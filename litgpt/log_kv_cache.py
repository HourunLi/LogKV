"""Log-Structured KV Cache with merged-position slots — strict O(B·log N).

Architecture (uniform 2:1 compaction at every level):
- Buffer (sliding window): recent_size raw tokens, no compression (exact keys).
- Level 0 (write level):   B entries, each = 2 tokens merged. Partially fillable
                            (0 to B entries). Filled one entry at a time from buffer
                            flush. When full, carries to level 1.
- Level 1+ (carry levels):  B entries per level, each = 2 entries from prev level
                            merged. All-or-nothing (0 or B entries). Binary carry
                            promotes full blocks upward.

Position handling ("expected RoPE", merged into the compressed state):
- Keys enter the cache as FULL post-RoPE vectors
      k_j = [ R(p_j)·k_pos_j ; k_content_j ]
  (partial rotary: only the leading rope_n_elem dims are rotated; any remaining
  content channel is position-free; full rotary means the content channel is
  empty — both are supported).
- Merging a block mean-pools the WHOLE key. Mean commutes with concatenation, so
  the merged key is [ mean_j R(p_j)·k_pos_j ; mean_j k_content_j ]: the block's
  position embedding is the weighted mean of its tokens' rotated position keys —
  the exact expectation of the rotation over the span. Per RoPE frequency θ over
  a width-w span centred at c this equals R(c)·sin(wθ/2)/(w·sin(θ/2)): RoPE at
  the span centre, damped per-frequency by a Dirichlet factor. High frequencies
  fade as spans widen, so positional resolution decays ∝ span width ∝ distance
  (constant relative error). The merged key is deliberately NOT renormalized:
  the norm shrinkage encodes the span's positional uncertainty.
- NO per-token state of any kind survives outside the recent window.

Slot attention (``log_kv_slot_attention``):
    score_s = (q · k_s) * scale + λ·log(w_s)
    out     = softmax(score) · v_s
The +λ·log(w_s) mass bias gives a w-token slot softmax mass ≈ w·exp(score),
first-order-matching per-token attention (log-sum-exp of w similar logits =
shared logit + log w + O(intra-slot score variance)). It is EXACT when the
tokens inside a slot are identical. Recent tokens are slots of w=1 (bias 0).

Memory:  O(recent_size + B·log(N/2)) slots — no Θ(N) term.
Compute: O(recent_size + B·log(N/2)) per query — no Θ(N) term.
"""

import math
from typing import NoReturn

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable


class LogStructuredKVCache(nn.Module):
    """Log-Structured KV Cache storing full post-RoPE keys in merged slots.

    Args:
        k_shape: (batch_size, n_groups, max_seq_length, k_dim) — k_dim is the
            FULL post-RoPE key width (RoPE'd position channel + content channel).
            max_seq_length only sizes the level hierarchy and the append-only
            token counter; no O(N) buffer is allocated.
        v_shape: (batch_size, n_groups, max_seq_length, v_dim)
        B: slots per level (default 512)
        recent_size: sliding window size (default 0 = 2)
        device / dtype: torch device / dtype
    """

    def __init__(
        self,
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        B: int = 512,
        recent_size: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        batch_size, n_groups, max_seq_length, k_dim = k_shape
        _, _, _, v_dim = v_shape

        self.batch_size = batch_size
        self.n_groups = n_groups
        self.max_seq_length = max_seq_length
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.B = B
        self.recent_size = recent_size if recent_size > 0 else 2
        # Explicit raise (not assert): must survive `python -O`.
        if self.recent_size < 2:
            raise ValueError(f"recent_size ({self.recent_size}) must be >= 2")

        denom = B * 2
        # +1 for the write level (level 0). The formula gives the number of carry
        # levels needed; total levels = carry + 1 write level.
        self.max_levels = max(2, math.ceil(math.log2(max((max_seq_length + 1) / denom, 1))) + 1)

        # Total tokens ever committed (scalar bookkeeping only — replaces the
        # former Θ(N) per-token position-key buffer).
        self.token_count: int = 0

        # ---- Sliding window: last < recent_size tokens (exact keys + values) ----
        self.register_buffer(
            "recent_k",
            torch.zeros(batch_size, n_groups, self.recent_size, k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "recent_v",
            torch.zeros(batch_size, n_groups, self.recent_size, v_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.recent_count: int = 0

        # ---- Hierarchical levels: B slots each (k, v, w) ----
        # Level 0: partially fillable (0 to B entries), written one entry at a time.
        # Level 1+: all-or-nothing (0 or B entries), written via binary carry.
        for ell in range(self.max_levels):
            self.register_buffer(
                f"level_k_{ell}",
                torch.zeros(batch_size, n_groups, B, k_dim, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"level_v_{ell}",
                torch.zeros(batch_size, n_groups, B, v_dim, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"level_w_{ell}",
                torch.zeros(batch_size, n_groups, B, device=device, dtype=dtype),
                persistent=False,
            )
        self.register_buffer(
            "level_count",
            torch.zeros(self.max_levels, dtype=torch.long, device=device),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Level accessors
    # ------------------------------------------------------------------

    def _get_level(self, ell: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            getattr(self, f"level_k_{ell}"),
            getattr(self, f"level_v_{ell}"),
            getattr(self, f"level_w_{ell}"),
        )

    def _set_level(
        self, ell: int, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor
    ) -> None:
        getattr(self, f"level_k_{ell}").copy_(k)
        getattr(self, f"level_v_{ell}").copy_(v)
        getattr(self, f"level_w_{ell}").copy_(w)
        self.level_count[ell] = self.B

    def _clear_level(self, ell: int) -> None:
        getattr(self, f"level_k_{ell}").zero_()
        getattr(self, f"level_v_{ell}").zero_()
        getattr(self, f"level_w_{ell}").zero_()
        self.level_count[ell] = 0

    # ------------------------------------------------------------------
    # Compact: compress tokens -> 1 entry via mean pooling (2:1 by default)
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_tokens(
        k: torch.Tensor,  # (B, G, n, k_dim) full post-RoPE keys
        v: torch.Tensor,  # (B, G, n, v_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress n tokens into a single compact entry via mean pooling.

        Mean-pooling the full key merges content and position in one step:
        the position sub-channel becomes the expected rotation over the span
        (see module docstring). No renormalization — the per-frequency norm
        shrinkage encodes the span's positional uncertainty.
        Returns k_entry (B,G,1,k_dim), v_entry (B,G,1,v_dim), w_entry (B,G,1).
        """
        n = k.size(2)
        k_entry = k.mean(dim=2, keepdim=True)
        v_entry = v.mean(dim=2, keepdim=True)
        w_entry = torch.full(
            (k.size(0), k.size(1), 1),
            float(n),
            device=k.device,
            dtype=k.dtype,
        )
        return k_entry, v_entry, w_entry

    # ------------------------------------------------------------------
    # Compact operation: merge two B-slot blocks -> one B-slot block (adjacent pairs)
    # ------------------------------------------------------------------

    @staticmethod
    def compact(
        k1: torch.Tensor, v1: torch.Tensor, w1: torch.Tensor,
        k2: torch.Tensor, v2: torch.Tensor, w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge two B-slot blocks into one B-slot block.

        Concatenates the two blocks in time order (k1=older, k2=newer) and pairs
        ADJACENT slots: (slot 2i, slot 2i+1) -> slot i. Every merged slot covers
        a contiguous span, and the weighted mean keeps each slot's key equal to
        the true weighted mean over all tokens it covers (mean-merge is
        associative), so no error accumulates across levels.
        """
        k_cat = torch.cat([k1, k2], dim=-2)  # (B, G, 2B, D)
        v_cat = torch.cat([v1, v2], dim=-2)
        w_cat = torch.cat([w1, w2], dim=-1)  # (B, G, 2B)

        ka = k_cat[..., 0::2, :]  # even slots (older half of each adjacent pair)
        kb = k_cat[..., 1::2, :]  # odd slots  (newer half of each adjacent pair)
        va = v_cat[..., 0::2, :]
        vb = v_cat[..., 1::2, :]
        wa = w_cat[..., 0::2]
        wb = w_cat[..., 1::2]

        w_total = wa + wb  # (B, G, B)
        alpha = (wa / w_total.clamp(min=1e-8)).unsqueeze(-1)  # (B, G, B, 1)

        k_out = alpha * ka + (1 - alpha) * kb
        v_out = alpha * va + (1 - alpha) * vb
        return k_out, v_out, w_total

    # ------------------------------------------------------------------
    # Add compact entry to level 0; carry to level 1+ when full
    # ------------------------------------------------------------------

    def _add_compact_entry(
        self, k_entry: torch.Tensor, v_entry: torch.Tensor, w_entry: torch.Tensor
    ) -> None:
        """Add one compact entry to level 0. If level 0 is full, binary carry to levels 1+."""
        idx = self.level_count[0].item()
        getattr(self, "level_k_0")[:, :, idx, :] = k_entry
        getattr(self, "level_v_0")[:, :, idx, :] = v_entry
        getattr(self, "level_w_0")[:, :, idx] = w_entry
        self.level_count[0] = idx + 1

        if self.level_count[0] >= self.B:
            lk, lv, lw = self._get_level(0)
            self._binary_carry(lk.clone(), lv.clone(), lw.clone())
            self._clear_level(0)

    # ------------------------------------------------------------------
    # Binary carry: promote B entries through levels 1+
    # ------------------------------------------------------------------

    def _binary_carry(self, block_k: torch.Tensor, block_v: torch.Tensor,
                      block_w: torch.Tensor) -> None:
        new_k, new_v, new_w = block_k, block_v, block_w
        for ell in range(1, self.max_levels):
            if self.level_count[ell] == 0:
                self._set_level(ell, new_k, new_v, new_w)
                return
            else:
                ek, ev, ew = self._get_level(ell)
                new_k, new_v, new_w = self.compact(ek, ev, ew, new_k, new_v, new_w)
                self._clear_level(ell)
        raise RuntimeError(
            f"LogStructuredKVCache: binary carry overflow! "
            f"All {self.max_levels} levels occupied. "
            f"max_seq_length={self.max_seq_length}, B={self.B}. "
            f"This is a bug — max_levels formula needs review."
        )

    # ------------------------------------------------------------------
    # Token accounting (append-only contract)
    # ------------------------------------------------------------------

    def _count_tokens(self, n: int) -> None:
        """Advance the committed-token counter, enforcing the sizing contract."""
        if self.token_count + n > self.max_seq_length:
            raise RuntimeError(
                f"LogStructuredKVCache: token overflow! "
                f"token_count={self.token_count}, n={n}, max_seq_length={self.max_seq_length}."
            )
        self.token_count += n

    # ------------------------------------------------------------------
    # Flush: compact oldest 2 tokens from buffer -> level 0
    # ------------------------------------------------------------------

    def _flush_recent(self) -> None:
        """Compact the oldest 2 tokens from the sliding window into level 0."""
        if self.recent_count < 2:
            return
        self._flush_pairs(2)

    def _flush_pairs(self, flush_len: int) -> None:
        """Batched equivalent of ``flush_len // 2`` sequential ``_flush_recent``
        calls: pairwise-merge the oldest ``flush_len`` window tokens into level-0
        entries in one shot, then shift the window once.

        The state trajectory is identical to the sequential version — the same
        (2i, 2i+1) pairs are merged with the same mean, entries reach level 0 in
        the same order, and carries fire at the same counts — but the O(window)
        clone+shift happens once instead of once per pair.
        """
        f = flush_len // 2
        rk = self.recent_k[:, :, :flush_len, :]
        rv = self.recent_v[:, :, :flush_len, :]
        B_, G_, _, kd = rk.shape
        vd = rv.size(-1)
        # Same arithmetic as _compact_tokens on each 2-token pair (mean over a
        # size-2 dim in the storage dtype), just for all f pairs at once.
        pk = rk.reshape(B_, G_, f, 2, kd).mean(dim=3)
        pv = rv.reshape(B_, G_, f, 2, vd).mean(dim=3)
        pw = torch.full((B_, G_, f), 2.0, device=rk.device, dtype=rk.dtype)
        self._append_level0(pk, pv, pw)

        # Shift the survivors to the front. Source/destination overlap in the
        # same storage; PyTorch copy_ with overlapping src/dst is undefined (may
        # corrupt silently on CUDA), so the source must be cloned first.
        remaining = self.recent_count - flush_len
        if remaining > 0:
            self.recent_k[:, :, :remaining, :] = self.recent_k[:, :, flush_len:self.recent_count, :].clone()
            self.recent_v[:, :, :remaining, :] = self.recent_v[:, :, flush_len:self.recent_count, :].clone()
        self.recent_k[:, :, remaining:self.recent_count, :].zero_()
        self.recent_v[:, :, remaining:self.recent_count, :].zero_()
        self.recent_count = remaining

    def _append_level0(self, pk: torch.Tensor, pv: torch.Tensor, pw: torch.Tensor) -> None:
        """Append f compact entries to level 0 in order, carrying when it fills.

        Trajectory-identical to f sequential ``_add_compact_entry`` calls: the
        binary carry fires exactly when the count reaches B, between the same
        two entries as in the sequential version.
        """
        f = pk.size(2)
        off = 0
        while off < f:
            idx = int(self.level_count[0].item())
            take = min(self.B - idx, f - off)
            getattr(self, "level_k_0")[:, :, idx:idx + take, :] = pk[:, :, off:off + take, :]
            getattr(self, "level_v_0")[:, :, idx:idx + take, :] = pv[:, :, off:off + take, :]
            getattr(self, "level_w_0")[:, :, idx:idx + take] = pw[:, :, off:off + take]
            self.level_count[0] = idx + take
            off += take
            if int(self.level_count[0].item()) >= self.B:
                lk, lv, lw = self._get_level(0)
                self._binary_carry(lk.clone(), lv.clone(), lw.clone())
                self._clear_level(0)

    # ------------------------------------------------------------------
    # Ingest chunk (testing): direct compact into level 0, bypass buffer
    # ------------------------------------------------------------------

    def ingest_chunk(
        self,
        k: torch.Tensor,   # (B, G, t_chunk, k_dim) full post-RoPE keys, detached
        v: torch.Tensor,   # (B, G, t_chunk, v_dim) detached
    ) -> None:
        """Compact a chunk of full keys + values straight into level 0.
        Bypasses the buffer. Intended for testing.
        """
        self._count_tokens(k.size(2))

        k_entry, v_entry, w_entry = self._compact_tokens(k, v)
        k_entry = k_entry.squeeze(2)
        v_entry = v_entry.squeeze(2)
        w_entry = w_entry.squeeze(2)

        self._add_compact_entry(k_entry, v_entry, w_entry)

    # ------------------------------------------------------------------
    # Add to recent (training): sliding window entry point
    # ------------------------------------------------------------------

    def add_recent(
        self,
        k: torch.Tensor,   # (B, G, n, k_dim) full post-RoPE keys, detached
        v: torch.Tensor,   # (B, G, n, v_dim) detached
    ) -> None:
        """Add tokens to the sliding window. When the window overflows, the
        oldest 2 tokens are flushed (compacted) into level 0.
        """
        n = k.size(2)
        # Explicit raise (not assert): must survive `python -O`.
        if n > self.recent_size:
            raise ValueError(f"chunk size {n} exceeds recent_size {self.recent_size}")

        self._count_tokens(n)

        # Batched flush: streaming lazily flushes the oldest pair each time the
        # window refills, so over this whole chunk it flushes ceil(overflow/2)
        # pairs — all taken from the CURRENT window front (appends only ever go
        # to the right). Flushing them up front in one batch reaches the exact
        # same final state with one shift instead of one per pair.
        overflow = self.recent_count + n - self.recent_size
        if overflow > 0:
            flush_len = 2 * ((overflow + 1) // 2)
            if flush_len <= self.recent_count:
                self._flush_pairs(flush_len)
            else:
                # Degenerate corner (odd recent_count with n == recent_size):
                # fall back to the lazy interleaved order.
                offset = 0
                while offset < n:
                    if self.recent_count == self.recent_size:
                        self._flush_recent()
                    capacity = self.recent_size - self.recent_count
                    if capacity <= 0:
                        raise RuntimeError(
                            "LogStructuredKVCache.add_recent() could not make room in the recent buffer; "
                            f"recent_count={self.recent_count}, recent_size={self.recent_size}."
                        )
                    take = min(capacity, n - offset)
                    self.recent_k[:, :, self.recent_count:self.recent_count + take, :] = k[:, :, offset:offset + take, :]
                    self.recent_v[:, :, self.recent_count:self.recent_count + take, :] = v[:, :, offset:offset + take, :]
                    self.recent_count += take
                    offset += take
                return

        self.recent_k[:, :, self.recent_count:self.recent_count + n, :] = k
        self.recent_v[:, :, self.recent_count:self.recent_count + n, :] = v
        self.recent_count += n

    # ------------------------------------------------------------------
    # nn.Module forward intentionally disabled
    # ------------------------------------------------------------------

    def forward(
        self,
        input_pos: torch.Tensor,
        k: torch.Tensor,   # (B, G, T, k_dim) post-RoPE
        v: torch.Tensor,   # (B, G, T, v_dim)
    ) -> NoReturn:
        """Disabled direct update API.

        LogKV attention must build a temporary state that includes the current
        token(s), compute slot attention from that state, and only then commit
        the token(s) to the cache. Calling the cache directly used to implement
        an incompatible raw-prefill / write-before-decode path, so it is
        intentionally disabled. ``input_pos`` is accepted only to keep the
        stale call site fail-fast and self-explanatory. Use
        ``GPT.set_log_kv_cache()`` and let ``CausalSelfAttention`` manage the
        update order, or call
        ``get_attention_state()`` and ``add_recent()`` explicitly.
        """
        raise RuntimeError(
            "LogStructuredKVCache.forward() is disabled/deprecated: direct cache calls cannot implement "
            "LogKV correctly because attention must include the current token(s) before committing them. "
            "Use GPT.set_log_kv_cache() so CausalSelfAttention owns the update order, or use the explicit "
            "get_attention_state()/add_recent() helpers in tests."
        )

    # ------------------------------------------------------------------
    # Build attention state: slot-granular, O(recent + B*log N) entries
    # ------------------------------------------------------------------

    def get_attention_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble the cache state for ``log_kv_slot_attention``.

        Returns:
            slot_k: (B, G, n_slots, k_dim) — time-ordered slot keys: compact
                    levels oldest (highest level) first down to level 0, then
                    the recent-window tokens as exact w=1 slots.
            slot_v: (B, G, n_slots, v_dim)
            slot_w: (B, G, n_slots) — token count per slot (1 for recent).

        Slots cover contiguous, time-ordered spans, so ``cumsum(slot_w)`` gives
        the token boundaries of every slot (used by exactness tests).
        """
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        w_parts: list[torch.Tensor] = []

        # Compact levels: oldest (highest level) first, down to level 0
        for ell in range(self.max_levels - 1, -1, -1):
            count = self.level_count[ell].item()
            if count > 0:
                lk, lv, lw = self._get_level(ell)
                k_parts.append(lk[:, :, :count, :])
                v_parts.append(lv[:, :, :count, :])
                w_parts.append(lw[:, :, :count])

        # Recent window: exact tokens, weight 1 each
        if self.recent_count > 0:
            k_parts.append(self.recent_k[:, :, :self.recent_count, :])
            v_parts.append(self.recent_v[:, :, :self.recent_count, :])
            w_parts.append(
                self.recent_k.new_ones(self.batch_size, self.n_groups, self.recent_count)
            )

        if w_parts:
            return (
                torch.cat(k_parts, dim=-2),
                torch.cat(v_parts, dim=-2),
                torch.cat(w_parts, dim=-1),
            )
        return (
            self.recent_k[:, :, :0, :],
            self.recent_v[:, :, :0, :],
            getattr(self, "level_w_0")[:, :, :0],
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _convert_dtype(self, dtype: torch.dtype) -> None:
        """Reconcile all data buffers with the activation dtype.

        Needed when the cache was allocated at a different (default) dtype than
        the running activations — e.g. bf16 inference where ``set_log_kv_cache``
        received no explicit ``dtype`` and the process default is fp32. Without
        this, ``get_attention_state`` / ``add_recent`` would try to ``cat`` /
        ``matmul`` fp32 buffers against bf16 activations and raise.

        Idempotent and cheap: a no-op once the buffers already match, so it is
        safe to call on every forward (mirrors ``KVCache.forward``'s reconcile).
        ``level_count`` stays ``long`` and is deliberately untouched. Compaction
        weights are always powers of two (uniform 2:1), hence exact in bf16/fp16,
        so converting ``level_w`` does not corrupt token counts.
        """
        if self.recent_k.dtype == dtype:
            return
        self.recent_k = self.recent_k.to(dtype)
        self.recent_v = self.recent_v.to(dtype)
        for ell in range(self.max_levels):
            setattr(self, f"level_k_{ell}", getattr(self, f"level_k_{ell}").to(dtype))
            setattr(self, f"level_v_{ell}", getattr(self, f"level_v_{ell}").to(dtype))
            setattr(self, f"level_w_{ell}", getattr(self, f"level_w_{ell}").to(dtype))

    def reset_parameters(self) -> None:
        """Reset all buffers to zero."""
        self.token_count = 0
        self.recent_k.zero_()
        self.recent_v.zero_()
        self.recent_count = 0
        for ell in range(self.max_levels):
            self._clear_level(ell)

    @property
    def total_slots(self) -> int:
        count = self.recent_count
        for ell in range(self.max_levels):
            count += self.level_count[ell].item()
        return count

    @property
    def total_tokens_covered(self) -> int:
        return self.token_count


# ======================================================================
# Slot attention: merged-position slots + log-multiplicity mass bias
# ======================================================================


def append_exact_tokens(
    slot_k: torch.Tensor,   # (B, G, S, k_dim)
    slot_v: torch.Tensor,   # (B, G, S, v_dim)
    slot_w: torch.Tensor,   # (B, G, S)
    k_new: torch.Tensor,    # (B, G, n, k_dim) exact tokens, appended in time order
    v_new: torch.Tensor,    # (B, G, n, v_dim)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append exact per-token entries (w=1 slots) after the cached slots.

    Used by the attention layer to make the current chunk (and any pending
    token) visible to attention BEFORE it is committed to the cache. Gradient
    flows through ``k_new``/``v_new``; the ones-weights are constants.
    """
    n_new = k_new.size(2)
    ones = slot_w.new_ones(slot_w.size(0), slot_w.size(1), n_new)
    return (
        torch.cat([slot_k, k_new], dim=2),
        torch.cat([slot_v, v_new], dim=2),
        torch.cat([slot_w, ones], dim=-1),
    )


def log_kv_slot_attention(
    q: torch.Tensor,        # (B, nh, T_q, k_dim) full post-RoPE queries
    slot_k: torch.Tensor,   # (B, G, S, k_dim) merged slot keys (position included)
    slot_v: torch.Tensor,   # (B, G, S, v_dim)
    slot_w: torch.Tensor,   # (B, G, S) token count per slot (>= 1)
    scale: float,
    mask: torch.Tensor | None = None,  # (T_q, S) bool, True = attend
    lam: float = 1.0,
    causal_tail: int = 0,
) -> torch.Tensor:
    """Slot-granular attention over merged-position entries.

        score_s = (q · k_s) * scale + λ·log(w_s)
        out     = softmax(score) · v_s

    One logit per SLOT — compute and memory are O(S) = O(recent + B·log N),
    never O(N). The position information lives inside ``slot_k`` (expected-RoPE
    sub-channel, see module docstring), so a single dot product covers both the
    content and the position score. The +λ·log(w_s) bias restores the softmax
    mass a w-token span would have contributed token-by-token; it is exact when
    the span's tokens are identical and first-order otherwise. λ=1 preserves
    mass ∝ token count; λ=0 yields a built-in ∝1/w long-range forgetting curve
    (ablation knob).

    Note the bias is added AFTER ``scale``: it is a multiplicity correction on
    the logits, not a similarity, so it must not be shrunk by 1/sqrt(d).

    Preconditions (enforced by construction in the callers):
      - ``slot_w`` entries are >= 1 (log is finite);
      - every ``mask`` row has at least one True (softmax row is finite) —
        chunk-causal masks always allow the diagonal.

    Args:
        q: (B, nh, T_q, k_dim)
        slot_k / slot_v / slot_w: from ``get_attention_state()`` (+ optionally
            ``append_exact_tokens`` for the in-flight chunk)
        scale: attention scale (applied to the dot product only)
        mask: optional (T_q, S) bool. True = allowed. General-purpose escape
            hatch (tests); the model uses ``causal_tail`` instead. None (and
            causal_tail == 0) = attend all.
        lam: weight of the log-multiplicity mass bias (default 1.0)
        causal_tail: if > 0, the LAST ``causal_tail`` slots are the in-flight
            chunk appended behind the frozen cache state (see
            ``append_exact_tokens``); they are masked causally against the
            queries (query i sees appended entry j iff j <= i; requires
            ``causal_tail == T_q``), while everything before them stays fully
            visible. Memory: this replaces a (T_q, S) bool mask + a
            masked_fill copy of the full fp32 score tensor with one in-place
            fill on the (T_q, causal_tail) tail slice — per streaming chunk,
            per layer. Mutually exclusive with ``mask``.

    Returns:
        (B, nh, T_q, v_dim)
    """
    B, nh, T_q, k_dim = q.shape
    v_dim = slot_v.size(-1)
    S = slot_k.size(2)
    if S == 0:
        return torch.zeros(B, nh, T_q, v_dim, device=q.device, dtype=q.dtype)

    if causal_tail:
        # Explicit raises (not asserts): survive `python -O`.
        if mask is not None:
            raise ValueError("causal_tail and mask are mutually exclusive")
        if causal_tail != T_q:
            raise ValueError(
                f"causal_tail ({causal_tail}) must equal T_q ({T_q}): the tail entries "
                "are the appended in-flight chunk, aligned one-to-one with the queries"
            )
        # (T_q, T_q) bool, True above the diagonal = blocked. Tiny (chunk-sized,
        # not S-sized) and the only allocation masking costs on this path.
        tail_blocked = torch.ones(T_q, causal_tail, dtype=torch.bool, device=q.device).triu_(1)

    # Scores in fp32: the log-w bias shifts logits by up to ~log(N) and bf16
    # resolution degrades with magnitude; fp32 keeps cross-level logit
    # differences intact. S is O(log N), so the fp32 buffer is small.
    #
    # In-place ops (mul_/add_/masked_fill_) are deliberate: none of these
    # intermediates is a saved tensor for backward (matmul saves its operands,
    # softmax saves its output, the scalar/bias ops save nothing), so mutating
    # the score buffer is autograd-safe and avoids three full-size fp32
    # temporaries per call — once per streaming chunk, per layer.
    nkv = slot_k.size(1)
    if nh != nkv:
        # --- GQA without materializing an rf× copy of the slots ---
        # repeat_interleave(rf, dim=1) on slot_k/v/w would allocate a fresh
        # (B, nh, S, ·) copy AND, in training, save that expanded tensor for
        # backward — rf× the slot memory for every one of the O(T) streaming
        # chunks, which is what runs the GPU out of memory. Instead fold the rf
        # query heads that share a KV group into their own axis and let matmul
        # broadcast the (B, nkv, 1, …) group across them: the K/V operands stay
        # (B, nkv, S, ·), nothing is duplicated, and the tensor autograd saves is
        # rf× smaller. Mathematically identical to the repeat_interleave path
        # (query head h ↔ KV group h // rf, matching model.py's GQA convention);
        # any difference is kernel-level FP reduction order, ULP-scale.
        rf = nh // nkv
        qg = q.reshape(B, nkv, rf, T_q, k_dim)               # (B, nkv, rf, T_q, k_dim)
        scores = torch.matmul(qg, slot_k.unsqueeze(2).mT)    # (B, nkv, rf, T_q, S)
        scores = scores.to(torch.float32)
        scores.mul_(scale)
        if lam != 0.0:
            # slot_w >= 1 by construction; guarded via lam gate so lam=0 can
            # never produce 0 * log(0) = NaN even on malformed input.
            scores.add_(lam * slot_w.to(torch.float32).log()[:, :, None, None, :])
        if mask is not None:
            scores.masked_fill_(~mask.view(1, 1, 1, T_q, S), float("-inf"))
        elif causal_tail:
            scores[..., S - causal_tail:].masked_fill_(tail_blocked, float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(q.dtype)     # (B, nkv, rf, T_q, S)
        out = torch.matmul(attn, slot_v.unsqueeze(2))        # (B, nkv, rf, T_q, v_dim)
        return out.reshape(B, nh, T_q, v_dim)

    # MHA (nh == nkv): one logit per slot, no head expansion needed.
    scores = torch.matmul(q, slot_k.mT).to(torch.float32)  # (B, nh, T_q, S)
    scores.mul_(scale)
    if lam != 0.0:
        # slot_w >= 1 by construction; guarded via lam gate so lam=0 can never
        # produce 0 * log(0) = NaN even on malformed input.
        scores.add_(lam * slot_w.to(torch.float32).log().unsqueeze(-2))
    if mask is not None:
        scores.masked_fill_(~mask.view(1, 1, T_q, S), float("-inf"))
    elif causal_tail:
        scores[..., S - causal_tail:].masked_fill_(tail_blocked, float("-inf"))
    attn = torch.softmax(scores, dim=-1).to(q.dtype)  # (B, nh, T_q, S)
    return torch.matmul(attn, slot_v)                 # (B, nh, T_q, v_dim)


# ======================================================================
# Low-memory training autograd: stream without a graph, replay in backward
# ======================================================================


def log_kv_chunk_attention(
    cache: LogStructuredKVCache,
    q_b: torch.Tensor,   # (B, nh, t, k_dim) current-chunk queries, post-RoPE
    k_b: torch.Tensor,   # (B, G, t, k_dim) current-chunk keys, post-RoPE
    v_b: torch.Tensor,   # (B, G, t, v_dim)
    scale: float,
) -> torch.Tensor:
    """One streaming attention step, WITHOUT committing the chunk.

    [frozen cache state (fully visible)] + [current chunk (causal)]. This is
    the shared building block of ``LogKVStreamTrainingAttention``: its forward
    stream and its backward replay must both compute chunks through this exact
    function so the recomputed graphs match the streamed outputs.
    """
    slot_k, slot_v, slot_w = cache.get_attention_state()
    k_all, v_all, w_all = append_exact_tokens(slot_k, slot_v, slot_w, k_b, v_b)
    return log_kv_slot_attention(
        q_b, k_all, v_all, w_all, scale=scale, causal_tail=q_b.size(2)
    )


class LogKVStreamTrainingAttention(torch.autograd.Function):
    """Constant-in-T-times-S memory autograd for logKV streaming training.

    Problem: the naive training graph saves, for every 2-token chunk, the
    concatenated [detached prefix + chunk] keys/values (O(S) each) that its
    two matmuls need for backward. Per layer that is O(T/2 * S) saved tensors
    — hundreds of GB for a 32K stream at B=512/recent=1024 — the long-context
    training OOM (the failing allocation is merely whichever op runs next
    once the accumulated graph has eaten the device).

    Forward here streams the whole sequence with NO recorded graph and returns
    the attention output. Backward resets the cache and REPLAYS the same stream:
    each block's attention is recomputed with a throwaway local graph that
    ``torch.autograd.grad`` consumes immediately, so at most one block graph is
    alive at any time. Peak memory: O(T + train_block*S) instead of O(T*S).
    Cost: one extra streaming pass plus the per-block backwards.

    Correctness:
      - For a fixed train_block, gradients are EXACT for that block-streaming
        objective: in the naive graph, gradient reaches q/k/v only through each
        token's own block (the cache commits detached copies), so per-block
        grads are complete and disjoint, and the replayed per-block
        ``autograd.grad`` reproduces them one-to-one.
      - The replay is deterministic: compaction is pure mean-pooling with
        count-based binary carries — no RNG, identical shapes take identical
        kernels — so the rebuilt per-block prefix states equal forward's.
      - Backward does not depend on the cache state forward left behind (it
        resets first), so interleaved forward/backward across micro-batches
        or activation-checkpoint recompute ordering cannot corrupt it.

    ``train_block=2`` is the strict 2-token streaming reference. Larger blocks
    freeze the prefix state at block start and use causal exact attention within
    the block, matching the inference prefill speed/memory tradeoff while
    preserving the exact final cache state. First-order only
    (``once_differentiable``) — CPT never needs grad-of-grad.
    """

    @staticmethod
    def forward(ctx, q, k, v, cache, scale, train_block):
        T = q.size(2)
        train_block = int(train_block)
        if train_block < 2:
            raise ValueError(f"logKV train_block must be >= 2, got {train_block}")
        if train_block > cache.recent_size:
            raise ValueError(
                f"logKV train_block ({train_block}) must be <= recent_size ({cache.recent_size})"
            )
        outputs: list[torch.Tensor] = []
        # Explicit no_grad: the memory guarantee of this whole scheme rests on
        # this pass recording nothing (Function.forward already runs detached;
        # this makes the invariant local and future-proof).
        with torch.no_grad():
            cache.reset_parameters()
            start = 0
            while start < T:  # mirrored in backward() — keep in sync
                end = min(start + train_block, T)
                outputs.append(
                    log_kv_chunk_attention(
                        cache,
                        q[:, :, start:end], k[:, :, start:end], v[:, :, start:end],
                        scale,
                    )
                )
                cache.add_recent(k[:, :, start:end], v[:, :, start:end])
                start = end
        ctx.save_for_backward(q, k, v)
        ctx.cache = cache
        ctx.scale = scale
        ctx.train_block = train_block
        return torch.cat(outputs, dim=2)  # (B, nh, T, v_dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        q, k, v = ctx.saved_tensors
        cache = ctx.cache
        scale = ctx.scale
        train_block = ctx.train_block
        T = q.size(2)
        # Blocks partition [0, T) and each position's grad comes from exactly
        # its own block, so the empty buffers are fully overwritten.
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        cache.reset_parameters()
        start = 0
        while start < T:  # mirrors forward() — keep in sync
            end = min(start + train_block, T)
            q_b = q[:, :, start:end].detach().requires_grad_(True)
            k_b = k[:, :, start:end].detach().requires_grad_(True)
            v_b = v[:, :, start:end].detach().requires_grad_(True)
            with torch.enable_grad():
                y_b = log_kv_chunk_attention(cache, q_b, k_b, v_b, scale)
            g_q, g_k, g_v = torch.autograd.grad(y_b, (q_b, k_b, v_b), grad_y[:, :, start:end])
            dq[:, :, start:end] = g_q
            dk[:, :, start:end] = g_k
            dv[:, :, start:end] = g_v
            with torch.no_grad():
                cache.add_recent(k[:, :, start:end], v[:, :, start:end])
            start = end
        return dq, dk, dv, None, None, None
