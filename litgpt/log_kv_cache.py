"""Log-Structured Semantic KV Cache with DeepSeek V4-style Content-Position Decoupling.

Architecture (uniform 2:1 compaction at every level):
- Buffer (sliding window): recent_size raw tokens, no compression.
- Level 0 (write level):   B entries, each = 2 tokens merged. Partially fillable
                            (0 to B entries). Filled one entry at a time from buffer
                            flush. When full, carries to level 1.
- Level 1+ (carry levels):  B entries per level, each = 2 entries from prev level
                            merged. All-or-nothing (0 or B entries). Binary carry
                            promotes full blocks upward.

Content-Position Decoupling:
- Content channel (head_size - d_pos dims, NO RoPE): hierarchically compressed.
  Position-free -> averaging is clean.
- Position channel (d_pos dims, RoPE'd): stored PER-TOKEN, NEVER compressed.
  Each token retains its exact position key.
- Value: compressed together with content via the hierarchy.

Decoupled attention (DeepSeek V4 style):
  score_j = q_content . k_content_{slot(j)}    (per-slot, shared across the span)
          + q_pos . k_pos_j                     (per-token, exact position)
  softmax over all tokens (compressed-expanded + exact recent), weighted sum of values.
  No +log(w) mass correction: per-token expansion naturally gives each slot attention
  mass proportional to its token count.

Memory:  O(B * log(N/t) * (content_dim + v_dim) + N * d_pos)
  Position is O(N * d_pos) (uncompressed, small dim); content+value O(B * log) (compressed).
Compute: O(N * (d_pos + v_dim)) per query.
"""

import math
from typing import NoReturn

import torch
import torch.nn as nn


class LogStructuredKVCache(nn.Module):
    """Log-Structured KV Cache with per-token position and compressed content/value.

    Args:
        k_content_shape: (batch_size, n_groups, max_seq_length, content_dim)
        v_shape: (batch_size, n_groups, max_seq_length, v_dim)
        d_pos: position key dim (RoPE'd, per-token, uncompressed)
        B: slots per level (default 512)
        recent_size: sliding window size (default 0 = 2)
        device / dtype: torch device / dtype
    """

    def __init__(
        self,
        k_content_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        d_pos: int,
        B: int = 512,
        recent_size: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        batch_size, n_groups, max_seq_length, content_dim = k_content_shape
        _, _, _, v_dim = v_shape

        self.batch_size = batch_size
        self.n_groups = n_groups
        self.max_seq_length = max_seq_length
        self.content_dim = content_dim
        self.v_dim = v_dim
        self.d_pos = d_pos
        self.B = B
        self.recent_size = recent_size if recent_size > 0 else 2
        # Explicit raise (not assert): must survive `python -O`.
        if self.recent_size < 2:
            raise ValueError(f"recent_size ({self.recent_size}) must be >= 2")

        denom = B * 2
        # +1 for the write level (level 0). The formula gives the number of carry
        # levels needed; total levels = carry + 1 write level.
        self.max_levels = max(2, math.ceil(math.log2(max((max_seq_length + 1) / denom, 1))) + 1)

        # ---- Per-token position keys (NEVER compressed) ----
        self.register_buffer(
            "pos_keys",
            torch.zeros(batch_size, n_groups, max_seq_length, d_pos, device=device, dtype=dtype),
            persistent=False,
        )
        self.pos_count: int = 0

        # ---- Sliding window: last < recent_size tokens (exact content + value) ----
        self.register_buffer(
            "recent_content",
            torch.zeros(batch_size, n_groups, self.recent_size, content_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "recent_values",
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
                torch.zeros(batch_size, n_groups, B, content_dim, device=device, dtype=dtype),
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
        k_content: torch.Tensor,  # (B, G, n, content_dim)
        v: torch.Tensor,          # (B, G, n, v_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress n tokens into a single compact entry via mean pooling.

        Content keys carry no RoPE -> averaging is position-clean.
        Returns k_entry (B,G,1,content_dim), v_entry (B,G,1,v_dim), w_entry (B,G,1).
        """
        n = k_content.size(2)
        k_entry = k_content.mean(dim=2, keepdim=True)
        v_entry = v.mean(dim=2, keepdim=True)
        w_entry = torch.full(
            (k_content.size(0), k_content.size(1), 1),
            float(n),
            device=k_content.device,
            dtype=k_content.dtype,
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
        ADJACENT slots: (slot 2i, slot 2i+1) -> slot i. This keeps every merged
        slot covering a contiguous span, so per-token position keys for the span
        can be gathered by [start : start + w).
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
    # Position key store (per-token, uncompressed)
    # ------------------------------------------------------------------

    def _store_pos_keys(self, pos_keys_chunk: torch.Tensor) -> None:
        """Append per-token position keys to the pos_keys buffer (in arrival order)."""
        n = pos_keys_chunk.size(2)
        if self.pos_count + n > self.max_seq_length:
            raise RuntimeError(
                f"LogStructuredKVCache: pos_keys overflow! "
                f"pos_count={self.pos_count}, n={n}, max_seq_length={self.max_seq_length}."
            )
        self.pos_keys[:, :, self.pos_count:self.pos_count + n, :] = pos_keys_chunk
        self.pos_count += n

    # ------------------------------------------------------------------
    # Flush: compact oldest 2 tokens from buffer -> level 0
    # ------------------------------------------------------------------

    def _flush_recent(self) -> None:
        """Compact the oldest 2 tokens from the sliding window into level 0."""
        if self.recent_count < 2:
            return

        k_entry, v_entry, w_entry = self._compact_tokens(
            self.recent_content[:, :, :2, :],
            self.recent_values[:, :, :2, :],
        )
        k_entry = k_entry.squeeze(2)   # (B, G, content_dim)
        v_entry = v_entry.squeeze(2)   # (B, G, v_dim)
        w_entry = w_entry.squeeze(2)   # (B, G)

        self._add_compact_entry(k_entry, v_entry, w_entry)

        # Shift remaining tokens to the front. The source slice [2:count] overlaps
        # the destination [0:count-2] in the same storage; PyTorch copy_ with
        # overlapping src/dst is undefined (may corrupt silently on CUDA), so the
        # source must be materialized via .clone() first.
        remaining = self.recent_count - 2
        if remaining > 0:
            self.recent_content[:, :, :remaining, :] = self.recent_content[:, :, 2:self.recent_count, :].clone()
            self.recent_values[:, :, :remaining, :] = self.recent_values[:, :, 2:self.recent_count, :].clone()
        self.recent_content[:, :, remaining:self.recent_count, :].zero_()
        self.recent_values[:, :, remaining:self.recent_count, :].zero_()
        self.recent_count = remaining

    # ------------------------------------------------------------------
    # Ingest chunk (testing): direct compact into level 0, bypass buffer
    # ------------------------------------------------------------------

    def ingest_chunk(
        self,
        k_content: torch.Tensor,   # (B, G, t_chunk, content_dim) detached
        v: torch.Tensor,           # (B, G, t_chunk, v_dim) detached
        pos_keys_chunk: torch.Tensor,  # (B, G, t_chunk, d_pos) detached
    ) -> None:
        """Store per-token position keys and compact content+value into level 0.
        Bypasses the buffer. Intended for testing.
        """
        self._store_pos_keys(pos_keys_chunk)

        k_entry, v_entry, w_entry = self._compact_tokens(k_content, v)
        k_entry = k_entry.squeeze(2)
        v_entry = v_entry.squeeze(2)
        w_entry = w_entry.squeeze(2)

        self._add_compact_entry(k_entry, v_entry, w_entry)

    # ------------------------------------------------------------------
    # Add to recent (training): sliding window entry point
    # ------------------------------------------------------------------

    def add_recent(
        self,
        k_content: torch.Tensor,   # (B, G, n, content_dim) detached
        v: torch.Tensor,           # (B, G, n, v_dim) detached
        pos_keys: torch.Tensor,    # (B, G, n, d_pos) detached
    ) -> None:
        """Training-only: add tokens to the sliding window. When the window
        overflows, the oldest 2 tokens are flushed (compacted) into level 0.
        """
        n = k_content.size(2)
        # Explicit raise (not assert): must survive `python -O`.
        if n > self.recent_size:
            raise ValueError(f"chunk size {n} exceeds recent_size {self.recent_size}")

        self._store_pos_keys(pos_keys)

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
            self.recent_content[:, :, self.recent_count:self.recent_count + take, :] = (
                k_content[:, :, offset:offset + take, :]
            )
            self.recent_values[:, :, self.recent_count:self.recent_count + take, :] = v[:, :, offset:offset + take, :]
            self.recent_count += take
            offset += take

    # ------------------------------------------------------------------
    # nn.Module forward intentionally disabled
    # ------------------------------------------------------------------

    def forward(
        self,
        input_pos: torch.Tensor,
        k: torch.Tensor,   # (B, G, T, head_size) post-RoPE
        v: torch.Tensor,   # (B, G, T, head_size)
    ) -> NoReturn:
        """Disabled direct update API.

        LogKV attention must build a temporary state that includes the current
        token(s), compute decoupled attention from that state, and only then
        commit the token(s) to the cache. Calling the cache directly used to
        implement an incompatible raw-prefill / write-before-decode path, so it
        is intentionally disabled. ``input_pos`` is accepted only to keep the
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
    # Build attention state: assemble all pieces for decoupled attention
    # ------------------------------------------------------------------

    def get_attention_state(self) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,  # compact k, v, w
        torch.Tensor,                               # compact per-token pos_keys
        torch.Tensor, torch.Tensor, torch.Tensor,  # recent content, values, pos_keys
    ]:
        """Assemble the cache state for ``log_kv_decoupled_attention``.

        Returns:
            compact_k:     (B, G, n_slots, content_dim) — time-ordered compact slots
            compact_v:     (B, G, n_slots, v_dim)
            compact_w:     (B, G, n_slots) — token count per slot
            compact_pos_keys: (B, G, n_compressed_tokens, d_pos) — per-token pos keys
                              for all tokens covered by compact slots (contiguous order)
            recent_content:(B, G, recent_count, content_dim) — exact
            recent_values: (B, G, recent_count, v_dim) — exact
            recent_pos_keys: (B, G, recent_count, d_pos) — exact per-token

        The compact slots are time-ordered (oldest level first) with contiguous
        spans, so ``cumsum(compact_w)`` gives span boundaries aligned with
        ``compact_pos_keys``.
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

        if w_parts:
            compact_k = torch.cat(k_parts, dim=-2)
            compact_v = torch.cat(v_parts, dim=-2)
            compact_w = torch.cat(w_parts, dim=-1)
        else:
            compact_k = self.recent_content[:, :, :0, :]
            compact_v = self.recent_values[:, :, :0, :]
            compact_w = getattr(self, "level_w_0")[:, :, :0]

        # Per-token position keys: compressed tokens first, then recent.
        n_compressed = self.pos_count - self.recent_count
        n_compressed = max(0, n_compressed)
        compact_pos_keys = self.pos_keys[:, :, :n_compressed, :]
        recent_content = self.recent_content[:, :, :self.recent_count, :]
        recent_values = self.recent_values[:, :, :self.recent_count, :]
        recent_pos_keys = self.pos_keys[:, :, n_compressed:n_compressed + self.recent_count, :]

        return (compact_k, compact_v, compact_w, compact_pos_keys,
                recent_content, recent_values, recent_pos_keys)

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
        if self.recent_content.dtype == dtype:
            return
        self.pos_keys = self.pos_keys.to(dtype)
        self.recent_content = self.recent_content.to(dtype)
        self.recent_values = self.recent_values.to(dtype)
        for ell in range(self.max_levels):
            setattr(self, f"level_k_{ell}", getattr(self, f"level_k_{ell}").to(dtype))
            setattr(self, f"level_v_{ell}", getattr(self, f"level_v_{ell}").to(dtype))
            setattr(self, f"level_w_{ell}", getattr(self, f"level_w_{ell}").to(dtype))

    def reset_parameters(self) -> None:
        """Reset all buffers to zero."""
        self.pos_keys.zero_()
        self.pos_count = 0
        self.recent_content.zero_()
        self.recent_values.zero_()
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
        return self.pos_count


# ======================================================================
# Decoupled attention (DeepSeek V4 style): content (per-slot) + position (per-token)
# ======================================================================


def log_kv_decoupled_attention(
    q_content: torch.Tensor,  # (B, nh, T_q, content_dim)
    q_pos: torch.Tensor,      # (B, nh, T_q, d_pos)
    cache_state: tuple,
    scale: float,
    mask: torch.Tensor | None = None,  # (T_q, n_total) bool, True = attend
) -> torch.Tensor:
    """Decoupled attention: content channel (per-slot, compressed) + position channel
    (per-token, exact).

    Each compact slot s covers w_s contiguous tokens. It contributes w_s logits to
    the softmax — one per token — sharing the slot's content score but using each
    token's own exact position score. Exact (recent / current-chunk) tokens are
    treated as slots of weight 1.

    score_{j} = q_content . k_content_{slot(j)} + q_pos . k_pos_j
    out = softmax(score) . v_{slot(j)}

    No +log(w): per-token expansion gives each slot mass proportional to w_s.

    Args:
        q_content: (B, nh, T_q, content_dim)
        q_pos: (B, nh, T_q, d_pos)
        cache_state: tuple from ``LogStructuredKVCache.get_attention_state()``
            (compact_k, compact_v, compact_w, compact_pos_keys,
             recent_content, recent_values, recent_pos_keys)
        scale: attention scale (applied to the combined score)
        mask: optional (T_q, n_total) bool. True = allowed. For training chunk
            causality (prefix fully visible, chunk causal). None for decode.

    Returns:
        (B, nh, T_q, v_dim)
    """
    (compact_k, compact_v, compact_w, compact_pos_keys,
     recent_content, recent_values, recent_pos_keys) = cache_state

    B, nh, T_q, content_dim = q_content.shape
    d_pos = q_pos.size(-1)
    device = q_content.device

    n_compact_slots = compact_k.size(2)
    n_compact_tokens = compact_pos_keys.size(2)
    n_recent = recent_content.size(2)
    n_total = n_compact_tokens + n_recent

    # --- Build unified slot arrays (at n_groups granularity) ---
    if n_compact_slots > 0:
        all_slot_k = torch.cat([compact_k, recent_content], dim=-2)
        all_slot_v = torch.cat([compact_v, recent_values], dim=-2)
        w_counts = compact_w[0, 0].long()  # (n_compact_slots,)
        slot_idx_compact = torch.repeat_interleave(
            torch.arange(n_compact_slots, device=device), w_counts
        )  # (n_compact_tokens,)
    else:
        all_slot_k = recent_content
        all_slot_v = recent_values
        slot_idx_compact = torch.empty(0, dtype=torch.long, device=device)

    n_all_slots = n_compact_slots + n_recent
    if n_recent > 0:
        slot_idx_recent = n_compact_slots + torch.arange(n_recent, device=device)
        token_to_slot = torch.cat([slot_idx_compact, slot_idx_recent], dim=0)  # (n_total,)
    else:
        token_to_slot = slot_idx_compact

    # Per-token position keys (all tokens)
    if n_total > 0:
        all_pos_keys = torch.cat([compact_pos_keys, recent_pos_keys], dim=2)  # (B, G, n_total, d_pos)
    else:
        v_dim = all_slot_v.size(-1)
        return torch.zeros(B, nh, T_q, v_dim, device=device, dtype=q_content.dtype)

    # --- GQA: expand k-side from n_groups to n_head ---
    nkv = all_slot_k.size(1)
    if nh != nkv:
        rf = nh // nkv
        all_slot_k = all_slot_k.repeat_interleave(rf, dim=1)
        all_slot_v = all_slot_v.repeat_interleave(rf, dim=1)
        all_pos_keys = all_pos_keys.repeat_interleave(rf, dim=1)

    # --- Scores ---
    content_scores_slots = torch.matmul(q_content, all_slot_k.mT)  # (B, nh, T_q, n_all_slots)
    position_scores = torch.matmul(q_pos, all_pos_keys.mT)         # (B, nh, T_q, n_total)

    # Expand content score to per-token: gather along the slot axis.
    idx_expand = token_to_slot.view(1, 1, 1, n_total).expand(B, nh, T_q, n_total)
    content_per_token = content_scores_slots.gather(3, idx_expand)  # (B, nh, T_q, n_total)

    total = (content_per_token + position_scores) * scale  # (B, nh, T_q, n_total)

    # --- Mask (training chunk causality) ---
    if mask is not None:
        total = total.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(total, dim=-1)  # (B, nh, T_q, n_total)

    # --- Value aggregation: segment-sum per slot, then matmul with slot values ---
    slot_attn = torch.zeros(B, nh, T_q, n_all_slots, device=device, dtype=attn.dtype)
    slot_attn.scatter_add_(3, idx_expand, attn)  # (B, nh, T_q, n_all_slots)
    out = torch.matmul(slot_attn, all_slot_v)    # (B, nh, T_q, v_dim)
    return out
