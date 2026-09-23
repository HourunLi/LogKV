"""SinkWindow KV cache: keep the first ``S`` tokens ("sink") plus the most
recent ``W`` tokens ("window") per layer, with real storage recycling at
inference and real token positions preserved through standard post-RoPE
storage -- the same convention plain ``KVCache`` already uses (RoPE is
applied once at write time using the caller's real position; nothing here
remaps or re-derives position at read time).

This is the baseline for the 32K Dense/SinkWindow/SemanticLogKV comparison.
Deliberately named SinkWindow, not "SWA" or "StreamingLLM": it implements
exactly "keep first S + most recent W", nothing more -- no claim is made
that it reproduces StreamingLLM's full training/position protocol.

Design note -- why this does NOT reuse ``KVCache``'s existing ring-buffer +
column-indexed-mask pattern (``is_sliding_window=True``): that path assumes
"column index in the returned buffer == real position" when it slices the
precomputed ``mask_cache`` by ``input_pos``. That assumption only holds
before the buffer first wraps, or for single-token decode. A chunked
multi-token commit whose position range straddles a wraparound boundary
breaks it: a query early in the chunk could read a slot whose *content* is a
later, still-future token from the very same chunk, while the mask still
treats that column as an already-valid old position -- a silent correctness
bug, and one this cache's ``window_size`` (much smaller than a 32K prompt)
would hit on essentially every real prefill.

The fix is architectural, not a patch: this module never builds a mask
indexed by physical storage slot at all. Whatever sink+window content is
already resident before a chunk is committed is *unconditionally* visible to
every query in that chunk (it strictly precedes the chunk by construction --
chunks are only committed into storage *after* being attended to), so
attention is computed as ``[fully-visible frozen prefix] + [causal within
the new chunk]`` -- the same two-part structure
``litgpt.log_kv_cache.log_kv_chunk_attention`` already uses for LogKV's own
chunked commits. Physical slot order inside the frozen prefix is irrelevant
(softmax is a sum over independently-scored keys, order-invariant), so no
per-slot position bookkeeping is needed for correctness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _batched_index_copy_1d(t: torch.Tensor, dim: int, idx: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """Thin re-export of litgpt.model.batched_index_copy_'s 1-D-index path
    (shared index across the whole batch -- see SinkWindowKVCache's
    single-shared-position-range contract, mirroring LogStructuredKVCache's
    own ``_assert_log_kv_input_pos_contiguous`` convention). Imported lazily
    to avoid a hard circular import (litgpt.model imports this module for
    the CausalSelfAttention wiring).
    """
    from litgpt.model import batched_index_copy_

    return batched_index_copy_(t, dim, idx, val)


class SinkWindowKVCache(nn.Module):
    """Fixed sink region (first ``sink_size`` real tokens, write-once) plus a
    ring-buffer window region (most recent ``window_size`` real tokens).
    Physical storage is ``O(sink_size + window_size)``, independent of
    sequence length -- the whole point of this baseline's bounded cache.

    Buffers store standard post-RoPE keys, exactly like ``KVCache``.

    Contract (mirrors ``LogStructuredKVCache``'s existing one): all batch
    rows share a single, contiguous, append-only position range. Multi-
    sample-with-independent-positions batching is not supported -- see
    ``assert_input_pos_contiguous``.
    """

    def __init__(
        self,
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        sink_size: int,
        window_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        batch_size, n_groups, _, k_dim = k_shape
        _, _, _, v_dim = v_shape
        if sink_size < 0:
            raise ValueError(f"sink_size must be >= 0, got {sink_size}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        self.batch_size = batch_size
        self.n_groups = n_groups
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)
        # Real tokens committed so far. Plain scalar, not per-batch-row --
        # valid only under the shared-position-range contract above.
        self.token_count: int = 0

        self.register_buffer(
            "sink_k",
            torch.zeros(batch_size, n_groups, self.sink_size, k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "sink_v",
            torch.zeros(batch_size, n_groups, self.sink_size, v_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "window_k",
            torch.zeros(batch_size, n_groups, self.window_size, k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "window_v",
            torch.zeros(batch_size, n_groups, self.window_size, v_dim, device=device, dtype=dtype),
            persistent=False,
        )

    def reset_parameters(self) -> None:
        """Reset all buffers to zero, in place -- no reallocation (same
        rationale as LogStructuredKVCache/KVCache's own reset_parameters:
        rebuilding per-request fragments the CUDA allocator on long eval
        runs).
        """
        self.token_count = 0
        torch.nn.init.zeros_(self.sink_k)
        torch.nn.init.zeros_(self.sink_v)
        torch.nn.init.zeros_(self.window_k)
        torch.nn.init.zeros_(self.window_v)

    def extra_live_tensors(self) -> list[torch.Tensor]:
        """Duck-typed hook for litgpt/cache_accounting.py. Every buffer here
        is already structural (a fixed function of sink_size/window_size
        alone) -- unlike LogStructuredKVCache's mid-decode packed workspace,
        SinkWindow has no timing-dependent transient allocation, so this is
        always empty.
        """
        return []

    @property
    def sink_filled(self) -> int:
        return min(self.token_count, self.sink_size)

    @property
    def window_filled(self) -> int:
        return min(max(0, self.token_count - self.sink_size), self.window_size)

    def read_frozen(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Currently-resident sink+window content, concatenated along the
        sequence dim. Order among the returned positions is unspecified and
        irrelevant for correctness -- see module docstring.
        """
        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []
        if self.sink_filled > 0:
            parts_k.append(self.sink_k[:, :, : self.sink_filled, :])
            parts_v.append(self.sink_v[:, :, : self.sink_filled, :])
        wf = self.window_filled
        if wf > 0:
            if wf < self.window_size:
                parts_k.append(self.window_k[:, :, :wf, :])
                parts_v.append(self.window_v[:, :, :wf, :])
            else:
                parts_k.append(self.window_k)
                parts_v.append(self.window_v)
        if not parts_k:
            empty_k = self.sink_k.new_zeros(self.batch_size, self.n_groups, 0, self.k_dim)
            empty_v = self.sink_v.new_zeros(self.batch_size, self.n_groups, 0, self.v_dim)
            return empty_k, empty_v
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    def commit(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Commit T newly-computed post-RoPE k/v into sink and/or window
        storage. The T tokens are assumed to be exactly
        [self.token_count, self.token_count + T) in real position -- the
        caller (CausalSelfAttention._sink_window_inference_forward) is
        responsible for that contract, mirroring
        LogStructuredKVCache's append-only assumption.
        """
        T = k_new.size(-2)
        if T == 0:
            return
        if self.sink_k.dtype != k_new.dtype:
            self.sink_k = self.sink_k.to(k_new.dtype)
            self.window_k = self.window_k.to(k_new.dtype)
        if self.sink_v.dtype != v_new.dtype:
            self.sink_v = self.sink_v.to(v_new.dtype)
            self.window_v = self.window_v.to(v_new.dtype)

        remaining_k, remaining_v = k_new, v_new
        if self.token_count < self.sink_size:
            n_fill = min(self.sink_size - self.token_count, T)
            self.sink_k[:, :, self.token_count : self.token_count + n_fill, :] = remaining_k[..., :n_fill, :]
            self.sink_v[:, :, self.token_count : self.token_count + n_fill, :] = remaining_v[..., :n_fill, :]
            remaining_k = remaining_k[..., n_fill:, :]
            remaining_v = remaining_v[..., n_fill:, :]
            self.token_count += n_fill

        n_window_new = remaining_k.size(-2)
        if n_window_new == 0:
            return
        if n_window_new > self.window_size:
            raise ValueError(
                f"commit() received {n_window_new} tokens for the window region in one call, but "
                f"window_size is only {self.window_size} -- the caller must chunk to at most "
                "window_size tokens per commit (a larger single call would need to write the same "
                "ring-buffer slot more than once, which batched_index_copy_ does not guarantee a "
                "deterministic result for)."
            )
        tokens_fed_to_window_before = self.token_count - self.sink_size
        write_head = tokens_fed_to_window_before % self.window_size
        slot_idx = (write_head + torch.arange(n_window_new, device=remaining_k.device)) % self.window_size
        _batched_index_copy_1d(self.window_k, -2, slot_idx, remaining_k)
        _batched_index_copy_1d(self.window_v, -2, slot_idx, remaining_v)
        self.token_count += n_window_new


def assert_input_pos_contiguous(cache: SinkWindowKVCache, input_pos: torch.Tensor, T: int) -> None:
    """Validate the append-only SinkWindowKVCache contract -- mirrors
    LogStructuredKVCache._assert_log_kv_input_pos_contiguous exactly (same
    reason: cache.commit() tracks position with a scalar counter, so
    inference must feed a single, shared, contiguous position range).
    """
    if input_pos.dim() == 2:
        if not torch.equal(input_pos, input_pos[:1].expand_as(input_pos)):
            raise ValueError(
                "SinkWindowKVCache does not support per-sample input_pos; all batch rows must "
                "share the same position range."
            )
        first_row = input_pos[0]
    else:
        first_row = input_pos
    if first_row.numel() != T:
        raise ValueError(f"input_pos.shape[-1] = {first_row.numel()} != {T} = T")
    expected = torch.arange(cache.token_count, cache.token_count + T, device=first_row.device)
    if not torch.equal(first_row, expected):
        raise ValueError(
            f"SinkWindowKVCache requires contiguous append-only input_pos: expected "
            f"[{cache.token_count}, {cache.token_count + T}), got {first_row.tolist()}"
        )


def sink_window_chunk_attention(
    q_chunk: torch.Tensor,
    k_chunk_new: torch.Tensor,
    v_chunk_new: torch.Tensor,
    frozen_k: torch.Tensor,
    frozen_v: torch.Tensor,
    *,
    scale: float,
    enable_gqa: bool = True,
) -> torch.Tensor:
    """Inference-time chunk attention: [frozen prefix, fully visible] +
    [current chunk, causal]. Mirrors log_kv_chunk_attention's structure.

    All tensors are (B, n_query_groups or n_head, T, head_size). Call this
    BEFORE committing k_chunk_new/v_chunk_new into the cache (frozen_k/v must
    not already include the current chunk).
    """
    Tc = q_chunk.size(-2)
    device = q_chunk.device
    n_frozen = frozen_k.size(-2)

    q_idx = torch.arange(Tc, device=device).view(-1, 1)
    k_idx = torch.arange(Tc, device=device).view(1, -1)
    causal_chunk = k_idx <= q_idx

    if n_frozen > 0:
        k_all = torch.cat([frozen_k, k_chunk_new], dim=-2)
        v_all = torch.cat([frozen_v, v_chunk_new], dim=-2)
        visible = torch.cat(
            [torch.ones(Tc, n_frozen, dtype=torch.bool, device=device), causal_chunk],
            dim=-1,
        )
    else:
        k_all, v_all = k_chunk_new, v_chunk_new
        visible = causal_chunk

    attn_mask = torch.zeros(Tc, visible.size(-1), dtype=q_chunk.dtype, device=device)
    attn_mask.masked_fill_(~visible, float("-inf"))
    return F.scaled_dot_product_attention(
        q_chunk, k_all, v_all, attn_mask=attn_mask.view(1, 1, Tc, -1), scale=scale, enable_gqa=enable_gqa,
    )


def sink_window_train_chunk_attention(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    *,
    sink_size: int,
    window_size: int,
    scale: float,
    chunk_size: int | None = None,
    enable_gqa: bool = True,
) -> torch.Tensor:
    """Full-sequence SinkWindow attention for training, chunked so the score
    matrix stays O(T * (sink_size + window_size + chunk_size)) instead of a
    single dense T*T additive mask (a real concern at 32K: T=32768 makes a
    T*T mask ~4GB in fp32 per layer and forces SDPA off the flash path).

    The task spec explicitly permits an "efficient implementation satisfying
    the same mask" for training (it does not need real storage recycling,
    only inference does), and this needs no autograd replay trick the way
    LogKV's training path does: SinkWindow's visible set is a static
    function of position alone, never of routed/clustered content, so there
    is no online decision process whose backward pass needs replaying
    against a since-mutated cache -- ordinary autograd over this chunked
    loop, slicing directly out of the already-materialized q_full/k_full/
    v_full, is correct as written.

    q_full/k_full/v_full are (B, n_query_groups or n_head, T, head_size)
    (post-RoPE, already GQA-ungrouped or not, matching enable_gqa).
    """
    T = q_full.size(-2)
    device = q_full.device
    chunk_size = chunk_size or window_size
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    outputs = []
    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        window_start = max(0, c0 - window_size + 1)
        if window_start <= sink_size:
            # Sink and window ranges overlap/merge at this point in the
            # sequence -- nothing has been evicted yet, every earlier
            # position is still visible (task spec: "长序列下最多有 S+W 个
            # 可见位置", i.e. short sequences see everything).
            key_positions = torch.arange(0, c1, device=device)
        else:
            key_positions = torch.cat(
                [
                    torch.arange(0, sink_size, device=device),
                    torch.arange(window_start, c1, device=device),
                ]
            )
        k_chunk = k_full.index_select(-2, key_positions)
        v_chunk = v_full.index_select(-2, key_positions)
        q_chunk = q_full[..., c0:c1, :]

        q_pos = torch.arange(c0, c1, device=device).view(-1, 1)
        k_pos = key_positions.view(1, -1)
        causal = k_pos <= q_pos
        in_sink = k_pos < sink_size
        in_window = (q_pos - k_pos) < window_size
        visible = causal & (in_sink | in_window)

        attn_mask = torch.zeros(c1 - c0, key_positions.numel(), dtype=q_full.dtype, device=device)
        attn_mask.masked_fill_(~visible, float("-inf"))
        out_chunk = F.scaled_dot_product_attention(
            q_chunk, k_chunk, v_chunk, attn_mask=attn_mask.view(1, 1, c1 - c0, -1), scale=scale, enable_gqa=enable_gqa,
        )
        outputs.append(out_chunk)
    return torch.cat(outputs, dim=-2)
