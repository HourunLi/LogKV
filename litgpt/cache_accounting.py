"""Byte-accurate persistent KV-cache accounting.

Walks the live tensors on a cache module (or a full ``GPT`` containing one
cache per layer) and reports *real allocated storage* bytes: not
``tensor.numel() * element_size()`` on a possibly-sliced view, and not
``torch.cuda.max_memory_allocated()`` (allocator-reserved headroom, which the
32K fair-comparison experiment's budget-matching protocol explicitly forbids
using as the matching metric -- it reflects the allocator's bookkeeping, not
what any mechanism actually holds).

Two numbers are reported, deliberately kept separate rather than merged:

- ``structural_bytes``: a pure function of static config -- the byte total of
  every unique tensor storage reachable via ``module.named_buffers()``. This
  is what W-calibration / cross-mechanism budget matching must use, because
  it is reproducible from a config alone (no live weights or a populated
  cache are needed -- see ``scripts/measure_cache_bytes.py``).
- ``live_extra_bytes``: transient, decode-timing-dependent workspace tensors
  that a cache class exposes through a duck-typed ``extra_live_tensors()``
  hook (e.g. ``LogStructuredKVCache``'s ``_mid_decode_state`` packed-KV
  buffers, only allocated under ``semantic_anchor_mode="mid"``, sized from
  ``recent_count`` at first decode use -- not a function of config alone).
  Informational only. Do not fold this into a frozen budget number: it
  varies run to run and step to step.

Storage, not buffer count, is what's summed: several named buffers can alias
the same underlying allocation (the clearest case in this codebase --
``LogStructuredKVCache.cos_cache``/``sin_cache`` are literally ``GPT.cos``/
``sin`` themselves when ``config.rope_indices is None``, the Qwen3 default,
so every layer's semantic cache instance shares one RoPE-table storage; a
naive per-buffer sum would multiply it by ``n_layer`` and call that "cache
memory", which is both wrong and not even about compression -- the RoPE
table is common infrastructure all three mechanisms need equally).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _storage_ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def _storage_nbytes(t: torch.Tensor) -> int:
    return t.untyped_storage().nbytes()


def structural_bytes(module: nn.Module, *, per_buffer: bool = False):
    """Sum of unique-storage bytes across ``module.named_buffers(recurse=True)``.

    ``nn.Module._named_members`` already skips buffers registered as
    ``None`` (e.g. the second-order ``level_sigma_*``/``level_gamma_*``
    buffers when ``allocate_second_order=False``), so no extra filtering for
    that case is needed here.

    Returns ``total_bytes`` (int), or ``(total_bytes, breakdown)`` when
    ``per_buffer=True`` -- ``breakdown`` is a list of per-named-buffer dicts
    with ``bytes=0``/``deduped=True`` for every buffer after the first that
    shares a storage, so re-summing ``breakdown[*]["bytes"]`` reproduces
    ``total_bytes`` without double counting.
    """
    named = [(name, t) for name, t in module.named_buffers(recurse=True) if t is not None]
    seen_bytes: dict[int, int] = {}
    breakdown: list[dict[str, Any]] = []
    for name, t in named:
        ptr = _storage_ptr(t)
        first_owner = ptr not in seen_bytes
        if first_owner:
            seen_bytes[ptr] = _storage_nbytes(t)
        if per_buffer:
            breakdown.append(
                {
                    "name": name,
                    "shape": tuple(t.shape),
                    "dtype": str(t.dtype),
                    "bytes": seen_bytes[ptr] if first_owner else 0,
                    "deduped": not first_owner,
                }
            )
    total = sum(seen_bytes.values())
    return (total, breakdown) if per_buffer else total


def live_extra_bytes(module: nn.Module, *, per_module: bool = False):
    """Sum of unique-storage bytes from every submodule's ``extra_live_tensors()``.

    Modules that don't define the hook are skipped (``getattr(..., None)``),
    so this works unchanged whether or not a given cache class implements it.
    """
    seen_bytes: dict[int, int] = {}
    per: list[dict[str, Any]] = []
    for name, m in module.named_modules():
        hook = getattr(m, "extra_live_tensors", None)
        if hook is None:
            continue
        extra = hook()
        if not extra:
            continue
        module_bytes = 0
        for t in extra:
            if t is None:
                continue
            ptr = _storage_ptr(t)
            if ptr not in seen_bytes:
                seen_bytes[ptr] = _storage_nbytes(t)
                module_bytes += seen_bytes[ptr]
        if per_module:
            per.append({"module": name, "bytes": module_bytes, "count": len(extra)})
    total = sum(seen_bytes.values())
    return (total, per) if per_module else total


def cache_bytes(module: nn.Module, *, per_buffer: bool = False) -> dict[str, Any]:
    """Primary entry point: structural + live_extra byte counts for ``module``.

    ``module`` can be a single cache instance (e.g. a freshly-built
    ``LogStructuredKVCache`` or ``SinkWindowKVCache`` for W-calibration) or a
    full ``GPT`` with one cache per layer (structural bytes then already
    correctly dedupe shared RoPE-table storage across layers).
    """
    if per_buffer:
        s_total, s_breakdown = structural_bytes(module, per_buffer=True)
        l_total, l_breakdown = live_extra_bytes(module, per_module=True)
        return {
            "structural_bytes": s_total,
            "live_extra_bytes": l_total,
            "total_bytes": s_total + l_total,
            "structural_breakdown": s_breakdown,
            "live_extra_breakdown": l_breakdown,
        }
    s_total = structural_bytes(module, per_buffer=False)
    l_total = live_extra_bytes(module, per_module=False)
    return {
        "structural_bytes": s_total,
        "live_extra_bytes": l_total,
        "total_bytes": s_total + l_total,
    }


def assert_no_training_only_state(cache: Any) -> None:
    """Defensive check for eval-time accounting scripts.

    ``op_log``/``op_log_len`` (LogKV's training-replay metadata) are plain
    instance attributes, not ``register_buffer``s (see
    ``LogStructuredKVCache.__init__``), so they are already structurally
    invisible to ``structural_bytes()`` -- nothing needs to filter them out
    of the buffer walk. ``docs/algorithm-spec.md`` states directly: "op_log
    不属于 serving cache；推理路径不分配" (op_log is not part of the serving
    cache; the inference path never allocates it).

    This assertion instead guards against a different mistake: measuring a
    cache that is still live from *training* (where ``begin_op_log()`` has
    allocated real tensors into these attributes) and mistaking that reading
    for a clean inference-time budget number. Call this before trusting a
    ``cache_bytes()`` result as the frozen persistent-cache budget.
    """
    op_log = getattr(cache, "op_log", None)
    op_log_len = getattr(cache, "op_log_len", None)
    if op_log is not None or op_log_len is not None:
        raise AssertionError(
            "cache has live training op_log state (op_log/op_log_len is not "
            "None) -- this is not part of the inference-time persistent "
            "cache budget. Measure a cache built for eval via "
            "GPT.set_log_kv_cache(), not one still attached from "
            "enable_log_kv_training() training mode."
        )
