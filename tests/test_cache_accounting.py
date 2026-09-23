"""Tests for litgpt/cache_accounting.py.

Written for the 32K Dense/SinkWindow/SemanticLogKV comparison's byte-accurate
persistent-cache budget matching. Not executed in the environment that wrote
this file (no GPU/torch there) -- run for real before trusting
measure_cache_bytes.py / calibrate_sinkwindow_w.py's numbers.
"""

import pytest
import torch
import torch.nn as nn

from litgpt.cache_accounting import (
    assert_no_training_only_state,
    cache_bytes,
    live_extra_bytes,
    structural_bytes,
)
from litgpt.log_kv_cache import LogStructuredKVCache
from litgpt.model import build_rope_cache


def make_plain_cache(*, batch_size=1, n_groups=2, max_seq_length=64, k_dim=8, v_dim=8, B=4, recent_size=2):
    """Non-semantic LogStructuredKVCache -- fewer buffers, easy to hand-compute."""
    return LogStructuredKVCache(
        (batch_size, n_groups, max_seq_length, k_dim),
        (batch_size, n_groups, max_seq_length, v_dim),
        B=B,
        recent_size=recent_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def make_semantic_cache(*, batch_size=1, n_groups=2, max_seq_length=64, k_dim=8, v_dim=8, B=4, recent_size=2, K_max=2,
                         cos=None, sin=None):
    if cos is None or sin is None:
        cos, sin = build_rope_cache(max_seq_length, k_dim)
    return LogStructuredKVCache(
        (batch_size, n_groups, max_seq_length, k_dim),
        (batch_size, n_groups, max_seq_length, v_dim),
        B=B,
        recent_size=recent_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
        semantic_clusters=True,
        cluster_k_max=K_max,
        semantic_s_h=1.0,
        cos_cache=cos,
        sin_cache=sin,
        rope_n_elem=k_dim,
    ), cos, sin


def test_structural_bytes_matches_hand_computed_total_for_plain_cache():
    cache = make_plain_cache(batch_size=1, n_groups=2, max_seq_length=64, k_dim=8, v_dim=8, B=4, recent_size=2)
    expected = 0
    seen_storage_ids = set()
    for _, t in cache.named_buffers(recurse=True):
        if t is None:
            continue
        ptr = t.untyped_storage().data_ptr()
        if ptr in seen_storage_ids:
            continue
        seen_storage_ids.add(ptr)
        expected += t.untyped_storage().nbytes()
    assert expected > 0, "fixture should have at least one real buffer"
    assert structural_bytes(cache) == expected


def test_structural_bytes_per_buffer_breakdown_resums_to_total():
    cache = make_plain_cache()
    total, breakdown = structural_bytes(cache, per_buffer=True)
    assert sum(entry["bytes"] for entry in breakdown) == total
    # every named buffer appears exactly once in the breakdown, deduped or not
    named_count = sum(1 for _, t in cache.named_buffers(recurse=True) if t is not None)
    assert len(breakdown) == named_count


def test_cos_sin_aliasing_does_not_scale_with_layer_count():
    """Regression pin for the storage-dedup fix: when several semantic caches
    share one underlying cos/sin storage (as every layer's LogStructuredKVCache
    does via GPT.cos/GPT.sin when config.rope_indices is None), the RoPE
    table's bytes must be counted once, not once per layer.
    """
    cos, sin = build_rope_cache(64, 8)
    cache_a, _, _ = make_semantic_cache(cos=cos, sin=sin)
    cache_b, _, _ = make_semantic_cache(cos=cos, sin=sin)
    assert cache_a.cos_cache.untyped_storage().data_ptr() == cache_b.cos_cache.untyped_storage().data_ptr(), (
        "fixture assumption broken: the two caches must actually alias the same cos storage for this test to mean anything"
    )

    one_layer = structural_bytes(cache_a)

    two_layers = nn.Module()
    two_layers.layer0 = cache_a
    two_layers.layer1 = cache_b
    two_layers_bytes = structural_bytes(two_layers)

    rope_bytes = cos.untyped_storage().nbytes() + sin.untyped_storage().nbytes()
    non_rope_bytes_per_layer = one_layer - rope_bytes
    # Two independent per-layer buffer sets, but the shared RoPE table counted once.
    assert two_layers_bytes == 2 * non_rope_bytes_per_layer + rope_bytes
    assert two_layers_bytes < 2 * one_layer, "naive per-buffer summing (no dedup) would double the RoPE table's bytes"


def test_extra_live_tensors_empty_until_mid_decode_state_set():
    cache, _, _ = make_semantic_cache()
    assert cache.extra_live_tensors() == []
    assert live_extra_bytes(cache) == 0


def test_extra_live_tensors_reports_mid_decode_workspace_separately_from_structural():
    cache, _, _ = make_semantic_cache()
    structural_before = structural_bytes(cache)

    workspace_k = torch.zeros(1, 2, 6, 8)
    workspace_v = torch.zeros(1, 2, 6, 8)
    cache._mid_decode_state = (("fake-key",), None, (workspace_k, workspace_v), 0)

    assert live_extra_bytes(cache) == (
        workspace_k.untyped_storage().nbytes() + workspace_v.untyped_storage().nbytes()
    )
    # Structural (config-only) number must be unaffected by transient decode state.
    assert structural_bytes(cache) == structural_before

    result = cache_bytes(cache)
    assert result["structural_bytes"] == structural_before
    assert result["live_extra_bytes"] > 0
    assert result["total_bytes"] == result["structural_bytes"] + result["live_extra_bytes"]


def test_op_log_is_invisible_to_named_buffers_and_assert_catches_it_anyway():
    cache, _, _ = make_semantic_cache()
    # op_log/op_log_len are plain instance attributes (see
    # LogStructuredKVCache.__init__), never register_buffer -- confirm they
    # genuinely don't show up in the buffer walk regardless.
    buffer_names = {name for name, _ in cache.named_buffers(recurse=True)}
    assert "op_log" not in buffer_names
    assert "op_log_len" not in buffer_names

    assert_no_training_only_state(cache)  # None/None at construction -- must not raise

    cache.op_log = torch.zeros(1, 2, 12, 4, dtype=torch.int32)
    cache.op_log_len = torch.zeros(1, 2, dtype=torch.int32)
    with pytest.raises(AssertionError):
        assert_no_training_only_state(cache)
    # And structural_bytes is unaffected either way -- op_log was never counted.
    assert "op_log" not in {name for name, _ in cache.named_buffers(recurse=True)}


def test_disabled_second_order_buffers_are_absent_not_zero_sized():
    cache = LogStructuredKVCache(
        (1, 1, 32, 8), (1, 1, 32, 8), B=4, allocate_second_order=False, device=torch.device("cpu"), dtype=torch.float32,
    )
    names = {name for name, _ in cache.named_buffers(recurse=True)}
    assert "level_sigma_u" not in names
    assert "level_sigma2" not in names
    assert "level_gamma_a" not in names
    assert "level_gamma_b" not in names
    assert "level_gamma" not in names
