from types import SimpleNamespace

import pytest
import torch

from litgpt.kv_compression import cache_compression_snapshot, summarize_compression
from litgpt.log_kv_cache import LogStructuredKVCache
from litgpt.model import KVCache


def model_with(*caches):
    layers = [SimpleNamespace(attn=SimpleNamespace(kv_cache=cache)) for cache in caches]
    return SimpleNamespace(transformer=SimpleNamespace(h=layers))


def log_cache():
    shape = (2, 3, 128, 8)
    return LogStructuredKVCache(shape, shape, B=4, recent_size=4)


def append_tokens(cache, count):
    generator = torch.Generator().manual_seed(17)
    for start in range(0, count, cache.recent_size):
        shape = (cache.batch_size, cache.n_groups, min(cache.recent_size, count - start), 8)
        cache.add_recent(torch.randn(shape, generator=generator), torch.randn(shape, generator=generator))


def test_dense_cache_ignores_preallocated_capacity():
    shape = (2, 3, 128, 8)
    caches = [KVCache(shape, shape) for _ in range(2)]
    for cache in caches:
        cache(torch.arange(5), torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8))
    snapshot = cache_compression_snapshot(model_with(*caches), 5)
    assert snapshot["layer_group_count"] == 12
    assert snapshot["recent_slots"] == snapshot["retained_slots"] == snapshot["dense_slots"] == 60
    assert snapshot["retained_slots_per_layer_group"] == 5
    assert snapshot["kv_retention_ratio"] == snapshot["kv_compression_factor"] == 1
    assert snapshot["kv_saving_ratio"] == 0


def test_sliding_window_counts_only_populated_positions():
    shape = (2, 3, 4, 8)
    cache = KVCache(shape, shape, is_sliding_window=True, sliding_window_size=4)
    for start, stop in ((0, 2), (2, 6), (6, 10)):
        cache(torch.arange(start, stop), torch.randn(2, 3, stop - start, 8), torch.randn(2, 3, stop - start, 8))
        snapshot = cache_compression_snapshot(model_with(cache), stop)
        assert snapshot["recent_slots"] == min(stop, 4) * 6
        assert snapshot["kv_retention_ratio"] == min(stop, 4) / stop


@pytest.mark.parametrize("token_count", [4, 64])
def test_log_cache_uses_committed_tokens_and_all_layers(token_count):
    caches = [log_cache(), log_cache()]
    for cache in caches:
        append_tokens(cache, token_count)
    snapshot = cache_compression_snapshot(model_with(*caches), processed_tokens=999)
    assert snapshot["processed_tokens"] == token_count
    assert snapshot["dense_slots"] == token_count * 12
    assert snapshot["retained_slots"] == sum(cache.total_slots * 6 for cache in caches)
    assert snapshot["kv_compression_factor"] == snapshot["dense_slots"] / snapshot["retained_slots"]
    if token_count == 4:
        assert snapshot["kv_retention_ratio"] == 1
    else:
        assert 0 < snapshot["kv_retention_ratio"] < 1


def test_alpha_counts_unfinished_spans_and_each_group():
    shape = (2, 2, 128, 8)
    cache = LogStructuredKVCache(
        shape, shape, B=8, recent_size=8, semantic_clusters=True, cluster_k_max=4,
        semantic_anchor_mode="mid", allocate_second_order=False, semantic_flush_granularity=8,
        cos_cache=torch.ones(128, 8), sin_cache=torch.zeros(128, 8), rope_n_elem=8,
        alpha_exact_tokens=8, alpha_span_max_tokens=4,
    )
    generator = torch.Generator().manual_seed(17)
    k, v = (torch.randn(2, 2, 96, 8, generator=generator) for _ in range(2))
    ends = [[i % 3 == 2 for i in range(96)], [i % 4 == 3 for i in range(96)]]
    saw_unfinished_span = saw_unequal_groups = False
    for start in range(0, 96, 8):
        cache.add_recent(k[:, :, start:start + 8], v[:, :, start:start + 8],
                         k_raw=k[:, :, start:start + 8], span_ends=[row[start:start + 8] for row in ends])
        snapshot = cache_compression_snapshot(model_with(cache), start + 8)
        exact = int(cache.alpha_valid.sum().item()) * 2
        hierarchical = int((cache.level_w > 0).sum().item())
        assert snapshot["exact_slots"] == exact
        assert snapshot["hierarchical_slots"] == hierarchical
        assert snapshot["retained_slots"] == cache.recent_count * 4 + exact + hierarchical
        saw_unfinished_span |= any(not closed for row in cache._alpha_spans for _, closed in row)
        saw_unequal_groups |= hierarchical != int((cache.level_w[0, 0] > 0).sum().item()) * 4
    assert saw_unfinished_span
    assert saw_unequal_groups


def test_summary_weights_entries_instead_of_request_ratios():
    short, long = log_cache(), log_cache()
    append_tokens(short, 4)
    append_tokens(long, 64)
    records = [cache_compression_snapshot(model_with(cache), cache.token_count) for cache in (short, long)]
    result = summarize_compression(records)
    expected = sum(row["retained_slots"] for row in records) / sum(row["dense_slots"] for row in records)
    assert result["request_count"] == 2
    assert result["processed_tokens"] == 68
    assert result["kv_retention_ratio"] == expected
    assert result["kv_retention_ratio"] != sum(row["kv_retention_ratio"] for row in records) / 2
    assert result["kv_saving_ratio"] == 1 - expected
    assert result["kv_compression_factor"] == 1 / expected


def test_empty_cache_and_empty_summary_have_no_ratio():
    for result in (cache_compression_snapshot(model_with(log_cache()), 0), summarize_compression([])):
        assert result["retained_slots"] == result["dense_slots"] == 0
        assert result["kv_retention_ratio"] is None
        assert result["kv_saving_ratio"] is None
        assert result["kv_compression_factor"] is None
    assert summarize_compression([])["request_count"] == 0


def test_inconsistent_layer_token_counts_are_rejected():
    caches = [log_cache(), log_cache()]
    append_tokens(caches[0], 4)
    with pytest.raises(ValueError, match="token counts differ"):
        cache_compression_snapshot(model_with(*caches), 4)


def test_missing_cache_is_rejected():
    with pytest.raises(ValueError, match="initialized cache"):
        cache_compression_snapshot(model_with(None), 4)
