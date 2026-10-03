"""Beta selection, mass-weighted compaction, lane boundaries and replay."""
from unittest.mock import patch

import pytest
import torch

from litgpt.alpha_log_kv import select_spans
from litgpt.beta_log_kv import make_layout, merge_blocks
from litgpt.log_kv_cache import _SemanticReplayUpdates
from test_alpha_log_kv import cache_for


def entry_block(values, weights, device="cpu", dtype=torch.float32):
    v = torch.tensor(values, device=device, dtype=dtype)[:, None].repeat(1, 8)
    w = torch.tensor(weights, device=device, dtype=torch.float32)
    pos = torch.arange(len(values), device=device, dtype=torch.int64)
    return (torch.ones_like(v), v, w, None, None, None, None, None,
            pos, pos.clone(), pos * w.long(), pos.clone(), torch.zeros_like(pos, dtype=torch.bool))


def test_novelty_rescues_singleton_and_handles_empty_history():
    k = torch.tensor([1., 3.]).view(1, 1, 2, 1).expand(1, 4, 2, 2)
    centers = torch.ones(1, 4, 2, 2)
    valid = torch.tensor([[[True, False]] * 4])
    args = (k, k, [[]], [[]], [[0, 1]], [[True, True]], 0, 1, 1)
    assert select_spans(*args).keep == [[0]]
    result = select_spans(*args, beta_novelty=True, centroids=centers, centroid_valid=valid)
    assert result.keep == [[1]] and result.archive == [[0]]
    result = select_spans(*args, beta_novelty=True, centroids=centers, centroid_valid=valid & False)
    assert result.keep == [[0]]


def test_top_groups_preserve_sparse_signal():
    # Span 0 is unusual in two groups; span 1 is moderately unusual in all four.
    # Top-2 should favor span 0, whereas averaging all groups would favor span 1.
    k = torch.ones(1, 4, 2, 1)
    k[:, :2, 0] = 3.
    k[:, :, 1] = 2.
    result = select_spans(k, k, [[]], [[]], [[0, 1]], [[True, True]], 0, 1, 1,
                          beta_novelty=True, centroids=torch.ones(1, 4, 1, 1),
                          centroid_valid=torch.ones(1, 4, 1, dtype=torch.bool))
    assert result.keep == [[0]]


def test_adaptive_compaction_conserves_mass_and_never_crosses_lanes():
    # Two lanes with two-entry tails; concatenating then grouping by four would
    # merge entries 4..7 across lanes and lose their independent order.
    assert make_layout([0, 6, 0, 2], "cpu").tolist() == [[0, 4, 0], [4, 2, 2], [6, 2, 3]]
    assert make_layout([], "cpu").shape == (0, 3)
    block = entry_block([10, 0, 0, 0, 7, 9, 20, 30], [1, 2, 3, 4, 1, 1, 2, 3])
    result, cuts = merge_blocks(block, [6, 2])
    assert cuts.tolist() == [1, 1, 1]
    torch.testing.assert_close(result[2], torch.tensor([1., 9., 2., 5.]))
    torch.testing.assert_close(result[1][:, 0], torch.tensor([10., 0., 8., 26.]))
    assert result[10].sum() == block[10].sum()
    for f in (0, 1):
        torch.testing.assert_close((result[f] * result[2][:, None]).sum(0),
                                   (block[f] * block[2][:, None]).sum(0))
    with patch("litgpt.beta_log_kv.select_cuts", side_effect=AssertionError("reselected")):
        replay, _ = merge_blocks(block, [6, 2], cuts=cuts)
    for first, second in zip(result, replay):
        if first is not None:
            torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_compaction_loss_never_exceeds_fixed_pairs():
    torch.manual_seed(83)
    block = list(entry_block([0] * 40, torch.randint(1, 10, (40,)).tolist()))
    block[0], block[1] = torch.randn(40, 7), torch.randn(40, 11)
    layout = make_layout([40], "cpu")
    _, cuts = merge_blocks(block, [40])
    for start, cut in zip(layout[:, 0].tolist(), cuts.tolist()):
        w = block[2][start:start + 4]

        def loss(split):
            result = 0.
            for x in block[:2]:
                x = x[start:start + 4]
                energy = (x.square().sum(-1) * w).sum() / w.sum()
                for part in (slice(0, split), slice(split, 4)):
                    mean = (x[part] * w[part, None]).sum(0) / w[part].sum()
                    result += ((x[part] - mean).square().sum(-1) * w[part]).sum() / energy
            return result

        assert loss(cut) <= loss(2) + 1e-5


def test_update_replay_uses_recorded_cuts_and_exact_selection():
    torch.manual_seed(29)
    cache = cache_for(beta_novelty=True, beta_adaptive_merge=True)
    updates = _SemanticReplayUpdates()
    k, v = torch.randn(2, 2, 64, 8), torch.randn(2, 2, 64, 8)
    positions = torch.arange(64).expand(2, -1)
    kwargs = dict(positions_host=positions.tolist(), span_ends=[[i % 4 == 3 for i in range(64)]] * 2)
    cache.begin_op_log()
    with cache._semantic_update_context(updates):
        cache.route_and_flush_batch(k, v, positions, record_op_log=True, **kwargs)
    actions = next(iter(updates.flushes.values()))[1]
    assert any(kind == "append_beta" and metadata[2] for kind, metadata, _ in actions)
    fields = ("level_k", "level_v", "level_w", "level_sum_wp", "level_order",
              "alpha_k_raw", "alpha_v", "alpha_pos", "alpha_valid", "centroid", "n_total")
    expected = {name: getattr(cache, name).clone() for name in fields}
    cache.reset_parameters()
    with cache._semantic_update_context(updates, replay=True), \
         patch("litgpt.alpha_log_kv.select_spans", side_effect=AssertionError("reselected span")), \
         patch("litgpt.beta_log_kv.select_cuts", side_effect=AssertionError("reselected cut")):
        cache.route_and_flush_batch(k, v, positions, replay_op_log_host=(), **kwargs)
    for name, value in expected.items():
        torch.testing.assert_close(getattr(cache, name), value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_compaction_matches_reference_and_replays(dtype):
    from litgpt.log_kv_updates_triton import merge_scatter

    storage = list(entry_block([10, 0, 0, 0, 7, 9, 11, 13, 0, 0], [1, 2, 3, 4, 1, 1, 2, 3, 0, 0], "cuda", dtype))
    stage = list(entry_block([20, 30, 30, 30, 17.13], [1, 2, 3, 4, 1], "cuda"))
    # Different, non-power-of-two feature sizes; FP32 staging into BF16 storage.
    for block in (storage, stage):
        block[0] = block[0][:, :7].contiguous()
        block[1] = block[1].repeat(1, 2)[:, :11].contiguous()
    si, gi = torch.arange(8, device="cuda"), torch.arange(5, device="cuda")
    pi = torch.tensor([0, 1, 2, 3, 4, 5, 8, 9, 10, 11], device="cuda")
    vi = torch.tensor([6, 7, 12], device="cuda")
    dst, clear = torch.tensor([1, 4, 8], device="cuda"), torch.tensor([6, 7], device="cuda")
    pool = tuple(torch.cat((f[si], s[gi].to(f.dtype))) if f is not None else None for f, s in zip(storage, stage))
    expected, expected_cuts = merge_blocks(tuple(x[pi] if x is not None else None for x in pool), [6, 4])

    def run(cuts=None, prepacked=False):
        fields = tuple(x.clone() if x is not None else None for x in storage)
        kwargs = dict(beta_lengths=[6, 4], beta_cuts=cuts)
        if prepacked:
            kwargs["beta_layout"] = make_layout([6, 4], "cuda")
            with patch("litgpt.beta_log_kv.make_layout", side_effect=AssertionError("redundant layout upload")):
                result = merge_scatter(fields, stage, si, gi, pi, vi, dst, clear, **kwargs)
        else:
            result = merge_scatter(fields, stage, si, gi, pi, vi, dst, clear, **kwargs)
        for f, original, p in zip(fields, storage, pool):
            if f is not None:
                updated = original.clone()
                updated[dst] = p[vi]
                updated[clear] = 0
                torch.testing.assert_close(f, updated, rtol=0, atol=0)
        return result

    actual, cuts = run(prepacked=True)
    torch.testing.assert_close(cuts, expected_cuts)
    replay, _ = run(cuts)
    for a, b, c in zip(expected, actual, replay):
        if a is not None:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(b, c, rtol=0, atol=0)
