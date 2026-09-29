"""Whole-flush candidate construction, budget reduction, and training replay."""

from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

from litgpt.config import Config
from litgpt.log_kv_cache import LOG_KV_OP_WARD_MERGE, LogStructuredKVCache, _SemanticReplayUpdates, _SemanticTreeCluster
from litgpt.log_kv_checkpoint import enable_logkv_checkpoint_replay
from litgpt.model import Block, GPT

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA'))]


def cache_for(*, unified=True, batch_size=1, groups=1, K=3, B=3, **kwargs):
    shape = (batch_size, groups, 256, 8)
    return LogStructuredKVCache(
        shape, shape, B=B, recent_size=16, semantic_flush_granularity=16,
        semantic_clusters=True, cluster_k_max=K, cluster_lambda_rel=.25,
        semantic_s_h=1., semantic_unified_route=unified,
        semantic_centroid_backend='parallel',
        semantic_anchor_mode='mid', semantic_pack_backend='torch', allocate_second_order=False,
        cos_cache=torch.ones(256, 8), sin_cache=torch.zeros(256, 8), rope_n_elem=8,
        **kwargs,
    )


def keys(values):
    k = torch.zeros(1, 1, len(values), 8)
    k[0, 0, :, 0] = torch.tensor(values)
    return k


def snapshot(cache):
    buffers = {name: x.clone() for name, x in cache.named_buffers() if not name.startswith('op_log')}
    mirrors = {name: deepcopy(getattr(cache, name)) for name in (
        '_semantic_counts', '_semantic_alive', '_semantic_p_hi_c',
        '_semantic_current_segment', '_semantic_level0_phase', '_semantic_n_total',
    )}
    return buffers, mirrors


def assert_snapshot(cache, expected):
    buffers, mirrors = snapshot(cache)
    assert buffers.keys() == expected[0].keys()
    for name, actual in buffers.items():
        torch.testing.assert_close(actual, expected[0][name], atol=0, rtol=0, msg=name)
    assert mirrors == expected[1]


@pytest.mark.parametrize('device', DEVICES)
def test_parallel_pairs_are_disjoint_and_handle_ties_without_serial_collapse(device):
    cost = torch.zeros(6, 6, device=device)
    cost.fill_diagonal_(float('inf'))
    pairs = LogStructuredKVCache._semantic_unified_pairs(cost, 3)
    assert pairs.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert pairs.unique().numel() == pairs.numel()
    assert LogStructuredKVCache._semantic_unified_pairs(cost, 1).size(0) == 1
    # XOR(1,4)=5: the masked tie sentinel must exceed even non-power-of-two sizes.
    sparse = torch.full((5, 5), float('inf'), device=device)
    sparse[1, 4] = sparse[4, 1] = 1.
    assert LogStructuredKVCache._semantic_unified_pairs(sparse, 2).tolist() == [[1, 4]]


@pytest.mark.parametrize('values', [
    [0., 10., 20., 30., 40., 50.],
    [0., .4, .8, 1.2, 1.6],
])
def test_candidates_are_compact_metadata_and_do_not_obey_final_budget(values):
    cache = cache_for(K=2)
    k = keys(values)
    before = snapshot(cache)
    candidates = cache._semantic_unified_candidates(0, 0, k, [list(range(len(values)))])
    assert_snapshot(cache, before)
    assert sorted(i for c in candidates for i in c.tokens) == list(range(len(values)))
    for candidate in candidates:
        members = k[0, 0, list(candidate.tokens)]
        assert candidate.existing == ()
        assert candidate.n_total == len(candidate.tokens)
        torch.testing.assert_close(candidate.centroid, members.mean(0))
        assert ((members - candidate.centroid).square().sum(-1) <= .25 + 1e-6).all()
    if values[1] == 10.:
        assert len(candidates) == len(values) > cache.K_max
        assert all(len(c.tokens) == 1 for c in candidates)
    else:
        assert len(candidates) > 1  # Adjacent-neighbor chains cannot join distant endpoints.


@pytest.mark.parametrize(('old', 'fresh', 'expected'), [
    ([0., .01], 100., [(2, .005), (4, 100.)]),
    ([0., 100.], .1, [(5, .08), (1, 100.)]),
])
def test_unified_budget_selects_old_old_or_new_old_merge(old, fresh, expected):
    cache = cache_for(K=2, B=8)
    for c, value in enumerate(old):
        k = keys([value])[0, 0, 0]
        cache._semantic_new_cluster(0, 0, c, c, k, k * 2, torch.tensor(c), record=False)
    k = keys([fresh] * 4)
    cache.route_and_flush_batch(k, k * 2, torch.arange(2, 6))
    live = cache.alive[0, 0]
    assert live.sum() == 2
    centers = cache.centroid[0, 0, live, 0]
    counts = cache.n_total[0, 0, live]
    order = centers.argsort()
    torch.testing.assert_close(centers[order], torch.tensor([center for _, center in expected]))
    assert counts[order].tolist() == [count for count, _ in expected]


def test_candidate_members_are_written_as_exact_entries_before_ladder_compaction():
    torch.manual_seed(19)
    cache = cache_for(B=32)
    k = keys([0., .1, .2, 10., 10.1, 20., 20.1, 20.2])
    v = torch.randn_like(k)
    positions = torch.arange(k.size(2)) * 3 + 7
    cache.route_and_flush_batch(k, v, positions)
    assert not cache.level_count[..., 1:].any()
    valid = cache.level_w[0, 0] > 0
    order = cache.level_p_lo[0, 0][valid].argsort()
    torch.testing.assert_close(cache.level_k[0, 0][valid][order], k[0, 0], atol=0, rtol=0)
    torch.testing.assert_close(cache.level_v[0, 0][valid][order], v[0, 0], atol=0, rtol=0)
    assert cache.level_w[0, 0][valid].tolist() == [1.] * k.size(2)
    for name in ('level_p_lo', 'level_p_hi', 'level_sum_wp'):
        torch.testing.assert_close(getattr(cache, name)[0, 0][valid][order], positions, atol=0, rtol=0)


def test_unified_router_allocates_no_extra_persistent_cache_buffers():
    baseline, unified = cache_for(unified=False), cache_for()
    layout = lambda cache: {name: (value.shape, value.dtype, value.numel() * value.element_size())
                            for name, value in cache.named_buffers()}
    expected = layout(baseline)
    assert layout(unified) == expected
    k = keys([float(i * 10) for i in range(16)])
    unified.route_and_flush_batch(k, k, torch.arange(16))
    assert layout(unified) == expected


def test_structural_merges_preserve_planner_order_before_any_new_token_write():
    cache = cache_for(K=4, B=2)
    for c, value in enumerate([0., 30., 20., 20.1]):
        k = keys([value])[0, 0, 0]
        cache._semantic_new_cluster(0, 0, c, c, k, k, torch.tensor(c), record=False)
    k = keys([-100., -101., 100., 101.])
    cache.begin_op_log()
    cache.route_and_flush_batch(k, k, torch.arange(4, 8), record_op_log=True)
    cache.take_op_log()
    rows = cache._last_op_log_host[0][0]
    assert [(row[1], row[2]) for row in rows if row[0] == LOG_KV_OP_WARD_MERGE] == [(2, 3), (1, 2)]
    assert all(row[0] == LOG_KV_OP_WARD_MERGE for row in rows[:2])
    assert all(row[0] != LOG_KV_OP_WARD_MERGE for row in rows[2:])


@pytest.mark.parametrize('saved_updates', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_unified_replay_restores_merges_compaction_mass_and_host_counters(saved_updates, dtype):
    torch.manual_seed(37)
    cache = cache_for(batch_size=2, groups=2, semantic_replay_updates=saved_updates, dtype=dtype)
    tape = _SemanticReplayUpdates()
    cache.begin_op_log()
    inputs, states = [], []
    sum_k = torch.zeros(2, 2, 8)
    sum_v = torch.zeros_like(sum_k)
    sum_positions = torch.zeros(2, dtype=torch.int64)
    for step in range(6):
        values = [0.] * 4 + [10.] * 4 + [20.] * 4 if step == 0 else [float(step * 100)] * 12
        k = keys(values).expand(2, 2, -1, -1).to(dtype).clone()
        k[..., 1] = torch.tensor([[0., 1000.], [2000., 3000.]])[:, :, None]
        v = torch.randn_like(k)
        host = [list(range(step * 12 + b * 100, (step + 1) * 12 + b * 100)) for b in range(2)]
        p = torch.tensor(host)
        inputs.append((k, v, p, host))
        context = cache._semantic_update_context(tape) if saved_updates else nullcontext()
        with torch.no_grad(), context:
            cache.route_and_flush_batch(k, v, p, positions_host=host, record_op_log=True)
        sum_k += k.float().sum(2)
        sum_v += v.float().sum(2)
        sum_positions += p.sum(1)
        torch.testing.assert_close(cache.level_w.sum((2, 3, 4)), torch.full((2, 2), float((step + 1) * 12)))
        # bf16 pooling rounds weighted means; replay below must still be bit-identical.
        if dtype == torch.float32:
            torch.testing.assert_close((cache.level_k * cache.level_w[..., None]).sum((2, 3, 4)), sum_k)
            torch.testing.assert_close((cache.level_v * cache.level_w[..., None]).sum((2, 3, 4)), sum_v, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cache.level_sum_wp.sum((2, 3, 4)), sum_positions[:, None].expand(2, 2))
        assert (cache.alive.sum(-1) <= cache.K_max).all()
        states.append(snapshot(cache))
    log, lengths = cache.take_op_log()
    host_log = deepcopy(cache._last_op_log_host)
    assert any(row[0] == LOG_KV_OP_WARD_MERGE for batch in host_log for group in batch for row in group)
    assert cache.level_count[..., 1:].any(), 'The replay fixture must exercise ladder compaction.'
    if saved_updates:
        assert tape.tensors and all(not x.requires_grad for x in tape.tensors)
    for _ in range(2):
        cache.reset_parameters()
        guard = patch.object(cache, '_route_and_flush_batch', side_effect=AssertionError('update replay rerouted')) \
            if saved_updates else patch.object(cache, '_semantic_route_unified', side_effect=AssertionError('op replay rerouted'))
        with guard:
            for (k, v, p, host), expected in zip(inputs, states):
                context = cache._semantic_update_context(tape, replay=True) if saved_updates else nullcontext()
                with torch.no_grad(), context:
                    cache.route_and_flush_batch(k, v, p, positions_host=host, replay_op_log=log,
                                               replay_op_log_len=lengths, replay_op_log_host=host_log)
                assert_snapshot(cache, expected)


@pytest.mark.parametrize('device', DEVICES)
def test_unified_checkpoint_gradients_match_without_checkpoint_and_skip_rerouting(device):
    torch.manual_seed(47)
    config = Config(block_size=48, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                    vocab_size=41, padding_multiple=1, rotary_percentage=1.)
    reference = GPT(config).to(device)
    models = [reference, deepcopy(reference)]
    for i, model in enumerate(models):
        model.enable_log_kv_training(
            batch_size=1, B=3, recent_size=8, train_block=8, second_order_scale=0.,
            semantic_clusters=True, cluster_k_max=3, cluster_lambda_rel=.25,
            semantic_flush_granularity=8, semantic_unified_route=True,
            semantic_anchor_mode='mid', semantic_pack_backend='torch', allocate_second_order=False,
            semantic_replay_updates=bool(i), semantic_centroid_backend='parallel',
            device=torch.device(device))
        if i:
            apply_activation_checkpointing(model, check_fn=lambda m: isinstance(m, Block))
            enable_logkv_checkpoint_replay(model, Block)
    x = torch.randint(41, (1, 40), device=device)
    target = torch.randint(41, x.shape, device=device)
    losses, gradients = [], []
    for i, model in enumerate(models):
        loss = model(x, targets=target, loss_chunk_size=7)
        guard = patch.object(LogStructuredKVCache, '_route_and_flush_batch',
                             side_effect=AssertionError('checkpoint backward rerouted')) if i else nullcontext()
        with guard:
            loss.backward()
        losses.append(loss.detach())
        gradients.append([p.grad.clone() for p in model.parameters()])
    for expected, actual in zip([losses[0]] + gradients[0], [losses[1]] + gradients[1]):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('candidate_pass', [False, True])
def test_batched_reduce_matches_independent_groups_with_uneven_convergence(device, candidate_pass):
    torch.manual_seed(92)
    cache = cache_for(device=device)
    groups = []
    for group, count in enumerate([0, 1, 3, 65, 41, 20]):
        points = torch.randn(count, 8, device=device) * (group + 1)
        groups.append([_SemanticTreeCluster(k, (i % 3) + 1, i, (i,) if i < 3 else (), (i,))
                       for i, k in enumerate(points.unbind(0))])
    radii = [0., 1., 100., 2., 30., .1]
    kwargs = {'radius_limits': radii} if candidate_pass else {'max_clusters': 3}
    actual = cache._semantic_unified_reduce_batch(groups, **kwargs)
    for i, (nodes, merges) in enumerate(actual):
        args = {'radius_limit': radii[i]} if candidate_pass else {'max_clusters': 3}
        expected, expected_merges = cache._semantic_unified_reduce(groups[i], **args)
        assert merges == expected_merges
        assert [(n.n_total, n.p_hi, n.existing, n.tokens) for n in nodes] == [
            (n.n_total, n.p_hi, n.existing, n.tokens) for n in expected]
        for a, b in zip(nodes, expected):
            torch.testing.assert_close(a.centroid, b.centroid, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_gemm_distance_keeps_fp32_under_autocast_and_large_common_offset(device):
    torch.manual_seed(124)
    points = torch.randn(2, 65, 8, device=device) + 10000
    expected = torch.cdist(points.double(), points.double(),
                           compute_mode='donot_use_mm_for_euclid_dist').square().float()
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            actual = LogStructuredKVCache._semantic_unified_distance2(points)
        assert actual.dtype == torch.float32
        assert torch.get_float32_matmul_precision() == 'high'
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=3e-6)
    finally:
        torch.set_float32_matmul_precision(precision)


def test_gemm_cancellation_cannot_widen_candidate_radius():
    cache = cache_for()
    # A large separation between two groups makes close within-group squared
    # distances vulnerable to cancellation even after subtracting one origin.
    k = keys([0.] + [10000. + i * .2 for i in range(64)])
    candidates = cache._semantic_unified_candidates(0, 0, k, [list(range(65))])
    assert sum(len(n.tokens) for n in candidates) == 65
    for n in candidates:
        members = k[0, 0, list(n.tokens)]
        assert ((members - n.centroid).norm(dim=-1) <= .501).all()


def test_batched_global_merge_relaxes_impossible_hard_cap_without_losing_mass():
    cache = cache_for(semantic_capacity_hard_cap_mult=.1)
    groups = [[_SemanticTreeCluster(torch.full((8,), float(i)), 1, i, (), (i,))
               for i in range(count)] for count in [7, 11]]
    for original, (nodes, _) in zip(groups, cache._semantic_unified_reduce_batch(groups, max_clusters=3)):
        assert len(nodes) == 3
        assert sum(n.n_total for n in nodes) == len(original)
        assert sorted(i for n in nodes for i in n.tokens) == list(range(len(original)))
