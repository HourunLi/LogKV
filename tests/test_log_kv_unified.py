"""Whole-flush candidate construction, budget reduction, and training replay."""

import math

from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import patch

import numpy as np
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/Triton')
def test_fused_round_pairs_match_reference_for_ties_padding_caps_and_radius():
    pytest.importorskip('triton')
    from unused.benchmark_log_kv_unified import check_round_pairs

    check_round_pairs(cache_for(device='cuda'), torch.device('cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/Triton')
def test_fused_merge_pack_matches_reference_and_keeps_inputs_immutable():
    pytest.importorskip('triton')
    from unused.benchmark_log_kv_unified import check_merge_pack

    check_merge_pack(cache_for(device='cuda'), torch.device('cuda'))


def test_merge_pack_preserves_centers_mass_radius_and_padding():
    mu = torch.tensor([[[1., 2.], [3., 6.], [7., 9.], [8., 5.]]])
    mass = torch.tensor([[2., 6., 3., 4.]])
    radius = torch.tensor([[.5, 1., 2., 3.]])
    indices = torch.tensor([[[0, 2, 3, -1]], [[1, -1, -1, -1]]])
    original = mu.clone(), mass.clone(), radius.clone()
    center, weight, bound = LogStructuredKVCache._semantic_unified_merge_pack(mu, mass, radius, indices, True)
    torch.testing.assert_close(center, torch.tensor([[[2.5, 5.], [7., 9.], [8., 5.], [0., 0.]]]))
    torch.testing.assert_close(weight, torch.tensor([[8., 3., 4., 0.]]))
    expected_radius = torch.maximum(.5 + .75 * (mu[0, 0] - mu[0, 1]).norm(),
                                    1. + .25 * (mu[0, 0] - mu[0, 1]).norm())
    torch.testing.assert_close(bound[0, 0], expected_radius)
    torch.testing.assert_close(bound[0, 1:], torch.tensor([2., 3., 0.]))
    for actual, before in zip((mu, mass, radius), original):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)


def test_short_route_profiler_exports_trace_and_python_summary(tmp_path):
    import json
    from unused.benchmark_log_kv_unified import profile_route

    torch.manual_seed(18)
    cache = cache_for()
    inputs = [(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8),
               torch.arange(start, start + 8), [list(range(start, start + 8))]) for start in (0, 8)]
    with torch.no_grad():
        result = profile_route(cache, inputs, torch.device('cpu'), tmp_path)
    trace = json.loads((tmp_path / 'route_trace.json').read_text())
    assert sum(e.get('name') == 'logkv/route_flush' for e in trace['traceEvents']) == 1
    names = {e.get('name') for e in trace['traceEvents']}
    assert {'logkv/global_merge', 'logkv/merge_round'} <= names
    summary = (tmp_path / 'route_summary.txt').read_text()
    assert 'aten::' in summary and 'route_and_flush_batch' in summary
    assert (tmp_path / 'route_python.prof').stat().st_size > 0
    assert result['profiled_flushes_per_pass'] == 1


def test_packed_candidates_reuse_unmerged_nodes_and_centers():
    cache = cache_for()
    mu = torch.arange(9.).unsqueeze(1).expand(9, 8).contiguous() * 10
    nodes = [_SemanticTreeCluster(None, 1, i, (), (i,)) for i in range(9)]
    result, merges, centers = cache._semantic_unified_reduce_packed([nodes], [mu], radius_limits=[.1])[0]
    assert not merges and centers is mu
    assert all(a is b for a, b in zip(result, nodes))
    result, _, centers = cache._semantic_unified_reduce_packed([nodes], [mu], max_clusters=3)[0]
    assert len(result) == len(centers) == 3
    for node, center in zip(result, centers):
        torch.testing.assert_close(center, mu[list(node.tokens)].mean(0))


@pytest.mark.parametrize('device', DEVICES)
def test_production_unified_route_does_not_unbind_token_centers(device):
    cache = cache_for(device=device, batch_size=2, groups=2)
    unbind = torch.Tensor.unbind

    def checked(tensor, dim=0):
        assert not (tensor.ndim == 2 and dim == 0 and tensor.size(0) > cache.K_max and tensor.size(1) == 8)
        return unbind(tensor, dim)

    with patch.object(torch.Tensor, 'unbind', checked):
        for start in (0, 65):
            k = torch.randn(2, 2, 65, 8, device=device)
            cache.route_and_flush_batch(k, k, torch.arange(start, start + 65, device=device),
                                        positions_host=[list(range(start, start + 65))] * 2)
    torch.testing.assert_close(cache.level_w.sum((2, 3, 4)), torch.full((2, 2), 130., device=device))


# Keep the former node/serial-write path as an independent state oracle.
def _node_route_reference(
    self, k_raw: torch.Tensor, v: torch.Tensor, positions: torch.Tensor,
    positions_host: list[list[int]], *, record: bool,
) -> None:
    jobs: list[tuple[int, int, int, tuple[int, ...]]] = []
    lanes = [(b, g) for b in range(k_raw.size(0)) for g in range(k_raw.size(1))]
    # Fused CUDA search needs one pair matrix, versus several for Torch.
    # ponytail: 16 groups bound CUDA scratch; tile rows if larger flushes need less memory.
    tile_size = 16 if k_raw.is_cuda else 4
    for start in range(0, len(lanes), tile_size):
        tile = lanes[start:start + tile_size]
        centers = [k_raw[b, g].detach().float() for b, g in tile]
        groups = [[_SemanticTreeCluster(None, 1, positions_host[b][i], (), (i,))
                   for i in range(k_raw.size(2))] for b, g in tile]
        candidates = self._semantic_unified_reduce_packed(
            groups, centers, radius_limits=[math.sqrt(self._semantic_tree_threshold(b, g)) for b, g in tile]
        )
        existing = [self._semantic_tree_existing_set(b, g) for b, g in tile]
        centers = [torch.cat((torch.stack([n.centroid for n in old]), candidate[2])) if old else candidate[2]
                   for old, candidate in zip(existing, candidates)]
        plans = self._semantic_unified_reduce_packed(
            [old + candidate[0] for old, candidate in zip(existing, candidates)], centers,
            max_clusters=self.K_max,
        )
        for (b, g), (nodes, old_merges, _) in zip(tile, plans):
            for keep, free in old_merges:
                self._semantic_ward_merge(b, g, keep, free, record=record)
            free_slots = iter(self._semantic_free_clusters(b, g))
            for node in nodes:
                offsets = tuple(sorted(node.tokens, key=lambda i: positions_host[b][i]))
                if not offsets:
                    continue
                if node.existing:
                    target = min(node.existing)
                else:
                    target = next(free_slots)
                    first, *rest = offsets
                    self._semantic_new_cluster(
                        b, g, target, positions_host[b][first], k_raw[b, g, first], v[b, g, first],
                        positions[b, first], record=record,
                    )
                    offsets = tuple(rest)
                if offsets:
                    jobs.append((b, g, target, offsets))
    self._semantic_commit_joins(sorted(jobs), k_raw, v, positions, positions_host, record=record)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('pattern', ['random', 'identical', 'clustered'])
@pytest.mark.parametrize('seg_gap', [None, 3])
def test_array_route_and_batched_writes_match_node_reference(device, pattern, seg_gap):
    from types import MethodType
    torch.manual_seed(167)
    cache = cache_for(batch_size=2, groups=2, B=8, device=device, seg_gap_max=seg_gap,
                      seg_block_level=2, seg_forget=1.)
    reference = deepcopy(cache)
    reference._semantic_route_unified = MethodType(_node_route_reference, reference)
    for c in (cache, reference):
        c.begin_op_log()
    for step in range(3):
        # More than 32 rows exercises GEMM, unequal rounds and old-old merges.
        k = torch.randn(2, 2, 65, 8, device=device)
        if pattern == 'identical':
            k.fill_(.1)
        elif pattern == 'clustered':
            k = (torch.arange(65, device=device) // 16)[None, None, :, None] + .001 * k
        v = torch.randn_like(k)
        pos = (torch.arange(step * 65, (step + 1) * 65, device=device) * 7).expand(2, -1)
        host = pos.cpu().tolist()
        for c in (cache, reference):
            c.route_and_flush_batch(k, v, pos, positions_host=host, record_op_log=True)
        assert_snapshot(cache, snapshot(reference))
        assert cache._op_log_host == reference._op_log_host
        torch.testing.assert_close(cache.op_log, reference.op_log, atol=0, rtol=0)
        torch.testing.assert_close(cache.op_log_len, reference.op_log_len, atol=0, rtol=0)
        assert not cache._pending_op_starts


def test_production_route_does_not_construct_token_nodes():
    cache = cache_for(batch_size=2, groups=2)
    k = torch.randn(2, 2, 65, 8)
    with patch('litgpt.log_kv_cache._SemanticTreeCluster', side_effect=AssertionError('token node allocated')):
        for step in range(2):
            cache.route_and_flush_batch(k, k, torch.arange(step * 65, (step + 1) * 65))
    assert cache.level_w.sum().item() == 2 * 2 * 65 * 2


@pytest.mark.parametrize('device', DEVICES)
def test_benchmark_verifies_batched_state_before_timing(device):
    from unused.benchmark_log_kv_unified import check_batched_state
    check_batched_state(torch.device(device))


@pytest.mark.parametrize('device', DEVICES)
def test_deferred_op_log_matches_immediate_writes_with_uneven_lanes(device):
    actual = cache_for(device=device, batch_size=2, groups=2)
    reference = deepcopy(actual)
    for cache in (actual, reference):
        cache.begin_op_log()
    for step in range(2):
        with actual._semantic_deferred_scalars():
            for b, g, count in [(0, 1, 3), (1, 0, 1), (0, 1, 2), (1, 1, 4)]:
                rows = [(2, 0, 0, step * 20 + i) for i in range(count)]
                actual._record_ops(b, g, rows)
                reference._record_ops(b, g, rows)
            # Taking a log is also a visibility boundary inside a deferred block.
            for got, want in zip(actual.take_op_log(), reference.take_op_log()):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.testing.assert_close(actual.op_log, reference.op_log, atol=0, rtol=0)
        assert actual._op_log_host == reference._op_log_host


@pytest.mark.parametrize('device', DEVICES)
def test_frozen_matching_sweeps_match_reference_and_preserve_constraints(device):
    """An independent peeling oracle, ragged groups, ties and guarded paths."""
    cache = cache_for(semantic_merge_passes=4)
    rng = torch.Generator().manual_seed(98)
    mu = torch.randint(-4, 5, (3, 65, 8), generator=rng).float().to(device)
    mu[0].zero_()
    mass = torch.ones(3, 65, device=device)
    mass[1, 51:] = 0
    mass[2] = 0
    radius = torch.zeros_like(mass)
    d2 = (mu[:, :, None] - mu[:, None, :]).square().sum(-1)
    cost = d2 * (mass[:, :, None] * mass[:, None, :] / (mass[:, :, None] + mass[:, None, :]).clamp_min(1))
    cost.masked_fill_(~((mass > 0)[:, :, None] & (mass > 0)[:, None, :]), math.inf)
    cost.diagonal(dim1=-2, dim2=-1).fill_(math.inf)
    expected = []
    for lane in range(3):
        remaining = cost[lane].clone()
        chosen = []
        for _ in range(4):
            pairs = cache._semantic_unified_pairs(remaining, 32)
            chosen.extend(pairs.tolist())
            remaining[pairs.flatten()] = math.inf
            remaining[:, pairs.flatten()] = math.inf
        expected.append(sorted(chosen, key=lambda pair: (float(cost[lane, pair[0], pair[1]]), pair[0])))
    actual = cache._semantic_unified_round_pairs(mu, mass, radius, None, math.inf, True)
    for rows, wanted in zip(actual.cpu(), expected):
        pairs = rows[rows[:, 0] >= 0]
        assert pairs.tolist() == wanted
        assert pairs.unique().numel() == pairs.numel()
    for limits, cap, overflow in ((mu.new_full((3,), .5), math.inf, False), (None, .1, True)):
        actual = cache._semantic_unified_round_pairs(mu, mass, radius, limits, cap, overflow)
        cache.semantic_merge_passes = 1
        expected = cache._semantic_unified_round_pairs(mu, mass, radius, limits, cap, overflow)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        cache.semantic_merge_passes = 4


def test_frozen_matching_reduction_reaches_budget_and_conserves_centroids():
    import numpy as np
    cache = cache_for(semantic_merge_passes=4)
    torch.manual_seed(11)
    centers = [torch.randn(n, 8) for n in (65, 19)]
    weights = [np.arange(1, len(mu) + 1, dtype=np.float32) for mu in centers]
    for mu, weight, (roots, traces, reduced) in zip(
        centers, weights, cache._semantic_unified_reduce_arrays(weights, centers, max_clusters=3)
    ):
        assert len(roots) == 3
        labels = cache._semantic_unified_labels(len(mu), roots, traces)
        for c in range(3):
            member = torch.from_numpy(labels == c)
            w = torch.from_numpy(weight)[member]
            torch.testing.assert_close(reduced[c], (mu[member] * w[:, None]).sum(0) / w.sum())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('candidate_pass', [False, True])
def test_incremental_rounds_match_full_recompute_without_ties(device, candidate_pass):
    # Distinct random costs: Lance-Williams updates must reproduce every pair
    # of the full-recompute rounds, in round and cost order.
    torch.manual_seed(211)
    cache = cache_for(device=device, K=4)
    groups = [np.random.RandomState(i).randint(1, 4, size=n).astype(np.float32) for i, n in enumerate([70, 33, 5, 90])]
    centers = [torch.randn(len(w), 8, device=device) * (i + 1) for i, w in enumerate(groups)]
    kwargs = {'radius_limits': [2., 3., 1., 4.]} if candidate_pass else {'max_clusters': 4}
    actual = cache._semantic_unified_reduce_arrays(groups, [c.clone() for c in centers], **kwargs)
    expected = cache._semantic_unified_reduce_rounds(groups, [c.clone() for c in centers], **kwargs)
    merged = 0
    for (roots, traces, mu), (want_roots, want_traces, want_mu) in zip(actual, expected):
        pairs = np.concatenate(traces) if traces else np.empty((0, 2), dtype=np.int64)
        want = np.concatenate(want_traces) if want_traces else np.empty((0, 2), dtype=np.int64)
        assert np.array_equal(roots, want_roots)
        assert np.array_equal(pairs, want)
        torch.testing.assert_close(mu, want_mu, atol=1e-5, rtol=1e-5)
        merged += len(pairs)
    assert merged


def test_device_ward_entry_order_matches_two_pointer_merge():
    rng = np.random.RandomState(7)
    for _ in range(300):
        metas, splits, sides = [], [], []
        for _ in range(rng.randint(1, 4)):
            keep, free = rng.randint(0, 6), rng.randint(0, 6)
            keep += keep + free == 0
            metas.append([(int(rng.randint(0, 5)), int(rng.randint(0, 8)), bool(rng.rand() < .7))
                          for _ in range(keep + free)])
            splits.append(keep)
            sides += [keep, free]
        expected, start = [], 0
        for meta, split in zip(metas, splits):
            expected += [start + i for i in LogStructuredKVCache._semantic_ward_entry_order(meta, split)]
            start += len(meta)
        flat = [m for meta in metas for m in meta]
        sides = np.asarray(sides)
        actual = LogStructuredKVCache._semantic_ward_entry_index(
            torch.tensor([m[0] for m in flat]), torch.tensor([m[1] for m in flat]),
            torch.tensor([m[2] for m in flat]), torch.from_numpy(np.repeat(np.arange(len(sides)), sides)),
            torch.from_numpy(np.repeat(np.cumsum(sides), sides)),
        )
        assert actual.tolist() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/Triton')
def test_fused_incremental_rounds_match_torch_reference():
    pytest.importorskip('triton')
    from unused.benchmark_log_kv_unified import check_incremental_reduce

    check_incremental_reduce(torch.device('cuda'))


def _reducer_type(device):
    from litgpt.log_kv_cache import _UnifiedReduceTorch, _triton_route
    fused = _triton_route() if device == 'cuda' else None
    return fused.UnifiedReduce if fused is not None else _UnifiedReduceTorch


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('limited', [False, True])
def test_incremental_distances_stay_exactly_symmetric(device, limited):
    # Lattice keys with ulp-level noise: pairs merged in the same round meet
    # near ties. An asymmetric entry can form a preference cycle with no mutual
    # pair, which used to abort routing with "no finite merge cost".
    torch.manual_seed(7)
    lanes, size = 4, 120
    mu = torch.randint(0, 3, (lanes, size, 3), device=device).float()
    mu = mu * 1000 + 1e-4 * torch.randn_like(mu)
    mass = torch.randint(1, 4, (lanes, size), device=device).float()
    dist = LogStructuredKVCache._semantic_unified_pair_matrix(mu)
    limits = torch.full((lanes,), 5., device=device) if limited else None
    reducer = _reducer_type(device)(dist, mu, mass, torch.full((lanes,), size, device=device), limits, 1 if limited else 3)
    for round_ in range(1, 2 * size + 8):
        active = reducer.step(round_)
        live = torch.as_tensor(reducer.alive, device=device).bool()
        both = live[:, :, None] & live[:, None, :]
        assert torch.equal(dist[both], dist.transpose(1, 2)[both]), f'asymmetric after round {round_}'
        if not bool(active.any()):
            break
    _, _, stuck = reducer.finish()
    if not limited:
        assert not np.asarray(stuck).any()


def test_stuck_incremental_lane_falls_back_to_full_recompute():
    from litgpt.log_kv_cache import _UnifiedReduceTorch

    torch.manual_seed(5)
    cache = cache_for(K=3)
    groups = [np.ones(n, dtype=np.float32) for n in (20, 17)]
    centers = [torch.randn(len(w), 8) for w in groups]
    finish = _UnifiedReduceTorch.finish

    def stuck_first_lane(self):
        traces, alive, stuck = finish(self)
        stuck = np.asarray(stuck).copy()
        stuck[0] = True
        return traces, alive, stuck

    LogStructuredKVCache._semantic_warned_stuck = False
    with patch.object(_UnifiedReduceTorch, 'finish', stuck_first_lane), pytest.warns(UserWarning, match='full-recompute'):
        actual = cache._semantic_unified_reduce_arrays(groups, [c.clone() for c in centers], max_clusters=3)
    expected = cache._semantic_unified_reduce_rounds(groups[:1], [centers[0].clone()], max_clusters=3)[0]
    roots, traces, mu = actual[0]
    assert np.array_equal(roots, expected[0])
    assert np.array_equal(np.concatenate(traces), np.concatenate(expected[1]))
    torch.testing.assert_close(mu, expected[2], rtol=0, atol=0)
    assert len(actual[1][0]) == 3


def test_production_route_honours_capacity_hard_cap():
    cache = cache_for(K=3, semantic_capacity_hard_cap_mult=.1)
    rounds = cache._semantic_unified_reduce_rounds
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get('max_clusters'))
        return rounds(*args, **kwargs)

    with patch.object(cache, '_semantic_unified_reduce_rounds', counted):
        k = keys([float(i * 10) for i in range(16)])
        cache.route_and_flush_batch(k, k, torch.arange(16))
    assert 3 in calls
    assert cache.alive.sum() <= 3
    assert cache.level_w.sum().item() == 16


@pytest.mark.parametrize('device', DEVICES)
def test_incremental_frozen_sweeps_match_full_recompute_sweeps(device):
    # Distinct random costs: restricted rescans inside a round must reproduce
    # the full-recompute frozen-center sweeps (AlphaLogKV merge_passes=4).
    torch.manual_seed(223)
    cache = cache_for(device=device, K=4, semantic_merge_passes=4)
    groups = [np.random.RandomState(i).randint(1, 4, size=n).astype(np.float32) for i, n in enumerate([70, 33, 5, 90])]
    centers = [torch.randn(len(w), 8, device=device) * (i + 1) for i, w in enumerate(groups)]
    actual = cache._semantic_unified_reduce_arrays(groups, [c.clone() for c in centers], max_clusters=4)
    expected = cache._semantic_unified_reduce_rounds(groups, [c.clone() for c in centers], max_clusters=4)
    for (roots, traces, mu), (want_roots, want_traces, want_mu) in zip(actual, expected):
        pairs = np.concatenate(traces) if traces else np.empty((0, 2), dtype=np.int64)
        want = np.concatenate(want_traces) if want_traces else np.empty((0, 2), dtype=np.int64)
        assert np.array_equal(roots, want_roots)
        assert np.array_equal(pairs, want)
        torch.testing.assert_close(mu, want_mu, atol=1e-5, rtol=1e-5)
