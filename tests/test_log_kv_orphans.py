"""Orphan planning batches lanes without changing token ownership or replay."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from litgpt.log_kv_cache import LogStructuredKVCache


def _case(groups, device="cpu", dtype=torch.float32, capped=False, mixed=True):
    torch.manual_seed(71)
    shape = (2, groups, 64, 8)
    cache = LogStructuredKVCache(
        shape, shape, B=3, recent_size=4, device=device, dtype=dtype,
        semantic_clusters=True, cluster_k_max=8, allocate_second_order=False,
        semantic_capacity_hard_cap_mult=0.5 if capped else 0.,
        seg_gap_max=2, seg_block_level=1, semantic_flush_granularity=4,
        cos_cache=torch.ones(128, 8, device=device), sin_cache=torch.zeros(128, 8, device=device),
        rope_n_elem=8,
    )
    # Strided inputs, different live/free slots, orphan counts and seed counts.
    keys = torch.randn(2, groups, 128, 8, device=device, dtype=dtype)[:, :, ::2]
    values = torch.randn_like(keys)
    positions = torch.arange(4, 132, 2, device=device).expand(2, -1)
    host = positions.cpu().tolist()
    orphans = [[[] for _ in range(groups)] for _ in range(2)]
    direct = []
    for b in range(2):
        for g in range(groups):
            variant = (b * groups + g) % 5 if mixed else 2
            live = ([], [1, 3, 5], list(range(8)), [0], [2, 6])[variant]
            count = (3, 1, 47, 0, 31)[variant]
            if variant == 0:
                keys[b, g].zero_()  # identical keys still need distinct seeds
            for c in live:
                key = keys[b, g, c]
                cache._semantic_new_cluster(b, g, c, c % 2, key, key, positions.new_tensor(c % 2), record=False)
            orphans[b][g] = list(range(1, count + 1))
            if live:
                direct.append((b, g, live[0], (0,)))
    novelty = keys.float().square().sum(-1)
    return cache, keys, values, positions, host, orphans, novelty, direct


def _plan(case, orphans=None, jobs=None):
    cache, keys, values, positions, host, original_orphans, novelty, direct = case
    with patch.object(cache, "_semantic_new_cluster") as new, patch.object(cache, "_semantic_commit_joins") as commit:
        cache._semantic_route_orphans_fast(
            original_orphans if orphans is None else orphans, keys, values, positions, host,
            novelty, direct if jobs is None else jobs, record=True,
        )
    created = [call.args[:4] for call in new.call_args_list]
    return created, commit.call_args.args[0]


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capped", [False, True])
def test_batched_orphans_match_independent_lanes_and_replay(device, dtype, capped):
    case = _case(4, device, dtype, capped)
    cache, keys, values, positions, host, orphans, novelty, direct = case
    centroid = cache.centroid.clone()
    created, jobs = _plan(case)
    single_created, single_jobs = [], []
    for b in range(2):
        for g in range(4):
            selected = [[[] for _ in range(4)] for _ in range(2)]
            selected[b][g] = orphans[b][g]
            new, joined = _plan(case, selected, [j for j in direct if j[:2] == (b, g)])
            single_created.extend(new)
            single_jobs.extend(joined)
    assert created == single_created
    assert jobs == single_jobs
    torch.testing.assert_close(cache.centroid, centroid, atol=0, rtol=0)

    # Full routing covers every token; both replay entry points reproduce all
    # writes, including the new clusters and segment padding, on mixed lanes.
    for host_log in (False, True):
        source, replay = deepcopy(cache), deepcopy(cache)
        source.begin_op_log()
        source.route_and_flush_batch(keys, values, positions, positions_host=host, record_op_log=True)
        log, lengths = source.take_op_log()
        replay.route_and_flush_batch(
            keys, values, positions, positions_host=host, replay_op_log=log, replay_op_log_len=lengths,
            replay_op_log_host=source._last_op_log_host if host_log else None,
        )
        for name, tensor in source.named_buffers():
            torch.testing.assert_close(tensor, dict(replay.named_buffers())[name], atol=0, rtol=0)
        assert source._semantic_n_total == replay._semantic_n_total
        assert all(sum(source._semantic_n_total[b][g]) == sum(cache._semantic_n_total[b][g]) + 64
                   for b in range(2) for g in range(4))


@pytest.mark.parametrize("mixed", [False, True])
def test_orphan_planning_ops_and_host_transfers_do_not_scale_with_lanes(mixed):
    class Counter(TorchDispatchMode):
        def __init__(self):
            self.n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.n += 1
            return func(*args, **(kwargs or {}))

    counts = []
    for groups in (4, 16):
        case = _case(groups, mixed=mixed)
        counter = Counter()
        original_cpu = torch.Tensor.cpu
        transfers = []

        def cpu(tensor, *args, **kwargs):
            # Stop at the decision transfer: actual cluster creation below
            # still indexes each new seed's K/V before committing it.
            transfers.append(counter.n)
            return original_cpu(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", cpu), counter:
            _plan(case)
        assert len(transfers) == 1
        counts.append(transfers[0])
    assert counts[1] <= counts[0] * 1.1, counts
