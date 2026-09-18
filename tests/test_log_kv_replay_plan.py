"""A CPU replay plan retains no device state and adds no device readbacks."""

from array import array
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from litgpt.log_kv_cache import LogStructuredKVCache, _SemanticReplayPlans


def _case(device, dtype):
    torch.manual_seed(73)
    shape = (2, 2, 64, 8)
    cache = LogStructuredKVCache(
        shape, shape, B=3, recent_size=4, device=device, dtype=dtype,
        semantic_clusters=True, cluster_k_max=8, allocate_second_order=False,
        seg_gap_max=1, seg_block_level=2,
        cos_cache=torch.ones(64, 8, device=device), sin_cache=torch.zeros(64, 8, device=device), rope_n_elem=8,
    )
    keys = torch.randn(2, 2, 32, 8, device=device, dtype=dtype)[:, :, ::2]
    values = torch.randn_like(keys)
    positions = torch.arange(0, 32, 2, device=device).expand(2, -1)
    host = positions.cpu().tolist()
    cache.begin_op_log()
    cache.route_and_flush_batch(keys, values, positions, positions_host=host, record_op_log=True)
    log, lengths = cache.take_op_log()
    return cache, keys, values, positions, host, log, lengths


class NoReadback(TorchDispatchMode):
    def __init__(self):
        self.preparing = True
        self.scalar_reads = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._local_scalar_dense.default:
            self.scalar_reads += 1
            assert not self.preparing, "plan preparation read a device scalar"
        if func == torch.ops.aten._to_copy.default and args[0].is_cuda:
            assert torch.device(kwargs.get("device") or args[0].device).type != "cpu", "device-to-host copy"
        return func(*args, **kwargs)


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_plan_skips_parsing_and_device_readbacks(device, dtype):
    cache, keys, values, positions, host, log, lengths = _case(device, dtype)
    expected = {name: tensor.clone() for name, tensor in cache.named_buffers()}
    original_host = cache._last_op_log_host
    plans = _SemanticReplayPlans(original_host)
    readbacks = []
    for iteration in range(2):
        cache.reset_parameters()
        guard = NoReadback()
        commit = cache._semantic_commit_runs

        def apply(*args):
            # Existing scalar pad/overflow writes can read pad_mask.item().
            # Plan construction and lookup must finish without any readback.
            guard.preparing = False
            return commit(*args)

        with patch.object(cache, "_semantic_build_replay_plan",
                          wraps=cache._semantic_build_replay_plan) as parse, guard, patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("unexpected CPU readback")
        ), patch.object(cache, "_semantic_commit_runs", side_effect=apply):
            cache.route_and_flush_batch(
                keys, values, positions, positions_host=host, replay_op_log=log,
                replay_op_log_len=lengths, replay_op_log_host=plans.host_log, replay_plans=plans,
            )
        assert parse.call_count == (1 if iteration == 0 else 0)
        readbacks.append(guard.scalar_reads)
        for name, tensor in cache.named_buffers():
            torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
        # Mutating the original mutable host log cannot change its frozen snapshot.
        if iteration == 0:
            original_host[0][0].clear()
    assert len(plans.plans) == 1
    assert readbacks[0] == readbacks[1]

    def check_cpu(value):
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for key, item in value.items():
                check_cpu(key)
                check_cpu(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_cpu(item)
        else:
            assert isinstance(value, (np.ndarray, array, int, str, bool, type(None)))

    check_cpu(vars(plans))

    cache.reset_parameters()
    with pytest.raises(RuntimeError, match="CPU positions"):
        cache.route_and_flush_batch(
            keys, values, positions, replay_op_log=log, replay_op_log_len=lengths,
            replay_op_log_host=plans.host_log, replay_plans=plans,
        )
    with pytest.raises(RuntimeError, match="rerouting is not allowed"):
        cache.route_and_flush_batch(keys, values, positions, positions_host=host, replay_plans=plans)
    with pytest.raises(RuntimeError, match="CPU op-log"):
        cache.route_and_flush_batch(
            keys, values, positions, positions_host=host, replay_op_log=log, replay_op_log_len=lengths,
            replay_op_log_host=original_host, replay_plans=plans,
        )
    cache.reset_parameters()
    changed_host = [row[:] for row in host]
    changed_host[0][-1] += 1
    with pytest.raises(RuntimeError, match="flush positions"):
        cache.route_and_flush_batch(
            keys, values, positions, positions_host=changed_host, replay_op_log=log, replay_op_log_len=lengths,
            replay_op_log_host=plans.host_log, replay_plans=plans,
        )


def test_cached_offsets_read_current_kv_instead_of_retaining_old_payload():
    cache, keys, values, positions, host, log, lengths = _case("cpu", torch.float32)
    plans = _SemanticReplayPlans(cache._last_op_log_host)
    results = []
    for multiplier, use_plan in ((1., True), (2., True), (2., False)):
        cache.reset_parameters()
        cache.route_and_flush_batch(
            keys * multiplier, values * multiplier, positions, positions_host=host,
            replay_op_log=log, replay_op_log_len=lengths, replay_op_log_host=plans.host_log,
            replay_plans=plans if use_plan else None,
        )
        results.append({name: tensor.clone() for name, tensor in cache.named_buffers()})
    assert not torch.equal(results[0]["level_k"], results[1]["level_k"])
    for name, tensor in results[1].items():
        torch.testing.assert_close(tensor, results[2][name], atol=0, rtol=0)
