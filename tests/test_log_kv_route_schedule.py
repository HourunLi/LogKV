"""Completion polling may skip host waits, but must not change routing results."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from litgpt.log_kv_cache import LogStructuredKVCache, _UnifiedReduceTorch


@pytest.mark.parametrize("stop_round", [1, 4, 5, 9, 39])
def test_cuda_schedule_polls_four_rounds_at_a_time_without_cuda(stop_round):
    # Exercise the actual host scheduler with CPU flags and a fake CUDA stream.
    size, steps = 16, []
    device = torch.device("cuda")
    mu = SimpleNamespace(device=device, size=lambda axis: size)
    cache = object.__new__(LogStructuredKVCache)
    torch.nn.Module.__init__(cache)
    event = Mock()

    class Reducer:
        def __init__(self, *args):
            pass

        def step(self, round_):
            steps.append(round_)
            return torch.tensor([round_ < stop_round], dtype=torch.int32)

        def finish(self):
            return [np.empty((0, 2), dtype=np.int64)], np.ones((1, size), dtype=bool), np.zeros(1, dtype=bool)

    with patch("litgpt.log_kv_cache._triton_route", return_value=SimpleNamespace(UnifiedReduce=Reducer)), \
         patch.object(cache, "_semantic_unified_pair_matrix", return_value=None), \
         patch.object(cache, "_SEMANTIC_ROUTE_FLAG_BUFFERS", {(device, 1): torch.zeros(1, dtype=torch.int32)}), \
         patch.object(torch.Tensor, "to", lambda self, *args, **kwargs: self), \
         patch.object(torch.Tensor, "pin_memory", lambda self: self), \
         patch("torch.cuda.Event", return_value=event), patch("torch.cuda.current_stream", return_value=object()):
        cache._semantic_unified_reduce_incremental(
            mu, [size], np.ones((1, size), dtype=np.float32),
            target=1, radius_limits=None, strict=False, original=None,
        )
    expected = min(((stop_round + 3) // 4) * 4, 2 * size + 7)
    assert steps == list(range(1, expected + 1))
    assert event.synchronize.call_count == (expected + 3) // 4
    assert event.record.call_count == event.synchronize.call_count


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA"))])
def test_finished_rounds_preserve_centroids_mass_and_merge_trace(device, limited):
    reducer_type = _UnifiedReduceTorch
    if device == "cuda":
        from litgpt.log_kv_route_triton import UnifiedReduce
        reducer_type = UnifiedReduce
    torch.manual_seed(1701)
    mu = torch.randn(3, 12, 4, device=device)
    mass = torch.ones(3, 12, device=device)
    count = torch.tensor([12, 12, 12], device=device)
    # Lanes finish for both reasons: no admissible candidates and target reached.
    limits = torch.tensor([0., .35, 100.], device=device) if limited else None
    dist = LogStructuredKVCache._semantic_unified_pair_matrix(mu)
    reducer = reducer_type(dist, mu, mass, count, limits, 1 if limited else 3)
    for round_ in range(1, 2 * mu.size(1) + 8):
        if not bool(reducer.step(round_).any()):
            break
    else:
        pytest.fail("reducer did not finish")
    fields = ("mu", "mass", "radius", "count", "alive", "stuck")
    expected = {name: getattr(reducer, name).clone() for name in fields}
    traces, alive, stuck = reducer.finish()
    # finish() can return numpy views into CPU tensors; freeze the expected data.
    traces, alive, stuck = [x.copy() for x in traces], alive.copy(), stuck.copy()
    for extra in range(1, 4):
        assert not bool(reducer.step(round_ + extra).any())
    for name, value in expected.items():
        torch.testing.assert_close(getattr(reducer, name), value, rtol=0, atol=0)
    actual_traces, actual_alive, actual_stuck = reducer.finish()
    for a, b in zip(traces, actual_traces):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(alive, actual_alive)
    np.testing.assert_array_equal(stuck, actual_stuck)
