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
    cache.semantic_merge_passes = 1
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


@pytest.mark.parametrize("limited", [False, True])
def test_status_polls_skip_rounds_that_cannot_finish_and_narrow_launches(limited):
    # A fake reducer halves its live count each round, the fastest possible
    # descent; it reports (active, live count) rows like the Triton reducer.
    size, target, steps, bounds = 300, 12, [], []
    device = torch.device("cuda")
    mu = SimpleNamespace(device=device, size=lambda axis: size)
    cache = object.__new__(LogStructuredKVCache)
    torch.nn.Module.__init__(cache)
    cache.semantic_merge_passes = 1
    event = Mock()

    class Reducer:
        def __init__(self, *args):
            self.status = torch.tensor([[1], [size]], dtype=torch.int32)

        def step(self, round_):
            steps.append(round_)
            live = max((int(self.status[1, 0]) + 1) // 2, target)
            self.status[:, 0] = torch.tensor([live > target, live], dtype=torch.int32)
            return self.status[0]

        def limit_live(self, count):
            bounds.append(count)

        def finish(self):
            return [np.empty((0, 2), dtype=np.int64)], np.ones((1, size), dtype=bool), np.zeros(1, dtype=bool)

    with patch("litgpt.log_kv_cache._triton_route", return_value=SimpleNamespace(UnifiedReduce=Reducer)), \
         patch.object(cache, "_semantic_unified_pair_matrix", return_value=None), \
         patch.object(cache, "_SEMANTIC_ROUTE_FLAG_BUFFERS", {(device, 1, 2): torch.zeros(2, 1, dtype=torch.int32)}), \
         patch.object(torch.Tensor, "to", lambda self, *args, **kwargs: self), \
         patch.object(torch.Tensor, "pin_memory", lambda self: self), \
         patch("torch.cuda.Event", return_value=event), patch("torch.cuda.current_stream", return_value=object()):
        cache._semantic_unified_reduce_incremental(
            mu, [size], np.ones((1, size), dtype=np.float32), target=target,
            radius_limits=[1.] if limited else None, strict=not limited, original=None,
        )
    finished = LogStructuredKVCache._semantic_min_rounds(size, target)
    assert finished == 5 and LogStructuredKVCache._semantic_min_rounds(target, target) == 0
    # Completion is still observed within three extra rounds.
    assert finished <= len(steps) <= finished + 3
    polls = event.synchronize.call_count
    assert polls == (2 if limited else 1)
    assert bounds and bounds[-1] == target and bounds == sorted(bounds, reverse=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("limited", [False, True])
def test_live_list_rounds_match_reference_with_narrowed_launches(limited):
    from litgpt.log_kv_route_triton import UnifiedReduce

    torch.manual_seed(1703)
    size = 300
    mu = torch.randint(0, 6, (3, size, 1), device="cuda").float() * 4 + .05 * torch.randn(3, size, 9, device="cuda")
    count = torch.tensor([size, size - 41, 7], device="cuda")
    mask = torch.arange(size, device="cuda")[None, :] < count[:, None]
    mu, mass = mu * mask[..., None], torch.randint(1, 4, (3, size), device="cuda").float() * mask
    limits = torch.tensor([.3, 1., 4.], device="cuda") if limited else None
    results = []
    for reducer_type in (_UnifiedReduceTorch, UnifiedReduce):
        state = mu.clone()
        reducer = reducer_type(LogStructuredKVCache._semantic_unified_pair_matrix(state), state, mass.clone(),
                               count, limits, 1 if limited else 12)
        for round_ in range(1, 2 * size + 8):
            active = reducer.step(round_)
            if hasattr(reducer, "limit_live"):
                reducer.limit_live(int(reducer.count.max()))
            if not bool(active.any()):
                break
        results.append((reducer.finish(), state))
    (expected, expected_mu), (actual, actual_mu) = results
    for a, b in zip(actual[0], expected[0]):
        if limited:
            assert sorted(map(tuple, a.tolist())) == sorted(map(tuple, b.tolist()))
        else:
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[1], expected[1])
    if not limited:
        torch.testing.assert_close(actual_mu, expected_mu, rtol=0, atol=0)
