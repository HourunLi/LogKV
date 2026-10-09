"""LOGKV_ROUTE_OVERLAP only moves decisions whose inputs predate the ready event."""

from types import SimpleNamespace

import pytest
import torch

from litgpt.log_kv_cache import LogStructuredKVCache

CUDA = torch.device("cuda")


class _Stream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


def _cache():
    n, dim = 64, 8
    cache = LogStructuredKVCache(
        (1, 2, n, dim), (1, 2, n, dim), B=4, recent_size=16, semantic_flush_granularity=16,
        semantic_clusters=True, cluster_k_max=4, semantic_anchor_mode="mid", allocate_second_order=False,
        semantic_replay_updates=True, alpha_exact_tokens=8, alpha_span_max_tokens=4,
        cos_cache=torch.ones(n, dim), sin_cache=torch.zeros(n, dim), rope_n_elem=dim,
    )
    stream = _Stream()
    cache._route_overlap = True
    cache._ROUTE_STREAMS = {CUDA: stream}
    cache._route_ready, cache._route_ready_rows = object(), 16
    return cache, stream


def _window(buffer, rows, offset=0):
    # Stands in for a CUDA slice of the recent window (no GPU needed).
    return SimpleNamespace(device=CUDA, data_ptr=lambda: buffer.data_ptr() + offset, size=lambda dim: rows)


def test_side_stream_needs_rows_that_existed_at_the_event():
    cache, stream = _cache()
    k, v = _window(cache.recent_k_raw, 16), _window(cache.recent_v, 16)
    assert cache._route_decision_stream(k, v) is stream and stream.waited == [cache._route_ready]
    # An unaligned chunk appended rows after the event: stay on the main stream.
    cache._route_ready_rows = 8
    assert cache._route_decision_stream(k, v) is None
    cache._route_ready_rows = 16
    # Inputs that are not the flush loop's own window keep the main stream.
    assert cache._route_decision_stream(_window(cache.recent_k_raw, 16, 4), v) is None
    assert cache._route_decision_stream(k, _window(cache.recent_k_raw, 16)) is None
    # Off, before the first flush, or on another route.
    cache._route_ready = None
    assert cache._route_decision_stream(k, v) is None
    cache._route_ready = object()
    cache.semantic_unified_route = True
    assert cache._route_decision_stream(k, v) is None
    cache.semantic_unified_route = False
    cache._route_overlap = False
    assert cache._route_decision_stream(k, v) is None
    assert len(stream.waited) == 1


def test_ready_event_records_the_window_rows(monkeypatch):
    cache, _ = _cache()
    recorded = []
    event = SimpleNamespace(record=recorded.append)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "main")
    cache._route_ready, cache.recent_count = event, 5
    cache._route_mark_ready(CUDA)
    assert recorded == ["main"] and cache._route_ready_rows == 5


@pytest.mark.parametrize("enabled,warn_only", [(False, False), (True, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_checker_backward_is_deterministic_and_restores_mode(enabled, warn_only, fail):
    from unused.check_route_overlap import backward_exact

    original = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        x = torch.tensor([2.], requires_grad=True)

        def check_mode(grad):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            if fail:
                raise RuntimeError("backward failed")
            return grad

        x.register_hook(check_mode)
        if fail:
            with pytest.raises(RuntimeError, match="backward failed"):
                backward_exact(x.square().sum())
        else:
            backward_exact(x.square().sum())
            assert torch.equal(x.grad, torch.tensor([4.]))
        assert torch.are_deterministic_algorithms_enabled() == enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])


def test_checker_keeps_zero_tolerance_and_reports_error():
    from unused.check_route_overlap import assert_equal

    reference = torch.tensor([1.])
    changed = torch.nextafter(reference, torch.tensor([2.]))
    with pytest.raises(AssertionError, match="(?s)training: dq.*Mismatched elements"):
        assert_equal(reference, changed, "training: dq")
