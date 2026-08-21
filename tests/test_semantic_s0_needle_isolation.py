import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "unused" / "semantic_s0_needle_isolation.py"
_SPEC = importlib.util.spec_from_file_location("semantic_s0_needle_isolation_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

NeedleIsolationAccumulator = _MODULE.NeedleIsolationAccumulator
_draw_random_intervals = _MODULE._draw_random_intervals
_surviving_needle_intervals = _MODULE._surviving_needle_intervals


def test_surviving_needle_intervals_clamps_to_used_token_window_and_merges() -> None:
    sample = {
        "prompt_token_offset": 10,
        "needle_spans": [
            {"used_token_start": -2, "used_token_end": 3, "survived_left_truncation": True},
            {"used_token_start": 2, "used_token_end": 6, "survived_left_truncation": True},
            {"used_token_start": 9, "used_token_end": 20, "survived_left_truncation": True},
            {"used_token_start": 4, "used_token_end": 8, "survived_left_truncation": False},
        ],
    }

    assert _surviving_needle_intervals(sample, token_count=10) == [(0, 6), (9, 10)]


def test_surviving_needle_intervals_supports_old_original_token_fields() -> None:
    sample = {
        "prompt_token_offset": 10,
        "needle_spans": [
            {"token_start": 12, "token_end": 16, "survived_left_truncation": True},
        ],
    }

    assert _surviving_needle_intervals(sample, token_count=8) == [(2, 6)]


def test_accumulator_reports_token_and_span_isolation_against_b_prime() -> None:
    cluster_ids = np.asarray([0, 0, 1, 1, 1, 2, 2], dtype=np.int32)
    cluster_sizes = [2, 3, 2]
    acc = NeedleIsolationAccumulator(b_prime=2)

    acc.add_route_meta(cluster_count=3, segment_count=4, sh_source="calibrated")
    acc.add_intervals(intervals=[(0, 3), (5, 7)], cluster_ids=cluster_ids, cluster_sizes=cluster_sizes)
    acc.add_random_intervals(intervals=[(2, 5)], cluster_ids=cluster_ids, cluster_sizes=cluster_sizes)
    row = acc.finalize()

    assert row["needle_token_count"] == 5
    assert row["needle_token_isolated_count"] == 4
    assert row["needle_token_isolated_rate"] == 0.8
    assert row["span_all_tokens_isolated_count"] == 1
    assert row["span_any_token_isolated_count"] == 2
    assert row["random_token_isolated_count"] == 0
    assert row["random_span_all_tokens_isolated_count"] == 0
    assert row["cluster_size_quantiles"]["p50"] == 2.0


def test_draw_random_intervals_preserves_lengths_and_bounds() -> None:
    rng = np.random.default_rng(7)
    intervals = _draw_random_intervals([(2, 5), (8, 10)], token_count=12, rng=rng, trials=4)

    assert len(intervals) == 8
    assert [end - start for start, end in intervals] == [3, 3, 3, 3, 2, 2, 2, 2]
    assert all(0 <= start < end <= 12 for start, end in intervals)
