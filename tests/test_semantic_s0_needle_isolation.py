import importlib.util
from pathlib import Path
import tempfile

import numpy as np

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "unused" / "semantic_s0_needle_isolation.py"
_SPEC = importlib.util.spec_from_file_location("semantic_s0_needle_isolation_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

NeedleIsolationAccumulator = _MODULE.NeedleIsolationAccumulator
_draw_random_intervals = _MODULE._draw_random_intervals
_groups_from_key_scale = _MODULE._groups_from_key_scale
parse_lambda_rel_list = _MODULE.parse_lambda_rel_list
_process_record_task = _MODULE._process_record_task
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


def test_parse_lambda_rel_list_accepts_comma_separated_values() -> None:
    assert parse_lambda_rel_list("0.25,0.5,1.0") == [0.25, 0.5, 1.0]
    assert parse_lambda_rel_list([0.75, 1]) == [0.75, 1.0]


def test_groups_from_key_scale_infers_group_ids_without_opening_npz() -> None:
    manifest = {"key_scale": {"3": {"s_h": [1.0, 2.0, 3.0]}}}

    assert _groups_from_key_scale(manifest, 3) == [0, 1, 2]
    assert _groups_from_key_scale(manifest, 4) is None


def test_process_record_task_is_deterministic_for_worker_shards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        k_raw = np.asarray(
            [
                [
                    [0.0, 0.0],
                    [0.1, 0.0],
                    [8.0, 0.0],
                    [0.2, 0.0],
                    [8.1, 0.0],
                    [0.3, 0.0],
                ]
            ],
            dtype=np.float32,
        )
        np.savez(base_dir / "sample_0000_layer_00.npz", k_raw=k_raw, v=k_raw.copy())
        sample = {
            "sample_id": "smoke",
            "prompt_token_offset": 0,
            "needle_spans": [
                {
                    "used_token_start": 2,
                    "used_token_end": 3,
                    "survived_left_truncation": True,
                }
            ],
        }
        task = {
            "sample": sample,
            "record": {"layer": 0, "path": "sample_0000_layer_00.npz"},
            "record_i": 1,
            "record_index": 0,
            "record_count": 1,
            "base_dir": str(base_dir),
            "scale_manifest": {"key_scale": {"0": {"s_h": [1.0]}}},
            "g_values": [float("inf"), 2.0],
            "group_filter": [0],
            "lambda_rel_values": [0.5, 1.0],
            "seg_forget": 0.5,
            "b_prime": 2,
            "random_trials": 3,
            "seed": 99,
            "allow_fallback_sh": False,
            "collect_by_layer_group": True,
        }

        first = _process_record_task(task)
        second = _process_record_task(task)

    assert first["processed_pairs"] == 1
    assert first["matched_group_pairs"] == 1
    assert first["comparable_pairs"] == 1
    assert sorted(first["overall"]) == [(0.5, "2"), (0.5, "inf"), (1.0, "2"), (1.0, "inf")]
    assert sorted(first["by_layer_group"]) == [
        (0.5, "2", 0, 0),
        (0.5, "inf", 0, 0),
        (1.0, "2", 0, 0),
        (1.0, "inf", 0, 0),
    ]
    assert first["overall"][(0.5, "2")].finalize()["random_token_count"] == 3
    assert first["overall"][(0.5, "2")].finalize() == second["overall"][(0.5, "2")].finalize()


def test_process_record_task_can_process_one_group_subset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        k_raw = np.asarray(
            [
                [[0.0, 0.0], [0.1, 0.0], [8.0, 0.0], [0.2, 0.0]],
                [[1.0, 0.0], [1.1, 0.0], [9.0, 0.0], [1.2, 0.0]],
            ],
            dtype=np.float32,
        )
        np.savez(base_dir / "sample_0000_layer_00.npz", k_raw=k_raw, v=k_raw.copy())
        task = {
            "sample": {
                "sample_id": "smoke",
                "prompt_token_offset": 0,
                "needle_spans": [
                    {"used_token_start": 2, "used_token_end": 3, "survived_left_truncation": True}
                ],
            },
            "record": {"layer": 0, "path": "sample_0000_layer_00.npz"},
            "record_i": 1,
            "record_index": 0,
            "record_count": 1,
            "base_dir": str(base_dir),
            "scale_manifest": {"key_scale": {"0": {"s_h": [1.0, 1.0]}}},
            "g_values": [float("inf")],
            "group_filter": None,
            "task_groups": [1],
            "lambda_rel_values": [1.0],
            "seg_forget": 0.5,
            "b_prime": 2,
            "random_trials": 1,
            "seed": 99,
            "allow_fallback_sh": False,
            "collect_by_layer_group": True,
        }

        result = _process_record_task(task)

    assert result["matched_group_pairs"] == 1
    assert result["processed_pairs"] == 1
    assert sorted(result["by_layer_group"]) == [(1.0, "inf", 0, 1)]


def test_process_record_task_skip_message_keeps_group_task_context() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        k_raw = np.asarray(
            [
                [[0.0], [0.1], [0.2]],
                [[1.0], [1.1], [1.2]],
            ],
            dtype=np.float32,
        )
        np.savez(base_dir / "sample_0000_layer_00.npz", k_raw=k_raw, v=k_raw.copy())
        task = {
            "sample": {"sample_id": "smoke", "needle_spans": []},
            "record": {"layer": 0, "path": "sample_0000_layer_00.npz"},
            "record_i": 1,
            "record_index": 0,
            "record_count": 1,
            "task_i": 2,
            "task_count": 4,
            "base_dir": str(base_dir),
            "scale_manifest": {"key_scale": {"0": {"s_h": [1.0, 1.0]}}},
            "g_values": [float("inf")],
            "group_filter": None,
            "task_groups": [1],
            "lambda_rel_values": [1.0],
            "seg_forget": 0.5,
            "b_prime": 2,
            "random_trials": 1,
            "seed": 99,
            "allow_fallback_sh": False,
            "collect_by_layer_group": True,
            "log_timing": True,
        }

        result = _process_record_task(task)

    assert result["matched_group_pairs"] == 1
    assert result["processed_pairs"] == 0
    assert "task=2/4" in result["message"]
    assert "groups=[1]" in result["message"]
    assert "elapsed=" in result["message"]
