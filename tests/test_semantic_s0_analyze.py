import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "unused" / "semantic_s0_analyze.py"
_SPEC = importlib.util.spec_from_file_location("semantic_s0_analyze_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_analysis = _MODULE.build_analysis
_parse_int_spec = _MODULE._parse_int_spec
_worst_layer_groups = _MODULE._worst_layer_groups


def _row(g_max: str, l_block: int, layer: int, group: int, key: float, value: float, span: float) -> dict:
    return {
        "g_max": g_max,
        "l_block": l_block,
        "layer": layer,
        "group": group,
        "token_weighted_key_var_relative": key,
        "token_weighted_value_var_relative": value,
        "entry_count_mean": 10.0,
        "entry_span_global_quantiles": {"p99": span},
    }


def _overall(g_max: str, l_block: int, key: float, value: float, span: float) -> dict:
    return {
        "g_max": g_max,
        "l_block": l_block,
        "token_weighted_key_var_relative": key,
        "token_weighted_value_var_relative": value,
        "entry_count_mean": 10.0,
        "pad_entry_count_mean": 0.0,
        "cluster_count_mean": 2.0,
        "segment_count_mean": 3.0,
        "entry_span_global_quantiles": {"p99": span},
        "entry_span_global_max": span,
    }


def _baseline(layer: int, group: int, key: float = 1.0, value: float = 1.0, span: float = 100.0) -> dict:
    return {
        "layer": layer,
        "group": group,
        "token_weighted_key_var_relative": key,
        "token_weighted_value_var_relative": value,
        "entry_count_mean": 20.0,
        "entry_span_global_quantiles": {"p99": span},
    }


def _payload() -> dict:
    return {
        "version": 1,
        "kind": "semantic_logkv_s0_0_sweep",
        "config": {"g_max": ["256", "inf"], "l_block": [0, 1], "lambda_rel": 1.0, "b_prime": 8},
        "overall_by_config": [
            _overall("256", 0, 0.50, 0.40, 90.0),
            _overall("inf", 0, 0.50, 0.40, 90.0),
            _overall("256", 1, 0.30, 0.60, 40.0),
        ],
        "by_layer_group": [
            _row("256", 0, 0, 0, 0.50, 0.40, 90.0),
            _row("inf", 0, 0, 0, 0.50, 0.40, 90.0),
            _row("256", 1, 0, 0, 0.30, 0.60, 40.0),
            _row("256", 0, 1, 0, 0.50, 0.40, 90.0),
            _row("inf", 0, 1, 0, 0.50, 0.40, 90.0),
            _row("256", 1, 1, 0, 0.20, 0.50, 30.0),
        ],
        "vanilla_logkv_compressed_prefix_baseline_by_layer_group": [
            _baseline(0, 0),
            _baseline(1, 0),
        ],
    }


def test_parse_int_spec_accepts_ranges_and_singletons() -> None:
    assert _parse_int_spec("23-25,28") == {23, 24, 25, 28}
    assert _parse_int_spec("3-1") == {1, 2, 3}
    assert _parse_int_spec(None) is None


def test_build_analysis_collapses_l_block_zero_duplicate_configs() -> None:
    analysis = build_analysis(_payload())

    configs = [(row["g_max"], row["l_block"]) for row in analysis["config_rankings"]]
    assert configs == [("256", 1), ("inf", 0)]
    assert any("collapsed 1 duplicate l_block=0 config rows" in warning for warning in analysis["warnings"])


def test_baseline_comparison_uses_layer_group_ratios_after_l0_dedupe() -> None:
    analysis = build_analysis(_payload())
    by_config = {(row["g_max"], row["l_block"]): row for row in analysis["baseline_comparison"]}

    assert by_config[("256", 1)]["key_ratio_median"] == 0.25
    assert by_config[("256", 1)]["value_ratio_median"] == 0.55
    assert by_config[("256", 1)]["entry_count_ratio_median"] == 0.5
    assert by_config[("inf", 0)]["layer_group_count"] == 2


def test_layer_filter_reaggregates_from_by_layer_group_not_overall_rows() -> None:
    analysis = build_analysis(_payload(), layers={1})
    by_config = {(row["g_max"], row["l_block"]): row for row in analysis["config_rankings"]}

    assert by_config[("256", 1)]["token_weighted_key_var_relative"] == 0.20
    assert by_config[("inf", 0)]["token_weighted_key_var_relative"] == 0.50


def test_worst_layer_groups_sort_by_largest_key_ratio() -> None:
    payload = _payload()
    rows = _worst_layer_groups(
        payload["by_layer_group"],
        payload["vanilla_logkv_compressed_prefix_baseline_by_layer_group"],
        config=("256", 1),
        layers=None,
        groups=None,
        collapse_l0=True,
        key_metric="token_weighted_key_var_relative",
        value_metric="token_weighted_value_var_relative",
        limit=1,
    )

    assert rows == [
        {
            "layer": 0,
            "group": 0,
            "key_ratio": 0.3,
            "value_ratio": 0.6,
            "span_p99_ratio": 0.4,
            "entry_count_ratio": 0.5,
            "key_rel": 0.3,
            "value_rel": 0.6,
        }
    ]
