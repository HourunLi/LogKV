import csv
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "unused" / "semantic_s0_analyze.py"
_SPEC = importlib.util.spec_from_file_location("semantic_s0_analyze_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_analysis = _MODULE.build_analysis
build_anchor_analysis = _MODULE.build_anchor_analysis
build_needle_analysis = _MODULE.build_needle_analysis
_parse_int_spec = _MODULE._parse_int_spec
_write_anchor_csv = _MODULE._write_anchor_csv
_worst_layer_groups = _MODULE._worst_layer_groups
_write_csv_by_layer = _MODULE._write_csv_by_layer
_write_needle_csv = _MODULE._write_needle_csv


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


def _needle_row(
    g_max: str,
    layer: int | None,
    group: int | None,
    *,
    needle_iso: int,
    needle_total: int,
    random_iso: int,
    random_total: int,
    span_all_iso: int = 1,
    random_span_all_iso: int = 0,
) -> dict:
    row = {
        "g_max": g_max,
        "sample_groups": 1,
        "sample_groups_with_needle": 1,
        "span_count": 1,
        "needle_token_count": needle_total,
        "needle_token_isolated_count": needle_iso,
        "needle_token_isolated_rate": needle_iso / needle_total,
        "span_all_tokens_isolated_count": span_all_iso,
        "span_all_tokens_isolated_rate": span_all_iso,
        "span_any_token_isolated_count": 1 if needle_iso else 0,
        "span_any_token_isolated_rate": 1.0 if needle_iso else 0.0,
        "random_span_count": 1,
        "random_token_count": random_total,
        "random_token_isolated_count": random_iso,
        "random_token_isolated_rate": random_iso / random_total,
        "random_span_all_tokens_isolated_count": random_span_all_iso,
        "random_span_all_tokens_isolated_rate": random_span_all_iso,
        "random_span_any_token_isolated_count": 1 if random_iso else 0,
        "random_span_any_token_isolated_rate": 1.0 if random_iso else 0.0,
        "token_isolation_lift": (needle_iso / needle_total) / (random_iso / random_total),
        "span_all_tokens_isolation_lift": None if random_span_all_iso == 0 else span_all_iso / random_span_all_iso,
        "span_any_token_isolation_lift": 1.0 if random_iso else None,
        "cluster_size_mean": 2.0,
        "cluster_size_quantiles": {"p50": 2.0, "p90": 3.0, "p99": 4.0},
        "cluster_size_max": 5,
        "cluster_count_mean": 3.0,
        "segment_count_mean": 4.0,
    }
    if layer is not None and group is not None:
        row.update({"layer": layer, "group": group})
    return row


def _needle_payload() -> dict:
    return {
        "version": 1,
        "kind": "semantic_logkv_s0_3_needle_isolation",
        "config": {"g_max": ["256", "inf"], "lambda_rel": 1.0, "b_prime": 8, "random_trials": 4},
        "overall_by_config": [
            _needle_row("256", None, None, needle_iso=3, needle_total=4, random_iso=1, random_total=4),
            _needle_row("inf", None, None, needle_iso=2, needle_total=4, random_iso=1, random_total=4),
        ],
        "by_layer_group": [
            _needle_row("256", 0, 0, needle_iso=3, needle_total=4, random_iso=1, random_total=4),
            _needle_row("inf", 0, 0, needle_iso=2, needle_total=4, random_iso=1, random_total=4),
            _needle_row("256", 1, 0, needle_iso=1, needle_total=2, random_iso=1, random_total=2),
            _needle_row("inf", 1, 0, needle_iso=2, needle_total=2, random_iso=1, random_total=2),
        ],
    }


def _anchor_row(
    scheme: str,
    g_max: str | None,
    l_block: int | None,
    *,
    entry: float,
    real: float,
    logical: float,
    fixed3: float,
    layer: int | None = None,
    group: int | None = None,
) -> dict:
    row = {
        "scheme": scheme,
        "g_max": g_max,
        "l_block": l_block,
        "sample_groups": 1,
        "entry_count_mean": entry,
        "real_entry_count_mean": real,
        "pad_entry_count_mean": entry - real,
        "token_count_mean": 128.0,
        "logical_anchor_count_mean": logical,
        "fixed3_anchor_count_mean": fixed3,
        "fixed3_entry_anchor_count_mean": fixed3,
        "fixed3_real_anchor_count_mean": real * 3.0,
        "lo_hi_anchor_count_mean": real * 2.0,
        "E_M": logical / real,
        "E_M_lo_hi": 2.0,
        "anchor_count_per_token": logical / 128.0,
        "fixed3_over_logical_ratio": fixed3 / logical,
        "fixed3_real_over_logical_ratio": real * 3.0 / logical,
        "logical_over_lo_hi_ratio": logical / (real * 2.0),
        "gather_savings_fraction_vs_fixed3": 1.0 - logical / fixed3,
        "m_counts": {"1": 2, "2": 5, "3": 3},
        "m_fractions": {"1": 0.2, "2": 0.5, "3": 0.3},
        "entry_width_max": 8,
        "entry_span_max": 16,
    }
    if layer is not None and group is not None:
        row.update({"layer": layer, "group": group})
    return row


def _anchor_payload() -> dict:
    single = "single_cluster_bprime_baseline"
    vanilla_full = "vanilla_logkv_full_cache_baseline"
    semantic = "semantic"
    return {
        "version": 1,
        "kind": "semantic_logkv_s0_6_anchor_dedup",
        "config": {"g_max": ["256", "inf"], "l_block": [1], "lambda_rel": 1.0, "b_prime": 8},
        "overall_by_config": [
            _anchor_row(single, None, None, entry=10.0, real=10.0, logical=10.0, fixed3=30.0),
            _anchor_row(vanilla_full, None, None, entry=100.0, real=100.0, logical=100.0, fixed3=300.0),
            _anchor_row(semantic, "256", 1, entry=32.0, real=30.0, logical=45.0, fixed3=96.0),
            _anchor_row(semantic, "inf", 1, entry=30.0, real=30.0, logical=60.0, fixed3=90.0),
        ],
        "by_layer_group": [
            _anchor_row(single, None, None, entry=10.0, real=10.0, logical=10.0, fixed3=30.0, layer=0, group=0),
            _anchor_row(vanilla_full, None, None, entry=100.0, real=100.0, logical=100.0, fixed3=300.0, layer=0, group=0),
            _anchor_row(semantic, "256", 1, entry=12.0, real=10.0, logical=12.0, fixed3=36.0, layer=0, group=0),
            _anchor_row(semantic, "inf", 1, entry=12.0, real=10.0, logical=20.0, fixed3=36.0, layer=0, group=0),
            _anchor_row(single, None, None, entry=10.0, real=10.0, logical=10.0, fixed3=30.0, layer=1, group=0),
            _anchor_row(vanilla_full, None, None, entry=100.0, real=100.0, logical=100.0, fixed3=300.0, layer=1, group=0),
            _anchor_row(semantic, "256", 1, entry=12.0, real=10.0, logical=15.0, fixed3=36.0, layer=1, group=0),
            _anchor_row(semantic, "inf", 1, entry=12.0, real=10.0, logical=11.0, fixed3=36.0, layer=1, group=0),
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


def test_auto_baseline_does_not_fall_back_to_position_or_single_cluster() -> None:
    # position_baseline_by_layer_group is the old field name that used to *be*
    # misread as "the vanilla baseline" before it was renamed to make clear it
    # is a same-B'-budget position-order CONTROL, not the deployed LogKV (see
    # BASELINE_KEYS["single_cluster"]'s backward-compat entry). --baseline
    # auto (the default) must never silently resolve to it, or to
    # single_cluster_bprime_baseline_by_layer_group, when no real vanilla
    # field is present -- doing so would produce ratio numbers that read as
    # "vs. deployed LogKV" while actually comparing against a control.
    payload = _payload()
    del payload["vanilla_logkv_compressed_prefix_baseline_by_layer_group"]
    payload["position_baseline_by_layer_group"] = [_baseline(0, 0), _baseline(1, 0)]
    payload["single_cluster_bprime_baseline_by_layer_group"] = [_baseline(0, 0), _baseline(1, 0)]

    analysis = build_analysis(payload, baseline="auto")

    assert analysis["baseline"]["field"] is None
    assert analysis["baseline"]["row_count"] == 0
    assert analysis["baseline_comparison"] == []
    assert any("no matching baseline rows were found" in warning for warning in analysis["warnings"])


def test_explicit_position_baseline_still_resolves_when_requested() -> None:
    # The old position_baseline_by_layer_group field is still usable, but only
    # when a caller explicitly opts into the position-order control via
    # --baseline position (or single_cluster) rather than getting it for free
    # under auto.
    payload = _payload()
    del payload["vanilla_logkv_compressed_prefix_baseline_by_layer_group"]
    payload["position_baseline_by_layer_group"] = [_baseline(0, 0), _baseline(1, 0)]

    analysis = build_analysis(payload, baseline="position")

    assert analysis["baseline"]["field"] == "position_baseline_by_layer_group"
    assert analysis["baseline"]["row_count"] == 2
    assert analysis["baseline_comparison"] != []


def test_write_csv_by_layer_persists_the_per_layer_table(tmp_path: Path) -> None:
    # --csv (see _write_csv) always writes the (g_max, l_block) table aggregated
    # across every layer/group, even when --by_layer was passed -- the per-layer
    # breakdown never made it into any file before this writer existed. Check it
    # actually persists one row per layer with that layer's own winning config,
    # not a copy of the g_max/l_block table (which has no "layer" column at all).
    analysis = build_analysis(_payload())
    out = tmp_path / "by_layer.csv"

    _write_csv_by_layer(out, analysis["best_by_layer"])

    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert fieldnames is not None and "layer" in fieldnames and "g_max" in fieldnames
    assert [row["layer"] for row in rows] == ["0", "1"]
    assert [row["g_max"] for row in rows] == ["256", "256"]
    assert [row["l_block"] for row in rows] == ["1", "1"]
    assert float(rows[0]["key_ratio_median"]) == 0.3
    assert float(rows[0]["value_ratio_median"]) == 0.6
    assert float(rows[1]["key_ratio_median"]) == 0.2
    assert float(rows[1]["value_ratio_median"]) == 0.5


def test_best_by_layer_is_populated_independent_of_any_by_layer_flag() -> None:
    # analysis["best_by_layer"] is computed unconditionally in build_analysis
    # whenever a baseline resolves -- it is not gated by the --by_layer CLI
    # flag (that flag only controls whether _print_report also prints it). This
    # is what lets --csv_by_layer work without also requiring --by_layer.
    analysis = build_analysis(_payload())
    assert len(analysis["best_by_layer"]) == 2


def test_csv_by_layer_has_nothing_to_write_without_a_baseline() -> None:
    payload = _payload()
    del payload["vanilla_logkv_compressed_prefix_baseline_by_layer_group"]
    analysis = build_analysis(payload, baseline="none")
    assert analysis.get("best_by_layer", []) == []


def test_build_needle_analysis_ranks_by_isolation_lift_and_reaggregates_layer_scope() -> None:
    analysis = build_needle_analysis(_needle_payload())

    assert [row["g_max"] for row in analysis["config_rankings"]] == ["256", "inf"]
    assert analysis["config_rankings"][0]["token_isolation_lift"] == 3.0
    assert [row["g_max"] for row in analysis["best_by_layer"]] == ["256", "inf"]

    scoped = build_needle_analysis(_needle_payload(), layers={1})

    assert scoped["config_source"] == "by_layer_group_count_aggregate"
    assert [row["g_max"] for row in scoped["config_rankings"]] == ["inf", "256"]
    assert scoped["config_rankings"][0]["needle_token_isolated_rate"] == 1.0


def test_write_needle_csv_persists_ranked_needle_fields(tmp_path: Path) -> None:
    analysis = build_needle_analysis(_needle_payload())
    out = tmp_path / "needle.csv"

    _write_needle_csv(out, analysis["csv_rows"])

    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert fieldnames is not None and "token_isolation_lift" in fieldnames
    assert rows[0]["rank"] == "1"
    assert rows[0]["g_max"] == "256"
    assert float(rows[0]["cluster_size_quantiles.p90"]) == 3.0


def test_build_anchor_analysis_ranks_by_e_m_and_attaches_baseline_ratios() -> None:
    analysis = build_anchor_analysis(_anchor_payload())

    best = analysis["config_rankings"][0]
    assert best["g_max"] == "256"
    assert best["E_M"] == 1.5
    assert best["entry_count_mean_ratio_vs_single_cluster"] == 3.2
    assert best["logical_anchor_count_mean_ratio_vs_single_cluster"] == 4.5
    assert best["current_scheme_physical_slot_count_mean_ratio_vs_vanilla_full"] == 0.96
    assert [row["g_max"] for row in analysis["best_by_layer"]] == ["256", "inf"]

    scoped = build_anchor_analysis(_anchor_payload(), layers={1})

    assert scoped["config_source"] == "by_layer_group_count_aggregate"
    assert [row["g_max"] for row in scoped["config_rankings"]] == ["inf", "256"]
    assert scoped["config_rankings"][0]["E_M"] == 1.1


def test_anchor_analysis_collapses_l_block_zero_duplicate_configs() -> None:
    payload = _anchor_payload()
    payload["overall_by_config"].extend(
        [
            _anchor_row("semantic", "128", 0, entry=8.0, real=8.0, logical=8.0, fixed3=24.0),
            _anchor_row("semantic", "inf", 0, entry=8.0, real=8.0, logical=8.0, fixed3=24.0),
        ]
    )

    analysis = build_anchor_analysis(payload)
    l0_configs = [(row["g_max"], row["l_block"]) for row in analysis["config_rankings"] if row["l_block"] == 0]

    assert l0_configs == [("inf", 0)]
    assert any("duplicate semantic l_block=0" in warning for warning in analysis["warnings"])

    keep = build_anchor_analysis(payload, collapse_l0=False)
    kept_l0 = [(row["g_max"], row["l_block"]) for row in keep["config_rankings"] if row["l_block"] == 0]
    assert kept_l0 == [("128", 0), ("inf", 0)]


def test_write_anchor_csv_persists_ranked_anchor_fields(tmp_path: Path) -> None:
    analysis = build_anchor_analysis(_anchor_payload())
    out = tmp_path / "anchor.csv"

    _write_anchor_csv(out, analysis["csv_rows"])

    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert fieldnames is not None and "E_M" in fieldnames and "m_fractions.3" in fieldnames
    assert rows[0]["rank"] == "1"
    assert rows[0]["g_max"] == "256"
    assert float(rows[0]["m_fractions.3"]) == 0.3
    assert float(rows[0]["current_scheme_physical_slot_count_mean_ratio_vs_vanilla_full"]) == 0.96


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
