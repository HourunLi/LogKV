#!/usr/bin/env python
"""Summarize SemanticLogKV Stage-0 analysis JSON output.

The Stage-0 JSON files are intentionally machine-readable and can be awkward
to inspect directly. This script turns them into compact reports:

  * S0.0 sweep:
  * top ``(g_max, l_block)`` configs by scale-comparable key/value variance;
  * per-layer/group ratios against the real vanilla LogKV compressed-prefix
    baseline;
  * span/entry-count tradeoff signals for S0.5;
  * optional layer-wise winners and CSV/JSON summary exports.
  * S0.3 needle isolation:
    top ``(lambda_rel, g_max, k_max)`` configs by exact-entry needle isolation
    lift over same-length random spans, plus the older small-cluster proxy,
    Ward/K_max binding probes, optional per-layer winners and CSV export.
  * S0.6 anchor dedup:
    top semantic ``(g_max, l_block, k_max)`` configs by ``E[M]``/gather
    potential, baseline rows, Ward/K_max binding probes, optional per-layer
    winners, and CSV export.

Examples:
    python unused/semantic_s0_analyze.py stage0_dump/s0_sweep.json
    python unused/semantic_s0_analyze.py stage0_dump/s0_3_needle_isolation.json --csv stage0_dump/s0_3.csv
    python unused/semantic_s0_analyze.py stage0_dump/s0_6_anchor_dedup.json --csv stage0_dump/s0_6.csv
    python unused/semantic_s0_analyze.py './stage0_dump/*s0_sweep*.json' --layers 23-26 --top 20
    python unused/semantic_s0_analyze.py stage0_dump/s0_sweep.json --csv stage0_dump/s0_summary.csv
    python unused/semantic_s0_analyze.py stage0_dump/s0_sweep.json \
      --csv_by_layer stage0_dump/s0_summary_by_layer.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


PRIMARY_KEY_METRIC = "token_weighted_key_var_relative"
PRIMARY_VALUE_METRIC = "token_weighted_value_var_relative"
SPAN_P99_METRIC = "entry_span_global_quantiles.p99"
ENTRY_COUNT_METRIC = "entry_count_mean"
KIND_SWEEP = "semantic_logkv_s0_0_sweep"
KIND_NEEDLE = "semantic_logkv_s0_3_needle_isolation"
KIND_ANCHOR = "semantic_logkv_s0_6_anchor_dedup"
SCHEME_SEMANTIC = "semantic"
SCHEME_SINGLE_CLUSTER = "single_cluster_bprime_baseline"
SCHEME_VANILLA_COMPRESSED = "vanilla_logkv_compressed_prefix_baseline"
SCHEME_VANILLA_FULL = "vanilla_logkv_full_cache_baseline"

NEEDLE_CSV_FIELDS = [
    "rank",
    "lambda_rel",
    "g_max",
    "k_max",
    "token_exact_entry_lift",
    "needle_token_exact_entry_rate",
    "random_token_exact_entry_rate",
    "token_isolation_lift",
    "needle_token_isolated_rate",
    "random_token_isolated_rate",
    "needle_cluster_merged_by_ward_rate",
    "needle_token_ward_merged_rate",
    "span_any_ward_merged_rate",
    "K_max_binding_rate",
    "ward_merge_count_mean",
    "k_max_binding_count_mean",
    "new_cluster_attempt_count_mean",
    "span_all_tokens_exact_entry_lift",
    "span_all_tokens_exact_entry_rate",
    "random_span_all_tokens_exact_entry_rate",
    "span_all_tokens_isolation_lift",
    "span_all_tokens_isolated_rate",
    "random_span_all_tokens_isolated_rate",
    "span_any_token_exact_entry_lift",
    "span_any_token_exact_entry_rate",
    "random_span_any_token_exact_entry_rate",
    "span_any_token_isolation_lift",
    "span_any_token_isolated_rate",
    "random_span_any_token_isolated_rate",
    "cluster_size_mean",
    "cluster_size_quantiles.p50",
    "cluster_size_quantiles.p90",
    "cluster_size_quantiles.p99",
    "cluster_size_max",
    "cluster_count_mean",
    "segment_count_mean",
    "sample_groups",
    "sample_groups_with_needle",
    "span_count",
    "needle_token_count",
    "random_token_count",
    "layer_group_count",
]

NEEDLE_LAYER_CSV_FIELDS = [
    "layer",
    "lambda_rel",
    "g_max",
    "k_max",
    "token_exact_entry_lift",
    "needle_token_exact_entry_rate",
    "random_token_exact_entry_rate",
    "token_isolation_lift",
    "needle_token_isolated_rate",
    "random_token_isolated_rate",
    "needle_cluster_merged_by_ward_rate",
    "needle_token_ward_merged_rate",
    "span_any_ward_merged_rate",
    "K_max_binding_rate",
    "ward_merge_count_mean",
    "k_max_binding_count_mean",
    "new_cluster_attempt_count_mean",
    "span_all_tokens_exact_entry_lift",
    "span_all_tokens_exact_entry_rate",
    "random_span_all_tokens_exact_entry_rate",
    "span_all_tokens_isolation_lift",
    "span_all_tokens_isolated_rate",
    "random_span_all_tokens_isolated_rate",
    "span_any_token_exact_entry_lift",
    "span_any_token_exact_entry_rate",
    "random_span_any_token_exact_entry_rate",
    "span_any_token_isolation_lift",
    "span_any_token_isolated_rate",
    "random_span_any_token_isolated_rate",
    "cluster_count_mean",
    "segment_count_mean",
    "sample_groups",
    "span_count",
    "needle_token_count",
    "layer_group_count",
]

ANCHOR_CSV_FIELDS = [
    "rank",
    "scheme",
    "lambda_rel",
    "g_max",
    "l_block",
    "k_max",
    "E_M",
    "K_max_binding_rate",
    "ward_merge_count_mean",
    "k_max_binding_count_mean",
    "new_cluster_attempt_count_mean",
    "m_fractions.1",
    "m_fractions.2",
    "m_fractions.3",
    "gather_savings_fraction_vs_fixed3",
    "logical_anchor_count_mean",
    "fixed3_anchor_count_mean",
    "current_scheme_physical_slot_count_mean",
    "entry_count_mean",
    "real_entry_count_mean",
    "pad_entry_count_mean",
    "entry_count_mean_ratio_vs_single_cluster",
    "logical_anchor_count_mean_ratio_vs_single_cluster",
    "fixed3_anchor_count_mean_ratio_vs_single_cluster",
    "current_scheme_physical_slot_count_mean_ratio_vs_vanilla_full",
    "sample_groups",
    "layer_group_count",
]

ANCHOR_LAYER_CSV_FIELDS = [
    "layer",
    "lambda_rel",
    "g_max",
    "l_block",
    "k_max",
    "E_M",
    "K_max_binding_rate",
    "ward_merge_count_mean",
    "k_max_binding_count_mean",
    "new_cluster_attempt_count_mean",
    "m_fractions.1",
    "m_fractions.2",
    "m_fractions.3",
    "gather_savings_fraction_vs_fixed3",
    "logical_anchor_count_mean",
    "fixed3_anchor_count_mean",
    "current_scheme_physical_slot_count_mean",
    "entry_count_mean",
    "real_entry_count_mean",
    "pad_entry_count_mean",
    "sample_groups",
    "layer_group_count",
]

BASELINE_KEYS = {
    "vanilla": (
        "vanilla_logkv_compressed_prefix_baseline_by_layer_group",
        "vanilla_logkv_baseline_by_layer_group",
    ),
    "vanilla_full": ("vanilla_logkv_full_cache_baseline_by_layer_group",),
    "single_cluster": (
        "single_cluster_bprime_baseline_by_layer_group",
        # One older intermediate version used this name before the baseline was
        # renamed to clarify that it is not the deployed vanilla LogKV.
        "position_baseline_by_layer_group",
    ),
    "position": ("position_baseline_by_layer_group",),
}


def _load_json(path_or_glob: str) -> tuple[Path, dict[str, Any]]:
    pattern = str(Path(path_or_glob).expanduser())
    matches = sorted(glob.glob(pattern))
    path = Path(matches[-1] if matches else pattern)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return path, json.load(f)


def _parse_int_spec(spec: str | None) -> set[int] | None:
    if spec is None:
        return None
    values: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(chunk))
    return values


def _format_layers(layers: set[int] | None) -> str:
    if layers is None:
        return "all"
    if not layers:
        return "none"
    ordered = sorted(layers)
    ranges: list[str] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def _metric(row: dict[str, Any] | None, dotted_key: str) -> float | None:
    if row is None:
        return None
    if dotted_key in row:
        return _finite_float(row.get(dotted_key))
    current: Any = row
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return _finite_float(current)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = min(max(float(q), 0.0), 1.0) * (len(ordered) - 1)
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return ordered[left]
    frac = pos - left
    return ordered[left] * (1.0 - frac) + ordered[right] * frac


def _ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return value / baseline


def _config_key(row: dict[str, Any], *, collapse_l0: bool) -> tuple[str, int]:
    l_block = int(row["l_block"])
    g_max = str(row["g_max"])
    if collapse_l0 and l_block == 0:
        # semantic_s0_sweep.py routes every l_block=0 cell with effective
        # g_max=inf, so the nominal finite-g rows are duplicate pure-semantic
        # endpoints. Collapsing keeps the report from being dominated by the
        # same row repeated once per swept g_max.
        g_max = "inf"
    return g_max, l_block


def _config_lg_key(row: dict[str, Any], *, collapse_l0: bool) -> tuple[str, int, int, int]:
    g_max, l_block = _config_key(row, collapse_l0=collapse_l0)
    return g_max, l_block, int(row["layer"]), int(row["group"])


def _row_in_scope(row: dict[str, Any], layers: set[int] | None, groups: set[int] | None) -> bool:
    if layers is not None and int(row.get("layer", -1)) not in layers:
        return False
    return not (groups is not None and int(row.get("group", -1)) not in groups)


def _dedupe_config_rows(
    rows: list[dict[str, Any]], *, collapse_l0: bool
) -> tuple[list[dict[str, Any]], int]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = _config_key(row, collapse_l0=collapse_l0)
        if key in out:
            duplicates += 1
            continue
        normalized = dict(row)
        normalized["g_max"], normalized["l_block"] = key
        out[key] = normalized
    return list(out.values()), duplicates


def _dedupe_config_layer_group_rows(
    rows: list[dict[str, Any]], *, collapse_l0: bool
) -> tuple[list[dict[str, Any]], int]:
    out: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = _config_lg_key(row, collapse_l0=collapse_l0)
        if key in out:
            duplicates += 1
            continue
        normalized = dict(row)
        normalized["g_max"], normalized["l_block"], normalized["layer"], normalized["group"] = key
        out[key] = normalized
    return list(out.values()), duplicates


def _baseline_layer_group_map(
    rows: list[dict[str, Any]], *, layers: set[int] | None, groups: set[int] | None
) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if not _row_in_scope(row, layers, groups):
            continue
        key = int(row["layer"]), int(row["group"])
        out.setdefault(key, row)
    return out


def _resolve_baseline(payload: dict[str, Any], name: str) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    if name == "none":
        return None, None, []
    candidates: list[str]
    if name == "auto":
        # "auto" only ever resolves to a real vanilla-LogKV baseline field --
        # never to single_cluster/position. Those are same-B'-budget /
        # position-order CONTROLS (see route_single_cluster_bprime_ladder's
        # and single_cluster_bprime_baseline's docstrings in
        # litgpt/semantic_s0.py and unused/semantic_s0_sweep.py), not the
        # deployed LogKV, and "position_baseline_by_layer_group" is the exact
        # field name that used to *be* misread as "the vanilla baseline"
        # before it was renamed (see BASELINE_KEYS["single_cluster"]'s
        # backward-compat entry above). Silently falling back to it here
        # would let a JSON sweep run without a real vanilla baseline still
        # produce ratio numbers that read as "vs. deployed LogKV" under
        # --baseline auto (the default). If no real vanilla field is present,
        # _resolve_baseline returns no rows and build_analysis's caller emits
        # a warning and skips the baseline comparison entirely -- callers who
        # actually want the position/single_cluster control must ask for it
        # explicitly via --baseline single_cluster / --baseline position.
        candidates = list(BASELINE_KEYS["vanilla"])
    else:
        candidates = list(BASELINE_KEYS[name])
    for key in candidates:
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            return name, key, rows
    return name, None, []


def _aggregate_config_rows(
    rows: list[dict[str, Any]],
    *,
    key_metric: str,
    value_metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_config_key(row, collapse_l0=False)].append(row)

    metrics = [
        key_metric,
        value_metric,
        "cluster_count_mean",
        "segment_count_mean",
        "entry_count_mean",
        "nonpad_entry_count_mean",
        "pad_entry_count_mean",
        "entry_width_global_quantiles.p90",
        "entry_width_global_quantiles.p99",
        "entry_width_global_max",
        "entry_span_global_quantiles.p90",
        "entry_span_global_quantiles.p99",
        "entry_span_global_max",
    ]
    out: list[dict[str, Any]] = []
    for (g_max, l_block), group_rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        row: dict[str, Any] = {
            "g_max": g_max,
            "l_block": l_block,
            "layer_group_count": len(group_rows),
            "source": "by_layer_group_mean",
        }
        for metric in metrics:
            values = [_metric(group_row, metric) for group_row in group_rows]
            clean_values = [value for value in values if value is not None]
            row[metric] = _mean(clean_values)
        out.append(row)
    return out


def _config_summary_rows(
    payload: dict[str, Any],
    *,
    layers: set[int] | None,
    groups: set[int] | None,
    collapse_l0: bool,
    key_metric: str,
    value_metric: str,
) -> tuple[list[dict[str, Any]], str, int]:
    if layers is None and groups is None:
        rows, duplicates = _dedupe_config_rows(
            payload.get("overall_by_config", []),
            collapse_l0=collapse_l0,
        )
        return rows, "overall_by_config", duplicates

    by_lg_rows, duplicates = _dedupe_config_layer_group_rows(
        payload.get("by_layer_group", []),
        collapse_l0=collapse_l0,
    )
    scoped = [row for row in by_lg_rows if _row_in_scope(row, layers, groups)]
    rows = _aggregate_config_rows(scoped, key_metric=key_metric, value_metric=value_metric)
    return rows, "by_layer_group_mean", duplicates


def _ratio_summary(
    rows: list[dict[str, Any]],
    baseline_by_lg: dict[tuple[int, int], dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    ratios: list[float] = []
    for row in rows:
        baseline = baseline_by_lg.get((int(row["layer"]), int(row["group"])))
        ratio = _ratio(_metric(row, metric), _metric(baseline, metric))
        if ratio is not None:
            ratios.append(ratio)
    return {
        "valid": len(ratios),
        "wins": sum(1 for value in ratios if value < 1.0),
        "mean": _mean(ratios),
        "median": _median(ratios),
        "p90": _quantile(ratios, 0.90),
        "max": max(ratios) if ratios else None,
    }


def _score_from_ratios(row: dict[str, Any]) -> float | None:
    key_ratio = _metric(row, "key_ratio_median")
    value_ratio = _metric(row, "value_ratio_median")
    if key_ratio is None:
        return None
    if value_ratio is None:
        return key_ratio
    return math.sqrt(max(key_ratio, 0.0) * max(value_ratio, 0.0))


def _compare_to_baseline(
    by_layer_group_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    layers: set[int] | None,
    groups: set[int] | None,
    collapse_l0: bool,
    key_metric: str,
    value_metric: str,
) -> tuple[list[dict[str, Any]], int]:
    deduped_rows, duplicates = _dedupe_config_layer_group_rows(by_layer_group_rows, collapse_l0=collapse_l0)
    baseline_by_lg = _baseline_layer_group_map(baseline_rows, layers=layers, groups=groups)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in deduped_rows:
        if _row_in_scope(row, layers, groups) and (int(row["layer"]), int(row["group"])) in baseline_by_lg:
            grouped[_config_key(row, collapse_l0=False)].append(row)

    out: list[dict[str, Any]] = []
    for (g_max, l_block), rows in grouped.items():
        key_stats = _ratio_summary(rows, baseline_by_lg, key_metric)
        value_stats = _ratio_summary(rows, baseline_by_lg, value_metric)
        span_stats = _ratio_summary(rows, baseline_by_lg, SPAN_P99_METRIC)
        entry_stats = _ratio_summary(rows, baseline_by_lg, ENTRY_COUNT_METRIC)
        row = {
            "g_max": g_max,
            "l_block": l_block,
            "layer_group_count": len(rows),
            "key_ratio_median": key_stats["median"],
            "key_ratio_mean": key_stats["mean"],
            "key_ratio_p90": key_stats["p90"],
            "key_ratio_max": key_stats["max"],
            "key_wins": key_stats["wins"],
            "key_valid": key_stats["valid"],
            "value_ratio_median": value_stats["median"],
            "value_ratio_mean": value_stats["mean"],
            "value_ratio_p90": value_stats["p90"],
            "value_ratio_max": value_stats["max"],
            "value_wins": value_stats["wins"],
            "value_valid": value_stats["valid"],
            "span_p99_ratio_median": span_stats["median"],
            "span_p99_ratio_p90": span_stats["p90"],
            "span_p99_wins": span_stats["wins"],
            "span_p99_valid": span_stats["valid"],
            "entry_count_ratio_median": entry_stats["median"],
            "entry_count_ratio_p90": entry_stats["p90"],
            "entry_count_valid": entry_stats["valid"],
        }
        row["score"] = _score_from_ratios(row)
        out.append(row)
    return out, duplicates


def _best_by_l_block(rows: list[dict[str, Any]], *, fallback_metric: str = PRIMARY_KEY_METRIC) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        score = _metric(row, "score")
        if score is None:
            score = _metric(row, fallback_metric)
        if score is None:
            continue
        l_block = int(row["l_block"])
        old = best.get(l_block)
        if old is None:
            old_score = None
        else:
            old_score = _metric(old, "score")
            if old_score is None:
                old_score = _metric(old, fallback_metric)
        if old is None or old_score is None or score < old_score:
            best[l_block] = row
    return [best[key] for key in sorted(best)]


def _best_by_layer(
    by_layer_group_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    layers: set[int] | None,
    groups: set[int] | None,
    collapse_l0: bool,
    key_metric: str,
    value_metric: str,
) -> list[dict[str, Any]]:
    deduped_rows, _duplicates = _dedupe_config_layer_group_rows(by_layer_group_rows, collapse_l0=collapse_l0)
    baseline_by_lg = _baseline_layer_group_map(baseline_rows, layers=layers, groups=groups)
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in deduped_rows:
        if not _row_in_scope(row, layers, groups):
            continue
        layer_group = int(row["layer"]), int(row["group"])
        if layer_group not in baseline_by_lg:
            continue
        g_max, l_block = _config_key(row, collapse_l0=False)
        grouped[(int(row["layer"]), g_max, l_block)].append(row)

    candidates: list[dict[str, Any]] = []
    for (layer, g_max, l_block), rows in grouped.items():
        base_for_layer = {key: value for key, value in baseline_by_lg.items() if key[0] == layer}
        key_stats = _ratio_summary(rows, base_for_layer, key_metric)
        value_stats = _ratio_summary(rows, base_for_layer, value_metric)
        row = {
            "layer": layer,
            "g_max": g_max,
            "l_block": l_block,
            "group_count": len(rows),
            "key_ratio_median": key_stats["median"],
            "value_ratio_median": value_stats["median"],
            "key_wins": key_stats["wins"],
            "key_valid": key_stats["valid"],
        }
        row["score"] = _score_from_ratios(row)
        candidates.append(row)

    by_layer: dict[int, dict[str, Any]] = {}
    for row in candidates:
        score = _metric(row, "score")
        if score is None:
            continue
        old = by_layer.get(int(row["layer"]))
        old_score = None if old is None else _metric(old, "score")
        if old is None or old_score is None or score < old_score:
            by_layer[int(row["layer"])] = row
    return [by_layer[layer] for layer in sorted(by_layer)]


def _parse_config_selector(text: str) -> tuple[str, int]:
    for sep in (":", ",", "/"):
        if sep in text:
            g_max, l_block = text.split(sep, 1)
            return g_max.strip(), int(l_block.strip())
    raise ValueError(f"config selector must look like 'g_max:l_block', got {text!r}")


def _worst_layer_groups(
    by_layer_group_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    config: tuple[str, int],
    layers: set[int] | None,
    groups: set[int] | None,
    collapse_l0: bool,
    key_metric: str,
    value_metric: str,
    limit: int,
) -> list[dict[str, Any]]:
    deduped_rows, _duplicates = _dedupe_config_layer_group_rows(by_layer_group_rows, collapse_l0=collapse_l0)
    baseline_by_lg = _baseline_layer_group_map(baseline_rows, layers=layers, groups=groups)
    records: list[dict[str, Any]] = []
    for row in deduped_rows:
        if _config_key(row, collapse_l0=False) != config or not _row_in_scope(row, layers, groups):
            continue
        baseline = baseline_by_lg.get((int(row["layer"]), int(row["group"])))
        if baseline is None:
            continue
        records.append(
            {
                "layer": int(row["layer"]),
                "group": int(row["group"]),
                "key_ratio": _ratio(_metric(row, key_metric), _metric(baseline, key_metric)),
                "value_ratio": _ratio(_metric(row, value_metric), _metric(baseline, value_metric)),
                "span_p99_ratio": _ratio(_metric(row, SPAN_P99_METRIC), _metric(baseline, SPAN_P99_METRIC)),
                "entry_count_ratio": _ratio(_metric(row, ENTRY_COUNT_METRIC), _metric(baseline, ENTRY_COUNT_METRIC)),
                "key_rel": _metric(row, key_metric),
                "value_rel": _metric(row, value_metric),
            }
        )

    def sort_key(record: dict[str, Any]) -> tuple[float, int, int]:
        key_ratio = record.get("key_ratio")
        return (float("-inf") if key_ratio is None else float(key_ratio), int(record["layer"]), int(record["group"]))

    return sorted(records, key=sort_key, reverse=True)[: max(int(limit), 0)]


def build_analysis(
    payload: dict[str, Any],
    *,
    layers: set[int] | None = None,
    groups: set[int] | None = None,
    baseline: str = "auto",
    collapse_l0: bool = True,
    key_metric: str = PRIMARY_KEY_METRIC,
    value_metric: str = PRIMARY_VALUE_METRIC,
) -> dict[str, Any]:
    config_rows, config_source, config_duplicates = _config_summary_rows(
        payload,
        layers=layers,
        groups=groups,
        collapse_l0=collapse_l0,
        key_metric=key_metric,
        value_metric=value_metric,
    )
    ranked_configs = sorted(
        config_rows,
        key=lambda row: (
            math.inf if _metric(row, key_metric) is None else float(_metric(row, key_metric)),
            math.inf if _metric(row, value_metric) is None else float(_metric(row, value_metric)),
        ),
    )

    baseline_name, baseline_key, baseline_rows = _resolve_baseline(payload, baseline)
    comparison: list[dict[str, Any]] = []
    comparison_duplicates = 0
    best_layers: list[dict[str, Any]] = []
    if baseline_rows:
        comparison, comparison_duplicates = _compare_to_baseline(
            payload.get("by_layer_group", []),
            baseline_rows,
            layers=layers,
            groups=groups,
            collapse_l0=collapse_l0,
            key_metric=key_metric,
            value_metric=value_metric,
        )
        comparison = sorted(
            comparison,
            key=lambda row: (
                math.inf if _metric(row, "score") is None else float(_metric(row, "score")),
                math.inf if _metric(row, "key_ratio_median") is None else float(_metric(row, "key_ratio_median")),
            ),
        )
        best_layers = _best_by_layer(
            payload.get("by_layer_group", []),
            baseline_rows,
            layers=layers,
            groups=groups,
            collapse_l0=collapse_l0,
            key_metric=key_metric,
            value_metric=value_metric,
        )

    warnings: list[str] = []
    if not ranked_configs:
        warnings.append("no config rows found for the selected scope")
    if baseline != "none" and not baseline_rows:
        warnings.append(f"baseline {baseline!r} was requested but no matching baseline rows were found")
    if config_duplicates:
        warnings.append(f"collapsed {config_duplicates} duplicate l_block=0 config rows")
    if comparison_duplicates and config_source != "overall_by_config":
        # Avoid repeating the same warning in the common no-filter case.
        warnings.append(f"collapsed {comparison_duplicates} duplicate l_block=0 by_layer_group rows")

    return {
        "kind": payload.get("kind"),
        "version": payload.get("version"),
        "config": payload.get("config", {}),
        "scope": {
            "layers": None if layers is None else sorted(layers),
            "groups": None if groups is None else sorted(groups),
            "collapse_l0": collapse_l0,
        },
        "config_source": config_source,
        "baseline": {"requested": baseline_name, "field": baseline_key, "row_count": len(baseline_rows)},
        "warnings": warnings,
        "config_rankings": ranked_configs,
        "baseline_comparison": comparison,
        "best_by_l_block": _best_by_l_block(comparison if comparison else ranked_configs, fallback_metric=key_metric),
        "best_by_layer": best_layers,
    }


def _scope_layer_group_rows(
    rows: list[dict[str, Any]],
    *,
    layers: set[int] | None,
    groups: set[int] | None,
) -> list[dict[str, Any]]:
    return [row for row in rows if _row_in_scope(row, layers, groups)]


def _weighted_mean_from_mean_rows(rows: list[dict[str, Any]], metric: str, *, weight: str = "sample_groups") -> float | None:
    num = 0.0
    den = 0.0
    for row in rows:
        value = _metric(row, metric)
        row_weight = _metric(row, weight)
        if value is None or row_weight is None:
            continue
        num += value * row_weight
        den += row_weight
    return num / den if den else None


def _fraction_from_counts(counts: dict[str, int], key: str, denominator: float) -> float | None:
    return float(counts.get(key, 0)) / denominator if denominator else None


def _needle_default_lambda_rel(payload: dict[str, Any]) -> float | None:
    config = payload.get("config") or {}
    value = config.get("lambda_rel")
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return _finite_float(value)


def _needle_row_with_default_lambda(row: dict[str, Any], default_lambda_rel: float | None) -> dict[str, Any]:
    normalized = dict(row)
    if "lambda_rel" not in normalized and default_lambda_rel is not None:
        normalized["lambda_rel"] = default_lambda_rel
    return normalized


def _needle_config_key(row: dict[str, Any]) -> tuple[float | None, str, str]:
    return _metric(row, "lambda_rel"), str(row["g_max"]), str(row.get("k_max", "unclipped"))


def _needle_config_sort_key(key: tuple[float | None, str, str]) -> tuple[float, str, str]:
    lambda_rel, g_max, k_max = key
    return (math.inf if lambda_rel is None else float(lambda_rel), g_max, k_max)


def _aggregate_needle_group(
    lambda_rel: float | None,
    g_max: str,
    k_max: str,
    rows: list[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sums: dict[str, float] = defaultdict(float)
    count_fields = [
        "sample_groups",
        "sample_groups_with_needle",
        "span_count",
        "needle_token_count",
        "needle_token_isolated_count",
        "needle_token_exact_entry_count",
        "span_all_tokens_isolated_count",
        "span_all_tokens_exact_entry_count",
        "span_any_token_isolated_count",
        "span_any_token_exact_entry_count",
        "random_span_count",
        "random_token_count",
        "random_token_isolated_count",
        "random_token_exact_entry_count",
        "random_span_all_tokens_isolated_count",
        "random_span_all_tokens_exact_entry_count",
        "random_span_any_token_isolated_count",
        "random_span_any_token_exact_entry_count",
        "needle_token_ward_merged_count",
        "needle_token_ward_touched_count",
        "span_any_ward_merged_count",
        "span_any_ward_touched_count",
        "random_token_ward_merged_count",
        "random_token_ward_touched_count",
        "random_span_any_ward_merged_count",
        "random_span_any_ward_touched_count",
    ]
    for row in rows:
        for field in count_fields:
            value = _metric(row, field)
            if value is not None:
                sums[field] += value

    token_rate = _ratio(sums["needle_token_isolated_count"], sums["needle_token_count"])
    exact_token_rate = _ratio(sums["needle_token_exact_entry_count"], sums["needle_token_count"])
    random_token_rate = _ratio(sums["random_token_isolated_count"], sums["random_token_count"])
    random_exact_token_rate = _ratio(sums["random_token_exact_entry_count"], sums["random_token_count"])
    span_all_rate = _ratio(sums["span_all_tokens_isolated_count"], sums["span_count"])
    span_all_exact_rate = _ratio(sums["span_all_tokens_exact_entry_count"], sums["span_count"])
    random_span_all_rate = _ratio(sums["random_span_all_tokens_isolated_count"], sums["random_span_count"])
    random_span_all_exact_rate = _ratio(
        sums["random_span_all_tokens_exact_entry_count"],
        sums["random_span_count"],
    )
    span_any_rate = _ratio(sums["span_any_token_isolated_count"], sums["span_count"])
    span_any_exact_rate = _ratio(sums["span_any_token_exact_entry_count"], sums["span_count"])
    random_span_any_rate = _ratio(sums["random_span_any_token_isolated_count"], sums["random_span_count"])
    random_span_any_exact_rate = _ratio(
        sums["random_span_any_token_exact_entry_count"],
        sums["random_span_count"],
    )
    needle_token_ward_merged_rate = _ratio(sums["needle_token_ward_merged_count"], sums["needle_token_count"])
    needle_token_ward_touched_rate = _ratio(sums["needle_token_ward_touched_count"], sums["needle_token_count"])
    span_any_ward_merged_rate = _ratio(sums["span_any_ward_merged_count"], sums["span_count"])
    span_any_ward_touched_rate = _ratio(sums["span_any_ward_touched_count"], sums["span_count"])
    random_token_ward_merged_rate = _ratio(sums["random_token_ward_merged_count"], sums["random_token_count"])
    random_token_ward_touched_rate = _ratio(sums["random_token_ward_touched_count"], sums["random_token_count"])
    random_span_any_ward_merged_rate = _ratio(sums["random_span_any_ward_merged_count"], sums["random_span_count"])
    random_span_any_ward_touched_rate = _ratio(sums["random_span_any_ward_touched_count"], sums["random_span_count"])
    row: dict[str, Any] = {
        "lambda_rel": lambda_rel,
        "g_max": g_max,
        "k_max": k_max,
        "sample_groups": int(sums["sample_groups"]),
        "sample_groups_with_needle": int(sums["sample_groups_with_needle"]),
        "span_count": int(sums["span_count"]),
        "needle_token_count": int(sums["needle_token_count"]),
        "needle_token_isolated_count": int(sums["needle_token_isolated_count"]),
        "needle_token_isolated_rate": token_rate,
        "needle_token_exact_entry_count": int(sums["needle_token_exact_entry_count"]),
        "needle_token_exact_entry_rate": exact_token_rate,
        "span_all_tokens_isolated_count": int(sums["span_all_tokens_isolated_count"]),
        "span_all_tokens_isolated_rate": span_all_rate,
        "span_all_tokens_exact_entry_count": int(sums["span_all_tokens_exact_entry_count"]),
        "span_all_tokens_exact_entry_rate": span_all_exact_rate,
        "span_any_token_isolated_count": int(sums["span_any_token_isolated_count"]),
        "span_any_token_isolated_rate": span_any_rate,
        "span_any_token_exact_entry_count": int(sums["span_any_token_exact_entry_count"]),
        "span_any_token_exact_entry_rate": span_any_exact_rate,
        "random_span_count": int(sums["random_span_count"]),
        "random_token_count": int(sums["random_token_count"]),
        "random_token_isolated_count": int(sums["random_token_isolated_count"]),
        "random_token_isolated_rate": random_token_rate,
        "random_token_exact_entry_count": int(sums["random_token_exact_entry_count"]),
        "random_token_exact_entry_rate": random_exact_token_rate,
        "random_span_all_tokens_isolated_count": int(sums["random_span_all_tokens_isolated_count"]),
        "random_span_all_tokens_isolated_rate": random_span_all_rate,
        "random_span_all_tokens_exact_entry_count": int(sums["random_span_all_tokens_exact_entry_count"]),
        "random_span_all_tokens_exact_entry_rate": random_span_all_exact_rate,
        "random_span_any_token_isolated_count": int(sums["random_span_any_token_isolated_count"]),
        "random_span_any_token_isolated_rate": random_span_any_rate,
        "random_span_any_token_exact_entry_count": int(sums["random_span_any_token_exact_entry_count"]),
        "random_span_any_token_exact_entry_rate": random_span_any_exact_rate,
        "needle_token_ward_merged_count": int(sums["needle_token_ward_merged_count"]),
        "needle_token_ward_merged_rate": needle_token_ward_merged_rate,
        "needle_token_ward_touched_count": int(sums["needle_token_ward_touched_count"]),
        "needle_token_ward_touched_rate": needle_token_ward_touched_rate,
        "span_any_ward_merged_count": int(sums["span_any_ward_merged_count"]),
        "span_any_ward_merged_rate": span_any_ward_merged_rate,
        "span_any_ward_touched_count": int(sums["span_any_ward_touched_count"]),
        "span_any_ward_touched_rate": span_any_ward_touched_rate,
        "needle_cluster_merged_by_ward_rate": span_any_ward_merged_rate,
        "random_token_ward_merged_count": int(sums["random_token_ward_merged_count"]),
        "random_token_ward_merged_rate": random_token_ward_merged_rate,
        "random_token_ward_touched_count": int(sums["random_token_ward_touched_count"]),
        "random_token_ward_touched_rate": random_token_ward_touched_rate,
        "random_span_any_ward_merged_count": int(sums["random_span_any_ward_merged_count"]),
        "random_span_any_ward_merged_rate": random_span_any_ward_merged_rate,
        "random_span_any_ward_touched_count": int(sums["random_span_any_ward_touched_count"]),
        "random_span_any_ward_touched_rate": random_span_any_ward_touched_rate,
        "token_isolation_lift": _ratio(token_rate, random_token_rate),
        "token_exact_entry_lift": _ratio(exact_token_rate, random_exact_token_rate),
        "span_all_tokens_isolation_lift": _ratio(span_all_rate, random_span_all_rate),
        "span_all_tokens_exact_entry_lift": _ratio(span_all_exact_rate, random_span_all_exact_rate),
        "span_any_token_isolation_lift": _ratio(span_any_rate, random_span_any_rate),
        "span_any_token_exact_entry_lift": _ratio(span_any_exact_rate, random_span_any_exact_rate),
        "cluster_count_mean": _weighted_mean_from_mean_rows(rows, "cluster_count_mean"),
        "segment_count_mean": _weighted_mean_from_mean_rows(rows, "segment_count_mean"),
        "new_cluster_attempt_count_mean": _weighted_mean_from_mean_rows(rows, "new_cluster_attempt_count_mean"),
        "k_max_binding_count_mean": _weighted_mean_from_mean_rows(rows, "k_max_binding_count_mean"),
        "K_max_binding_rate": _ratio(
            sum((_metric(row, "k_max_binding_count_mean") or 0.0) * (_metric(row, "sample_groups") or 0.0) for row in rows),
            sum((_metric(row, "new_cluster_attempt_count_mean") or 0.0) * (_metric(row, "sample_groups") or 0.0) for row in rows),
        ),
        "ward_merge_count_mean": _weighted_mean_from_mean_rows(rows, "ward_merge_count_mean"),
        "novelty_suppressed_count_mean": _weighted_mean_from_mean_rows(rows, "novelty_suppressed_count_mean"),
        "cluster_size_mean": _weighted_mean_from_mean_rows(rows, "cluster_size_mean"),
        "layer_group_count": len(rows),
    }
    if len(rows) == 1:
        row["cluster_size_quantiles"] = rows[0].get("cluster_size_quantiles")
        row["cluster_size_max"] = rows[0].get("cluster_size_max")
        row["random_cluster_size_quantiles"] = rows[0].get("random_cluster_size_quantiles")
    else:
        row["cluster_size_quantiles"] = {"p50": None, "p90": None, "p99": None}
        row["random_cluster_size_quantiles"] = {"p50": None, "p90": None, "p99": None}
        max_values = [_metric(source, "cluster_size_max") for source in rows]
        clean_max_values = [value for value in max_values if value is not None]
        row["cluster_size_max"] = max(clean_max_values) if clean_max_values else None
    if extra:
        row.update(extra)
    return row


def _needle_config_rows(
    payload: dict[str, Any],
    *,
    layers: set[int] | None,
    groups: set[int] | None,
) -> tuple[list[dict[str, Any]], str]:
    default_lambda_rel = _needle_default_lambda_rel(payload)
    if layers is None and groups is None:
        return [
            dict(
                _needle_row_with_default_lambda(row, default_lambda_rel),
                layer_group_count=row.get("sample_groups"),
            )
            for row in payload.get("overall_by_config", [])
        ], "overall_by_config"

    rows = [
        _needle_row_with_default_lambda(row, default_lambda_rel)
        for row in payload.get("by_layer_group", [])
    ]
    scoped = _scope_layer_group_rows(rows, layers=layers, groups=groups)
    grouped: dict[tuple[float | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scoped:
        grouped[_needle_config_key(row)].append(row)
    return [
        _aggregate_needle_group(lambda_rel, g_max, k_max, rows)
        for (lambda_rel, g_max, k_max), rows in sorted(grouped.items(), key=lambda item: _needle_config_sort_key(item[0]))
    ], "by_layer_group_count_aggregate"


def _needle_best_by_layer(
    payload: dict[str, Any],
    *,
    layers: set[int] | None,
    groups: set[int] | None,
) -> list[dict[str, Any]]:
    default_lambda_rel = _needle_default_lambda_rel(payload)
    rows = [
        _needle_row_with_default_lambda(row, default_lambda_rel)
        for row in payload.get("by_layer_group", [])
    ]
    scoped = _scope_layer_group_rows(rows, layers=layers, groups=groups)
    grouped: dict[tuple[int, float | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scoped:
        lambda_rel, g_max, k_max = _needle_config_key(row)
        grouped[(int(row["layer"]), lambda_rel, g_max, k_max)].append(row)
    candidates = [
        _aggregate_needle_group(lambda_rel, g_max, k_max, rows, extra={"layer": layer})
        for (layer, lambda_rel, g_max, k_max), rows in grouped.items()
    ]
    best: dict[int, dict[str, Any]] = {}
    for row in candidates:
        layer = int(row["layer"])
        old = best.get(layer)
        if old is None or _needle_rank_key(row) < _needle_rank_key(old):
            best[layer] = row
    return [best[layer] for layer in sorted(best)]


def _needle_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str, str]:
    exact_lift = _metric(row, "token_exact_entry_lift")
    exact_rate = _metric(row, "needle_token_exact_entry_rate")
    exact_span_lift = _metric(row, "span_all_tokens_exact_entry_lift")
    lift = exact_lift if exact_lift is not None else _metric(row, "token_isolation_lift")
    rate = exact_rate if exact_rate is not None else _metric(row, "needle_token_isolated_rate")
    span_lift = (
        exact_span_lift
        if exact_span_lift is not None
        else _metric(row, "span_all_tokens_isolation_lift")
    )
    binding = _metric(row, "K_max_binding_rate")
    lambda_rel = _metric(row, "lambda_rel")
    return (
        -(lift if lift is not None else -math.inf),
        -(rate if rate is not None else -math.inf),
        -(span_lift if span_lift is not None else -math.inf),
        binding if binding is not None else 0.0,
        lambda_rel if lambda_rel is not None else math.inf,
        str(row.get("g_max")),
        str(row.get("k_max", "unclipped")),
    )


def build_needle_analysis(
    payload: dict[str, Any],
    *,
    layers: set[int] | None = None,
    groups: set[int] | None = None,
) -> dict[str, Any]:
    rows, source = _needle_config_rows(payload, layers=layers, groups=groups)
    ranked = sorted(rows, key=_needle_rank_key)
    best_layers = _needle_best_by_layer(payload, layers=layers, groups=groups)
    warnings: list[str] = []
    if not rows:
        warnings.append("no needle-isolation rows found for the selected scope")
    if (layers is not None or groups is not None) and not payload.get("by_layer_group"):
        warnings.append("selected scope requires by_layer_group rows, but the JSON has none")
    return {
        "kind": payload.get("kind"),
        "version": payload.get("version"),
        "config": payload.get("config", {}),
        "scope": {
            "layers": None if layers is None else sorted(layers),
            "groups": None if groups is None else sorted(groups),
        },
        "config_source": source,
        "warnings": warnings,
        "config_rankings": ranked,
        "best_by_layer": best_layers,
        "csv_rows": ranked,
    }


def _anchor_scheme_physical_width(row: dict[str, Any]) -> None:
    scheme = row.get("scheme")
    if row.get("current_scheme_physical_slot_count_mean") is not None:
        return
    if scheme in {SCHEME_SEMANTIC, SCHEME_SINGLE_CLUSTER}:
        row["current_scheme_physical_slot_count_mean"] = row.get("fixed3_anchor_count_mean")
    elif scheme in {SCHEME_VANILLA_COMPRESSED, SCHEME_VANILLA_FULL}:
        row["current_scheme_physical_slot_count_mean"] = row.get("entry_count_mean")
    else:
        row["current_scheme_physical_slot_count_mean"] = None


def _anchor_default_lambda_rel(payload: dict[str, Any]) -> float | None:
    config = payload.get("config") or {}
    return _finite_float(config.get("lambda_rel"))


def _anchor_row_with_default_lambda(row: dict[str, Any], default_lambda_rel: float | None) -> dict[str, Any]:
    normalized = dict(row)
    if "lambda_rel" not in normalized and default_lambda_rel is not None:
        normalized["lambda_rel"] = default_lambda_rel
    return normalized


def _normalize_anchor_config(row: dict[str, Any], *, collapse_l0: bool) -> dict[str, Any]:
    normalized = dict(row)
    if (
        collapse_l0
        and normalized.get("scheme") == SCHEME_SEMANTIC
        and normalized.get("l_block") is not None
        and int(normalized["l_block"]) == 0
    ):
        normalized["g_max"] = "inf"
    return normalized


def _dedupe_anchor_rows(rows: list[dict[str, Any]], *, collapse_l0: bool) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    duplicates = 0
    for row in rows:
        normalized = _normalize_anchor_config(row, collapse_l0=collapse_l0)
        key = (
            normalized.get("scheme"),
            normalized.get("lambda_rel"),
            normalized.get("g_max"),
            normalized.get("l_block"),
            normalized.get("k_max"),
            normalized.get("layer"),
            normalized.get("group"),
        )
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        out.append(normalized)
    return out, duplicates


def _anchor_aggregate_group(
    scheme: str,
    lambda_rel: float | None,
    g_max: str | None,
    l_block: int | None,
    k_max: str | None,
    rows: list[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_groups = sum(_metric(row, "sample_groups") or 0.0 for row in rows)
    mean_fields = [
        "entry_count_mean",
        "real_entry_count_mean",
        "pad_entry_count_mean",
        "token_count_mean",
        "logical_anchor_count_mean",
        "fixed3_anchor_count_mean",
        "fixed3_entry_anchor_count_mean",
        "fixed3_real_anchor_count_mean",
        "lo_hi_anchor_count_mean",
        "new_cluster_attempt_count_mean",
        "k_max_binding_count_mean",
        "ward_merge_count_mean",
        "novelty_suppressed_count_mean",
    ]
    sums = {
        field: sum((_metric(row, field) or 0.0) * (_metric(row, "sample_groups") or 0.0) for row in rows)
        for field in mean_fields
    }
    m_counts = {"1": 0, "2": 0, "3": 0}
    lo_hi_m_counts = {"1": 0, "2": 0}
    for row in rows:
        for key, value in (row.get("m_counts") or {}).items():
            m_counts[str(key)] = m_counts.get(str(key), 0) + int(value)
        for key, value in (row.get("lo_hi_m_counts") or {}).items():
            lo_hi_m_counts[str(key)] = lo_hi_m_counts.get(str(key), 0) + int(value)

    denom = sample_groups or 1.0
    logical = sums["logical_anchor_count_mean"]
    real_entries = sums["real_entry_count_mean"]
    fixed3 = sums["fixed3_anchor_count_mean"]
    fixed3_real = sums["fixed3_real_anchor_count_mean"]
    lo_hi = sums["lo_hi_anchor_count_mean"]
    row: dict[str, Any] = {
        "scheme": scheme,
        "lambda_rel": lambda_rel,
        "g_max": g_max,
        "l_block": l_block,
        "k_max": k_max,
        "sample_groups": int(sample_groups),
        "layer_group_count": len(rows),
    }
    for field in mean_fields:
        row[field] = sums[field] / denom
    row.update(
        {
            "E_M": _ratio(logical, real_entries),
            "E_M_lo_hi": _ratio(lo_hi, real_entries),
            "anchor_count_per_token": _ratio(logical, sums["token_count_mean"]),
            "fixed3_over_logical_ratio": _ratio(fixed3, logical),
            "fixed3_real_over_logical_ratio": _ratio(fixed3_real, logical),
            "logical_over_lo_hi_ratio": _ratio(logical, lo_hi),
            "gather_savings_fraction_vs_fixed3": None if fixed3 <= 0 else 1.0 - logical / fixed3,
            "gather_savings_fraction_vs_fixed3_real": None if fixed3_real <= 0 else 1.0 - logical / fixed3_real,
            "lo_hi_savings_fraction_vs_fixed3": None if fixed3 <= 0 else 1.0 - lo_hi / fixed3,
            "lo_hi_savings_fraction_vs_fixed3_real": None if fixed3_real <= 0 else 1.0 - lo_hi / fixed3_real,
            "K_max_binding_rate": _ratio(
                sums["k_max_binding_count_mean"],
                sums["new_cluster_attempt_count_mean"],
            ),
            "m_counts": {key: int(m_counts.get(key, 0)) for key in ("1", "2", "3")},
            "m_fractions": {key: _fraction_from_counts(m_counts, key, real_entries) for key in ("1", "2", "3")},
            "lo_hi_m_counts": {key: int(lo_hi_m_counts.get(key, 0)) for key in ("1", "2")},
            "entry_width_max": max([value for value in (_metric(source, "entry_width_max") for source in rows) if value is not None], default=None),
            "entry_span_max": max([value for value in (_metric(source, "entry_span_max") for source in rows) if value is not None], default=None),
        }
    )
    if len(rows) == 1:
        row["M_quantiles"] = rows[0].get("M_quantiles")
        row["entry_width_quantiles"] = rows[0].get("entry_width_quantiles")
        row["entry_span_quantiles"] = rows[0].get("entry_span_quantiles")
    else:
        row["M_quantiles"] = {"p50": None, "p90": None, "p99": None}
        row["entry_width_quantiles"] = {"p50": None, "p90": None, "p99": None}
        row["entry_span_quantiles"] = {"p50": None, "p90": None, "p99": None}
    _anchor_scheme_physical_width(row)
    if extra:
        row.update(extra)
    return row


def _anchor_add_ratios(rows: list[dict[str, Any]]) -> None:
    baselines = {row["scheme"]: row for row in rows if row.get("scheme") != SCHEME_SEMANTIC}
    metrics = [
        "entry_count_mean",
        "real_entry_count_mean",
        "logical_anchor_count_mean",
        "fixed3_anchor_count_mean",
        "fixed3_entry_anchor_count_mean",
        "fixed3_real_anchor_count_mean",
        "current_scheme_physical_slot_count_mean",
        "lo_hi_anchor_count_mean",
    ]
    suffixes = {
        "single_cluster": SCHEME_SINGLE_CLUSTER,
        "vanilla_compressed": SCHEME_VANILLA_COMPRESSED,
        "vanilla_full": SCHEME_VANILLA_FULL,
    }
    for row in rows:
        if row.get("scheme") != SCHEME_SEMANTIC:
            continue
        for suffix, baseline_scheme in suffixes.items():
            baseline = baselines.get(baseline_scheme)
            if baseline is None:
                continue
            for metric in metrics:
                row[f"{metric}_ratio_vs_{suffix}"] = _ratio(
                    _metric(row, metric),
                    _metric(baseline, metric),
                )


def _anchor_rows(
    payload: dict[str, Any],
    *,
    layers: set[int] | None,
    groups: set[int] | None,
    collapse_l0: bool,
) -> tuple[list[dict[str, Any]], str, int]:
    default_lambda_rel = _anchor_default_lambda_rel(payload)
    if layers is None and groups is None:
        rows = [_anchor_row_with_default_lambda(row, default_lambda_rel) for row in payload.get("overall_by_config", [])]
        for row in rows:
            _anchor_scheme_physical_width(row)
        rows, duplicates = _dedupe_anchor_rows(rows, collapse_l0=collapse_l0)
        return rows, "overall_by_config", duplicates

    deduped, duplicates = _dedupe_anchor_rows(
        [_anchor_row_with_default_lambda(row, default_lambda_rel) for row in payload.get("by_layer_group", [])],
        collapse_l0=collapse_l0,
    )
    scoped = _scope_layer_group_rows(deduped, layers=layers, groups=groups)
    grouped: dict[tuple[str, float | None, str | None, int | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in scoped:
        key = row["scheme"], _metric(row, "lambda_rel"), row.get("g_max"), row.get("l_block"), row.get("k_max")
        grouped[key].append(row)
    rows = [
        _anchor_aggregate_group(
            scheme,
            lambda_rel,
            g_max,
            None if l_block is None else int(l_block),
            k_max,
            group_rows,
        )
        for (scheme, lambda_rel, g_max, l_block, k_max), group_rows in sorted(
            grouped.items(),
            key=lambda item: (
                item[0][0],
                math.inf if item[0][1] is None else float(item[0][1]),
                "" if item[0][2] is None else item[0][2],
                -1 if item[0][3] is None else item[0][3],
                "" if item[0][4] is None else item[0][4],
            ),
        )
    ]
    _anchor_add_ratios(rows)
    return rows, "by_layer_group_count_aggregate", duplicates


def _anchor_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str, int, str]:
    e_m = _metric(row, "E_M")
    savings = _metric(row, "gather_savings_fraction_vs_fixed3")
    phys_ratio = _metric(row, "current_scheme_physical_slot_count_mean_ratio_vs_vanilla_full")
    binding = _metric(row, "K_max_binding_rate")
    lambda_rel = _metric(row, "lambda_rel")
    return (
        e_m if e_m is not None else math.inf,
        -(savings if savings is not None else -math.inf),
        phys_ratio if phys_ratio is not None else math.inf,
        binding if binding is not None else 0.0,
        lambda_rel if lambda_rel is not None else math.inf,
        str(row.get("g_max")),
        int(row.get("l_block") or -1),
        str(row.get("k_max", "unclipped")),
    )


def _anchor_best_by_layer(
    payload: dict[str, Any],
    *,
    layers: set[int] | None,
    groups: set[int] | None,
    collapse_l0: bool,
) -> list[dict[str, Any]]:
    default_lambda_rel = _anchor_default_lambda_rel(payload)
    deduped, _duplicates = _dedupe_anchor_rows(
        [_anchor_row_with_default_lambda(row, default_lambda_rel) for row in payload.get("by_layer_group", [])],
        collapse_l0=collapse_l0,
    )
    scoped = _scope_layer_group_rows(deduped, layers=layers, groups=groups)
    grouped: dict[tuple[int, float | None, str, int, str | None], list[dict[str, Any]]] = defaultdict(list)
    baseline_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scoped:
        if row.get("scheme") == SCHEME_SEMANTIC:
            grouped[
                (
                    int(row["layer"]),
                    _metric(row, "lambda_rel"),
                    str(row.get("g_max")),
                    int(row.get("l_block")),
                    row.get("k_max"),
                )
            ].append(row)
        else:
            baseline_groups[(int(row["layer"]), str(row["scheme"]))].append(row)

    # Each baseline scheme has one raw row per (layer, group) in the source
    # data (unaffected by g_max/l_block/k_max), so aggregate them per layer
    # with the same sample_groups-weighted logic used for semantic rows --
    # not just "keep whichever group's row is seen last" -- before handing
    # them to _anchor_add_ratios below.
    baselines_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (layer, scheme), base_rows in baseline_groups.items():
        baselines_by_layer[layer].append(
            _anchor_aggregate_group(scheme, None, None, None, None, base_rows, extra={"layer": layer})
        )

    candidates = [
        _anchor_aggregate_group(SCHEME_SEMANTIC, lambda_rel, g_max, l_block, k_max, rows, extra={"layer": layer})
        for (layer, lambda_rel, g_max, l_block, k_max), rows in grouped.items()
    ]
    # _anchor_rank_key reads current_scheme_physical_slot_count_mean_ratio_
    # vs_vanilla_full as a tie-break, but that field is only ever populated by
    # _anchor_add_ratios -- which nothing called on these candidates before,
    # silently making the tie-break inert (always None -> math.inf for every
    # candidate, so it never actually discriminates between tied E[M]/savings
    # rows, e.g. the l_block=0 collapse group). _anchor_add_ratios keys its
    # baseline lookup purely by scheme, assuming one row per scheme, so it
    # must be called per layer (each layer's own baseline aggregate) rather
    # than on one flat list, or a later layer's baseline would silently
    # overwrite an earlier layer's in the lookup dict.
    for candidate in candidates:
        layer_baselines = baselines_by_layer.get(int(candidate["layer"]), [])
        if layer_baselines:
            _anchor_add_ratios([*layer_baselines, candidate])

    best: dict[int, dict[str, Any]] = {}
    for row in candidates:
        layer = int(row["layer"])
        old = best.get(layer)
        if old is None or _anchor_rank_key(row) < _anchor_rank_key(old):
            best[layer] = row
    return [best[layer] for layer in sorted(best)]


def build_anchor_analysis(
    payload: dict[str, Any],
    *,
    layers: set[int] | None = None,
    groups: set[int] | None = None,
    collapse_l0: bool = True,
) -> dict[str, Any]:
    rows, source, duplicates = _anchor_rows(
        payload,
        layers=layers,
        groups=groups,
        collapse_l0=collapse_l0,
    )
    baseline_rows = [row for row in rows if row.get("scheme") != SCHEME_SEMANTIC]
    semantic_rows = [row for row in rows if row.get("scheme") == SCHEME_SEMANTIC]
    _anchor_add_ratios(rows)
    ranked = sorted(semantic_rows, key=_anchor_rank_key)
    best_layers = _anchor_best_by_layer(
        payload,
        layers=layers,
        groups=groups,
        collapse_l0=collapse_l0,
    )
    warnings: list[str] = []
    if not rows:
        warnings.append("no anchor-dedup rows found for the selected scope")
    if (layers is not None or groups is not None) and not payload.get("by_layer_group"):
        warnings.append("selected scope requires by_layer_group rows, but the JSON has none")
    if duplicates:
        warnings.append(f"collapsed {duplicates} duplicate semantic l_block=0 anchor rows")
    return {
        "kind": payload.get("kind"),
        "version": payload.get("version"),
        "config": payload.get("config", {}),
        "scope": {
            "layers": None if layers is None else sorted(layers),
            "groups": None if groups is None else sorted(groups),
            "collapse_l0": collapse_l0,
        },
        "config_source": source,
        "warnings": warnings,
        "baseline_rows": baseline_rows,
        "config_rankings": ranked,
        "best_by_layer": best_layers,
        "csv_rows": ranked,
    }


def _fmt_num(value: Any, digits: int = 4) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e5:
        return f"{number:.3e}"
    return f"{number:.{digits}g}"


def _fmt_ratio(value: Any) -> str:
    number = _finite_float(value)
    return "n/a" if number is None else f"{number:.3g}x"


def _fmt_pct(value: Any) -> str:
    number = _finite_float(value)
    return "n/a" if number is None else f"{number:.1%}"


def _fmt_int(value: Any) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return str(int(number))


def _fmt_win(row: dict[str, Any], prefix: str) -> str:
    wins = row.get(f"{prefix}_wins")
    valid = row.get(f"{prefix}_valid")
    if not isinstance(wins, int) or not isinstance(valid, int) or valid <= 0:
        return "n/a"
    return f"{wins}/{valid}"


def _fmt_config(row: dict[str, Any]) -> str:
    return f"{row.get('g_max')}:{row.get('l_block')}"


def _table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str | Callable[[dict[str, Any]], Any], Callable[[Any], str]]],
    *,
    limit: int | None = None,
) -> str:
    clipped = rows if limit is None else rows[:limit]
    if not clipped:
        return "n/a"
    rendered: list[list[str]] = []
    for row in clipped:
        rendered_row: list[str] = []
        for _header, getter, formatter in columns:
            value = getter(row) if callable(getter) else _metric(row, getter)
            if value is None and isinstance(getter, str) and "." not in getter:
                value = row.get(getter)
            rendered_row.append(formatter(value))
        rendered.append(rendered_row)
    headers = [header for header, _getter, _formatter in columns]
    widths = [
        max(len(headers[i]), *(len(rendered_row[i]) for rendered_row in rendered))
        for i in range(len(headers))
    ]
    lines = ["  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))]
    lines.append("  ".join("-" * width for width in widths))
    for rendered_row in rendered:
        lines.append("  ".join(rendered_row[i].ljust(widths[i]) for i in range(len(widths))))
    return "\n".join(lines)


def _ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row, rank=i) for i, row in enumerate(rows, start=1)]


def _print_report(
    path: Path,
    analysis: dict[str, Any],
    *,
    top: int,
    by_layer: bool,
    worst_rows: list[dict[str, Any]],
    inspected_config: tuple[str, int] | None,
    key_metric: str,
    value_metric: str,
) -> None:
    config = analysis.get("config", {})
    baseline = analysis.get("baseline", {})
    print("== Semantic S0 sweep summary ==")
    print(f"file: {path}")
    print(f"kind/version: {analysis.get('kind')}/{analysis.get('version')}")
    print(
        "scope: "
        f"layers={_format_layers(None if analysis['scope']['layers'] is None else set(analysis['scope']['layers']))} "
        f"groups={_format_layers(None if analysis['scope']['groups'] is None else set(analysis['scope']['groups']))} "
        f"source={analysis.get('config_source')}"
    )
    print(
        "sweep: "
        f"g_max={config.get('g_max')} l_block={config.get('l_block')} "
        f"lambda_rel={config.get('lambda_rel')} b_prime={config.get('b_prime')}"
    )
    if baseline.get("field"):
        print(f"baseline: {baseline.get('field')} ({baseline.get('row_count')} rows)")
    else:
        print("baseline: none")
    for warning in analysis.get("warnings", []):
        print(f"warning: {warning}")

    print(f"\n== Top configs by {key_metric} ==")
    print(
        _table(
            _ranked(analysis.get("config_rankings", [])),
            [
                ("#", "rank", _fmt_int),
                ("cfg", _fmt_config, str),
                ("key_rel", key_metric, _fmt_num),
                ("value_rel", value_metric, _fmt_num),
                ("entries", "entry_count_mean", _fmt_num),
                ("pad", "pad_entry_count_mean", _fmt_num),
                ("clusters", "cluster_count_mean", _fmt_num),
                ("segments", "segment_count_mean", _fmt_num),
                ("span_p99", SPAN_P99_METRIC, _fmt_num),
                ("span_max", "entry_span_global_max", _fmt_num),
            ],
            limit=top,
        )
    )

    comparison = analysis.get("baseline_comparison", [])
    if comparison:
        print("\n== Ratios vs baseline, lower is better ==")
        print(
            _table(
                _ranked(comparison),
                [
                    ("#", "rank", _fmt_int),
                    ("cfg", _fmt_config, str),
                    ("score", "score", _fmt_ratio),
                    ("key_med", "key_ratio_median", _fmt_ratio),
                    ("key_p90", "key_ratio_p90", _fmt_ratio),
                    ("key_win", lambda row: _fmt_win(row, "key"), str),
                    ("val_med", "value_ratio_median", _fmt_ratio),
                    ("val_win", lambda row: _fmt_win(row, "value"), str),
                    ("span_p99", "span_p99_ratio_median", _fmt_ratio),
                    ("entries", "entry_count_ratio_median", _fmt_ratio),
                    ("n", "layer_group_count", _fmt_int),
                ],
                limit=top,
            )
        )

        print("\n== Best config per l_block ==")
        print(
            _table(
                _ranked(analysis.get("best_by_l_block", [])),
                [
                    ("#", "rank", _fmt_int),
                    ("cfg", _fmt_config, str),
                    ("score", "score", _fmt_ratio),
                    ("key_med", "key_ratio_median", _fmt_ratio),
                    ("val_med", "value_ratio_median", _fmt_ratio),
                    ("span_p99", "span_p99_ratio_median", _fmt_ratio),
                    ("entries", "entry_count_ratio_median", _fmt_ratio),
                ],
            )
        )

    if by_layer and analysis.get("best_by_layer"):
        print("\n== Best config per layer ==")
        print(
            _table(
                analysis["best_by_layer"],
                [
                    ("layer", "layer", _fmt_int),
                    ("cfg", _fmt_config, str),
                    ("score", "score", _fmt_ratio),
                    ("key_med", "key_ratio_median", _fmt_ratio),
                    ("key_win", lambda row: _fmt_win(row, "key"), str),
                    ("val_med", "value_ratio_median", _fmt_ratio),
                    ("groups", "group_count", _fmt_int),
                ],
            )
        )

    if worst_rows and inspected_config is not None:
        print(f"\n== Worst layer/groups for {inspected_config[0]}:{inspected_config[1]} by key ratio ==")
        print(
            _table(
                worst_rows,
                [
                    ("layer", "layer", _fmt_int),
                    ("group", "group", _fmt_int),
                    ("key_ratio", "key_ratio", _fmt_ratio),
                    ("value_ratio", "value_ratio", _fmt_ratio),
                    ("span_p99", "span_p99_ratio", _fmt_ratio),
                    ("entries", "entry_count_ratio", _fmt_ratio),
                    ("key_rel", "key_rel", _fmt_num),
                    ("value_rel", "value_rel", _fmt_num),
                ],
            )
        )


def _print_needle_report(path: Path, analysis: dict[str, Any], *, top: int, by_layer: bool) -> None:
    config = analysis.get("config", {})
    print("== Semantic S0.3 needle isolation summary ==")
    print(f"file: {path}")
    print(f"kind/version: {analysis.get('kind')}/{analysis.get('version')}")
    print(
        "scope: "
        f"layers={_format_layers(None if analysis['scope']['layers'] is None else set(analysis['scope']['layers']))} "
        f"groups={_format_layers(None if analysis['scope']['groups'] is None else set(analysis['scope']['groups']))} "
        f"source={analysis.get('config_source')}"
    )
    print(
        "config: "
        f"g_max={config.get('g_max')} "
        f"lambda_rel={config.get('lambda_rel_values', config.get('lambda_rel'))} "
        f"b_prime={config.get('b_prime')} random_trials={config.get('random_trials')}"
    )
    for warning in analysis.get("warnings", []):
        print(f"warning: {warning}")

    print("\n== Top lambda_rel/g_max/k_max by exact-entry needle isolation lift, higher is better ==")
    print(
        _table(
            _ranked(analysis.get("config_rankings", [])),
            [
                ("#", "rank", _fmt_int),
                ("lambda", "lambda_rel", _fmt_num),
                ("g_max", "g_max", str),
                ("k_max", "k_max", str),
                ("exact", "needle_token_exact_entry_rate", _fmt_pct),
                ("random", "random_token_exact_entry_rate", _fmt_pct),
                ("lift", "token_exact_entry_lift", _fmt_ratio),
                ("small", "needle_token_isolated_rate", _fmt_pct),
                ("small_lift", "token_isolation_lift", _fmt_ratio),
                ("span_exact", "span_all_tokens_exact_entry_rate", _fmt_pct),
                ("span_lift", "span_all_tokens_exact_entry_lift", _fmt_ratio),
                ("cluster_p50", "cluster_size_quantiles.p50", _fmt_num),
                ("cluster_p90", "cluster_size_quantiles.p90", _fmt_num),
                ("clusters", "cluster_count_mean", _fmt_num),
                ("segments", "segment_count_mean", _fmt_num),
                ("n", "sample_groups", _fmt_int),
            ],
            limit=top,
        )
    )

    if by_layer and analysis.get("best_by_layer"):
        print("\n== Best g_max per layer ==")
        print(
            _table(
                analysis["best_by_layer"],
                [
                    ("layer", "layer", _fmt_int),
                    ("lambda", "lambda_rel", _fmt_num),
                    ("g_max", "g_max", str),
                    ("k_max", "k_max", str),
                    ("exact", "needle_token_exact_entry_rate", _fmt_pct),
                    ("random", "random_token_exact_entry_rate", _fmt_pct),
                    ("lift", "token_exact_entry_lift", _fmt_ratio),
                    ("small", "needle_token_isolated_rate", _fmt_pct),
                    ("small_lift", "token_isolation_lift", _fmt_ratio),
                    ("span_lift", "span_all_tokens_exact_entry_lift", _fmt_ratio),
                    ("groups", "layer_group_count", _fmt_int),
                    ("n", "sample_groups", _fmt_int),
                ],
            )
        )


def _print_anchor_report(path: Path, analysis: dict[str, Any], *, top: int, by_layer: bool) -> None:
    config = analysis.get("config", {})
    print("== Semantic S0.6 anchor dedup summary ==")
    print(f"file: {path}")
    print(f"kind/version: {analysis.get('kind')}/{analysis.get('version')}")
    print(
        "scope: "
        f"layers={_format_layers(None if analysis['scope']['layers'] is None else set(analysis['scope']['layers']))} "
        f"groups={_format_layers(None if analysis['scope']['groups'] is None else set(analysis['scope']['groups']))} "
        f"source={analysis.get('config_source')}"
    )
    print(
        "config: "
        f"g_max={config.get('g_max')} l_block={config.get('l_block')} "
        f"lambda_rel={config.get('lambda_rel')} b_prime={config.get('b_prime')}"
    )
    print("note: lower E[M] means more anchor dedup potential; current v1 physical width is the phys column.")
    for warning in analysis.get("warnings", []):
        print(f"warning: {warning}")

    baseline_rows = analysis.get("baseline_rows", [])
    if baseline_rows:
        print("\n== Baselines ==")
        print(
            _table(
                baseline_rows,
                [
                    ("scheme", "scheme", str),
                    ("E[M]", "E_M", _fmt_num),
                    ("M=3", "m_fractions.3", _fmt_pct),
                    ("logical", "logical_anchor_count_mean", _fmt_num),
                    ("fixed3", "fixed3_anchor_count_mean", _fmt_num),
                    ("phys", "current_scheme_physical_slot_count_mean", _fmt_num),
                    ("entries", "entry_count_mean", _fmt_num),
                    ("n", "sample_groups", _fmt_int),
                ],
            )
        )

    print("\n== Semantic configs by E[M], lower is better for gather potential ==")
    print(
        _table(
            _ranked(analysis.get("config_rankings", [])),
            [
                ("#", "rank", _fmt_int),
                ("cfg", _fmt_config, str),
                ("E[M]", "E_M", _fmt_num),
                ("M=3", "m_fractions.3", _fmt_pct),
                ("save", "gather_savings_fraction_vs_fixed3", _fmt_pct),
                ("logical", "logical_anchor_count_mean", _fmt_num),
                ("fixed3", "fixed3_anchor_count_mean", _fmt_num),
                ("phys", "current_scheme_physical_slot_count_mean", _fmt_num),
                ("entries", "entry_count_mean", _fmt_num),
                ("entry/single", "entry_count_mean_ratio_vs_single_cluster", _fmt_ratio),
                ("logical/single", "logical_anchor_count_mean_ratio_vs_single_cluster", _fmt_ratio),
                ("phys/vanilla", "current_scheme_physical_slot_count_mean_ratio_vs_vanilla_full", _fmt_ratio),
            ],
            limit=top,
        )
    )

    if by_layer and analysis.get("best_by_layer"):
        print("\n== Best anchor-dedup config per layer ==")
        print(
            _table(
                analysis["best_by_layer"],
                [
                    ("layer", "layer", _fmt_int),
                    ("cfg", _fmt_config, str),
                    ("E[M]", "E_M", _fmt_num),
                    ("M=3", "m_fractions.3", _fmt_pct),
                    ("save", "gather_savings_fraction_vs_fixed3", _fmt_pct),
                    ("logical", "logical_anchor_count_mean", _fmt_num),
                    ("phys", "current_scheme_physical_slot_count_mean", _fmt_num),
                    ("groups", "layer_group_count", _fmt_int),
                ],
            )
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "g_max",
        "l_block",
        "score",
        "key_ratio_median",
        "key_ratio_mean",
        "key_ratio_p90",
        "key_ratio_max",
        "key_wins",
        "key_valid",
        "value_ratio_median",
        "value_ratio_mean",
        "value_ratio_p90",
        "value_ratio_max",
        "value_wins",
        "value_valid",
        "span_p99_ratio_median",
        "span_p99_ratio_p90",
        "entry_count_ratio_median",
        "entry_count_ratio_p90",
        "layer_group_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_csv_by_layer(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``best_by_layer`` (one row per layer, its winning config) to CSV.

    ``--csv`` writes ``baseline_comparison`` -- the (g_max, l_block) table
    aggregated *across* every layer/group -- regardless of whether ``--by_layer``
    was passed; ``--by_layer`` only ever controlled an extra section of the
    *stdout* report (``_print_report``'s ``if by_layer and analysis.get(
    "best_by_layer")`` branch). So there was previously no way to persist the
    per-layer breakdown at all: a caller who ran ``--by_layer --csv out.csv``
    got a CSV byte-identical to one without ``--by_layer``, silently. This
    writer, and the paired ``--csv_by_layer`` flag, close that gap. Note
    ``analysis["best_by_layer"]`` is computed unconditionally in
    ``build_analysis`` whenever a baseline resolves (not gated by the
    ``--by_layer`` CLI flag, which only affects printing), so ``--csv_by_layer``
    does not require ``--by_layer`` to also be passed.
    """
    fieldnames = [
        "layer",
        "g_max",
        "l_block",
        "score",
        "key_ratio_median",
        "value_ratio_median",
        "key_wins",
        "key_valid",
        "group_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _csv_cell(row: dict[str, Any], field: str) -> Any:
    if field in row:
        value = row.get(field)
    else:
        value = _metric(row, field)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_cell(row, name) for name in fieldnames})


def _write_needle_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_rows_csv(path, _ranked(rows), NEEDLE_CSV_FIELDS)


def _write_needle_csv_by_layer(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_rows_csv(path, rows, NEEDLE_LAYER_CSV_FIELDS)


def _write_anchor_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_rows_csv(path, _ranked(rows), ANCHOR_CSV_FIELDS)


def _write_anchor_csv_by_layer(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_rows_csv(path, rows, ANCHOR_LAYER_CSV_FIELDS)


def _write_summary_json(path: Path, analysis: dict[str, Any], *, top: int) -> None:
    compact = dict(analysis)
    compact["config_rankings"] = compact.get("config_rankings", [])[:top]
    compact["baseline_comparison"] = compact.get("baseline_comparison", [])[:top]
    compact["csv_rows"] = compact.get("csv_rows", [])[:top]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage0_json",
        help="Path or glob for SemanticLogKV S0 analysis JSON; .gz is supported",
    )
    parser.add_argument("--top", type=int, default=12, help="Rows to print in each ranked table")
    parser.add_argument("--layers", help="Optional layer filter, e.g. '23-26' or '0,7,14'")
    parser.add_argument("--groups", help="Optional KV-group filter, e.g. '0-3'")
    parser.add_argument(
        "--baseline",
        choices=("auto", "none", "vanilla", "vanilla_full", "single_cluster", "position"),
        default="auto",
        help="S0.0 only: baseline for ratios. auto prefers the real vanilla compressed-prefix baseline.",
    )
    parser.add_argument(
        "--keep_duplicate_l0",
        action="store_true",
        help="Do not collapse l_block=0 rows whose effective g_max is always inf.",
    )
    parser.add_argument("--key_metric", default=PRIMARY_KEY_METRIC)
    parser.add_argument("--value_metric", default=PRIMARY_VALUE_METRIC)
    parser.add_argument("--by_layer", action="store_true", help="Also print the best config per layer")
    parser.add_argument(
        "--inspect_config",
        help="Config to drill into for worst layer/groups, formatted as 'g_max:l_block'. Defaults to best ratio row.",
    )
    parser.add_argument("--worst", type=int, default=8, help="Worst layer/groups to print for --inspect_config")
    parser.add_argument("--csv", type=Path, help="Write the compact ranked table to CSV")
    parser.add_argument(
        "--csv_by_layer",
        type=Path,
        help="Write the per-layer best-config table to CSV. Independent of --by_layer, which only "
        "controls whether this table is also printed to stdout.",
    )
    parser.add_argument("--summary_json", type=Path, help="Write a compact top-N JSON summary")
    args = parser.parse_args()

    layers = _parse_int_spec(args.layers)
    groups = _parse_int_spec(args.groups)
    path, payload = _load_json(args.stage0_json)
    top = max(args.top, 1)
    kind = payload.get("kind")

    if kind == KIND_NEEDLE:
        analysis = build_needle_analysis(payload, layers=layers, groups=groups)
        _print_needle_report(path, analysis, top=top, by_layer=args.by_layer)
        if args.csv:
            _write_needle_csv(args.csv, analysis.get("csv_rows", []))
            print(f"\nwrote CSV: {args.csv}")
        if args.csv_by_layer:
            best_layers = analysis.get("best_by_layer", [])
            if not best_layers:
                raise SystemExit("--csv_by_layer needs by_layer_group rows in the S0.3 JSON")
            _write_needle_csv_by_layer(args.csv_by_layer, best_layers)
            print(f"wrote per-layer CSV: {args.csv_by_layer}")
        if args.summary_json:
            _write_summary_json(args.summary_json, analysis, top=top)
            print(f"wrote compact JSON: {args.summary_json}")
        return

    if kind == KIND_ANCHOR:
        analysis = build_anchor_analysis(
            payload,
            layers=layers,
            groups=groups,
            collapse_l0=not args.keep_duplicate_l0,
        )
        _print_anchor_report(path, analysis, top=top, by_layer=args.by_layer)
        if args.csv:
            _write_anchor_csv(args.csv, analysis.get("csv_rows", []))
            print(f"\nwrote CSV: {args.csv}")
        if args.csv_by_layer:
            best_layers = analysis.get("best_by_layer", [])
            if not best_layers:
                raise SystemExit("--csv_by_layer needs by_layer_group rows in the S0.6 JSON")
            _write_anchor_csv_by_layer(args.csv_by_layer, best_layers)
            print(f"wrote per-layer CSV: {args.csv_by_layer}")
        if args.summary_json:
            _write_summary_json(args.summary_json, analysis, top=top)
            print(f"wrote compact JSON: {args.summary_json}")
        return

    if kind not in (None, KIND_SWEEP):
        raise SystemExit(f"unsupported SemanticLogKV S0 analysis kind: {kind!r}")

    analysis = build_analysis(
        payload,
        layers=layers,
        groups=groups,
        baseline=args.baseline,
        collapse_l0=not args.keep_duplicate_l0,
        key_metric=args.key_metric,
        value_metric=args.value_metric,
    )

    comparison = analysis.get("baseline_comparison", [])
    baseline_field = analysis.get("baseline", {}).get("field")
    inspected_config: tuple[str, int] | None = None
    worst_rows: list[dict[str, Any]] = []
    if baseline_field and args.worst > 0:
        if args.inspect_config:
            inspected_config = _parse_config_selector(args.inspect_config)
            if not args.keep_duplicate_l0 and inspected_config[1] == 0:
                inspected_config = ("inf", 0)
        elif comparison:
            best = comparison[0]
            inspected_config = str(best["g_max"]), int(best["l_block"])
        if inspected_config is not None:
            worst_rows = _worst_layer_groups(
                payload.get("by_layer_group", []),
                payload.get(baseline_field, []),
                config=inspected_config,
                layers=layers,
                groups=groups,
                collapse_l0=not args.keep_duplicate_l0,
                key_metric=args.key_metric,
                value_metric=args.value_metric,
                limit=args.worst,
            )

    _print_report(
        path,
        analysis,
        top=top,
        by_layer=args.by_layer,
        worst_rows=worst_rows,
        inspected_config=inspected_config,
        key_metric=args.key_metric,
        value_metric=args.value_metric,
    )

    if args.csv:
        if not comparison:
            raise SystemExit("--csv needs a baseline comparison; pass --baseline other than 'none'")
        _write_csv(args.csv, comparison)
        print(f"\nwrote CSV: {args.csv}")
    if args.csv_by_layer:
        best_layers = analysis.get("best_by_layer", [])
        if not best_layers:
            raise SystemExit("--csv_by_layer needs a baseline comparison; pass --baseline other than 'none'")
        _write_csv_by_layer(args.csv_by_layer, best_layers)
        print(f"wrote per-layer CSV: {args.csv_by_layer}")
    if args.summary_json:
        _write_summary_json(args.summary_json, analysis, top=top)
        print(f"wrote compact JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
