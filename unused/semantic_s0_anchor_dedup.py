#!/usr/bin/env python
"""Run the SemanticLogKV S0.6 anchor-dedup ``E[M]`` analysis on a Stage-0 dump.

S0.6 measures how many distinct anchors each real entry has after deduplicating
``[p_lo, p_mid, p_hi]``. It is a future gather/packed-readout signal: current
v1 still pays a fixed three-anchor physical readout width, while ``E[M]`` says
how much of that width is logically useful after deduplication.

Example:
    python unused/semantic_s0_anchor_dedup.py \
      --dump stage0_dump \
      --output stage0_dump/s0_6_anchor_dedup.json \
      --g_max inf,8192,4096,2048,1024,256 \
      --l_block 0,1,2,3 \
      --b_prime 8
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_S0_PATH = REPO_ROOT / "litgpt" / "semantic_s0.py"
_S0_SPEC = importlib.util.spec_from_file_location("semantic_s0_local", _S0_PATH)
if _S0_SPEC is None or _S0_SPEC.loader is None:
    raise RuntimeError(f"could not load {_S0_PATH}")
_S0 = importlib.util.module_from_spec(_S0_SPEC)
sys.modules[_S0_SPEC.name] = _S0
_S0_SPEC.loader.exec_module(_S0)

OfflineEntry = _S0.OfflineEntry
format_g_max = _S0.format_g_max
load_manifest = _S0.load_manifest
manifest_base_dir = _S0.manifest_base_dir
parse_g_max_list = _S0.parse_g_max_list
parse_int_list = _S0.parse_int_list
route_dpmeans_segments = _S0.route_dpmeans_segments
route_single_cluster_bprime_ladder = _S0.route_single_cluster_bprime_ladder
simulate_segment_ladders = _S0.simulate_segment_ladders
vanilla_logkv_compressed_entries = _S0.vanilla_logkv_compressed_entries
vanilla_logkv_full_cache_entries = _S0.vanilla_logkv_full_cache_entries


SCHEME_SEMANTIC = "semantic"
SCHEME_SINGLE_CLUSTER = "single_cluster_bprime_baseline"
SCHEME_VANILLA_COMPRESSED = "vanilla_logkv_compressed_prefix_baseline"
SCHEME_VANILLA_FULL = "vanilla_logkv_full_cache_baseline"

RATIO_BASELINES = {
    "single_cluster": SCHEME_SINGLE_CLUSTER,
    "vanilla_compressed": SCHEME_VANILLA_COMPRESSED,
    "vanilla_full": SCHEME_VANILLA_FULL,
}


def _wanted(values: str | None) -> set[int] | None:
    return None if values is None else set(parse_int_list(values))


def _fallback_scale(x: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float32)
    mean = xx.mean(axis=0, keepdims=True)
    return float(np.square(xx - mean, dtype=np.float32).sum(axis=1).mean(dtype=np.float64))


def _manifest_sh(
    manifest: dict[str, Any],
    layer: int,
    group: int,
    k: np.ndarray,
    *,
    allow_fallback: bool,
) -> tuple[float, str]:
    reason: str | None = None
    value: float | None = None
    try:
        value = float(manifest["key_scale"][str(layer)]["s_h"][group])
    except Exception as exc:
        reason = f"manifest['key_scale'][{str(layer)!r}]['s_h'][{group}] missing or malformed: {exc!r}"
    else:
        if not (math.isfinite(value) and value > 0):
            reason = f"manifest['key_scale'][{str(layer)!r}]['s_h'][{group}] = {value!r} is invalid"
            value = None

    if value is not None:
        return value, "calibrated"
    if not allow_fallback:
        raise ValueError(
            f"{reason}. layer={layer} group={group} has no usable calibrated key_scale. "
            f"Pass --allow_fallback_sh to substitute this one prompt's local variance instead "
            f"(noisier and not comparable across layers/prompts)."
        )
    return max(_fallback_scale(k), 1e-12), "fallback_local"


def _effective_g_max(g_max: float, l_block: int) -> float:
    # Keep the same S0.0 convention: the l_block=0 column is the true
    # pure-semantic endpoint, regardless of the nominal swept g_max.
    return math.inf if int(l_block) == 0 else g_max


def _iter_records(
    manifest: dict[str, Any],
    *,
    layer_filter: set[int] | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sample in manifest.get("samples", []):
        for record in sample.get("layers", []):
            layer = int(record["layer"])
            if layer_filter is not None and layer not in layer_filter:
                continue
            out.append((sample, record))
    return out


def _mid_anchor(members: list[int]) -> int:
    if not members:
        raise ValueError("_mid_anchor requires at least one real member")
    p_lo = min(int(p) for p in members)
    p_hi = max(int(p) for p in members)
    w = len(members)
    sum_wp = sum(int(p) for p in members)
    p_mid = (2 * sum_wp + w) // (2 * w)
    return min(max(int(p_mid), p_lo), p_hi)


def _entry_anchor_stats(entry: Any) -> dict[str, Any] | None:
    members = [int(p) for p in entry.members]
    if not members:
        return None
    p_lo = min(members)
    p_hi = max(members)
    p_mid = _mid_anchor(members)
    anchors = sorted({p_lo, p_mid, p_hi})
    lo_hi_anchors = sorted({p_lo, p_hi})
    return {
        "w": len(members),
        "span": p_hi - p_lo,
        "p_lo": p_lo,
        "p_mid": p_mid,
        "p_hi": p_hi,
        "m": len(anchors),
        "lo_hi_m": len(lo_hi_anchors),
        "mid_is_new": p_mid not in {p_lo, p_hi},
    }


def _quantiles(values: list[float], qs: tuple[float, ...] = (0.5, 0.9, 0.99)) -> dict[str, float | None]:
    if not values:
        return {f"p{int(q * 100):02d}": None for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


def _rate(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def _ratio(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None or float(den) <= 0:
        return None
    return float(num) / float(den)


class AnchorDedupAccumulator:
    def __init__(self) -> None:
        self.sample_groups = 0
        self.entry_count_sum = 0.0
        self.real_entry_count_sum = 0.0
        self.pad_entry_count_sum = 0.0
        self.token_count_sum = 0.0
        self.logical_anchor_count_sum = 0.0
        self.fixed3_entry_anchor_count_sum = 0.0
        self.fixed3_real_anchor_count_sum = 0.0
        self.lo_hi_anchor_count_sum = 0.0
        self.m_counts = {1: 0, 2: 0, 3: 0}
        self.lo_hi_m_counts = {1: 0, 2: 0}
        self.mid_new_count = 0
        self.widths: list[float] = []
        self.spans: list[float] = []
        self.m_values: list[float] = []
        self.lo_hi_m_values: list[float] = []
        self.sh_sources: set[str] = set()

    def add(self, *, entries: list[Any], ladder_meta: dict[str, Any], sh_source: str = "n/a") -> None:
        self.sample_groups += 1
        entry_count = float(ladder_meta.get("entry_count", len(entries)))
        self.entry_count_sum += entry_count
        # Current v1 materializes a fixed three-anchor axis per ladder entry.
        # E[M] below intentionally skips pad/dead entries, so keep both widths.
        self.fixed3_entry_anchor_count_sum += 3.0 * entry_count
        self.pad_entry_count_sum += float(ladder_meta.get("pad_entry_count", 0))
        self.sh_sources.add(sh_source)
        for entry in entries:
            stats = _entry_anchor_stats(entry)
            if stats is None:
                continue
            m = int(stats["m"])
            lo_hi_m = int(stats["lo_hi_m"])
            width = int(stats["w"])
            self.real_entry_count_sum += 1
            self.token_count_sum += width
            self.logical_anchor_count_sum += m
            self.fixed3_real_anchor_count_sum += 3
            self.lo_hi_anchor_count_sum += lo_hi_m
            self.m_counts[m] = self.m_counts.get(m, 0) + 1
            self.lo_hi_m_counts[lo_hi_m] = self.lo_hi_m_counts.get(lo_hi_m, 0) + 1
            self.mid_new_count += int(bool(stats["mid_is_new"]))
            self.widths.append(float(width))
            self.spans.append(float(stats["span"]))
            self.m_values.append(float(m))
            self.lo_hi_m_values.append(float(lo_hi_m))

    def merge(self, other: "AnchorDedupAccumulator") -> None:
        self.sample_groups += other.sample_groups
        self.entry_count_sum += other.entry_count_sum
        self.real_entry_count_sum += other.real_entry_count_sum
        self.pad_entry_count_sum += other.pad_entry_count_sum
        self.token_count_sum += other.token_count_sum
        self.logical_anchor_count_sum += other.logical_anchor_count_sum
        self.fixed3_entry_anchor_count_sum += other.fixed3_entry_anchor_count_sum
        self.fixed3_real_anchor_count_sum += other.fixed3_real_anchor_count_sum
        self.lo_hi_anchor_count_sum += other.lo_hi_anchor_count_sum
        for key, count in other.m_counts.items():
            self.m_counts[key] = self.m_counts.get(key, 0) + count
        for key, count in other.lo_hi_m_counts.items():
            self.lo_hi_m_counts[key] = self.lo_hi_m_counts.get(key, 0) + count
        self.mid_new_count += other.mid_new_count
        self.widths.extend(other.widths)
        self.spans.extend(other.spans)
        self.m_values.extend(other.m_values)
        self.lo_hi_m_values.extend(other.lo_hi_m_values)
        self.sh_sources.update(other.sh_sources)

    def finalize(self) -> dict[str, Any]:
        denom = max(self.sample_groups, 1)
        real_entries = self.real_entry_count_sum
        logical = self.logical_anchor_count_sum
        fixed3_entry = self.fixed3_entry_anchor_count_sum
        fixed3_real = self.fixed3_real_anchor_count_sum
        lo_hi = self.lo_hi_anchor_count_sum
        return {
            "sample_groups": int(self.sample_groups),
            "entry_count_mean": self.entry_count_sum / denom,
            "real_entry_count_mean": real_entries / denom,
            "pad_entry_count_mean": self.pad_entry_count_sum / denom,
            "token_count_mean": self.token_count_sum / denom,
            "logical_anchor_count_mean": logical / denom,
            "fixed3_anchor_count_mean": fixed3_entry / denom,
            "fixed3_entry_anchor_count_mean": fixed3_entry / denom,
            "fixed3_real_anchor_count_mean": fixed3_real / denom,
            "lo_hi_anchor_count_mean": lo_hi / denom,
            "E_M": _rate(logical, real_entries),
            "E_M_lo_hi": _rate(lo_hi, real_entries),
            "anchor_count_per_token": _rate(logical, self.token_count_sum),
            "fixed3_over_logical_ratio": _ratio(fixed3_entry, logical),
            "fixed3_real_over_logical_ratio": _ratio(fixed3_real, logical),
            "logical_over_lo_hi_ratio": _ratio(logical, lo_hi),
            "gather_savings_fraction_vs_fixed3": None if fixed3_entry <= 0 else 1.0 - logical / fixed3_entry,
            "gather_savings_fraction_vs_fixed3_real": None
            if fixed3_real <= 0
            else 1.0 - logical / fixed3_real,
            "lo_hi_savings_fraction_vs_fixed3": None if fixed3_entry <= 0 else 1.0 - lo_hi / fixed3_entry,
            "lo_hi_savings_fraction_vs_fixed3_real": None if fixed3_real <= 0 else 1.0 - lo_hi / fixed3_real,
            "m_counts": {str(k): int(self.m_counts.get(k, 0)) for k in (1, 2, 3)},
            "m_fractions": {str(k): _rate(self.m_counts.get(k, 0), real_entries) for k in (1, 2, 3)},
            "lo_hi_m_counts": {str(k): int(self.lo_hi_m_counts.get(k, 0)) for k in (1, 2)},
            "mid_new_count": int(self.mid_new_count),
            "mid_new_fraction": _rate(self.mid_new_count, real_entries),
            "M_quantiles": _quantiles(self.m_values),
            "lo_hi_M_quantiles": _quantiles(self.lo_hi_m_values),
            "entry_width_quantiles": _quantiles(self.widths),
            "entry_span_quantiles": _quantiles(self.spans),
            "entry_width_max": max(self.widths) if self.widths else None,
            "entry_span_max": max(self.spans) if self.spans else None,
            "sh_source": sorted(self.sh_sources),
        }


def _add_ratios(rows: list[dict[str, Any]]) -> None:
    baselines = {row["scheme"]: row for row in rows if row.get("scheme") in RATIO_BASELINES.values()}
    metrics = (
        "entry_count_mean",
        "real_entry_count_mean",
        "logical_anchor_count_mean",
        "fixed3_anchor_count_mean",
        "fixed3_entry_anchor_count_mean",
        "fixed3_real_anchor_count_mean",
        "current_scheme_physical_slot_count_mean",
        "lo_hi_anchor_count_mean",
    )
    for row in rows:
        if row.get("scheme") != SCHEME_SEMANTIC:
            continue
        for suffix, baseline_scheme in RATIO_BASELINES.items():
            baseline = baselines.get(baseline_scheme)
            if baseline is None:
                continue
            for metric in metrics:
                row[f"{metric}_ratio_vs_{suffix}"] = _ratio(row.get(metric), baseline.get(metric))


def _add_ratios_by_layer_group(rows: list[dict[str, Any]]) -> None:
    """Same idea as ``_add_ratios``, computed independently per (layer, group).

    ``_add_ratios``'s baseline lookup keys purely by ``scheme`` and assumes at
    most one row per scheme -- true for ``overall_rows`` (one row per scheme,
    no layer/group axis) but false for ``by_layer_group`` rows, which have one
    row per (scheme, layer, group). Calling ``_add_ratios`` directly on those
    would silently pick one arbitrary (layer, group)'s baseline row per scheme
    (whichever the dict comprehension's iteration order happens to keep last)
    and divide *every other* (layer, group)'s semantic numbers by it --
    comparing one layer's semantic entries against a different layer's
    baseline. This groups rows by (layer, group) first so each semantic row is
    only ever compared against its own (layer, group)'s baseline rows, and is
    what actually populates the ``_ratio_vs_*`` fields on ``by_layer_group``
    output -- previously nothing called this (or ``_add_ratios``) on those
    rows at all, so they had no ratio fields, only absolute counts, making it
    impossible to check per CLAUDE.md S2.5 whether the E[M]/entry-count story
    is uniform across layers or concentrated in a few (the same class of gap
    as semantic_s0_analyze.py's missing --csv_by_layer).
    """
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["group"]))].append(row)
    for group_rows in grouped.values():
        _add_ratios(group_rows)


def _attach_scheme_physical_width(row: dict[str, Any]) -> None:
    scheme = row.get("scheme")
    if scheme in {SCHEME_SEMANTIC, SCHEME_SINGLE_CLUSTER}:
        row["current_scheme_physical_slot_count_mean"] = row.get("fixed3_anchor_count_mean")
        row["current_scheme_physical_slot_note"] = "anchor_entry_x3_no_gather"
    elif scheme == SCHEME_VANILLA_COMPRESSED:
        row["current_scheme_physical_slot_count_mean"] = row.get("entry_count_mean")
        row["current_scheme_physical_slot_note"] = "vanilla_compressed_prefix_only_recent_window_omitted"
    elif scheme == SCHEME_VANILLA_FULL:
        row["current_scheme_physical_slot_count_mean"] = row.get("entry_count_mean")
        row["current_scheme_physical_slot_note"] = "vanilla_full_cache_compressed_prefix_plus_exact_recent"
    else:
        row["current_scheme_physical_slot_count_mean"] = None
        row["current_scheme_physical_slot_note"] = "unknown_scheme"


def _fmt_num(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.3g}"


def _fmt_pct(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _print_summary(rows: list[dict[str, Any]], *, top: int) -> None:
    baseline_rows = [row for row in rows if row.get("scheme") != SCHEME_SEMANTIC]
    semantic_rows = [row for row in rows if row.get("scheme") == SCHEME_SEMANTIC]
    print("== S0.6 anchor dedup E[M] ==")
    print("口径: 每个 real entry 的 [p_lo,p_mid,p_hi] 去重后锚点数 M；pad/dead entry 不计入 E[M]。")
    print(
        "注意: fixed3 对 anchor-based scheme 是当前 v1 无 gather 的 entry×3 宽度；"
        "跨 scheme 物理宽度看 phys，E[M] 只衡量未来 packed/gather 潜力。"
    )
    if baseline_rows:
        print("\n== Baselines ==")
        print("scheme                              E[M]  M=1    M=2    M=3    logical  fixed3  phys  entries")
        print("----------------------------------  ----  -----  -----  -----  -------  ------  ----  -------")
        for row in baseline_rows:
            m_frac = row.get("m_fractions") or {}
            print(
                f"{str(row['scheme']).ljust(34)}  "
                f"{_fmt_num(row.get('E_M')).rjust(4)}  "
                f"{_fmt_pct(m_frac.get('1')).rjust(5)}  "
                f"{_fmt_pct(m_frac.get('2')).rjust(5)}  "
                f"{_fmt_pct(m_frac.get('3')).rjust(5)}  "
                f"{_fmt_num(row.get('logical_anchor_count_mean')).rjust(7)}  "
                f"{_fmt_num(row.get('fixed3_anchor_count_mean')).rjust(6)}  "
                f"{_fmt_num(row.get('current_scheme_physical_slot_count_mean')).rjust(4)}  "
                f"{_fmt_num(row.get('entry_count_mean')).rjust(7)}"
            )

    if semantic_rows:
        print("\n== Semantic configs ==")
        print(
            "cfg          E[M]  M=3    save   logical  fixed3  phys  entries  "
            "entry/single  logical/single  phys/vanilla"
        )
        print(
            "-----------  ----  -----  -----  -------  ------  ----  -------  "
            "------------  --------------  ------------"
        )
        ranked = sorted(semantic_rows, key=lambda row: (float(row.get("E_M") or 0), str(row.get("g_max"))))
        for row in ranked[: max(int(top), 1)]:
            m_frac = row.get("m_fractions") or {}
            cfg = f"{row.get('g_max')}:{row.get('l_block')}"
            print(
                f"{cfg.ljust(11)}  "
                f"{_fmt_num(row.get('E_M')).rjust(4)}  "
                f"{_fmt_pct(m_frac.get('3')).rjust(5)}  "
                f"{_fmt_pct(row.get('gather_savings_fraction_vs_fixed3')).rjust(5)}  "
                f"{_fmt_num(row.get('logical_anchor_count_mean')).rjust(7)}  "
                f"{_fmt_num(row.get('fixed3_anchor_count_mean')).rjust(6)}  "
                f"{_fmt_num(row.get('current_scheme_physical_slot_count_mean')).rjust(4)}  "
                f"{_fmt_num(row.get('entry_count_mean')).rjust(7)}  "
                f"{_fmt_num(row.get('entry_count_mean_ratio_vs_single_cluster')).rjust(12)}  "
                f"{_fmt_num(row.get('logical_anchor_count_mean_ratio_vs_single_cluster')).rjust(14)}  "
                f"{_fmt_num(row.get('current_scheme_physical_slot_count_mean_ratio_vs_vanilla_full')).rjust(12)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="Stage-0 dump directory or manifest.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--g_max", default="inf,8192,4096,2048,1024,256")
    parser.add_argument("--l_block", default="0,1,2,3")
    parser.add_argument("--lambda_rel", type=float, default=1.0)
    parser.add_argument("--seg_forget", type=float, default=0.5)
    parser.add_argument("--b_prime", type=int, default=8)
    parser.add_argument("--vanilla_B", type=int, default=512)
    parser.add_argument("--vanilla_recent_size", type=int, default=1024)
    parser.add_argument("--layers", help="Optional comma-separated layer filter")
    parser.add_argument("--groups", help="Optional comma-separated KV-group filter")
    parser.add_argument("--top", type=int, default=16, help="Rows to print in the terminal summary")
    parser.add_argument(
        "--allow_fallback_sh",
        action="store_true",
        help="Use this prompt's local key variance if calibrated manifest key_scale is missing/corrupt.",
    )
    parser.add_argument(
        "--skip_by_layer_group",
        action="store_true",
        help="Omit per-(scheme, g_max, l_block, layer, group) rows and write only overall_by_config.",
    )
    args = parser.parse_args()

    g_values = parse_g_max_list(args.g_max)
    l_values = parse_int_list(args.l_block)
    if not g_values:
        raise ValueError(f"--g_max {args.g_max!r} parsed to an empty list -- pass at least one value")
    if not l_values:
        raise ValueError(f"--l_block {args.l_block!r} parsed to an empty list -- pass at least one value")
    if not (math.isfinite(args.lambda_rel) and args.lambda_rel > 0.0):
        raise ValueError(f"--lambda_rel must be finite and > 0, got {args.lambda_rel}")
    if not (math.isfinite(args.seg_forget) and 0.0 <= args.seg_forget <= 1.0):
        raise ValueError(f"--seg_forget must be in [0, 1], got {args.seg_forget}")
    bad_g_max = [g for g in g_values if not (g == math.inf or (math.isfinite(g) and g >= 0.0))]
    if bad_g_max:
        raise ValueError(f"--g_max values must be finite >= 0, or inf, got {bad_g_max}")
    if args.b_prime <= 0 or args.b_prime % 2 != 0:
        raise ValueError(f"--b_prime must be a positive even integer, got {args.b_prime}")
    if args.vanilla_B <= 0:
        raise ValueError(f"--vanilla_B must be a positive integer, got {args.vanilla_B}")
    if args.vanilla_recent_size < 2:
        raise ValueError(f"--vanilla_recent_size must be >= 2, got {args.vanilla_recent_size}")

    manifest = load_manifest(args.dump)
    base_dir = manifest_base_dir(args.dump)
    layer_filter = _wanted(args.layers)
    group_filter = _wanted(args.groups)
    records = _iter_records(manifest, layer_filter=layer_filter)
    if not records:
        raise ValueError("No layer records matched the requested filters")

    overall: dict[tuple[str, str | None, int | None], AnchorDedupAccumulator] = {}
    by_layer_group: dict[tuple[str, str | None, int | None, int, int], AnchorDedupAccumulator] = {}
    processed_pairs = 0
    groups_seen: set[int] = set()

    def add_stats(
        *,
        scheme: str,
        entries: list[Any],
        ladder_meta: dict[str, Any],
        sh_source: str,
        layer: int,
        group: int,
        g_label: str | None = None,
        l_block: int | None = None,
    ) -> None:
        overall.setdefault((scheme, g_label, l_block), AnchorDedupAccumulator()).add(
            entries=entries,
            ladder_meta=ladder_meta,
            sh_source=sh_source,
        )
        by_layer_group.setdefault((scheme, g_label, l_block, layer, group), AnchorDedupAccumulator()).add(
            entries=entries,
            ladder_meta=ladder_meta,
            sh_source=sh_source,
        )

    for record_i, (sample, record) in enumerate(records, start=1):
        layer = int(record["layer"])
        payload = np.load(base_dir / record["path"])
        k_raw = payload["k_raw"]
        groups: Any = range(k_raw.shape[0])
        groups_seen.update(groups)
        if group_filter is not None:
            groups = [g for g in groups if g in group_filter]

        for group in groups:
            processed_pairs += 1
            k_group = np.asarray(k_raw[group], dtype=np.float32)
            sh, sh_source = _manifest_sh(
                manifest,
                layer,
                int(group),
                k_group,
                allow_fallback=args.allow_fallback_sh,
            )
            lambda_new = float(args.lambda_rel) * sh

            single_route = route_single_cluster_bprime_ladder(k_group.shape[0])
            single_entries, single_meta = simulate_segment_ladders(single_route, b_prime=args.b_prime, l_block=0)
            add_stats(
                scheme=SCHEME_SINGLE_CLUSTER,
                entries=single_entries,
                ladder_meta=single_meta,
                sh_source=sh_source,
                layer=layer,
                group=int(group),
            )

            vanilla_compressed_entries, vanilla_compressed_meta = vanilla_logkv_compressed_entries(
                k_group.shape[0],
                b=args.vanilla_B,
                recent_size=args.vanilla_recent_size,
            )
            add_stats(
                scheme=SCHEME_VANILLA_COMPRESSED,
                entries=vanilla_compressed_entries,
                ladder_meta=vanilla_compressed_meta,
                sh_source=sh_source,
                layer=layer,
                group=int(group),
            )

            vanilla_full_entries, vanilla_full_meta = vanilla_logkv_full_cache_entries(
                k_group.shape[0],
                b=args.vanilla_B,
                recent_size=args.vanilla_recent_size,
            )
            add_stats(
                scheme=SCHEME_VANILLA_FULL,
                entries=vanilla_full_entries,
                ladder_meta=vanilla_full_meta,
                sh_source=sh_source,
                layer=layer,
                group=int(group),
            )

            route_cache: dict[float, Any] = {}
            for g_max in g_values:
                g_label = format_g_max(g_max)
                for l_block in l_values:
                    effective_g_max = _effective_g_max(g_max, int(l_block))
                    route = route_cache.get(effective_g_max)
                    if route is None:
                        route = route_dpmeans_segments(
                            k_group,
                            lambda_new=lambda_new,
                            g_max=effective_g_max,
                            gamma=args.seg_forget,
                        )
                        route_cache[effective_g_max] = route
                    entries, meta = simulate_segment_ladders(route, b_prime=args.b_prime, l_block=int(l_block))
                    add_stats(
                        scheme=SCHEME_SEMANTIC,
                        entries=entries,
                        ladder_meta=meta,
                        sh_source=sh_source,
                        layer=layer,
                        group=int(group),
                        g_label=g_label,
                        l_block=int(l_block),
                    )

        print(
            f"[s0-anchor] {record_i}/{len(records)} sample={sample.get('sample_id')} "
            f"layer={layer} path={record['path']}",
            flush=True,
        )

    if processed_pairs == 0:
        raise ValueError(
            f"--groups {sorted(group_filter) if group_filter is not None else group_filter} matched none "
            f"of the KV groups actually present in the dump ({sorted(groups_seen)}); "
            f"0 (layer, group) pairs were processed."
        )

    overall_rows = []
    for (scheme, g_label, l_block), acc in sorted(
        overall.items(),
        key=lambda item: (
            item[0][0],
            "" if item[0][1] is None else item[0][1],
            -1 if item[0][2] is None else item[0][2],
        ),
    ):
        row = {"scheme": scheme, "g_max": g_label, "l_block": l_block}
        row.update(acc.finalize())
        _attach_scheme_physical_width(row)
        overall_rows.append(row)
    _add_ratios(overall_rows)

    by_lg_rows = []
    for (scheme, g_label, l_block, layer, group), acc in sorted(
        by_layer_group.items(),
        key=lambda item: (
            item[0][0],
            "" if item[0][1] is None else item[0][1],
            -1 if item[0][2] is None else item[0][2],
            item[0][3],
            item[0][4],
        ),
    ):
        row = {"scheme": scheme, "g_max": g_label, "l_block": l_block, "layer": layer, "group": group}
        row.update(acc.finalize())
        _attach_scheme_physical_width(row)
        by_lg_rows.append(row)
    _add_ratios_by_layer_group(by_lg_rows)

    result = {
        "version": 1,
        "kind": "semantic_logkv_s0_6_anchor_dedup",
        "source_manifest": str(Path(args.dump)),
        "config": {
            "lambda_rel": args.lambda_rel,
            "seg_forget": args.seg_forget,
            "b_prime": args.b_prime,
            "vanilla_B": args.vanilla_B,
            "vanilla_recent_size": args.vanilla_recent_size,
            "g_max": [format_g_max(x) for x in g_values],
            "l_block": l_values,
            "layers": None if layer_filter is None else sorted(layer_filter),
            "groups": None if group_filter is None else sorted(group_filter),
            "routing_eta": 0.0,
            "routing_mode": "strict_serial_dpmeans_unclipped",
            "l_block_zero_forces_g_max_inf": True,
        },
        "summary": {
            "record_count": len(records),
            "processed_layer_group_pairs": processed_pairs,
            "metric_note": (
                "E[M] skips pad/dead entries and counts distinct anchors after dedup([p_lo,p_mid,p_hi]). "
                "fixed3_anchor_count_mean aliases fixed3_entry_anchor_count_mean and is the anchor-expanded "
                "entry*3 width; it is current v1 physical width only for anchor-based semantic/single-cluster "
                "schemes, and hypothetical for vanilla baselines. current_scheme_physical_slot_count_mean is "
                "the comparable scheme-native physical width. fixed3_real_anchor_count_mean is the pad/dead-"
                "filtered auxiliary lower-bound width. logical_anchor_count_mean is the future gather/packed "
                "width implied by dedup."
            ),
        },
        "overall_by_config": overall_rows,
        "by_layer_group": [] if args.skip_by_layer_group else by_lg_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _print_summary(overall_rows, top=args.top)
    print(f"[s0-anchor] wrote {args.output}")


if __name__ == "__main__":
    main()
