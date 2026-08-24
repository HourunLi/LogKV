#!/usr/bin/env python
"""Run the SemanticLogKV S0.0 `(g_max, l_block)` CPU sweep on a Stage-0 dump.

Example:
    python unused/semantic_s0_sweep.py \
      --dump stage0_dump \
      --output stage0_dump/s0_sweep.json \
      --g_max inf,8192,4096,2048,1024,256 \
      --l_block 0,1,2,3 \
      --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
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

RouteResult = _S0.RouteResult
SweepAccumulator = _S0.SweepAccumulator
format_g_max = _S0.format_g_max
load_manifest = _S0.load_manifest
manifest_base_dir = _S0.manifest_base_dir
parse_g_max_list = _S0.parse_g_max_list
parse_int_list = _S0.parse_int_list
route_dpmeans_segments = _S0.route_dpmeans_segments
route_single_cluster_bprime_ladder = _S0.route_single_cluster_bprime_ladder
simulate_segment_ladders = _S0.simulate_segment_ladders
summarize_entries = _S0.summarize_entries
vanilla_logkv_compressed_entries = _S0.vanilla_logkv_compressed_entries
vanilla_logkv_full_cache_entries = _S0.vanilla_logkv_full_cache_entries


def _wanted(values: str | None) -> set[int] | None:
    return None if values is None else set(parse_int_list(values))


def _groups_from_key_scale(manifest: dict[str, Any], layer: int) -> list[int] | None:
    try:
        values = manifest["key_scale"][str(int(layer))]["s_h"]
    except Exception:
        return None
    if not isinstance(values, list):
        return None
    return list(range(len(values)))


def _fallback_scale(x: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float32)
    mean = xx.mean(axis=0, keepdims=True)
    return float(np.square(xx - mean, dtype=np.float32).sum(axis=1).mean(dtype=np.float64))


def _manifest_scale(
    manifest: dict[str, Any],
    field: str,
    layer: int,
    group: int,
    x: np.ndarray,
    *,
    allow_fallback: bool,
) -> tuple[float, str]:
    """Returns ``(scale, source)``, ``source`` one of ``"calibrated"``/``"fallback_local"``.

    ``field`` is ``"key_scale"`` or ``"value_scale"`` -- the two are calibrated
    independently (key-space and value-space vectors have no reason to share a
    norm scale) and must not be cross-substituted. The calibrated value is the
    whole-calibration-set global ``E||x-xbar||^2`` the dump script wrote into
    ``manifest[field]`` (``algorithm-spec.md`` S5.2: thresholds/normalizers
    must be relative to this, not an online/per-sample estimate -- "S0.2 扫
    出来的 K_eff 曲线在层间完全没有可比性" otherwise). A missing/corrupted
    calibration entry hard-fails by default rather than silently substituting
    this one sample's own local variance, which is a materially noisier,
    non-comparable-across-layers quantity; pass ``allow_fallback=True`` to opt
    into that substitution instead. ``manifest[field]`` missing entirely
    (dumps written before value-scale calibration existed) is treated the
    same as a missing per-layer entry, not a hard crash on the outer lookup.
    """
    reason: str | None = None
    value: float | None = None
    try:
        value = float(manifest[field][str(layer)]["s_h"][group])
    except Exception as exc:
        reason = f"manifest[{field!r}][{str(layer)!r}]['s_h'][{group}] missing or malformed: {exc!r}"
    else:
        if not (math.isfinite(value) and value > 0):
            reason = f"manifest[{field!r}][{str(layer)!r}]['s_h'][{group}] = {value!r} is not a finite positive scale"
            value = None

    if value is not None:
        return value, "calibrated"

    if not allow_fallback:
        raise ValueError(
            f"{reason}. layer={layer} group={group} has no usable calibrated {field} from the "
            f"dump manifest. Pass --allow_fallback_sh to substitute this one prompt's own "
            f"local variance instead (noisier, and not comparable across layers/prompts -- see "
            f"algorithm-spec.md S5.2)."
        )
    return max(_fallback_scale(x), 1e-12), "fallback_local"


def _manifest_sh(
    manifest: dict[str, Any], layer: int, group: int, k: np.ndarray, *, allow_fallback: bool
) -> tuple[float, str]:
    return _manifest_scale(manifest, "key_scale", layer, group, k, allow_fallback=allow_fallback)


def _manifest_vh(
    manifest: dict[str, Any], layer: int, group: int, v: np.ndarray, *, allow_fallback: bool
) -> tuple[float, str]:
    return _manifest_scale(manifest, "value_scale", layer, group, v, allow_fallback=allow_fallback)


def _effective_g_max(g_max: float, l_block: int) -> float:
    """The g_max routing should actually use for one (g_max, l_block) sweep cell.

    algorithm-spec.md S5.3 defines the pure-semantic reference point as
    eta=0 AND (g_max=inf OR l_block=0). eta=0 is structural (route_dpmeans_
    segments has no eta parameter at all), but l_block=0 alone does not make
    routing behave like g_max=inf: with a finite g_max, gamma-decay still
    fires on every segment break, shifting centroid trajectories -- and
    therefore cluster_ids -- away from what pure (g_max=inf) DP-means would
    produce. Verified empirically (see tests): with identical data and eta=0,
    g_max=inf gives 2 clusters ([0,0,1]) while g_max=0 gives 1 ([0,0,0]).
    So the l_block=0 column must route with g_max=inf regardless of which
    g_max is nominally being swept, or it silently isn't the pure-semantic
    endpoint the sweep's whole point is to compare against.
    """
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


def _merge_sweep_accumulator(dst: Any, src: Any) -> None:
    dst.sample_groups += src.sample_groups
    dst.cluster_count_sum += src.cluster_count_sum
    dst.segment_count_sum += src.segment_count_sum
    dst.entry_count_sum += src.entry_count_sum
    dst.nonpad_entry_count_sum += src.nonpad_entry_count_sum
    dst.pad_entry_count_sum += src.pad_entry_count_sum
    dst.token_count_sum += src.token_count_sum
    dst.key_sse_sum += src.key_sse_sum
    dst.value_sse_sum += src.value_sse_sum
    dst.key_sse_norm_sum += src.key_sse_norm_sum
    dst.value_sse_norm_sum += src.value_sse_norm_sum
    dst.widths.extend(src.widths)
    dst.spans.extend(src.spans)
    dst.all_widths.extend(src.all_widths)
    dst.all_spans.extend(src.all_spans)
    dst.sh_sources.update(src.sh_sources)
    dst.vh_sources.update(src.vh_sources)
    dst.value_var_available = bool(dst.value_var_available and src.value_var_available)
    dst.value_var_relative_available = bool(dst.value_var_relative_available and src.value_var_relative_available)
    for key, value in src.meta_sums.items():
        dst.meta_sums[key] = dst.meta_sums.get(key, 0.0) + value
    for key, value in src.meta_counts.items():
        dst.meta_counts[key] = dst.meta_counts.get(key, 0) + value
    dst.coverage_source_token_sum += src.coverage_source_token_sum
    dst.coverage_token_sum += src.coverage_token_sum


def _merge_accumulator_map(dst: dict[Any, Any], src: dict[Any, Any]) -> None:
    for key, acc in src.items():
        _merge_sweep_accumulator(dst.setdefault(key, SweepAccumulator()), acc)


def _process_record_task(task: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    sample = task["sample"]
    record = task["record"]
    layer = int(record["layer"])
    record_i = int(task["record_i"])
    record_count = int(task["record_count"])
    base_dir = Path(task["base_dir"])
    g_values = [float(x) for x in task["g_values"]]
    l_values = [int(x) for x in task["l_values"]]
    group_filter = None if task["group_filter"] is None else set(int(x) for x in task["group_filter"])
    task_groups = None if task.get("task_groups") is None else set(int(x) for x in task["task_groups"])

    by_layer_group: dict[tuple[str, int, int, int], SweepAccumulator] = {}
    overall: dict[tuple[str, int], SweepAccumulator] = {}
    single_cluster_bprime_baseline: dict[tuple[int, int], SweepAccumulator] = {}
    vanilla_logkv_compressed_prefix_baseline: dict[tuple[int, int], SweepAccumulator] = {}
    vanilla_logkv_full_cache_baseline: dict[tuple[int, int], SweepAccumulator] = {}
    processed_pairs = 0
    timing_events: list[dict[str, Any]] = []

    with np.load(base_dir / record["path"]) as payload:
        k_raw = payload["k_raw"]
        v = None if bool(task["skip_value_var"]) else payload["v"]

    groups = list(range(k_raw.shape[0]))
    groups_seen = set(groups)
    if group_filter is not None:
        groups = [g for g in groups if g in group_filter]
    if task_groups is not None:
        groups = [g for g in groups if g in task_groups]

    for group in groups:
        processed_pairs += 1
        k_group = np.asarray(k_raw[group], dtype=np.float32)
        v_group = None if v is None else np.asarray(v[group], dtype=np.float32)
        sh, sh_source = _manifest_sh(
            task["scale_manifest"],
            layer,
            int(group),
            k_group,
            allow_fallback=bool(task["allow_fallback_sh"]),
        )
        lambda_rel = float(task["lambda_rel"])
        lambda_new = lambda_rel * sh
        vh, vh_source = (
            _manifest_vh(
                task["scale_manifest"],
                layer,
                int(group),
                v_group,
                allow_fallback=bool(task["allow_fallback_sh"]),
            )
            if v_group is not None
            else (None, None)
        )

        sc_route = route_single_cluster_bprime_ladder(k_group.shape[0])
        sc_entries, sc_ladder_meta = simulate_segment_ladders(
            sc_route,
            b_prime=int(task["b_prime"]),
            l_block=0,
        )
        sc_summary = summarize_entries(k_group, v_group, sc_entries)
        single_cluster_bprime_baseline.setdefault((layer, int(group)), SweepAccumulator()).add(
            route=sc_route,
            ladder_meta=sc_ladder_meta,
            summary=sc_summary,
            sh=sh,
            sh_source=sh_source,
            vh=vh,
            vh_source=vh_source,
        )

        vanilla_route = RouteResult(
            cluster_ids=np.zeros(0, dtype=np.int32),
            segment_ids=np.zeros(0, dtype=np.int32),
            cluster_count=1,
            segment_count=1,
            cluster_sizes=[k_group.shape[0]],
        )
        vanilla_compressed_entries, vanilla_compressed_meta = vanilla_logkv_compressed_entries(
            k_group.shape[0],
            b=int(task["vanilla_B"]),
            recent_size=int(task["vanilla_recent_size"]),
        )
        vanilla_compressed_summary = summarize_entries(k_group, v_group, vanilla_compressed_entries)
        vanilla_logkv_compressed_prefix_baseline.setdefault((layer, int(group)), SweepAccumulator()).add(
            route=vanilla_route,
            ladder_meta=vanilla_compressed_meta,
            summary=vanilla_compressed_summary,
            sh=sh,
            sh_source=sh_source,
            vh=vh,
            vh_source=vh_source,
        )

        vanilla_full_entries, vanilla_full_meta = vanilla_logkv_full_cache_entries(
            k_group.shape[0],
            b=int(task["vanilla_B"]),
            recent_size=int(task["vanilla_recent_size"]),
        )
        vanilla_full_summary = summarize_entries(k_group, v_group, vanilla_full_entries)
        vanilla_logkv_full_cache_baseline.setdefault((layer, int(group)), SweepAccumulator()).add(
            route=vanilla_route,
            ladder_meta=vanilla_full_meta,
            summary=vanilla_full_summary,
            sh=sh,
            sh_source=sh_source,
            vh=vh,
            vh_source=vh_source,
        )

        route_cache: dict[float, Any] = {}
        for g_max in g_values:
            g_label = format_g_max(g_max)
            for l_block in l_values:
                cell_started_at = time.perf_counter()
                effective_g_max = _effective_g_max(g_max, int(l_block))
                effective_g_label = format_g_max(effective_g_max)
                route = route_cache.get(effective_g_max)
                route_cached = route is not None
                route_elapsed = 0.0
                if route is None:
                    route_started_at = time.perf_counter()
                    route = route_dpmeans_segments(
                        k_group,
                        lambda_new=lambda_new,
                        g_max=effective_g_max,
                        gamma=float(task["seg_forget"]),
                    )
                    route_elapsed = time.perf_counter() - route_started_at
                    route_cache[effective_g_max] = route
                entries, ladder_meta = simulate_segment_ladders(
                    route,
                    b_prime=int(task["b_prime"]),
                    l_block=int(l_block),
                )
                summary = summarize_entries(k_group, v_group, entries)
                key = (g_label, int(l_block), layer, int(group))
                by_layer_group.setdefault(key, SweepAccumulator()).add(
                    route=route,
                    ladder_meta=ladder_meta,
                    summary=summary,
                    sh=sh,
                    sh_source=sh_source,
                    vh=vh,
                    vh_source=vh_source,
                )
                if task.get("log_timing"):
                    timing_events.append(
                        {
                            "task_i": task.get("task_i"),
                            "task_count": task.get("task_count"),
                            "record_i": record_i,
                            "record_count": record_count,
                            "layer": layer,
                            "group": int(group),
                            "lambda_rel": lambda_rel,
                            "g_max": g_label,
                            "l_block": int(l_block),
                            "effective_g_max": effective_g_label,
                            "route_cached": route_cached,
                            "route_elapsed_seconds": route_elapsed,
                            "cell_elapsed_seconds": time.perf_counter() - cell_started_at,
                            "cluster_count": int(route.cluster_count),
                            "segment_count": int(route.segment_count),
                        }
                    )
                overall.setdefault((g_label, int(l_block)), SweepAccumulator()).add(
                    route=route,
                    ladder_meta=ladder_meta,
                    summary=summary,
                    sh=sh,
                    sh_source=sh_source,
                    vh=vh,
                    vh_source=vh_source,
                )

    task_i = task.get("task_i")
    task_count = task.get("task_count")
    task_prefix = f"task={task_i}/{task_count} " if task_i is not None and task_count is not None else ""
    group_suffix = f" groups={sorted(task_groups)}" if task_groups is not None else ""
    elapsed = time.perf_counter() - started_at
    timing_suffix = f" elapsed={elapsed:.2f}s" if task.get("log_timing") else ""
    message = (
        f"[s0-sweep] {task_prefix}record={record_i}/{record_count} sample={sample.get('sample_id')} "
        f"layer={layer} path={record['path']}{group_suffix}{timing_suffix}"
    )
    return {
        "by_layer_group": by_layer_group,
        "overall": overall,
        "single_cluster_bprime_baseline": single_cluster_bprime_baseline,
        "vanilla_logkv_compressed_prefix_baseline": vanilla_logkv_compressed_prefix_baseline,
        "vanilla_logkv_full_cache_baseline": vanilla_logkv_full_cache_baseline,
        "processed_pairs": processed_pairs,
        "groups_seen": groups_seen,
        "elapsed_seconds": elapsed,
        "timing_events": timing_events,
        "message": message,
    }


def _format_timing_event(event: dict[str, Any]) -> str:
    cached = "cached" if event.get("route_cached") else "computed"
    return (
        "[s0-sweep-timing] "
        f"task={event.get('task_i')}/{event.get('task_count')} "
        f"record={event.get('record_i')}/{event.get('record_count')} "
        f"layer={event.get('layer')} group={event.get('group')} "
        f"lambda_rel={float(event['lambda_rel']):g} g_max={event.get('g_max')} "
        f"l_block={event.get('l_block')} effective_g_max={event.get('effective_g_max')} "
        f"route={float(event.get('route_elapsed_seconds') or 0.0):.3f}s({cached}) "
        f"cell={float(event.get('cell_elapsed_seconds') or 0.0):.3f}s "
        f"clusters={event.get('cluster_count')} segments={event.get('segment_count')}"
    )


def _add_timing_stat(stats: dict[tuple[float, str, int, str], dict[str, Any]], event: dict[str, Any]) -> None:
    key = (float(event["lambda_rel"]), str(event["g_max"]), int(event["l_block"]), str(event["effective_g_max"]))
    row = stats.setdefault(
        key,
        {
            "lambda_rel": key[0],
            "g_max": key[1],
            "l_block": key[2],
            "effective_g_max": key[3],
            "cell_count": 0,
            "route_calls": 0,
            "route_elapsed_total": 0.0,
            "route_elapsed_max": 0.0,
            "cell_elapsed_total": 0.0,
            "cell_elapsed_max": 0.0,
            "cluster_count_max": 0,
            "segment_count_max": 0,
        },
    )
    route_elapsed = float(event.get("route_elapsed_seconds") or 0.0)
    cell_elapsed = float(event.get("cell_elapsed_seconds") or 0.0)
    row["cell_count"] += 1
    row["cell_elapsed_total"] += cell_elapsed
    row["cell_elapsed_max"] = max(float(row["cell_elapsed_max"]), cell_elapsed)
    row["cluster_count_max"] = max(int(row["cluster_count_max"]), int(event.get("cluster_count") or 0))
    row["segment_count_max"] = max(int(row["segment_count_max"]), int(event.get("segment_count") or 0))
    if not event.get("route_cached"):
        row["route_calls"] += 1
        row["route_elapsed_total"] += route_elapsed
        row["route_elapsed_max"] = max(float(row["route_elapsed_max"]), route_elapsed)


def _timing_rows(stats: dict[tuple[float, str, int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in stats.values():
        out = dict(row)
        out["route_elapsed_mean"] = (
            float(out["route_elapsed_total"]) / int(out["route_calls"]) if int(out["route_calls"]) else 0.0
        )
        out["cell_elapsed_mean"] = (
            float(out["cell_elapsed_total"]) / int(out["cell_count"]) if int(out["cell_count"]) else 0.0
        )
        rows.append(out)
    return sorted(rows, key=lambda x: (float(x["lambda_rel"]), str(x["g_max"]), int(x["l_block"])))


def _print_timing_summary(rows: list[dict[str, Any]], *, top: int = 20) -> None:
    if not rows:
        return
    print("== S0.0 routing timing by (lambda_rel, g_max, l_block) ==")
    print(
        "lambda_rel  g_max  l_block  eff_g  route_total  route_mean  route_max  "
        "cell_total  cell_mean  clusters_max  segments_max  cells"
    )
    for row in sorted(rows, key=lambda x: float(x["route_elapsed_total"]), reverse=True)[:top]:
        print(
            f"{float(row['lambda_rel']):>10g}  "
            f"{str(row['g_max']).rjust(5)}  "
            f"{int(row['l_block']):7d}  "
            f"{str(row['effective_g_max']).rjust(5)}  "
            f"{float(row['route_elapsed_total']):11.3f}s  "
            f"{float(row['route_elapsed_mean']):10.3f}s  "
            f"{float(row['route_elapsed_max']):9.3f}s  "
            f"{float(row['cell_elapsed_total']):10.3f}s  "
            f"{float(row['cell_elapsed_mean']):9.3f}s  "
            f"{int(row['cluster_count_max']):12d}  "
            f"{int(row['segment_count_max']):12d}  "
            f"{int(row['cell_count'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="Dump directory or manifest.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--g_max", default="inf,8192,4096,2048,1024,256")
    parser.add_argument("--l_block", default="0,1,2,3")
    parser.add_argument("--lambda_rel", type=float, default=1.0)
    parser.add_argument("--seg_forget", type=float, default=0.5)
    parser.add_argument("--b_prime", type=int, default=8)
    parser.add_argument(
        "--vanilla_B",
        type=int,
        default=512,
        help="B for the vanilla_logkv_compressed/full_cache_baseline reference points -- "
        "matches LogStructuredKVCache's --log_kv_B / self.B, default 512 to match the deployed "
        "exp/qwen1.7b-32k/base.yaml config. NOT the same knob as --b_prime, which only controls "
        "the semantic sweep cells and single_cluster_bprime_baseline. Unlike --b_prime, this "
        "does not need to be even -- the real LogStructuredKVCache places no such constraint on "
        "B (verified empirically, see vanilla_logkv_compressed_entries's docstring); --b_prime's "
        "evenness requirement is for a semantic-design-specific reason (PAD_INSERT alignment) "
        "that does not apply here.",
    )
    parser.add_argument(
        "--vanilla_recent_size",
        type=int,
        default=1024,
        help="recent_size for the vanilla_logkv_compressed/full_cache_baseline reference points -- matches "
        "LogStructuredKVCache's --log_kv_recent_size, default 1024 to match the deployed "
        "exp/qwen1.7b-32k/base.yaml config. The semantic sweep and single_cluster_bprime_baseline "
        "have no recent-window concept at all (every token goes through compaction), so this only "
        "affects the two vanilla_logkv_*_baseline outputs.",
    )
    parser.add_argument("--layers", help="Optional comma-separated layer filter")
    parser.add_argument("--groups", help="Optional comma-separated KV-group filter")
    parser.add_argument("--skip_value_var", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Use 1 for deterministic single-process execution.",
    )
    parser.add_argument(
        "--parallel_unit",
        choices=("record", "group"),
        default="record",
        help=(
            "Task granularity for multiprocessing. 'record' keeps the original one-layer-record task; "
            "'group' splits each layer record into one task per KV group to reduce straggler imbalance."
        ),
    )
    parser.add_argument(
        "--log_timing",
        action="store_true",
        help="Append task elapsed seconds and print per-(lambda_rel, g_max, l_block) routing timing details.",
    )
    parser.add_argument(
        "--allow_fallback_sh",
        action="store_true",
        help="If a (layer, group) has no usable calibrated s_h (key scale) or vh (value scale, "
        "only looked up when --skip_value_var is not set) in the dump manifest, substitute "
        "this one prompt's own local variance instead of hard-failing. algorithm-spec.md S5.2 "
        "requires the calibrated, whole-calibration-set scale for cross-layer comparability -- "
        "only use this to unblock a smoke test on a corrupted/partial manifest (e.g. one written "
        "before value-scale calibration existed), not for a real S0.0 run.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.dump)
    base_dir = manifest_base_dir(args.dump)
    g_values = parse_g_max_list(args.g_max)
    l_values = parse_int_list(args.l_block)
    if not g_values:
        raise ValueError(f"--g_max {args.g_max!r} parsed to an empty list -- pass at least one value")
    if not l_values:
        raise ValueError(f"--l_block {args.l_block!r} parsed to an empty list -- pass at least one value")
    # Fail fast on degenerate sweep parameters instead of silently producing a
    # complete-looking but meaningless result (route_dpmeans_segments enforces
    # the same bounds on lambda_new/g_max, but only once routing actually
    # starts -- after loading the dump; checking here catches it before any
    # of that work happens). -inf is deliberately rejected alongside negative
    # finite values: math.isinf(-inf) is True, so it would otherwise slip
    # through a naive "isinf or >= 0" check.
    if not (math.isfinite(args.lambda_rel) and args.lambda_rel > 0.0):
        raise ValueError(f"--lambda_rel must be finite and > 0, got {args.lambda_rel}")
    if not (math.isfinite(args.seg_forget) and 0.0 <= args.seg_forget <= 1.0):
        raise ValueError(f"--seg_forget must be in [0, 1], got {args.seg_forget}")
    bad_g_max = [g for g in g_values if not (g == math.inf or (math.isfinite(g) and g >= 0.0))]
    if bad_g_max:
        raise ValueError(f"--g_max values must be finite >= 0, or inf, got {bad_g_max}")
    if args.vanilla_B <= 0:
        raise ValueError(f"--vanilla_B must be a positive integer, got {args.vanilla_B}")
    if args.vanilla_recent_size < 2:
        raise ValueError(f"--vanilla_recent_size must be >= 2, got {args.vanilla_recent_size}")
    if args.workers <= 0:
        raise ValueError(f"--workers must be a positive integer, got {args.workers}")
    layer_filter = _wanted(args.layers)
    group_filter = _wanted(args.groups)

    by_layer_group: dict[tuple[str, int, int, int], SweepAccumulator] = {}
    overall: dict[tuple[str, int], SweepAccumulator] = {}
    # S0.4 needs semantic-cluster intra-entry variance compared against the
    # existing position-bucketed LogKV's intra-slot variance -- three
    # different reference points for that, none producible by
    # route_dpmeans_segments (even g_max=inf is still semantic DP-means, not
    # arrival-order routing). All keyed by (layer, group) only, since none
    # vary with (g_max, l_block): computed once per group, not per sweep
    # cell.
    #
    # single_cluster_bprime_baseline: same B'-budget ladder mechanics as the
    # semantic sweep cells, but with no clustering at all -- isolates what
    # semantic grouping itself contributes, holding the ladder mechanics
    # fixed. Not a reproduction of the real deployed LogKV (see
    # route_single_cluster_bprime_ladder's docstring for the three respects
    # in which it differs).
    #
    # vanilla_logkv_compressed_prefix_baseline / vanilla_logkv_full_cache_
    # baseline: the actual existing/deployed LogKV -- real log_kv_B /
    # log_kv_recent_size, real recent-window carve-out, real w=2 level-0
    # pairing (see vanilla_logkv_compressed_entries's docstring). The two
    # differ in whether the exact recent-window tokens are folded in --
    # compressed_prefix (no) is the primary S0.4 metric, full_cache (yes) is
    # a secondary sanity check -- see vanilla_logkv_full_cache_entries's
    # docstring for why.
    single_cluster_bprime_baseline: dict[tuple[int, int], SweepAccumulator] = {}
    vanilla_logkv_compressed_prefix_baseline: dict[tuple[int, int], SweepAccumulator] = {}
    vanilla_logkv_full_cache_baseline: dict[tuple[int, int], SweepAccumulator] = {}
    records = _iter_records(manifest, layer_filter=layer_filter)
    if not records:
        raise ValueError("No layer records matched the requested filters")

    processed_pairs = 0
    groups_seen: set[int] = set()
    timing_stats: dict[tuple[float, str, int, str], dict[str, Any]] = {}
    scale_manifest = {
        "key_scale": manifest.get("key_scale", {}),
        "value_scale": manifest.get("value_scale", {}),
    }
    base_tasks = []
    for record_i, (sample, record) in enumerate(records, start=1):
        base_task = {
            "sample": sample,
            "record": record,
            "record_i": record_i,
            "record_count": len(records),
            "base_dir": str(base_dir),
            "scale_manifest": scale_manifest,
            "g_values": g_values,
            "l_values": l_values,
            "group_filter": None if group_filter is None else sorted(group_filter),
            "lambda_rel": args.lambda_rel,
            "seg_forget": args.seg_forget,
            "b_prime": args.b_prime,
            "vanilla_B": args.vanilla_B,
            "vanilla_recent_size": args.vanilla_recent_size,
            "skip_value_var": args.skip_value_var,
            "allow_fallback_sh": args.allow_fallback_sh,
            "log_timing": args.log_timing,
        }
        base_tasks.append(base_task)

    if args.parallel_unit == "record":
        tasks = base_tasks
    else:
        tasks = []
        for base_task in base_tasks:
            layer = int(base_task["record"]["layer"])
            if group_filter is None:
                groups_for_record = _groups_from_key_scale(manifest, layer)
                if groups_for_record is None:
                    raise ValueError(
                        "--parallel_unit group requires calibrated manifest key_scale to infer KV groups. "
                        "Pass --groups explicitly, or run with --parallel_unit record."
                    )
            else:
                groups_for_record = sorted(group_filter)
            for group in groups_for_record:
                task = dict(base_task)
                task["task_groups"] = [int(group)]
                tasks.append(task)

    for task_i, task in enumerate(tasks, start=1):
        task["task_i"] = task_i
        task["task_count"] = len(tasks)

    def consume_result(result: dict[str, Any]) -> None:
        nonlocal processed_pairs
        processed_pairs += int(result["processed_pairs"])
        groups_seen.update(int(x) for x in result["groups_seen"])
        _merge_accumulator_map(by_layer_group, result["by_layer_group"])
        _merge_accumulator_map(overall, result["overall"])
        _merge_accumulator_map(single_cluster_bprime_baseline, result["single_cluster_bprime_baseline"])
        _merge_accumulator_map(
            vanilla_logkv_compressed_prefix_baseline,
            result["vanilla_logkv_compressed_prefix_baseline"],
        )
        _merge_accumulator_map(vanilla_logkv_full_cache_baseline, result["vanilla_logkv_full_cache_baseline"])
        if args.log_timing:
            for event in result.get("timing_events", []):
                print(_format_timing_event(event), flush=True)
                _add_timing_stat(timing_stats, event)
        print(result["message"], flush=True)

    if args.workers == 1:
        for task in tasks:
            consume_result(_process_record_task(task))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_to_record = {executor.submit(_process_record_task, task): task["record_i"] for task in tasks}
            for future in as_completed(future_to_record):
                consume_result(future.result())

    if processed_pairs == 0:
        # _iter_records already ruled out an empty --layers filter; getting here
        # means every (layer, group) pair the layer filter left standing was then
        # excluded by --groups, so overall_by_config/by_layer_group would
        # otherwise be silently written out empty with exit code 0.
        raise ValueError(
            f"--groups {sorted(group_filter) if group_filter is not None else group_filter} matched none "
            f"of the KV groups actually present in the dump ({sorted(groups_seen)}); "
            f"0 (layer, group) pairs were processed."
        )

    by_lg_rows = []
    for (g_label, l_block, layer, group), acc in sorted(
        by_layer_group.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])
    ):
        row = {
            "g_max": g_label,
            "l_block": l_block,
            "layer": layer,
            "group": group,
        }
        row.update(acc.finalize())
        by_lg_rows.append(row)

    overall_rows = []
    for (g_label, l_block), acc in sorted(overall.items(), key=lambda item: (item[0][0], item[0][1])):
        row = {"g_max": g_label, "l_block": l_block}
        row.update(acc.finalize())
        overall_rows.append(row)

    single_cluster_bprime_baseline_rows = []
    for (layer, group), acc in sorted(
        single_cluster_bprime_baseline.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        row = {"layer": layer, "group": group}
        row.update(acc.finalize())
        single_cluster_bprime_baseline_rows.append(row)

    vanilla_logkv_compressed_prefix_baseline_rows = []
    for (layer, group), acc in sorted(
        vanilla_logkv_compressed_prefix_baseline.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        row = {"layer": layer, "group": group}
        row.update(acc.finalize())
        vanilla_logkv_compressed_prefix_baseline_rows.append(row)

    vanilla_logkv_full_cache_baseline_rows = []
    for (layer, group), acc in sorted(
        vanilla_logkv_full_cache_baseline.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        row = {"layer": layer, "group": group}
        row.update(acc.finalize())
        vanilla_logkv_full_cache_baseline_rows.append(row)

    result = {
        "version": 1,
        "kind": "semantic_logkv_s0_0_sweep",
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
            "skip_value_var": bool(args.skip_value_var),
            "workers": args.workers,
            "parallel_unit": args.parallel_unit,
            "task_count": len(tasks),
            "routing_eta": 0.0,
            "routing_mode": "strict_serial_dpmeans_unclipped",
            "l_block_zero_forces_g_max_inf": True,
        },
        "overall_by_config_caveat": (
            "CLAUDE.md S2.5 / experiments.md require Stage 0 statistics to be reported "
            "per layer x KV-group, because key-vector scale (s_h) and value-vector scale "
            "(vh) can each differ by orders of magnitude across layers/groups "
            "(algorithm-spec.md S5.2). overall_by_config's token_weighted_key_var/value_var "
            "sum absolute SSE across every swept layer/group and can therefore be "
            "dominated by a single high-magnitude one, hiding the trend in all the "
            "others -- do not use them alone to make the S0.0 call. Use "
            "token_weighted_key_var_relative (normalized by each contributor's own "
            "calibrated s_h) / token_weighted_value_var_relative (normalized by each "
            "contributor's own calibrated vh, NOT s_h -- key and value scales are "
            "independent) for a scale-comparable cross-layer summary, or inspect "
            "by_layer_group directly. token_weighted_value_var/_relative and "
            "value_var_available/vh_source can be None/empty if any contributing sample "
            "had no value data (--skip_value_var) or no usable calibrated vh -- see "
            "SweepAccumulator's docstring."
        ),
        "overall_by_config": overall_rows,
        "by_layer_group": by_lg_rows,
        "single_cluster_bprime_baseline_note": (
            "NOT the vanilla/existing-LogKV reference -- see the two vanilla_logkv_*_baseline_"
            "note fields for that. This is a same-B' single-cluster position-order CONTROL: "
            "every token in one sequential cluster/segment (no semantic clustering, no "
            "g_max/l_block segmentation), run through the same simulate_segment_ladders b_prime "
            "bounded-carry construction as the semantic sweep cells (route_single_cluster_bprime_"
            "ladder), so it answers 'how much does semantic grouping itself help, holding the "
            "ladder budget/mechanics fixed to what the semantic sweep is already using' -- a "
            "real, separate question from S0.4's 'how does this compare to the currently "
            "deployed LogKV'. Independent of (g_max, l_block) -- reported once per "
            "(layer, group), not swept."
        ),
        "single_cluster_bprime_baseline_by_layer_group": single_cluster_bprime_baseline_rows,
        "vanilla_logkv_compressed_prefix_baseline_note": (
            "PRIMARY S0.4 reference: docs/experiments.md's decision gate ('key variance should "
            "be significantly lower than the existing position slot') should be judged against "
            "this one, not vanilla_logkv_full_cache_baseline -- see the latter's note for why. "
            "Built with vanilla_logkv_compressed_entries at --vanilla_B/--vanilla_recent_size "
            "(defaults 512/1024, matching the deployed exp/qwen1.7b-32k/base.yaml config): a "
            "real recent-window carve-out (the last vanilla_recent_size tokens are never "
            "compacted, exactly like LogStructuredKVCache.add_recent()) and real vanilla "
            "level-0 pairing (2 raw tokens pre-merged into one w=2 entry, exactly like "
            "log_kv_cache.py's _flush_pairs()) before the shared binary-carry ladder. Only "
            "covers the compacted prefix (compactable_token_count of token_count) -- the exact "
            "recent-window tokens are NOT represented here at all, not even as zero-variance "
            "entries, so real_token_count/entry_count read 0 whenever token_count <= "
            "vanilla_recent_size (correctly: nothing has been compressed yet, not 'zero "
            "variance'). compactable_token_count_mean/recent_count_mean/coverage_token_count_"
            "mean/covered_token_fraction in the row itself (see SweepAccumulator's docstring) "
            "make this coverage gap checkable directly instead of only inferrable from this "
            "note -- covered_token_fraction well below 1.0 here is expected and correct, not a "
            "bug, whenever token_count is not much larger than vanilla_recent_size. Validated "
            "end-to-end against a real LogStructuredKVCache -- see vanilla_logkv_compressed_"
            "entries's docstring and tests/test_semantic_s0.py. "
            "cluster_count_mean/segment_count_mean read 1.0/1.0 for every row: vanilla has no "
            "cluster/segment concept, those fields are a SweepAccumulator interface placeholder, "
            "not a measurement. Independent of (g_max, l_block) -- reported once per "
            "(layer, group), not swept."
        ),
        "vanilla_logkv_compressed_prefix_baseline_by_layer_group": vanilla_logkv_compressed_prefix_baseline_rows,
        "vanilla_logkv_full_cache_baseline_note": (
            "SECONDARY sanity check, not the primary S0.4 metric -- prefer "
            "vanilla_logkv_compressed_prefix_baseline for the actual decision gate. This is "
            "vanilla_logkv_compressed_prefix_baseline's compressed entries PLUS the "
            "recent_count exact recent-window tokens (up to vanilla_recent_size -- possibly "
            "fewer: token_count for prompts shorter than vanilla_recent_size, or "
            "vanilla_recent_size - 1 after an odd-overflow add_recent() carve-out, see "
            "vanilla_logkv_compressed_entries's docstring), each added back as its own trivial "
            "width-1, zero-variance entry (vanilla_logkv_full_cache_entries), so every position "
            "in the prompt is covered by exactly one entry -- useful for confirming total token "
            "coverage, or as an end-to-end 'whole attention state' reference. recent_count_mean/"
            "coverage_token_count_mean/covered_token_fraction in the row itself (see "
            "SweepAccumulator's docstring) make this checkable directly instead of only via this "
            "note; covered_token_fraction should read ~1.0 here. Not the primary metric because "
            "those ~recent_count structurally-zero-variance slots dilute the token-weighted "
            "variance average, more so the smaller token_count is relative to "
            "vanilla_recent_size -- this measures something closer to 'how much of the whole "
            "context is well-represented' than 'how good is compression itself'. Same "
            "cluster_count_mean/segment_count_mean placeholder caveat as the compressed-prefix "
            "baseline above. Independent of (g_max, l_block) -- reported once per (layer, "
            "group), not swept."
        ),
        "vanilla_logkv_full_cache_baseline_by_layer_group": vanilla_logkv_full_cache_baseline_rows,
        "timing_by_config": _timing_rows(timing_stats) if args.log_timing else [],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if args.log_timing:
        _print_timing_summary(result["timing_by_config"])
    print(f"[s0-sweep] wrote {args.output}")


if __name__ == "__main__":
    main()
