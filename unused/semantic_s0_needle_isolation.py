#!/usr/bin/env python
"""Run the SemanticLogKV S0.3 needle-isolation analysis on a Stage-0 dump.

S0.3 asks whether the needle lands in a small semantic cluster. The decision
gate in docs/experiments.md is the fraction of needle tokens/spans whose
containing cluster has size <= B', compared with a same-length random-span
baseline from the same prompt.

Example:
    python unused/semantic_s0_needle_isolation.py \
      --dump stage0_dump \
      --output stage0_dump/s0_3_needle_isolation.json \
      --g_max inf,8192,4096,2048,1024,256 \
      --b_prime 8 \
      --random_trials 100
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

format_g_max = _S0.format_g_max
load_manifest = _S0.load_manifest
manifest_base_dir = _S0.manifest_base_dir
parse_g_max_list = _S0.parse_g_max_list
parse_int_list = _S0.parse_int_list
route_dpmeans_segments = _S0.route_dpmeans_segments


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


def _surviving_needle_intervals(sample: dict[str, Any], token_count: int) -> list[tuple[int, int]]:
    """Return needle intervals in this dump record's used-token coordinates.

    ``semantic_stage0_dump.py`` stores original prompt coordinates plus
    ``used_token_start``/``used_token_end`` after left truncation. S0.3 must
    analyze the latter, clamped to the actual ``k_raw`` length, because the
    route result is indexed in the model's used prompt window.
    """
    intervals: list[tuple[int, int]] = []
    offset = int(sample.get("prompt_token_offset") or 0)
    for span in sample.get("needle_spans", []):
        if not span.get("survived_left_truncation"):
            continue
        if "used_token_start" in span and "used_token_end" in span:
            start = int(span["used_token_start"])
            end = int(span["used_token_end"])
        else:
            start = int(span["token_start"]) - offset
            end = int(span["token_end"]) - offset
        start = max(0, start)
        end = min(int(token_count), end)
        if end > start:
            intervals.append((start, end))
    return _merge_intervals(intervals)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _cluster_sizes_for_interval(
    cluster_ids: np.ndarray,
    cluster_sizes: list[int],
    start: int,
    end: int,
) -> list[int]:
    return [int(cluster_sizes[int(cluster_ids[pos])]) for pos in range(int(start), int(end))]


class NeedleIsolationAccumulator:
    def __init__(self, *, b_prime: int) -> None:
        self.b_prime = int(b_prime)
        self.sample_groups = 0
        self.sample_groups_with_needle = 0
        self.span_count = 0
        self.needle_token_count = 0
        self.needle_token_isolated = 0
        self.span_all_isolated = 0
        self.span_any_isolated = 0
        self.random_span_count = 0
        self.random_token_count = 0
        self.random_token_isolated = 0
        self.random_span_all_isolated = 0
        self.random_span_any_isolated = 0
        self.cluster_size_values: list[float] = []
        self.random_cluster_size_values: list[float] = []
        self.cluster_count_sum = 0.0
        self.segment_count_sum = 0.0
        self.sh_sources: set[str] = set()

    def add_route_meta(self, *, cluster_count: int, segment_count: int, sh_source: str) -> None:
        self.sample_groups += 1
        self.cluster_count_sum += int(cluster_count)
        self.segment_count_sum += int(segment_count)
        self.sh_sources.add(sh_source)

    def add_intervals(
        self,
        *,
        intervals: list[tuple[int, int]],
        cluster_ids: np.ndarray,
        cluster_sizes: list[int],
    ) -> None:
        if not intervals:
            return
        self.sample_groups_with_needle += 1
        for start, end in intervals:
            sizes = _cluster_sizes_for_interval(cluster_ids, cluster_sizes, start, end)
            if not sizes:
                continue
            isolated = [size <= self.b_prime for size in sizes]
            self.span_count += 1
            self.needle_token_count += len(sizes)
            self.needle_token_isolated += sum(1 for value in isolated if value)
            self.span_all_isolated += int(all(isolated))
            self.span_any_isolated += int(any(isolated))
            self.cluster_size_values.extend(float(size) for size in sizes)

    def add_random_intervals(
        self,
        *,
        intervals: list[tuple[int, int]],
        cluster_ids: np.ndarray,
        cluster_sizes: list[int],
    ) -> None:
        for start, end in intervals:
            sizes = _cluster_sizes_for_interval(cluster_ids, cluster_sizes, start, end)
            if not sizes:
                continue
            isolated = [size <= self.b_prime for size in sizes]
            self.random_span_count += 1
            self.random_token_count += len(sizes)
            self.random_token_isolated += sum(1 for value in isolated if value)
            self.random_span_all_isolated += int(all(isolated))
            self.random_span_any_isolated += int(any(isolated))
            self.random_cluster_size_values.extend(float(size) for size in sizes)

    def merge(self, other: "NeedleIsolationAccumulator") -> None:
        self.sample_groups += other.sample_groups
        self.sample_groups_with_needle += other.sample_groups_with_needle
        self.span_count += other.span_count
        self.needle_token_count += other.needle_token_count
        self.needle_token_isolated += other.needle_token_isolated
        self.span_all_isolated += other.span_all_isolated
        self.span_any_isolated += other.span_any_isolated
        self.random_span_count += other.random_span_count
        self.random_token_count += other.random_token_count
        self.random_token_isolated += other.random_token_isolated
        self.random_span_all_isolated += other.random_span_all_isolated
        self.random_span_any_isolated += other.random_span_any_isolated
        self.cluster_size_values.extend(other.cluster_size_values)
        self.random_cluster_size_values.extend(other.random_cluster_size_values)
        self.cluster_count_sum += other.cluster_count_sum
        self.segment_count_sum += other.segment_count_sum
        self.sh_sources.update(other.sh_sources)

    def finalize(self) -> dict[str, Any]:
        token_rate = _rate(self.needle_token_isolated, self.needle_token_count)
        random_token_rate = _rate(self.random_token_isolated, self.random_token_count)
        span_all_rate = _rate(self.span_all_isolated, self.span_count)
        random_span_all_rate = _rate(self.random_span_all_isolated, self.random_span_count)
        span_any_rate = _rate(self.span_any_isolated, self.span_count)
        random_span_any_rate = _rate(self.random_span_any_isolated, self.random_span_count)
        denom = max(self.sample_groups, 1)
        return {
            "sample_groups": int(self.sample_groups),
            "sample_groups_with_needle": int(self.sample_groups_with_needle),
            "span_count": int(self.span_count),
            "needle_token_count": int(self.needle_token_count),
            "needle_token_isolated_count": int(self.needle_token_isolated),
            "needle_token_isolated_rate": token_rate,
            "span_all_tokens_isolated_count": int(self.span_all_isolated),
            "span_all_tokens_isolated_rate": span_all_rate,
            "span_any_token_isolated_count": int(self.span_any_isolated),
            "span_any_token_isolated_rate": span_any_rate,
            "cluster_size_mean": _mean(self.cluster_size_values),
            "cluster_size_quantiles": _quantiles(self.cluster_size_values),
            "cluster_size_max": max(self.cluster_size_values) if self.cluster_size_values else None,
            "random_span_count": int(self.random_span_count),
            "random_token_count": int(self.random_token_count),
            "random_token_isolated_count": int(self.random_token_isolated),
            "random_token_isolated_rate": random_token_rate,
            "random_span_all_tokens_isolated_count": int(self.random_span_all_isolated),
            "random_span_all_tokens_isolated_rate": random_span_all_rate,
            "random_span_any_token_isolated_count": int(self.random_span_any_isolated),
            "random_span_any_token_isolated_rate": random_span_any_rate,
            "random_cluster_size_mean": _mean(self.random_cluster_size_values),
            "random_cluster_size_quantiles": _quantiles(self.random_cluster_size_values),
            "token_isolation_lift": _lift(token_rate, random_token_rate),
            "span_all_tokens_isolation_lift": _lift(span_all_rate, random_span_all_rate),
            "span_any_token_isolation_lift": _lift(span_any_rate, random_span_any_rate),
            "cluster_count_mean": self.cluster_count_sum / denom,
            "segment_count_mean": self.segment_count_sum / denom,
            "sh_source": sorted(self.sh_sources),
        }


def _rate(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def _lift(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return value / baseline


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantiles(values: list[float], qs: tuple[float, ...] = (0.5, 0.9, 0.99)) -> dict[str, float | None]:
    if not values:
        return {f"p{int(q * 100):02d}": None for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


def _draw_random_intervals(
    intervals: list[tuple[int, int]],
    *,
    token_count: int,
    rng: np.random.Generator,
    trials: int,
) -> list[tuple[int, int]]:
    if trials <= 0 or token_count <= 0:
        return []
    out: list[tuple[int, int]] = []
    for start, end in intervals:
        length = min(int(end) - int(start), int(token_count))
        if length <= 0:
            continue
        high_exclusive = int(token_count) - length + 1
        for _ in range(int(trials)):
            random_start = int(rng.integers(0, high_exclusive))
            out.append((random_start, random_start + length))
    return out


def _rng_for_record_group(*, seed: int, record_index: int, layer: int, group: int) -> np.random.Generator:
    words = [
        int(seed) & 0xFFFFFFFF,
        (int(seed) >> 32) & 0xFFFFFFFF,
        int(record_index) & 0xFFFFFFFF,
        int(layer) & 0xFFFFFFFF,
        int(group) & 0xFFFFFFFF,
    ]
    return np.random.default_rng(np.random.SeedSequence(words))


def _merge_accumulator_map(
    dst: dict[Any, NeedleIsolationAccumulator],
    src: dict[Any, NeedleIsolationAccumulator],
    *,
    b_prime: int,
) -> None:
    for key, acc in src.items():
        dst.setdefault(key, NeedleIsolationAccumulator(b_prime=b_prime)).merge(acc)


def _process_record_task(task: dict[str, Any]) -> dict[str, Any]:
    sample = task["sample"]
    record = task["record"]
    layer = int(record["layer"])
    record_i = int(task["record_i"])
    record_count = int(task["record_count"])
    record_index = int(task["record_index"])
    base_dir = Path(task["base_dir"])
    g_values = [float(x) for x in task["g_values"]]
    group_filter = None if task["group_filter"] is None else set(int(x) for x in task["group_filter"])
    b_prime = int(task["b_prime"])
    collect_by_layer_group = bool(task["collect_by_layer_group"])

    overall: dict[str, NeedleIsolationAccumulator] = {}
    by_layer_group: dict[tuple[str, int, int], NeedleIsolationAccumulator] = {}
    processed_pairs = 0
    comparable_pairs = 0
    records_without_surviving_needle = 0

    path = base_dir / record["path"]
    with np.load(path) as payload:
        k_raw = payload["k_raw"]

    groups = list(range(k_raw.shape[0]))
    groups_seen = set(groups)
    if group_filter is not None:
        groups = [g for g in groups if g in group_filter]
    matched_group_pairs = len(groups)

    intervals = _surviving_needle_intervals(sample, int(k_raw.shape[1]))
    if not intervals:
        records_without_surviving_needle = 1
        message = (
            f"[s0-needle] {record_i}/{record_count} sample={sample.get('sample_id')} "
            f"layer={layer} skipped_no_surviving_needle"
        )
        return {
            "overall": overall,
            "by_layer_group": by_layer_group,
            "processed_pairs": processed_pairs,
            "matched_group_pairs": matched_group_pairs,
            "comparable_pairs": comparable_pairs,
            "groups_seen": groups_seen,
            "records_without_surviving_needle": records_without_surviving_needle,
            "message": message,
        }

    for group in groups:
        processed_pairs += 1
        comparable_pairs += 1
        k_group = np.asarray(k_raw[group], dtype=np.float32)
        sh, sh_source = _manifest_sh(
            task["scale_manifest"],
            layer,
            int(group),
            k_group,
            allow_fallback=bool(task["allow_fallback_sh"]),
        )
        lambda_new = float(task["lambda_rel"]) * sh
        route_cache: dict[str, Any] = {}
        random_intervals = _draw_random_intervals(
            intervals,
            token_count=int(k_group.shape[0]),
            rng=_rng_for_record_group(
                seed=int(task["seed"]),
                record_index=record_index,
                layer=layer,
                group=int(group),
            ),
            trials=int(task["random_trials"]),
        )
        for g_max in g_values:
            g_label = format_g_max(g_max)
            route = route_cache.get(g_label)
            if route is None:
                route = route_dpmeans_segments(
                    k_group,
                    lambda_new=lambda_new,
                    g_max=g_max,
                    gamma=float(task["seg_forget"]),
                )
                route_cache[g_label] = route

            accs = [
                overall.setdefault(g_label, NeedleIsolationAccumulator(b_prime=b_prime)),
            ]
            if collect_by_layer_group:
                accs.append(
                    by_layer_group.setdefault(
                        (g_label, layer, int(group)),
                        NeedleIsolationAccumulator(b_prime=b_prime),
                    )
                )
            for acc in accs:
                acc.add_route_meta(
                    cluster_count=route.cluster_count,
                    segment_count=route.segment_count,
                    sh_source=sh_source,
                )
                acc.add_intervals(
                    intervals=intervals,
                    cluster_ids=route.cluster_ids,
                    cluster_sizes=route.cluster_sizes,
                )
                acc.add_random_intervals(
                    intervals=random_intervals,
                    cluster_ids=route.cluster_ids,
                    cluster_sizes=route.cluster_sizes,
                )

    message = (
        f"[s0-needle] {record_i}/{record_count} sample={sample.get('sample_id')} "
        f"layer={layer} path={record['path']}"
    )
    return {
        "overall": overall,
        "by_layer_group": by_layer_group,
        "processed_pairs": processed_pairs,
        "matched_group_pairs": matched_group_pairs,
        "comparable_pairs": comparable_pairs,
        "groups_seen": groups_seen,
        "records_without_surviving_needle": records_without_surviving_needle,
        "message": message,
    }


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_num(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.3g}"


def _print_summary(rows: list[dict[str, Any]], *, top: int) -> None:
    print("== S0.3 needle isolation ==")
    print(
        "口径: needle token/span 所在 semantic cluster 的成员数 <= B'；"
        "random 为同 prompt 等长随机 span。"
    )
    print("注意: 这是 cluster-level 判据，l_block 不参与路由；需要扫的是 g_max/lambda_rel。")
    if not rows:
        print("no rows")
        return
    print(
        "g_max  token_iso  random  lift  span_all  random  lift  "
        "cluster_p50  cluster_p90  clusters  segments  n"
    )
    print(
        "-----  ---------  ------  ----  --------  ------  ----  "
        "-----------  -----------  --------  --------  -"
    )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("token_isolation_lift") or -1.0),
            -float(row.get("needle_token_isolated_rate") or -1.0),
        ),
    )
    for row in ranked[: max(int(top), 1)]:
        q = row.get("cluster_size_quantiles") or {}
        print(
            f"{str(row['g_max']).ljust(5)}  "
            f"{_fmt_pct(row.get('needle_token_isolated_rate')).rjust(9)}  "
            f"{_fmt_pct(row.get('random_token_isolated_rate')).rjust(6)}  "
            f"{_fmt_num(row.get('token_isolation_lift')).rjust(4)}  "
            f"{_fmt_pct(row.get('span_all_tokens_isolated_rate')).rjust(8)}  "
            f"{_fmt_pct(row.get('random_span_all_tokens_isolated_rate')).rjust(6)}  "
            f"{_fmt_num(row.get('span_all_tokens_isolation_lift')).rjust(4)}  "
            f"{_fmt_num(q.get('p50')).rjust(11)}  "
            f"{_fmt_num(q.get('p90')).rjust(11)}  "
            f"{_fmt_num(row.get('cluster_count_mean')).rjust(8)}  "
            f"{_fmt_num(row.get('segment_count_mean')).rjust(8)}  "
            f"{row.get('sample_groups')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="Stage-0 dump directory or manifest.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--g_max", default="inf,8192,4096,2048,1024,256")
    parser.add_argument("--lambda_rel", type=float, default=1.0)
    parser.add_argument("--seg_forget", type=float, default=0.5)
    parser.add_argument("--b_prime", type=int, default=8)
    parser.add_argument("--layers", help="Optional comma-separated layer filter")
    parser.add_argument("--groups", help="Optional comma-separated KV-group filter")
    parser.add_argument("--random_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes over layer records. Use 1 for deterministic single-process execution.",
    )
    parser.add_argument("--top", type=int, default=12, help="Rows to print in the terminal summary")
    parser.add_argument(
        "--allow_fallback_sh",
        action="store_true",
        help="Use this prompt's local key variance if calibrated manifest key_scale is missing/corrupt.",
    )
    parser.add_argument(
        "--skip_by_layer_group",
        action="store_true",
        help="Omit per-(g_max, layer, group) rows and write only overall_by_config.",
    )
    args = parser.parse_args()

    if not (math.isfinite(args.lambda_rel) and args.lambda_rel > 0.0):
        raise ValueError(f"--lambda_rel must be finite and > 0, got {args.lambda_rel}")
    if not (math.isfinite(args.seg_forget) and 0.0 <= args.seg_forget <= 1.0):
        raise ValueError(f"--seg_forget must be in [0, 1], got {args.seg_forget}")
    if args.b_prime <= 0:
        raise ValueError(f"--b_prime must be a positive integer, got {args.b_prime}")
    if args.random_trials < 0:
        raise ValueError(f"--random_trials must be >= 0, got {args.random_trials}")
    if args.workers <= 0:
        raise ValueError(f"--workers must be a positive integer, got {args.workers}")

    manifest = load_manifest(args.dump)
    base_dir = manifest_base_dir(args.dump)
    g_values = parse_g_max_list(args.g_max)
    if not g_values:
        raise ValueError(f"--g_max {args.g_max!r} parsed to an empty list -- pass at least one value")
    bad_g_max = [g for g in g_values if not (g == math.inf or (math.isfinite(g) and g >= 0.0))]
    if bad_g_max:
        raise ValueError(f"--g_max values must be finite >= 0, or inf, got {bad_g_max}")
    layer_filter = _wanted(args.layers)
    group_filter = _wanted(args.groups)

    records = _iter_records(manifest, layer_filter=layer_filter)
    if not records:
        raise ValueError("No layer records matched the requested filters")

    overall: dict[str, NeedleIsolationAccumulator] = {}
    by_layer_group: dict[tuple[str, int, int], NeedleIsolationAccumulator] = {}
    processed_pairs = 0
    matched_group_pairs = 0
    comparable_pairs = 0
    groups_seen: set[int] = set()
    records_without_surviving_needle = 0
    scale_manifest = {"key_scale": manifest.get("key_scale", {})}
    tasks = [
        {
            "sample": sample,
            "record": record,
            "record_i": record_i,
            "record_index": record_i - 1,
            "record_count": len(records),
            "base_dir": str(base_dir),
            "scale_manifest": scale_manifest,
            "g_values": g_values,
            "group_filter": None if group_filter is None else sorted(group_filter),
            "lambda_rel": args.lambda_rel,
            "seg_forget": args.seg_forget,
            "b_prime": args.b_prime,
            "random_trials": args.random_trials,
            "seed": args.seed,
            "allow_fallback_sh": args.allow_fallback_sh,
            "collect_by_layer_group": not args.skip_by_layer_group,
        }
        for record_i, (sample, record) in enumerate(records, start=1)
    ]

    def consume_result(result: dict[str, Any]) -> None:
        nonlocal processed_pairs
        nonlocal matched_group_pairs
        nonlocal comparable_pairs
        nonlocal records_without_surviving_needle
        processed_pairs += int(result["processed_pairs"])
        matched_group_pairs += int(result["matched_group_pairs"])
        comparable_pairs += int(result["comparable_pairs"])
        records_without_surviving_needle += int(result["records_without_surviving_needle"])
        groups_seen.update(int(x) for x in result["groups_seen"])
        _merge_accumulator_map(overall, result["overall"], b_prime=args.b_prime)
        if not args.skip_by_layer_group:
            _merge_accumulator_map(by_layer_group, result["by_layer_group"], b_prime=args.b_prime)
        print(result["message"], flush=True)

    if args.workers == 1:
        for task in tasks:
            consume_result(_process_record_task(task))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_to_record = {executor.submit(_process_record_task, task): task["record_i"] for task in tasks}
            for future in as_completed(future_to_record):
                consume_result(future.result())

    if matched_group_pairs == 0:
        raise ValueError(
            f"--groups {sorted(group_filter) if group_filter is not None else group_filter} matched none "
            f"of the KV groups actually present in the dump ({sorted(groups_seen)}); "
            f"0 (layer, group) pairs were processed."
        )
    if comparable_pairs == 0 or not overall:
        raise ValueError(
            "No comparable needle spans were found in the selected records. "
            "Use a dump made from NIAH samples and, preferably, dump with --require_needle_span."
        )

    overall_rows = []
    for g_label, acc in sorted(overall.items(), key=lambda item: item[0]):
        row = {"g_max": g_label}
        row.update(acc.finalize())
        overall_rows.append(row)

    by_lg_rows = []
    for (g_label, layer, group), acc in sorted(by_layer_group.items(), key=lambda item: item[0]):
        row = {"g_max": g_label, "layer": layer, "group": group}
        row.update(acc.finalize())
        by_lg_rows.append(row)

    result = {
        "version": 1,
        "kind": "semantic_logkv_s0_3_needle_isolation",
        "source_manifest": str(Path(args.dump)),
        "config": {
            "lambda_rel": args.lambda_rel,
            "seg_forget": args.seg_forget,
            "b_prime": args.b_prime,
            "g_max": [format_g_max(x) for x in g_values],
            "layers": None if layer_filter is None else sorted(layer_filter),
            "groups": None if group_filter is None else sorted(group_filter),
            "random_trials": args.random_trials,
            "seed": args.seed,
            "workers": args.workers,
            "routing_eta": 0.0,
            "routing_mode": "strict_serial_dpmeans_unclipped",
            "random_baseline_rng": "per_record_layer_group_seedsequence_v1",
            "l_block_caveat": (
                "S0.3 is a cluster-membership metric. l_block only affects the later segment/ladder "
                "packing path, not route_dpmeans_segments cluster_ids, so this analyzer reports by g_max."
            ),
        },
        "summary": {
            "record_count": len(records),
            "processed_layer_group_pairs": processed_pairs,
            "comparable_layer_group_pairs": comparable_pairs,
            "records_without_surviving_needle": records_without_surviving_needle,
        },
        "overall_by_config": overall_rows,
        "by_layer_group": [] if args.skip_by_layer_group else by_lg_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _print_summary(overall_rows, top=args.top)
    print(f"[s0-needle] wrote {args.output}")


if __name__ == "__main__":
    main()
