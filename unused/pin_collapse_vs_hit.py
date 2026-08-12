"""Join ``eval.py --log_kv_pin_score_diag_output`` and ``--log_kv_pin_diag_output``
JSON files by ``sample_id`` and check whether softmax-mass "collapse" onto
salience pins correlates with the pin actually landing near the NIAH needle.

Both diagnostics must come from the SAME eval run (same benchmark, same
checkpoint) so their ``sample_id`` values line up — see eval.py's
``_request_sample_id()``.

Usage:
    python unused/pin_collapse_vs_hit.py \\
        --pin-score-diag pin_score_diag_step_1400_..._20260812_115411.json \\
        --pin-diag pin_diag_step_1400_..._20260812_115411.json \\
        [--hit-mode near|exact] [--out-csv joined.csv]

Glob patterns are accepted for both paths (last match wins), so you can also
point at a directory pattern like './pin_score_diag/*.json'.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path_or_glob: str) -> dict[str, Any]:
    matches = sorted(glob.glob(path_or_glob))
    path = Path(matches[-1] if matches else path_or_glob).expanduser()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _task_from_sample_id(sample_id: str) -> str | None:
    for part in sample_id.split("|"):
        if part.startswith("task="):
            return part[len("task="):]
    return None


def _pin_diag_sample_summary(sample: dict[str, Any]) -> dict[str, Any]:
    """Reduce one pin_diag sample record (many layers x kv-groups) to
    per-sample hit-rate scalars.

    A sample's pins are selected independently per (layer, kv_group), so
    "did this sample hit" is not a single boolean in the raw record — it is
    aggregated here as a rate over every (layer, group) that actually had
    pins, which is what gets thresholded into hit/miss buckets below.
    """
    n_groups_with_pins = 0
    n_groups_near_hit = 0
    n_groups_exact_hit = 0
    min_distance: int | None = None
    for layer in sample.get("layers", []):
        if not layer.get("has_pin_indices"):
            continue
        for group in layer.get("groups", []):
            if not group.get("pin_count"):
                continue
            n_groups_with_pins += 1
            if group.get("near_hit_count", 0) > 0:
                n_groups_near_hit += 1
            if group.get("exact_hit_count", 0) > 0:
                n_groups_exact_hit += 1
            d = group.get("min_distance")
            if d is not None and (min_distance is None or d < min_distance):
                min_distance = d
    return {
        "n_groups_with_pins": n_groups_with_pins,
        "near_hit_rate": (n_groups_near_hit / n_groups_with_pins) if n_groups_with_pins else None,
        "exact_hit_rate": (n_groups_exact_hit / n_groups_with_pins) if n_groups_with_pins else None,
        "min_distance": min_distance,
        "comparable_needle_count": sample.get("comparable_needle_count", 0) or 0,
    }


def _describe(values: list[float]) -> dict[str, Any]:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    values.sort()
    n = len(values)
    return {
        "n": n,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values) if n > 1 else 0.0,
        "min": values[0],
        "max": values[-1],
    }


def _fmt(stats: dict[str, Any]) -> str:
    if stats["n"] == 0:
        return "n=0 (no data)"
    return (
        f"n={stats['n']} mean={stats['mean']:.4f} median={stats['median']:.4f} "
        f"stdev={stats['stdev']:.4f} min={stats['min']:.4f} max={stats['max']:.4f}"
    )


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pin-score-diag", required=True, help="path or glob to pin_score_diag_*.json")
    parser.add_argument("--pin-diag", required=True, help="path or glob to pin_diag_*.json")
    parser.add_argument(
        "--hit-mode", choices=["near", "exact"], default="near",
        help="which hit-rate defines the hit/miss split (default: near_hit_rate > 0)",
    )
    parser.add_argument("--out-csv", default=None, help="optional path to dump the joined per-sample table")
    parser.add_argument("--top", type=int, default=10, help="how many top-collapse samples to print per group")
    args = parser.parse_args()

    score_doc = _load_json(args.pin_score_diag)
    diag_doc = _load_json(args.pin_diag)

    score_root = score_doc.get("pin_score_diag", score_doc)
    diag_root = diag_doc.get("pin_diag", diag_doc)

    score_samples = {s["sample_id"]: s for s in score_root.get("samples", []) if s.get("sample_id")}
    diag_samples = {s["sample_id"]: s for s in diag_root.get("samples", []) if s.get("sample_id")}

    score_ids = set(score_samples)
    diag_ids = set(diag_samples)
    joined_ids = score_ids & diag_ids

    print(f"pin_score_diag samples: {len(score_ids)}")
    print(f"pin_diag samples:       {len(diag_ids)}")
    print(f"joined (intersection):  {len(joined_ids)}")
    only_score = score_ids - diag_ids
    only_diag = diag_ids - score_ids
    if only_score:
        print(f"  WARNING: {len(only_score)} sample(s) only in pin_score_diag, e.g. {sorted(only_score)[:3]}")
    if only_diag:
        print(f"  WARNING: {len(only_diag)} sample(s) only in pin_diag, e.g. {sorted(only_diag)[:3]}")
    if not joined_ids:
        print("No overlapping sample_id — nothing to analyze. Check both files come from the SAME eval run.")
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for sid in sorted(joined_ids):
        s = score_samples[sid]
        d = _pin_diag_sample_summary(diag_samples[sid])
        last_q = s.get("last_query") or {}
        rows.append({
            "sample_id": sid,
            "task": _task_from_sample_id(sid),
            "comparable_needle_count": d["comparable_needle_count"],
            "near_hit_rate": d["near_hit_rate"],
            "exact_hit_rate": d["exact_hit_rate"],
            "min_distance": d["min_distance"],
            "max_log10_ratio": s.get("max_pin_to_pooled_mass_per_slot_log10_ratio"),
            "max_ratio": s.get("max_pin_to_pooled_mass_per_slot_ratio"),
            "last_q_abs_pos": last_q.get("q_abs_pos"),
            "last_q_log10_ratio": last_q.get("max_pin_to_pooled_mass_per_slot_log10_ratio"),
            "last_q_mean_ratio": last_q.get("mean_pin_to_pooled_mass_per_slot_ratio"),
        })

    # 没有可比对针的样本（截断/未找到 needle）排除在 hit/miss 对比之外——它们的
    # hit_rate 恒为 None/0，混进 miss 组会制造假的"没命中"信号。
    no_needle = [r for r in rows if r["comparable_needle_count"] == 0]
    scoreable = [r for r in rows if r["comparable_needle_count"] > 0]

    hit_key = "near_hit_rate" if args.hit_mode == "near" else "exact_hit_rate"
    hit_rows = [r for r in scoreable if (r[hit_key] or 0) > 0]
    miss_rows = [r for r in scoreable if (r[hit_key] or 0) == 0]

    print()
    print(f"samples with no comparable needle (excluded from hit/miss): {len(no_needle)}")
    print(f"=== Grouped by {hit_key} > 0 (hit) vs == 0 (miss), n={len(scoreable)} scoreable samples ===")
    print(f"hit group:  n={len(hit_rows)}")
    print(f"miss group: n={len(miss_rows)}")

    for label, subset in (("HIT", hit_rows), ("MISS", miss_rows)):
        print(f"\n--- {label} group ---")
        for field in ("max_log10_ratio", "last_q_log10_ratio", "last_q_mean_ratio"):
            print(f"  {field}: {_fmt(_describe([r[field] for r in subset]))}")

    print()
    print("=== Per-task breakdown (all scoreable samples) ===")
    by_task: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for r in scoreable:
        by_task[r["task"]].append(r)
    for task, subset in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        hit_n = sum(1 for r in subset if (r[hit_key] or 0) > 0)
        stats = _describe([r["max_log10_ratio"] for r in subset])
        mean_str = f"{stats['mean']:.4f}" if stats["n"] else "n/a"
        print(f"  {task}: n={len(subset)} hit={hit_n} miss={len(subset) - hit_n} max_log10_ratio mean={mean_str}")

    xs = [r[hit_key] for r in scoreable if r[hit_key] is not None and r["max_log10_ratio"] is not None]
    ys = [r["max_log10_ratio"] for r in scoreable if r[hit_key] is not None and r["max_log10_ratio"] is not None]
    r_value = _pearson(xs, ys)
    print()
    if r_value is None:
        print(f"Pearson r({hit_key}, max_log10_ratio): not enough data")
    else:
        print(f"Pearson r({hit_key}, max_log10_ratio) = {r_value:.4f}  (n={len(xs)})")
        print("  接近 0：塌缩程度和是否命中针基本无关；正值：命中越准塌缩越猛；负值：命中越准反而越不塌缩。")

    for label, subset in (("HIT", hit_rows), ("MISS", miss_rows)):
        ranked = sorted(subset, key=lambda r: (r["max_log10_ratio"] if r["max_log10_ratio"] is not None else -math.inf), reverse=True)
        print(f"\n--- Top {args.top} most-collapsed samples in {label} group ---")
        for r in ranked[: args.top]:
            print(
                f"  {r['sample_id']}: max_log10_ratio={r['max_log10_ratio']:.3f} "
                f"{hit_key}={r[hit_key]:.2f} min_distance={r['min_distance']} "
                f"last_q_log10_ratio={r['last_q_log10_ratio']}"
            )

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nJoined per-sample table (all {len(rows)} samples, incl. no-needle) written to {args.out_csv}")


if __name__ == "__main__":
    main()
