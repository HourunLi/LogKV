"""Summarize ``eval.py --log_kv_pin_diag_output`` JSON files.

Usage:
    python unused/pin_diag.py ./pin_diag_smoke/pin_diag_xxx.json
    python unused/pin_diag.py './pin_diag_smoke/*.json' --band 23-26

The script reports:
  1. sample-level hit rates: whether any layer/group hits the needle;
  2. sample-level hit rates inside a selected layer band;
  3. the recorded group-level rates from the JSON summary;
  4. full ``pin_indices`` spatial spread when the JSON was recorded with
     ``--log_kv_pin_diag_include_indices``;
  5. a random-pin baseline with the same pin_size/recent_size/radius.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path
from typing import Any


def _load_json(path_or_glob: str) -> tuple[Path, dict[str, Any]]:
    matches = sorted(glob.glob(path_or_glob))
    path = Path(matches[-1] if matches else path_or_glob).expanduser()
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return path, payload


def _pin_diag(payload: dict[str, Any]) -> dict[str, Any]:
    if "pin_diag" in payload:
        return payload["pin_diag"]
    return payload


def _config(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("config", {})


def _pct(num: int | float | None, den: int | float | None = None) -> str:
    if den is not None:
        if den == 0:
            return "n/a"
        value = float(num or 0) / float(den)
    else:
        if num is None:
            return "n/a"
        value = float(num)
    return f"{value:.1%}"


def _parse_layer_spec(spec: str) -> set[int]:
    layers: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                start, end = end, start
            layers.update(range(start, end + 1))
        else:
            layers.add(int(chunk))
    return layers


def _parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values


def _has_group_hit(sample: dict[str, Any], key: str, layers: set[int] | None = None) -> bool:
    for layer in sample.get("layers", []):
        if not layer.get("has_pin_indices"):
            continue
        if layers is not None and int(layer.get("layer", -1)) not in layers:
            continue
        if any(group.get(key, 0) > 0 for group in layer.get("groups", [])):
            return True
    return False


def _distance_to_spans(index: int, spans: list[dict[str, Any]]) -> int | None:
    best: int | None = None
    for span in spans:
        token_start = int(span["token_start"])
        token_end = int(span["token_end"])
        if token_start <= index < token_end:
            distance = 0
        elif index < token_start:
            distance = token_start - index
        else:
            distance = index - (token_end - 1)
        best = distance if best is None else min(best, distance)
    return best


def _iter_comparable_samples(pin_diag: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        sample
        for sample in pin_diag.get("samples", [])
        if sample.get("comparable_needle_count", 0) > 0
    ]


def print_sample_hit_summary(pin_diag: dict[str, Any], band_layers: set[int]) -> None:
    samples = _iter_comparable_samples(pin_diag)
    total = len(samples)

    exact_any = sum(_has_group_hit(sample, "exact_hit_count") for sample in samples)
    near_any = sum(_has_group_hit(sample, "near_hit_count") for sample in samples)
    exact_band = sum(_has_group_hit(sample, "exact_hit_count", band_layers) for sample in samples)
    near_band = sum(_has_group_hit(sample, "near_hit_count", band_layers) for sample in samples)

    band_text = _format_layers(band_layers)
    print("== Sample-level hit rates ==")
    print(f"有效样本数: {total}")
    print(f"任意层/组至少一处精确命中: {exact_any}/{total} = {_pct(exact_any, total)}")
    print(f"任意层/组至少一处 radius 内命中: {near_any}/{total} = {_pct(near_any, total)}")
    print(f"{band_text} 层里至少一处精确命中: {exact_band}/{total} = {_pct(exact_band, total)}")
    print(f"{band_text} 层里至少一处 radius 内命中: {near_band}/{total} = {_pct(near_band, total)}")


def _iter_groups(sample: dict[str, Any], layers: set[int] | None = None):
    for layer in sample.get("layers", []):
        if not layer.get("has_pin_indices"):
            continue
        if layers is not None and int(layer.get("layer", -1)) not in layers:
            continue
        for group in layer.get("groups", []):
            yield layer, group


def print_compressed_slot_summary(pin_diag: dict[str, Any], band_layers: set[int]) -> None:
    samples = _iter_comparable_samples(pin_diag)
    total = len(samples)
    any_compressed = 0
    any_recent = 0
    band_compressed = 0
    group_trials = 0
    group_compressed = 0
    widths: list[int] = []
    levels: dict[int, int] = {}

    for sample in samples:
        sample_has_compressed = False
        sample_has_recent = False
        sample_band_has_compressed = False
        for layer, group in _iter_groups(sample):
            covering = group.get("needle_covering_slots", [])
            compressed = [slot for slot in covering if slot.get("is_compressed")]
            recent = [slot for slot in covering if slot.get("kind") == "recent"]
            if compressed:
                sample_has_compressed = True
                if int(layer.get("layer", -1)) in band_layers:
                    sample_band_has_compressed = True
            if recent:
                sample_has_recent = True
            if covering:
                group_trials += 1
                if compressed:
                    group_compressed += 1
            for slot in compressed:
                widths.append(int(slot.get("slot_width", 0) or 0))
                level = slot.get("level")
                if level is not None:
                    levels[int(level)] = levels.get(int(level), 0) + 1

        any_compressed += sample_has_compressed
        any_recent += sample_has_recent
        band_compressed += sample_band_has_compressed

    band_text = _format_layers(band_layers)
    print("\n== Needle compressed-slot coverage ==")
    print(f"任意层/组 needle 落在压缩 level slot: {any_compressed}/{total} = {_pct(any_compressed, total)}")
    print(f"任意层/组 needle 仍在 recent exact slot: {any_recent}/{total} = {_pct(any_recent, total)}")
    print(f"{band_text} 层里 needle 落在压缩 level slot: {band_compressed}/{total} = {_pct(band_compressed, total)}")
    print(f"group-level compressed coverage: {group_compressed}/{group_trials} = {_pct(group_compressed, group_trials)}")
    if widths:
        widths_sorted = sorted(widths)
        median_width = widths_sorted[len(widths_sorted) // 2]
        print(f"compressed slot width: min={min(widths)} median={median_width} max={max(widths)}")
    if levels:
        level_text = ", ".join(f"L{level}:{count}" for level, count in sorted(levels.items()))
        print(f"compressed levels covering needle: {level_text}")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return ordered[left]
    frac = pos - left
    return ordered[left] * (1.0 - frac) + ordered[right] * frac


def _fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _candidate_bounds(sample: dict[str, Any], config: dict[str, Any], recent_size: int | None) -> tuple[int, int] | None:
    prompt_tokens = sample.get("prompt_tokens", {})
    used_tokens = prompt_tokens.get("used")
    if used_tokens is None:
        return None
    offset = int(prompt_tokens.get("left_truncated_tokens") or 0)
    actual_recent_size = int(recent_size if recent_size is not None else config.get("log_kv_recent_size", 1024))
    candidate_count = int(used_tokens) - actual_recent_size
    if candidate_count <= 0:
        return None
    return offset, offset + candidate_count


def _pin_histogram(pins: list[int], start: int, end: int, bins: int) -> list[int]:
    counts = [0] * bins
    width = max(1, end - start)
    for pin in pins:
        if pin < start or pin >= end:
            continue
        idx = min(bins - 1, int((pin - start) * bins / width))
        counts[idx] += 1
    return counts


def _hist_entropy_norm(counts: list[int]) -> float | None:
    total = sum(counts)
    nonzero = [c for c in counts if c > 0]
    if total <= 0 or len(counts) <= 1:
        return None
    entropy = -sum((c / total) * math.log(c / total) for c in nonzero)
    return entropy / math.log(len(counts))


def _max_window(pins: list[int], window: int) -> tuple[int, int, int]:
    if not pins:
        return 0, 0, 0
    best_count = 0
    best_start = pins[0]
    right = 0
    for left, start in enumerate(pins):
        while right < len(pins) and pins[right] < start + window:
            right += 1
        count = right - left
        if count > best_count:
            best_count = count
            best_start = start
    return best_count, best_start, best_start + window


def _pin_records(
    pin_diag: dict[str, Any],
    config: dict[str, Any],
    *,
    layers: set[int] | None,
    bins: int,
    windows: list[int],
    recent_size: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_i, sample in enumerate(pin_diag.get("samples", [])):
        bounds = _candidate_bounds(sample, config, recent_size)
        for layer, group in _iter_groups(sample, layers):
            raw_pins = group.get("pin_indices")
            if not raw_pins:
                continue
            pins = sorted(int(pin) for pin in raw_pins)
            unique_pins = sorted(set(pins))
            if bounds is None:
                start = min(unique_pins)
                end = max(unique_pins) + 1
            else:
                start, end = bounds
            candidate_count = max(1, end - start)
            pin_span = max(unique_pins) - min(unique_pins) + 1
            gaps = [b - a for a, b in zip(unique_pins, unique_pins[1:])]
            hist = _pin_histogram(unique_pins, start, end, bins)
            occupied_bins = sum(1 for count in hist if count > 0)
            max_bin_count = max(hist) if hist else 0
            window_stats = {
                window: _max_window(unique_pins, window)
                for window in windows
                if window > 0
            }
            records.append(
                {
                    "sample": sample_i,
                    "layer": int(layer.get("layer", -1)),
                    "batch": group.get("batch"),
                    "kv_group": group.get("kv_group"),
                    "pin_count": len(pins),
                    "unique_pin_count": len(unique_pins),
                    "candidate_start": start,
                    "candidate_end": end,
                    "candidate_count": candidate_count,
                    "min_pin": min(unique_pins),
                    "max_pin": max(unique_pins),
                    "pin_span": pin_span,
                    "pin_span_frac": pin_span / candidate_count,
                    "gap_min": min(gaps) if gaps else None,
                    "gap_median": _median([float(g) for g in gaps]),
                    "gap_p90": _quantile([float(g) for g in gaps], 0.90),
                    "gap_max": max(gaps) if gaps else None,
                    "occupied_bins": occupied_bins,
                    "occupied_bin_frac": occupied_bins / bins if bins else None,
                    "max_bin_count": max_bin_count,
                    "max_bin_frac": max_bin_count / len(unique_pins) if unique_pins else None,
                    "hist_entropy_norm": _hist_entropy_norm(hist),
                    "histogram": hist,
                    "window_stats": window_stats,
                }
            )
    return records


def _ascii_hist(counts: list[int], width: int = 48) -> str:
    if not counts:
        return ""
    max_count = max(counts)
    if max_count <= 0:
        return " ".join("0" for _ in counts)
    bars = []
    for count in counts:
        bar_len = max(1, round(count / max_count * width)) if count > 0 else 0
        bars.append("#" * bar_len if bar_len else ".")
    return " ".join(bars)


def print_pin_distribution_summary(
    pin_diag: dict[str, Any],
    config: dict[str, Any],
    *,
    band_layers: set[int],
    bins: int,
    windows: list[int],
    recent_size: int | None,
    top_groups: int,
) -> None:
    records = _pin_records(
        pin_diag,
        config,
        layers=band_layers,
        bins=bins,
        windows=windows,
        recent_size=recent_size,
    )
    print("\n== Pin spatial distribution ==")
    if not records:
        print("没有找到 group.pin_indices；需要重新跑诊断并加 --log_kv_pin_diag_include_indices。")
        return

    span_fracs = [float(r["pin_span_frac"]) for r in records]
    occupied_fracs = [float(r["occupied_bin_frac"]) for r in records if r["occupied_bin_frac"] is not None]
    entropy_values = [float(r["hist_entropy_norm"]) for r in records if r["hist_entropy_norm"] is not None]
    gap_medians = [float(r["gap_median"]) for r in records if r["gap_median"] is not None]
    max_bin_fracs = [float(r["max_bin_frac"]) for r in records if r["max_bin_frac"] is not None]

    print(f"分析层段: {_format_layers(band_layers)}")
    print(f"含完整 pin_indices 的 group 数: {len(records)}")
    print(
        "pin 覆盖跨度 / candidate_horizon: "
        f"median={_pct(_median(span_fracs))} mean={_pct(_mean(span_fracs))}"
    )
    print(
        f"{bins} bins 占用比例: "
        f"median={_pct(_median(occupied_fracs))} mean={_pct(_mean(occupied_fracs))}"
    )
    print(
        "hist entropy(normalized): "
        f"median={_fmt_num(_median(entropy_values), 3)} mean={_fmt_num(_mean(entropy_values), 3)}"
    )
    print(
        "相邻 pin gap: "
        f"median_group_median={_fmt_num(_median(gap_medians))} "
        f"mean_group_median={_fmt_num(_mean(gap_medians))}"
    )
    print(
        "单个 bin 最大 pin 占比: "
        f"median={_pct(_median(max_bin_fracs))} mean={_pct(_mean(max_bin_fracs))}"
    )

    for window in windows:
        fracs = []
        counts = []
        for record in records:
            stat = record["window_stats"].get(window)
            if stat is None:
                continue
            count, _start, _end = stat
            counts.append(float(count))
            fracs.append(count / max(1, int(record["unique_pin_count"])))
        print(
            f"最密 {window}-token 窗口: "
            f"median_count={_fmt_num(_median(counts))} "
            f"median_pin_frac={_pct(_median(fracs))} "
            f"mean_pin_frac={_pct(_mean(fracs))}"
        )

    # A compact heuristic, meant to flag the thing worth eyeballing rather than
    # prove a statistical property.
    primary_window = 128 if 128 in windows else (windows[0] if windows else None)
    if primary_window is not None:
        primary_fracs = [
            record["window_stats"][primary_window][0] / max(1, int(record["unique_pin_count"]))
            for record in records
            if primary_window in record["window_stats"]
        ]
        clustered = (_median(primary_fracs) or 0.0) >= 0.35 or (_median(occupied_fracs) or 1.0) <= 0.35
        print(
            "粗判: "
            + (
                "存在明显抱团迹象，建议看下面 top groups 的 histogram。"
                if clustered
                else "没有明显整体抱团；仍可看 top groups 是否有局部异常。"
            )
        )

    ranked = sorted(
        records,
        key=lambda r: (
            -(max((stat[0] / max(1, int(r["unique_pin_count"]))) for stat in r["window_stats"].values()) if r["window_stats"] else 0.0),
            r["occupied_bin_frac"] if r["occupied_bin_frac"] is not None else 1.0,
            -float(r["max_bin_frac"] or 0.0),
        ),
    )
    if top_groups > 0:
        print(f"\nTop {min(top_groups, len(ranked))} densest groups:")
        for record in ranked[:top_groups]:
            best_window = None
            best_stat = None
            best_frac = -1.0
            for window, stat in record["window_stats"].items():
                frac = stat[0] / max(1, int(record["unique_pin_count"]))
                if frac > best_frac:
                    best_frac = frac
                    best_window = window
                    best_stat = stat
            if best_stat is None:
                dense_text = "dense_window=n/a"
            else:
                count, start, end = best_stat
                dense_text = (
                    f"dense{best_window}=[{start},{end}) "
                    f"count={count}/{record['unique_pin_count']} ({_pct(best_frac)})"
                )
            print(
                f"sample={record['sample']} layer={record['layer']} group={record['kv_group']} "
                f"span={record['pin_span']} ({_pct(record['pin_span_frac'])}) "
                f"bins={record['occupied_bins']}/{bins} "
                f"gap_med={_fmt_num(record['gap_median'])} "
                f"{dense_text}"
            )
            print(f"  hist: {_ascii_hist(record['histogram'])}")


def _format_layers(layers: set[int]) -> str:
    if not layers:
        return "<none>"
    ordered = sorted(layers)
    ranges: list[str] = []
    start = prev = ordered[0]
    for layer in ordered[1:]:
        if layer == prev + 1:
            prev = layer
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = layer
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def print_json_summary(pin_diag: dict[str, Any]) -> None:
    overall = pin_diag.get("overall", {})
    print("\n== JSON group-level summary ==")
    print(f"verdict: {pin_diag.get('verdict')}")
    print(f"sample_count: {pin_diag.get('sample_count')}")
    print(f"samples_with_needle: {pin_diag.get('samples_with_needle')}")
    print(f"samples_with_comparable_needle: {pin_diag.get('samples_with_comparable_needle')}")
    print(
        "overall: "
        f"near={_pct(overall.get('near_hit_rate'))} "
        f"exact={_pct(overall.get('exact_hit_rate'))} "
        f"median_dist={overall.get('median_min_distance')} "
        f"mean_dist={overall.get('mean_min_distance')}"
    )

    by_layer = pin_diag.get("by_layer", [])
    if by_layer:
        print("\nby_layer:")
    for layer in by_layer:
        print(
            f"layer {int(layer['layer']):02d}: "
            f"near={_pct(layer.get('near_hit_rate'))} "
            f"exact={_pct(layer.get('exact_hit_rate'))} "
            f"median_dist={layer.get('median_min_distance')}"
        )


def print_random_baseline(
    pin_diag: dict[str, Any],
    config: dict[str, Any],
    *,
    seed: int,
    recent_size: int | None,
    pin_size: int | None,
) -> None:
    actual_recent_size = int(recent_size if recent_size is not None else config.get("log_kv_recent_size", 1024))
    actual_pin_size = int(pin_size if pin_size is not None else config.get("log_kv_pin_size", 256))
    radius = int(pin_diag.get("radius", 16))

    rng = random.Random(seed)
    exact_hits = 0
    near_hits = 0
    trials = 0

    for sample in pin_diag.get("samples", []):
        if sample.get("comparable_needle_count", 0) <= 0:
            continue

        prompt_tokens = sample.get("prompt_tokens", {})
        used_tokens = prompt_tokens.get("used")
        token_offset = int(prompt_tokens.get("left_truncated_tokens") or 0)
        if used_tokens is None:
            continue

        candidate_count = int(used_tokens) - actual_recent_size
        if candidate_count <= 0:
            continue

        spans = [span for span in sample.get("needle_spans", []) if span.get("survived_left_truncation")]
        if not spans:
            continue

        random_pin_count = min(actual_pin_size, candidate_count)
        for layer in sample.get("layers", []):
            if not layer.get("has_pin_indices"):
                continue
            for _group in layer.get("groups", []):
                random_pins = [
                    idx + token_offset
                    for idx in rng.sample(range(candidate_count), random_pin_count)
                ]
                distances = [_distance_to_spans(int(pin), spans) for pin in random_pins]
                distances = [dist for dist in distances if dist is not None]
                if not distances:
                    continue
                trials += 1
                exact_hits += any(dist == 0 for dist in distances)
                near_hits += any(dist <= radius for dist in distances)

    print("\n== Random pin baseline ==")
    print(
        f"random seed={seed}, pin_size={actual_pin_size}, "
        f"recent_size={actual_recent_size}, radius={radius}"
    )
    print(f"随机 pin 基线（{trials} 组 sample×layer×group）:")
    print(f"  exact_hit_rate ~ {_pct(exact_hits, trials)}")
    print(f"  near_hit_rate  ~ {_pct(near_hits, trials)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", help="pin_diag JSON path, or a glob such as './pin_diag/*.json'.")
    parser.add_argument("--band", default="23-26", help="Layer band for sample-level summary, e.g. 23-26 or 20,23-26.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the random-pin baseline.")
    parser.add_argument("--recent_size", type=int, default=None, help="Override log_kv_recent_size for the baseline.")
    parser.add_argument("--pin_size", type=int, default=None, help="Override log_kv_pin_size for the baseline.")
    parser.add_argument("--pin_bins", type=int, default=32, help="Number of bins for pin_indices histograms.")
    parser.add_argument(
        "--pin_windows",
        default="64,128,256,512",
        help="Comma-separated token window sizes for dense-window pin clustering metrics.",
    )
    parser.add_argument("--pin_top_groups", type=int, default=12, help="How many densest groups to print.")
    parser.add_argument("--no_random_baseline", action="store_true", help="Skip the random-pin baseline.")
    args = parser.parse_args()

    path, payload = _load_json(args.json)
    pin_diag = _pin_diag(payload)
    config = _config(payload)
    band_layers = _parse_layer_spec(args.band)

    print(f"file: {path}")
    print_sample_hit_summary(pin_diag, band_layers)
    print_compressed_slot_summary(pin_diag, band_layers)
    print_pin_distribution_summary(
        pin_diag,
        config,
        band_layers=band_layers,
        bins=args.pin_bins,
        windows=_parse_int_list(args.pin_windows),
        recent_size=args.recent_size,
        top_groups=args.pin_top_groups,
    )
    print_json_summary(pin_diag)
    if not args.no_random_baseline:
        print_random_baseline(
            pin_diag,
            config,
            seed=args.seed,
            recent_size=args.recent_size,
            pin_size=args.pin_size,
        )


if __name__ == "__main__":
    main()
