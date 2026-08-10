"""Summarize ``eval.py --log_kv_pin_diag_output`` JSON files.

Usage:
    python unused/pin_diag.py ./pin_diag_smoke/pin_diag_xxx.json
    python unused/pin_diag.py './pin_diag_smoke/*.json' --band 23-26

The script reports:
  1. sample-level hit rates: whether any layer/group hits the needle;
  2. sample-level hit rates inside a selected layer band;
  3. the recorded group-level rates from the JSON summary;
  4. a random-pin baseline with the same pin_size/recent_size/radius.
"""

from __future__ import annotations

import argparse
import glob
import json
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
    parser.add_argument("--no_random_baseline", action="store_true", help="Skip the random-pin baseline.")
    args = parser.parse_args()

    path, payload = _load_json(args.json)
    pin_diag = _pin_diag(payload)
    config = _config(payload)
    band_layers = _parse_layer_spec(args.band)

    print(f"file: {path}")
    print_sample_hit_summary(pin_diag, band_layers)
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
