"""Diagnostics for LogKV salience pins on needle-in-a-haystack prompts.

This module is deliberately read-only with respect to the model: it only looks
at ``CausalSelfAttention._log_kv_pin_indices`` after prefill and compares those
absolute token indices with the token span of the needle text in the prompt.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
import statistics
from typing import Any

import torch


_RULER_MAGIC_RE = re.compile(
    r"(?:One of the )?special magic [^.\n]{1,200}? for [^.\n]{1,200}? is:?\s*[^.\n]{1,200}?\.",
    re.IGNORECASE,
)
_SAN_FRANCISCO_RE = re.compile(
    r"The best thing to do in San Francisco is [^.\n]{1,240}?\.",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NeedleSpan:
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_len": max(0, self.token_end - self.token_start),
            "source": self.source,
        }


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(_as_strings(v))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            out.extend(_as_strings(v))
        return out
    return []


def _dedup_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        v = str(v)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _find_all(text: str, needle: str) -> list[tuple[int, int]]:
    out = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            return out
        out.append((i, i + len(needle)))
        start = i + max(1, len(needle))


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left_candidates = [text.rfind(mark, 0, start) for mark in ("\n", ".", "?", "!")]
    left = max(left_candidates)
    left = 0 if left < 0 else left + 1
    while left < len(text) and text[left].isspace():
        left += 1

    right_candidates = [text.find(mark, end) for mark in ("\n", ".", "?", "!")]
    right_candidates = [r for r in right_candidates if r >= 0]
    right = min(right_candidates) + 1 if right_candidates else end
    return left, right


def _token_len(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text)
    if isinstance(encoded, torch.Tensor):
        return int(encoded.numel())
    return len(encoded)


def _char_span_to_token_span(tokenizer: Any, prompt: str, start: int, end: int) -> tuple[int, int]:
    token_start = _token_len(tokenizer, prompt[:start])
    token_end = _token_len(tokenizer, prompt[:end])
    if token_end <= token_start:
        token_end = token_start + 1
    return token_start, token_end


def _doc_get(doc: Any, key: str) -> Any:
    if isinstance(doc, dict):
        return doc.get(key)
    return getattr(doc, key, None)


def _add_span(
    spans: list[tuple[int, int, str]],
    seen: set[tuple[int, int]],
    start: int,
    end: int,
    source: str,
) -> None:
    if start < 0 or end <= start:
        return
    key = (start, end)
    if key in seen:
        return
    seen.add(key)
    spans.append((start, end, source))


def find_needle_spans(
    tokenizer: Any,
    prompt: str,
    doc: Any | None = None,
    explicit_needles: list[str] | None = None,
) -> list[NeedleSpan]:
    """Find likely needle spans and map them to token coordinates.

    RULER NIAH docs usually expose the answer in ``outputs`` and place the
    actual needle as a full sentence in ``input``. The fallback therefore finds
    answer occurrences, expands them to the surrounding sentence, and also scans
    for the common RULER "special magic" template.
    """

    search_text = prompt

    spans: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    needle_values = list(explicit_needles or [])
    for key in ("needle", "needles", "needle_text", "needle_texts"):
        needle_values.extend(_as_strings(_doc_get(doc, key)))
    for needle in _dedup_strings(needle_values):
        for start, end in _find_all(search_text, needle):
            _add_span(spans, seen, start, end, "explicit_needle")

    answer_values: list[str] = []
    for key in ("outputs", "output", "answer", "answers", "target", "targets"):
        answer_values.extend(_as_strings(_doc_get(doc, key)))
    answer_values = _dedup_strings(answer_values)

    for pattern, source in ((_RULER_MAGIC_RE, "ruler_magic_regex"), (_SAN_FRANCISCO_RE, "sf_needle_regex")):
        for match in pattern.finditer(search_text):
            sentence = match.group(0)
            if not answer_values or any(ans in sentence for ans in answer_values):
                _add_span(spans, seen, match.start(), match.end(), source)

    for answer in answer_values:
        # Very short answers can occur in unrelated prompt text. Keep them only
        # when they sit inside a plausible needle sentence.
        for start, end in _find_all(search_text, answer):
            left, right = _sentence_bounds(search_text, start, end)
            sentence = search_text[left:right]
            if len(answer) < 6 and not _RULER_MAGIC_RE.search(sentence):
                continue
            _add_span(spans, seen, left, right, "answer_sentence")

    out: list[NeedleSpan] = []
    for start, end, source in spans:
        token_start, token_end = _char_span_to_token_span(tokenizer, search_text, start, end)
        out.append(
            NeedleSpan(
                text=search_text[start:end],
                char_start=start,
                char_end=end,
                token_start=token_start,
                token_end=token_end,
                source=source,
            )
        )
    return sorted(out, key=lambda s: (s.token_start, s.token_end, s.source))


def _distance_to_spans(index: int, spans: list[NeedleSpan]) -> int | None:
    if not spans:
        return None
    best: int | None = None
    for span in spans:
        if span.token_start <= index < span.token_end:
            dist = 0
        elif index < span.token_start:
            dist = span.token_start - index
        else:
            dist = index - (span.token_end - 1)
        best = dist if best is None else min(best, dist)
    return best


def _cache_slot_layout(
    cache: Any,
    *,
    batch_i: int,
    group_i: int,
    prompt_token_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct current LogKV slot spans from level widths.

    LogStructuredKVCache deliberately does not keep an O(N) token->slot map, but
    its slots are contiguous and time-ordered (ignoring salience-pin duplicates).
    Therefore cumulative ``level_w`` plus the exact recent window is enough to
    recover which original token span each compressed slot currently covers.
    """

    slots: list[dict[str, Any]] = []
    cursor = 0

    max_levels = int(getattr(cache, "max_levels", 0) or 0)
    for level in range(max_levels - 1, -1, -1):
        count = int(cache.level_count[batch_i, group_i, 0, level].item())
        if count <= 0:
            continue
        widths = cache.level_w[batch_i, group_i, 0, level, :count].detach().to("cpu").tolist()
        for slot_i, width_value in enumerate(widths):
            width = int(round(float(width_value)))
            used_start = cursor
            used_end = cursor + width
            slots.append(
                {
                    "kind": "level",
                    "level": level,
                    "slot_index": slot_i,
                    "width": width,
                    "used_token_start": used_start,
                    "used_token_end": used_end,
                    "token_start": used_start + int(prompt_token_offset),
                    "token_end": used_end + int(prompt_token_offset),
                }
            )
            cursor = used_end

    recent_count = int(getattr(cache, "recent_count", 0) or 0)
    for slot_i in range(recent_count):
        used_start = cursor
        used_end = cursor + 1
        slots.append(
            {
                "kind": "recent",
                "level": None,
                "slot_index": slot_i,
                "width": 1,
                "used_token_start": used_start,
                "used_token_end": used_end,
                "token_start": used_start + int(prompt_token_offset),
                "token_end": used_end + int(prompt_token_offset),
            }
        )
        cursor = used_end

    meta = {
        "token_count": int(getattr(cache, "token_count", 0) or 0),
        "recent_count": recent_count,
        "pin_count": int(getattr(cache, "pin_count", 0) or 0),
        "slot_count_without_pins": len(slots),
        "covered_tokens_without_pins": cursor,
        "layout_matches_token_count": cursor == int(getattr(cache, "token_count", cursor) or cursor),
    }
    return slots, meta


def _needle_covering_slots(
    slots: list[dict[str, Any]],
    spans: list[NeedleSpan],
) -> list[dict[str, Any]]:
    covers: list[dict[str, Any]] = []
    for span_i, span in enumerate(spans):
        for slot in slots:
            left = max(int(slot["token_start"]), span.token_start)
            right = min(int(slot["token_end"]), span.token_end)
            if right <= left:
                continue
            covers.append(
                {
                    "needle_span_index": span_i,
                    "needle_source": span.source,
                    "needle_token_start": span.token_start,
                    "needle_token_end": span.token_end,
                    "kind": slot["kind"],
                    "level": slot["level"],
                    "slot_index": slot["slot_index"],
                    "slot_width": slot["width"],
                    "slot_token_start": slot["token_start"],
                    "slot_token_end": slot["token_end"],
                    "slot_used_token_start": slot["used_token_start"],
                    "slot_used_token_end": slot["used_token_end"],
                    "overlap_tokens": right - left,
                    "is_compressed": slot["kind"] == "level",
                }
            )
    return covers


def _safe_request_metadata(req: Any | None) -> dict[str, Any]:
    if req is None:
        return {}
    meta: dict[str, Any] = {}
    for key in ("request_type", "task_name", "doc_id", "idx", "metadata"):
        value = getattr(req, key, None)
        if value is not None:
            meta[key] = value
    return meta


class PinDiagRecorder:
    """Collect and summarize ``_log_kv_pin_indices`` vs true needle spans."""

    def __init__(
        self,
        radius: int = 16,
        max_samples: int | None = None,
        include_indices: bool = False,
        nearest_k: int = 8,
    ) -> None:
        self.radius = int(radius)
        self.max_samples = max_samples
        self.include_indices = bool(include_indices)
        self.nearest_k = int(nearest_k)
        self.samples: list[dict[str, Any]] = []

    def should_record(self) -> bool:
        return self.max_samples is None or len(self.samples) < self.max_samples

    def merge_samples(self, parts: list[list[dict[str, Any]]]) -> None:
        samples = [sample for part in parts for sample in part]
        self.samples = samples[: self.max_samples] if self.max_samples is not None else samples

    def record(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt: str,
        sample_id: str | None = None,
        doc: Any | None = None,
        request: Any | None = None,
        prompt_token_offset: int = 0,
        original_prompt_tokens: int | None = None,
        used_prompt_tokens: int | None = None,
    ) -> None:
        if not self.should_record():
            return
        if doc is None and request is not None:
            doc = getattr(request, "doc", None)

        spans = find_needle_spans(tokenizer, prompt, doc=doc)
        shifted_span_dicts = []
        comparable_spans = []
        for span in spans:
            d = span.to_dict()
            d["used_token_start"] = span.token_start - int(prompt_token_offset)
            d["used_token_end"] = span.token_end - int(prompt_token_offset)
            d["survived_left_truncation"] = d["used_token_end"] > 0 and (
                used_prompt_tokens is None or d["used_token_start"] < used_prompt_tokens
            )
            shifted_span_dicts.append(d)
            if d["survived_left_truncation"]:
                comparable_spans.append(span)

        sample: dict[str, Any] = {
            "sample_id": sample_id,
            "request": _safe_request_metadata(request),
            "prompt_tokens": {
                "original": original_prompt_tokens,
                "used": used_prompt_tokens,
                "left_truncated_tokens": int(prompt_token_offset),
            },
            "needle_spans": shifted_span_dicts,
            "comparable_needle_count": len(comparable_spans),
            "layers": [],
        }

        for layer_idx, block in enumerate(model.transformer.h):
            idx = getattr(block.attn, "_log_kv_pin_indices", None)
            if idx is None:
                sample["layers"].append({"layer": layer_idx, "has_pin_indices": False, "groups": []})
                continue

            cache = getattr(block.attn, "kv_cache", None)
            idx_cpu = idx.detach().to(device="cpu", dtype=torch.long)
            # _log_kv_pin_indices are coordinates in the used/truncated prompt.
            # Shift them back to original prompt token coordinates for comparison
            # with spans found in the original prompt text.
            idx_full = idx_cpu + int(prompt_token_offset)
            groups = []
            for batch_i in range(idx_full.size(0)):
                for group_i in range(idx_full.size(1)):
                    if cache is not None:
                        slot_layout, slot_layout_meta = _cache_slot_layout(
                            cache,
                            batch_i=batch_i,
                            group_i=group_i,
                            prompt_token_offset=int(prompt_token_offset),
                        )
                        needle_covering_slots = _needle_covering_slots(slot_layout, comparable_spans)
                    else:
                        slot_layout_meta = None
                        needle_covering_slots = []
                    pins = idx_full[batch_i, group_i].tolist()
                    scored = []
                    for pin in pins:
                        dist = _distance_to_spans(int(pin), comparable_spans)
                        if dist is not None:
                            scored.append((int(dist), int(pin)))
                    scored.sort(key=lambda x: (x[0], x[1]))
                    exact_hits = [pin for dist, pin in scored if dist == 0]
                    near_hits = [pin for dist, pin in scored if dist <= self.radius]
                    group = {
                        "batch": batch_i,
                        "kv_group": group_i,
                        "pin_count": len(pins),
                        "exact_hit_count": len(exact_hits),
                        "near_hit_count": len(near_hits),
                        "min_distance": scored[0][0] if scored else None,
                        "nearest_indices": [
                            {"index": pin, "distance": dist}
                            for dist, pin in scored[: self.nearest_k]
                        ],
                        "needle_covering_slots": needle_covering_slots,
                        "needle_compressed_slot_count": sum(
                            1 for slot in needle_covering_slots if slot.get("is_compressed")
                        ),
                        "needle_recent_slot_count": sum(
                            1 for slot in needle_covering_slots if slot.get("kind") == "recent"
                        ),
                    }
                    if slot_layout_meta is not None:
                        group["slot_layout"] = slot_layout_meta
                    if self.include_indices:
                        group["pin_indices"] = pins
                    groups.append(group)
            sample["layers"].append({"layer": layer_idx, "has_pin_indices": True, "groups": groups})

        sample["summary"] = self._summarize_sample(sample)
        self.samples.append(sample)

    def _group_records(self) -> list[tuple[int, dict[str, Any]]]:
        records: list[tuple[int, dict[str, Any]]] = []
        for sample in self.samples:
            if sample.get("comparable_needle_count", 0) <= 0:
                continue
            for layer in sample.get("layers", []):
                if not layer.get("has_pin_indices"):
                    continue
                layer_idx = int(layer["layer"])
                for group in layer.get("groups", []):
                    records.append((layer_idx, group))
        return records

    def _summarize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        records: list[tuple[int, dict[str, Any]]] = []
        if sample.get("comparable_needle_count", 0) <= 0:
            return {
                "groups": 0,
                "exact_hit_groups": 0,
                "near_hit_groups": 0,
                "exact_hit_rate": None,
                "near_hit_rate": None,
                "mean_min_distance": None,
                "median_min_distance": None,
            }
        for layer in sample.get("layers", []):
            if not layer.get("has_pin_indices"):
                continue
            layer_idx = int(layer["layer"])
            for group in layer.get("groups", []):
                records.append((layer_idx, group))
        return self._summarize_records(records)

    def _summarize_records(self, records: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
        distances = [g["min_distance"] for _, g in records if g.get("min_distance") is not None]
        total = len(records)
        exact = sum(1 for _, g in records if g.get("exact_hit_count", 0) > 0)
        near = sum(1 for _, g in records if g.get("near_hit_count", 0) > 0)
        return {
            "groups": total,
            "exact_hit_groups": exact,
            "near_hit_groups": near,
            "exact_hit_rate": exact / total if total else None,
            "near_hit_rate": near / total if total else None,
            "mean_min_distance": statistics.fmean(distances) if distances else None,
            "median_min_distance": statistics.median(distances) if distances else None,
        }

    def summary(self) -> dict[str, Any]:
        records = self._group_records()
        by_layer: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for layer_idx, group in records:
            by_layer[layer_idx].append((layer_idx, group))

        samples_with_needle = sum(1 for s in self.samples if s.get("needle_spans"))
        samples_with_comparable_needle = sum(1 for s in self.samples if s.get("comparable_needle_count", 0) > 0)
        missing_needle = len(self.samples) - samples_with_needle
        overall = self._summarize_records(records)

        if samples_with_needle == 0:
            verdict = "no_needle_spans_found"
        elif samples_with_comparable_needle == 0:
            verdict = "needle_truncated_out_of_actual_prompt"
        else:
            verdict = "no_pin_indices_recorded"
        near_rate = overall.get("near_hit_rate")
        exact_rate = overall.get("exact_hit_rate")
        if records:
            if (exact_rate or 0.0) >= 0.25 or (near_rate or 0.0) >= 0.75:
                verdict = "pins_often_hit_needle_or_neighborhood_check_train_infer_pin_mismatch"
            else:
                verdict = "pins_mostly_miss_needle_check_pin_selection"

        return {
            "radius": self.radius,
            "include_indices": self.include_indices,
            "token_span_method": "char_offsets_reencoded_as_prefix_lengths_approx",
            "exact_hit_caveat": (
                "BPE merges across char cut points can shift token_start/token_end by a few tokens; "
                "prefer near_hit_rate when exact_hit_rate is low but near_hit_rate is high."
            ),
            "max_samples_scope": "per_rank_before_all_gather_then_truncated_after_merge",
            "compressed_slot_method": (
                "reconstruct_current_logkv_slot_spans_from_cumulative_level_w; "
                "kind=level means compressed hierarchy slot, kind=recent means exact recent-window token"
            ),
            "sample_count": len(self.samples),
            "samples_with_needle": samples_with_needle,
            "samples_with_comparable_needle": samples_with_comparable_needle,
            "samples_missing_needle": missing_needle,
            "overall": overall,
            "by_layer": [
                {"layer": layer_idx, **self._summarize_records(layer_records)}
                for layer_idx, layer_records in sorted(by_layer.items())
            ],
            "verdict": verdict,
            "samples": self.samples,
        }
