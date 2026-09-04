"""Needle-in-a-haystack span helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
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
    """Find likely needle spans and map them to token coordinates."""

    spans: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    needle_values = list(explicit_needles or [])
    for key in ("needle", "needles", "needle_text", "needle_texts"):
        needle_values.extend(_as_strings(_doc_get(doc, key)))
    for needle in _dedup_strings(needle_values):
        for start, end in _find_all(prompt, needle):
            _add_span(spans, seen, start, end, "explicit_needle")

    answer_values: list[str] = []
    for key in ("outputs", "output", "answer", "answers", "target", "targets"):
        answer_values.extend(_as_strings(_doc_get(doc, key)))
    answer_values = _dedup_strings(answer_values)

    for pattern, source in ((_RULER_MAGIC_RE, "ruler_magic_regex"), (_SAN_FRANCISCO_RE, "sf_needle_regex")):
        for match in pattern.finditer(prompt):
            sentence = match.group(0)
            if not answer_values or any(ans in sentence for ans in answer_values):
                _add_span(spans, seen, match.start(), match.end(), source)

    for answer in answer_values:
        for start, end in _find_all(prompt, answer):
            left, right = _sentence_bounds(prompt, start, end)
            sentence = prompt[left:right]
            if len(answer) < 6 and not _RULER_MAGIC_RE.search(sentence):
                continue
            _add_span(spans, seen, left, right, "answer_sentence")

    out: list[NeedleSpan] = []
    for start, end, source in spans:
        token_start, token_end = _char_span_to_token_span(tokenizer, prompt, start, end)
        out.append(
            NeedleSpan(
                text=prompt[start:end],
                char_start=start,
                char_end=end,
                token_start=token_start,
                token_end=token_end,
                source=source,
            )
        )
    return sorted(out, key=lambda s: (s.token_start, s.token_end, s.source))
