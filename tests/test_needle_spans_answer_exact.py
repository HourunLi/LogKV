"""Tests for the answer_exact span added to litgpt/needle_spans.py.

Pure string/logic tests -- no model, no real tokenizer needed (a whitespace
tokenizer stand-in is enough to exercise the char->token span conversion).
Not executed in the environment that wrote this file; run for real before
trusting scripts/export_niah_samples.py's span annotations.
"""

import torch

from litgpt.needle_spans import find_needle_spans


class _WhitespaceTokenizer:
    """Minimal encode()-only stand-in: one token per whitespace-split word,
    enough to exercise _char_span_to_token_span's tokenizer.encode() contract
    (returns something with either .numel() or plain len()).
    """

    def encode(self, text: str):
        return torch.tensor([0] * len(text.split()))


def test_answer_exact_is_the_tight_substring_span():
    prompt = "Some filler text. The special magic number for testing is: 42. More filler follows here."
    doc = {"outputs": ["42"]}
    spans = find_needle_spans(_WhitespaceTokenizer(), prompt, doc=doc)
    exact = [s for s in spans if s.source == "answer_exact"]
    assert len(exact) == 1
    assert exact[0].text == "42"
    assert prompt[exact[0].char_start : exact[0].char_end] == "42"


def test_answer_exact_is_strictly_narrower_than_answer_sentence():
    prompt = "Padding before. The special magic number for the test-case is: 12345. Padding after this line."
    doc = {"outputs": ["12345"]}
    spans = find_needle_spans(_WhitespaceTokenizer(), prompt, doc=doc)
    exact = next(s for s in spans if s.source == "answer_exact")
    sentence = next(s for s in spans if s.source == "answer_sentence")
    assert exact.char_end - exact.char_start < sentence.char_end - sentence.char_start
    assert sentence.char_start <= exact.char_start
    assert exact.char_end <= sentence.char_end
    # the sentence span must actually contain the exact span as a substring
    assert prompt[exact.char_start:exact.char_end] in prompt[sentence.char_start:sentence.char_end]


def test_answer_exact_respects_the_short_answer_guard():
    """A short answer (<6 chars) with no RULER-magic-pattern sentence around it
    is filtered by the same guard as answer_sentence -- answer_exact must not
    sneak a low-confidence match past that guard.
    """
    prompt = "This paragraph mentions the number 42 in an unrelated context with no special phrasing at all."
    doc = {"outputs": ["42"]}
    spans = find_needle_spans(_WhitespaceTokenizer(), prompt, doc=doc)
    assert not any(s.source in ("answer_sentence", "answer_exact") for s in spans)


def test_multiple_occurrences_each_get_their_own_answer_exact_span():
    prompt = "The special magic uuid for alpha is: abc-123. Later, alpha's uuid abc-123 is repeated verbatim."
    doc = {"outputs": ["abc-123"]}
    spans = find_needle_spans(_WhitespaceTokenizer(), prompt, doc=doc)
    exact = [s for s in spans if s.source == "answer_exact"]
    assert len(exact) == 2
    starts = sorted(s.char_start for s in exact)
    assert starts[0] < starts[1]


def test_no_answer_exact_span_when_answer_not_present():
    prompt = "Nothing relevant is in this prompt at all."
    doc = {"outputs": ["needle-content-not-here"]}
    spans = find_needle_spans(_WhitespaceTokenizer(), prompt, doc=doc)
    assert not any(s.source == "answer_exact" for s in spans)


def test_answer_exact_token_span_is_computed_correctly():
    prompt = "one two three four the special magic code for beta is: five six seven eight nine ten"
    doc = {"outputs": ["five six"]}
    spans = find_needle_spans(_WhitespaceTokenizer(), prompt, doc=doc)
    exact = next(s for s in spans if s.source == "answer_exact")
    # Whitespace-tokenizer: token count up to char_start should equal the
    # word index at which "five six" begins.
    prefix_word_count = len(prompt[: exact.char_start].split())
    assert exact.token_start == prefix_word_count
