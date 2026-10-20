"""Gamma literals: exact-pool selection values spans holding the first occurrence of a random-looking literal."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from litgpt.alpha_log_kv import (CUE_CAP, CUE_FLOOR, LITERAL_FLOOR, LITERAL_TIER_CAP, CueStream, LiteralStream,
                                 SummedValues, add_cue_values, cut_spans, literal_code)
from litgpt.config import Config
from litgpt.log_kv_cache import LogStructuredKVCache
from litgpt.model import GPT

# Digits, the space token Qwen puts before a number, hex letters, connectors, a CJK
# piece, punctuation that ends a literal, and digit-free words.
TEXTS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ", "f", "a", "c", "-", ".", "编号", "[", "To",
         " 12", "\n"] + [f" w{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(130)]
WORDS = range(21, len(TEXTS))


def test_literal_code_measures_how_random_the_symbols_are():
    random7, _ = literal_code("4829173")
    assert random7 > LITERAL_FLOOR + 5
    for low in ("1995", "1,000,000", "2006To", "$100,000", "x86"):
        assert literal_code(low)[0] < LITERAL_FLOOR, low
    uuid, symbols = literal_code("8f3a2c1e-4b5d-4c6e-9fbc-1234567890ab")
    assert uuid > CUE_CAP and symbols == "8f3a2c1e4b5d4c6e9fbc1234567890"  # "ab" ends it, so it is a word
    assert literal_code("special magic") == (0., "")
    assert literal_code("4,829,173") == literal_code("4829173")
    # The code is the adaptive order-0 length: 0.5 prior over 10 digits.
    assert literal_code("12")[0] == pytest.approx(-math.log(.5 / 5) - math.log(.5 / 6))


def test_tokenizer_literal_texts_keep_the_space_of_word_initial_pieces():
    from litgpt.tokenizer import Tokenizer
    pieces, decoded = ["▁the", "Ġcat", "12", "▁"], ["the", " cat", "12", ""]
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.backend = "huggingface"
    tokenizer.processor = SimpleNamespace(get_vocab_size=lambda **_: 4, decode=lambda ids: decoded[ids[0]],
                                          id_to_token=lambda i: pieces[i])
    assert tokenizer.gamma_literal_texts == (" the", " cat", "12", " ")


def _ids(*parts):
    table = {t: i for i, t in enumerate(TEXTS)}
    return np.array([table[p] for part in parts for p in part], dtype=np.int64)


def test_only_the_first_occurrence_of_a_literal_counts():
    number = [" ", "4", "8", "2", "9", "1", "7", "3"]
    ids = _ids([" wab"] * 5, number, ["."], [" wac"] * 5, number, [" wad", "编号"], number[1:], ["编号"],
               [" wae", " ", "1", "9", "9", "5"])
    stream = LiteralStream(TEXTS)
    stream.ingest(ids[None], [0])
    spans = [np.arange(0, 5), np.arange(5, 14), np.arange(14, 27), np.arange(27, 37), np.arange(37, len(ids))]
    values = stream.span_values(0, np.concatenate(spans), [len(s) for s in spans], len(ids))
    code = literal_code("4829173")[0]
    assert values[0] == 0 and values[1] == pytest.approx((math.exp(code) - math.exp(LITERAL_FLOOR)) / 9)
    # Repeats code to 0, also inside CJK text; a year stays below the floor.
    assert values[2] == values[3] == values[4] == 0
    row = stream._rows[0]
    # The space before a number is a token of its own and not part of the run; the year is still open.
    assert row.code[1:] == [0., 0.] and row.first == {"4829173": 6} and row.open_lo == len(ids) - 4


def _reference(ids, stop, lengths, floor):
    """Literal runs of ids[:stop] found directly, then the same span value."""
    stream = LiteralStream(TEXTS)
    lit, digit, lead = stream._tables(np.asarray(ids[:stop]))
    code, first, at = {}, {}, 0
    while at < stop:
        if not lit[at]:
            at += 1
            continue
        end = at + 1
        while end < stop and lit[end] and not lead[end]:
            end += 1
        if digit[at:end].any():
            value, symbols = literal_code("".join(TEXTS[i] for i in ids[at:end]))
            if first.setdefault(symbols, at) == at:
                code.update({p: value for p in range(at, end)})
        at = end
    out, start = [], 0
    for length in lengths:
        best = max(code.get(p, 0.) for p in range(start, start + length))
        out.append(max(math.exp(min(best, CUE_CAP)) - math.exp(floor), 0.) / length)
        start += length
    return out


@patch("litgpt.alpha_log_kv.LITERAL_FLOOR", 0.)  # Short random literals then carry value.
def test_literal_values_are_causal_and_independent_of_forward_chunking():
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(20, 160))
        ids = np.where(rng.random(n) < .6, rng.integers(0, 16, n), rng.integers(0, len(TEXTS), n))
        whole = LiteralStream(TEXTS)
        whole.ingest(ids[None], [0])
        chunked, seen = LiteralStream(TEXTS), 0
        while seen < n:
            step = int(rng.integers(1, 19))
            chunked.ingest(ids[None, seen:seen + step], [seen])
            seen = min(n, seen + step)
            # A run that continues past stop counts its tokens before it, whatever arrived later.
            stop = int(rng.integers(1, seen + 1))
            cut = sorted(set(rng.integers(1, stop, 3).tolist())) if stop > 1 else []
            lengths = np.diff([0, *cut, stop])
            got = chunked.span_values(0, np.arange(stop), lengths, stop)
            np.testing.assert_array_equal(got, whole.span_values(0, np.arange(stop), lengths, stop))
            np.testing.assert_allclose(got, _reference(ids, stop, lengths, 0.), rtol=1e-12)
    with pytest.raises(ValueError, match="expects position"):
        chunked.ingest(ids[None, :4], [3])


def test_summed_values_add_cue_and_literal_values_in_candidate_order():
    number = [" ", "4", "8", "2", "9", "1", "7", "3"]
    ids = _ids([" wab", " wac", "\n"], [" wad"] * 13, number, ["."], [" wab"] * 7, [" wae"] * 8)
    kinds = np.zeros(len(ids), dtype=np.int64)
    kinds[2] = 2
    cues, literals = CueStream(3), LiteralStream(TEXTS)
    cues.ingest(np.stack([ids, ids]), np.stack([kinds, kinds]), [0, 0])
    literals.ingest(np.stack([ids, ids]), [0, 0])
    ends = [[(i + 1) % 8 == 0 for i in range(32)]] * 2
    cut = cut_spans([[], []], [[], []], [list(range(8, 40))] * 2, ends, 0, 64)
    summed = add_cue_values(cut, [0.5] * 8, SummedValues([cues, literals]), [40, 40])
    parts = [add_cue_values(cut, [0.] * 8, stream, [40, 40]) for stream in (cues, literals)]
    assert summed == pytest.approx([0.5 + a + b for a, b in zip(*parts)])
    # The number and its full stop touch the second and third span of each row.
    assert [i for i, value in enumerate(parts[1]) if value > 0] == [1, 2, 5, 6]


def test_literal_values_sit_one_tier_below_cues_with_the_query_cue():
    five = [" ", "4", "8", "2", "9", "1"]
    ids = _ids([" wab"] * 4, five, ["."], [" wac"] * 3, [" ", "6", "0", "5", "3", "7", "1", "9"], ["."], [" wad"] * 4,
               [" ", "1", "9", "9", "5"])
    plain, tiered = LiteralStream(TEXTS), LiteralStream(TEXTS, below_cue=True)
    lengths = [4, 7, 3, 9, 4, 5]
    for stream in (plain, tiered):
        stream.ingest(ids[None], [0])
    got = tiered.span_values(0, np.arange(len(ids)), lengths, len(ids))
    codes = [0., literal_code("48291")[0], 0., literal_code("6053719")[0], 0., literal_code("1995")[0]]
    assert LITERAL_FLOOR < codes[1] < LITERAL_FLOOR + LITERAL_TIER_CAP - CUE_FLOOR < codes[3] and codes[5] < LITERAL_FLOOR
    expect = [max(math.exp(min(c - LITERAL_FLOOR + CUE_FLOOR, LITERAL_TIER_CAP)) - math.exp(CUE_FLOOR), 0.) / n
              for c, n in zip(codes, lengths)]
    assert got == pytest.approx(expect) and got[1] > 0 and got[3] == pytest.approx(
        (math.exp(LITERAL_TIER_CAP) - math.exp(CUE_FLOOR)) / 9)
    # The same literals pass the same floor in both tiers.
    assert ((plain.span_values(0, np.arange(len(ids)), lengths, len(ids)) > 0) == (got > 0)).all()


def test_literal_cuts_are_cached_per_stop_without_copying_the_runs():
    number = [" ", "4", "8", "2", "9", "1", "7", "3"]
    ids = _ids([" wab"] * 3, number, ["."], [" wac"] * 3, number[:4])
    stream = LiteralStream(TEXTS)
    stream.ingest(ids[None, :8], [0])  # The number is still open at 8.
    row = stream._rows[0]
    first = stream.span_values(0, np.arange(8), [8], 8)
    assert row.at_stop[8] == (0, None, pytest.approx(literal_code("4829")[0]))
    stream.span_values(0, np.arange(8), [8], 30)
    assert 30 not in row.at_stop  # Past the tokens seen: later tokens could change it.
    stream.ingest(ids[None, 8:], [8])  # Closes the number (cache dropped), then opens its prefix again.
    assert row.at_stop == {}
    np.testing.assert_array_equal(stream.span_values(0, np.arange(8), [8], 8), first)
    n, last, opened = row.at_stop[8]
    assert (n, opened) == (1, None) and last == pytest.approx(literal_code("4829")[0])
    whole = LiteralStream(TEXTS)
    whole.ingest(ids[None], [0])
    for stop in (8, 12, len(ids)):
        np.testing.assert_array_equal(stream.span_values(0, np.arange(stop), [stop], stop),
                                      whole.span_values(0, np.arange(stop), [stop], stop))


def _alpha_cache(literals, cues=0, **overrides):
    args = dict(B=8, recent_size=16, semantic_clusters=True, cluster_k_max=4, semantic_anchor_mode="mid",
                allocate_second_order=False, semantic_replay_updates=True, semantic_flush_granularity=16,
                cos_cache=torch.ones(4096, 8), sin_cache=torch.zeros(4096, 8), rope_n_elem=8,
                alpha_exact_tokens=16, alpha_span_max_tokens=8, beta_novelty=True, beta_adaptive_merge=True,
                gamma_level0_reinsert=True, gamma_top_merge="lightest", gamma_cue_tokens=cues,
                gamma_literals=literals)
    args.update(overrides)
    return LogStructuredKVCache((2, 2, 4096, 8), (2, 2, 4096, 8), **args)


@pytest.mark.parametrize("route", ["attach", "unified"])
def test_literals_keep_the_span_with_a_random_number_in_the_exact_pool(route):
    torch.manual_seed(8)
    n = 4000
    rng = np.random.default_rng(8)
    ids = rng.choice(np.array(WORDS), n)
    ends = np.zeros(n, dtype=bool)
    ends[rng.choice(np.arange(8, n), 480, replace=False)] = True
    years = rng.choice(np.arange(100, 2900), 40, replace=False)
    for at in years:  # Years and repeated numbers code below the floor.
        ids[at:at + 5] = _ids([" ", "1", "9", "9", "5"])
    needle = list(range(3000, 3008))
    ids[needle] = _ids([" ", "4", "8", "2", "9", "1", "7", "3"])
    ends[needle[0] - 1] = ends[needle[-1]] = True
    ends[needle[0]:needle[-1]] = False
    k, v = torch.randn(2, 2, n, 8), torch.randn(2, 2, n, 8)
    kept = {}
    for literals in (False, True):
        c = _alpha_cache(literals, semantic_unified_route=route == "unified")
        if literals:
            c.literal_stream = LiteralStream(TEXTS)
            c.literal_stream.ingest(np.stack([ids, ids]), [0, 0])
        with torch.no_grad():
            for s in range(0, n, 16):
                c.add_recent(k[:, :, s:s + 16], v[:, :, s:s + 16], k_raw=k[:, :, s:s + 16],
                             span_ends=[ends[s:s + 16].tolist()] * 2)
        kept[literals] = [set(row) for row in c._alpha_positions]
        mass = c.level_w.sum((2, 3, 4)) + c.alpha_valid.sum(-1)[:, None] + c.recent_count
        assert (mass == n).all()
    assert all(set(needle) <= row for row in kept[True])
    assert not any(set(needle) <= row for row in kept[False])


def test_literal_switch_is_validated_and_needs_the_model_stream():
    with pytest.raises(ValueError, match="requires alpha_exact_tokens"):
        _alpha_cache(True, alpha_exact_tokens=0, beta_novelty=False, beta_adaptive_merge=False)
    with pytest.raises(ValueError, match="token texts"):
        LiteralStream([])
    c = _alpha_cache(True)
    k = torch.randn(2, 2, 48, 8)
    with pytest.raises(RuntimeError, match="LiteralStream"), torch.no_grad():
        for s in range(0, 48, 16):
            c.add_recent(k[:, :, s:s + 16], k[:, :, s:s + 16], k_raw=k[:, :, s:s + 16],
                         span_ends=[[i % 5 == 4 for i in range(16)]] * 2)
    both = _alpha_cache(True, cues=8)
    both.cue_stream, both.literal_stream = CueStream(8), LiteralStream(TEXTS)
    stream, stops = both._gamma_cues([np.arange(16), np.arange(8)])
    assert isinstance(stream, SummedValues) and stops == [16, 8]
    assert stream.streams == (both.cue_stream, both.literal_stream)
    # With the query-aware cue, the literal stream must sit below the cue.
    queried = _alpha_cache(True, gamma_query_cue=True)
    queried.cue_stream, queried.literal_stream = CueStream(0), LiteralStream(TEXTS)
    with pytest.raises(RuntimeError, match="tier does not match"):
        queried._gamma_cues([np.arange(16), np.arange(8)])
    queried.literal_stream = LiteralStream(TEXTS, below_cue=True)
    assert queried._gamma_cues([np.arange(16), np.arange(8)])[0].streams[1].below_cue


def _model(vocab=len(TEXTS)):
    torch.manual_seed(5)
    config = Config(block_size=128, n_layer=2, n_embd=32, n_head=4, n_query_groups=2, vocab_size=vocab,
                    padding_multiple=1, rotary_percentage=1.)
    return GPT(config).eval()


KWARGS = dict(batch_size=2, B=8, recent_size=8, second_order_scale=0., semantic_clusters=True, cluster_k_max=4,
              semantic_flush_granularity=8, semantic_anchor_mode="mid", semantic_replay_updates=True,
              alpha_exact_tokens=8, alpha_span_max_tokens=4)


@patch("litgpt.alpha_log_kv.LITERAL_FLOOR", 0.)
def test_model_prefill_and_token_streaming_keep_the_same_exact_spans_with_literals():
    def pools(literals, stream_tokens):
        model = _model()
        model.set_alpha_span_boundary_ids([15, 20], literal_texts=TEXTS)
        model.set_log_kv_cache(prefill_block=8, gamma_literals=literals, **KWARGS)
        rng = np.random.default_rng(105)
        x = torch.as_tensor(np.where(rng.random((2, 64)) < .5, rng.integers(0, 11, (2, 64)),
                                     rng.integers(21, len(TEXTS), (2, 64))))
        with torch.no_grad():
            if stream_tokens:
                for t in range(64):
                    model(x[:, t:t + 1], input_pos=torch.tensor([t]))
            else:
                model(x, input_pos=torch.arange(64))
        return [block.attn.kv_cache._alpha_positions for block in model.transformer.h]

    assert pools(True, False) == pools(True, True)
    assert pools(True, False)[0] != pools(False, False)[0]  # Literals change what is kept.


def test_model_feeds_one_shared_literal_stream_and_needs_the_token_texts():
    model = _model()
    model.set_alpha_span_boundary_ids([15, 20])
    with pytest.raises(RuntimeError, match="literal_texts"):
        model.set_log_kv_cache(prefill_block=8, gamma_literals=True, **KWARGS)
    model.set_alpha_span_boundary_ids([15, 20], line_end_ids=[20], literal_texts=TEXTS)
    model.set_log_kv_cache(prefill_block=8, gamma_literals=True, gamma_cue_tokens=6, **KWARGS)
    stream = model.log_kv_literal_stream
    assert all(block.attn.kv_cache.literal_stream is stream and block.attn.kv_cache.cue_stream is
               model.log_kv_cue_stream for block in model.transformer.h)
    x = torch.as_tensor(_ids([" wab", " ", "4", "8", "2", "9", "1", "7", "3", "."] * 2)).repeat(2, 1)
    with torch.no_grad():
        model(x[:, :13], input_pos=torch.arange(13))
        model(x[:, 13:14], input_pos=torch.tensor([13]))
    assert [row.seen for row in stream._rows] == [14, 14]
    # The first number closed at the next word; the repeat is still open at position 12.
    assert [(row.lo, row.open_lo) for row in stream._rows] == [([2], 12)] * 2
    assert stream.span_values(0, np.arange(14), [10, 4], 14)[1] == 0  # It repeats the first one.
    assert [row.seen for row in model.log_kv_cue_stream._rows] == [14, 14]
    model.reset_log_kv_cache()
    with torch.no_grad():
        model(x[:, :5], input_pos=torch.arange(5))  # A new request restarts the rows.
    assert [row.seen for row in stream._rows] == [5, 5] and [row.lo for row in stream._rows] == [[], []]
    model.clear_kv_cache()
    assert model.log_kv_literal_stream is None
    model.set_log_kv_cache(prefill_block=8, **KWARGS)
    assert model.log_kv_literal_stream is None and model.transformer.h[0].attn.kv_cache.literal_stream is None
    model.set_log_kv_cache(prefill_block=8, gamma_literals=True, gamma_query_cue=True, **KWARGS)
    assert model.log_kv_literal_stream.below_cue and model.log_kv_cue_stream.tokens == 0
    # Every stream built from the same token texts shares one table build.
    assert model.log_kv_literal_stream.lit is LiteralStream(model.alpha_literal_texts).lit
    assert not LiteralStream(TEXTS).below_cue


def test_literals_bind_training_replay():
    c = _alpha_cache(True)
    c.literal_stream = LiteralStream(TEXTS)
    ids = np.arange(100)[None].repeat(2, 0) % len(TEXTS)
    c.literal_stream.ingest(ids, [0, 0])
    from litgpt.log_kv_cache import LogKVStreamTrainingAttention
    q, k, v = [torch.randn(2, 2, 48, 8, requires_grad=True) for _ in range(3)]
    ends = [[i % 5 == 4 for i in range(48)]] * 2
    y = LogKVStreamTrainingAttention.apply(q, k, v, c, .3, 16, 0., k, ends)
    c.gamma_literals = False
    with pytest.raises(RuntimeError, match="configuration changed"):
        y.sum().backward()
