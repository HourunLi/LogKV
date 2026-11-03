import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from litgpt.alpha_log_kv import SURPRISAL_BASE, SURPRISAL_SCALE, SurprisalStream
from litgpt.config import Config
from litgpt.model import GPT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unused"))
from niah_surprisal import token_surprisal  # noqa: E402

KWARGS = dict(batch_size=1, B=8, prefill_block=8, second_order_scale=0., semantic_clusters=True, cluster_k_max=4,
              semantic_flush_granularity=8, semantic_anchor_mode="mid", semantic_replay_updates=True,
              alpha_exact_tokens=8, alpha_span_max_tokens=4)


def _model():
    torch.manual_seed(7)
    config = Config(block_size=256, n_layer=2, n_embd=32, n_head=4, n_query_groups=2, vocab_size=41,
                    padding_multiple=1, rotary_percentage=1.)
    model = GPT(config).eval()
    model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7])
    return model


def test_span_values_grow_exponentially_with_the_mean_or_top4_surprisal():
    def value(bits):
        return SURPRISAL_SCALE * SURPRISAL_BASE ** bits
    stream = SurprisalStream("top4")
    stream.add(0, 0, [np.nan, 1., 2., 3., 4., 5., 9.])
    assert stream.span_values(0, np.arange(7), [2, 5], 7).tolist() == pytest.approx(
        [value(0.5), value((3 + 4 + 5 + 9) / 4)])
    stream = SurprisalStream("mean")
    stream.add(0, 0, [np.nan, 1., 2., 3.])
    stream.add(0, 4, [5.])
    assert stream.span_values(0, np.array([1, 2, 3, 4]), [3, 1], 5).tolist() == pytest.approx(
        [value(2.), value(5.)])
    with pytest.raises(RuntimeError, match="not known yet"):
        stream.span_values(0, np.array([5]), [1], 6)
    stream = SurprisalStream("mix")
    stream.add(0, 0, [1., 1., 1., 1., 1., 9.])
    assert stream.span_values(0, np.arange(6), [6], 6).tolist() == pytest.approx([value((14 / 6 + 3) / 2)])
    # "sum": the inverse probability of the span; seven 3.3-bit digits outweigh one 12-bit word.
    stream = SurprisalStream("sum")
    stream.add(0, 0, [3.3] * 7 + [12.])
    number, word = stream.span_values(0, np.arange(8), [7, 1], 8)
    assert number == pytest.approx(SURPRISAL_SCALE * 2 ** 23.1) and number > word


def test_multi_scores_a_span_by_the_dimension_it_stands_out_in():
    from litgpt.alpha_log_kv import SURPRISAL_MIN_SPREAD
    # 20 dull 4-token spans, a uniformly surprising one (mean) and a long one with one peak (top4).
    dull = [1., 2., 1., 2.] * 20
    bits = np.r_[dull, [6.] * 4, [1.] * 15 + [12.]]
    lengths = [4] * 20 + [4, 16]
    stream = SurprisalStream("multi")
    stream.add(0, 0, bits)
    values = stream.span_values(0, np.arange(len(bits)), lengths, len(bits))
    assert values[20] > values[0] * 100 and values[21] > values[0] * 100
    # A later call for the same flush (another layer) sees the same statistics.
    assert stream.span_values(0, np.arange(len(bits)), lengths, len(bits)).tolist() == pytest.approx(values.tolist())
    assert stream._stats[0].count == 22 and SURPRISAL_MIN_SPREAD > 0


def test_the_model_feeds_each_tokens_surprisal_in_chunks_and_returns_the_asked_logits():
    model = _model()
    ids = torch.randint(0, 41, (40,))
    plain = token_surprisal(model, ids.tolist())
    with torch.no_grad():
        full = model(ids[None]).float()
    # Inside the recent window the cache is exact, so chunked prefill reproduces the uncached model.
    model.set_log_kv_cache(recent_size=64, gamma_surprisal="mean", **KWARGS)
    assert model.log_kv_surprisal_chunk == 8
    with torch.no_grad():
        out = model(ids[None], input_pos=torch.arange(40), lm_head_start=35).float()
    assert out.shape == (1, 5, full.size(-1)) and torch.allclose(out, full[:, 35:], atol=1e-4)
    stream = model.log_kv_surprisal_stream._rows[0]
    assert stream[0] == 0. and stream[1:].tolist() == pytest.approx(plain[1:].tolist(), abs=1e-3)
    # Token-by-token decode feeds the same values.
    model.set_log_kv_cache(recent_size=64, gamma_surprisal="mean", **KWARGS)
    with torch.no_grad():
        for t in range(40):
            model(ids[None, t:t + 1], input_pos=torch.tensor([t]))
    assert model.log_kv_surprisal_stream._rows[0][1:].tolist() == pytest.approx(plain[1:].tolist(), abs=1e-3)


def test_a_surprising_span_takes_the_exact_pool():
    model = _model()
    ids = torch.full((64,), 1, dtype=torch.long)
    ids[3::4] = 3  # spans of four tokens
    bits = np.full(64, 1.)
    bits[20:24] = 12.  # one surprising span, long out of the recent window by the end
    pools = {}
    for score in ("mean", "top4"):
        model.set_log_kv_cache(recent_size=16, gamma_surprisal=score, **KWARGS)
        model.log_kv_surprisal_stream.add(0, 0, bits)  # given values: run the cache without the model's feed
        model._surprisal_inner = True
        try:
            with torch.no_grad():
                model(ids[None], input_pos=torch.arange(64))
        finally:
            model._surprisal_inner = False
        pools[score] = [set(block.attn.kv_cache._alpha_positions[0]) for block in model.transformer.h]
    assert all(set(range(20, 24)) <= pool for layers in pools.values() for pool in layers)
    with pytest.raises(ValueError, match="requires alpha_exact_tokens"):
        model.set_log_kv_cache(recent_size=16, gamma_surprisal="mean", **{**KWARGS, "alpha_exact_tokens": 0})


def test_a_short_surprising_span_evicts_a_long_dull_one():
    from litgpt.alpha_log_kv import SpanCut, choose_spans
    # The pool holds one 64-token span of 3 bits; a 20-token span of 6 bits arrives.
    cut = SpanCut([np.arange(84)], [np.arange(84)], [np.array([64, 20])], [np.array([True, False])],
                  [np.array([True, True])])
    stream = SurprisalStream("mean")
    stream.add(0, 0, np.r_[np.full(64, 3.), np.full(20, 6.)])
    exponential = stream.span_values(0, np.arange(84), [64, 20], 84).tolist()
    assert choose_spans(cut, exponential, 64).keep[0] == list(range(64, 84))
    linear = [3. * SURPRISAL_SCALE, 6. * SURPRISAL_SCALE]  # 20 x 6 < 64 x 3 x 1.1: the swap would fail
    assert choose_spans(cut, linear, 64).keep[0] == list(range(64))


def test_information_runs_cut_whole_words_by_their_code_length():
    from litgpt.alpha_log_kv import information_ends, word_start_table
    assert word_start_table([" the", "ing", "\n", "中", "", "-"]).tolist() == [True, False, True, True, True, False]
    # 30 one-token words of 2 bits, then a 7-token number of 3.5-bit digits (one word of 24.5 bits: none of
    # its tokens stands out alone), then a word "obnoxious-sale" of 12 + 0.2 + 10 bits, then 30 more words.
    bits = np.r_[np.full(30, 2.), np.full(7, 3.5), [12., 0.2, 10.], np.full(30, 2.)]
    starts = np.r_[np.ones(30), [1] + [0] * 6, [1, 0, 0], np.ones(30)].astype(bool)
    ends = np.flatnonzero(information_ends(bits, 64, starts)).tolist()
    assert ends == [29, 39]  # one run of both words, cut only between words


def test_information_spans_mark_the_caches_span_ends():
    from litgpt.alpha_log_kv import information_ends, word_start_table
    model = _model()
    texts = [" w" if i % 3 == 0 else "x" for i in range(41)]  # every third id begins a word
    with pytest.raises(RuntimeError, match="token texts"):
        model.set_log_kv_cache(recent_size=64, gamma_surprisal="mean", gamma_surprisal_spans=True, **KWARGS)
    model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7], literal_texts=texts)
    ids = torch.randint(0, 41, (40,))
    model.set_log_kv_cache(recent_size=64, gamma_surprisal="mean", gamma_surprisal_spans=True, **KWARGS)
    with torch.no_grad():
        model(ids[None], input_pos=torch.arange(40))
    words = word_start_table(texts)[ids.numpy()]
    expected = np.flatnonzero(information_ends(model.log_kv_surprisal_stream._rows[0], 8, words))
    assert len(expected)
    for block in model.transformer.h:
        assert all(block.attn.kv_cache._recent_span_ends[0][p] for p in expected)
    with pytest.raises(ValueError, match="requires gamma_surprisal"):
        model.set_log_kv_cache(recent_size=64, gamma_surprisal_spans=True, **KWARGS)
