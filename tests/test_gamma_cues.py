"""Gamma cues: exact-pool selection values spans that repeat rare tokens of the stream's opening."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from litgpt.alpha_log_kv import (CUE_CAP, CUE_FLOOR, CUE_MAX_COUNT, CUE_PRIOR, CUE_TAU, CueStream, add_cue_values,
                                 cut_spans, query_region)
from litgpt.config import Config
from litgpt.log_kv_cache import LogStructuredKVCache
from litgpt.model import GPT


def test_tokenizer_line_ends_are_span_ends_that_end_a_line():
    from litgpt.tokenizer import Tokenizer
    texts = [".\n", "\n\"", "x\n ", ".", "\r\n", "<eos>", "abc"]
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.backend, tokenizer.eos_id = "huggingface", 5
    tokenizer.processor = SimpleNamespace(get_vocab_size=lambda **_: len(texts), decode=lambda ids: texts[ids[0]])
    assert tokenizer.alpha_span_boundary_ids == (0, 1, 2, 3, 4, 5)
    assert tokenizer.alpha_line_end_ids == (0, 1, 2, 4, 5)


def test_cue_region_ends_at_the_last_line_end_else_span_end_else_whole_head():
    ids = np.arange(10, 20)[None]
    stream = CueStream(8)
    stream.ingest(ids, np.array([[0, 1, 0, 2, 0, 1, 0, 0, 0, 0]]), [0])
    assert stream._rows[0].region == 4 and stream._rows[0].cues.tolist() == [10, 11, 12, 13]
    stream.ingest(ids, np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0]]), [0])  # a row restarts at 0
    assert stream._rows[0].region == 6
    stream.ingest(ids, np.zeros((1, 10)), [0])
    assert stream._rows[0].region == 8
    # The region is used only once a decision's stop covers it, even if later tokens arrived.
    assert stream.span_values(0, np.arange(4, 8), [4], 7).tolist() == [0.]
    with pytest.raises(ValueError, match="expects position 10"):
        stream.ingest(ids, np.zeros((1, 10)), [12])
    with pytest.raises(ValueError, match=">= 0"):
        CueStream(-1)


def _stream_tokens(n=20_000, seed=0):
    """Opening line [100..103], unique filler ids, a frequent cue (100) and a few rare ones."""
    rng = np.random.default_rng(seed)
    ids = 100_000 + rng.permutation(n)
    ids[:4] = [100, 101, 102, 103]
    ids[4::40] = 100  # 2.5% of the stream: never evidence.
    ids[2000] = 101  # Alone and early: evidence below the floor.
    ids[[10_000, 10_002, 10_003]] = [101, 101, 102]  # One span with two distinct rare cues.
    ids[18_000] = 103
    kinds = np.zeros(n, dtype=np.int64)
    kinds[3] = 2
    return ids, kinds


def _value(evidence, length):
    return max(np.exp(evidence) - np.exp(CUE_FLOOR), 0.) / length


def test_cue_evidence_sums_distinct_rare_cues_outside_the_region():
    ids, kinds = _stream_tokens()
    stream = CueStream(16)
    stream.ingest(ids[None], kinds[None], [0])
    spans = [np.arange(9998, 10_006), np.arange(0, 8), np.arange(40, 48), np.arange(17_996, 18_004),
             np.arange(12_000, 12_008)]
    lengths = [len(s) for s in spans]
    stop = 15_000
    values = stream.span_values(0, np.concatenate(spans), lengths, stop)

    def weight(token, stop=stop):
        count = int((ids[:stop] == token).sum())
        return max(np.log((stop + 1 + CUE_PRIOR) / (count + 1)) - CUE_TAU, 0.)

    # 101 twice and 102 once: two distinct cues; 103 at 18000 is not yet seen at this stop.
    assert values[0] == pytest.approx(_value(weight(101) + weight(102), 8)) and values[0] > 0
    assert values[1] == 0  # The opening region itself carries no evidence.
    assert values[2] == 0  # A cue seen in more than 1% of the stream.
    assert values[3] == pytest.approx(_value(weight(103), 8)) and values[3] > 0
    assert values[4] == 0  # No cue.
    # Weak evidence stays below the floor: 101 alone, early in the stream.
    early = stream.span_values(0, np.arange(1996, 2004), [8], 2500)
    assert 0 < weight(101, 2500) < CUE_FLOOR and early[0] == 0
    # Later occurrences count once seen: 103 recurs at 18000, which lowers its weight after that.
    later = stream.span_values(0, spans[3], [8], 19_000)
    assert int((ids[:19_000] == 103).sum()) == 2
    assert later[0] == pytest.approx(_value(np.log((19_001 + CUE_PRIOR) / 3) - CUE_TAU, 8))
    assert later[0] < _value(np.log((19_001 + CUE_PRIOR) / 2) - CUE_TAU, 8)


def test_cue_values_are_causal_and_independent_of_forward_chunking():
    ids, kinds = _stream_tokens(seed=3)
    whole = CueStream(16)
    whole.ingest(np.stack([ids, ids[::-1]]), np.stack([kinds, kinds[::-1]]), [0, 0])
    chunked = CueStream(16)
    for a, b in ((0, 1), (1, 9), (9, 17), (17, 1200), (1200, len(ids))):
        chunked.ingest(np.stack([ids[a:b], ids[::-1][a:b]]), np.stack([kinds[a:b], kinds[::-1][a:b]]), [a, a])
    prefix = CueStream(16)
    prefix.ingest(ids[None, :12_000], kinds[None, :12_000], [0])
    positions = np.arange(9960, 10_060)
    lengths = [4] * 25
    for row in (0, 1):
        np.testing.assert_array_equal(whole.span_values(row, positions, lengths, 12_000),
                                      chunked.span_values(row, positions, lengths, 12_000))
    # Evidence as of a position never depends on tokens after it (layer-major prefill).
    np.testing.assert_array_equal(whole.span_values(0, positions, lengths, 12_000),
                                  prefix.span_values(0, positions, lengths, 12_000))
    assert whole.span_values(0, positions, lengths, 12_000).max() > 0


def test_cue_value_is_capped_and_unstarted_rows_score_zero():
    stream = CueStream(16)
    head = np.arange(1, 17)[None]  # Sixteen distinct cues; the line end closes the region.
    kinds = np.zeros((1, 16), dtype=np.int64)
    kinds[0, -1] = 2
    stream.ingest(head, kinds, [0])
    stream.ingest(np.full((1, 100_000), 99), np.zeros((1, 100_000)), [16])
    stream.ingest(head, np.zeros((1, 16)), [100_016])
    values = stream.span_values(0, np.arange(100_016, 100_032), [16], 100_032)
    # Sixteen cues of about 6.3 nats each: far above the cap.
    assert values[0] == pytest.approx((np.exp(CUE_CAP) - np.exp(CUE_FLOOR)) / 16)
    assert stream.span_values(1, np.arange(4), [4], 8).tolist() == [0.]
    short = CueStream(64)
    short.ingest(head, np.zeros((1, 4)), [0])  # The opening region is not complete yet.
    assert short.span_values(0, np.arange(4), [4], 4).tolist() == [0.]


def test_add_cue_values_follows_candidate_order_across_rows():
    ids, kinds = _stream_tokens()
    stream = CueStream(16)
    stream.ingest(np.stack([ids, ids]), np.stack([kinds, kinds]), [0, 0])
    ends = [[(i + 1) % 8 == 0 for i in range(16)]] * 2
    cut = cut_spans([[], []], [[], []], [list(range(9992, 10_008)), list(range(17_992, 18_008))], ends, 0, 64)
    scores = add_cue_values(cut, [0.5] * 4, stream, [15_000, 19_000])
    direct = [stream.span_values(0, np.arange(9992, 10_008), [8, 8], 15_000),
              stream.span_values(1, np.arange(17_992, 18_008), [8, 8], 19_000)]
    assert scores == pytest.approx([0.5 + x for x in np.concatenate(direct)])
    assert scores[1] > 0.5 and scores[3] > 0.5 and scores[0] == scores[2] == 0.5


def _alpha_cache(cues, **overrides):
    args = dict(B=8, recent_size=16, semantic_clusters=True, cluster_k_max=4, semantic_anchor_mode="mid",
                allocate_second_order=False, semantic_replay_updates=True, semantic_flush_granularity=16,
                cos_cache=torch.ones(4096, 8), sin_cache=torch.zeros(4096, 8), rope_n_elem=8,
                alpha_exact_tokens=16, alpha_span_max_tokens=8, beta_novelty=True, beta_adaptive_merge=True,
                gamma_level0_reinsert=True, gamma_top_merge="lightest", gamma_cue_tokens=cues)
    args.update(overrides)
    return LogStructuredKVCache((2, 2, 4096, 8), (2, 2, 4096, 8), **args)


@pytest.mark.parametrize("route", ["attach", "unified"])
def test_cues_keep_the_span_that_repeats_the_opening_in_the_exact_pool(route):
    torch.manual_seed(7)
    n = 4000  # Long enough for the needle's cues to clear the evidence floor when it competes.
    rng = np.random.default_rng(7)
    ids = 5_000 + rng.permutation(n)
    ids[:6] = [11, 12, 13, 14, 15, 16]
    ends = np.zeros(n, dtype=bool)
    ends[rng.choice(np.arange(8, n), 480, replace=False)] = True
    ends[5] = True
    needle = list(range(3000, 3006))
    ids[needle] = [20, 12, 21, 14, 22, 15]  # Repeats three opening tokens.
    ends[needle[0] - 1] = ends[needle[-1]] = True
    ends[needle[0]:needle[-1]] = False
    kinds = ends.astype(np.int64)
    kinds[5] = 2
    k, v = torch.randn(2, 2, n, 8), torch.randn(2, 2, n, 8)
    kept = {}
    for cues in (0, 32):
        c = _alpha_cache(cues, semantic_unified_route=route == "unified")
        if cues:
            c.cue_stream = CueStream(cues)
            c.cue_stream.ingest(np.stack([ids, ids]), np.stack([kinds, kinds]), [0, 0])
        with torch.no_grad():
            for s in range(0, n, 16):
                c.add_recent(k[:, :, s:s + 16], v[:, :, s:s + 16], k_raw=k[:, :, s:s + 16],
                             span_ends=[ends[s:s + 16].tolist()] * 2)
        kept[cues] = [set(row) for row in c._alpha_positions]
        mass = c.level_w.sum((2, 3, 4)) + c.alpha_valid.sum(-1)[:, None] + c.recent_count
        assert (mass == n).all()
    assert all(set(needle) <= row for row in kept[32])
    assert not any(set(needle) <= row for row in kept[0])


@patch("litgpt.alpha_log_kv.CUE_TAU", 0.)  # Short streams then carry cue evidence.
@patch("litgpt.alpha_log_kv.CUE_FLOOR", 0.)
@patch("litgpt.alpha_log_kv.CUE_PRIOR", 0)
def test_cue_evidence_does_not_depend_on_how_input_is_chunked():
    """Prefill blocks and token-by-token decode archive the same blocks with the same evidence."""
    import litgpt.alpha_log_kv as al
    n = 96
    rng = np.random.default_rng(1)
    ids = rng.integers(5, 12, (2, n))
    kinds = np.zeros((2, n), dtype=np.int64)
    kinds[:, 3] = 2
    ends = rng.random((2, n)) < .3
    ends[:, 3] = True
    torch.manual_seed(3)
    k, v = torch.randn(2, 2, n, 8), torch.randn(2, 2, n, 8)
    results = []
    for chunk in (8, 2, 1):
        c = LogStructuredKVCache((2, 2, 128, 8), (2, 2, 128, 8), B=8, recent_size=8, semantic_clusters=True,
                                 cluster_k_max=4, semantic_anchor_mode="mid", allocate_second_order=False,
                                 semantic_replay_updates=True, semantic_flush_granularity=8,
                                 cos_cache=torch.ones(128, 8), sin_cache=torch.zeros(128, 8), rope_n_elem=8,
                                 alpha_exact_tokens=8, alpha_span_max_tokens=4, gamma_cue_tokens=4)
        c.cue_stream = CueStream(4)
        c.cue_stream.ingest(ids, kinds, [0, 0])
        stops, original = [], al.add_cue_values

        def spy(cut, scores, stream, rows):
            stops.append(list(rows))
            return original(cut, scores, stream, rows)

        with patch.object(al, "add_cue_values", side_effect=spy), torch.no_grad():
            for s in range(0, n, chunk):
                c.add_recent(k[:, :, s:s + chunk], v[:, :, s:s + chunk], k_raw=k[:, :, s:s + chunk],
                             span_ends=ends[:, s:s + chunk].tolist())
        results.append((stops, c._alpha_positions))
    # Every decision reads evidence as of the end of the block it archives.
    assert results[0][0] == [[end, end] for end in range(8, n - 7, 8)]
    assert results[0] == results[1] == results[2]


@patch("litgpt.alpha_log_kv.CUE_TAU", 0.)
@patch("litgpt.alpha_log_kv.CUE_FLOOR", 0.)
@patch("litgpt.alpha_log_kv.CUE_PRIOR", 0)
def test_model_prefill_and_token_streaming_keep_the_same_exact_spans_with_cues():
    def pools(cues, stream_tokens):
        torch.manual_seed(5)
        config = Config(block_size=128, n_layer=2, n_embd=32, n_head=4, n_query_groups=2, vocab_size=41,
                        padding_multiple=1, rotary_percentage=1.)
        model = GPT(config).eval()
        model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7])
        model.set_log_kv_cache(batch_size=2, B=8, recent_size=8, prefill_block=8, second_order_scale=0.,
                               semantic_clusters=True, cluster_k_max=4, semantic_flush_granularity=8,
                               semantic_anchor_mode="mid", semantic_replay_updates=True, alpha_exact_tokens=8,
                               alpha_span_max_tokens=4, gamma_cue_tokens=cues)
        x = torch.randint(8, 20, (2, 64), generator=torch.Generator().manual_seed(105))
        x[:, 2] = 7  # The opening region is the first line: three tokens.
        with torch.no_grad():
            if stream_tokens:
                for t in range(64):
                    model(x[:, t:t + 1], input_pos=torch.tensor([t]))
            else:
                model(x, input_pos=torch.arange(64))
        return [block.attn.kv_cache._alpha_positions for block in model.transformer.h]

    assert pools(4, False) == pools(4, True)
    assert pools(4, False)[0] != pools(0, False)[0]  # Cues change what is kept.


def test_cue_switch_is_validated_and_needs_the_model_stream():
    with pytest.raises(ValueError, match="requires alpha_exact_tokens"):
        _alpha_cache(8, alpha_exact_tokens=0, beta_novelty=False, beta_adaptive_merge=False)
    with pytest.raises(ValueError, match=">= 0"):
        _alpha_cache(-1)
    c = _alpha_cache(8)
    k = torch.randn(2, 2, 48, 8)
    with pytest.raises(RuntimeError, match="CueStream"), torch.no_grad():
        for s in range(0, 48, 16):
            c.add_recent(k[:, :, s:s + 16], k[:, :, s:s + 16], k_raw=k[:, :, s:s + 16],
                         span_ends=[[i % 5 == 4 for i in range(16)]] * 2)


def test_model_feeds_one_shared_stream_with_line_ends_and_positions():
    torch.manual_seed(5)
    config = Config(block_size=64, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                    vocab_size=41, padding_multiple=1, rotary_percentage=1.)
    model = GPT(config).eval()
    model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7])
    kwargs = dict(batch_size=2, B=8, recent_size=8, second_order_scale=0., semantic_clusters=True, cluster_k_max=4,
                  semantic_flush_granularity=8, semantic_anchor_mode="mid", semantic_replay_updates=True,
                  alpha_exact_tokens=8, alpha_span_max_tokens=4, gamma_cue_tokens=6)
    model.set_log_kv_cache(prefill_block=8, **kwargs)
    stream = model.log_kv_cue_stream
    assert all(block.attn.kv_cache.cue_stream is stream for block in model.transformer.h)
    x = torch.randint(8, 41, (2, 24))
    x[0, 2], x[0, 4], x[1, 1] = 7, 3, 3
    with torch.no_grad():
        model(x[:, :17], input_pos=torch.arange(17))
        model(x[:, 17:18], input_pos=torch.tensor([17]))
    assert [row.seen for row in stream._rows] == [18, 18]
    assert [row.region for row in stream._rows] == [3, 2]  # Line end at 2, else the last span end.
    model.reset_log_kv_cache()
    with torch.no_grad():
        model(x[:, :9], input_pos=torch.arange(9))  # A new request restarts the rows.
    assert [row.seen for row in stream._rows] == [9, 9]
    model.clear_kv_cache()
    assert model.log_kv_cue_stream is None
    model.set_log_kv_cache(prefill_block=8, **{**kwargs, "gamma_cue_tokens": 0})
    assert model.log_kv_cue_stream is None and model.transformer.h[0].attn.kv_cache.cue_stream is None


def test_cue_tokens_bind_training_replay():
    c = _alpha_cache(8)
    c.cue_stream = CueStream(8)
    ids = np.arange(100)[None].repeat(2, 0)
    c.cue_stream.ingest(ids, np.zeros_like(ids), [0, 0])
    from litgpt.log_kv_cache import LogKVStreamTrainingAttention
    q, k, v = [torch.randn(2, 2, 48, 8, requires_grad=True) for _ in range(3)]
    ends = [[i % 5 == 4 for i in range(48)]] * 2
    y = LogKVStreamTrainingAttention.apply(q, k, v, c, .3, 16, 0., k, ends)
    c.gamma_cue_tokens = 0
    with pytest.raises(RuntimeError, match="configuration changed"):
        y.sum().backward()


def test_cue_guard_needs_a_line_ended_opening():
    ids = np.arange(10, 20)[None]
    for kinds, cues in (([0, 1, 0, 2, 0, 1, 0, 0, 0, 0], [10, 11, 12, 13]), ([0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [])):
        stream = CueStream(8, guard=True)
        stream.ingest(ids, np.array([kinds]), [0])
        stream.ingest(np.array([[10, 11, 12, 13] * 3]), np.zeros((1, 12)), [10])
        assert stream._rows[0].cues.tolist() == cues
        values = stream.span_values(0, np.arange(10, 22), [4, 4, 4], 22)
        assert (values > 0).all() if cues else (values == 0).all()
    # Without the guard, the same unended head falls back to its last span end.
    plain = CueStream(8)
    plain.ingest(ids, np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0]]), [0])
    assert plain._rows[0].region == 6


def test_cue_guard_drops_cues_counted_more_than_the_cap():
    ids, kinds = _stream_tokens()
    ids[5_000:5_000 + 8 * (CUE_MAX_COUNT - 1):8] = 102  # 102 now occurs CUE_MAX_COUNT + 1 times
    guarded, plain = CueStream(16, guard=True), CueStream(16)
    for stream in (guarded, plain):
        stream.ingest(ids[None], kinds[None], [0])
    span = np.arange(9998, 10_006)  # holds 101 (3 times so far) and 102
    stop = 15_000
    assert int((ids[:stop] == 102).sum()) == CUE_MAX_COUNT + 1 and int((ids[:stop] == 101).sum()) <= CUE_MAX_COUNT

    def weight(token):
        count = int((ids[:stop] == token).sum())
        return max(np.log((stop + 1 + CUE_PRIOR) / (count + 1)) - CUE_TAU, 0.)

    assert plain.span_values(0, span, [8], stop)[0] == pytest.approx(_value(weight(101) + weight(102), 8))
    assert guarded.span_values(0, span, [8], stop)[0] == pytest.approx(_value(weight(101), 8))


def test_cue_guard_is_validated_bound_to_replay_and_must_match_the_stream():
    with pytest.raises(ValueError, match="requires gamma_cue_tokens"):
        _alpha_cache(0, gamma_cue_guard=True)
    c = _alpha_cache(8, gamma_cue_guard=True)
    c.cue_stream = CueStream(8)  # unguarded stream, guarded cache
    k = torch.randn(2, 2, 48, 8)
    with pytest.raises(RuntimeError, match="does not match"), torch.no_grad():
        for s in range(0, 48, 16):
            c.add_recent(k[:, :, s:s + 16], k[:, :, s:s + 16], k_raw=k[:, :, s:s + 16],
                         span_ends=[[i % 5 == 4 for i in range(16)]] * 2)
    c = _alpha_cache(8, gamma_cue_guard=True)
    c.cue_stream = CueStream(8, guard=True)
    ids = np.arange(100)[None].repeat(2, 0)
    c.cue_stream.ingest(ids, np.zeros_like(ids), [0, 0])
    from litgpt.log_kv_cache import LogKVStreamTrainingAttention
    q, k, v = [torch.randn(2, 2, 48, 8, requires_grad=True) for _ in range(3)]
    ends = [[i % 5 == 4 for i in range(48)]] * 2
    y = LogKVStreamTrainingAttention.apply(q, k, v, c, .3, 16, 0., k, ends)
    c.gamma_cue_guard = False
    with pytest.raises(RuntimeError, match="configuration changed"):
        y.sum().backward()


def test_model_attaches_a_guarded_stream():
    torch.manual_seed(5)
    config = Config(block_size=64, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                    vocab_size=41, padding_multiple=1, rotary_percentage=1.)
    model = GPT(config).eval()
    model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7])
    model.set_log_kv_cache(batch_size=2, B=8, recent_size=8, prefill_block=8, second_order_scale=0.,
                           semantic_clusters=True, cluster_k_max=4, semantic_flush_granularity=8,
                           semantic_anchor_mode="mid", semantic_replay_updates=True, alpha_exact_tokens=8,
                           alpha_span_max_tokens=4, gamma_cue_tokens=6, gamma_cue_guard=True)
    assert model.log_kv_cue_stream.guard
    assert all(block.attn.kv_cache.gamma_cue_guard for block in model.transformer.h)
    x = torch.randint(8, 41, (2, 24))
    x[0, 2] = 7  # Row 0 has a line end in its head; row 1 only span ends.
    x[1, 1] = 3
    with torch.no_grad():
        model(x, input_pos=torch.arange(24))
    assert len(model.log_kv_cue_stream._rows[0].cues) == 3 and len(model.log_kv_cue_stream._rows[1].cues) == 0


def test_query_region_is_the_closing_question_after_the_first_line_end_of_its_tail():
    line_end = np.zeros(50, dtype=bool)
    line_end[7] = True
    ids = list(range(20, 40)) + [7, 41, 42, 43, 7, 44, 45]
    assert query_region(ids, line_end, tokens=8).tolist() == [41, 42, 43, 7, 44, 45]
    assert query_region(ids[:20], line_end, tokens=8).tolist() == list(range(32, 40))  # no line end: all
    assert query_region(ids[:21], line_end, tokens=8).tolist() == list(range(33, 40)) + [7]  # only the last
    assert query_region([7, 3, 99], line_end).tolist() == [3, 99]  # short prompt; ids past the table
    assert query_region(np.array([], dtype=np.int64), line_end).tolist() == []


def test_query_cues_join_the_opening_once_and_only_for_the_next_request():
    ids = np.arange(10, 20)[None]
    kinds = np.array([[0, 1, 0, 2, 0, 0, 0, 0, 0, 0]])
    stream = CueStream(8)
    stream.set_queries([[31, 13, 30, 31]])
    stream.ingest(ids, kinds, [0])
    assert stream._rows[0].cues.tolist() == [10, 11, 12, 13, 30, 31]  # sorted, each cue once
    stream.ingest(ids, kinds, [0])  # The next request has no question.
    assert stream._rows[0].cues.tolist() == [10, 11, 12, 13]
    stream.set_queries([[30]])
    stream.set_queries([None])  # None withdraws it.
    stream.ingest(ids, kinds, [0])
    assert stream._rows[0].cues.tolist() == [10, 11, 12, 13]
    # Without an opening region the question is the only cue, from position 0.
    only = CueStream(0)
    only.set_queries([[31, 30], None])
    only.ingest(np.repeat(ids, 2, 0), np.zeros((2, 10)), [0, 0])
    assert only._rows[0].cues.tolist() == [30, 31] and only._rows[0].region == 0
    assert len(only._rows[1].cues) == 0
    assert only.span_values(1, np.arange(4), [4], 10).tolist() == [0.]
    # The guard drops an opening without a line end but keeps the question.
    guarded = CueStream(8, guard=True)
    guarded.set_queries([[30]])
    guarded.ingest(ids, np.zeros((1, 10)), [0])
    assert guarded._rows[0].cues.tolist() == [30] and guarded._rows[0].region == 8


def test_query_cue_values_spans_that_repeat_the_question():
    ids, kinds = _stream_tokens()
    ids[[6, 10_001, 10_004]] = [200, 201, 200]  # Question words at the start and in one span.
    ids[6_000:6_000 + 8 * CUE_MAX_COUNT:8] = 202  # A question word counted CUE_MAX_COUNT times before.
    ids[12_002] = 202
    stream = CueStream(0, guard=True)
    stream.set_queries([[200, 201, 202, 203]])
    stream.ingest(ids[None], kinds[None], [0])
    spans = [np.arange(9998, 10_006), np.arange(0, 8), np.arange(11_998, 12_006), np.arange(12_400, 12_408)]
    stop = 15_000
    values = stream.span_values(0, np.concatenate(spans), [8] * 4, stop)

    def weight(token):
        count = int((ids[:stop] == token).sum())
        return max(np.log((stop + 1 + CUE_PRIOR) / (count + 1)) - CUE_TAU, 0.) if count <= CUE_MAX_COUNT else 0.

    assert values[0] == pytest.approx(_value(weight(200) + weight(201), 8)) and values[0] > 0
    # Without an opening region, the stream's first tokens are evidence too.
    assert values[1] == pytest.approx(_value(weight(200), 8)) and values[1] > 0
    assert int((ids[:stop] == 202).sum()) == CUE_MAX_COUNT + 1 and values[2] == 0
    assert values[3] == 0


def test_query_cue_values_do_not_depend_on_how_input_is_chunked():
    ids, kinds = _stream_tokens(seed=3)
    question = [ids[10_000], ids[10_002], 777]
    whole, chunked = CueStream(16, guard=True), CueStream(16, guard=True)
    whole.set_queries([question, None])
    whole.ingest(np.stack([ids, ids[::-1]]), np.stack([kinds, kinds[::-1]]), [0, 0])
    chunked.set_queries([question, None])
    for a, b in ((0, 1), (1, 9), (9, 17), (17, 1200), (1200, len(ids))):
        chunked.ingest(np.stack([ids[a:b], ids[::-1][a:b]]), np.stack([kinds[a:b], kinds[::-1][a:b]]), [a, a])
    positions, lengths = np.arange(9960, 10_060), [4] * 25
    for row in (0, 1):
        np.testing.assert_array_equal(whole.span_values(row, positions, lengths, 12_000),
                                      chunked.span_values(row, positions, lengths, 12_000))
    assert whole.span_values(0, positions, lengths, 12_000).max() > 0


def test_query_cues_keep_the_span_that_repeats_the_question_in_the_exact_pool():
    torch.manual_seed(7)
    n = 4000
    rng = np.random.default_rng(7)
    ids = 5_000 + rng.permutation(n)
    ends = np.zeros(n, dtype=bool)
    ends[rng.choice(np.arange(8, n), 480, replace=False)] = True
    needle = list(range(3000, 3006))
    ids[needle] = [20, 12, 21, 14, 22, 15]
    ends[needle[0] - 1] = ends[needle[-1]] = True
    ends[needle[0]:needle[-1]] = False
    k, v = torch.randn(2, 2, n, 8), torch.randn(2, 2, n, 8)
    kept = {}
    for question in (None, [12, 14, 15, 99]):
        c = _alpha_cache(0, gamma_query_cue=True, gamma_cue_guard=True)
        c.cue_stream = CueStream(0, guard=True)
        c.cue_stream.set_queries([question, question])
        c.cue_stream.ingest(np.stack([ids, ids]), np.zeros((2, n)), [0, 0])
        with torch.no_grad():
            for s in range(0, n, 16):
                c.add_recent(k[:, :, s:s + 16], v[:, :, s:s + 16], k_raw=k[:, :, s:s + 16],
                             span_ends=[ends[s:s + 16].tolist()] * 2)
        kept[question is not None] = [set(row) for row in c._alpha_positions]
    assert all(set(needle) <= row for row in kept[True])
    assert not any(set(needle) <= row for row in kept[False])


def test_query_cue_is_validated_bound_to_replay_and_must_match_the_stream():
    with pytest.raises(ValueError, match="requires alpha_exact_tokens"):
        _alpha_cache(0, gamma_query_cue=True, alpha_exact_tokens=0, beta_novelty=False, beta_adaptive_merge=False)
    _alpha_cache(0, gamma_query_cue=True, gamma_cue_guard=True)  # The question alone is enough for the guard.
    k = torch.randn(2, 2, 48, 8)
    for stream, message in ((None, "CueStream"), (CueStream(8), "does not match")):
        c = _alpha_cache(0, gamma_query_cue=True)
        c.cue_stream = stream
        with pytest.raises(RuntimeError, match=message), torch.no_grad():
            for s in range(0, 48, 16):
                c.add_recent(k[:, :, s:s + 16], k[:, :, s:s + 16], k_raw=k[:, :, s:s + 16],
                             span_ends=[[i % 5 == 4 for i in range(16)]] * 2)
    c = _alpha_cache(0, gamma_query_cue=True)
    c.cue_stream = CueStream(0)
    c.cue_stream.set_queries([[3, 4], [5]])
    ids = np.arange(100)[None].repeat(2, 0)
    c.cue_stream.ingest(ids, np.zeros_like(ids), [0, 0])
    from litgpt.log_kv_cache import LogKVStreamTrainingAttention
    q, k, v = [torch.randn(2, 2, 48, 8, requires_grad=True) for _ in range(3)]
    ends = [[i % 5 == 4 for i in range(48)]] * 2
    y = LogKVStreamTrainingAttention.apply(q, k, v, c, .3, 16, 0., k, ends)
    c.gamma_query_cue = False
    with pytest.raises(RuntimeError, match="configuration changed"):
        y.sum().backward()


def test_model_hands_each_request_its_question():
    torch.manual_seed(5)
    config = Config(block_size=64, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                    vocab_size=41, padding_multiple=1, rotary_percentage=1.)
    model = GPT(config).eval()
    model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7])
    with pytest.raises(RuntimeError, match="gamma_query_cue"):
        model.set_gamma_queries([[30]])
    kwargs = dict(batch_size=2, B=8, recent_size=8, prefill_block=8, second_order_scale=0., semantic_clusters=True,
                  cluster_k_max=4, semantic_flush_granularity=8, semantic_anchor_mode="mid",
                  semantic_replay_updates=True, alpha_exact_tokens=8, alpha_span_max_tokens=4)
    model.set_log_kv_cache(gamma_cue_guard=True, gamma_query_cue=True, **kwargs)
    stream = model.log_kv_cue_stream
    assert stream.tokens == 0 and stream.guard
    assert all(block.attn.kv_cache.cue_stream is stream for block in model.transformer.h)
    model.set_gamma_queries([[31, 30], None])
    x = torch.randint(8, 41, (2, 24))
    with torch.no_grad():
        model(x, input_pos=torch.arange(24))
    assert stream._rows[0].cues.tolist() == [30, 31] and len(stream._rows[1].cues) == 0
    model.reset_log_kv_cache()
    with torch.no_grad():
        model(x[:, :9], input_pos=torch.arange(9))  # The next request has no question.
    assert len(stream._rows[0].cues) == 0
    model.set_log_kv_cache(gamma_cue_tokens=6, **kwargs)
    with pytest.raises(RuntimeError, match="gamma_query_cue"):
        model.set_gamma_queries([[30]])


def test_eval_hands_each_prompt_question_to_the_model():
    # Exercise the real eval method without importing GPU/HF evaluation dependencies.
    source = Path(__file__).resolve().parents[1] / "eval.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "LogKVLM")
    cls.bases, cls.decorator_list = [], []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_set_gamma_query"]
    scope = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), scope)
    lm, handed = scope["LogKVLM"](), []
    lm.model = SimpleNamespace(set_gamma_queries=handed.append)
    lm._query_line_ends = np.zeros(200, dtype=bool)
    lm._query_line_ends[7] = True
    lm._set_gamma_query(torch.tensor(list(range(100, 190)) + [7, 11, 12, 13]))
    assert [q.tolist() for q in handed[0]] == [[11, 12, 13]]
