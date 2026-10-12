"""Gamma cues: exact-pool selection values spans that repeat rare tokens of the stream's opening."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from litgpt.alpha_log_kv import CUE_CAP, CUE_FLOOR, CUE_TAU, CueStream, add_cue_values, cut_spans
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
    with pytest.raises(ValueError, match="expects position 10"):
        stream.ingest(ids, np.zeros((1, 10)), [12])
    with pytest.raises(ValueError, match=">= 1"):
        CueStream(0)


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
        return max(np.log((stop + 1) / (count + 1)) - CUE_TAU, 0.)

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
    assert later[0] == pytest.approx(_value(np.log(19_001 / 3) - CUE_TAU, 8))
    assert later[0] < _value(np.log(19_001 / 2) - CUE_TAU, 8)


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
    stream = CueStream(4)
    head = np.array([[1, 2, 3, 4]])
    stream.ingest(head, np.array([[0, 0, 0, 2]]), [0])
    stream.ingest(np.full((1, 100_000), 99), np.zeros((1, 100_000)), [4])
    stream.ingest(np.array([[1, 2, 3, 4]]), np.zeros((1, 4)), [100_004])
    values = stream.span_values(0, np.arange(100_004, 100_008), [4], 100_008)
    assert np.isfinite(values).all() and 0 < values[0] <= np.exp(CUE_CAP) / 4
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
