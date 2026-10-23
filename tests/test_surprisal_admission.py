import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from litgpt.alpha_log_kv import SURPRISAL_SCALE, SurprisalStream
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


def test_span_values_are_the_scaled_mean_or_top4_surprisal():
    stream = SurprisalStream("top4")
    stream.add(0, 0, [np.nan, 1., 2., 3., 4., 5., 9.])
    assert stream.span_values(0, np.arange(7), [2, 5], 7).tolist() == pytest.approx(
        [SURPRISAL_SCALE * 0.5, SURPRISAL_SCALE * (3 + 4 + 5 + 9) / 4])
    stream = SurprisalStream("mean")
    stream.add(0, 0, [np.nan, 1., 2., 3.])
    stream.add(0, 4, [5.])
    assert stream.span_values(0, np.array([1, 2, 3, 4]), [3, 1], 5).tolist() == pytest.approx(
        [SURPRISAL_SCALE * 2., SURPRISAL_SCALE * 5.])
    with pytest.raises(RuntimeError, match="not known yet"):
        stream.span_values(0, np.array([5]), [1], 6)


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
