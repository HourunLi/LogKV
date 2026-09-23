"""Tests for litgpt/sinkwindow_cache.py and its litgpt/model.py wiring.

Not executed in the environment that wrote this file (no GPU/torch there);
run for real before trusting a SinkWindow training or eval run. The
"chunked prefill vs decode-by-decode across wraparound" test is the one
specifically written to catch the masking bug found and fixed during design
(see sinkwindow_cache.py's module docstring) -- treat a failure there as a
real correctness regression, not a flaky test.
"""

import math

import pytest
import torch

from litgpt import Config
from litgpt.model import GPT
from litgpt.sinkwindow_cache import (
    SinkWindowKVCache,
    assert_input_pos_contiguous,
    sink_window_chunk_attention,
    sink_window_train_chunk_attention,
)


def _tiny_config(**overrides) -> Config:
    kwargs = dict(
        name="sinkwindow-test-tiny",
        block_size=64,
        n_layer=2,
        n_embd=32,
        n_head=4,
        n_query_groups=2,  # GQA: 2 query heads share each of 2 groups
        vocab_size=97,
        padded_vocab_size=97,
        bias=False,
        norm_eps=1e-5,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


# ---------------------------------------------------------------------------
# SinkWindowKVCache unit tests
# ---------------------------------------------------------------------------


def _make_cache(batch=1, groups=2, k_dim=8, v_dim=8, sink_size=2, window_size=3):
    return SinkWindowKVCache(
        (batch, groups, 0, k_dim), (batch, groups, 0, v_dim),
        sink_size=sink_size, window_size=window_size, device=torch.device("cpu"), dtype=torch.float32,
    )


def test_reset_parameters_zeros_everything_and_resets_token_count():
    cache = _make_cache()
    cache.commit(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
    assert cache.token_count == 3
    cache.reset_parameters()
    assert cache.token_count == 0
    assert torch.equal(cache.sink_k, torch.zeros_like(cache.sink_k))
    assert torch.equal(cache.window_k, torch.zeros_like(cache.window_k))


def test_sink_fills_before_window_and_freezes():
    cache = _make_cache(sink_size=2, window_size=3)
    k = torch.arange(5 * 8, dtype=torch.float32).view(1, 1, 5, 8).expand(1, 2, 5, 8).contiguous()
    v = k.clone()
    cache.commit(k[..., :2, :], v[..., :2, :])  # fills sink exactly
    assert cache.sink_filled == 2
    assert cache.window_filled == 0
    assert torch.equal(cache.sink_k[:, :, 0, :], k[:, :, 0, :])
    assert torch.equal(cache.sink_k[:, :, 1, :], k[:, :, 1, :])

    cache.commit(k[..., 2:5, :], v[..., 2:5, :])  # 3 more tokens -> all into window
    assert cache.sink_filled == 2
    assert cache.window_filled == 3
    # sink content must be unchanged (write-once) after later commits
    assert torch.equal(cache.sink_k[:, :, 0, :], k[:, :, 0, :])
    assert torch.equal(cache.sink_k[:, :, 1, :], k[:, :, 1, :])


def test_window_ring_buffer_overwrites_oldest_after_wrap():
    cache = _make_cache(sink_size=0, window_size=3)
    k = torch.arange(5 * 8, dtype=torch.float32).view(1, 1, 5, 8).expand(1, 2, 5, 8).contiguous()
    cache.commit(k[..., :3, :], k[..., :3, :])
    assert cache.window_filled == 3
    frozen_k_before, _ = cache.read_frozen()
    assert set(frozen_k_before[0, 0, :, 0].tolist()) == {0.0, 8.0, 16.0}  # tokens 0,1,2

    cache.commit(k[..., 3:5, :], k[..., 3:5, :])  # tokens 3,4 -> evict tokens 0,1
    assert cache.window_filled == 3  # still capped at window_size
    frozen_k_after, _ = cache.read_frozen()
    assert set(frozen_k_after[0, 0, :, 0].tolist()) == {16.0, 24.0, 32.0}  # tokens 2,3,4


def test_commit_rejects_chunk_larger_than_window_size():
    cache = _make_cache(sink_size=0, window_size=3)
    with pytest.raises(ValueError, match="window_size"):
        cache.commit(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))


def test_extra_live_tensors_always_empty():
    cache = _make_cache()
    assert cache.extra_live_tensors() == []
    cache.commit(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
    assert cache.extra_live_tensors() == []


def test_assert_input_pos_contiguous_accepts_expected_and_rejects_gap():
    cache = _make_cache(sink_size=2, window_size=3)
    cache.commit(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))  # token_count -> 2
    assert_input_pos_contiguous(cache, torch.arange(2, 5), 3)  # must not raise
    with pytest.raises(ValueError):
        assert_input_pos_contiguous(cache, torch.arange(3, 6), 3)  # gap at position 2


# ---------------------------------------------------------------------------
# sink_window_chunk_attention / sink_window_train_chunk_attention correctness
# ---------------------------------------------------------------------------


def _brute_force_sink_window_attention(q, k, v, sink_size, window_size, scale):
    """Reference: one dense (T, T) mask, no chunking at all -- independent of
    the implementation under test.
    """
    T = q.size(-2)
    device = q.device
    i = torch.arange(T, device=device).view(-1, 1)
    j = torch.arange(T, device=device).view(1, -1)
    causal = j <= i
    in_sink = j < sink_size
    in_window = (i - j) < window_size
    visible = causal & (in_sink | in_window)
    mask = torch.zeros(T, T, dtype=q.dtype, device=device)
    mask.masked_fill_(~visible, float("-inf"))
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask.view(1, 1, T, T), scale=scale, enable_gqa=q.size(1) != k.size(1),
    )


@pytest.mark.parametrize("sink_size,window_size,T", [(0, 3, 10), (2, 3, 10), (4, 5, 17), (2, 100, 10)])
def test_train_chunk_attention_matches_brute_force_dense_mask(sink_size, window_size, T):
    torch.manual_seed(0)
    B, n_head, n_group, hs = 1, 4, 2, 8
    q = torch.randn(B, n_head, T, hs)
    k = torch.randn(B, n_group, T, hs)
    v = torch.randn(B, n_group, T, hs)
    scale = 1.0 / math.sqrt(hs)

    chunked = sink_window_train_chunk_attention(
        q, k, v, sink_size=sink_size, window_size=window_size, scale=scale, enable_gqa=True,
    )
    k_expanded = k.repeat_interleave(n_head // n_group, dim=1)
    v_expanded = v.repeat_interleave(n_head // n_group, dim=1)
    reference = _brute_force_sink_window_attention(q, k_expanded, v_expanded, sink_size, window_size, scale)
    torch.testing.assert_close(chunked, reference, atol=1e-5, rtol=1e-4)


def test_train_chunk_attention_degenerates_to_plain_causal_when_sink_plus_window_covers_all():
    torch.manual_seed(1)
    T = 6
    q = torch.randn(1, 2, T, 8)
    k = torch.randn(1, 2, T, 8)
    v = torch.randn(1, 2, T, 8)
    scale = 1.0 / math.sqrt(8)
    out = sink_window_train_chunk_attention(q, k, v, sink_size=T, window_size=T, scale=scale, enable_gqa=False)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Model-level integration
# ---------------------------------------------------------------------------


def test_dense_equivalence_when_sink_plus_window_covers_whole_sequence():
    """sink_size + window_size >= T means nothing is ever evicted -- SinkWindow
    inference must match plain dense causal attention exactly.
    """
    torch.manual_seed(2)
    config = _tiny_config()
    model = GPT(config)
    model.eval()
    T = 8
    idx = torch.randint(0, config.vocab_size, (1, T))

    with torch.no_grad():
        dense_logits = model(idx)  # training-mode dense forward, established correct by test_dense_bypass.py

        model.set_sink_window_cache(batch_size=1, sink_size=4, window_size=8, device="cpu")
        sink_window_logits = model(idx, input_pos=torch.arange(T))

    torch.testing.assert_close(sink_window_logits, dense_logits, atol=1e-4, rtol=1e-3)


def test_chunked_prefill_matches_decode_by_decode_across_wraparound():
    """THE regression test for the wraparound masking bug (see
    sinkwindow_cache.py's module docstring). sink_size=2, window_size=3,
    T=13 forces the ring buffer to wrap 3+ times within a single prefill
    call's internal chunking (chunk_size == window_size == 3). Decoding the
    same 13 tokens one at a time must produce identical cache state and
    logits at every position.
    """
    torch.manual_seed(3)
    config = _tiny_config()
    model = GPT(config)
    model.eval()
    T = 13
    sink_size, window_size = 2, 3
    idx = torch.randint(0, config.vocab_size, (1, T))

    with torch.no_grad():
        model.set_sink_window_cache(batch_size=1, sink_size=sink_size, window_size=window_size, device="cpu")
        prefill_logits = model(idx, input_pos=torch.arange(T))
        prefill_window_k = model.transformer.h[0].attn.kv_cache.window_k.clone()
        prefill_sink_k = model.transformer.h[0].attn.kv_cache.sink_k.clone()

        model.reset_sink_window_cache()
        decode_logits = []
        for t in range(T):
            out = model(idx[:, t : t + 1], input_pos=torch.tensor([t]))
            decode_logits.append(out)
        decode_logits = torch.cat(decode_logits, dim=1)
        decode_window_k = model.transformer.h[0].attn.kv_cache.window_k.clone()
        decode_sink_k = model.transformer.h[0].attn.kv_cache.sink_k.clone()

    torch.testing.assert_close(prefill_logits, decode_logits, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(prefill_sink_k, decode_sink_k)
    # Window buffer content must match as a SET per slot semantics: both
    # paths must have committed the exact same final ring-buffer state.
    torch.testing.assert_close(prefill_window_k, decode_window_k)


def test_chunked_prefill_in_two_different_chunk_groupings_agree():
    """Same 17 tokens, prefill split as one call of 17 vs two calls (10 then
    7) -- both must agree with each other (and transitively with decode,
    covered above). Exercises a prefill call that does NOT align with
    window_size boundaries.
    """
    torch.manual_seed(4)
    config = _tiny_config()
    model = GPT(config)
    model.eval()
    T = 17
    sink_size, window_size = 3, 4
    idx = torch.randint(0, config.vocab_size, (1, T))

    with torch.no_grad():
        model.set_sink_window_cache(batch_size=1, sink_size=sink_size, window_size=window_size, device="cpu")
        one_shot = model(idx, input_pos=torch.arange(T))

        model.reset_sink_window_cache()
        first = model(idx[:, :10], input_pos=torch.arange(0, 10))
        second = model(idx[:, 10:], input_pos=torch.arange(10, T))
        split = torch.cat([first, second], dim=1)

    torch.testing.assert_close(one_shot, split, atol=1e-4, rtol=1e-3)


def test_storage_stays_bounded_after_many_tokens():
    config = _tiny_config()
    model = GPT(config)
    model.eval()
    sink_size, window_size = 2, 3
    model.set_sink_window_cache(batch_size=1, sink_size=sink_size, window_size=window_size, device="cpu")
    cache = model.transformer.h[0].attn.kv_cache
    with torch.no_grad():
        for _ in range(5):
            idx = torch.randint(0, config.vocab_size, (1, window_size))
            model(idx, input_pos=torch.arange(cache.token_count, cache.token_count + window_size))
    assert cache.sink_k.shape == (1, config.n_query_groups, sink_size, config.head_size)
    assert cache.window_k.shape == (1, config.n_query_groups, window_size, config.head_size)
    assert cache.token_count == 5 * window_size


def test_reset_between_samples_leaves_no_cross_contamination():
    torch.manual_seed(5)
    config = _tiny_config()
    model = GPT(config)
    model.eval()
    model.set_sink_window_cache(batch_size=1, sink_size=2, window_size=3, device="cpu")
    with torch.no_grad():
        idx_a = torch.randint(0, config.vocab_size, (1, 6))
        model(idx_a, input_pos=torch.arange(6))
        model.reset_sink_window_cache()
        cache = model.transformer.h[0].attn.kv_cache
        assert cache.token_count == 0
        assert torch.equal(cache.window_k, torch.zeros_like(cache.window_k))

        idx_b = torch.randint(0, config.vocab_size, (1, 6))
        logits_b_fresh = model(idx_b, input_pos=torch.arange(6))

    # A cache reused after reset must behave identically to a brand-new one.
    model2 = GPT(config)
    model2.load_state_dict(model.state_dict())
    model2.eval()
    model2.set_sink_window_cache(batch_size=1, sink_size=2, window_size=3, device="cpu")
    with torch.no_grad():
        logits_b_clean = model2(idx_b, input_pos=torch.arange(6))
    torch.testing.assert_close(logits_b_fresh, logits_b_clean, atol=1e-4, rtol=1e-3)


def test_gqa_fast_path_matches_manual_repeat_interleave_reference():
    torch.manual_seed(6)
    B, n_head, n_group, T, hs = 1, 4, 2, 5, 8
    q = torch.randn(B, n_head, T, hs)
    k_new = torch.randn(B, n_group, T, hs)
    v_new = torch.randn(B, n_group, T, hs)
    frozen_k = torch.randn(B, n_group, 3, hs)
    frozen_v = torch.randn(B, n_group, 3, hs)
    scale = 1.0 / math.sqrt(hs)

    gqa_out = sink_window_chunk_attention(q, k_new, v_new, frozen_k, frozen_v, scale=scale, enable_gqa=True)

    rep = n_head // n_group
    manual_out = sink_window_chunk_attention(
        q,
        k_new.repeat_interleave(rep, dim=1),
        v_new.repeat_interleave(rep, dim=1),
        frozen_k.repeat_interleave(rep, dim=1),
        frozen_v.repeat_interleave(rep, dim=1),
        scale=scale,
        enable_gqa=False,
    )
    torch.testing.assert_close(gqa_out, manual_out, atol=1e-5, rtol=1e-4)
