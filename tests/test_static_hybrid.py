"""Static local/local/local/full attention, including strict window semantics."""

import math
from copy import deepcopy
from types import MethodType

import torch
import torch.nn.functional as F

from litgpt import Config
from litgpt.model import GPT


def _tiny_config(**overrides):
    values = dict(
        block_size=64,
        n_layer=4,
        n_embd=32,
        n_head=4,
        n_query_groups=2,
        vocab_size=31,
        padded_vocab_size=31,
        bias=False,
    )
    values.update(overrides)
    return Config(**values)


def test_28_layer_layout_and_cache_capacity():
    with torch.device("meta"):
        model = GPT(_tiny_config(n_layer=28, block_size=32768))
    model.enable_sink_window_training(sink_size=16, window_size=2048, full_attention_interval=4)
    full_layers = [i for i, block in enumerate(model.transformer.h, start=1) if not block.attn.training_sink_window]
    assert full_layers == [4, 8, 12, 16, 20, 24, 28]

    model.set_sink_window_cache(
        batch_size=1, sink_size=16, window_size=2048, full_attention_interval=4, device="meta"
    )
    for i, block in enumerate(model.transformer.h, start=1):
        cache = block.attn.kv_cache
        expected_sink, expected_window = (0, 32768) if i in full_layers else (16, 2048)
        assert (cache.sink_size, cache.window_size) == (expected_sink, expected_window)
        assert cache.sink_k.size(-2) == cache.sink_v.size(-2) == expected_sink
        assert cache.window_k.size(-2) == cache.window_v.size(-2) == expected_window
    assert model.mask_cache is None


def test_training_logits_and_gradients_match_dense_mask_reference(monkeypatch):
    torch.manual_seed(40)
    model = GPT(_tiny_config()).double()
    reference = deepcopy(model)
    model.enable_sink_window_training(
        sink_size=2, window_size=3, train_chunk_size=4, full_attention_interval=4
    )

    def dense_reference(attn, q, k, v, mask=None):
        # Construct each visible set directly; the oracle has no chunks or cache.
        length = q.size(-2)
        visible = torch.zeros(length, length, dtype=torch.bool, device=q.device)
        for position in range(length):
            if attn.block_idx == 3:
                visible[position, : position + 1] = True
            else:
                visible[position, : min(2, position + 1)] = True
                visible[position, max(0, position - 2) : position + 1] = True
        scale = attn.mscale**2 / math.sqrt(attn.config.attention_scores_scalar or attn.config.head_size)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=visible, scale=scale).transpose(1, 2)

    for block in reference.transformer.h:
        monkeypatch.setattr(
            block.attn, "scaled_dot_product_attention", MethodType(dense_reference, block.attn)
        )
    tokens = torch.randint(0, model.config.vocab_size, (2, 11))
    actual, expected = model(tokens), reference(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        assert parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=1e-9, rtol=1e-8)


def test_training_prefill_chunked_decode_and_reset_agree():
    torch.manual_seed(41)
    model = GPT(_tiny_config()).double().eval()
    model.enable_sink_window_training(sink_size=2, window_size=3, full_attention_interval=4)
    tokens = torch.randint(0, model.config.vocab_size, (1, 17))
    with torch.no_grad():
        expected = model(tokens)
        model.set_sink_window_cache(
            batch_size=1, sink_size=2, window_size=3, full_attention_interval=4, dtype=torch.float64
        )
        expected_buffers = None
        for chunks in ([17], [5, 1, 7, 4], [1] * 17, [17]):
            model.reset_sink_window_cache()
            assert all(block.attn.kv_cache.token_count == 0 for block in model.transformer.h)
            outputs = []
            start = 0
            for length in chunks:
                stop = start + length
                outputs.append(model(tokens[:, start:stop], input_pos=torch.arange(start, stop)))
                start = stop
            torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-9, rtol=1e-8)
            buffers = {
                (layer, name): buffer.clone()
                for layer, block in enumerate(model.transformer.h)
                for name, buffer in block.attn.kv_cache.named_buffers()
            }
            assert all(block.attn.kv_cache.token_count == 17 for block in model.transformer.h)
            if expected_buffers is None:
                expected_buffers = buffers
            else:
                for key, buffer in buffers.items():
                    torch.testing.assert_close(buffer, expected_buffers[key], atol=1e-9, rtol=1e-8)


def test_interval_one_matches_full_causal_attention():
    torch.manual_seed(42)
    model = GPT(_tiny_config()).double().eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, 17))
    with torch.no_grad():
        expected = model(tokens)
        model.enable_sink_window_training(sink_size=2, window_size=3, full_attention_interval=1)
        assert not any(block.attn.training_sink_window for block in model.transformer.h)
        torch.testing.assert_close(model(tokens), expected)
        model.set_sink_window_cache(
            batch_size=1, sink_size=2, window_size=3, full_attention_interval=1, dtype=torch.float64
        )
        actual = model(tokens, input_pos=torch.arange(17))
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
        model.reset_sink_window_cache()
        prefix = model(tokens[:, :5], input_pos=torch.arange(5))
        suffix = model(tokens[:, 5:], input_pos=torch.arange(5, 17))
        torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), expected, atol=1e-9, rtol=1e-8)
