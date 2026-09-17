"""Regression checks for first-order storage, replay plans and bounded LM-head loss."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from litgpt.chunked_loss import chunked_linear_cross_entropy
from litgpt.log_kv_cache import LogStructuredKVCache, LogKVStreamTrainingAttention


@pytest.mark.parametrize("semantic,kmax,legacy,chunk", [
    (False, 1, False, 0), (True, 1, False, 0), (True, 8, False, 0),
    (True, 2, True, 0), (True, 8, False, 4),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_first_order_storage_and_replay(semantic, kmax, legacy, chunk, dtype):
    torch.manual_seed(123)
    q = torch.randn(1, 4, 32, 8, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 2, 32, 8, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    upstream = torch.randn_like(q)

    def run(allocate, reuse):
        cache = LogStructuredKVCache(
            k.shape, v.shape, B=3, recent_size=4, dtype=dtype,
            semantic_clusters=semantic, cluster_k_max=kmax, semantic_legacy_route=legacy,
            semantic_flush_granularity=4, semantic_cluster_chunk_size=chunk,
            seg_gap_max=1, seg_block_level=1, allocate_second_order=allocate,
            cos_cache=torch.ones(32, 8), sin_cache=torch.zeros(32, 8), rope_n_elem=8,
        )
        original = cache.get_attention_state

        def state(with_stats=False, *, plan=None):
            return original(with_stats, plan=plan if reuse else None)

        with patch.object(cache, "get_attention_state", side_effect=state), patch.object(
            cache, "_semantic_attention_plan", wraps=cache._semantic_attention_plan,
        ) as planner:
            out = LogKVStreamTrainingAttention.apply(q, k, v, cache, 8 ** -0.5, 4, 0., k)
            gradients = torch.autograd.grad(out, (q, k, v), upstream)
            if semantic and reuse:
                assert planner.call_count == 8  # forward only; zero planning in replay
        if not allocate:
            for name in ("level_sigma_u", "level_sigma2", "level_gamma_a", "level_gamma_b", "level_gamma"):
                assert getattr(cache, name) is None
            assert cache.get_attention_state(with_stats=True).slot_sigma_u is None
        state = cache.get_attention_state()
        result = (out, *gradients, state.slot_k, state.slot_v, state.slot_w)
        cache.reset_parameters()
        cache._convert_dtype(torch.float32)
        if allocate:
            cache.second_order = True  # zero-to-nonzero warmup remains possible after reset
        else:
            with pytest.raises(RuntimeError, match="storage is disabled"):
                cache.second_order = True
        return result

    expected = run(True, False)
    actual = run(False, True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("softcap,all_ignored", [(None, False), (3., False), (None, True)])
def test_chunked_head_loss_and_gradients(dtype, softcap, all_ignored):
    torch.manual_seed(9)
    hidden = torch.randn(2, 11, 8, dtype=dtype, requires_grad=True)
    weight = torch.randn(31, 8, dtype=dtype, requires_grad=True)
    bias = torch.randn(31, dtype=dtype, requires_grad=True)
    targets = torch.randint(31, (2, 11))
    targets[:, :3] = -100
    if all_ignored:
        targets.fill_(-100)
    logits = F.linear(hidden, weight, bias)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    reference = F.cross_entropy(logits.float().reshape(-1, 31), targets.reshape(-1), reduction="sum")
    reference = reference / (targets != -100).sum().clamp_min(1)
    saved_shapes = []

    def save(x):
        saved_shapes.append(tuple(x.shape))
        return x

    with torch.autograd.graph.saved_tensors_hooks(save, lambda x: x):
        actual = chunked_linear_cross_entropy(hidden, weight, targets, bias, chunk_size=4, softcap=softcap)
    assert (22, 31) not in saved_shapes and (2, 11, 31) not in saved_shapes
    eg = torch.autograd.grad(reference, (hidden, weight, bias))
    ag = torch.autograd.grad(actual, (hidden, weight, bias))
    tol = 0.02 if dtype == torch.bfloat16 else 2e-6
    for a, b in zip((actual, *ag), (reference, *eg)):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_model_loss_with_block_checkpoint_and_warmup(dtype):
    from collections import Counter
    from copy import deepcopy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
    from litgpt.config import Config
    from litgpt.model import Block, GPT

    torch.manual_seed(8)
    config = Config(n_layer=2, n_embd=32, n_head=4, n_query_groups=2, block_size=32,
                    vocab_size=31, padding_multiple=1, intermediate_size=64,
                    mlp_class_name="LLaMAMLP", parallel_residual=False,
                    norm_class_name="RMSNorm", norm_qk=True)
    reference = GPT(config).to(dtype=dtype)
    actual = deepcopy(reference)
    for model in (reference, actual):
        model.enable_log_kv_training(
            batch_size=1, B=3, recent_size=4, train_block=4, second_order_scale=0.,
            allocate_second_order=False, semantic_clusters=True, cluster_k_max=8,
            semantic_flush_granularity=4,
        )
    counts = Counter()

    def count(name):
        def hook(*args):
            counts[name] += 1
        return hook

    for block in actual.transformer.h:
        block.attn.register_forward_pre_hook(count("attention"))
        block.mlp.register_forward_pre_hook(count("mlp"))
    apply_activation_checkpointing(actual, check_fn=lambda module: isinstance(module, Block))
    inputs = torch.randint(31, (1, 24))
    targets = torch.randint(31, inputs.shape)
    targets[:, :3] = -100
    logits = reference(inputs)
    expected = F.cross_entropy(logits.float().reshape(-1, 31), targets.reshape(-1))
    expected.backward()
    saved_shapes = []

    def save(x):
        saved_shapes.append(tuple(x.shape))
        return x

    with torch.autograd.graph.saved_tensors_hooks(save, lambda x: x):
        loss = actual(inputs, targets=targets, loss_chunk_size=7)
    # Whole-block checkpointing must discard the per-layer Q/K/V and QK-norm
    # activations; MLP-only checkpointing retains these 4-D tensors.
    assert not any(len(shape) == 4 for shape in saved_shapes)
    loss.backward()
    torch.testing.assert_close(loss, expected)
    for a, b in zip(actual.parameters(), reference.parameters()):
        tol = 0.02 if dtype == torch.bfloat16 else 2e-6
        torch.testing.assert_close(a.grad, b.grad, atol=tol, rtol=10 * tol)
    assert counts == {"attention": 4, "mlp": 4}

    # Warmup callers explicitly allocate for their nonzero target while starting at zero.
    warmup = GPT(config).to(dtype=dtype)
    apply_activation_checkpointing(warmup, check_fn=lambda module: isinstance(module, Block))
    warmup.enable_log_kv_training(batch_size=1, B=3, recent_size=4, train_block=4,
                                  second_order_scale=0., allocate_second_order=True)
    warmup(inputs).sum().backward()
    warmup.set_log_kv_second_order_scale(0.2)
    warmup(inputs).sum().backward()
    assert all(block.attn.kv_cache.second_order for block in warmup.transformer.h)
