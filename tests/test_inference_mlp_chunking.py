"""Bound inference MLP temporaries while preserving attention, caches, and training gradients."""
import ast
import copy
import os
from pathlib import Path

import pytest
import torch

from litgpt import Config, GPT
from litgpt.kv_compression import cache_compression_snapshot


def config(**kwargs):
    values = dict(block_size=48, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                  padded_vocab_size=64, rotary_percentage=1., bias=False,
                  mlp_class_name="LLaMAMLP", intermediate_size=96, parallel_residual=False,
                  norm_class_name="RMSNorm")
    values.update(kwargs)
    return Config(**values)


@pytest.mark.parametrize("mlp", ["LLaMAMLP", "GptNeoxMLP", "GemmaMLP", "LLaMAMoE"])
@pytest.mark.parametrize("residual", ["sequential", "parallel", "shared"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_norm_mlp_residual_matches_full_forward(mlp, residual, dtype):
    torch.manual_seed(231)
    base = GPT(config(mlp_class_name=mlp, parallel_residual=residual != "sequential",
                      shared_attention_norm=residual == "shared", post_mlp_norm=True,
                      n_expert=3, n_expert_per_token=2, moe_intermediate_size=64)).to(dtype).eval()
    low = copy.deepcopy(base)
    low.config.inference_mlp_chunk_size = 4
    ids = torch.randint(0, 64, (2, 19))
    original = ids.clone()
    calls = []
    attention_calls = []
    for block in low.transformer.h:
        block.mlp.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
        block.attn.register_forward_pre_hook(lambda module, args: attention_calls.append(args[0].shape[1]))
    with torch.no_grad():
        expected = base(ids)
        actual = low(ids)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.equal(ids, original)
    assert calls == [4, 4, 4, 4, 3] * 2
    assert attention_calls == [19, 19]  # attention/routing is never split by the MLP optimization
    assert base.state_dict().keys() == low.state_dict().keys()


@pytest.mark.parametrize("training,grad", [(True, True), (True, False), (False, True)])
def test_training_and_grad_enabled_eval_keep_the_original_path(training, grad):
    torch.manual_seed(315)
    base = GPT(config()).train(training)
    low = copy.deepcopy(base)
    low.config.inference_mlp_chunk_size = 3
    calls = []
    low.transformer.h[0].mlp.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
    ids = torch.randint(0, 64, (1, 13))
    with torch.set_grad_enabled(grad):
        expected, actual = base(ids), low(ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if grad:
            expected.square().sum().backward()
            actual.square().sum().backward()
            for a, b in zip(base.parameters(), low.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    assert calls == [13]


@pytest.mark.parametrize("backend", ["dense", "logkv"])
def test_prefill_decode_and_cache_accounting_match(backend):
    torch.manual_seed(71)
    base = GPT(config()).eval()
    low = copy.deepcopy(base)
    low.config.inference_mlp_chunk_size = 4
    for model in (base, low):
        if backend == "dense":
            model.set_kv_cache(batch_size=1)
        else:
            model.set_log_kv_cache(batch_size=1, B=2, recent_size=4, prefill_block=2, second_order_scale=0.)
    ids = torch.randint(0, 64, (1, 25))
    with torch.inference_mode():
        for start, end in ((0, 19), (19, 24), (24, 25)):
            positions = torch.arange(start, end)
            a = base(ids[:, start:end], input_pos=positions, input_pos_maxp1=end)
            b = low(ids[:, start:end], input_pos=positions, input_pos_maxp1=end)
            torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-5)
            assert cache_compression_snapshot(base, end) == cache_compression_snapshot(low, end)
        if backend == "dense":
            for a, b in zip(base.transformer.h, low.transformer.h):
                torch.testing.assert_close(a.attn.kv_cache.k, b.attn.kv_cache.k, atol=3e-6, rtol=1e-5)
                torch.testing.assert_close(a.attn.kv_cache.v, b.attn.kv_cache.v, atol=3e-6, rtol=1e-5)


def test_prefill_input_is_not_modified_and_wide_projection_is_bounded():
    torch.manual_seed(118)
    model = GPT(config(inference_mlp_chunk_size=3)).eval()
    block = model.transformer.h[0]
    x = torch.randn(1, 37, 32)
    before = x.clone()
    sizes = []
    block.mlp.fc_1.register_forward_hook(lambda module, args, out: sizes.append(out.numel()))
    with torch.inference_mode():
        result = block(x, model.cos[:37][None], model.sin[:37][None])
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert result.data_ptr() != x.data_ptr()
    assert max(sizes) == 3 * 96 and sum(sizes) == 37 * 96


def eval_memory_configurer():
    path = Path(__file__).resolve().parents[1] / "eval.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_configure_inference_memory")
    scope = dict(os=os)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[node.name]


def test_eval_enables_the_same_budget_for_all_methods_and_allows_opt_out(monkeypatch):
    apply = eval_memory_configurer()
    monkeypatch.delenv("LOGKV_EVAL_MLP_CHUNK", raising=False)
    cfg = config()
    assert cfg.inference_mlp_chunk_size == 0  # model/training defaults retain the original forward
    assert apply(cfg) == cfg.inference_mlp_chunk_size == 2048
    assert apply(cfg, {"inference_mlp_chunk_size": 1024}) == 1024
    monkeypatch.setenv("LOGKV_EVAL_MLP_CHUNK", "0")
    assert apply(cfg, {"inference_mlp_chunk_size": 1024}) == cfg.inference_mlp_chunk_size == 0


@pytest.mark.parametrize("value", ["-1", "invalid"])
def test_invalid_eval_budget_is_rejected(monkeypatch, value):
    monkeypatch.setenv("LOGKV_EVAL_MLP_CHUNK", value)
    with pytest.raises(ValueError, match="integer >= 0"):
        eval_memory_configurer()(config())
    with pytest.raises(ValueError, match="inference_mlp_chunk_size"):
        config(inference_mlp_chunk_size=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for allocator peak comparison")
def test_cuda_peak_is_lower_for_long_prefill():
    cfg = config(block_size=8192, n_embd=512, n_head=8, n_query_groups=2, intermediate_size=2048)
    base = GPT(cfg).cuda().bfloat16().eval()
    low = copy.deepcopy(base)
    low.config.inference_mlp_chunk_size = 256
    tokens = torch.randint(0, 64, (1, 8192), device="cuda")
    peaks = []
    with torch.inference_mode():
        for model in (base, low):
            model(tokens[:, :16], lm_head_start=15)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            model(tokens, lm_head_start=8191)
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated())
    assert peaks[1] < peaks[0] * .8
