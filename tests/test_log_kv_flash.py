"""First-order SDPA transform, gradients, dispatch, and streaming replay."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

import litgpt.log_kv_cache as kv


def _case(device, dtype, heads=4, dim=128):
    torch.manual_seed(42)
    q = torch.randn(2, heads, 5, dim, device=device, dtype=dtype).requires_grad_()
    k = torch.randn(2, 2, 12, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    # Different lane lengths, including an entirely empty pooled prefix.
    valid = torch.tensor([[[True, False, True, False], [False] * 4],
                          [[True] * 4, [False, True, False, True]]], device=device)
    # Garbage in masked slots must never leak into attention or gradients.
    k[..., :4, :].masked_fill_(~valid.unsqueeze(-1), 1000)
    v[..., :4, :].masked_fill_(~valid.unsqueeze(-1), 1000)
    w = torch.ones(2, 2, 12, device=device)
    w[..., :4] = torch.tensor([1., 3., 128., 32768.], device=device)
    m = torch.tensor([1., 3., 2., 3.], device=device).expand(2, 2, 4)
    return q, k.requires_grad_(), v.requires_grad_(), w, m, valid


def _math_sdpa(q, k, v, w, scale, lam, causal_tail, slot_M, slot_valid, check_valid):
    # Exercise the production transform with math SDPA on CPU. CUDA tests below
    # exercise the actual automatic Flash dispatcher without this substitution.
    qa, ka, va = kv._slot_sdpa_inputs(q, k, v, w, slot_M, slot_valid, scale, lam)
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            qa, ka, va, scale=scale, enable_gqa=q.size(1) != k.size(1),
            attn_mask=causal_lower_right(q.size(2), k.size(2)) if causal_tail else None,
        )
    return out[..., :v.size(-1)]


@pytest.mark.parametrize("heads,tail,lam", [(2, 0, 1.0), (4, 5, 1.0), (4, 5, 0.0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_augmented_sdpa_output_and_gradients(heads, tail, lam, dtype):
    q, k, v, w, m, valid = _case("cpu", dtype, heads)
    kwargs = dict(scale=128 ** -0.5, lam=lam, causal_tail=tail, slot_M=m,
                  slot_valid=valid, second_order_scale=0.0)
    reference = kv.log_kv_slot_attention(q, k, v, w, **kwargs)
    with patch.object(kv, "_slot_flash_attention", _math_sdpa):
        actual = kv.log_kv_slot_attention(q, k, v, w, **kwargs)
    grad = torch.randn_like(reference)
    expected_grads = torch.autograd.grad(reference, (q, k, v), grad)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    tol = 2e-5 if dtype == torch.float32 else 0.04 if dtype == torch.bfloat16 else 0.005
    for a, b in zip((actual, *actual_grads), (reference, *expected_grads)):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
    for g in actual_grads[1:]:
        assert torch.count_nonzero(g[..., :4, :].masked_select(~valid.unsqueeze(-1))) == 0
    # Future exact-tail tokens cannot affect earlier queries.
    if tail:
        changed = v.detach().clone()
        changed[..., -1, :] += 100
        out = _math_sdpa(q, k, changed, w, kwargs["scale"], lam, tail, m, valid, True)
        torch.testing.assert_close(out[..., :-1, :], actual[..., :-1, :])


def test_nonzero_second_order_and_custom_mask_bypass_flash():
    q, k, v, w, m, valid = _case("cpu", torch.float32)
    with patch.object(kv, "_slot_flash_attention", side_effect=AssertionError("unexpected Flash dispatch")):
        kv.log_kv_slot_attention(q, k, v, w, scale=0.1, second_order_scale=0.2)
        kv.log_kv_slot_attention(q, k, v, w, scale=0.1, second_order_scale=0.0,
                                 mask=torch.ones(5, 12, dtype=torch.bool))


def test_streaming_replay_with_sdpa_matches_original():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 24, 8, requires_grad=True)
    k = torch.randn(1, 2, 24, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    cache = kv.LogStructuredKVCache(
        k.shape, v.shape, B=2, recent_size=4, dtype=torch.float32,
        semantic_clusters=True, cluster_k_max=8, semantic_flush_granularity=4,
        rope_n_elem=8, cos_cache=torch.ones(24, 8), sin_cache=torch.zeros(24, 8),
    )
    upstream = torch.randn_like(q)

    def run():
        y = kv.LogKVStreamTrainingAttention.apply(q, k, v, cache, 8 ** -0.5, 4, 0.0, k)
        grads = torch.autograd.grad(y, (q, k, v), upstream)
        return y, grads, cache.level_k.clone(), cache.level_w.clone(), cache.take_op_log()

    expected, eg, ek, ew, elog = run()
    with patch.object(kv, "_slot_flash_attention", side_effect=_math_sdpa) as dispatch:
        actual, ag, ak, aw, alog = run()
        assert dispatch.call_count == 12  # six forward blocks + six replay blocks
    for a, b in zip((actual, *ag), (expected, *eg)):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    for a, b in zip((ak, aw, *alog), (ek, ew, *elog)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tail", [0, 5])
def test_cuda_real_flash_forward_backward(dtype, tail):
    q, k, v, w, m, valid = _case("cuda", dtype)
    args = (q, k, v, w, 128 ** -0.5, 1.0, tail, m, valid, True)
    fast = kv._slot_flash_attention(*args)
    if fast is None:
        pytest.skip("Flash backend does not support this GPU/shape")
    with patch.object(kv, "_slot_flash_attention", return_value=None):
        reference = kv.log_kv_slot_attention(q, k, v, w, scale=args[4], causal_tail=tail,
                                             slot_M=m, slot_valid=valid, second_order_scale=0.0)
    grad = torch.randn_like(fast)
    fg = torch.autograd.grad(fast, (q, k, v), grad)
    rg = torch.autograd.grad(reference, (q, k, v), grad)
    tol = 0.04 if dtype == torch.bfloat16 else 0.005
    for a, b in zip((fast, *fg), (reference, *rg)):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
    # The public entry point must use Flash, including its backward, not math SDPA.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        out = kv.log_kv_slot_attention(q, k, v, w, scale=args[4], causal_tail=tail,
                                      slot_M=m, slot_valid=valid, second_order_scale=0.0)
        torch.autograd.grad(out, (q, k, v), grad)
    names = {event.key for event in prof.key_averages()}
    assert any("_scaled_dot_product_flash_attention" in name and "backward" not in name for name in names)
    assert any("_scaled_dot_product_flash_attention_backward" in name for name in names)
    assert not any("_scaled_dot_product_attention_math" in name for name in names)
