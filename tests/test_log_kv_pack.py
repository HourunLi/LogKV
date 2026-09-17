"""Mid-anchor packing: independent readout reference, gradients and cache lifetime.

CPU tests exercise the same pack/replay integration using math SDPA. CUDA tests
require the actual Triton kernel and Flash forward/backward; they never emulate
GPU execution or silently substitute the torch packer.
"""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

import litgpt.log_kv_cache as kv
import litgpt.log_kv_pack as packing


def _cache(dtype=torch.float32, device="cpu", dim=16, rope=None, populated=True, cos_dtype=torch.float32):
    rope = dim if rope is None else rope
    phases = torch.outer(torch.arange(128, device=device).float(),
                         10000 ** (-torch.arange(0, rope, 2, device=device).float() / max(rope, 1))).repeat(1, 2)
    cache = kv.LogStructuredKVCache(
        (2, 2, 128, dim), (2, 2, 128, dim), B=4, recent_size=8,
        dtype=dtype, device=torch.device(device), semantic_clusters=True, cluster_k_max=8,
        semantic_anchor_mode="mid", semantic_pack_backend="torch", semantic_flush_granularity=4,
        allocate_second_order=False, rope_n_elem=rope,
        cos_cache=phases.cos().to(cos_dtype), sin_cache=phases.sin().to(cos_dtype),
    )
    if populated:
        # Widely separated positions, carries, unequal lane lengths and one empty lane.
        for b, g, c, positions in [(0, 0, 0, [1, 2, 7, 33, 34, 80]),
                                    (0, 0, 3, [4, 9]), (1, 0, 0, [0, 127]), (1, 1, 2, [5])]:
            x = torch.randn(len(positions), dim, device=device, dtype=dtype)
            cache._semantic_append_token_entries(b, g, c, x, x * 2, torch.tensor(positions, device=device))
        cache._semantic_insert_pad(1, 1, 2, 2, record=False)
        cache.recent_count = 3
        cache.recent_k[:, :, :3].normal_()
        cache.recent_v[:, :, :3].normal_()
    return cache


def _math_packed(q, k, v, scale, tail, v_dim):
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            q, k, v, scale=scale, dropout_p=0., enable_gqa=q.size(1) != k.size(1),
            attn_mask=causal_lower_right(q.size(2), k.size(2)) if tail else None,
        )
    return out[..., :v_dim]


def _reference_pack(cache, q, k, v, plan):
    state = kv.append_exact_tokens(cache.get_attention_state(plan=plan), k, v)
    return kv._slot_sdpa_inputs(q, state.slot_k, state.slot_v, state.slot_w,
                                state.M_s, state.slot_valid, q.size(-1) ** -0.5, 1.)


def test_mid_plan_ignores_endpoints_and_never_sorts_or_reads_device_scalars():
    from torch.utils._python_dispatch import TorchDispatchMode
    cache = _cache()
    expected = cache._semantic_attention_plan()
    # Independent midpoint/reference ordering from the existing positional helper.
    from litgpt.log_kv_position import mid_anchor
    mids = mid_anchor(cache.level_p_lo, cache.level_p_hi, cache.level_sum_wp, cache.level_w).flatten(2)
    torch.testing.assert_close(expected[1][expected[3]], mids.gather(2, expected[0])[expected[3]])
    for b in range(cache.batch_size):
        for g in range(cache.n_groups):
            old_order = [c * cache.L_alloc * cache.B_prime + level * cache.B_prime + i
                         for c in range(cache.K_max) for level in reversed(range(cache.L_alloc))
                         for i in range(cache._semantic_counts[b][g][c][level])
                         if cache.level_w[b, g, c, level, i] > 0]
            assert expected[0][b, g][expected[3][b, g]].tolist() == old_order
    # Corrupting endpoints must not alter mid readout or trigger an endpoint gather.
    cache.level_p_lo.fill_(100000)
    cache.level_p_hi.fill_(-100000)
    names = []

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            names.append(str(func))
            return func(*args, **(kwargs or {}))

    with Count(), patch.object(kv, "dedup_anchors", side_effect=AssertionError("multi-anchor path")):
        actual = cache._semantic_attention_plan()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not any("sort" in n or "flip" in n or "_local_scalar_dense" in n for n in names)
    assert actual[3].sum() == (cache.level_w > 0).sum()
    assert torch.all(actual[2] == 1)
    assert not actual[3][0, 1].any()  # entirely empty lane


def test_multi_anchor_mode_preserves_original_three_positions():
    cache = _cache(populated=False)
    cache.semantic_anchor_mode = "multi"
    x = torch.randn(2, 16)
    cache._semantic_append_token_entries(0, 0, 0, x, x, torch.tensor([1, 101]))
    # Force one carry with two distinct, distant positions in the first pair.
    cache._semantic_append_token_entries(0, 0, 0, torch.randn(4, 16), torch.randn(4, 16), torch.tensor([102, 103, 104, 105]))
    multi = cache._semantic_attention_plan()
    assert multi[1][0, 0][multi[3][0, 0]][:3].tolist() == [1, 51, 101]
    cache.semantic_anchor_mode = "mid"
    mid = cache._semantic_attention_plan()
    assert mid[1][0, 0][mid[3][0, 0]][0].item() == 51
    assert mid[3].sum() < multi[3].sum()


def test_pack_rejects_incomplete_workspace_reuse():
    cache = _cache()
    plan = cache._semantic_attention_plan()
    k = torch.randn(2, 2, 1, 16)
    for kwargs in ({"skip_pooled": True}, {"recent_start": 1}, {"backend": "typo"}):
        with pytest.raises(ValueError):
            packing.pack_mid_kv(cache, plan, k, k, 24, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rope,populated", [(0, True), (8, True), (16, True), (16, False)])
def test_torch_pack_matches_existing_materialization_and_current_gradients(dtype, rope, populated):
    cache = _cache(dtype, rope=rope, populated=populated)
    # Noncontiguous token and feature strides, as in real QKV projections/slices.
    k = torch.randn(2, 2, 10, 32, dtype=dtype)[:, :, ::2, ::2].requires_grad_()
    v = torch.randn_like(k).transpose(2, 3).contiguous().transpose(2, 3).requires_grad_()
    q = torch.randn(2, 4, 5, 16, dtype=dtype)
    plan = cache._semantic_attention_plan()
    _, ek, ev = _reference_pack(cache, q, k, v, plan)
    ak, av = packing.pack_mid_kv(cache, plan, k, v, 24)
    torch.testing.assert_close(ak, ek, atol=0, rtol=0)
    torch.testing.assert_close(av, ev, atol=0, rtol=0)
    gk, gv = torch.randn_like(ak), torch.randn_like(av)
    expected = torch.autograd.grad((ek, ev), (k, v), (gk, gv))
    actual = torch.autograd.grad((ak, av), (k, v), (gk, gv))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("segments", [False, True])
def test_mid_packed_streaming_replay_and_gradients(segments):
    torch.manual_seed(18)
    cache = _cache(populated=False)
    if segments:
        cache.seg_gap_max, cache.seg_block_level = 1, 2
    q = torch.randn(2, 4, 40, 16, requires_grad=True)
    k = torch.randn(2, 2, 40, 16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    upstream = torch.randn_like(q)

    def run():
        y = kv.LogKVStreamTrainingAttention.apply(q, k, v, cache, .25, 8, 0., k)
        grads = torch.autograd.grad(y, (q, k, v), upstream)
        return (y, *grads, cache.level_k.clone(), cache.level_w.clone())

    with patch.object(kv, "_mid_flash_supported", return_value=False):
        expected = run()
    with patch.object(kv, "_mid_flash_supported", return_value=True), \
            patch.object(kv, "_packed_flash_attention", _math_packed), \
            patch.object(cache, "_semantic_attention_plan", wraps=cache._semantic_attention_plan) as planner:
        actual = run()
        assert planner.call_count == 5  # saved plans are reused in backward
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    assert cache._mid_decode_state is None  # no persistent workspace during training


def test_decode_workspace_reuses_prefix_and_invalidates_on_flush_reset_dtype_and_rope():
    torch.manual_seed(4)
    cache = _cache(populated=False)
    q = torch.randn(2, 4, 1, 16)
    with torch.no_grad(), patch.object(kv, "_mid_flash_supported", return_value=True), \
            patch.object(kv, "_packed_flash_attention", _math_packed), \
            patch.object(packing, "pack_mid_kv", wraps=packing.pack_mid_kv) as packer:
        for step in range(22):
            k, v = torch.randn(2, 2, 1, 16), torch.randn(2, 2, 1, 16)
            had_workspace = cache._mid_decode_state is not None
            actual = kv.log_kv_chunk_attention(cache, q, k, v, .25, 0., reuse_prefix=True)
            assert packer.call_args.kwargs["skip_pooled"] == had_workspace
            if had_workspace:
                assert packer.call_args.kwargs["recent_start"] == cache.recent_count - 1
            reference = kv.log_kv_chunk_attention(cache, q, k, v, .25, 0., reuse_prefix=False)
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
            will_flush = cache.recent_count == cache.recent_size
            cache.add_recent(k, v, k_raw=k)
            if will_flush:
                assert cache._mid_decode_state is None
        # RoPE replacement/in-place update rebuilds the prefix on next access.
        cache.cos_cache.mul_(.9)
        kv.log_kv_chunk_attention(cache, q, k, v, .25, 0., reuse_prefix=True)
        assert not packer.call_args.kwargs["skip_pooled"]
        cache._convert_dtype(torch.bfloat16)
        assert cache._mid_decode_state is None
        cache._mid_decode_state = "sentinel"
        cache.to(dtype=torch.float32)
        assert cache._mid_decode_state is None
        cache._mid_decode_state = "sentinel"
        cache.reset_parameters()
        assert cache._mid_decode_state is None


def test_mid_full_model_checkpoint_and_decode_pending_token():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
    from litgpt.config import Config
    from litgpt.model import GPT, Block
    torch.manual_seed(8)
    config = Config(block_size=32, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                    vocab_size=41, padding_multiple=1, rotary_percentage=1.0)
    reference = GPT(config)
    fast = deepcopy(reference)
    options = dict(batch_size=1, B=3, recent_size=4, second_order_scale=0.,
                   semantic_clusters=True, cluster_k_max=8, semantic_flush_granularity=4,
                   semantic_anchor_mode="mid", semantic_pack_backend="torch")
    for model in (reference, fast):
        model.enable_log_kv_training(**options, train_block=4, allocate_second_order=False)
    apply_activation_checkpointing(fast, check_fn=lambda m: isinstance(m, Block))
    inputs, targets = torch.randint(41, (1, 24)), torch.randint(41, (1, 24))
    loss = reference(inputs, targets=targets, loss_chunk_size=7)
    loss.backward()
    with patch.object(kv, "_mid_flash_supported", return_value=True), patch.object(kv, "_packed_flash_attention", _math_packed):
        actual = fast(inputs, targets=targets, loss_chunk_size=7)
        actual.backward()
    torch.testing.assert_close(actual, loss, atol=2e-5, rtol=2e-5)
    for a, b in zip(fast.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-4)

    # Exercise the inference caller's odd prefill and pending-single-token branch.
    reference.disable_log_kv_training()
    reference.eval()
    expected = deepcopy(reference)
    for model in (reference, expected):
        model.set_log_kv_cache(**options, prefill_block=4)
    with torch.no_grad():
        for start, end in [(0, 7), *[(i, i + 1) for i in range(7, 18)]]:
            pos = torch.arange(start, end)
            y = expected(inputs[:, start:end], pos)
            with patch.object(kv, "_mid_flash_supported", return_value=True), patch.object(kv, "_packed_flash_attention", _math_packed):
                z = reference(inputs[:, start:end], pos)
            torch.testing.assert_close(z, y, atol=3e-5, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rope,cos_half", [(0, False), (64, False), (128, False), (128, True)])
def test_cuda_triton_pack_and_flash_gradients(dtype, rope, cos_half):
    assert packing._triton_packer() is not None  # Missing Triton must fail CUDA acceptance.
    cache = _cache(dtype, "cuda", dim=128, rope=rope, cos_dtype=dtype if cos_half else torch.float32)
    cache.semantic_pack_backend = "triton"
    q = torch.randn(2, 4, 5, 128, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, 10, 256, device="cuda", dtype=dtype)[:, :, ::2, ::2].requires_grad_()
    v = torch.randn_like(k, requires_grad=True)
    assert kv._mid_flash_supported(cache, q, k, v, 128 ** -.5)
    plan = cache._semantic_attention_plan()
    qa, ek, ev = _reference_pack(cache, q, k, v, plan)
    ak, av = packing.pack_mid_kv(cache, plan, k, v, 136)
    tol = .02 if dtype == torch.bfloat16 else .003
    torch.testing.assert_close(ak, ek, atol=tol, rtol=tol)
    torch.testing.assert_close(av, ev, atol=0, rtol=0)
    expected = _math_packed(qa, ek, ev, 128 ** -.5, 5, 128)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        actual = kv.log_kv_chunk_attention(cache, q, k, v, 128 ** -.5, 0., attention_plan=plan)
        grad = torch.randn_like(actual)
        ag = torch.autograd.grad(actual, (q, k, v), grad)
    eg = torch.autograd.grad(expected, (q, k, v), grad)
    for a, b in zip((actual, *ag), (expected, *eg)):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
    names = {e.key for e in prof.key_averages()}
    assert any("_scaled_dot_product_flash_attention_backward" in n for n in names)
    assert not any("_scaled_dot_product_attention_math" in n for n in names)
    with torch.no_grad():
        for step in range(12):
            ki, vi, qi = k[:, :, :1].detach(), v[:, :, :1].detach(), q[:, :, :1].detach()
            actual = kv.log_kv_chunk_attention(cache, qi, ki, vi, 128 ** -.5, 0., reuse_prefix=True)
            reference = kv.log_kv_chunk_attention(cache, qi, ki, vi, 128 ** -.5, 0.)
            torch.testing.assert_close(actual, reference, atol=tol, rtol=tol)
            cache.add_recent(ki, vi, k_raw=ki)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
def test_cuda_triton_streaming_replay_bf16():
    assert packing._triton_packer() is not None
    torch.manual_seed(57)
    cache = _cache(torch.bfloat16, "cuda", dim=128, populated=False)
    q = torch.randn(2, 4, 40, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 2, 40, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    assert kv._mid_flash_supported(cache, q, k, v, 128 ** -.5)
    grad = torch.randn_like(q)

    def run(backend):
        cache.semantic_pack_backend = backend
        y = kv.LogKVStreamTrainingAttention.apply(q, k, v, cache, 128 ** -.5, 8, 0., k)
        return y, *torch.autograd.grad(y, (q, k, v), grad)

    expected = run("torch")
    actual = run("triton")
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=.025, rtol=.025)
