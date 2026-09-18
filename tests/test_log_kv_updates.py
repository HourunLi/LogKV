"""Exact reference checks for fused cache writes and ordered centroid updates."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

import litgpt.log_kv_cache as kv


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")


def cache_for(device, dtype, B=3, dim=8, groups=2, vdim=None):
    return kv.LogStructuredKVCache(
        (2, groups, 512, dim), (2, groups, 512, vdim or dim), B=B, recent_size=4,
        device=device, dtype=dtype, semantic_clusters=True, cluster_k_max=8,
        allocate_second_order=False, seg_gap_max=2, seg_block_level=2, seg_forget=.3,
        cos_cache=torch.ones(512, dim, device=device), sin_cache=torch.zeros(512, dim, device=device),
        rope_n_elem=dim,
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [8, 128])
@pytest.mark.parametrize("forget", [0., .3, 1.])
def test_centroid_metadata_preserves_run_order(device, dtype, dim, forget):
    if device == "cuda":
        assert kv._triton_updates() is not None
    cache = cache_for(device, dtype, dim=dim)
    cache.seg_forget = forget
    # Interleaved clusters, uneven runs, repeated updates, non-integral forget,
    # zero old mass, and cancellation distinguish sequential from tree sums.
    values = [2.**24, 1., -2.**24, 1., 2., 3., 4., 5., 6., 7., 8.]
    k = torch.tensor(values, dtype=dtype, device=device).view(1, 1, -1, 1).expand(1, 1, -1, dim)
    runs = [[0, 0, 4, False], [1, 0, 2, True], [0, 1, 3, False], [2, 0, 2, True]]
    cache.centroid.fill_(.125)
    cache.n_eff.flatten()[0] = 7.
    expected_mu, expected_n = cache.centroid.clone().view(-1, dim), cache.n_eff.clone().flatten()
    starts, start = [], 0
    for _, _, n, _ in runs:
        starts.append(start)
        start += n
    for j in sorted(range(len(runs)), key=lambda j: runs[j][0]):
        _, ci, n, new = runs[j]
        acc = torch.zeros(dim, device=device)
        for i in range(starts[j], starts[j] + n):
            acc = acc + k[0, 0, i].float()
        pre = expected_n[ci] * (forget if new else 1.)
        denom = pre + n
        expected_mu[ci] = torch.where(pre > 0, (pre * expected_mu[ci] + acc) / denom, acc / n)
        expected_n[ci] = denom
    n = len(values)
    with patch.object(cache, "_semantic_append_entries_batched"):
        cache._semantic_apply_join_plan([(0, 0, 0)], [n], list(range(n)), list(range(n)),
                                        list(range(n)), runs, [], n, k, k,
                                        torch.arange(n, device=device).view(1, -1))
    torch.testing.assert_close(cache.centroid.view(-1, dim), expected_mu, atol=0, rtol=0)
    torch.testing.assert_close(cache.n_eff.flatten(), expected_n, atol=0, rtol=0)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("B,dim,vdim", [(3, 8, 12), (64, 128, 128)])
def test_fused_ladder_carry_survivors_and_clears(dtype, B, dim, vdim):
    assert kv._triton_updates() is not None
    torch.manual_seed(52)
    actual = cache_for("cuda", dtype, B, dim, vdim=vdim)
    expected = deepcopy(actual)
    lanes = [(0, 0, 0), (0, 1, 7), (1, 0, 3), (1, 1, 1)]
    # Several appends force mixed fills, cross-source pairs, survivor moves,
    # multiple carry levels; the B=64 case also reaches the allocated top level.
    for step in range(4):
        counts = [B + 1, 3 * B + step, 1, 0 if step == 0 else B - 1]
        n = sum(counts)
        # fp32 staging tests cast-before-merge, even with bf16 storage.
        k, v = torch.randn(n, dim, device="cuda"), torch.randn(n, vdim, device="cuda")
        w = torch.randint(0, 5, (n,), device="cuda").float()
        p = torch.arange(n, device="cuda") + 2**33
        block = (k, v, w, None, None, None, None, None, p, p + 3, p * w.long(), p, w == 0)
        with patch.object(kv, "_triton_updates", return_value=None):
            expected._semantic_append_entries_batched(lanes, counts, block)
        actual._semantic_append_entries_batched(lanes, counts, block)
        torch.cuda.synchronize()
        assert actual._semantic_counts == expected._semantic_counts
        for name, tensor in actual.named_buffers():
            torch.testing.assert_close(tensor, dict(expected.named_buffers())[name], atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_route_replay_output_and_gradients(dtype, device):
    # The CPU case also validates this fixture when CUDA tests are skipped.
    if device == "cuda":
        assert kv._triton_updates() is not None
    torch.manual_seed(53)
    q = torch.randn(2, 4, 96, 8, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, 128, 8, device=device, dtype=dtype)[:, :, 16:112].requires_grad_()
    v = torch.randn(2, 128, 2, 8, device=device, dtype=dtype).transpose(1, 2)[:, :, 16:112].requires_grad_()
    upstream = torch.randn_like(q)

    def run():
        cache = cache_for(device, dtype)
        with patch.object(cache, "route_and_flush_batch", wraps=cache.route_and_flush_batch) as route:
            out = kv.LogKVStreamTrainingAttention.apply(q, k, v, cache, 8 ** -.5, cache.recent_size, 0., k)
            grad = torch.autograd.grad(out, (q, k, v), upstream)
        assert any(call.kwargs.get("record_op_log") for call in route.call_args_list)
        assert any(call.kwargs.get("replay_op_log") is not None for call in route.call_args_list)
        return (out, *grad), {name: t.clone() for name, t in cache.named_buffers()}

    with patch.object(kv, "_triton_updates", return_value=None):
        reference, buffers = run()
    actual, actual_buffers = run()
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for name, t in actual_buffers.items():
        torch.testing.assert_close(t, buffers[name], atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("corrupt", [False, True])
def test_update_diagnostic_keeps_centroid_error_details(corrupt):
    from types import SimpleNamespace
    from unused.benchmark_log_kv_updates import diagnose_updates

    def update(k, mu, ne, meta, start, count):
        mu.copy_(torch.tensor([[2., 3.]]))
        ne.fill_(2.)
        if corrupt:
            mu[0, 1] += .25

    backend = SimpleNamespace(centroid=update)
    # One run, two tokens, no previous centroid mass.
    k = torch.tensor([[1., 2.], [3., 4.]])
    meta = torch.tensor([[0], [0], [0], [2], [1065353216]])  # fp32 1.0 bits
    mu, ne = torch.zeros(1, 2), torch.zeros(1)
    with diagnose_updates(backend) as calls:
        if corrupt:
            with pytest.raises(AssertionError, match=r"centroid update 1[\s\S]*Mismatched elements"):
                backend.centroid(k, mu, ne, meta, 0, 1)
        else:
            backend.centroid(k, mu, ne, meta, 0, 1)
        assert calls["centroid"] == 1
    assert backend.centroid is update  # instrumentation does not leak into timing


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [128, 256])
def test_centroid_feature_warps_read_same_old_count(dtype, dim):
    backend = kv._triton_updates()
    assert backend is not None
    nr, length = 256, 292
    # Production failure matched using 4106 instead of the old count 3814.
    # Read-only computation must finish before any cluster state is committed.
    k = torch.zeros(nr, length, dim, device="cuda", dtype=dtype)
    k[:, 0, :] = 162.806640625
    k = k.view(-1, dim)
    mu = torch.full((nr, dim), .5797467827796936, device="cuda")
    ne = torch.full((nr,), 3814., device="cuda")
    expected_mu, expected_ne = mu.clone(), ne.clone()
    rows = torch.arange(nr, device="cuda")
    lengths = torch.full_like(rows, length)
    meta = torch.stack((rows, rows, rows * length, lengths, torch.full_like(rows, 1065353216)))
    sums = torch.segment_reduce(k.float(), "sum", lengths=lengths, unsafe=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(32):
            denom = expected_ne + length
            expected_mu = (expected_ne[:, None] * expected_mu + sums) / denom[:, None]
            expected_ne = denom
            if _ == 0:
                before_mu, before_ne = mu.clone(), ne.clone()
                out_mu, out_ne = torch.empty_like(mu), torch.empty_like(ne)
                backend._centroid[(nr,)](k, mu, ne, out_mu, out_ne, meta, nr, 0, dim, dim,
                                         num_warps=4, enable_fp_fusion=False)
                torch.testing.assert_close(mu, before_mu, atol=0, rtol=0)
                torch.testing.assert_close(ne, before_ne, atol=0, rtol=0)
                torch.testing.assert_close(out_mu, expected_mu, atol=0, rtol=0)
                torch.testing.assert_close(out_ne, expected_ne, atol=0, rtol=0)
            backend.centroid(k, mu, ne, meta, 0, nr)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(mu, expected_mu, atol=0, rtol=0)
    torch.testing.assert_close(ne, expected_ne, atol=0, rtol=0)


@pytest.mark.parametrize("B", [2, 3, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_ladder_plan_partitions_rows_and_matches_scalar(B, dtype):
    torch.manual_seed(713)
    cache = cache_for("cpu", dtype, B=B)
    reference = deepcopy(cache)
    lanes = [(0, 0, 0), (0, 1, 7), (1, 0, 4), (1, 1, 1)]
    make_indices = cache._semantic_index_tensors

    def checked_indices(*args):
        fs, fd, si, gi, pi, vi, sd, cd = result = make_indices(*args)
        for index in (fs, fd, si, gi, sd, cd):
            assert index.unique().numel() == index.numel()
        assert pi.numel() % 2 == 0
        # Each pool row participates in exactly one pair OR survives.
        torch.testing.assert_close(torch.cat((pi, vi)).sort().values,
                                   torch.arange(si.numel() + gi.numel()))
        destinations = torch.cat((sd, cd))
        assert destinations.unique().numel() == destinations.numel()
        return result

    mass = 0.
    with patch.object(cache, "_semantic_index_tensors", side_effect=checked_indices):
        for step in range(8):
            counts = torch.randint(0, 3 * B + 3, (len(lanes),)).tolist()
            n = sum(counts)
            k, v = torch.randn(n, 8, dtype=dtype), torch.randn(n, 8, dtype=dtype)
            w = torch.randint(0, 5, (n,)).float()
            mass += float(w.sum())
            pos = torch.arange(n) + 2**33 + 1000 * step
            block = (k, v, w, None, None, None, None, None, pos, pos + 1, pos * w.long(), pos, w == 0)
            cache._semantic_append_entries_batched(lanes, counts, block)
            start = 0
            for (b, g, c), count in zip(lanes, counts):
                for i in range(start, start + count):
                    reference._semantic_append_entry(b, g, c, tuple(x[i] if x is not None else None for x in block))
                start += count
            assert float(cache.level_w.sum()) == mass
            assert cache._semantic_counts == reference._semantic_counts
            for name, tensor in cache.named_buffers():
                torch.testing.assert_close(tensor, dict(reference.named_buffers())[name], atol=0, rtol=0,
                                           msg=lambda detail: f"B={B}, step={step}, {name}\n{detail}")


def test_benchmark_checks_later_iterations(monkeypatch):
    import sys
    from unused import benchmark_log_kv_updates as benchmark

    monkeypatch.setattr(sys, "argv", ["benchmark", "--device", "cpu", "--sequence", "32", "--chunk", "8",
                                      "--batch", "1", "--groups", "1", "--dim", "8", "--iters", "1"])
    route = kv.LogStructuredKVCache.route_and_flush_batch
    recorded = 0

    def corrupt_second_warmup(cache, *args, **kwargs):
        nonlocal recorded
        route(cache, *args, **kwargs)
        if kwargs.get("record_op_log"):
            recorded += 1  # reference, first warmup, second warmup
            if recorded == 3:
                cache.n_eff[0, 0, 0] += 1

    with patch.object(kv.LogStructuredKVCache, "route_and_flush_batch", corrupt_second_warmup):
        with pytest.raises(AssertionError, match="torch/route/iteration=1: n_eff"):
            benchmark.main()
