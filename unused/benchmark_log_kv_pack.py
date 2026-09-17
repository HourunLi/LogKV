"""A/B anchor count and input preparation on a real CUDA device, without a checkpoint.

python unused/benchmark_log_kv_pack.py --device cuda
CPU smoke: --device cpu --sequence 128 --chunk 8 --dim 16 --groups 2 --iters 3
Times include host dispatch and device work; compilation is excluded by warmup.
This benchmarks one layer/chunk, not optimizer-step throughput.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import litgpt.log_kv_cache as kv
from litgpt.log_kv_pack import pack_mid_kv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence", type=int, default=32768)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    if args.sequence < args.chunk or args.chunk < 2 or args.dim % 2 or args.iters < 1:
        parser.error("require sequence >= chunk >= 2, even dim and positive iters")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu for a small smoke check")
    torch.manual_seed(49)
    torch.set_num_threads(1)
    dev, dtype = torch.device(args.device), torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    n, t, d, g = args.sequence, args.chunk, args.dim, args.groups
    positions = torch.arange(n, device=dev).float()
    phases = torch.outer(positions, 10000 ** (-torch.arange(0, d, 2, device=dev).float() / d)).repeat(1, 2)
    cache = kv.LogStructuredKVCache(
        (1, g, n, d), (1, g, n, d), B=64, recent_size=t, device=dev, dtype=dtype,
        semantic_clusters=True, cluster_k_max=8, semantic_anchor_mode="mid",
        semantic_flush_granularity=t, allocate_second_order=False,
        cos_cache=phases.cos(), sin_cache=phases.sin(), rope_n_elem=d,
    )
    # Recurrent prototypes populate semantic clusters instead of benchmarking an empty cache.
    prototypes = torch.randn(1, g, 8, d, device=dev, dtype=dtype)
    with torch.no_grad():
        for start in range(0, n - t, t):
            count = min(t, n - t - start)
            labels = torch.randint(8, (1, g, count), device=dev)
            raw = prototypes.gather(2, labels[..., None].expand(-1, -1, -1, d))
            raw = raw + .1 * torch.randn_like(raw)
            rotated = torch.cat((-raw[..., d // 2:], raw[..., :d // 2]), dim=-1)
            k = (raw * cache.cos_cache[start:start + count] + rotated * cache.sin_cache[start:start + count]).to(dtype)
            cache.add_recent(k, torch.randn_like(k), k_raw=raw)
    q = torch.randn(1, g * 2, t, d, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(1, g, t, d, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    upstream = torch.randn_like(q)
    scale, dim = d ** -.5, ((d + 8) // 8) * 8
    if dev.type == "cuda" and not kv._mid_flash_supported(cache, q, k, v, scale):
        raise RuntimeError("This layout cannot use Flash SDPA; refusing a misleading Flash benchmark")

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def measure(fn):
        for _ in range(3):
            fn()
        sync()
        baseline = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        timings = []
        for _ in range(args.iters):
            start = time.perf_counter()
            fn()
            sync()
            timings.append(1000 * (time.perf_counter() - start))
        peak = torch.cuda.max_memory_allocated(dev) - baseline if dev.type == "cuda" else None
        return round(statistics.median(timings), 3), round(peak / 2**20, 2) if peak is not None else None

    variants = [("multi_reference", "multi", None), ("mid_reference", "mid", None), ("mid_torch", "mid", "torch")]
    if dev.type == "cuda":
        variants.append(("mid_triton", "mid", "triton"))
    mid_reference = None
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                      "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev),
                      "sequence": n, "chunk": t, "groups": g, "dim": d}), flush=True)
    for name, mode, backend in variants:
        cache.semantic_anchor_mode = mode
        plan = cache._semantic_attention_plan()

        def prepare():
            if backend is None:
                state = kv.append_exact_tokens(cache.get_attention_state(plan=plan), k, v)
                return kv._slot_sdpa_inputs(q, state.slot_k, state.slot_v, state.slot_w,
                                            state.M_s, state.slot_valid, scale, 1.)
            ka, va = pack_mid_kv(cache, plan, k, v, dim, backend=backend)
            qa = F.pad(q, (0, dim - d))
            qa[..., d] = 1. / scale
            return qa, ka, va

        def forward_backward(return_result=False):
            qa, ka, va = prepare()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION if dev.type == "cuda" else SDPBackend.MATH):
                y = F.scaled_dot_product_attention(qa, ka, va, scale=scale, enable_gqa=True,
                                                   attn_mask=causal_lower_right(t, ka.size(2)))[..., :d]
            grads = torch.autograd.grad(y, (q, k, v), upstream)
            if return_result:
                return y.detach(), *(x.detach() for x in grads)

        result = forward_backward(True)
        if name == "mid_reference":
            mid_reference = result
        elif backend:
            for a, b in zip(result, mid_reference):
                torch.testing.assert_close(a, b, atol=.025 if dev.type == "cuda" else 2e-5,
                                           rtol=.025 if dev.type == "cuda" else 2e-5)
        del result
        plan_ms, _ = measure(cache._semantic_attention_plan)
        prep_ms, prep_peak = measure(prepare)
        fb_ms, fb_peak = measure(forward_backward)
        valid, slots = int(plan[3].sum()), plan[3].numel()
        total = plan[0].size(-1) + cache.recent_count + t
        print(json.dumps({"variant": name, "pooled_width": plan[0].size(-1), "total_width": total,
                          "pooled_padding_fraction": 1 - valid / max(slots, 1),
                          "total_padding_fraction": (slots - valid) / max(g * total, 1),
                          "plan_ms": plan_ms, "prepare_ms": prep_ms, "forward_backward_ms": fb_ms,
                          "prepare_peak_MiB": prep_peak, "forward_backward_peak_MiB": fb_peak}), flush=True)


if __name__ == "__main__":
    main()
