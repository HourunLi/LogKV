"""Same-state route/replay A/B, including host dispatch and device work.

python unused/benchmark_log_kv_updates.py --iters 10
CPU smoke: --device cpu --sequence 128 --chunk 16 --batch 1 --groups 2 --dim 8 --iters 2
Compilation and cache restoration are excluded; this is not full-step throughput.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import litgpt.log_kv_cache as kv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sequence", type=int, default=32768)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    if min(args.chunk, args.batch, args.groups, args.dim, args.iters) < 1 or args.sequence < 2 * args.chunk:
        p.error("require positive sizes/iters and sequence >= 2*chunk")
    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    fused = kv._triton_updates() if dev.type == "cuda" else None
    if dev.type == "cuda" and fused is None:
        raise RuntimeError("Triton unavailable; refusing to label a fallback as fused")
    torch.manual_seed(54)
    torch.set_num_threads(1)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    b, g, n, d, t = args.batch, args.groups, args.sequence, args.dim, args.chunk
    base = kv.LogStructuredKVCache(
        (b, g, n, d), (b, g, n, d), B=64, recent_size=t, device=dev, dtype=dtype,
        semantic_clusters=True, cluster_k_max=8, semantic_anchor_mode="mid", allocate_second_order=False,
        cos_cache=torch.ones(n, d, device=dev), sin_cache=torch.zeros(n, d, device=dev), rope_n_elem=d,
    )
    prototypes = torch.randn(b, g, 8, d, device=dev, dtype=dtype)

    def inputs(start, count):
        labels = torch.randint(8, (b, g, count), device=dev)
        k = prototypes.gather(2, labels[..., None].expand(-1, -1, -1, d))
        k = k + .1 * torch.randn_like(k)
        pos = torch.arange(start, start + count, device=dev).expand(b, -1)
        return k, torch.randn_like(k), pos, [list(range(start, start + count)) for _ in range(b)]

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    with torch.no_grad(), patch.object(kv, "_triton_updates", return_value=None):
        for start in range(0, n - t, t):
            k, v, pos, host = inputs(start, min(t, n - t - start))
            base.route_and_flush_batch(k, v, pos, positions_host=host)
        k, v, pos, host = inputs(n - t, t)
        expected = deepcopy(base)
        expected.begin_op_log()
        expected.route_and_flush_batch(k, v, pos, positions_host=host, record_op_log=True)
        log, lengths = expected.take_op_log()
        plans = kv._SemanticReplayPlans(expected._last_op_log_host)
    print(json.dumps({**vars(args), "torch": torch.__version__, "cuda": torch.version.cuda,
                      "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev)}), flush=True)
    variants = [("torch", None)] + ([("triton", fused)] if fused is not None else [])
    for name, backend in variants:
        for phase in ("route", "replay"):
            times, peaks = [], []
            for iteration in range(3 + args.iters):
                cache = deepcopy(base)
                if phase == "route":
                    cache.begin_op_log()
                sync()
                baseline = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                with torch.no_grad(), patch.object(kv, "_triton_updates", return_value=backend):
                    start = time.perf_counter()
                    if phase == "route":
                        cache.route_and_flush_batch(k, v, pos, positions_host=host, record_op_log=True)
                    else:
                        cache.route_and_flush_batch(k, v, pos, positions_host=host, replay_op_log=log,
                            replay_op_log_len=lengths, replay_op_log_host=plans.host_log, replay_plans=plans)
                    sync()
                    elapsed = (time.perf_counter() - start) * 1000
                peak = torch.cuda.max_memory_allocated(dev) - baseline if dev.type == "cuda" else 0
                if iteration == 0:
                    for field, tensor in expected.named_buffers():
                        if not field.startswith("op_log"):
                            torch.testing.assert_close(dict(cache.named_buffers())[field], tensor, atol=0, rtol=0, msg=field)
                    if phase == "route":
                        got_log, got_len = cache.take_op_log()
                        torch.testing.assert_close(got_log, log, atol=0, rtol=0)
                        torch.testing.assert_close(got_len, lengths, atol=0, rtol=0)
                if iteration >= 3:
                    times.append(elapsed)
                    peaks.append(peak / 2**20)
                del cache
            print(json.dumps({"variant": name, "phase": phase, "median_ms": round(statistics.median(times), 3),
                              "peak_extra_MiB": round(max(peaks), 2), "exact_state": True}), flush=True)


if __name__ == "__main__":
    main()
