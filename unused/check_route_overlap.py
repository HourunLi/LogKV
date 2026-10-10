"""GPU check for LOGKV_ROUTE_OVERLAP: overlapped route decisions must change nothing.

Runs the default route (attach + Alpha + Beta + Gamma) through the real training path
(LogKVStreamTrainingAttention forward, then backward replay) and through
prefill-style add_recent, once with decisions on the main stream and once on
the side stream. Outputs, gradients, every cache buffer and the host mirrors
must match bit for bit; the JSON line also reports both wall times.
Backward uses strict deterministic algorithms: FlashAttention's default dQ
reduction is not bitwise reproducible, even with overlap disabled. Forward
timings retain the normal execution mode; this setting is only for the check.

    python unused/check_route_overlap.py --blocks 16 --groups 8
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

# Set before CUDA creates a cuBLAS handle; replay may use cuBLAS under the
# strict backward check. Respect an explicit workspace configuration.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litgpt.log_kv_cache import LogStructuredKVCache, LogKVStreamTrainingAttention

HOST_STATE = ("_semantic_counts", "_semantic_alive", "_semantic_n_total", "_semantic_p_hi_c",
              "_semantic_level0_phase", "_alpha_spans", "_alpha_positions", "alpha_count")


def make_cache(args, overlap, device, batch):
    n = args.blocks * args.tokens + args.tokens
    cache = LogStructuredKVCache(
        (batch, args.groups, n, args.dim), (batch, args.groups, n, args.dim), B=args.B,
        recent_size=args.tokens, semantic_flush_granularity=args.tokens, semantic_clusters=True,
        cluster_k_max=args.clusters, semantic_anchor_mode="mid", semantic_centroid_backend="parallel",
        allocate_second_order=False, semantic_replay_updates=True, alpha_exact_tokens=args.exact,
        alpha_span_max_tokens=args.span, beta_novelty=True, beta_adaptive_merge=True,
        gamma_level0_reinsert=True, gamma_top_merge="lightest",
        device=device, dtype=torch.bfloat16, cos_cache=torch.ones(n, args.dim, device=device),
        sin_cache=torch.zeros(n, args.dim, device=device), rope_n_elem=args.dim,
    )
    cache._route_overlap = overlap
    return cache


def compare(a, b, label):
    for name, value in a.named_buffers():
        assert_equal(value, b.get_buffer(name), f"{label}: buffer {name}")
    for name in HOST_STATE:
        assert getattr(a, name) == getattr(b, name), f"{label}: host state {name} differs"


def assert_equal(a, b, label):
    # Zero tolerance keeps real stream/replay errors visible, with the mismatch
    # count and greatest absolute/relative differences in the failure report.
    torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda message: f"{label}: {message}")


def backward_exact(loss):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        # warn_only=True would still run FlashAttention's nondeterministic path.
        torch.use_deterministic_algorithms(True, warn_only=False)
        loss.backward()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in [("blocks", 8), ("tokens", 2048), ("dim", 128), ("groups", 8), ("heads", 16),
                          ("clusters", 12), ("B", 128), ("exact", 256), ("span", 64)]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--device", default="cuda", help="cpu only dry-runs the harness: no streams, nothing overlaps")
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required: the overlap only exists on CUDA streams")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    torch.manual_seed(7)
    total = args.blocks * args.tokens
    protos = torch.randn(args.groups, 24, args.dim, device=device) * 4
    labels = torch.randint(24, (args.groups, total), device=device)
    k = (protos.gather(1, labels[..., None].expand(-1, -1, args.dim))
         + torch.randn(args.groups, total, args.dim, device=device))[None].to(torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(1, args.heads, total, args.dim, device=device, dtype=torch.bfloat16)
    ends = [[(i % 11) == 10 for i in range(total)]]
    report = {}

    # Training path: streamed attention reads the cache between flushes.
    results = []
    for overlap in (False, True):
        cache = make_cache(args, overlap, device, 1)
        qi, ki, vi = (x.clone().requires_grad_() for x in (q, k, v))
        sync()
        start = time.perf_counter()
        y = LogKVStreamTrainingAttention.apply(qi, ki, vi, cache, args.dim ** -.5, args.tokens, 0., ki, ends)
        sync()
        forward = time.perf_counter() - start
        backward_exact(y.float().square().mean())
        sync()
        report[f"train_forward_s_overlap_{overlap}"] = forward
        results.append((cache, y.detach(), qi.grad, ki.grad, vi.grad))
    (base, *tensors), (fast, *fast_tensors) = results
    compare(base, fast, "training")
    for name, a, b in zip(("y", "dq", "dk", "dv"), tensors, fast_tensors):
        assert_equal(a, b, f"training: {name} differs with overlap")

    # Prefill path: add_recent with a busy main stream between blocks.
    caches = []
    for overlap in (False, True):
        cache = make_cache(args, overlap, device, 1)
        busy = torch.randn(4096 if device.type == 'cuda' else 64, 4096 if device.type == 'cuda' else 64,
                           device=device, dtype=torch.bfloat16)
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            for block in range(args.blocks):
                rows = slice(block * args.tokens, (block + 1) * args.tokens)
                cache._semantic_attention_plan()
                busy = busy @ busy / 64  # Stand-in for the attention queued before each flush.
                cache.add_recent(k[:, :, rows], v[:, :, rows], k_raw=k[:, :, rows],
                                 span_ends=[row[rows] for row in ends])
        sync()
        report[f"prefill_s_overlap_{overlap}"] = time.perf_counter() - start
        caches.append(cache)
    compare(*caches, "prefill")
    print(json.dumps({**vars(args), **report, "bitwise_equal": True, "deterministic_backward": True,
                      "torch": torch.__version__,
                      "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}),
          flush=True)


if __name__ == "__main__":
    main()
