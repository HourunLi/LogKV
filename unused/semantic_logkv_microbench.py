"""Synthetic SemanticLogKV routing/op-log microbenchmark.

Usage:
    python unused/semantic_logkv_microbench.py
    python unused/semantic_logkv_microbench.py --device cuda --k-max 1,2,4 --orphan-ratios 0,0.1,0.5,1

This is intentionally not a model eval. It times only semantic cache routing,
op_log recording/cloning, and replay against synthetic pre-RoPE K/V.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("litgpt")
pkg.__path__ = [str(root / "litgpt")]
sys.modules.setdefault("litgpt", pkg)

from litgpt.log_kv_cache import LogStructuredKVCache


@dataclass
class Case:
    k_max: int
    orphan_ratio: float


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _time_ms(device: torch.device, fn) -> float:
    _sync(device)
    start = time.perf_counter()
    fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0


def _rope_cache(seq_len: int, dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    idx_theta = torch.outer(torch.arange(seq_len, device=device).float(), theta).repeat(1, 2)
    return torch.cos(idx_theta), torch.sin(idx_theta)


def _make_cache(args: argparse.Namespace, k_max: int, device: torch.device) -> LogStructuredKVCache:
    cos, sin = _rope_cache(args.seq_len, args.dim, device)
    cache = LogStructuredKVCache(
        (args.batch_size, args.groups, args.seq_len, args.dim),
        (args.batch_size, args.groups, args.seq_len, args.dim),
        B=args.B,
        recent_size=args.recent_size,
        device=device,
        dtype=torch.float32,
        semantic_clusters=True,
        cluster_k_max=k_max,
        cluster_lambda_rel=args.cluster_lambda_rel,
        seg_gap_max=args.seg_gap_max,
        seg_block_level=args.seg_block_level,
        semantic_flush_granularity=args.semantic_flush_granularity,
        semantic_s_h=1.0,
        cos_cache=cos,
        sin_cache=sin,
        rope_n_elem=args.dim,
    )
    cache.second_order = not args.first_order
    return cache


def _prototypes(k_max: int, dim: int, device: torch.device) -> torch.Tensor:
    proto = torch.zeros(k_max, dim, device=device)
    proto[:, 0] = torch.arange(k_max, device=device).float() * 100.0
    if dim > 1:
        proto[:, 1] = 1.0
    return proto


def _seed_full_clusters(cache: LogStructuredKVCache, proto: torch.Tensor) -> None:
    zero_v = torch.zeros(cache.v_dim, device=proto.device)
    for b in range(cache.batch_size):
        for g in range(cache.n_groups):
            for c in range(cache.K_max):
                cache._semantic_new_cluster(
                    b, g, c, -cache.K_max + c, proto[c], zero_v, torch.tensor(c, device=proto.device), record=False
                )


def _inputs(args: argparse.Namespace, k_max: int, orphan_ratio: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proto = _prototypes(k_max, args.dim, device)
    k = torch.empty(args.batch_size, args.groups, args.tokens, args.dim, device=device)
    v = torch.randn(args.batch_size, args.groups, args.tokens, args.dim, device=device)
    n_orphan = round(args.tokens * orphan_ratio)
    n_direct = args.tokens - n_orphan
    for i in range(n_direct):
        c = i % k_max
        k[:, :, i] = proto[c] + args.direct_noise * torch.randn(args.batch_size, args.groups, args.dim, device=device)
    for i in range(n_orphan):
        j = n_direct + i
        k[:, :, j].zero_()
        k[:, :, j, 0] = 10000.0 + i * 100.0
        if args.dim > 1:
            k[:, :, j, 1] = torch.arange(args.batch_size * args.groups, device=device).reshape(args.batch_size, args.groups)
    pos = torch.arange(args.tokens, device=device, dtype=torch.int64)
    return k, v, pos


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    return ys[len(ys) // 2]


def _bench_case(args: argparse.Namespace, case: Case, device: torch.device) -> dict[str, float]:
    k, v, pos = _inputs(args, case.k_max, case.orphan_ratio, device)
    route_ms: list[float] = []
    record_ms: list[float] = []
    clone_ms: list[float] = []
    replay_ms: list[float] = []
    ops = 0

    total = args.warmup + args.repeat
    for rep in range(total):
        cache = _make_cache(args, case.k_max, device)
        _seed_full_clusters(cache, _prototypes(case.k_max, args.dim, device))
        route = _time_ms(device, lambda: cache.route_and_flush_batch(k, v, pos, record_op_log=False))

        cache = _make_cache(args, case.k_max, device)
        _seed_full_clusters(cache, _prototypes(case.k_max, args.dim, device))
        cache.begin_op_log()
        record = _time_ms(device, lambda: cache.route_and_flush_batch(k, v, pos, record_op_log=True))
        clone = _time_ms(device, lambda: cache.take_op_log())
        op_log, op_log_len = cache.take_op_log()

        replay_cache = _make_cache(args, case.k_max, device)
        _seed_full_clusters(replay_cache, _prototypes(case.k_max, args.dim, device))
        replay = _time_ms(device, lambda: replay_cache.route_and_flush_batch(k, v, pos, replay_op_log=op_log, replay_op_log_len=op_log_len))

        if rep >= args.warmup:
            route_ms.append(route)
            record_ms.append(record)
            clone_ms.append(clone)
            replay_ms.append(replay)
            ops = int(op_log_len.sum().item())

    return {
        "route_ms": _median(route_ms),
        "record_ms": _median(record_ms),
        "oplog_clone_ms": _median(clone_ms),
        "replay_ms": _median(replay_ms),
        "ops": float(ops),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--recent-size", type=int, default=128)
    p.add_argument("--semantic-flush-granularity", type=int, default=2)
    p.add_argument("--k-max", default="1,2,4")
    p.add_argument("--orphan-ratios", default="0,0.1,0.5,1")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--cluster-lambda-rel", type=float, default=0.01)
    p.add_argument("--seg-gap-max", type=float, default=None)
    p.add_argument("--seg-block-level", type=int, default=0)
    p.add_argument("--direct-noise", type=float, default=0.001)
    p.add_argument("--first-order", action="store_true")
    args = p.parse_args()

    device = _device(args.device)
    if device.type != "cuda":
        print(f"# device={device}; use --device cuda on the target GPU for real timings")
    print("K_max\torphan_ratio\tops\troute_ms\trecord_ms\toplog_clone_ms\treplay_ms")
    for k_max in _parse_ints(args.k_max):
        for ratio in _parse_floats(args.orphan_ratios):
            stats = _bench_case(args, Case(k_max, ratio), device)
            print(
                f"{k_max}\t{ratio:g}\t{int(stats['ops'])}\t"
                f"{stats['route_ms']:.3f}\t{stats['record_ms']:.3f}\t"
                f"{stats['oplog_clone_ms']:.3f}\t{stats['replay_ms']:.3f}"
            )


if __name__ == "__main__":
    main()
