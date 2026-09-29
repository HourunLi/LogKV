"""Bounded synthetic unified-route benchmark; no model or training job needed.

python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8
CPU smoke: --device cpu --tokens 64 --dim 8 --groups 2 --iters 1
Uninstrumented route timings include device completion. A separate diagnostic
pass synchronizes stage boundaries and records merge rounds; do not add its
numbers to the uninstrumented timing. Synthetic keys do not predict NIAH quality.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litgpt.log_kv_cache import LogStructuredKVCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--pattern', choices=('random', 'clustered'), default='random')
    for name, default in [('tokens', 2048), ('dim', 128), ('batch', 1), ('groups', 1),
                          ('clusters', 12), ('B', 256), ('flushes', 2), ('iters', 3)]:
        p.add_argument('--' + name, type=int, default=default)
    args = p.parse_args()
    if min(args.tokens, args.dim, args.batch, args.groups, args.clusters, args.B, args.flushes, args.iters) < 1:
        p.error('all sizes and iteration counts must be positive')
    if args.B < 2 or args.dim % 2:
        p.error('B must be >= 2 and dim must be even')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; use --device cpu for a functional check')
    torch.set_num_threads(1)
    torch.manual_seed(17)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    n = max(32768, args.flushes * args.tokens)
    shape = (args.batch, args.groups, n, args.dim)
    cache = LogStructuredKVCache(
        shape, shape, B=args.B, recent_size=args.tokens, semantic_flush_granularity=args.tokens,
        semantic_clusters=True, cluster_k_max=args.clusters, semantic_unified_route=True,
        semantic_anchor_mode='mid', semantic_centroid_backend='parallel', allocate_second_order=False,
        device=device, dtype=dtype, cos_cache=torch.ones(n, args.dim, device=device),
        sin_cache=torch.zeros(n, args.dim, device=device), rope_n_elem=args.dim,
    )
    prototypes = torch.randn(args.batch, args.groups, args.clusters, args.dim, device=device, dtype=dtype)
    inputs = []
    for step in range(args.flushes):
        k = torch.randn(args.batch, args.groups, args.tokens, args.dim, device=device, dtype=dtype)
        if args.pattern == 'clustered':
            labels = torch.randint(args.clusters, k.shape[:-1], device=device)
            k = .03 * k + prototypes.gather(2, labels[..., None].expand_as(k))
        pos = torch.arange(step * args.tokens, (step + 1) * args.tokens, device=device)
        host = [list(range(step * args.tokens, (step + 1) * args.tokens))] * args.batch
        inputs.append((k, torch.randn_like(k), pos, host))

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def run():
        for step, (k, v, pos, host) in enumerate(inputs):
            cache.route_and_flush_batch(k, v, pos, positions_host=host)
        return args.flushes * args.tokens

    def validate(mass):
        torch.testing.assert_close(cache.level_w.sum((2, 3, 4)),
                                   torch.full((args.batch, args.groups), float(mass), device=device), rtol=0, atol=0)
        assert (cache.alive.sum(-1) <= args.clusters).all()

    times, peaks = [], []
    with torch.no_grad():
        for iteration in range(args.iters + 1):
            cache.reset_parameters()
            sync()
            baseline = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            mass = run()
            sync()
            elapsed = time.perf_counter() - start
            peak = (torch.cuda.max_memory_allocated(device) - baseline) / 2**20 if device.type == 'cuda' else None
            validate(mass)
            if iteration:
                times.append(elapsed)
                peaks.append(peak)
        print(json.dumps({**vars(args), 'torch': torch.__version__,
                          'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
                          'route_median_s_per_flush': statistics.median(times) / args.flushes,
                          'peak_extra_MiB': max(peaks) if device.type == 'cuda' else None,
                          'mass_verified': True}), flush=True)

        stages = defaultdict(float)
        rounds, phase = [], ['']
        reduce = cache._semantic_unified_reduce_batch
        pairs = cache._semantic_unified_pairs

        def reduce_timed(*a, **kw):
            phase[0] = 'candidates' if kw.get('radius_limits') is not None else 'global_merge'
            return timed(phase[0], reduce)(*a, **kw)

        def pairs_counted(cost, count):
            result = pairs(cost, count)
            sizes = (result[..., 0] >= 0).sum(-1).cpu().tolist()
            rounds.append({'phase': phase[0], 'matrix_size': cost.size(-1), 'proposed_pairs_per_group': sizes})
            return result

        def timed(name, fn):
            def wrapped(*a, **kw):
                sync()
                start = time.perf_counter()
                result = fn(*a, **kw)
                sync()
                stages[name] += time.perf_counter() - start
                return result
            return wrapped

        cache.reset_parameters()
        with ExitStack() as stack:
            stack.enter_context(patch.object(cache, '_semantic_unified_reduce_batch', reduce_timed))
            stack.enter_context(patch.object(cache, '_semantic_unified_pairs', pairs_counted))
            for name, stage in [('_semantic_ward_merge', 'old_kv_merge'),
                                ('_semantic_new_cluster', 'new_cluster_write'),
                                ('_semantic_commit_joins', 'batched_write')]:
                stack.enter_context(patch.object(cache, name, timed(stage, getattr(cache, name))))
            validate(run())
        print(json.dumps({'diagnostic_stage_s': dict(stages), 'pairing_searches': rounds,
                          'note': 'Separate synchronized diagnostic pass; includes instrumentation overhead.'}), flush=True)


if __name__ == '__main__':
    main()
