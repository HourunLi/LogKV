"""Bounded synthetic unified-route benchmark; no model or training job needed.

python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8
CPU smoke: --device cpu --tokens 64 --dim 8 --groups 2 --iters 1
Uninstrumented route timings include device completion. A separate diagnostic
pass synchronizes stage boundaries and records merge rounds; do not add its
numbers to the uninstrumented timing. Synthetic keys do not predict NIAH quality.
"""
import argparse
import cProfile
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litgpt.log_kv_cache import LogStructuredKVCache, _UnifiedReduceTorch, _triton_route, _triton_updates


def cache_mass(cache):
    mass = cache.level_w.sum((2, 3, 4))
    if cache.alpha_exact_tokens:
        mass = mass + cache.alpha_valid.sum(-1)[:, None]
    return mass


def flush_route(cache, k, v, pos, host):
    # Synthetic punctuation only: this benchmarks mechanics, not selection quality.
    ends = [[(p + b * 7) % max(1, cache.alpha_span_max_tokens // 2) == 0 for p in row]
            for b, row in enumerate(host)] if cache.alpha_exact_tokens else None
    cache.route_and_flush_batch(k, v, pos, positions_host=host, span_ends=ends)


def check_round_pairs(cache, device):
    """Cheap fused-vs-Torch check before timing; no checkpoint or training needed."""
    generator = torch.Generator(device=device).manual_seed(91)
    for size in (5, 33, 65):
        mu = torch.randint(-4, 5, (4, size, 8), device=device, generator=generator).float()
        mu[0].zero_()  # Exact ties, including a non-power-of-two row length.
        mass = torch.randint(1, 4, (4, size), device=device, generator=generator).float()
        mass[0].fill_(1)
        mass[2, size // 2:] = 0  # Ragged padding.
        mass[3].zero_()         # No live pairs.
        radius = torch.zeros_like(mass)
        limits = mu.new_tensor([0., 8., 2., 1.])
        for bound, cap, overflow in ((None, float('inf'), True), (None, 4., True),
                                     (None, .1, True), (limits, 4., False)):
            with patch('litgpt.log_kv_cache._triton_route', return_value=None):
                expected = cache._semantic_unified_round_pairs(mu, mass, radius, bound, cap, overflow)
            actual = cache._semantic_unified_round_pairs(mu, mass, radius, bound, cap, overflow)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def check_merge_pack(cache, device):
    """Centroid rounding affects later pairs: verify updates as well as selection."""
    generator = torch.Generator(device=device).manual_seed(93)
    for dim in (7, 128):
        mu = torch.randn(3, 11, dim, device=device, generator=generator)
        mass = torch.randint(1, 500000, (3, 11), device=device, generator=generator).float()
        radius = torch.zeros_like(mass)
        src = torch.arange(0, 30, 2, device=device).view(3, 5)
        partner = src + 1
        src[0, 1] = -1
        src[2] = -1
        partner[1, 2] = -1
        indices = torch.stack((src, partner))
        original = mu.clone(), mass.clone()
        with patch('litgpt.log_kv_cache._triton_route', return_value=None):
            expected = cache._semantic_unified_merge_pack(mu, mass, radius, indices, False)
        actual = cache._semantic_unified_merge_pack(mu, mass, radius, indices, False)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(mu, original[0], rtol=0, atol=0)
        torch.testing.assert_close(mass, original[1], rtol=0, atol=0)


def check_pair_distances(device):
    """Fused distance epilogue against the unfused expression on the same GEMM."""
    fused = _triton_route()
    generator = torch.Generator(device=device).manual_seed(96)
    for lanes, size, dim in ((3, 33, 7), (2, 130, 64), (1, 257, 128)):
        mu = torch.randn(lanes, size, dim, device=device, generator=generator)
        centered = mu - mu[:, :1]
        norm = centered.square().sum(-1)
        gram = torch.bmm(centered, centered.transpose(1, 2))
        dist = ((norm[:, :, None] + norm[:, None, :]) - 2 * gram).clamp_min(0)
        expected = dist.triu() + dist.triu(1).transpose(1, 2)
        actual = fused.pair_distances(gram.clone(), norm)
        assert torch.equal(actual, expected), 'fused pair distances differ from the unfused expression'


def check_incremental_reduce(device):
    """Fused incremental rounds against the Torch reference on the same device.

    The global pass (one sweep or frozen sweeps) is elementwise-identical and
    must match bit for bit. The candidate pass sums exact squared differences
    in a different order, so it compares partitions; random inputs make exact
    bound ties negligible. The fused reducer runs twice: as configured (CUDA
    graph replay, in-kernel sort, live-bound recapture) and with eager launches
    plus argsort.
    """
    fused = _triton_route()
    generator = torch.Generator(device=device).manual_seed(95)
    for size, dim, pattern in ((5, 8, 'ints'), (33, 7, 'ties'), (65, 128, 'random'), (300, 16, 'clusters'),
                               (160, 3, 'lattice')):
        count = torch.tensor([size, size - 3, size // 2, 2], device=device)
        mu = torch.randn(4, size, dim, device=device, generator=generator)
        if pattern == 'ints':
            mu = mu.mul(2).round()
        elif pattern == 'ties':
            mu[0] = .5  # Exact ties exercise the XOR rule and packed keys.
        elif pattern == 'clusters':
            mu = torch.randint(0, 5, (4, size, 1), device=device, generator=generator).float() * 4 + .05 * mu
        elif pattern == 'lattice':
            # Ulp-level near ties between clusters merged in the same round.
            mu = torch.randint(0, 3, mu.shape, device=device, generator=generator).float() * 1000 + 1e-4 * mu
        mask = torch.arange(size, device=device)[None, :] < count[:, None]
        mu = mu * mask[..., None]
        mass = torch.randint(1, 4, (4, size), device=device, generator=generator).float() * mask
        # passes > 1 covers the frozen-center sweeps (merge_passes=4).
        for limits, target, passes in ((None, 3, 1), (None, 3, 4), (mu.new_tensor([1., 2., 30., .1]), 1, 1)):
            results = []
            for reducer_type, eager in ((_UnifiedReduceTorch, False), (fused.UnifiedReduce, False),
                                        (fused.UnifiedReduce, True)):
                state = mu.clone()
                dist = LogStructuredKVCache._semantic_unified_pair_matrix(state)
                reducer = reducer_type(dist, state, mass.clone(), count, limits, target, passes)
                if eager:
                    reducer.sort, reducer.graph = False, False
                for round_ in range(1, 2 * size + 8):
                    flag = reducer.step(round_)
                    if not eager and hasattr(reducer, 'limit_live') and round_ % 2 == 0:
                        reducer.limit_live(int(reducer.count.max()))  # Exercises graph recapture.
                    if not bool(flag.any()):
                        break
                # Symmetry guarantees a mutual pair whenever a finite cost exists.
                assert torch.equal(dist, dist.transpose(1, 2)), 'incremental distance matrix lost symmetry'
                results.append((reducer.finish(), state))
            (expected, expected_mu), *fused_results = results
            for actual, actual_mu in fused_results:
                if limits is None:
                    for a, b in zip(actual[0], expected[0]):
                        assert (a == b).all(), 'fused global merge trace differs from Torch'
                    assert (actual[1] == expected[1]).all() and (actual[2] == expected[2]).all()
                    torch.testing.assert_close(actual_mu, expected_mu, rtol=0, atol=0)
                else:
                    for lane, (a, b) in enumerate(zip(actual[0], expected[0])):
                        assert sorted(map(tuple, a.tolist())) == sorted(map(tuple, b.tolist())), \
                            f'fused candidate partition differs from Torch (lane {lane})'
                    assert (actual[1] == expected[1]).all()
            # Graph replay and in-kernel sort change launches only: bit-identical.
            (replayed, replayed_mu), (launched, launched_mu) = fused_results
            assert all((a == b).all() for a, b in zip(replayed[0], launched[0])), 'graph/sort trace differs from eager'
            assert torch.equal(replayed_mu, launched_mu)


def check_batched_state(device):
    """Exercise old merges/new slots against serial physical writes before timing."""
    shape = (1, 2, 256, 8)
    actual = LogStructuredKVCache(
        shape, shape, B=3, recent_size=16, semantic_clusters=True, cluster_k_max=3,
        semantic_unified_route=True, allocate_second_order=False, device=device,
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        cos_cache=torch.ones(256, 8, device=device), sin_cache=torch.zeros(256, 8, device=device), rope_n_elem=8,
    )
    for g in range(2):
        for c, value in enumerate((0., .01, 100.)):
            k = torch.full((8,), value, device=device, dtype=actual.level_k.dtype)
            actual._semantic_new_cluster(0, g, c, c, k, k, torch.tensor(c, device=device), record=False)
    reference = deepcopy(actual)
    for cache in (actual, reference):
        cache.begin_op_log()

    def serial_merge(jobs, *, record):
        for b, g, keep, free in jobs:
            reference._semantic_ward_merge(b, g, keep, free, record=record)

    def serial_new(jobs, k, v, pos, host, *, record):
        for b, g, c, i in jobs:
            reference._semantic_new_cluster(b, g, c, host[b][i], k[b, g, i], v[b, g, i], pos[b, i], record=record)

    with patch.object(reference, '_semantic_ward_merge_batch', serial_merge), \
         patch.object(reference, '_semantic_new_clusters', serial_new):
        for step in range(2):
            k = torch.tensor([1000. + step * 2000] * 8 + [2000. + step * 2000] * 8,
                             device=device, dtype=actual.level_k.dtype).view(1, 1, 16, 1).expand(1, 2, 16, 8)
            pos = torch.arange(3 + step * 16, 19 + step * 16, device=device)
            for cache in (actual, reference):
                cache.route_and_flush_batch(k, k, pos, record_op_log=True)
            for name, value in actual.named_buffers():
                torch.testing.assert_close(value, reference.get_buffer(name), rtol=0, atol=0, msg=name)
            for name in actual._UPDATE_HOST_FIELDS + ('_semantic_counts', '_op_log_host'):
                assert getattr(actual, name) == getattr(reference, name), name


def profile_route(cache, inputs, device, directory):
    """Profile only the last flush, with its preceding cache state built outside capture."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def prepare():
        cache.reset_parameters()
        for k, v, pos, host in inputs[:-1]:
            flush_route(cache, k, v, pos, host)
        sync()

    def flush():
        k, v, pos, host = inputs[-1]
        flush_route(cache, k, v, pos, host)

    def annotated(fn, label):
        def wrapped(*a, **kw):
            name = label
            if label == 'reduce':
                name = 'candidates' if kw.get('radius_limits') is not None else 'global_merge'
            with torch.profiler.record_function('logkv/' + name):
                return fn(*a, **kw)
        return wrapped

    prepare()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # No per-stage synchronization or pair-count readbacks in this capture.
    with ExitStack() as stack:
        if cache.alpha_exact_tokens:
            from litgpt import alpha_log_kv

            stack.enter_context(patch.object(alpha_log_kv, 'select_spans', annotated(alpha_log_kv.select_spans, 'exact_select')))
            updates = _triton_updates() if device.type == 'cuda' else None
            if updates is not None:
                stack.enter_context(patch.object(updates, 'alpha_partition', annotated(updates.alpha_partition, 'exact_partition')))
                stack.enter_context(patch.object(updates, 'merge_scatter', annotated(updates.merge_scatter, 'ladder_merge_scatter')))
        for method, label in (
            ('_semantic_unified_reduce_device', 'reduce'),
            ('_semantic_unified_round', 'merge_round'),
            ('_semantic_unified_pair_matrix', 'pair_matrix'),
            # Full-recompute stages: finite capacity or the stalled-lane fallback.
            ('_semantic_unified_round_pairs', 'pair_search'),
            ('_semantic_unified_merge_pack', 'merge_pack'),
            ('_semantic_ward_merge_batch', 'old_kv_merge'),
            ('_semantic_new_clusters', 'new_cluster_write'),
            ('_semantic_commit_joins', 'batched_write'),
            ('_alpha_route_flush', 'alpha_select_and_archive'),
            ('_alpha_commit_joins', 'alpha_archive'),
        ):
            stack.enter_context(patch.object(cache, method, annotated(getattr(cache, method), label)))
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            with torch.profiler.record_function('logkv/route_flush'):
                flush()
            sync()
    prof.export_chrome_trace(str(out / 'route_trace.json'))
    averages = prof.key_averages(group_by_input_shape=True)
    cuda_events = sum(e.device_type == torch.autograd.DeviceType.CUDA for e in prof.events())
    report = io.StringIO()
    report.write('One warmed final flush; preceding flushes and cache reset excluded.\n'
                 'Profiler overhead is included. Use the uninstrumented JSON timing for speed comparisons.\n'
                 'CPU operator time can include CUDA waits; it is not pure CPU computation.\n'
                 'Nested total times overlap; do not add them together.\n\n')
    report.write('CPU operators, sorted by self CPU time:\n')
    report.write(averages.table(sort_by='self_cpu_time_total', row_limit=30))
    if device.type == 'cuda':
        report.write('\n\nCUDA operators, sorted by self device time:\n')
        report.write(averages.table(sort_by='self_device_time_total', row_limit=30))
        report.write(f'\nCUDA device events captured: {cuda_events}\n')
        if not cuda_events:
            report.write('WARNING: no CUDA device events captured; GPU kernel attribution is unavailable.\n')

    # Separate pass: cProfile reveals Python overhead without torch-profiler hooks.
    prepare()
    python_profile = cProfile.Profile()
    python_profile.runcall(flush)
    sync()
    python_profile.dump_stats(str(out / 'route_python.prof'))
    report.write('\n\nPython call sites, sorted by self time (includes blocking native calls):\n')
    pstats.Stats(python_profile, stream=report).strip_dirs().sort_stats('tottime').print_stats(30)
    report.write('\nPython call sites, sorted by cumulative time:\n')
    pstats.Stats(python_profile, stream=report).strip_dirs().sort_stats('cumtime').print_stats(30)
    expected = len(inputs) * inputs[0][0].size(2)
    torch.testing.assert_close(cache_mass(cache),
                               torch.full(inputs[0][0].shape[:2], float(expected), device=device), rtol=0, atol=0)
    (out / 'route_summary.txt').write_text(report.getvalue())
    return {'profile_summary': str(out / 'route_summary.txt'),
            'profile_trace': str(out / 'route_trace.json'),
            'python_profile': str(out / 'route_python.prof'),
            'cuda_device_events': cuda_events, 'profiled_flushes_per_pass': 1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--pattern', choices=('random', 'clustered'), default='random')
    p.add_argument('--profile-dir', help='Write CPU/CUDA operator tables, trace and Python profile; replaces stage diagnostics')
    for name, default in [('tokens', 2048), ('dim', 128), ('batch', 1), ('groups', 1),
                          ('clusters', 12), ('B', 256), ('flushes', 2), ('iters', 3)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--route', choices=('unified', 'attach'), default='unified',
                   help='Archive route; attach is the default training route (stage diagnostics cover unified only)')
    p.add_argument('--merge-passes', type=int, default=1, help='Frozen-center global matching sweeps (1 = original)')
    p.add_argument('--alpha-exact-tokens', type=int, default=0)
    p.add_argument('--alpha-span-max-tokens', type=int, default=64)
    p.add_argument('--beta-novelty', action='store_true')
    p.add_argument('--beta-adaptive-merge', action='store_true')
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
        semantic_clusters=True, cluster_k_max=args.clusters, semantic_unified_route=args.route == 'unified',
        semantic_anchor_mode='mid', semantic_centroid_backend='parallel', allocate_second_order=False,
        semantic_merge_passes=args.merge_passes,
        alpha_exact_tokens=args.alpha_exact_tokens, alpha_span_max_tokens=args.alpha_span_max_tokens,
        beta_novelty=args.beta_novelty, beta_adaptive_merge=args.beta_adaptive_merge,
        device=device, dtype=dtype, cos_cache=torch.ones(n, args.dim, device=device),
        sin_cache=torch.zeros(n, args.dim, device=device), rope_n_elem=args.dim,
    )
    fused = device.type == 'cuda' and _triton_route() is not None
    if fused:
        check_round_pairs(cache, device)
        check_merge_pack(cache, device)
        check_pair_distances(device)
        check_incremental_reduce(device)
    check_batched_state(device)
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
            flush_route(cache, k, v, pos, host)
        return args.flushes * args.tokens

    def validate(mass):
        torch.testing.assert_close(cache_mass(cache),
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
                          'route_backend': 'triton' if fused else 'torch',
                          'pairing_rule': 'frozen_mnn' if args.merge_passes > 1 else 'mnn',
                          'route_group_tile': cache._semantic_unified_tile(args.batch * args.groups, args.tokens, device),
                          'effective_B': cache.B, 'exact_count': cache.alpha_count,
                          'reference_checks_scope': 'shared_unified_route_and_pairwise_updates',
                          'reference_pairs_verified': True if fused else None,
                          'reference_updates_verified': True if fused else None,
                          'reference_incremental_verified': True if fused else None,
                          'reference_state_verified': True,
                          'mass_verified': True}), flush=True)

        if args.profile_dir:
            print(json.dumps(profile_route(cache, inputs, device, args.profile_dir)), flush=True)
            return

        stages = defaultdict(float)
        rounds, phase = [], ['']
        reduce = cache._semantic_unified_reduce_device
        merge_round = cache._semantic_unified_round
        pairs = cache._semantic_unified_round_pairs

        def reduce_timed(*a, **kw):
            phase[0] = 'candidates' if kw.get('radius_limits') is not None else 'global_merge'
            return timed(phase[0], reduce)(*a, **kw)

        def round_counted(reducer, round_):
            before = reducer.count.cpu().clone()
            result = merge_round(reducer, round_)
            merged = (before - reducer.count.cpu()).tolist()
            rounds.append({'phase': phase[0], 'matrix_size': reducer.mass.size(1), 'round': round_,
                           'merged_pairs_per_group': merged})
            return result

        def pairs_counted(mu, *a, **kw):
            result = pairs(mu, *a, **kw)
            sizes = (result[..., 0] >= 0).sum(-1).cpu().tolist()
            rounds.append({'phase': phase[0], 'matrix_size': mu.size(1),
                           'proposed_pairs_per_group': sizes})
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
            stack.enter_context(patch.object(cache, '_semantic_unified_reduce_device', reduce_timed))
            stack.enter_context(patch.object(cache, '_semantic_unified_round', round_counted))
            stack.enter_context(patch.object(cache, '_semantic_unified_round_pairs', pairs_counted))
            for name, stage in [('_semantic_ward_merge_batch', 'old_kv_merge'),
                                ('_semantic_new_clusters', 'new_cluster_write'),
                                ('_semantic_commit_joins', 'batched_write')]:
                stack.enter_context(patch.object(cache, name, timed(stage, getattr(cache, name))))
            validate(run())
        print(json.dumps({'diagnostic_stage_s': dict(stages), 'merge_rounds': rounds,
                          'note': 'Separate synchronized diagnostic pass; includes instrumentation overhead.'}), flush=True)


if __name__ == '__main__':
    main()
