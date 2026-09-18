"""Compare centroid, summary, and saved-update changes at the same token count.

python unused/benchmark_log_kv_summary_updates.py --iters 10
Each variant validates route/replay against its own forward (summaries change
approximation). Compilation, cache reset, and validation are outside timing.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litgpt.log_kv_cache import LogStructuredKVCache, _SemanticReplayPlans, _SemanticReplayUpdates, _triton_updates
from benchmark_log_kv_updates import assert_buffers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('sequence', 32768), ('chunk', 2048), ('batch', 1), ('groups', 8),
                          ('dim', 128), ('iters', 10), ('summary_size', 8)]:
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if min(args.chunk, args.batch, args.groups, args.dim, args.iters) < 1 or args.sequence < 2 * args.chunk:
        parser.error('require positive sizes and sequence >= 2*chunk')
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or _triton_updates() is None):
        parser.error('CUDA and Triton are required; use --device cpu for a functional smoke test')
    torch.set_num_threads(1)
    torch.manual_seed(54)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    b, g, n, d, t = args.batch, args.groups, args.sequence, args.dim, args.chunk
    prototype = torch.randn(b, g, 8, d, device=device, dtype=dtype)
    inputs = []
    for start in range(0, n, t):
        count = min(t, n - start)
        labels = torch.randint(8, (b, g, count), device=device)
        keys = prototype.gather(2, labels[..., None].expand(-1, -1, -1, d))
        keys += .1 * torch.randn_like(keys)
        host = [list(range(start, start + count)) for _ in range(b)]
        inputs.append((keys, torch.randn_like(keys), torch.tensor(host, device=device), host))

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    print(json.dumps({**vars(args), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device)}), flush=True)
    for label, centroid, summary, reuse in [('sequential', 'sequential', 1, False),
                                           ('parallel', 'parallel', 1, False),
                                           ('summary', 'parallel', args.summary_size, False),
                                           ('saved_summary', 'parallel', args.summary_size, True)]:
        base = LogStructuredKVCache(
            (b, g, n, d), (b, g, n, d), B=64, recent_size=t, device=device, dtype=dtype,
            semantic_clusters=True, cluster_k_max=8, semantic_anchor_mode='mid', allocate_second_order=False,
            semantic_centroid_backend=centroid, semantic_summary_size=summary, semantic_replay_updates=reuse,
            cos_cache=torch.ones(n, d, device=device), sin_cache=torch.zeros(n, d, device=device), rope_n_elem=d)
        with torch.no_grad():
            for k, v, pos, host in inputs[:-1]:
                base.route_and_flush_batch(k, v, pos, positions_host=host)
            k, v, pos, host = inputs[-1]
            expected = deepcopy(base)
            expected.begin_op_log()
            updates = _SemanticReplayUpdates() if reuse else None
            with expected._semantic_update_context(updates):
                expected.route_and_flush_batch(k, v, pos, positions_host=host, record_op_log=True)
            log, lengths = expected.take_op_log()
            plans = _SemanticReplayPlans(expected._last_op_log_host)
        payload_mib = sum(x.numel() * x.element_size() for x in updates.tensors) / 2**20 if reuse else 0.
        for phase in ('route', 'replay'):
            times, peaks = [], []
            for iteration in range(3 + args.iters):
                cache = deepcopy(base)
                current = _SemanticReplayUpdates() if reuse and phase == 'route' else updates
                if phase == 'route':
                    cache.begin_op_log()
                sync()
                baseline = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad(), cache._semantic_update_context(current, replay=phase == 'replay'):
                    start_time = time.perf_counter()
                    if phase == 'route':
                        cache.route_and_flush_batch(k, v, pos, positions_host=host, record_op_log=True)
                    else:
                        cache.route_and_flush_batch(k, v, pos, positions_host=host, replay_op_log=log,
                                                   replay_op_log_len=lengths, replay_op_log_host=plans.host_log,
                                                   replay_plans=plans)
                    sync()
                    elapsed = 1000 * (time.perf_counter() - start_time)
                peak = (torch.cuda.max_memory_allocated(device) - baseline) / 2**20 if device.type == 'cuda' else 0.
                assert_buffers(cache, expected, f'{label}/{phase}/{iteration}')
                if phase == 'route':
                    got_log, got_len = cache.take_op_log()
                    torch.testing.assert_close(got_log, log, atol=0, rtol=0)
                    torch.testing.assert_close(got_len, lengths, atol=0, rtol=0)
                if iteration >= 3:
                    times.append(elapsed)
                    peaks.append(peak)
                del cache, current
            print(json.dumps({'variant': label, 'phase': phase, 'median_ms': round(statistics.median(times), 3),
                              'peak_extra_MiB': round(max(peaks), 2), 'saved_update_MiB_per_flush': round(payload_mib, 3),
                              'entries_lane0': expected.total_slots, 'exact_replay_state': True,
                              'checked_iterations': 3 + args.iters}), flush=True)


if __name__ == '__main__':
    main()
