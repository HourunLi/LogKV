"""Same-state route/replay A/B, including host dispatch and device work.

python unused/benchmark_log_kv_updates.py --iters 10
First-mismatch diagnosis: python unused/benchmark_log_kv_updates.py --diagnose
CPU smoke: --device cpu --sequence 128 --chunk 16 --batch 1 --groups 2 --dim 8 --iters 2
Compilation and cache restoration are excluded; this is not full-step throughput.
"""

import argparse
from contextlib import contextmanager
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



def assert_buffers(actual, expected, context):
    buffers = dict(actual.named_buffers())
    for field, tensor in expected.named_buffers():
        if not field.startswith("op_log"):
            torch.testing.assert_close(buffers[field], tensor, atol=0, rtol=0,
                                       msg=lambda detail: f"{context}: {field}\n{detail}")


@contextmanager
def diagnose_updates(backend):
    """Untimed, immediate comparisons isolate the first faulty shared update."""
    append = kv.LogStructuredKVCache._semantic_append_entries_batched
    centroid = backend.centroid
    calls = {"ladder": 0, "centroid": 0}

    def checked_append(cache, lanes, counts, block):
        reference = deepcopy(cache)
        with patch.object(kv, "_triton_updates", return_value=None):
            append(reference, lanes, counts, block)
        append(cache, lanes, counts, block)
        calls["ladder"] += 1
        assert_buffers(cache, reference, f"ladder update {calls['ladder']}, counts={counts}")

    def checked_centroid(k, mu, ne, meta, start, count):
        # Reconstruct the original run order used by the PyTorch implementation.
        lengths = meta[3].index_select(0, torch.argsort(meta[1]))
        sums = torch.segment_reduce(k.float(), "sum", lengths=lengths, unsafe=True)
        end = start + count
        ci, sel = meta[0, start:end], meta[1, start:end]
        assert ci.unique().numel() == count, "centroid ordinal contains duplicate cluster writers"
        n = meta[3, start:end].float()
        forget = meta[4, start:end].to(torch.int32).view(torch.float32)
        pre = ne.index_select(0, ci) * forget
        denom = pre + n
        sums = sums.index_select(0, sel)
        old_mu = mu.index_select(0, ci)
        numerator = pre[:, None] * old_mu + sums
        expected = torch.where((pre > 0)[:, None], numerator / denom[:, None], sums / n[:, None])
        centroid(k, mu, ne, meta, start, count)
        calls["centroid"] += 1
        label = f"centroid update {calls['centroid']}, ordinal offset={start}, runs={count}"
        actual = mu.index_select(0, ci)
        try:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0,
                                       msg=lambda detail: f"{label}: centroid\n{detail}")
        except AssertionError as exc:
            row, dim = divmod(int((actual - expected).abs().argmax().item()), k.size(1))
            values = {"cluster": int(ci[row]), "dim": dim, "old_mu": float(old_mu[row, dim]),
                      "sum": float(sums[row, dim]), "pre": float(pre[row]), "n": float(n[row]),
                      "numerator": float(numerator[row, dim]), "denom": float(denom[row]),
                      "actual": float(actual[row, dim]), "expected": float(expected[row, dim])}
            raise AssertionError(f"{exc}\nLargest-difference inputs: {json.dumps(values)}") from exc
        torch.testing.assert_close(ne.index_select(0, ci), denom, atol=0, rtol=0,
                                   msg=lambda detail: f"{label}: n_eff\n{detail}")

    with patch.object(kv.LogStructuredKVCache, "_semantic_append_entries_batched", checked_append), \
         patch.object(backend, "centroid", checked_centroid):
        yield calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sequence", type=int, default=32768)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--diagnose", action="store_true", help="check each ladder/centroid update; do not time")
    args = p.parse_args()
    if min(args.chunk, args.batch, args.groups, args.dim, args.iters) < 1 or args.sequence < 2 * args.chunk:
        p.error("require positive sizes/iters and sequence >= 2*chunk")
    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    fused = kv._triton_updates() if dev.type == "cuda" else None
    if dev.type == "cuda" and fused is None:
        raise RuntimeError("Triton unavailable; refusing to label a fallback as fused")
    if args.diagnose and fused is None:
        p.error("--diagnose requires CUDA/Triton")
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
    if args.diagnose:
        with torch.no_grad(), diagnose_updates(fused) as calls:
            for phase in ("route", "replay"):
                cache = deepcopy(base)
                if phase == "route":
                    cache.begin_op_log()
                    cache.route_and_flush_batch(k, v, pos, positions_host=host, record_op_log=True)
                else:
                    cache.route_and_flush_batch(k, v, pos, positions_host=host, replay_op_log=log,
                        replay_op_log_len=lengths, replay_op_log_host=plans.host_log, replay_plans=plans)
                assert_buffers(cache, expected, f"diagnose/{phase}/final")
                if phase == "route":
                    got_log, got_len = cache.take_op_log()
                    torch.testing.assert_close(got_log, log, atol=0, rtol=0)
                    torch.testing.assert_close(got_len, lengths, atol=0, rtol=0)
        assert calls["ladder"] and calls["centroid"], "diagnostic did not exercise both updates"
        print(json.dumps({"diagnose": "passed", **calls}), flush=True)
        return
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
                    assert_buffers(cache, expected, f"{name}/{phase}")
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
