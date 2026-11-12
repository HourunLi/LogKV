"""Fused Alpha span scores, join staging, slot clears and top-level merges vs PyTorch on CPU.

Run with TRITON_INTERPRET=1 LOGKV_TRITON_CPU=1 (tests/test_log_kv_triton_interpret.py does).
"""
import copy
from unittest.mock import patch

import numpy as np
import torch

import litgpt.alpha_log_kv as al
import litgpt.log_kv_cache as kv
from litgpt import log_kv_updates_triton as fused


def main():
    assert kv._fused_on(torch.device("cpu")), "run with TRITON_INTERPRET=1 LOGKV_TRITON_CPU=1"
    check_span_scores()
    check_stage_and_clear()
    check_top_lightest()
    check_top_level_bursts()
    check_cache_trajectory()
    print("update kernels match the Torch reference")


def check_span_scores():
    """One program per span against the batched reference, Beta and plain."""
    generator = torch.Generator().manual_seed(98)
    rng = np.random.default_rng(98)
    for groups, dim, dtype, beta in ((1, 7, torch.float32, True), (3, 16, torch.bfloat16, True),
                                     (2, 16, torch.float32, False), (3, 7, torch.bfloat16, False)):
        old_spans = [[(3, True), (2, True), (1, False)], [(4, True)]]
        old_positions = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3]]
        ends = [(rng.random(20) < .3).tolist() for _ in range(2)]
        cut = al.cut_spans(old_spans, old_positions, [list(range(10, 30))] * 2, ends, 6, 5)
        k = torch.randn(2, groups, 26, dim, generator=generator).to(dtype)
        v = (3 * torch.randn(2, groups, 26, dim, generator=generator)).to(dtype)
        k[0, :, 3:5] = 0  # A span without energy.
        centroids = torch.randn(2, groups, 5, dim, generator=generator)
        valid = torch.rand(2, groups, 5, generator=generator) < .6
        valid[1, 0] = False  # A lane without live clusters.
        options = dict(beta_novelty=beta, centroids=centroids, centroid_valid=valid)
        # niahLogKV: a learned head's span values join on the device in both paths.
        from litgpt.retention import RetentionHead
        head = RetentionHead(groups, dim, dim, hidden=8)
        torch.nn.init.normal_(head.out.weight, std=3., generator=generator)
        torch.nn.init.constant_(head.out.bias, 1.)
        for retention in (None, head):
            with patch.object(kv, "_triton_alpha", lambda: None):
                expected = al.score_spans(k, v, cut, retention=retention, **options)
            actual = al.score_spans(k, v, cut, retention=retention, **options)
            # Sums run in another order: rounding only.
            torch.testing.assert_close(actual, expected.float(), rtol=1e-5, atol=1e-6,
                                       msg=f"fused span scores differ (groups={groups}, dim={dim}, beta={beta}, "
                                           f"head={retention is not None})")


def check_stage_and_clear():
    """Staged rows equal the concatenate-then-gather reference; clears equal `index_fill_`."""
    generator = torch.Generator().manual_seed(99)
    for dtype, k_dtype in ((torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16),
                           (torch.bfloat16, torch.float32)):
        cache = kv.LogStructuredKVCache(
            (2, 3, 64, 8), (2, 3, 64, 5), B=4, recent_size=16, dtype=dtype, semantic_clusters=True,
            cluster_k_max=3, allocate_second_order=False, semantic_flush_granularity=16,
            cos_cache=torch.ones(64, 8), sin_cache=torch.zeros(64, 8), rope_n_elem=8)
        fields = cache._semantic_flat_level_fields()
        for i, field in enumerate(fields):
            if field is not None:
                field.copy_(torch.randint(-50, 50, field.shape, generator=generator).to(field.dtype))
        # Strided token sources, like slices of the recent window.
        k_raw = torch.randn(2, 3, 20, 8, generator=generator).to(k_dtype)[:, :, 2:18]
        v = torch.randn(2, 3, 20, 5, generator=generator).to(k_dtype)[:, :, 1:17]
        positions = torch.randint(0, 1000, (2, 20), generator=generator)[:, 3:19]
        head = [torch.randint(0, n, (7,), generator=generator) for n in (2, 3, 16)]
        new = [torch.randint(0, n, (5,), generator=generator) for n in (2, 3, 16)]
        ladder = torch.randperm(fields[0].size(0), generator=generator)[:6]
        perm = torch.randperm(11, generator=generator)
        head_order = torch.randint(0, 99, (7,), generator=generator)
        phase = torch.randint(0, 99, (11,), generator=generator)
        block = cache._semantic_stage_block(18, k_raw.dtype, v.dtype)
        fused.stage(block, fields, k_raw, v, positions, head, head_order)
        fused.stage(block, fields, k_raw, v, positions, new, phase, start=7, ladder_rows=ladder, perm=perm)

        def tokens(index, order):
            b, g, t = index
            p = positions[b, t]
            return (k_raw[b, g, t], v[b, g, t], torch.ones(len(t)), None, None, None, None, None,
                    p, p, p, order, torch.zeros(len(t), dtype=torch.bool))

        for i, (out, field) in enumerate(zip(block, fields)):
            if out is None:
                continue
            fresh = tokens(new, phase)[i]
            tail = torch.cat((field[ladder].to(out.dtype), fresh.to(out.dtype)))[perm]
            if i == 11:
                tail = phase
            expected = torch.cat((tokens(head, head_order)[i].to(out.dtype), tail.to(out.dtype)))
            assert torch.equal(out, expected), f"staged field {i} differs (dtype={dtype}, tokens {k_dtype})"

        expected = [None if f is None else f.clone() for f in fields]
        imp = cache.level_imp.view(-1)
        imp.copy_(torch.rand(imp.shape, generator=generator))
        expected_imp = imp.clone()
        starts = torch.tensor([cache.B * 3, 0, cache.B * 7])
        for start in starts.tolist():
            for x in expected + [expected_imp]:
                if x is not None:
                    x[start:start + 2 * cache.B] = 0
        fused.clear(fields, imp, starts, 2 * cache.B)
        for i, (actual, want) in enumerate(zip(list(fields) + [imp], expected + [expected_imp])):
            if want is not None:
                assert torch.equal(actual, want), f"cleared field {i} differs (dtype={dtype})"


def check_top_lightest():
    """Saturated Gamma top levels: the one-program-per-level kernel against the batched PyTorch steps.

    Masses repeat, so ties (the oldest pair merges) are exercised. Merge decisions
    and metadata are exact in every dtype; bf16 K/V differ only by the
    interpreter's truncating fp32->bf16 casts (CUDA rounds to nearest, like Torch).
    """
    for dtype, stage_dtype, B in ((torch.float32, torch.float32, 2), (torch.float32, torch.float32, 5),
                                  (torch.bfloat16, torch.float32, 5), (torch.bfloat16, torch.bfloat16, 4)):
        generator = torch.Generator().manual_seed(B)
        cache = kv.LogStructuredKVCache(
            (2, 3, 512, 8), (2, 3, 512, 6), B=B, recent_size=16, dtype=dtype, semantic_clusters=True,
            cluster_k_max=3, allocate_second_order=False, semantic_flush_granularity=16, gamma_top_merge="lightest",
            cos_cache=torch.ones(512, 8), sin_cache=torch.zeros(512, 8), rope_n_elem=8)
        fields = cache._semantic_flat_level_fields()
        n = fields[0].size(0)

        def rows(count, dims, low):
            k, v = torch.randn(count, dims[0], generator=generator), torch.randn(count, dims[1], generator=generator)
            lo = torch.randint(0, 900, (count,), generator=generator)
            return (k, v, torch.randint(low, 6, (count,), generator=generator).float(), lo,
                    lo + torch.randint(0, 50, (count,), generator=generator),
                    torch.randint(0, 9999, (count,), generator=generator),
                    torch.randint(0, 99, (count,), generator=generator), torch.rand(count, generator=generator) < .3)

        for i, x in zip(fused._FIELDS, rows(n, (8, 6), 0)):
            fields[i].copy_(x)
        cache.level_imp.copy_(torch.rand(cache.level_imp.shape, generator=generator))
        stage = [None] * 13
        for i, x in zip(fused._FIELDS, rows(60, (8, 6), 1)):
            stage[i] = x.to(stage_dtype) if i < 2 else x
        stage = tuple(stage)
        # Lanes overflow by 1 up to 5B entries; one lane per (b, g, c).
        jobs = [(0, 0, 1, 0, 1), (0, 2, 0, 1, 12), (1, 1, 2, 13, 3), (1, 2, 1, 20, 7), (1, 0, 0, 30, 25)]
        reference = copy.deepcopy(cache)
        reference._semantic_top_level_lightest_batched(jobs, stage)
        cache._semantic_top_level_lightest_batched(jobs, stage, fused)
        for (name, actual), (_, expected) in zip(cache.named_buffers(), reference.named_buffers()):
            if dtype == torch.bfloat16 and name in ("level_k", "level_v"):
                torch.testing.assert_close(actual, expected, rtol=2 ** -6, atol=2 ** -6, msg=f"top level {name} ({dtype})")
            else:
                assert torch.equal(actual, expected), f"top level {name} differs (B={B}, {dtype}, stage {stage_dtype})"


def check_top_level_bursts():
    """Bursts that keep Gamma top levels saturated, through the cache's own append path."""
    def run(use_kernel):
        torch.manual_seed(13)
        cache = kv.LogStructuredKVCache(
            (2, 2, 16, 8), (2, 2, 16, 8), B=3, recent_size=4, semantic_clusters=True, cluster_k_max=2,
            allocate_second_order=False, semantic_anchor_mode="mid", semantic_unified_route=True,
            alpha_exact_tokens=2, alpha_span_max_tokens=2, beta_adaptive_merge=True, beta_novelty=True,
            gamma_top_merge="lightest", cos_cache=torch.ones(64, 8), sin_cache=torch.zeros(64, 8), rope_n_elem=8)
        lanes = [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]
        capacity = cache.B * (2 ** cache.L_alloc - 1)
        steps = []
        overflow = kv.LogStructuredKVCache._semantic_top_level_lightest_batched

        def spy(self, jobs, stage, kernel=None):
            assert kernel is fused, "the append path did not hand the fused kernels to the top level"
            steps.append(max(job[4] for job in jobs))
            return overflow(self, jobs, stage, kernel if use_kernel else None)

        position = 0
        with patch.object(kv.LogStructuredKVCache, "_semantic_top_level_lightest_batched", spy):
            for counts in ([capacity - 1, 2, capacity, 1], [5, capacity + 3, 1, 7], [9, 4, 2 * capacity, 3],
                           [3 * capacity, capacity, capacity, 2 * capacity]):
                total = sum(counts)
                k, v = torch.randn(total, 8), torch.randn(total, 8)
                pos = torch.arange(position, position + total)
                position += total
                block = (k, v, torch.ones(total), *cache._empty_stats(k, v), pos, pos, pos,
                         torch.arange(total), torch.zeros(total, dtype=torch.bool))
                cache._semantic_append_entries_batched(lanes, counts, block)
        return cache, steps

    reference, steps = run(False)
    actual, _ = run(True)
    assert len(steps) >= 3 and max(steps) > reference.B, f"the bursts did not keep top levels saturated: {steps}"
    assert actual._semantic_counts == reference._semantic_counts
    for (name, a), (_, b) in zip(actual.named_buffers(), reference.named_buffers()):
        assert torch.equal(a, b), f"cache buffer {name} differs between the fused and PyTorch top-level overflow"


def check_cache_trajectory():
    """Whole Gamma flushes: fused staging leaves the same cache as PyTorch staging."""
    def run(staged):
        torch.manual_seed(5)
        rng = np.random.default_rng(5)
        cache = kv.LogStructuredKVCache(
            (2, 2, 256, 8), (2, 2, 256, 8), B=8, recent_size=16, semantic_clusters=True, cluster_k_max=4,
            semantic_anchor_mode="mid", allocate_second_order=False, semantic_flush_granularity=16,
            cos_cache=torch.ones(256, 8), sin_cache=torch.zeros(256, 8), rope_n_elem=8,
            alpha_exact_tokens=16, alpha_span_max_tokens=8, beta_novelty=True, beta_adaptive_merge=True,
            gamma_level0_reinsert=True, gamma_top_merge="lightest")
        protos = torch.randn(2, 2, 6, 8)
        calls = []
        stage = fused.stage

        def counted(*args, **kwargs):
            calls.append("tail" if kwargs.get("perm") is not None else "head")
            return stage(*args, **kwargs)

        for step in range(12):
            labels = torch.randint(6, (2, 2, 4)).repeat_interleave(4, -1)
            k = protos.gather(2, labels[..., None].expand(-1, -1, -1, 8)) + .3 * torch.randn(2, 2, 16, 8)
            host = [list(range(16 * step, 16 * step + 16))] * 2
            ends = [(rng.random(16) < .3).tolist() for _ in range(2)]
            with patch.object(fused, "stage", counted), \
                    patch.object(kv.LogStructuredKVCache, "_semantic_fused_stage",
                                 kv.LogStructuredKVCache._semantic_fused_stage if staged else lambda *_: None):
                cache.route_and_flush_batch(k, torch.randn(2, 2, 16, 8), torch.arange(16 * step, 16 * step + 16),
                                            positions_host=host, span_ends=ends)
        return cache, calls

    reference, _ = run(False)
    actual, calls = run(True)
    assert "tail" in calls and "head" in calls, f"the fused staging paths did not run: {calls}"
    for (name, a), (_, b) in zip(actual.named_buffers(), reference.named_buffers()):
        assert torch.equal(a, b), f"cache buffer {name} differs between fused and PyTorch staging"


if __name__ == "__main__":
    main()
