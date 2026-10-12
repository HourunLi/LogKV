"""Fused Alpha span scores, join staging and slot clears vs PyTorch on CPU.

Run with TRITON_INTERPRET=1 LOGKV_TRITON_CPU=1 (tests/test_log_kv_triton_interpret.py does).
"""
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
        with patch.object(kv, "_triton_alpha", lambda: None):
            expected = al.score_spans(k, v, cut, **options)
        actual = al.score_spans(k, v, cut, **options)
        # Sums run in another order: rounding only.
        torch.testing.assert_close(actual, expected.float(), rtol=1e-5, atol=1e-6,
                                   msg=f"fused span scores differ (groups={groups}, dim={dim}, beta={beta})")


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
