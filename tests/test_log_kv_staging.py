"""Strided K/V gathers and read-only staging reuse preserve cache writes."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from litgpt.log_kv_cache import LogStructuredKVCache, LogKVStreamTrainingAttention


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pads", [False, True])
def test_strided_staging_matches_reference_without_full_chunk_copies(device, dtype, pads):
    torch.manual_seed(15)
    k = torch.randn(2, 3, 80, 8, device=device, dtype=dtype)[:, :, 7:47]
    v = torch.randn(2, 80, 3, 12, device=device, dtype=dtype).transpose(1, 2)[:, :, 7:47]
    pos = torch.arange(80, device=device)[7:47].expand(2, -1)
    assert not k.is_contiguous() and not v.is_contiguous() and not pos.is_contiguous()
    before_k, before_v = k.clone(), v.clone()
    cache = LogStructuredKVCache(
        (2, 3, 128, 8), (2, 3, 128, 12), B=3, recent_size=4, device=device, dtype=dtype,
        semantic_clusters=True, cluster_k_max=8, allocate_second_order=True,
        seg_gap_max=2 if pads else None, seg_block_level=2 if pads else 0,
        cos_cache=torch.ones(128, 8, device=device), sin_cache=torch.zeros(128, 8, device=device),
        rope_n_elem=8,
    )
    cache.second_order = True
    for b in range(2):
        for g in range(3):
            cache._semantic_new_cluster(b, g, g, 0, k[b, g, 0], v[b, g, 0], pos.new_zeros(()), record=False)
    jobs = [(b, g, g, (0, 1, 5, 6, 12)) for b in range(2) for g in range(3)]
    expected_k = torch.stack([k[b, g, i] for b, g, _, offsets in jobs for i in offsets])
    expected_v = torch.stack([v[b, g, i] for b, g, _, offsets in jobs for i in offsets])
    expected_p = torch.stack([pos[b, i] for b, _, _, offsets in jobs for i in offsets])

    class Copies(TorchDispatchMode):
        def __init__(self):
            self.calls = []
            self.gathered = []
            self.active = True

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if self.active:
                self.calls.append(str(func))
                if func == torch.ops.aten.index.Tensor:
                    self.gathered.append(result)
            return result

    copies = Copies()
    append = cache._semantic_append_entries_batched

    def checked_append(lanes, counts, block):
        copies.active = False  # inspect staging preparation, not the ladder itself
        assert "aten.clone.default" not in copies.calls
        assert "aten.index_copy.default" not in copies.calls
        if not pads:
            assert "aten.index_copy_.default" not in copies.calls
            assert block[0].data_ptr() == copies.gathered[0].data_ptr()
            assert block[1].data_ptr() == copies.gathered[1].data_ptr()
        assert block[8].data_ptr() == block[9].data_ptr() == block[10].data_ptr()
        real = (~block[12]).nonzero().flatten()
        assert bool(block[12].any()) == pads
        for field, expected in ((0, expected_k), (1, expected_v), (8, expected_p),
                                (9, expected_p), (10, expected_p)):
            reference = torch.zeros_like(block[field]).index_copy_(0, real, expected)
            torch.testing.assert_close(block[field], reference, atol=0, rtol=0)
        saved = [x.clone() if x is not None else None for x in block]
        append(lanes, counts, block)
        for value, original in zip(block, saved):
            if value is not None:
                torch.testing.assert_close(value, original, atol=0, rtol=0)

    reference = deepcopy(cache)
    with patch.object(cache, "_semantic_append_entries_batched", checked_append), copies:
        cache._semantic_commit_joins(jobs, k, v, pos, pos.cpu().tolist())
    reference._semantic_commit_joins(jobs, k.contiguous(), v.contiguous(), pos.contiguous(), pos.cpu().tolist())
    reference_buffers = dict(reference.named_buffers())
    for name, value in cache.named_buffers():
        torch.testing.assert_close(value, reference_buffers[name], atol=0, rtol=0)
    torch.testing.assert_close(k, before_k, atol=0, rtol=0)
    torch.testing.assert_close(v, before_v, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strided_training_output_and_gradients_match_contiguous(dtype):
    torch.manual_seed(17)
    q = torch.randn(2, 4, 40, 8, dtype=dtype)[:, :, 3:27].requires_grad_()
    k = torch.randn(2, 2, 40, 8, dtype=dtype)[:, :, 3:27].requires_grad_()
    v = torch.randn(2, 40, 2, 8, dtype=dtype).transpose(1, 2)[:, :, 3:27].requires_grad_()
    upstream = torch.randn_like(q)

    def run(q, k, v):
        cache = LogStructuredKVCache(
            k.shape, v.shape, B=3, recent_size=4, dtype=dtype,
            semantic_clusters=True, cluster_k_max=8, allocate_second_order=False,
            semantic_flush_granularity=4, seg_gap_max=1, seg_block_level=2,
            cos_cache=torch.ones(40, 8), sin_cache=torch.zeros(40, 8), rope_n_elem=8,
        )
        output = LogKVStreamTrainingAttention.apply(q, k, v, cache, 8 ** -0.5, 4, 0., k)
        gradients = torch.autograd.grad(output, (q, k, v), upstream)
        return (output, *gradients)

    actual = run(q, k, v)
    expected = run(*(x.detach().contiguous().requires_grad_() for x in (q, k, v)))
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
