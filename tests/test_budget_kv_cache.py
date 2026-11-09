"""Fixed-budget eviction baselines (litgpt/budget_kv_cache.py): exact while everything fits, never more than
the budget, each policy keeps what it should, and the byte budget matches the buffers."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from litgpt.budget_kv_cache import POLICIES, BudgetKVCache, capacity_for, slot_bytes
from litgpt.config import Config
from litgpt.kv_compression import cache_compression_snapshot
from litgpt.model import GPT

YARN = dict(factor=4.0, beta_fast=32, beta_slow=1, original_max_seq_len=64)


def tiny_model():
    torch.manual_seed(5)
    config = Config(block_size=256, n_layer=2, n_embd=32, n_head=4, n_query_groups=2, vocab_size=41,
                    padding_multiple=1, rotary_percentage=1., rope_adjustments=YARN)
    return GPT(config).eval()


@pytest.mark.parametrize("policy", POLICIES)
def test_a_budget_that_holds_everything_matches_the_uncached_model(policy):
    model = tiny_model()
    ids = torch.randint(0, 41, (1, 56))
    with torch.no_grad():
        full = model(ids).float()
        model.set_budget_kv_cache(1, 64, policy, chunk=8, window=4)
        cached = model(ids[:, :48], input_pos=torch.arange(48)).float()
        steps = [model(ids[:, i:i + 1], input_pos=torch.tensor([i])).float() for i in range(48, 56)]
    assert torch.allclose(cached, full[:, :48], atol=1e-4)  # chunked prefill (YaRN's mscale included)
    assert torch.allclose(torch.cat(steps, dim=1), full[:, 48:], atol=1e-4)  # decode
    model.reset_budget_kv_cache()
    assert all(block.attn.kv_cache.count == 0 for block in model.transformer.h)


@pytest.mark.parametrize("policy", POLICIES)
def test_the_cache_never_holds_more_than_its_budget(policy):
    model = tiny_model()
    ids = torch.randint(0, 41, (1, 100))
    with torch.no_grad():
        model.set_budget_kv_cache(1, 16, policy, chunk=8, window=4)
        model(ids[:, :90], input_pos=torch.arange(90))
        for i in range(90, 100):
            model(ids[:, i:i + 1], input_pos=torch.tensor([i]))
    for block in model.transformer.h:
        cache = block.attn.kv_cache
        assert cache.count == 16 and cache.seen == 100 and cache.k.size(2) == 16
        for row in cache.positions[0]:
            kept = set(row.tolist())
            assert len(kept) == 16 and max(kept) == 99 and min(kept) >= 0
            assert set(range(100 - cache.recent, 100)) <= kept  # the newest entries stay
            if policy == "streaming":
                assert kept == set(range(4)) | set(range(88, 100))
    snapshot = cache_compression_snapshot(model, 100)
    assert snapshot["exact_slots"] == 2 * 2 * 16 and snapshot["kv_compression_factor"] == pytest.approx(100 / 16)
    with pytest.raises(ValueError, match="append in order"):
        model(ids[:, :1], input_pos=torch.tensor([7]))


class _Attn:
    """The parts of CausalSelfAttention the cache uses: one head of size 4."""
    config = SimpleNamespace(attention_logit_softcapping=None, attention_scores_scalar=None, head_size=4)
    mscale = 1.0

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1 / math.sqrt(4)).transpose(1, 2)


@pytest.mark.parametrize("policy, keeps_the_hub", [("streaming", False), ("h2o", True), ("snapkv", True)])
def test_attention_scores_keep_the_token_every_query_reads(policy, keeps_the_hub):
    # Every query points along the first axis; only position 9's key does too, so it takes the attention.
    length, hub = 60, 9
    q = torch.zeros(1, 1, length, 4)
    q[..., 0] = 4.0
    k = torch.zeros(1, 1, length, 4)
    k[..., 1] = 1.0
    k[0, 0, hub] = torch.tensor([4.0, 0, 0, 0])
    v = torch.randn(1, 1, length, 4)
    cache = BudgetKVCache(1, 1, 4, 12, policy=policy, recent=4, window=4, chunk=8)
    for start in range(0, length, 8):
        cache.attend(_Attn, q[:, :, start:start + 8], k[:, :, start:start + 8], v[:, :, start:start + 8],
                     torch.arange(start, min(start + 8, length)))
    kept = cache.positions[0, 0].tolist()
    assert kept == sorted(kept) and len(set(kept)) == 12  # kept in position order
    assert (hub in kept) is keeps_the_hub and set(range(length - 4, length)) <= set(kept)


@pytest.mark.parametrize("policy", POLICIES)
def test_the_byte_budget_counts_the_buffers(policy):
    model = tiny_model()
    element = torch.empty((), dtype=torch.bfloat16).element_size()
    per_entry = slot_bytes(policy, 2, 8, element)
    budget = 2 * 37 * per_entry + per_entry - 1  # 37 entries in each of 2 layers, and not quite one more
    capacity = capacity_for(budget, policy, 2, 2, 8, element)
    assert capacity == 37
    model.set_budget_kv_cache(1, capacity, policy, dtype=torch.bfloat16)
    buffers = sum(b.untyped_storage().nbytes() for block in model.transformer.h
                  for b in block.attn.kv_cache.buffers())
    assert buffers == 2 * capacity * per_entry <= budget


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="unknown baseline"):
        BudgetKVCache(1, 1, 4, 8, policy="lru")
    with pytest.raises(ValueError, match="kernel"):
        BudgetKVCache(1, 1, 4, 8, policy="snapkv", kernel=4)
    with pytest.raises(ValueError, match="recent"):
        BudgetKVCache(1, 1, 4, 8, policy="h2o", recent=9)
