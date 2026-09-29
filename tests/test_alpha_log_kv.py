"""Whole-span ownership, delayed archival, fused packing and replay invariants."""
from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

from litgpt.alpha_log_kv import select_spans
from litgpt.config import Config
from litgpt.log_kv_cache import LogStructuredKVCache, LogKVStreamTrainingAttention, log_kv_chunk_attention
from litgpt.log_kv_checkpoint import enable_logkv_checkpoint_replay
from litgpt.log_kv_pack import pack_mid_kv
from litgpt.model import GPT, Block


def cache_for(device="cpu", dtype=torch.float32, **overrides):
    args = dict(B=8, recent_size=8, semantic_clusters=True, cluster_k_max=4,
                semantic_unified_route=True, semantic_merge_passes=4, semantic_anchor_mode="mid", allocate_second_order=False,
                semantic_replay_updates=True, semantic_flush_granularity=8,
                cos_cache=torch.ones(128, 8, device=device), sin_cache=torch.zeros(128, 8, device=device),
                rope_n_elem=8, alpha_exact_tokens=8, alpha_span_max_tokens=4)
    args.update(overrides)
    return LogStructuredKVCache((2, 2, 128, 8), (2, 2, 128, 8), device=device, dtype=dtype, **args)


def boundaries(n):
    return [[i % 3 == 2 for i in range(n)], [i % 4 == 3 for i in range(n)]]


def test_tokenizer_boundary_table_keeps_identifier_hyphens_and_colons():
    from litgpt.tokenizer import Tokenizer
    texts = ["abc", "-", ":", ".", "; ", "。”,", "。\"", "\n  ", "?", "<eos>"]
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.backend, tokenizer.eos_id = "huggingface", 9
    tokenizer.processor = SimpleNamespace(get_vocab_size=lambda **_: len(texts), decode=lambda ids: texts[ids[0]])
    assert tokenizer.alpha_span_boundary_ids == (3, 4, 6, 7, 8, 9)


def test_spans_are_whole_and_pending_crosses_flush_and_cap():
    k = torch.randn(1, 2, 7, 8)
    result = select_spans(k, k, [[(2, False)]], [[0, 1]], [[2, 3, 4, 5, 6]],
                          [[False, True, False, False, False]], 2, 7, 4)
    assert result.keep == [list(range(7))]
    assert result.spans == [[(4, True), (3, False)]]
    assert result.archive == [[]]
    # Completed old spans cannot be partially evicted to fill a small hole.
    k[:, :, :4] = 1.
    result = select_spans(k, k, [[(4, True)]], [[0, 1, 2, 3]], [[4, 5, 6]],
                          [[False, False, True]], 4, 4, 4)
    assert result.keep == [[4, 5, 6]]
    assert result.archive == [[0, 1, 2, 3]]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mass_payload_positions_and_budget_are_conserved(dtype, device):
    torch.manual_seed(17)
    c = cache_for(dtype=dtype, device=device)
    baseline = cache_for(dtype=dtype, device=device, alpha_exact_tokens=0)
    payload_bytes = lambda cache: sum(x.numel() * x.element_size() for x in cache.buffers())
    assert payload_bytes(c) <= payload_bytes(baseline)
    assert c.alpha_budget_bytes[1] <= c.alpha_budget_bytes[0]
    assert c.recent_size == baseline.recent_size == 8 and c.B < baseline.B
    k, v = (torch.randn(2, 2, 96, 8, device=device).to(dtype) for _ in range(2))
    ends = boundaries(96)
    delayed = c._alpha_commit_joins
    with patch.object(c, "_alpha_commit_joins", wraps=delayed) as joins:
        for start in range(0, 96, 8):
            stop = start + 8
            c.add_recent(k[:, :, start:stop], v[:, :, start:stop], k_raw=k[:, :, start:stop],
                         span_ends=[row[start:stop] for row in ends])
            mass = c.level_w.sum((2, 3, 4)) + c.alpha_valid.sum(-1)[:, None] + c.recent_count
            torch.testing.assert_close(mass, torch.full_like(mass, stop))
            for b, spans in enumerate(c._alpha_spans):
                assert sum(n for n, _ in spans) <= 8
                assert all(n <= 4 for n, _ in spans)
                positions = c.alpha_pos[b, c.alpha_valid[b]]
                assert positions.tolist() == c._alpha_positions[b]
                assert len(set(positions.tolist())) == len(positions)
                assert not set(positions.tolist()) & set(c._recent_pos_host[b][:c.recent_count])
                torch.testing.assert_close(c.alpha_k_raw[b, :, :len(positions)], k[b, :, positions], rtol=0, atol=0)
        assert joins.call_count
    for level, exact, recent, source in ((c.level_k, c.alpha_k_raw, c.recent_k_raw, k),
                                          (c.level_v, c.alpha_v, c.recent_v, v)):
        total = (level.float() * c.level_w[..., None]).sum((2, 3, 4))
        total += (exact.float() * c.alpha_valid[:, None, :, None]).sum(2)
        total += recent[:, :, :c.recent_count].float().sum(2)
        torch.testing.assert_close(total, source.float().sum(2), atol=.12 if dtype == torch.bfloat16 else 3e-5, rtol=.01)


def test_delayed_archive_updates_hi_and_preserves_original_weighted_positions():
    c = cache_for()
    k = torch.randn(2, 2, 8, 8)
    with torch.no_grad():
        c._semantic_new_clusters([(0, 0, 0, 0)], k, k, torch.tensor([[40] * 8, [0] * 8]), [[40] * 8, [0] * 8], record=False)
        c._alpha_commit_joins([(0, 0, 0, [0, 1])], k, k,
                              torch.tensor([[3, 4] + [0] * 6, [0] * 8]), [[3, 4] + [0] * 6, [0] * 8], record=False)
    assert c._semantic_p_hi_c[0][0][0] == 40
    assert c.level_w[0, 0, 0].sum() == 3
    assert c.level_sum_wp[0, 0, 0].sum() == 47
    assert c._semantic_n_total[0][0][0] == 3


def test_backward_replays_exact_pool_without_reselection_and_matches_naive_gradients():
    torch.manual_seed(31)
    originals = [torch.randn(2, 2, 40, 8) for _ in range(3)]
    results, grads = [], []
    ends = boundaries(40)
    for lowmem in (False, True):
        c = cache_for()
        q, k, v = [x.clone().requires_grad_() for x in originals]
        if lowmem:
            y = LogKVStreamTrainingAttention.apply(q, k, v, c, .3, 8, 0., k, ends)
        else:
            outputs = []
            for start in range(0, 40, 8):
                stop = start + 8
                outputs.append(log_kv_chunk_attention(c, q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop], .3, 0.))
                with torch.no_grad():
                    c.add_recent(k[:, :, start:stop], v[:, :, start:stop], k_raw=k[:, :, start:stop],
                                 span_ends=[row[start:stop] for row in ends])
            y = torch.cat(outputs, 2)
        with patch.object(c, "_alpha_route_flush", side_effect=AssertionError("rerouted during backward")):
            y.square().sum().backward()
        results.append(y.detach())
        grads.append([x.grad for x in (q, k, v)])
    for a, b in zip([results[0]] + grads[0], [results[1]] + grads[1]):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packing_includes_exact_rope_mask_and_current_gradients(device, dtype):
    c = cache_for(device=device, dtype=dtype)
    c.alpha_count = 3
    c.alpha_valid[:, :3] = torch.tensor([[True, True, True], [True, False, False]], device=device)
    c.alpha_pos[:, :3] = torch.tensor([[7, 9, 11], [5, 0, 0]], device=device)
    c.alpha_k_raw.normal_()
    c.alpha_v.normal_()
    phase = torch.arange(128, device=device).float()[:, None] * .1
    c.cos_cache.copy_(phase.cos().expand(-1, 8))
    c.sin_cache.copy_(phase.sin().expand(-1, 8))
    plan = c._semantic_attention_plan()
    k, v = [torch.randn(2, 2, 2, 8, device=device, dtype=dtype, requires_grad=True) for _ in range(2)]
    ka, va = pack_mid_kv(c, plan, k, v, 16, backend="torch")
    if device == "cuda":
        actual = pack_mid_kv(c, plan, k, v, 16, backend="triton")
        for a, b in zip(actual, (ka, va)):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    raw = c.alpha_k_raw[:, :, :3].float()
    cos = c.cos_cache[c.alpha_pos[:, :3]][:, None]
    sin = c.sin_cache[c.alpha_pos[:, :3]][:, None]
    expected = (cos * raw + sin * torch.cat((-raw[..., 4:], raw[..., :4]), -1)).to(dtype)
    expected.masked_fill_(~c.alpha_valid[:, None, :3, None], 0)
    torch.testing.assert_close(ka[:, :, :3, :8], expected)
    assert (ka[1, :, 1:3, 8] == -10000).all()
    (ka.sum() + va.sum()).backward()
    torch.testing.assert_close(k.grad, torch.ones_like(k))
    torch.testing.assert_close(v.grad, torch.ones_like(v))
    buffers = (torch.empty_like(ka), torch.empty_like(va))
    with torch.no_grad():
        a = pack_mid_kv(c, plan, k.detach(), v.detach(), 16, buffers=buffers)
        b = pack_mid_kv(c, plan, k.detach(), v.detach(), 16, buffers=buffers, skip_pooled=True)
        for x, y in zip(b, (ka, va)):
            torch.testing.assert_close(x, y)


def test_model_checkpoint_and_odd_prefill_decode():
    torch.manual_seed(55)
    config = Config(block_size=64, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                    vocab_size=41, padding_multiple=1, rotary_percentage=1.)
    original = GPT(config)
    models = [deepcopy(original), deepcopy(original)]
    kwargs = dict(batch_size=2, B=8, recent_size=8, second_order_scale=0., semantic_clusters=True,
                  cluster_k_max=4, semantic_unified_route=True, semantic_merge_passes=4, semantic_flush_granularity=8,
                  semantic_anchor_mode="mid", allocate_second_order=False, semantic_replay_updates=True,
                  alpha_exact_tokens=8, alpha_span_max_tokens=4)
    for model in models:
        model.set_alpha_span_boundary_ids([0, 3, 7])
        model.enable_log_kv_training(train_block=8, **kwargs)
    apply_activation_checkpointing(models[1], check_fn=lambda m: isinstance(m, Block))
    enable_logkv_checkpoint_replay(models[1], Block)
    xs = [torch.randint(41, (2, n)) for n in (24, 32)]
    losses, grads = [], []
    for model in models:
        ys = [model(x).square().mean() for x in xs]
        with patch.object(LogStructuredKVCache, "_alpha_route_flush", side_effect=AssertionError("rerouted")):
            ys[1].backward()
            ys[0].backward()
        losses.append(torch.stack([y.detach() for y in ys]))
        grads.append([p.grad.clone() for p in model.parameters()])
    for a, b in zip([losses[0]] + grads[0], [losses[1]] + grads[1]):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    model = original.eval()
    model.set_alpha_span_boundary_ids([0, 3, 7])
    model.set_log_kv_cache(prefill_block=8, **{k: v for k, v in kwargs.items() if k != "allocate_second_order"})
    with torch.no_grad():
        model(xs[0][:, :17], input_pos=torch.arange(17))
        model(xs[0][:, 17:18], input_pos=torch.tensor([17]))
    for block in model.transformer.h:
        c = block.attn.kv_cache
        assert c.token_count == 18
        assert block.attn._log_kv_pending is None
        assert (c.level_w.sum((2, 3, 4)) + c.alpha_valid.sum(-1)[:, None] + c.recent_count == 18).all()
