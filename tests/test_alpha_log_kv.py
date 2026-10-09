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


ROUTES = ["attach", "unified"]


def cache_for(device="cpu", dtype=torch.float32, route="attach", **overrides):
    args = dict(B=8, recent_size=8, semantic_clusters=True, cluster_k_max=4,
                semantic_unified_route=route == "unified", semantic_merge_passes=1, semantic_anchor_mode="mid", allocate_second_order=False,
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


def _token_loop_runs(new_pos, ends, pending, pending_pos, max_span):
    """The per-token span rule that `_new_runs` vectorizes."""
    current, out, pending_closed = list(range(-pending, 0)), [], False
    last = pending_pos
    for j, (pos, end) in enumerate(zip(new_pos, ends)):
        if current and pos != last + 1:
            if current[0] < 0 and current[-1] < 0:
                pending_closed = True
            else:
                out.append((max(current[0], 0), current[-1] + 1, True))
            current = []
        current.append(j)
        last = pos
        if end or len(current) == max_span:
            out.append((max(current[0], 0), current[-1] + 1, True))
            current = []
    if current and current[-1] >= 0:
        out.append((max(current[0], 0), current[-1] + 1, False))
    return pending_closed, out


def test_vectorized_span_runs_match_token_loop():
    from litgpt.alpha_log_kv import _new_runs

    generator = torch.Generator().manual_seed(11)
    for _ in range(400):
        count, max_span = int(torch.randint(0, 30, (), generator=generator)), int(torch.randint(1, 6, (), generator=generator))
        pending = int(torch.randint(0, max_span + 2, (), generator=generator))
        steps = 1 + (torch.rand(count, generator=generator) < .1).long() * torch.randint(1, 4, (count,), generator=generator)
        new_pos = (100 + steps.cumsum(0)).tolist()
        ends = (torch.rand(count, generator=generator) < .2).tolist()
        pending_pos = new_pos[0] - int(torch.randint(1, 3, (), generator=generator)) if count else 0
        closed, starts, stops, done = _new_runs(new_pos, ends, pending, pending_pos, max_span)
        assert (closed, list(zip(starts.tolist(), stops.tolist(), done.tolist()))) == \
            _token_loop_runs(new_pos, ends, pending, pending_pos, max_span)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("archive_width", [0, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_exact_partition_copies_payload_and_padding(archive_width, dtype):
    from litgpt.log_kv_updates_triton import alpha_partition

    k = torch.randn(2, 3, 13, 7, device="cuda", dtype=dtype)
    v = torch.randn(2, 3, 13, 11, device="cuda", dtype=dtype)
    pos = torch.arange(26, device="cuda").view(2, 13)
    index = torch.tensor([[12, 7, -1, -1, 0, 5, 9, 3, 4],
                          [0, 1, 12, -1, 7, 8, -1, -1, -1]], device="cuda")[:, :4 + archive_width].contiguous()
    exact = (k.new_full((2, 3, 4, 7), float("nan")), v.new_full((2, 3, 4, 11), float("nan")),
             pos.new_full((2, 4), -9), torch.ones(2, 4, device="cuda", dtype=torch.bool))
    archive = alpha_partition(k, v, pos, index, exact)
    mask, src = index >= 0, index.clamp_min(0)
    for expected, actual in ((k.gather(2, src[:, None, :, None].expand(-1, 3, -1, 7)), torch.cat((exact[0], archive[0]), 2)),
                              (v.gather(2, src[:, None, :, None].expand(-1, 3, -1, 11)), torch.cat((exact[1], archive[1]), 2))):
        expected.masked_fill_(~mask[:, None, :, None], 0)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat((exact[2], archive[2]), 1), pos.gather(1, src).masked_fill(~mask, 0), rtol=0, atol=0)
    torch.testing.assert_close(exact[3], mask[:, :4], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("route", ROUTES)
def test_mass_payload_positions_and_budget_are_conserved(dtype, device, beta, route):
    torch.manual_seed(17)
    c = cache_for(dtype=dtype, device=device, route=route, beta_novelty=beta, beta_adaptive_merge=beta)
    baseline = cache_for(dtype=dtype, device=device, route=route, alpha_exact_tokens=0)
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
            # Delayed archival keeps each live cluster's p_hi/n_total consistent with its ladder.
            live = c.level_w.sum(-1).sum(-1) > 0
            assert torch.equal(live, c.alive)
            assert (c.level_p_hi.amax((-1, -2)).masked_fill(~live, 0) <= c.p_hi_c.masked_fill(~live, 0)).all()
            torch.testing.assert_close(c.level_w.sum((-1, -2)), c.n_total.to(c.level_w.dtype))
        assert joins.call_count
    for level, exact, recent, source in ((c.level_k, c.alpha_k_raw, c.recent_k_raw, k),
                                          (c.level_v, c.alpha_v, c.recent_v, v)):
        total = (level.float() * c.level_w[..., None]).sum((2, 3, 4))
        total += (exact.float() * c.alpha_valid[:, None, :, None]).sum(2)
        total += recent[:, :, :c.recent_count].float().sum(2)
        torch.testing.assert_close(total, source.float().sum(2), atol=.12 if dtype == torch.bfloat16 else 3e-5, rtol=.01)


@pytest.mark.parametrize("route", ROUTES)
def test_ordinary_and_delayed_joins_share_one_ladder_append(route):
    torch.manual_seed(5)
    c = cache_for(route=route, beta_novelty=True, beta_adaptive_merge=True)
    c.begin_op_log()
    commit, append = c._alpha_commit_joins, c._semantic_append_entries_batched
    checked = []

    def split_reference(jobs, k_raw, v, positions, host, *, record):
        late = [min(host[b][i] for i in offsets) < c._semantic_p_hi_c[b][g][cluster]
                for b, g, cluster, offsets in jobs]
        reference = deepcopy(c)
        for kind in (False, True):
            part = [job for job, is_late in zip(jobs, late) if is_late == kind]
            # deepcopy also copies the instance patch; call the real method.
            LogStructuredKVCache._alpha_commit_joins(reference, part, k_raw, v, positions, host, record=record)
        with patch.object(c, "_semantic_append_entries_batched", wraps=append) as appends:
            commit(jobs, k_raw, v, positions, host, record=record)
        if any(late) and not all(late):
            assert appends.call_count == 1
            checked.append(True)
        for name, value in c.named_buffers():
            torch.testing.assert_close(value, reference.get_buffer(name), rtol=0, atol=0, msg=name)
        for name in c._UPDATE_HOST_FIELDS + ("_semantic_counts", "_op_log_host"):
            assert getattr(c, name) == getattr(reference, name), name

    k, v = torch.randn(2, 2, 96, 8), torch.randn(2, 2, 96, 8)
    ends = boundaries(96)
    with patch.object(c, "_alpha_commit_joins", side_effect=split_reference):
        for start in range(0, 96, 8):
            c.add_recent(k[:, :, start:start + 8], v[:, :, start:start + 8], k_raw=k[:, :, start:start + 8],
                         span_ends=[row[start:start + 8] for row in ends], record_op_log=True)
    assert checked


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


@pytest.mark.parametrize("merge_passes", [1, 4])
def test_ragged_archive_excludes_padding_and_preserves_empty_lanes(merge_passes):
    torch.manual_seed(71)
    c = cache_for(route="unified", semantic_merge_passes=merge_passes)
    expected_mass = torch.zeros(2, 2)
    expected_sum = torch.zeros(2, 2, 8)
    # One lane per tile also exercises a whole empty tile, not only padding
    # next to a nonempty lane. Later flushes archive older exact positions.
    with patch.object(c, "_semantic_unified_tile", return_value=1), torch.no_grad():
        for step, counts in enumerate(([0, 1], [7, 0], [1, 7], [0, 0])):
            k = torch.randn(2, 2, 7, 8)
            pos = torch.arange(30 - 7 * step, 23 - 7 * step, -1).expand(2, -1)
            for b, count in enumerate(counts):
                expected_mass[b] += count
                expected_sum[b] += k[b, :, :count].sum(1)
                k[b, :, count:] = 10000  # Padding must never become an entry.
            c._semantic_route_unified(k, k, pos, pos.tolist(), record=False, token_counts=counts)
            torch.testing.assert_close(c.level_w.sum((2, 3, 4)), expected_mass)
            actual = (c.level_k.float() * c.level_w[..., None]).sum((2, 3, 4))
            torch.testing.assert_close(actual, expected_sum, atol=3e-6, rtol=3e-5)
            assert (c.alive.sum(-1) <= c.K_max).all()


def test_attach_ragged_archive_excludes_padding_and_inserts_old_positions():
    torch.manual_seed(73)
    c = cache_for(route="attach")
    expected_mass = torch.zeros(2, 2)
    expected_sum = torch.zeros(2, 2, 8)
    delayed = c._alpha_commit_joins
    with patch.object(c, "_alpha_commit_joins", wraps=delayed) as joins, torch.no_grad():
        # Later flushes archive older positions, as evicted exact spans do.
        for step, counts in enumerate(([0, 1], [7, 0], [1, 7], [0, 0], [5, 3])):
            k = torch.randn(2, 2, 7, 8)
            pos = torch.arange(40 - 7 * step, 33 - 7 * step, -1).expand(2, -1)
            for b, count in enumerate(counts):
                expected_mass[b] += count
                expected_sum[b] += k[b, :, :count].sum(1)
                k[b, :, count:] = 10000  # Padding must never become an entry.
            with c._semantic_deferred_scalars():
                c._semantic_route_three_phase(k, k, pos, pos.tolist(), record=False, token_counts=counts)
            torch.testing.assert_close(c.level_w.sum((2, 3, 4)), expected_mass)
            torch.testing.assert_close(c.n_total.sum(-1).float(), expected_mass)
            actual = (c.level_k.float() * c.level_w[..., None]).sum((2, 3, 4))
            torch.testing.assert_close(actual, expected_sum, atol=3e-6, rtol=3e-5)
            assert (c.alive.sum(-1) <= c.K_max).all()
            assert (c.level_p_hi.amax((-1, -2)) <= c.p_hi_c).all()
    assert joins.call_count


def test_alpha_flush_routes_by_configured_route():
    k = torch.randn(2, 2, 24, 8)
    ends = boundaries(24)
    for route, used, unused in (("attach", "_semantic_route_three_phase", "_semantic_route_unified"),
                                ("unified", "_semantic_route_unified", "_semantic_route_three_phase")):
        c = cache_for(route=route)
        with patch.object(c, unused, side_effect=AssertionError(unused)), \
             patch.object(c, used, wraps=getattr(c, used)) as called:
            for start in range(0, 24, 8):
                c.add_recent(k[:, :, start:start + 8], k[:, :, start:start + 8], k_raw=k[:, :, start:start + 8],
                             span_ends=[row[start:start + 8] for row in ends])
        assert called.call_count
        assert all(call.kwargs["token_counts"] is not None for call in called.call_args_list)


@pytest.mark.parametrize("overrides", [dict(semantic_legacy_route=True), dict(semantic_cluster_chunk_size=4),
                                       dict(semantic_anchor_mode="multi"), dict(cluster_k_max=1)])
def test_alpha_rejects_routes_and_layouts_it_cannot_archive_into(overrides):
    with pytest.raises(ValueError):
        cache_for(**overrides)


@pytest.mark.parametrize("route,merge_passes", [("attach", 1), ("unified", 1), ("unified", 4)])
@pytest.mark.parametrize("beta", [False, True])
def test_backward_replays_exact_pool_without_reselection_and_matches_naive_gradients(route, merge_passes, beta):
    torch.manual_seed(31)
    originals = [torch.randn(2, 2, 40, 8) for _ in range(3)]
    results, grads = [], []
    ends = boundaries(40)
    for lowmem in (False, True):
        c = cache_for(route=route, semantic_merge_passes=merge_passes, beta_novelty=beta, beta_adaptive_merge=beta)
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
        with patch.object(c, "_alpha_route_flush", side_effect=AssertionError("rerouted during backward")), \
             patch("litgpt.beta_log_kv.select_cuts", side_effect=AssertionError("reselected cuts during backward")):
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


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("route", ROUTES)
def test_model_checkpoint_and_odd_prefill_decode(beta, route):
    torch.manual_seed(55)
    config = Config(block_size=64, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                    vocab_size=41, padding_multiple=1, rotary_percentage=1.)
    original = GPT(config)
    models = [deepcopy(original), deepcopy(original)]
    kwargs = dict(batch_size=2, B=8, recent_size=8, second_order_scale=0., semantic_clusters=True,
                  cluster_k_max=4, semantic_unified_route=route == "unified", semantic_merge_passes=1,
                  semantic_flush_granularity=8,
                  semantic_anchor_mode="mid", allocate_second_order=False, semantic_replay_updates=True,
                  alpha_exact_tokens=8, alpha_span_max_tokens=4, beta_novelty=beta, beta_adaptive_merge=beta)
    for model in models:
        model.set_alpha_span_boundary_ids([0, 3, 7])
        model.enable_log_kv_training(train_block=8, **kwargs)
    apply_activation_checkpointing(models[1], check_fn=lambda m: isinstance(m, Block))
    enable_logkv_checkpoint_replay(models[1], Block)
    xs = [torch.randint(41, (2, n)) for n in (24, 32)]
    losses, grads = [], []
    for model in models:
        ys = [model(x).square().mean() for x in xs]
        with patch.object(LogStructuredKVCache, "_alpha_route_flush", side_effect=AssertionError("rerouted")), \
             patch("litgpt.beta_log_kv.select_cuts", side_effect=AssertionError("reselected cuts")):
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


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("beta", [False, True])
def test_score_prefetch_reproduces_on_demand_selection(route, beta):
    torch.manual_seed(91)
    k, v = torch.randn(2, 2, 96, 8), torch.randn(2, 2, 96, 8)
    ends = boundaries(96)
    caches, hits = [], []
    for prefetch in (False, True):
        c = cache_for(route=route, beta_novelty=beta, beta_adaptive_merge=beta)
        used, original = [], c._alpha_take_prefetch

        def take(*args):
            result = original(*args)
            used.append(result is not None)
            return result

        with patch.object(LogStructuredKVCache, "_alpha_prefetch_on_cpu", prefetch), \
             patch.object(c, "_alpha_take_prefetch", side_effect=take):
            for start in range(0, 96, 8):
                c.add_recent(k[:, :, start:start + 8], v[:, :, start:start + 8], k_raw=k[:, :, start:start + 8],
                             span_ends=[row[start:start + 8] for row in ends])
        hits.append(sum(used))
        caches.append(c)
    assert hits[0] == 0 and hits[1] > 0
    for name, value in caches[0].named_buffers():
        torch.testing.assert_close(value, caches[1].get_buffer(name), rtol=0, atol=0, msg=name)
    assert caches[0]._alpha_spans == caches[1]._alpha_spans
    assert caches[0]._alpha_positions == caches[1]._alpha_positions
