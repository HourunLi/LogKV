"""Real Block checkpoints must replay the matching forward log, never reroute."""

from copy import deepcopy
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from unittest.mock import patch
import gc
import os
import weakref

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
)

from litgpt.config import Config
from litgpt.model import GPT, Block
from litgpt.log_kv_cache import LogStructuredKVCache
from litgpt.log_kv_checkpoint import (
    enable_logkv_checkpoint_replay, logkv_checkpoint_contexts, checkpoint_route_replay,
    checkpoint_record_routes, _route_pass,
)
from litgpt.log_kv_timing import logkv_begin_step, logkv_take_host_stats


def _models(*, device="cpu", scale=0., segments=False, mode="mid"):
    torch.manual_seed(91)
    config = Config(block_size=32, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                    vocab_size=41, padding_multiple=1, rotary_percentage=1.0)
    reference = GPT(config).to(device)
    fast = deepcopy(reference)
    for model in (reference, fast):
        model.enable_log_kv_training(
            batch_size=1, B=3, recent_size=4, train_block=4, second_order_scale=scale,
            semantic_clusters=True, cluster_k_max=8, semantic_flush_granularity=4,
            semantic_anchor_mode=mode, semantic_pack_backend="torch", allocate_second_order=scale != 0.,
            seg_gap_max=1 if segments else None, seg_block_level=2 if segments else 0,
            device=torch.device(device),
        )
        apply_activation_checkpointing(model, check_fn=lambda m: isinstance(m, Block))
    assert enable_logkv_checkpoint_replay(fast, Block) == 2
    return reference, fast


@pytest.mark.parametrize("scale,segments,mode", [(0., False, "mid"), (0., True, "multi"), (.2, False, "mid")])
def test_checkpoint_replay_matches_loss_gradients_and_halves_routing(scale, segments, mode):
    reference, fast = _models(scale=scale, segments=segments, mode=mode)
    inputs = [torch.randint(41, (1, 24)) for _ in range(2)]
    targets = [torch.randint(41, (1, 24)) for _ in range(2)]

    def run(model):
        logkv_begin_step()
        losses = []
        for x, target in zip(inputs, targets):
            loss = model(x, targets=target, loss_chunk_size=7)
            losses.append(loss.detach())
            guard = patch.object(LogStructuredKVCache, "_semantic_route_three_phase",
                                 side_effect=AssertionError("checkpoint backward rerouted")) if model is fast else nullcontext()
            with guard:
                (loss / 2).backward()
        states = [(m.level_k.clone(), m.level_w.clone()) for m in model.modules() if isinstance(m, LogStructuredKVCache)]
        return losses, [p.grad.clone() for p in model.parameters()], states, logkv_take_host_stats()

    with patch.object(LogStructuredKVCache, "_semantic_build_replay_plan",
                      wraps=LogStructuredKVCache._semantic_build_replay_plan) as parse:
        before = run(reference)
        old_parses = parse.call_count
        parse.reset_mock()
        after = run(fast)
        new_parses = parse.call_count
    for a, b in zip(before[0] + before[1], after[0] + after[1]):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    for a, b in zip(before[2], after[2]):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    old, new = before[3], after[3]
    assert new["route_n"] > 0 and old["route_n"] == 2 * new["route_n"]
    assert new["replay_n"] == 2 * old["replay_n"]
    assert old_parses == new_parses == old["replay_n"]  # second replay reuses each parsed flush
    assert new["plan_n"] == old["plan_n"]
    assert _route_pass.get() is None


def test_two_outstanding_forwards_and_retained_graph_use_their_own_records():
    reference, fast = _models()
    inputs = [torch.randint(41, (1, n)) for n in (20, 24)]
    targets = [torch.randint(41, (1, n)) for n in (20, 24)]

    def run(model):
        losses = [model(x, targets=t, loss_chunk_size=7) for x, t in zip(inputs, targets)]
        losses[0].backward(retain_graph=True)
        losses[1].backward()
        losses[0].backward()
        return [loss.detach() for loss in losses], [p.grad for p in model.parameters()]

    before, after = run(reference), run(fast)
    for a, b in zip(before[0] + before[1], after[0] + after[1]):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert _route_pass.get() is None


def test_missing_or_mismatched_record_raises_and_context_is_restored():
    _, model = _models()
    cache = next(m for m in model.modules() if isinstance(m, LogStructuredKVCache))
    forward, replay = logkv_checkpoint_contexts()
    with pytest.raises(RuntimeError, match="no matching"):
        with replay:
            checkpoint_route_replay(cache, ("input",))
    assert _route_pass.get() is None
    log = (torch.zeros(1), torch.zeros(1), [])
    with forward:
        checkpoint_record_routes(cache, ("first input",), log)
    with pytest.raises(RuntimeError, match="does not match"):
        with replay:
            checkpoint_route_replay(cache, ("different input",))
    assert _route_pass.get() is None


def test_record_tensors_release_when_checkpoint_graph_is_discarded():
    _, model = _models()
    refs = []
    original = LogStructuredKVCache.take_op_log

    def take(cache):
        log, lengths = original(cache)
        refs.append(weakref.ref(log))
        return log, lengths

    with patch.object(LogStructuredKVCache, "take_op_log", take):
        loss = model(torch.randint(41, (1, 24)), targets=torch.randint(41, (1, 24)), loss_chunk_size=7)
    assert refs and all(ref() is not None for ref in refs)
    del loss
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert _route_pass.get() is None


def test_cpu_replay_plans_release_with_retained_checkpoint_graph():
    import litgpt.log_kv_cache as cache_module

    _, model = _models()
    refs = []
    original = cache_module._SemanticReplayPlans

    def create(log):
        plans = original(log)
        refs.append(weakref.ref(plans))
        return plans

    with patch.object(cache_module, "_SemanticReplayPlans", create):
        loss = model(torch.randint(41, (1, 24)), targets=torch.randint(41, (1, 24)), loss_chunk_size=7)
        loss.backward(retain_graph=True)
        assert len(refs) == 2 and all(ref().plans for ref in refs)
        with patch.object(LogStructuredKVCache, "_semantic_build_replay_plan",
                          side_effect=AssertionError("retained graph re-parsed a cached flush")):
            loss.backward()
    del loss
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_reentrant_or_unwrapped_checkpoint_is_rejected():
    model = GPT(Config(block_size=16, n_layer=1, n_embd=16, n_head=2, vocab_size=16, padding_multiple=1))
    with pytest.raises(RuntimeError, match="every Block"):
        enable_logkv_checkpoint_replay(model, Block)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT),
        check_fn=lambda m: isinstance(m, Block),
    )
    with pytest.raises(RuntimeError, match="non-reentrant"):
        enable_logkv_checkpoint_replay(model, Block)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA, bf16 and Triton")
@pytest.mark.parametrize("distributed", [False, True], ids=["cuda", "fsdp"])
def test_cuda_checkpoint_replay_with_flash(distributed):
    """Run the fsdp case with torchrun --nproc_per_node=2 -m pytest ... -k fsdp."""
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from litgpt.log_kv_pack import _triton_packer

    if distributed and int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("run under torchrun with at least two CUDA ranks")
    assert _triton_packer() is not None
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    owns_group = distributed and not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(37)
        raw = GPT(Config(block_size=32, n_layer=2, n_embd=512, n_head=4, n_query_groups=2,
                         vocab_size=41, padding_multiple=1, rotary_percentage=1.)).to(device, dtype=torch.bfloat16)
        models = []
        for reuse in (False, True):
            model = deepcopy(raw)
            if distributed:
                model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({Block}), device_id=device,
                             sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, use_orig_params=True)
            apply_activation_checkpointing(model, check_fn=lambda m: isinstance(m, Block))
            if reuse:
                assert enable_logkv_checkpoint_replay(model, Block) == 2
            # Match Fabric: attach cache buffers after FSDP/checkpoint wrapping.
            model.enable_log_kv_training(
                batch_size=1, B=3, recent_size=4, train_block=4, second_order_scale=0.,
                semantic_clusters=True, cluster_k_max=8, semantic_flush_granularity=4,
                semantic_anchor_mode="mid", semantic_pack_backend="triton",
                allocate_second_order=False, device=device, dtype=torch.bfloat16,
            )
            models.append(model)
        torch.manual_seed(91 + (dist.get_rank() if distributed else 0))
        inputs = [torch.randint(41, (1, 24), device=device) for _ in range(2)]
        targets = [torch.randint(41, (1, 24), device=device) for _ in range(2)]
        results = []
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            for model in models:
                logkv_begin_step()
                losses = []
                for i, (x, target) in enumerate(zip(inputs, targets)):
                    with model.no_sync() if distributed and i == 0 else nullcontext():
                        loss = model(x, targets=target, loss_chunk_size=7)
                        losses.append(loss.detach())
                        (loss / 2).backward()
                grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
                results.append((losses, grads, logkv_take_host_stats()))
        before, after = results
        for a, b in zip(before[0] + before[1], after[0] + after[1]):
            if a is None or b is None:
                assert a is b
            else:
                torch.testing.assert_close(a, b, atol=.02, rtol=.02)
        assert before[2]["route_n"] == 2 * after[2]["route_n"] > 0
        assert after[2]["replay_n"] == 2 * before[2]["replay_n"]
        assert any("_scaled_dot_product_flash_attention_backward" in e.key for e in prof.key_averages())
        assert _route_pass.get() is None
    finally:
        if owns_group:
            dist.destroy_process_group()
