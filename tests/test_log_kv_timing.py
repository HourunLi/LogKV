"""Timing must separate replay/planning without synchronizing normal steps."""

from unittest.mock import Mock, patch
import ast
import inspect
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from litgpt.log_kv_cache import LogStructuredKVCache, LogKVStreamTrainingAttention
import litgpt.log_kv_timing as timing


def test_profile_step_cli_list_is_not_parsed_as_a_scalar_int():
    source = ast.parse((Path(__file__).resolve().parents[1] / "utils.py").read_text())
    parse = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_coerce_cli_value")
    parse.returns = None
    scope = {"inspect": inspect, "yaml": yaml}
    exec(compile(ast.Module(body=[parse], type_ignores=[]), "utils.py", "exec"), scope)
    for annotation, value, expected in [
        ("list[int] | None", "[3,4,5]", [3, 4, 5]),
        ("list[int] | None", "null", None),
        ("list[bool]", "[true,false]", [True, False]),
        ("int", "3", 3),
        ("float", "0.5", .5),
    ]:
        param = inspect.Parameter("steps", inspect.Parameter.KEYWORD_ONLY, annotation=annotation, default=None)
        assert scope["_coerce_cli_value"](value, param) == expected


def test_host_counts_span_accumulation_and_reset_without_cuda_calls():
    cache = LogStructuredKVCache(
        (1, 2, 32, 8), (1, 2, 32, 8), B=3, recent_size=4,
        semantic_clusters=True, cluster_k_max=8, semantic_anchor_mode="mid",
        semantic_flush_granularity=4, allocate_second_order=False,
        cos_cache=torch.ones(32, 8), sin_cache=torch.zeros(32, 8), rope_n_elem=8,
    )
    with patch.object(torch.cuda, "Event", side_effect=AssertionError("normal step created a CUDA event")), \
            patch.object(torch.cuda, "synchronize", side_effect=AssertionError("normal step synchronized")):
        timing.logkv_begin_step()
        for _ in range(2):
            q = torch.randn(1, 4, 32, 8, requires_grad=True)
            k = torch.randn(1, 2, 32, 8, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            with timing.logkv_timed("forward"):
                y = LogKVStreamTrainingAttention.apply(q, k, v, cache, 8 ** -.5, 4, 0., k)
            with timing.logkv_timed("backward"):
                y.sum().backward()
        result = timing.logkv_take_host_stats()
        assert not result["cuda_profiled"]
        assert result["forward_n"] == result["backward_n"] == 2
        assert result["route_n"] == result["replay_n"] > 0
        assert result["plan_n"] == 16  # forward plans only; replay reuses them
        assert result["attn_fwd_n"] == 32  # forward and backward reconstruction
        assert result["attn_bwd_n"] == 16
        assert result["attn_s"] == result["plan_s"] + result["pack_s"]
        assert all(result[stage + "_s"] > 0 for stage in timing.CACHE_STAGES)
        cleared = timing.logkv_take_host_stats()
        assert all(cleared[stage + "_n"] == 0 for stage in timing.CACHE_STAGES + timing.STEP_STAGES)


def test_cuda_events_are_consumed_once_at_step_end_and_disabled_afterwards():
    # Exercise lifecycle on CPU too; real CUDA event execution is checked below.
    events = []

    def event(**kwargs):
        result = Mock()
        result.elapsed_time.return_value = 12.5
        events.append(result)
        return result

    with patch.object(torch.cuda, "Event", side_effect=event), \
            patch.object(torch.cuda, "current_stream", return_value="stream"), \
            patch.object(torch.cuda, "synchronize") as sync:
        timing.logkv_begin_step(profile_cuda=True, device="cuda:0")
        for _ in range(2):
            with timing.logkv_timed("forward"):
                with timing.logkv_timed("plan"):
                    pass
        sync.assert_not_called()
        with pytest.raises(RuntimeError, match="consume"):
            timing.logkv_begin_step()
        result = timing.logkv_take_host_stats()
        sync.assert_called_once_with(torch.device("cuda:0"))
        assert len(events) == 8
        assert result["forward_cuda_s"] == result["plan_cuda_s"] == .025
        assert result["replay_cuda_s"] == 0
        for ev in events:
            ev.record.assert_called_once_with("stream")
        with timing.logkv_timed("pack"):
            pass
        assert len(events) == 8
        assert not timing.logkv_take_host_stats()["cuda_profiled"]
        assert sync.call_count == 1


def test_training_loop_profiles_global_steps_and_excludes_checkpoint_time():
    # Execute the actual training loop without loading datasets/model checkpoints.
    source = ast.parse((Path(__file__).resolve().parents[1] / "demo.py").read_text())
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(node for node in main.body if isinstance(node, ast.While))
    clock, selections = [0.0], []

    def advance(seconds):
        clock[0] += seconds

    def begin(**kwargs):
        selections.append(kwargs["profile_cuda"])
        timing.logkv_begin_step()  # CUDA lifecycle is covered separately.

    def forward(*args, **kwargs):
        advance(1)
        return torch.tensor(1., requires_grad=True)

    def save(path, **kwargs):
        advance(100)
        return path

    fabric = Mock()
    fabric.device, fabric.global_rank = torch.device("cpu"), 0
    fabric.no_backward_sync.side_effect = lambda *args, **kwargs: nullcontext()
    fabric.backward.side_effect = lambda loss: advance(2)
    model = Mock(side_effect=forward)
    optimizer = Mock()
    optimizer.param_groups = [{"lr": 0.}]
    optimizer.step.side_effect = lambda: advance(3)
    step_stats = Mock()
    step_stats.averages.return_value = {}
    scope = dict(
        global_step=2, max_steps=4, training_finished=False, step_active=False,
        profile_steps={3}, micro_batch_idx=0, gradient_accumulation_steps=2,
        loader_iter=iter([torch.ones(1, 5)] * 4), context_length=4, data_epoch=1, num_epochs=1,
        fabric=fabric, model=model, optimizer=optimizer, step_stats=step_stats,
        log_kv_second_order_warmup_steps=0, log_kv_second_order_scale=0., entropy_chunk_size=4,
        get_log_kv_second_order_scale=lambda *args: 0., get_lr=lambda *args: .1,
        total_steps=4, warmup_steps=0, learning_rate=.1, min_lr=0.,
        time=SimpleNamespace(perf_counter=lambda: clock[0]), datetime=datetime,
        logkv_begin_step=begin, logkv_timed=timing.logkv_timed, logkv_take_host_stats=timing.logkv_take_host_stats,
        CACHE_STAGES=timing.CACHE_STAGES, STEP_STAGES=timing.STEP_STAGES,
        save_ckpt=True, save_interval=3, save_path="unused", _unique_save_dir=lambda path: path,
        _save_training_checkpoint=save,
    )
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "demo.py", "exec"), scope)
    assert selections == [True, False]  # resumed global steps 3 and 4
    assert scope["global_step"] == 4 and not scope["step_active"]
    assert clock[0] == 218  # two 9-second steps plus two 100-second checkpoint writes
    for call in fabric.log_dict.call_args_list:
        metrics = call.args[0]
        assert metrics["train/step_host_s"] == 9
        assert metrics["train/step_forward_calls"] == metrics["train/step_backward_calls"] == 2
        assert metrics["train/step_optimizer_calls"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_event_spans_cover_forward_and_backward():
    timing.logkv_begin_step(profile_cuda=True, device="cuda")
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    with timing.logkv_timed("attn_fwd"):
        y = x @ x
    with timing.logkv_timed("attn_bwd"):
        y.sum().backward()
    result = timing.logkv_take_host_stats()
    assert result["cuda_profiled"]
    assert result["attn_fwd_cuda_s"] > 0
    assert result["attn_bwd_cuda_s"] > 0
