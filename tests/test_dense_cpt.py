"""Check training mode selection without loading checkpoints or launching GPUs."""
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _main_and_defaults():
    tree = ast.parse((ROOT / "demo.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    defaults = {arg.arg: ast.literal_eval(value)
                for arg, value in zip(main.args.args[-len(main.args.defaults):], main.args.defaults)}
    return main, defaults


def _calls(node, name):
    return any(isinstance(child, ast.Call) and
               ((isinstance(child.func, ast.Name) and child.func.id == name) or
                (isinstance(child.func, ast.Attribute) and child.func.attr == name))
               for child in ast.walk(node))


def _execute(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "demo.py", "exec"), scope)


@pytest.mark.parametrize("dense", [False, True])
def test_training_attention_mode(dense):
    main, scope = _main_and_defaults()
    assert scope["log_kv_dense_mode"] is False
    setup = next(node for node in main.body
                 if isinstance(node, ast.If) and _calls(node, "enable_log_kv_training"))
    model = Mock()
    model.parameters.return_value = iter([SimpleNamespace(dtype="bf16")])
    model.transformer = SimpleNamespace(h=[SimpleNamespace(attn=SimpleNamespace(
        kv_cache=SimpleNamespace(B=128, L_alloc=8)))])
    load_scales = Mock(side_effect=AssertionError("dense training must not load semantic scales"))
    scope.update(log_kv_dense_mode=dense, model=model, fabric=SimpleNamespace(device="cpu", print=Mock()),
                 config_obj=SimpleNamespace(n_layer=1, n_query_groups=1), global_step=0,
                 initial_second_order_scale=0., semantic_s_h=None, math=math,
                 get_log_kv_second_order_scale=Mock(return_value=0.), load_log_kv_semantic_s_h=load_scales,
                 log_kv_semantic_clusters=dense, log_kv_alpha_exact_tokens=256 if dense else 0,
                 log_kv_semantic_s_h_path="/missing/semantic-scales.pt")
    _execute([setup], scope)
    assert model.disable_log_kv_training.call_count == int(dense)
    assert model.enable_log_kv_training.call_count == int(not dense)
    model.set_alpha_span_boundary_ids.assert_not_called()
    model.set_kv_cache.assert_not_called()
    load_scales.assert_not_called()


@pytest.mark.parametrize("dense", [False, True])
def test_dense_skips_routing_replay(dense):
    main, scope = _main_and_defaults()
    replay = next(node for node in main.body
                  if isinstance(node, ast.If) and _calls(node, "enable_logkv_checkpoint_replay"))
    enable_replay = Mock(return_value=2)
    scope.update(log_kv_dense_mode=dense, use_fsdp=True, activation_checkpointing=True,
                 log_kv_semantic_clusters=True, log_kv_cluster_k_max=12,
                 model=Mock(), Block=object, fabric=SimpleNamespace(print=Mock()),
                 enable_logkv_checkpoint_replay=enable_replay)
    _execute([replay], scope)
    assert enable_replay.call_count == int(not dense)


@pytest.mark.parametrize("dense", [False, True])
def test_training_eval_keeps_attention_mode(monkeypatch, dense):
    main, scope = _main_and_defaults()
    helper = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "_run_eval")
    evaluate = Mock()
    monkeypatch.setitem(sys.modules, "eval", SimpleNamespace(main=evaluate))
    scope["log_kv_dense_mode"] = dense
    _execute([helper], scope)
    scope["_run_eval"]("/checkpoints/cpt", "niah_multikey_1")
    assert evaluate.call_args.kwargs["log_kv_dense_mode"] is dense
    assert evaluate.call_args.kwargs["checkpoint_dir"] == "/checkpoints/cpt"
