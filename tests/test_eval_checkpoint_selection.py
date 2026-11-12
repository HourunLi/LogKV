"""Checkpoint resolution respects an explicitly selected CPT output directory, without GPUs."""
import ast
import __future__
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helpers(**extra):
    tree = ast.parse((ROOT / "eval.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"_resolve_checkpoint_dir", "_load_lit_model_checkpoint"}]
    scope = dict(Path=Path, os=os, _is_main=lambda: True, _dist_ready=lambda: False,
                 _broadcast_obj=lambda value, group=None: value, **extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "eval.py", "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return scope


def test_final_root_weights_win_over_older_step_saves(tmp_path):
    torch = pytest.importorskip("torch")
    root = tmp_path / "cpt"
    step = root / "step_1600"
    step.mkdir(parents=True)
    torch.save({"global_step": 1600, "model": {"weight": torch.tensor([1600])}}, step / "lit_model.pth")
    torch.save({"global_step": 2000, "model": {"weight": torch.tensor([2000])}}, root / "lit_model.pth")
    scope = helpers(torch=torch)
    selected = scope["_resolve_checkpoint_dir"](str(root))
    assert selected == str(root)
    loaded = scope["_load_lit_model_checkpoint"](selected, "cpu")
    assert loaded["global_step"] == 2000 and loaded["model"]["weight"].item() == 2000
    # An explicitly selected intermediate checkpoint remains selectable.
    assert scope["_resolve_checkpoint_dir"](str(step)) == str(step)


def test_step_fallback_is_used_only_when_the_root_has_no_weights(tmp_path):
    for number in (400, 1200, 1600):
        directory = tmp_path / f"step_{number}"
        directory.mkdir()
        (directory / "lit_model.pth").write_bytes(b"weights")
    assert helpers()["_resolve_checkpoint_dir"](str(tmp_path)) == str(tmp_path / "step_1600")


def test_other_ranks_use_the_broadcast_without_inspecting_the_filesystem():
    scope = helpers()
    scope["_is_main"] = lambda: False
    scope["Path"] = Mock(side_effect=AssertionError("only rank 0 may inspect checkpoints"))
    group = object()
    broadcast = Mock(return_value="/chosen/cpt")
    scope["_broadcast_obj"] = broadcast
    assert scope["_resolve_checkpoint_dir"]("/input/cpt", group) == "/chosen/cpt"
    broadcast.assert_called_once_with(None, group=group)
