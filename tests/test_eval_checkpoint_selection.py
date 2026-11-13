"""Checkpoint resolution respects an explicitly selected CPT output directory, without GPUs."""
import ast
import __future__
import os
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helpers(**extra):
    tree = ast.parse((ROOT / "eval.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"_resolve_checkpoint_dir", "_load_lit_model_checkpoint", "_checkpoint_file_for_loading"}]
    scope = dict(Path=Path, os=os, time=time, _is_main=lambda: True, _dist_ready=lambda: False,
                 _rank_label=lambda: "CPU rank 0",
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


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("suffix", [b"", b"\n", b"\x00\r\n"])
def test_flattened_checkpoint_link_loads_the_referenced_binary_without_writes(tmp_path, monkeypatch, relative, suffix):
    torch = pytest.importorskip("torch")
    root = tmp_path / "cpt"
    root.mkdir()
    binary = tmp_path / "weights with spaces.pth"
    torch.save({"global_step": 2000, "model": {"weight": torch.tensor([2000])}}, binary)
    pointer = root / "lit_model.pth"
    target = os.path.relpath(binary, root) if relative else str(binary)
    pointer.write_bytes(target.encode() + suffix)
    original = pointer.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint loading must not create links or mutate the shared mount")

    monkeypatch.setattr(os, "symlink", forbidden)
    monkeypatch.setattr(os, "rename", forbidden)
    monkeypatch.setattr(os, "replace", forbidden)
    scope = helpers(torch=torch)
    selected = scope["_resolve_checkpoint_dir"](str(root))
    loaded = scope["_load_lit_model_checkpoint"](selected, "cpu")
    assert selected == str(root)
    assert loaded["global_step"] == 2000 and loaded["model"]["weight"].item() == 2000
    assert pointer.read_bytes() == original and not pointer.is_symlink()


def test_checkpoint_path_references_can_form_a_chain(tmp_path):
    torch = pytest.importorskip("torch")
    root = tmp_path / "cpt"
    root.mkdir()
    binary = tmp_path / "blob-without-extension"
    torch.save({"global_step": 2000}, binary)
    (root / "alias.pth").write_text("../blob-without-extension\n")
    (root / "lit_model.pth").write_text("./alias.pth\x00")
    assert helpers(torch=torch)["_load_lit_model_checkpoint"](str(root), "cpu")["global_step"] == 2000


@pytest.mark.parametrize("zip_format", [False, True])
def test_binary_checkpoints_keep_working_in_both_torch_formats(tmp_path, zip_format):
    torch = pytest.importorskip("torch")
    torch.save({"global_step": 2000}, tmp_path / "lit_model.pth", _use_new_zipfile_serialization=zip_format)
    assert helpers(torch=torch)["_load_lit_model_checkpoint"](str(tmp_path), "cpu")["global_step"] == 2000


def test_missing_pointer_target_never_falls_back_to_older_weights(tmp_path):
    torch = pytest.importorskip("torch")
    step = tmp_path / "step_1600"
    step.mkdir()
    torch.save({"global_step": 1600}, step / "lit_model.pth")
    missing = tmp_path / "missing-final.pth"
    (tmp_path / "lit_model.pth").write_text(str(missing))
    scope = helpers(torch=torch)
    selected = scope["_resolve_checkpoint_dir"](str(tmp_path))
    assert selected == str(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        scope["_load_lit_model_checkpoint"](selected, "cpu")
    assert str(missing) in str(exc.value) and str(tmp_path / "lit_model.pth") in str(exc.value)
    assert "older" in str(exc.value)


@pytest.mark.parametrize("self_reference", [False, True])
def test_checkpoint_reference_cycles_fail_before_unpickling(tmp_path, self_reference):
    (tmp_path / "lit_model.pth").write_text("./lit_model.pth" if self_reference else "./alias.pth")
    if not self_reference:
        (tmp_path / "alias.pth").write_text("./lit_model.pth")
    load = Mock(side_effect=AssertionError("a text reference must never reach torch.load"))
    scope = helpers(torch=type("Torch", (), {"load": load}))
    with pytest.raises(ValueError, match="cycle"):
        scope["_load_lit_model_checkpoint"](str(tmp_path), "cpu")
    load.assert_not_called()


@pytest.mark.parametrize("content,problem", [
    (b"", "empty"),
    (b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n", "Git LFS"),
])
def test_checkpoint_placeholders_are_reported_before_unpickling(tmp_path, content, problem):
    (tmp_path / "lit_model.pth").write_bytes(content)
    scope = helpers(torch=type("Torch", (), {"load": Mock(side_effect=AssertionError("not binary"))}))
    with pytest.raises(ValueError, match=problem):
        scope["_load_lit_model_checkpoint"](str(tmp_path), "cpu")


def test_other_rank_waits_for_the_referenced_file_to_be_visible(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    binary = tmp_path / "actual.pth"
    torch.save({"global_step": 2000}, binary)
    (tmp_path / "lit_model.pth").write_text(str(binary))
    exists = Path.exists
    probes, waits, clock = [], [], [0.]

    def visible(path):
        if path == binary:
            probes.append(path)
            return len(probes) >= 3
        return exists(path)

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(Path, "exists", visible)
    scope = helpers(torch=torch)
    scope.update(_is_main=lambda: False, _dist_ready=lambda: True,
                 time=type("Clock", (), {"monotonic": staticmethod(lambda: clock[0]), "sleep": staticmethod(sleep)}))
    assert scope["_load_lit_model_checkpoint"](str(tmp_path), "cpu", wait_s=10)["global_step"] == 2000
    assert waits == [2., 2.]


def test_invalid_binary_error_includes_the_actual_path_and_header(tmp_path):
    torch = pytest.importorskip("torch")
    (tmp_path / "lit_model.pth").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError) as exc:
        helpers(torch=torch)["_load_lit_model_checkpoint"](str(tmp_path), "cpu")
    assert str(tmp_path / "lit_model.pth") in str(exc.value)
    assert "not a checkpoint" in str(exc.value) and "host" in str(exc.value)
