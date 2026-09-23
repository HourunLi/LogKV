"""Tests for demo.py's _dump_resolved_config and the locals()-snapshot
mechanism its call site in main() uses.

Follows tests/test_demo_launch.py's own pattern of extracting a single
function's AST out of demo.py and exec-ing it in a minimal scope, rather
than importing demo.py itself -- demo.py pulls in lightning/litdata at
module import time, which this test doesn't need and shouldn't require just
to check a YAML-dumping helper's logic. Not executed in the environment that
wrote this file; run for real before trusting resolved_config.yaml's
contents.
"""

import __future__
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_demo_function(name: str):
    tree = ast.parse((ROOT / "demo.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    scope = {"os": __import__("os"), "yaml": yaml}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "demo.py", "exec", flags=__future__.annotations.compiler_flag),
        scope,
    )
    return scope[name]


def _load_normal_path():
    # _dump_resolved_config calls the module-level _normal_path helper.
    tree = ast.parse((ROOT / "demo.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_normal_path"
    )
    scope = {"os": __import__("os"), "Path": Path}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "demo.py", "exec", flags=__future__.annotations.compiler_flag),
        scope,
    )
    return scope["_normal_path"]


@pytest.fixture
def dump_resolved_config():
    fn = _load_demo_function("_dump_resolved_config")
    fn.__globals__["_normal_path"] = _load_normal_path()
    fn.__globals__["os"] = __import__("os")
    fn.__globals__["yaml"] = yaml
    return fn


def _fake_fabric(global_rank: int):
    calls = SimpleNamespace(barrier_count=0)

    def barrier():
        calls.barrier_count += 1

    return SimpleNamespace(global_rank=global_rank, barrier=barrier), calls


def test_dump_round_trips_through_yaml(tmp_path, dump_resolved_config):
    fabric, calls = _fake_fabric(0)
    resolved = {"arch_name": "Qwen/Qwen3-1.7B-Base", "context_length": 32768, "learning_rate": 5e-5, "log_kv_dense_mode": True}
    dump_resolved_config(fabric, tmp_path / "run1", resolved)

    out_file = tmp_path / "run1" / "resolved_config.yaml"
    assert out_file.is_file()
    with open(out_file, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded == resolved
    assert calls.barrier_count == 1


def test_only_rank_zero_writes(tmp_path, dump_resolved_config):
    fabric, calls = _fake_fabric(1)
    dump_resolved_config(fabric, tmp_path / "run2", {"a": 1})
    assert not (tmp_path / "run2").exists()
    assert calls.barrier_count == 1  # non-zero ranks still barrier, just don't write


def test_parameter_intersection_excludes_non_serializable_locals():
    """Pins the bug a raw locals() call (instead of intersecting with
    inspect.signature(main).parameters) would hit: main()'s scope also holds
    _yaml (a dict), _valid, _banned, and critically _o itself (a function
    object) -- none of those are main() parameters, and yaml.safe_dump would
    crash on the function object if it leaked through.
    """

    def main(arch_name: str = "x", context_length: int = 4096):
        _yaml = {"arch_name": "y"}  # noqa: F841 -- simulates demo.py's real locals
        _valid = {"arch_name", "context_length"}  # noqa: F841

        def _o(name, current):
            return current

        arch_name = _o("arch_name", arch_name)
        context_length = _o("context_length", context_length)
        resolved = {k: v for k, v in locals().items() if k in inspect.signature(main).parameters}
        return resolved

    resolved = main()
    assert resolved == {"arch_name": "x", "context_length": 4096}
    assert "_yaml" not in resolved
    assert "_valid" not in resolved
    assert "_o" not in resolved
    # and it must actually be YAML-serializable
    yaml.safe_dump(resolved)


def test_parameter_intersection_self_updates_when_main_gains_a_parameter():
    """A hand-written field list would need updating every time main() grows
    a new parameter; the intersection approach doesn't.
    """

    def main(arch_name: str = "x", a_brand_new_param: int = 7):
        resolved = {k: v for k, v in locals().items() if k in inspect.signature(main).parameters}
        return resolved

    resolved = main()
    assert resolved == {"arch_name": "x", "a_brand_new_param": 7}
