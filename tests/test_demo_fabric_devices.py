"""Check actual Fabric topology arguments without importing training dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_fabric_devices_follow_launcher():
    source = ast.parse((Path(__file__).resolve().parents[1] / "demo.py").read_text())
    fabric = next(
        node for node in ast.walk(source)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "L" and node.func.attr == "Fabric"
    )
    topology = {kw.arg: compile(ast.Expression(kw.value), "demo.py", "eval")
                for kw in fabric.keywords if kw.arg in ("devices", "num_nodes")}
    for env, configured, expected in [
        ({}, 1, (1, 1)),
        ({}, 4, (4, 1)),
        ({"LOCAL_WORLD_SIZE": "2", "GROUP_WORLD_SIZE": "1", "WORLD_SIZE": "2"}, 8, (2, 1)),
        ({"LOCAL_WORLD_SIZE": "4", "GROUP_WORLD_SIZE": "1", "WORLD_SIZE": "4"}, 8, (4, 1)),
        ({"LOCAL_WORLD_SIZE": "8", "GROUP_WORLD_SIZE": "1", "WORLD_SIZE": "8"}, 1, (8, 1)),
        ({"LOCAL_WORLD_SIZE": "8", "GROUP_WORLD_SIZE": "2", "WORLD_SIZE": "16"}, 1, (8, 2)),
        ({"LOCAL_WORLD_SIZE": "1", "GROUP_WORLD_SIZE": "1", "WORLD_SIZE": "1"}, 8, (1, 1)),
    ]:
        scope = {"os": SimpleNamespace(environ=env), "num_devices": configured}
        devices, nodes = (eval(topology[key], scope) for key in ("devices", "num_nodes"))
        assert (devices, nodes) == expected
        if "WORLD_SIZE" in env:
            assert devices * nodes == int(env["WORLD_SIZE"])


def _batch_setup():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "demo.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    start = next(index for index, node in enumerate(main.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "launch")
    stop = next(index for index, node in enumerate(main.body)
                if any(isinstance(child, ast.Constant) and child.value == "Tensorboard root:"
                       for child in ast.walk(node)))
    return compile(ast.Module(body=main.body[start + 1:stop], type_ignores=[]), "demo.py", "exec")


@pytest.mark.parametrize("world_size,expected", [(1, 32), (2, 16), (4, 8), (8, 4)])
def test_accumulation_preserves_global_batch(world_size, expected):
    fabric = SimpleNamespace(world_size=world_size, print=Mock())
    scope = dict(fabric=fabric, global_batch_size=32, micro_batch_size=1)
    exec(_batch_setup(), scope)
    assert scope["gradient_accumulation_steps"] == expected
    assert expected * world_size * scope["micro_batch_size"] == scope["global_batch_size"]
    log = fabric.print.call_args.args[0]
    assert f"world_size={world_size}" in log and "micro_batch_size=1" in log
    assert "global_batch_size=32" in log and f"gradient_accumulation_steps={expected}" in log


@pytest.mark.parametrize("global_batch,micro_batch,world_size,error", [
    (33, 1, 8, "must be divisible"),
    (4, 1, 8, "must be divisible"),
    (32, 3, 4, "must be divisible"),
    (0, 1, 1, "positive integers"),
    (-32, 1, 1, "positive integers"),
    (32, 0, 1, "positive integers"),
    (32, -1, 1, "positive integers"),
    (32.0, 1, 1, "positive integers"),
    (32, 1.5, 1, "positive integers"),
    (True, 1, 1, "positive integers"),
])
def test_invalid_batches_fail_before_model_loading(global_batch, micro_batch, world_size, error):
    scope = dict(fabric=SimpleNamespace(world_size=world_size, print=Mock()),
                 global_batch_size=global_batch, micro_batch_size=micro_batch)
    with pytest.raises(ValueError, match=error):
        exec(_batch_setup(), scope)
