"""Check actual Fabric topology arguments without importing training dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace


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
        ({"LOCAL_WORLD_SIZE": "8", "GROUP_WORLD_SIZE": "1", "WORLD_SIZE": "8"}, 1, (8, 1)),
        ({"LOCAL_WORLD_SIZE": "8", "GROUP_WORLD_SIZE": "2", "WORLD_SIZE": "16"}, 1, (8, 2)),
        ({"LOCAL_WORLD_SIZE": "1", "GROUP_WORLD_SIZE": "1", "WORLD_SIZE": "1"}, 8, (1, 1)),
    ]:
        scope = {"os": SimpleNamespace(environ=env), "num_devices": configured}
        devices, nodes = (eval(topology[key], scope) for key in ("devices", "num_nodes"))
        assert (devices, nodes) == expected
        if "WORLD_SIZE" in env:
            assert devices * nodes == int(env["WORLD_SIZE"])
