"""Check the real launch-script readers and SinkWindow/static-hybrid routing."""

import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("mode", "sink_size", "window_size", "full_attention_interval")


def _eval_args(script, config):
    source = (ROOT / script).read_text()
    if script == "majob.sh":
        start = source.index("read -r LOG_KV_B ")
        stop = source.index('\n)"', start) + len('\n)"')
        read_config = source[start:stop]
        start = source.index('LOG_KV_ARGS="')
        stop = source.index('\nDIAG_ARGS=', start)
        build_args = source[start:stop]
        print_args = "printf '%s\\n' ${EVAL_LOG_KV_ARGS}"
    else:
        start = source.index('CONFIG_EXPORTS=$(python ')
        stop = source.index('eval "${CONFIG_EXPORTS}"', start) + len('eval "${CONFIG_EXPORTS}"')
        read_config = source[start:stop]
        start = source.index('LOG_KV_ARG_LIST=(')
        stop = source.index('\nDIAG_ARGS=', start)
        build_args = source[start:stop]
        print_args = "printf '%s\\n' \"${LOG_KV_ARG_LIST[@]}\""
    shell = '\n'.join(('set -e', 'CONFIG_FILE=$1', read_config, build_args, print_args))
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    result = subprocess.run(["bash", "-c", shell, "bash", str(config)],
                            capture_output=True, text=True, check=True, env=env)
    args = result.stdout.splitlines()
    return {name: args[args.index("--log_kv_sink_window_" + name) + 1] for name in FIELDS}


@pytest.mark.parametrize("script", ["majob.sh", "eval.sh"])
@pytest.mark.parametrize("setting,expected", [
    (None, ("false", "4", "1024", "0")),
    ("null", ("false", "4", "1024", "0")),
    ("hybrid", ("true", "16", "2048", "4")),
])
def test_sink_window_arguments_follow_inheritance(tmp_path, script, setting, expected):
    parent = tmp_path / "base.yaml"
    values = "save_path: /tmp/unused-checkpoint\n"
    if setting == "null":
        values += "".join(f"log_kv_sink_window_{name}: null\n" for name in FIELDS)
    elif setting == "hybrid":
        values += "".join(f"log_kv_sink_window_{name}: {value}\n" for name, value in zip(FIELDS, expected))
    parent.write_text(values)
    config = tmp_path / "run.yaml"
    config.write_text("config: base.yaml\n")
    assert _eval_args(script, config) == dict(zip(FIELDS, expected))


@pytest.mark.parametrize("script", ["majob.sh", "eval.sh"])
@pytest.mark.parametrize("kind", ["train", "eval"])
def test_static_hybrid_experiment_reaches_eval(script, kind):
    config = ROOT / "exp/qwen1.7b-32k" / f"statichybrid_stage1_{kind}.yaml"
    assert _eval_args(script, config) == dict(zip(FIELDS, ("true", "16", "2048", "4")))


def test_training_rejects_hybrid_without_sink_window_mode():
    tree = ast.parse((ROOT / "demo.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_check_sink_window_mode_compatible")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "demo.py", "exec"), scope)
    check = scope[node.name]
    check(True, False, 4)
    check(False, False)
    with pytest.raises(ValueError, match="requires log_kv_sink_window_mode=True"):
        check(False, False, 4)
    with pytest.raises(ValueError, match="must be >= 0"):
        check(True, False, -1)


def test_eval_builds_hybrid_cache_and_reuses_it():
    tree = ast.parse((ROOT / "eval.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LogKVLM")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_set_eval_cache")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "eval.py", "exec"), scope)
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="bf16")]),
                            max_seq_length=32768, set_sink_window_cache=Mock())
    adapter = SimpleNamespace(
        model=model, _device="cpu", _eval_cache_ready=False, _reset_eval_cache=Mock(),
        log_kv_dense_mode=False, log_kv_sink_window_mode=True,
        log_kv_sink_window_sink_size=16, log_kv_sink_window_window_size=2048,
        log_kv_sink_window_full_attention_interval=4,
    )
    scope[node.name](adapter)
    model.set_sink_window_cache.assert_called_once_with(
        batch_size=1, sink_size=16, window_size=2048, full_attention_interval=4, device="cpu", dtype="bf16",
    )
    scope[node.name](adapter)
    adapter._reset_eval_cache.assert_called_once_with()
    assert model.set_sink_window_cache.call_count == 1
