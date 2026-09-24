"""Run with python tests/test_eval_cache_diag.py; no GPU or HF packages required."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def test_cache_failure_is_visible_and_original_behavior_is_preserved():
    source = Path(__file__).resolve().parents[1] / "eval.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_trace_hf_cache_resolution")
    scope = dict(contextlib=contextlib, os=os, json=json, Path=Path,
                 __file__=str(source), _global_rank=lambda: 7)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    result = object()
    failure = PermissionError("cannot enumerate cached configurations")

    class Factory:
        name = "allenai/social_i_qa"
        cache_dir = None

        def get_module(self):
            if self.cache_dir:
                raise failure
            return result

    original = Factory.get_module
    modules = {name: ModuleType(name) for name in ("datasets", "huggingface_hub", "lm_eval")}
    for module in modules.values():
        module.__version__, module.__file__ = "test", "/test/package.py"
    modules["datasets"].load_dataset = lambda *args, **kwargs: result
    modules.update({
        "datasets.load": SimpleNamespace(CachedDatasetModuleFactory=Factory),
        "datasets.config": SimpleNamespace(HF_DATASETS_CACHE="/test/cache", HF_DATASETS_OFFLINE=True),
        "huggingface_hub.constants": SimpleNamespace(HF_HUB_CACHE="/test/hub", HF_HUB_OFFLINE=True),
    })
    output = io.StringIO()
    with patch.dict(sys.modules, modules), contextlib.redirect_stderr(output):
        with scope["_trace_hf_cache_resolution"]():
            assert Factory().get_module() is result
        assert Factory.get_module is original
        try:
            with scope["_trace_hf_cache_resolution"]():
                cached = Factory()
                cached.cache_dir = "/explicit/cache"
                cached.get_module()
        except PermissionError as exc:
            assert exc is failure
        else:
            raise AssertionError("diagnostic swallowed the original exception")
        assert Factory.get_module is original
    events = [json.loads(line.removeprefix("[hf-cache] ")) for line in output.getvalue().splitlines()]
    assert [event["event"] for event in events] == ["runtime", "runtime", "local_cache_failure"]
    assert events[-1]["rank"] == 7
    assert events[-1]["effective_cache_dir"] == "/explicit/cache"
    assert "PermissionError: cannot enumerate cached configurations" in events[-1]["traceback"]


def test_eval_launch_uses_selected_python_for_both_jobs():
    source = (Path(__file__).resolve().parents[1] / "eval.sh").read_text()
    start = source.index('if [ "${BENCHMARKS}" != "none" ] && [ -n "${BENCHMARKS}" ]; then')
    # The trailing barrier cleanup is inert with NODE_RANK=1.
    shell = '''
PYTHON_BIN=selected_python
BENCHMARKS=social_iqa
NIAH_BENCHMARKS=niah_single_1
NODE_RANK=1
SAVE_DIR="/checkpoint with spaces"
selected_python() { printf 'CALL'; printf ' <%s>' "$@"; printf '\\n'; }
torchrun() { echo WRONG_INTERPRETER; return 99; }
'''
    result = subprocess.run(["bash", "-c", shell + source[start:]], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("CALL <-m> <torch.distributed.run>") == 2
    assert result.stdout.count("</checkpoint with spaces>") >= 2
    assert "WRONG_INTERPRETER" not in result.stdout


if __name__ == "__main__":
    test_cache_failure_is_visible_and_original_behavior_is_preserved()
    test_eval_launch_uses_selected_python_for_both_jobs()
    print("HF cache diagnostic check passed")
