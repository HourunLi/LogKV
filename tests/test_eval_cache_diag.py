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
                 __file__=str(source), _global_rank=lambda: 7, _eval_verbose=lambda: False)
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
    assert [event["event"] for event in events] == ["local_cache_failure"]
    assert events[-1]["rank"] == 7
    assert events[-1]["effective_cache_dir"] == "/explicit/cache"
    assert "PermissionError: cannot enumerate cached configurations" in events[-1]["traceback"]


def test_path_walk_names_the_component_that_blocks_traversal():
    import tempfile
    source = Path(__file__).resolve().parents[1] / "eval.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_walk_cache_path")
    scope = dict(os=os, Path=Path)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    walk = scope["_walk_cache_path"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        snapshot = Path("default/0.1.0/674d85e4/dataset_info.json")
        (root / "social_i_qa" / snapshot).parent.mkdir(parents=True)
        (root / "social_i_qa" / snapshot).write_text("{}")
        (root / "healthy").symlink_to("social_i_qa")
        (root / "flattened").write_text("social_i_qa")
        (root / "to_file").symlink_to("flattened")
        (root / "dangling").symlink_to("missing_dir")

        steps = walk(root / "healthy" / snapshot)
        assert not any(step.get("blocks_traversal") for step in steps)
        assert steps[0]["path"] == "/" and "fstype" in steps[0]["mount"]
        link = next(step for step in steps if step.get("path") == str(root / "healthy"))
        assert (link["type"], link["link"], link["target_type"]) == ("lnk", "social_i_qa", "dir")
        assert steps[-1]["entries"] == ["dataset_info.json"]

        for name, expected in (("flattened", dict(type="reg", head="social_i_qa")),
                               ("to_file", dict(type="lnk", target_type="reg", head="social_i_qa")),
                               ("dangling", dict(type="lnk", link="missing_dir"))):
            steps = walk(root / name / snapshot)
            blocking = [step for step in steps if step.get("blocks_traversal")]
            assert [step["path"] for step in blocking] == [str(root / name)], (name, steps)
            assert expected.items() <= blocking[0].items(), (name, blocking)
            assert steps[-1]["listing_of"] == str(root)


def test_unreadable_dataset_info_is_named_with_its_context():
    import tempfile
    source = Path(__file__).resolve().parents[1] / "eval.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_unreadable_dataset_infos")
    scope = dict(json=json, Path=Path, time=__import__("time"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    find = scope["_unreadable_dataset_infos"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = root / "allenai___social_i_qa" / "default" / "0.1.0" / "aaaa" / "dataset_info.json"
        bad = root / "allenai___social_i_qa" / "default" / "0.1.0" / "bbbb" / "dataset_info.json"
        other = root / "other___dataset" / "default" / "0.1.0" / "cccc" / "dataset_info.json"
        for info in (good, bad, other):
            info.parent.mkdir(parents=True)
        good.write_text('{"config_name": "default"}')
        # A trailing comma; Python 3.10 reports "Expecting property name enclosed in double quotes".
        bad.write_text('{\n    "config_name": "default",\n}')
        other.write_text("not json")
        found = find(root, "allenai/social_i_qa")
        assert [item["path"] for item in found] == [str(bad)]
        # Python 3.10 points at the brace, 3.13 at the comma; the context shows both.
        assert found[0]["context"][-2:] == ['    "config_name": "default",', "}"]
        # Size and time say whether a copy was cut short, and when it was written.
        assert found[0]["size"] == bad.stat().st_size and found[0]["modified"][:4].isdigit()
        assert find(root / "missing", "allenai/social_i_qa")[0]["error"].startswith("FileNotFoundError")


def test_repair_hf_cache_runs_on_node_zero_before_its_barrier():
    import tempfile
    root = Path(__file__).resolve().parents[1]
    for script in ("majob.sh", "eval.sh"):
        source = (root / script).read_text()
        start = source.index("HF_DATASETS_ROOT=${HF_DATASETS_ROOT:-")
        block = source[start:source.index("\nfi\n", start) + 4]
        assert source.index(block) < source.index('touch "${BARRIER_DIR}/node_${NODE_RANK}"'), script
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp, "calls")
            stub = Path(tmp, "python")
            stub.write_text(f'#!/bin/sh\necho "$@" >> {log}\nexit "${{STUB_STATUS:-0}}"\n')
            stub.chmod(0o755)
            run = lambda **env: subprocess.run(["bash", "-c", block + "\necho reached"], capture_output=True,
                                               text=True, env=dict(os.environ, PYTHON_BIN=str(stub), **env))
            assert "reached" in run(NODE_RANK="0").stdout and not log.exists()  # Off by default.
            assert "reached" in run(NODE_RANK="1", REPAIR_HF_CACHE="1").stdout and not log.exists()
            assert "reached" in run(NODE_RANK="0", REPAIR_HF_CACHE="1", HF_DATASETS_ROOT="/cache").stdout
            assert log.read_text() == "unused/repair_dataset_info.py --scan /cache --write --check\n"
            failed = run(NODE_RANK="0", REPAIR_HF_CACHE="1", STUB_STATUS="1")
            assert failed.returncode == 1 and "reached" not in failed.stdout, script


def test_both_launchers_default_to_every_niah_task():
    root = Path(__file__).resolve().parents[1]
    variants = sorted(p.stem for p in (root / "eval_tasks" / "niah_cue").glob("niah_*.yaml"))
    ruler = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
             "niah_multikey_3", "niah_multivalue", "niah_multiquery"]
    try:  # lm-eval's own RULER needle tasks, when it is installed here.
        import lm_eval.tasks
        found = sorted(p.stem for p in (Path(lm_eval.tasks.__file__).parent / "ruler").glob("niah_*.yaml"))
        assert found == sorted(ruler)
    except ImportError:
        pass
    assert len(variants) == 9
    defaults = {}
    for script in ("majob.sh", "eval.sh"):
        source = (root / script).read_text()
        lines = [line for line in source.splitlines() if line.startswith("DEFAULT_NIAH_BENCHMARKS=")]
        result = subprocess.run(["bash", "-c", "\n".join(lines) + '\necho "$DEFAULT_NIAH_BENCHMARKS"'],
                                capture_output=True, text=True, check=True)
        defaults[script] = result.stdout.strip().split(",")
        assert "NIAH_BENCHMARKS:-${DEFAULT_NIAH_BENCHMARKS}" in source or \
            "NIAH_BENCHMARKS=${3:-${DEFAULT_NIAH_BENCHMARKS}}" in source, script
    assert defaults["majob.sh"] == defaults["eval.sh"]
    assert sorted(defaults["eval.sh"]) == sorted(ruler + variants) and len(set(defaults["eval.sh"])) == 17


def test_task_selection_skips_blank_or_none_stages_and_supports_niah_only():
    root = Path(__file__).resolve().parents[1]

    def select(script, env=(), args=()):
        source = (root / script).read_text()
        block = source[source.index('DEFAULT_BENCHMARKS="boolq'):source.index("\nMETA=")]
        result = subprocess.run(["bash", "-c", block + '\necho "MAIN=$BENCHMARKS"; echo "NIAH=$NIAH_BENCHMARKS"',
                                 script, *args], capture_output=True, text=True,
                                env={"PATH": os.environ["PATH"], "NODE_RANK": "0", **dict(env)})
        lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if line[:5] in ("MAIN=", "NIAH="))
        return result.returncode, lines.get("MAIN"), lines.get("NIAH"), result.stdout

    for script, as_env in (("majob.sh", True), ("eval.sh", False)):
        def run(main=None, niah=None, **env):
            if as_env:
                env.update({k: v for k, v in (("BENCHMARKS", main), ("NIAH_BENCHMARKS", niah)) if v is not None})
                return select(script, env)
            return select(script, env, ["config.yaml", main or "", niah or ""])  # eval.sh <yaml> <main> <niah>

        code, main, niah, _ = run()
        assert code == 0 and main.startswith("boolq,piqa") and niah.startswith("niah_single_1,"), script
        assert run(main="none")[1] == "none" and run(main="none")[2] == niah
        assert run(NIAH_ONLY="1")[1:3] == ("none", niah), script
        assert run(main=" ", niah="NONE")[1:3] == ("none", "none"), script
        assert run(main=" piqa, ,boolq ,", niah="niah_single_2 ,niah_single_3")[1:3] == (
            "piqa,boolq", "niah_single_2,niah_single_3"), script
        code, _, _, out = run(main="--limit")
        assert code == 1 and "DIAG_ARGS" in out, script
    for script in ("majob.sh", "eval.sh"):
        source = (root / script).read_text()
        assert '--benchmark "${BENCHMARKS}"' in source and '--benchmark "${NIAH_BENCHMARKS}"' in source


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
    test_path_walk_names_the_component_that_blocks_traversal()
    test_unreadable_dataset_info_is_named_with_its_context()
    test_repair_hf_cache_runs_on_node_zero_before_its_barrier()
    test_both_launchers_default_to_every_niah_task()
    test_task_selection_skips_blank_or_none_stages_and_supports_niah_only()
    test_eval_launch_uses_selected_python_for_both_jobs()
    print("HF cache diagnostic check passed")
