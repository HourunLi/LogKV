"""Required LongBench v2 passes and task loading on the pinned lm-eval version, without GPUs or downloads."""
import ast
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import re
import shutil
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_module(monkeypatch):
    name = "litgpt.longbench2_patch"
    spec = importlib.util.spec_from_file_location(name, ROOT / "litgpt/longbench2_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def eval_helpers():
    tree = ast.parse((ROOT / "eval.py").read_text())
    names = {"_task_manager", "_check_task_lengths", "_check_task_data", "_skip_optional_tasks",
             "_longbench2_files_problem"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
             or isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_LENGTH_NAMED_TASK"]
    scope = dict(inspect=inspect, os=os, Path=Path, re=re, EVAL_TASKS_DIR=ROOT / "eval_tasks", _is_main=lambda: True)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "eval.py", "exec"), scope)
    return scope


def test_longbench2_is_required_even_with_a_stale_optional_list(data_module, monkeypatch):
    monkeypatch.setenv("EVAL_OPTIONAL_TASKS", "longbench2_64k")
    monkeypatch.setattr(data_module, "snapshot", lambda: (_ for _ in ()).throw(FileNotFoundError("empty cache")))
    helpers = eval_helpers()
    tasks = ["piqa", "longbench2_64k"]
    assert helpers["_skip_optional_tasks"](tasks, 66560) == tasks
    with pytest.raises(FileNotFoundError, match="--longbench2 --no-babilong"):
        helpers["_check_task_data"](tasks)
    with pytest.raises(ValueError, match="longbench2_64k needs a model window"):
        helpers["_check_task_lengths"](tasks, 32768)


def test_missing_and_damaged_subsets_fail_but_complete_and_flattened_cache_loads(data_module, monkeypatch, tmp_path):
    files = [f"{name}.parquet" for name in data_module.subsets()]
    assert len(files) == 20
    for name in files:
        (tmp_path / name).write_bytes(b"PAR1" + b"x" * 16 + b"PAR1")
    monkeypatch.setattr(data_module, "snapshot", lambda: (str(tmp_path), files))
    assert data_module.data_problem() is None
    monkeypatch.setattr(data_module, "snapshot", lambda: (str(tmp_path), files[:-1]))
    assert "missing subsets" in data_module.data_problem()
    monkeypatch.setattr(data_module, "snapshot", lambda: (str(tmp_path), files))
    (tmp_path / files[0]).write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    assert "not whole parquet" in data_module.data_problem()
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "whole").write_bytes(b"PAR1" + b"y" * 16 + b"PAR1")
    (tmp_path / files[0]).write_text("./blobs/whole")
    assert data_module.data_problem() is None


def test_every_length_loads_all_twenty_tasks_on_lm_eval(monkeypatch, data_module):
    pytest.importorskip("lm_eval")
    import datasets
    from lm_eval.evaluator_utils import get_subtask_list
    from lm_eval.tasks import get_task_dict

    seen = set()

    def fake(path=None, name=None, **kwargs):
        assert path == data_module.REPO
        seen.add(name)
        return datasets.DatasetDict({"train": datasets.Dataset.from_dict(
            {"context": ["context"], "question": ["question"], "choices": [["a", "b", "c", "d"]], "answer": [0]})})

    monkeypatch.setattr(datasets, "load_dataset", fake)
    manager = eval_helpers()["_task_manager"]({"pretrained": "unused"})
    expected = {yaml.safe_load(path.read_text())["task"] for path in data_module.TASKS_DIR.glob("*.yaml")
                if "dataset_name" in yaml.safe_load(path.read_text())}
    assert len(expected) == 20
    for length in ("32k", "64k", "128k", "256k"):
        group = f"longbench2_{length}"
        loaded = get_task_dict([group], manager)
        assert set(get_subtask_list(loaded)[group]) == expected
        assert seen == set(data_module.subsets())
    code = get_task_dict(["longbench2_code"], manager)["longbench2_code"]
    doc = code.dataset["train"][0]
    assert code.doc_to_choice(doc) == ["A", "B", "C", "D"]
    assert "<text>\ncontext\n</text>" in code.doc_to_text(doc)
    assert code.get_config("output_type") == "multiple_choice"


def test_download_discovers_all_subsets_without_installed_task_definitions(data_module):
    source = ast.parse((ROOT / "unused/fetch_long_eval_data.py").read_text())
    discover = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == "longbench2_subsets")
    scope = {}
    exec(compile(ast.Module(body=[discover], type_ignores=[]), "fetch_long_eval_data.py", "exec"), scope)
    assert scope["longbench2_subsets"]() == data_module.subsets()
    assert len(scope["longbench2_subsets"]()) == 20


def test_plain_copy_survives_the_hub_blobs_disappearing(data_module, monkeypatch, tmp_path):
    datasets = pytest.importorskip("datasets")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    monkeypatch.setenv("HF_HOME", str(tmp_path / "shared"))
    monkeypatch.delenv("LONGBENCH2_DIR", raising=False)
    hub = tmp_path / "hub"
    root = hub / "snapshots" / "revision"
    blobs = hub / "blobs"
    blobs.mkdir(parents=True)
    names = data_module.subsets()
    files = []
    for index, name in enumerate(names):
        count = 26 if index < 3 else 25  # 503 synthetic rows across the 20 subsets.
        blob = blobs / name
        pq.write_table(pa.table({"context": ["context"] * count, "question": ["question"] * count,
                                 "choices": [["a", "b", "c", "d"]] * count, "answer": [0] * count}), blob)
        relative = f"data/{name}/train.parquet"
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_text(os.path.relpath(blob, path.parent))  # How copied hub links appear on the training mount.
        files.append(relative)

    source = ast.parse((ROOT / "unused/fetch_long_eval_data.py").read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)
             and node.name in {"is_parquet", "materialize_longbench2"}]
    scope = dict(Path=Path, hashlib=hashlib, shutil=shutil, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "fetch_long_eval_data.py", "exec"), scope)
    lines = []
    plain = scope["materialize_longbench2"](str(root), files, lines)
    assert "20 subsets, 503 questions; no cache links" in lines[-1]
    assert all((plain / name).is_file() and not (plain / name).is_symlink() for name in files)
    # A partial copy and a flattened cache link are repaired as ordinary bytes, without renaming on the mount.
    (plain / files[0]).write_bytes(b"PAR1 cut short")
    (plain / files[1]).unlink()
    (plain / files[1]).write_text(str(root / files[1]))
    scope["materialize_longbench2"](str(root), files, [])
    assert not (plain / files[1]).is_symlink()
    for blob in blobs.iterdir():
        blob.unlink()

    def no_snapshot(*args, **kwargs):
        raise AssertionError("evaluation must not depend on the broken hub snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_snapshot)
    assert data_module.snapshot() == (str(plain), sorted(files))
    assert data_module.data_problem() is None
    real_load = datasets.load_dataset

    def only_plain(path=None, name=None, **kwargs):
        assert path == "parquet", "an existing datasets/hub cache must not override the verified plain files"
        return real_load(path, name=name, cache_dir=str(tmp_path / "datasets"), **kwargs)

    monkeypatch.setattr(datasets, "load_dataset", only_plain)
    data_module.apply_patch()
    rows = datasets.load_dataset(data_module.REPO, name=names[0], split="train")
    assert len(rows) == 26 and rows[0]["context"] == "context"


def test_explicit_plain_directory_and_bad_file_diagnostics(data_module, monkeypatch, tmp_path):
    monkeypatch.setenv("LONGBENCH2_DIR", str(tmp_path / "plain"))
    assert data_module.snapshot() == (str(tmp_path / "plain"), [])
    assert "missing subsets" in data_module.data_problem()
    files = [f"{name}.parquet" for name in data_module.subsets()]
    (tmp_path / "plain").mkdir()
    for name in files:
        (tmp_path / "plain" / name).write_bytes(b"PAR1" + b"x" * 16 + b"PAR1")
    bad = tmp_path / "plain" / files[0]
    bad.write_text("../../blobs/missing")
    problem = data_module.data_problem()
    assert "Observed on host" in problem and "symlink=False" in problem
    assert "../../blobs/missing" in problem and "starts=" in problem


@pytest.mark.parametrize("size", ["qwen1.7b-32k", "qwen8b-32k"])
def test_all_full_configs_schedule_longbench2_at_four_lengths(size):
    spec = importlib.util.spec_from_file_location("long_eval_plan", ROOT / "long_eval_plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def resolve(path):
        cfg = yaml.safe_load(path.read_text()) or {}
        return {**resolve(path.parent / cfg.pop("config")), **cfg} if "config" in cfg else cfg

    for path in (ROOT / "exp" / size).glob("eval_full_*.yaml"):
        cfg = resolve(path)
        passes = module.plan(",".join(cfg["eval_lengths"]))
        assert [p["name"] for p in passes] == ["32k", "64k", "128k", "256k"]
        for entry in passes:
            task = f"longbench2_{entry['name']}"
            assert task in entry["main"] and task not in entry["optional"], path
