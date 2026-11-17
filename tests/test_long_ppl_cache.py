"""PPL cache I/O under worker races and bucket failures, without tokenizer/GPU dependencies."""
import ast
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cache_helpers():
    path = ROOT / "eval_tasks/long_ppl/long_ppl_utils.py"
    names = {"_documents", "_write_selection_cache", "select_documents", "cache_dir", "cache_key",
             "_files", "_texts", "_missing"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(Path=Path, json=json, hashlib=hashlib, os=os, shutil=shutil, tempfile=tempfile,
                 time=time, warnings=warnings, LONGEST=131072, COLUMNS=("text", "content", "code"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def document_loader(tmp_path, monkeypatch):
    scope = cache_helpers()
    docs = [[1, 2, 3, 4]]
    select = Mock(return_value=docs)
    scope.update(_tokenizer=lambda kwargs: ("local-tokenizer", object()),
                 cache_key=lambda *args: "selection", cache_dir=lambda: tmp_path / "cache",
                 select_documents=select)
    monkeypatch.setenv("LONG_PPL_BOOKS", str(tmp_path / "books.jsonl"))
    return scope, docs, select


def test_workers_with_the_same_pid_have_independent_staging_files(tmp_path, monkeypatch):
    scope = cache_helpers()
    cache = tmp_path / "bucket" / "selection.json"
    docs = [[1, 2, 3] * 100]
    workers = 24
    barrier = threading.Barrier(workers, timeout=15)
    sources = []
    lock = threading.Lock()
    original = shutil.copyfile

    def copy(source, target):
        with lock:
            sources.append(Path(source))
        assert json.loads(Path(source).read_text()) == docs
        assert Path(source).parent != cache.parent
        barrier.wait()
        return original(source, target)

    monkeypatch.setattr(shutil, "copyfile", copy)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda _: scope["_write_selection_cache"](cache, docs), range(workers)))
    assert len(set(sources)) == workers
    assert all(not source.exists() for source in sources)
    assert json.loads(cache.read_text()) == docs
    assert list(cache.parent.iterdir()) == [cache]


def test_directory_visibility_retries_the_same_complete_local_file(tmp_path, monkeypatch):
    scope = cache_helpers()
    cache = tmp_path / "bucket" / "selection.json"
    docs = [[1, 2, 3]]
    sources, waits = [], []
    original = shutil.copyfile

    def copy(source, target):
        sources.append(Path(source))
        assert json.loads(Path(source).read_text()) == docs
        if len(sources) < 3:
            raise FileNotFoundError(2, "bucket directory not visible", str(target))
        return original(source, target)

    monkeypatch.setattr(shutil, "copyfile", copy)
    monkeypatch.setattr(time, "sleep", waits.append)
    scope["_write_selection_cache"](cache, docs)
    assert len(set(sources)) == 1 and waits == [0.2, 0.4]
    assert json.loads(cache.read_text()) == docs and not sources[0].exists()


@pytest.mark.parametrize("failure", ["directory", "copy"])
def test_cache_write_failure_keeps_documents_and_cleans_staging(tmp_path, monkeypatch, failure):
    scope, docs, select = document_loader(tmp_path, monkeypatch)
    original = tempfile.NamedTemporaryFile
    staging = []

    def temporary_file(*args, **kwargs):
        result = original(*args, **kwargs)
        staging.append(Path(result.name))
        return result

    def forbidden(*args, **kwargs):
        raise PermissionError(13, "bucket is read-only")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", temporary_file)
    if failure == "directory":
        monkeypatch.setattr(Path, "mkdir", forbidden)
    else:
        monkeypatch.setattr(shutil, "copyfile", forbidden)
    with pytest.warns(RuntimeWarning, match="LONG_PPL_CACHE.*node-local"):
        assert scope["_documents"]({}, longest=1048576, env="LONG_PPL_BOOKS") == docs
    select.assert_called_once()
    assert staging and all(not path.exists() for path in staging)


def test_persistent_visibility_failure_is_bounded_and_nonfatal(tmp_path, monkeypatch):
    scope, docs, select = document_loader(tmp_path, monkeypatch)
    copy = Mock(side_effect=FileNotFoundError(2, "bucket directory not visible"))
    waits = []
    monkeypatch.setattr(shutil, "copyfile", copy)
    monkeypatch.setattr(time, "sleep", waits.append)
    with pytest.warns(RuntimeWarning, match="Continuing with the selected documents"):
        assert scope["_documents"]({}, env="LONG_PPL_BOOKS") == docs
    assert copy.call_count == 3 and waits == [0.2, 0.4]
    select.assert_called_once()


@pytest.mark.parametrize("state", ["valid", "partial", "unreadable"])
def test_cache_reads_reuse_valid_documents_and_rebuild_unavailable_copies(tmp_path, monkeypatch, state):
    scope, docs, select = document_loader(tmp_path, monkeypatch)
    cache = tmp_path / "cache" / "selection.json"
    cache.parent.mkdir()
    cache.write_text("[" if state == "partial" else json.dumps(docs))
    if state == "unreadable":
        original = Path.open

        def open_file(path, mode="r", *args, **kwargs):
            if path == cache and mode == "r":
                raise PermissionError(13, "bucket cache read denied")
            return original(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_file)
    assert scope["_documents"]({}, env="LONG_PPL_BOOKS") == docs
    assert select.call_count == (0 if state == "valid" else 1)


def test_source_failure_propagates_without_becoming_a_cache_warning(tmp_path, monkeypatch):
    scope, _, select = document_loader(tmp_path, monkeypatch)
    error = FileNotFoundError(2, "source book missing", str(tmp_path / "books.jsonl"))
    select.side_effect = error
    with pytest.raises(FileNotFoundError) as caught:
        scope["_documents"]({}, env="LONG_PPL_BOOKS")
    assert caught.value is error
    assert not (tmp_path / "cache").exists()


def test_real_document_selection_is_reused_across_curve_points_and_separated_by_minimum_length(tmp_path, monkeypatch):
    scope = cache_helpers()
    source = tmp_path / "books.jsonl"
    source.write_text("\n".join(json.dumps({"text": " ".join(f"w{i}" for i in range(count))})
                                for count in (20, 100, 200)))
    monkeypatch.setenv("LONG_PPL_BOOKS", str(source))
    monkeypatch.setenv("LONG_PPL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LONG_PPL_DOCS", "2")
    monkeypatch.setenv("LONG_PPL_CONCAT", "0")

    def tokenizer(text, **kwargs):
        return {"input_ids": [int(word[1:]) for word in text.split()]}

    scope["_tokenizer"] = lambda kwargs: ("test-tokenizer", tokenizer)
    select = Mock(wraps=scope["select_documents"])
    scope["select_documents"] = select
    first = scope["_documents"]({"point": 32}, longest=64, env="LONG_PPL_BOOKS")
    second = scope["_documents"]({"point": 64}, longest=64, env="LONG_PPL_BOOKS")
    assert first == second == [list(range(64)), list(range(64))]
    assert select.call_count == 1
    longer = scope["_documents"]({}, longest=128, env="LONG_PPL_BOOKS")
    assert longer == [list(range(128))] and select.call_count == 2
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2
