"""PPL reading and preparation must survive mount-flattened Hub links without mutating snapshots."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval_tasks.long_ppl.input_files import open_parquet, resolve_input
from unused import long_eval_download as prep

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("suffix", ["", "\x00\n"])
def test_ppl_reads_flattened_pointer_chains_without_writes(tmp_path, monkeypatch, capsys, absolute, suffix):
    utils = load("ppl_link_read", "eval_tasks/long_ppl/long_ppl_utils.py")
    blob = tmp_path / "blobs/bytes"
    blob.parent.mkdir()
    pq.write_table(pa.table({"text": ["same book", "second book"]}), blob)
    pointer = tmp_path / "snapshot/data/train.parquet"
    pointer.parent.mkdir(parents=True)
    link = tmp_path / "snapshot/second-link"
    link.write_text(str(blob) if absolute else "../blobs/bytes")
    pointer.write_text((str(link) if absolute else "../second-link") + suffix)
    before = pointer.read_bytes(), link.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must not create links, rename files, or repair shared snapshots")

    monkeypatch.setattr(os, "symlink", forbidden)
    monkeypatch.setattr(os, "rename", forbidden)
    monkeypatch.setattr(os, "replace", forbidden)
    assert list(utils._texts(pointer)) == ["same book", "second book"]
    assert (pointer.read_bytes(), link.read_bytes()) == before
    output = capsys.readouterr().out
    assert "[PPL input]" in output and "footer=b'PAR1'" in output
    assert str(pointer) in output and str(blob) in output


@pytest.mark.parametrize("raw", [b"PAR1truncated", b"<html>access denied</html>",
                                  b"version https://git-lfs.github.com/spec/v1\n", b""])
def test_bad_parquet_names_the_file_and_bytes_before_arrow(tmp_path, monkeypatch, capsys, raw):
    path = tmp_path / "train.parquet"
    path.write_bytes(raw)
    monkeypatch.setattr(pq, "ParquetFile", lambda *a, **kw: pytest.fail("invalid bytes must fail before Arrow"))
    with pytest.raises(ValueError, match="not a whole Parquet") as error:
        open_parquet(path)
    assert str(path) in str(error.value) and "footer=" in str(error.value)
    assert f"size={len(raw)}" in capsys.readouterr().out


def test_missing_blob_and_link_cycle_fail_with_paths(tmp_path):
    path = tmp_path / "train.parquet"
    path.write_bytes(b"../blobs/missing\0")
    with pytest.raises(ValueError, match="missing or unreadable") as error:
        open_parquet(path)
    assert "blobs/missing" in str(error.value) and "LONG_PPL_BOOKS" in str(error.value)
    path.write_text("./second")
    (tmp_path / "second").write_text("./train.parquet")
    with pytest.raises(ValueError, match="cycle"):
        resolve_input(path)


def test_invalid_metadata_with_par1_markers_reports_the_input(tmp_path):
    path = tmp_path / "metadata-broken.parquet"
    path.write_bytes(b"PAR1" + b"x" * 20 + b"PAR1")
    with pytest.raises(ValueError, match="Cannot open PPL Parquet") as error:
        open_parquet(path)
    assert str(path) in str(error.value) and "footer=b'PAR1'" in str(error.value)


def test_plain_download_reuses_verified_blob_and_keeps_snapshot_unchanged(tmp_path, monkeypatch):
    import requests

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    cache = tmp_path / "hub/datasets--someone--books"
    blob = cache / "blobs/content"
    blob.parent.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["long book"]}), blob)
    raw = blob.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    pointer = cache / "snapshots/pin/data/train.parquet"
    pointer.parent.mkdir(parents=True)
    pointer.write_bytes(b"../../../blobs/content\0")
    before = pointer.read_bytes()
    monkeypatch.setattr(prep, "hub_info", lambda repo: SimpleNamespace(sha="pin", siblings=[
        SimpleNamespace(rfilename="data/train.parquet", size=len(raw), lfs={"sha256": digest})]))
    forbidden = lambda *a, **kw: pytest.fail("intact cached bytes need no download or link/rename")
    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(os, "symlink", forbidden)
    monkeypatch.setattr(os, "rename", forbidden)
    monkeypatch.setattr(os, "replace", forbidden)
    lines = []
    result = prep.download_plain("someone/books", "data/train.parquet", tmp_path / "ordinary", lines)
    assert result.read_bytes() == raw and not result.is_symlink()
    assert pointer.read_bytes() == before and any("reused verified Hub cache" in line for line in lines)


def test_wrong_cached_hash_is_downloaded_again_not_copied(tmp_path, monkeypatch):
    import requests

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"text": ["valid"]}), source)
    raw = source.read_bytes()
    bad = tmp_path / "hub/datasets--someone--books/snapshots/pin/data/train.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(raw.replace(b"valid", b"other"))  # same size and PAR1 markers, different contents
    assert len(bad.read_bytes()) == len(raw) and bad.read_bytes() != raw
    monkeypatch.setattr(prep, "hub_info", lambda repo: SimpleNamespace(sha="pin", siblings=[
        SimpleNamespace(rfilename="data/train.parquet", size=len(raw),
                        lfs={"sha256": hashlib.sha256(raw).hexdigest()})]))
    calls = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, **kwargs): yield raw

    def get(*args, **kwargs):
        calls.append(args)
        return Response()

    monkeypatch.setattr(requests, "get", get)
    result = prep.download_plain("someone/books", "data/train.parquet", tmp_path / "ordinary", [])
    assert result.read_bytes() == raw and len(calls) == 1


def test_ppl_hub_always_returns_ordinary_files(tmp_path, monkeypatch):
    import huggingface_hub

    fetch = load("fetch_plain_ppl", "unused/fetch_long_eval_data.py")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", lambda *a, **kw: ["README.md", "test/a.parquet"])
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: pytest.fail("no snapshot output"))

    def download(repo, name, folder, lines):
        path = Path(folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"text": ["held out"]}), path)
        return path

    monkeypatch.setattr(prep, "download_plain", download)
    lines = []
    root = fetch.fetch_ppl_hub("someone/test", lines)
    assert root == str(tmp_path / "long_ppl_data/someone--test")
    assert list(fetch._long_ppl_utils()._texts(Path(root) / "test/a.parquet")) == ["held out"]
    assert lines[-1].endswith(f"LONG_PPL_DATA={root}")


def test_preflight_checks_raw_inputs_even_if_selection_is_cached(tmp_path, monkeypatch):
    checker = load("check_ppl_raw", "unused/check_long_eval_data.py")
    path = tmp_path / "bad.parquet"
    path.write_bytes(b"not parquet")
    monkeypatch.setenv("LONG_PPL_BOOKS", str(tmp_path))
    reports = []
    # validate_inputs must run before _documents, which could return a previously warmed selection.
    checker.check_ppl(lambda ok, message: reports.append((ok, message)), "unused-tokenizer", ["32k"], ["books"])
    assert len(reports) == 1 and reports[0][0] is False
    assert str(path) in reports[0][1] and "footer=" in reports[0][1]
