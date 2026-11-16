"""The storage-only check must work inside a job without datasets/tokenizer dependencies."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def check(tmp_path, books=None, data=None):
    env = {k: v for k, v in os.environ.items() if k not in ("LONG_PPL_BOOKS", "LONG_PPL_DATA")}
    if books is not None:
        env["LONG_PPL_BOOKS"] = str(books)
    if data is not None:
        env["LONG_PPL_DATA"] = str(data)
    # -S removes site-packages: this mode must run without datasets, pyarrow, torch, or transformers.
    return subprocess.run([sys.executable, "-S", "-u", str(ROOT / "unused/check_long_eval_data.py"),
                           "--storage-only"], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=10)


def parquet(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PAR1abcdefghPAR1")  # readable markers; this mode does not validate Arrow metadata
    return path


def test_visible_sources_pass_without_tokenizer_or_cache(tmp_path):
    books = parquet(tmp_path / "books/emozilla--pg19/data/train.parquet")
    data = parquet(tmp_path / "data/emozilla--pg19-test/test.parquet")
    result = check(tmp_path, books.parent.parent, data.parent)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "host=" in result.stdout and "filesystem=" in result.stdout
    assert "0 storage problem(s)" in result.stdout and "footer=b'PAR1'" in result.stdout


def test_missing_sources_report_nearest_parent_and_actual_environment(tmp_path):
    (tmp_path / "present").mkdir()
    result = check(tmp_path, tmp_path / "absent/books", tmp_path / "absent/data")
    assert result.returncode == 1
    assert "2 storage problem(s)" in result.stdout and "FileNotFoundError" in result.stdout
    assert f"nearest existing path: {tmp_path}" in result.stdout and "present" in result.stdout
    assert str(tmp_path / "absent/books") in result.stdout
    assert not (tmp_path / "absent").exists()  # the diagnostic must never hide a missing mount with mkdir


def test_unexported_job_environment_is_reported(tmp_path):
    result = check(tmp_path)
    assert result.returncode == 1
    assert "LONG_PPL_BOOKS=<unset>" in result.stdout and "LONG_PPL_DATA=<unset>" in result.stdout
    assert "not exported in this job process" in result.stdout


def test_empty_folders_do_not_pass_as_visible_data(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = check(tmp_path, empty, empty)
    assert result.returncode == 1 and "no visible parquet/jsonl files" in result.stdout


@pytest.mark.parametrize("content", [b"<html>error</html>", b"../blobs/missing\0", b""])
def test_bad_or_dangling_inputs_report_the_filename(tmp_path, content):
    path = tmp_path / "train.parquet"
    path.write_bytes(content)
    data = parquet(tmp_path / "data/test.parquet")
    result = check(tmp_path, path, data)
    assert result.returncode == 1
    assert str(path) in result.stdout and "MISSING" in result.stdout


def test_flattened_input_is_inspected_without_changing_it(tmp_path):
    blob = parquet(tmp_path / "blobs/book")
    pointer = tmp_path / "train.parquet"
    pointer.write_bytes(b"./blobs/book\0")
    before = pointer.read_bytes()
    result = check(tmp_path, pointer, blob)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PPL link]" in result.stdout and pointer.read_bytes() == before
