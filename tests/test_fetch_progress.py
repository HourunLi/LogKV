"""Preparation must remain observable before downloads and tokenization complete."""
import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from unused.fetch_progress import ProgressLines, run_stage

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_is_visible_and_persisted_before_work_finishes(tmp_path, capsys):
    summary = tmp_path / "nested/progress.summary.txt"
    with ProgressLines(summary) as lines:
        lines.append("environment ready")

        def work():
            assert "START download source" in summary.read_text()
            assert "environment ready" in capsys.readouterr().out
            assert "DONE download source" not in summary.read_text()
            return 42

        assert run_stage(lines, "download source", work) == 42
        assert lines[-1].startswith("DONE download source")
    assert not lines._thread.is_alive()
    assert "DONE download source" in summary.read_text()


def test_blocked_stage_heartbeats_and_interrupts_are_persisted(tmp_path, monkeypatch):
    summary = tmp_path / "progress.txt"
    emitted = threading.Event()
    with ProgressLines(summary, interval=.01) as lines:
        original = lines._emit

        def emit(message):
            original(message)
            if message.startswith("WAIT "):
                emitted.set()

        monkeypatch.setattr(lines, "_emit", emit)
        with pytest.raises(KeyboardInterrupt):
            with lines.stage("PPL shard"):
                lines.append("tokenizing book row 3")
                assert emitted.wait(2), "a blocked call must still produce progress"
                raise KeyboardInterrupt()
        text = summary.read_text()
        assert "WAIT PPL shard" in text and "tokenizing book row 3" in text
        assert "FAILED PPL shard" in text and "KeyboardInterrupt" in text
    assert not lines._thread.is_alive()


def test_download_reports_bytes_before_end_and_checks_content(tmp_path, monkeypatch):
    import requests
    from unused import long_eval_download as prep

    raw = b"abcdefgh"
    monkeypatch.setattr(prep, "hub_info", lambda repo: SimpleNamespace(sha="pin", siblings=[
        SimpleNamespace(rfilename="train.jsonl", size=len(raw),
                        lfs={"sha256": hashlib.sha256(raw).hexdigest()})]))
    ticks = iter(range(0, 1000, 6))
    monkeypatch.setattr(prep.time, "monotonic", lambda: next(ticks))
    lines = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, **kwargs):
            yield raw[:4]
            assert any("MB/s" in line for line in lines)
            assert not any("ordinary file" in line for line in lines)
            yield raw[4:]

    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(requests, "get", get)
    path = prep.download_plain("someone/books", "train.jsonl", tmp_path, lines)
    assert path.read_bytes() == raw and not path.is_symlink()
    assert hashlib.sha256(raw).hexdigest() in lines[-1]
    prep.download_plain("someone/books", "train.jsonl", tmp_path, lines)
    assert len(calls) == 1  # rechecking an intact file must not download again


def test_ppl_callback_reports_tokenization_without_changing_selection(tmp_path, capsys):
    utils = load("progress_ppl", "eval_tasks/long_ppl/long_ppl_utils.py")
    path = tmp_path / "books.jsonl"
    path.write_text("\n".join(json.dumps({"text": text}) for text in ("x", "a " * 12, "b " * 12)))
    events = []

    class Tokenizer:
        def __call__(self, text, **kwargs):
            if events:
                assert events[-1].startswith("PPL tokenize ")
            return {"input_ids": text.split()}

    expected = utils.select_documents(str(path), Tokenizer(), 2, longest=8)
    assert capsys.readouterr().out == ""
    actual = utils.select_documents(str(path), Tokenizer(), 2, longest=8, progress=events.append)
    assert actual == expected == [["a"] * 8, ["b"] * 8]
    assert any("12 tokens" in line for line in events)
    assert events[-1] == "PPL selected 2/2 documents"


def test_cli_writes_live_summary_even_when_later_stage_fails(tmp_path, monkeypatch, capsys):
    fetch = load("progress_fetch", "unused/fetch_long_eval_data.py")
    out = tmp_path / "fetch.json"
    summary = tmp_path / "fetch.summary.txt"
    monkeypatch.setattr(fetch, "patch_requests", lambda verify: None)
    monkeypatch.setattr(fetch.sys, "argv", ["fetch", "--no-babilong", "--no-ruler-qa", "--out", str(out)])

    def datasets(*args, **kwargs):
        text = summary.read_text()
        assert "HF_HOME=" in text and "START BABILong/LongBench v2" in text
        assert not out.exists(), "summary must be visible before the final JSON is written"
        args[2].append("source: FAILED example")
        return [], False

    monkeypatch.setattr(fetch, "fetch_datasets", datasets)
    with pytest.raises(SystemExit) as error:
        fetch.main()
    assert error.value.code == 1
    result = json.loads(out.read_text())
    assert result["ok"] is False and "source: FAILED example" in result["lines"]
    assert "Preparation FAILED" in summary.read_text()
    assert "Preparation FAILED" in capsys.readouterr().out
