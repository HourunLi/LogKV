import hashlib
import importlib.util
import io
import json
import sys
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from litgpt.query_detector import LABELS, QueryDetector, TaskSpan, candidates, cue_ids, validate_local_model


def downloader():
    path = Path(__file__).resolve().parents[1] / "unused/prepare_query_detector.py"
    spec = importlib.util.spec_from_file_location("prepare_query_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download_response(url, body=b"", status=200, headers=None):
    response = requests.Response()
    response.status_code = status
    response.url = url
    response.headers.update(headers or {})
    response.raw = io.BytesIO(body)
    return response


def test_detects_middle_of_long_input_preserving_offsets_and_rejecting_all():
    calls = []

    def classifier(texts, labels, **kwargs):
        calls.extend(texts)
        return [[{"label": label, "score": (0.99 if label == LABELS[0] and
                 "What is the account code?" in text.split("<candidate>")[1].split("</candidate>")[0]
                 else 0.01)} for label in labels] for text in texts]

    detector = QueryDetector(None, pipeline=classifier, batch_size=2)
    text = "The road was empty. " * 200 + "What is the account code?\n" + "The sky was clear. " * 200
    spans = detector.detect(text)
    assert [text[s.start:s.end] for s in spans] == ["What is the account code?"]
    assert len(calls) == 1 and len(calls[0]) < 1400
    assert detector.detect("The sky was clear.") == []


def test_candidate_scan_covers_late_unpunctuated_text_and_can_disable_rule_filter():
    text = "background " * 1000 + "remember the account owner"
    ranges = list(candidates(text))
    assert ranges and "remember the account owner" in text[ranges[-1][0]:ranges[-1][1]]
    assert list(candidates("ordinary text")) == []
    assert list(candidates("ordinary text", use_rules=False)) == [(0, 13)]


def test_no_forced_positive_and_negative_class_can_reject_confident_question():
    def classifier(texts, labels, **kwargs):
        return [[{"label": labels[0], "score": 0.9}, {"label": labels[1], "score": 0.99}]
                for _ in texts]
    detector = QueryDetector(None, pipeline=classifier)
    assert detector.detect('The reporter asked, "What is the budget?"') == []


def test_cues_use_original_lm_ids_instead_of_detector_tokenization():
    tokenizer = SimpleNamespace(backend="huggingface", processor=SimpleNamespace(
        encode=lambda text: SimpleNamespace(ids=[10, 20, 30], offsets=[(0, 4), (4, 8), (8, 12)])))
    assert cue_ids("abc def ghij", [TaskSpan(5, 10, 1)], tokenizer) == [20, 30]
    assert cue_ids("abc def ghij", [], tokenizer) == []


def test_local_loading_rejects_hub_ids_incomplete_files_and_symlinks(tmp_path):
    with pytest.raises(ValueError, match="local ordinary"):
        validate_local_model("knowledgator/not-a-local-model")
    with pytest.raises(FileNotFoundError):
        validate_local_model(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "config.json").write_text(json.dumps({"architecture_type": "uni-encoder", "encoder_config": {}}))
    assert validate_local_model(tmp_path) == tmp_path
    (tmp_path / "bad-link").symlink_to(tmp_path / "config.json")
    with pytest.raises(ValueError, match="symlink"):
        validate_local_model(tmp_path)


def test_bundle_verifies_bytes_not_just_file_existence(tmp_path):
    path = Path(__file__).resolve().parents[1] / "unused/prepare_query_detector.py"
    spec = importlib.util.spec_from_file_location("prepare_query_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"ordinary bytes")
    (tmp_path / "bundle.json").write_text(json.dumps(
        {"revision": "test", "sha256": {artifact.name: module.sha256(artifact)}}))
    module.verify(tmp_path)
    artifact.write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="corrupt"):
        module.verify(tmp_path)


@pytest.mark.parametrize("verify_ssl", [False, True])
def test_download_certificate_policy_covers_only_required_model_files(tmp_path, monkeypatch, verify_ssl):
    path = Path(__file__).resolve().parents[1] / "unused/prepare_query_detector.py"
    spec = importlib.util.spec_from_file_location("prepare_query_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {"config.json": b"{}", "model.safetensors": b"weights",
             "tokenizer.json": b"{}", "tokenizer_config.json": b"{}"}
    info = {"sha": "fixed-revision", "siblings": [
        {"rfilename": name, "size": len(content), "lfs": {"sha256": hashlib.sha256(content).hexdigest()}}
        for name, content in files.items()] + [{"rfilename": "unused.bin", "size": 100}]}
    calls = []

    def get(session, url, *, timeout, verify, stream):
        calls.append(url)
        assert verify is verify_ssl and timeout == (60, 300)
        assert session.headers["Cache-Control"] == "no-cache"
        assert session.headers["Accept-Encoding"] == "identity"
        assert url.startswith("https://hf-mirror.com/")
        assert stream is ("/api/models/" not in url)
        body = json.dumps(info).encode() if not stream else files[
            urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]]
        return download_response(url, body)

    monkeypatch.setattr(requests.Session, "get", get)
    flags = ["--verify-ssl"] if verify_ssl else []
    monkeypatch.setattr(sys, "argv", [str(path), "--out", str(tmp_path), *flags])
    module.main()
    assert len(calls) == len(files) + 1
    assert (tmp_path / "model.safetensors").read_bytes() == b"weights"
    module.verify(tmp_path)
    assert set(json.loads((tmp_path / "bundle.json").read_text())["sha256"]) == set(files)
    assert not (tmp_path / "wheels").exists()
    assert not (tmp_path / "detector-requirements.txt").exists()


def test_403_refreshes_the_mirror_url_instead_of_reusing_signed_redirect(tmp_path, monkeypatch, capsys):
    module = downloader()
    url = "https://hf-mirror.com/repo/resolve/revision/model.safetensors"
    calls, waits, responses = [], [], []
    target = tmp_path / "model.safetensors.part"
    target.write_bytes(b"previous partial download")

    def get(address, **kwargs):
        calls.append(address)
        if len(calls) == 1:
            response = download_response("https://cdn.example/weights?signature=secret", b"Forbidden", 403)
        else:
            assert target.read_bytes() == b"previous partial download"
            response = download_response("https://cdn.example/weights?signature=new", b"weights")
        responses.append(response)
        return response

    monkeypatch.setattr(module.time, "sleep", waits.append)
    module.fetch(SimpleNamespace(get=get), url, target=target, expected_size=7,
                 expected_hash=hashlib.sha256(b"weights").hexdigest(), retries=1)
    assert calls[0] == url and calls[1].startswith(url + "?_logkv_retry=")
    assert waits == [1] and target.read_bytes() == b"weights"
    assert responses[0].raw.closed  # Release the rejected response before requesting a fresh redirect.
    output = capsys.readouterr().out
    assert "HTTP 403 from cdn.example" in output and "secret" not in output


def test_429_obeys_retry_after_and_404_does_not_retry(tmp_path, monkeypatch):
    module = downloader()
    waits = []
    responses = iter([download_response("https://hf-mirror.com/api", status=429, headers={"Retry-After": "7"}),
                      download_response("https://hf-mirror.com/api", b'{"sha": "revision"}')])
    monkeypatch.setattr(module.time, "sleep", waits.append)
    assert module.fetch(SimpleNamespace(get=lambda *a, **k: next(responses)),
                        "https://hf-mirror.com/api", retries=1) == {"sha": "revision"}
    assert waits == [7]
    with pytest.raises(RuntimeError, match="HTTP 404"):
        module.fetch(SimpleNamespace(get=lambda *a, **k: download_response("https://hf-mirror.com/file", status=404)),
                     "https://hf-mirror.com/file", target=tmp_path / "file.part", retries=5)
    assert waits == [7] and not (tmp_path / "file.part").exists()


@pytest.mark.parametrize("failure", ["stream", "size", "hash"])
def test_failed_stream_and_integrity_errors_restart_partial_file(tmp_path, monkeypatch, failure):
    module = downloader()
    target = tmp_path / "model.part"
    calls, waits = [], []

    def broken_stream(**kwargs):
        yield b"incomplete"
        raise requests.exceptions.ChunkedEncodingError("https://cdn.example/?signature=secret")

    def get(url, **kwargs):
        calls.append(url)
        response = download_response(url, b"weights" if len(calls) > 1 else b"badfile" if failure == "hash" else b"bad")
        if len(calls) == 1 and failure == "stream":
            response.iter_content = broken_stream
        return response

    monkeypatch.setattr(module.time, "sleep", waits.append)
    module.fetch(SimpleNamespace(get=get), "https://hf-mirror.com/file", target=target,
                 expected_size=7, expected_hash=hashlib.sha256(b"weights").hexdigest(), retries=1)
    assert len(calls) == 2 and waits == [1] and target.read_bytes() == b"weights"


def test_exhausted_403_is_bounded_and_preserves_completed_file(tmp_path, monkeypatch):
    module = downloader()
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"previous weights")
    calls, waits = [], []

    def get(url, **kwargs):
        calls.append(url)
        return download_response("https://cdn.example/weights?signature=secret", b"Forbidden", 403)

    monkeypatch.setattr(module.time, "sleep", waits.append)
    names = [target.name, *sorted(module.REQUIRED_FILES - {target.name})]
    info = {"sha": "fixed", "siblings": [{"rfilename": name} for name in names]}
    with pytest.raises(RuntimeError, match="after 3 attempts: HTTP 403 from cdn.example"):
        module.download_files(SimpleNamespace(get=get), info, tmp_path, "https://hf-mirror.com", {"retries": 2})
    assert len(calls) == 3 and waits == [1, 2]
    assert target.read_bytes() == b"previous weights"
    assert not (tmp_path / "bundle.json").exists()


@pytest.mark.parametrize("flag", ["--install-deps", "--wheels"])
def test_downloader_rejects_old_package_management_flags_before_creating_files(tmp_path, monkeypatch, flag):
    path = Path(__file__).resolve().parents[1] / "unused/prepare_query_detector.py"
    spec = importlib.util.spec_from_file_location("prepare_query_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", [str(path), "--out", str(target), flag])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2 and not target.exists()


def test_project_keeps_original_transformers_range_without_detector_dependency_override():
    source = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert '"transformers>=4.51.3,<4.57"' in source
    assert "gliclass==" not in source and "optional-dependencies.query-detector" not in source


def test_download_and_smoke_default_to_the_requested_hf_cache():
    from litgpt.query_detector import DEFAULT_MODEL_DIR

    path = Path(__file__).resolve().parents[1] / "unused/prepare_query_detector.py"
    spec = importlib.util.spec_from_file_location("prepare_query_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert str(module.DEFAULT_MODEL_DIR) == DEFAULT_MODEL_DIR
    assert DEFAULT_MODEL_DIR == (
        "/workspace/data/hf_cache/gliclass-instruct-edge")
