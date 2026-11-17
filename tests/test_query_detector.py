import hashlib
import importlib.util
import io
import json
import ssl
import sys
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import pytest

from litgpt.query_detector import LABELS, QueryDetector, TaskSpan, candidates, cue_ids, validate_local_model


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
        {"rfilename": name, "lfs": {"sha256": hashlib.sha256(content).hexdigest()}}
        for name, content in files.items()]}
    requests = []

    def urlopen(url, *, timeout, context):
        requests.append(url)
        assert context.check_hostname is verify_ssl
        assert context.verify_mode == (ssl.CERT_REQUIRED if verify_ssl else ssl.CERT_NONE)
        address = url.full_url if hasattr(url, "full_url") else url
        body = json.dumps(info).encode() if "/api/models/" in address else files[
            urllib.parse.urlsplit(address).path.rsplit("/", 1)[-1]]
        return io.BytesIO(body)

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    flags = ["--verify-ssl"] if verify_ssl else []
    monkeypatch.setattr(sys, "argv", [str(path), "--out", str(tmp_path), *flags])
    module.main()
    assert len(requests) == len(files) + 1
    assert (tmp_path / "model.safetensors").read_bytes() == b"weights"
    module.verify(tmp_path)
    assert set(json.loads((tmp_path / "bundle.json").read_text())["sha256"]) == set(files)
    assert not (tmp_path / "wheels").exists()
    assert not (tmp_path / "detector-requirements.txt").exists()


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
        "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/gliclass-instruct-edge")
