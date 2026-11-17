import importlib.util
import json
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
