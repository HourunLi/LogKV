"""Multi-needle RULER tasks in eval_tasks/: registration, sample generation and recall scoring.

Uses a toy word-level tokenizer and an in-memory essay haystack, so no HF download is needed.
"""

from pathlib import Path

import pytest

pytest.importorskip("lm_eval")
pytest.importorskip("transformers")
prepare_niah = pytest.importorskip("lm_eval.tasks.ruler.prepare_niah")  # needs wonderwords + nltk

import nltk  # noqa: E402
from lm_eval import evaluator  # noqa: E402
from lm_eval.api.model import LM  # noqa: E402
from lm_eval.tasks import TaskManager, get_task_dict  # noqa: E402
from lm_eval.utils import load_yaml_config  # noqa: E402

TASK_DIR = Path(__file__).resolve().parents[1] / "eval_tasks"
NEEDLE = "One of the special magic numbers for"


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [tokenizers.pre_tokenizers.Whitespace(), tokenizers.pre_tokenizers.Digits(individual_digits=True)]
    )
    path = tmp_path_factory.mktemp("toy_tokenizer")
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(path)
    return str(path)


@pytest.fixture
def offline_essay(monkeypatch):
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        pytest.skip("nltk punkt_tab is not installed")
    original = prepare_niah.get_haystack
    words = " ".join(f"Essay sentence {i} discusses topic {i % 7} in plain words." for i in range(3000)).split(" ")
    # Task YAMLs re-import niah_multi on load, which then binds this patched function.
    monkeypatch.setattr(
        prepare_niah,
        "get_haystack",
        lambda type_haystack: words if type_haystack == "essay" else original(type_haystack),
    )


def task_manager(tokenizer_dir):
    metadata = {"pretrained": tokenizer_dir, "max_seq_lengths": [1024, 2048], "niah_num_samples": 3}
    return TaskManager(include_path=str(TASK_DIR), metadata=metadata)


def test_task_configs_reserve_their_generation_budget():
    paths = sorted((TASK_DIR / "niah_multi").glob("niah_*.yaml"))
    assert len(paths) == 8
    for path in paths:
        cfg = load_yaml_config(str(path), mode="simple")
        n = int(cfg["task"].rsplit("_n", 1)[1])
        key = "num_needle_v" if "multivalue" in cfg["task"] else "num_needle_q"
        assert cfg["dataset_kwargs"][key] == n
        assert cfg["dataset_kwargs"].get("tokens_to_generate", 128) == cfg["generation_kwargs"]["max_gen_toks"]


@pytest.mark.parametrize("name, n", [("niah_multivalue_n8", 8), ("niah_multiquery_n2", 2)])
def test_docs_hold_all_needles_and_score_partial_recall(tokenizer_dir, offline_essay, name, n):
    task = get_task_dict([name], task_manager(tokenizer_dir))[name]
    docs = list(task.test_docs())
    assert [doc["max_length"] for doc in docs] == [1024] * 3 + [2048] * 3
    for doc in docs:
        assert len(doc["outputs"]) == n
        assert doc["input"].count(NEEDLE) == n
        assert all(answer in doc["input"] for answer in doc["outputs"])
        assert doc["length"] <= doc["max_length"]
        metrics = task.process_results(doc, [", ".join(doc["outputs"][: n // 2])])
        assert metrics[str(doc["max_length"])] == pytest.approx(0.5)


def test_lengths_that_cannot_fit_all_needles_are_skipped(tokenizer_dir, offline_essay):
    # 32 needles + question + 768 reserved tokens do not fit in 1024; lm-eval would retry forever.
    task = get_task_dict(["niah_multiquery_n32"], task_manager(tokenizer_dir))["niah_multiquery_n32"]
    assert {doc["max_length"] for doc in task.test_docs()} == {2048}


def test_simple_evaluate_reports_recall_per_length(tokenizer_dir, offline_essay):
    class HalfRecall(LM):
        def loglikelihood(self, requests):
            raise NotImplementedError

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError

        def generate_until(self, requests):
            for req in requests:
                prompt, gen_kwargs = req.args
                assert prompt.endswith("mentioned in the provided text are")
                assert gen_kwargs["max_gen_toks"] == 256
            return [", ".join(req.doc["outputs"][:8]) for req in requests]

    results = evaluator.simple_evaluate(
        model=HalfRecall(), tasks=["niah_multivalue_n16"], task_manager=task_manager(tokenizer_dir)
    )["results"]["niah_multivalue_n16"]
    assert results["1024,none"] == pytest.approx(0.5)
    assert results["2048,none"] == pytest.approx(0.5)
