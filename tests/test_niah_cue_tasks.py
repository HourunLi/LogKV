"""eval_tasks/niah_cue: RULER single_2/3 with the prompt around the haystack changed, loaded the way eval.py does."""
import ast
import inspect
from pathlib import Path
import re

import pytest

pytest.importorskip("lm_eval.tasks.ruler.prepare_niah", exc_type=ImportError)  # needs wonderwords and nltk
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
TASKS = [f"niah_single_{n}_{v}" for n in (2, 3) for v in ("qend", "preamble", "paraphrase")]
WORDS = ("Paul wrote about startups and cities. Founders learn by doing hard work every day. "
         "Good ideas often look bad at first. ").split() * 400


@pytest.fixture
def ruler(tmp_path, monkeypatch):
    """A whitespace tokenizer on disk, an offline essay haystack and a regex sentence splitter."""
    import lm_eval.tasks.ruler.prepare_niah as prepare

    vocab = {"[UNK]": 0}
    model = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    model.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=model, unk_token="[UNK]").save_pretrained(tmp_path)
    monkeypatch.setattr(prepare, "cached_sent_tokenize", lambda text: re.split(r"(?<=[.!?])\s+", text))
    # Before lm-eval imports niah_cue_utils from the YAML's directory, which binds this name.
    monkeypatch.setattr(prepare, "get_haystack", lambda type_haystack: WORDS)
    import importlib.util
    spec = importlib.util.spec_from_file_location("niah_cue_utils", ROOT / "eval_tasks" / "niah_cue" / "niah_cue_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils, dict(pretrained=str(tmp_path), max_seq_lengths=[2048])


def test_variants_change_only_the_prompt_around_the_haystack(ruler):
    utils, metadata = ruler
    for name in TASKS:
        doc = getattr(utils, name)(**dict(metadata))["test"][0]
        text, (answer,) = doc["input"], doc["outputs"]
        kind = "number" if name.startswith("niah_single_2") else "uuid"
        needle = re.search(rf"One of the special magic {kind}s for (\S+) is: (\S+)\.", text)
        assert needle and needle[2] == answer, name
        assert text.rstrip().endswith(f"What is the special magic {kind} for {needle[1]} mentioned in the provided text?")
        assert doc["gen_prefix"] == f"The special magic {kind} for {needle[1]} mentioned in the provided text is"
        opening = text[:text.index(needle[0])]
        if name.endswith("qend"):
            assert text.startswith("Paul wrote") and "special magic" not in opening
        elif name.endswith("preamble"):
            assert text.startswith(utils.PREAMBLE + f"A special magic {kind} is hidden within the following text.")
            assert len(utils.PREAMBLE.split()) >= 70  # Longer than the 64-token cue region in any tokenizer.
        else:
            assert text.startswith(utils.PARAPHRASE) and "special" not in opening and "magic" not in opening


def test_eval_py_task_manager_finds_and_builds_the_variants(ruler):
    utils, metadata = ruler
    source = ROOT / "eval.py"
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_task_manager"]
    scope = dict(inspect=inspect, EVAL_TASKS_DIR=ROOT / "eval_tasks")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    manager = scope["_task_manager"](metadata)
    assert set(TASKS) | {"niah_single_2"} <= set(manager.all_tasks)

    from lm_eval.tasks import get_task_dict

    # The YAML's !function resolves next to it, and the manager's metadata reaches the dataset.
    tasks = get_task_dict(["niah_single_2_qend", "niah_single_3_preamble"], manager)
    assert set(tasks) == {"niah_single_2_qend", "niah_single_3_preamble"}
    for name, task in tasks.items():
        docs = list(task.test_docs())
        assert len(docs) == 500 and {doc["max_length"] for doc in docs} == {2048}, name
        assert docs[0]["input"].startswith("Paul wrote" if name.endswith("qend") else utils.PREAMBLE)
