"""eval_tasks/infinitebench: InfiniteBench's ten tasks at 64K / 128K / 256K, as the official code (OpenBMB/InfiniteBench)
prompts, cuts and scores them."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "eval_tasks" / "infinitebench"


@pytest.fixture(scope="module")
def ib():
    spec = importlib.util.spec_from_file_location("infinitebench_utils", FOLDER / "infinitebench_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_scoring(ib):
    s = ib.SCORERS
    assert s["passkey"]("71432. The pass key is 71432", ["71432"]) == 1.0
    assert s["passkey"](" 7143", ["71432"]) == 0.0
    assert s["number_string"](" 1234567890.", ["1234567890"]) == 1.0
    assert s["kv_retrieval"](' "a1b2-c3": "d4e5-f6"}', ["d4e5-f6"]) == 1.0
    assert s["kv_retrieval"](" d4e5", ["d4e5-f6"]) == 0.0
    assert s["longbook_choice_eng"](" B. The ship", ["The ship", "B"]) == 1.0
    assert s["longbook_choice_eng"](" C", ["The ship", "B"]) == 0.0
    assert s["code_debug"](" C", ["func_x", "C"]) == 1.0
    assert s["math_find"](" 42\n", [42]) == 1.0 and s["math_find"](" 41", [42]) == 0.0
    assert s["longdialogue_qa_eng"](" Gandalf, the", ["GANDALF"]) == 1.0
    assert s["longbook_qa_eng"]("the red ship", ["A red ship", "boat"]) == 1.0
    assert s["longbook_qa_eng"]("red", ["red ship"]) == pytest.approx(2 / 3)
    assert s["longbook_qa_chn"]("红色的船。", ["红色的船"]) == 1.0
    assert 0 < s["longbook_sum_eng"]("the hero leaves home", ["The hero leaves home and returns."]) < 1


def test_prompts_answers_and_budgets(ib):
    eg = dict(context="CTX", input="The largest number of the list is?", answer=[9], options=[])
    assert ib.prompt_of("math_find", eg) == "What is the largest number in the following list?\n\nCTX\n\n" + eg["input"]
    eg = dict(context="BOOK", input="Who?", answer="Bob", options=["Ann", "Bob", "Cy", "Di"])
    prompt = ib.prompt_of("longbook_choice_eng", eg)
    assert "Question: Who?\nA. Ann\nB. Bob\nC. Cy\nD. Di\n\nThe letter of the correct answer is" in prompt
    assert ib.answer_of("longbook_choice_eng", eg) == ["Bob", "B"]
    assert ib.budget("passkey", "64k") == 65536  # within the window's 1024-token margin
    assert ib.budget("longbook_sum_eng", "256k") == 262144 + 1024 - 1200 - 64  # its 1200-token summary
    assert ib.truncate_middle(list(range(10)), 4) == [0, 1, 8, 9]


class Words:
    """A tokenizer whose tokens are the text's words."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split(" ")}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def _write_data(folder):
    filler = " ".join(f"w{i}" for i in range(70000))  # longer than 64K words
    rows = {
        "passkey": dict(input="What is the pass key?", answer=["12345"]),
        "number_string": dict(input="What is the sequence?", answer=["1234567890"]),
        "kv_retrieval": dict(input="Key: k", answer=["v"]),
        "longbook_sum_eng": dict(input="", answer=["A summary."]),
        "longbook_qa_eng": dict(input="Who?", answer=["Bob"]),
        "longbook_choice_eng": dict(input="Who?", answer=["Bob"], options=["Ann", "Bob", "Cy", "Di"]),
        "longbook_qa_chn": dict(input="谁？", answer=["鲍勃"]),
        "longdialogue_qa_eng": dict(input="", answer=["BOB"]),
        "code_debug": dict(input="", answer="f2", options=["f1", "f2", "f3", "f4"]),
        "math_find": dict(input="The largest number of the list is?", answer=[7]),
    }
    for task, row in rows.items():
        with open(folder / f"{task}.jsonl", "w") as f:
            for i in range(3):
                f.write(json.dumps({"id": i, "context": filler if i == 0 else "short context", "options": [], **row}) + "\n")


def test_the_groups_cut_long_contexts_in_the_middle(tmp_path, monkeypatch):
    pytest.importorskip("lm_eval")
    import datasets
    import transformers
    from lm_eval.tasks import get_task_dict

    _write_data(tmp_path)
    monkeypatch.setenv("INFINITEBENCH_DIR", str(tmp_path))
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Words())
    source = ROOT / "eval.py"
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_task_manager"]
    scope = dict(inspect=inspect, EVAL_TASKS_DIR=ROOT / "eval_tasks")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    manager = scope["_task_manager"]({"pretrained": "unused", "max_seq_lengths": [65536]})
    assert {f"infinitebench_{n}" for n in ("32k", "64k", "128k", "256k")} <= set(manager.all_groups)
    tasks = get_task_dict(["infinitebench_64k"], manager)
    flat = {name: task for group in tasks.values() for name, task in group.items()} \
        if all(isinstance(v, dict) for v in tasks.values()) else tasks
    assert len(flat) == 10
    for name, task in flat.items():
        task_name = getattr(task, "config").task
        task.build_all_requests(limit=2, rank=0, world_size=1)
        long, short = (inst.args for inst in task.instances)
        words = long[0].split(" ")
        kind = task_name[len("infinitebench_"):-len("_64k")]
        # the instruction and the first half of the book, the last half and the question; not the middle
        assert len(words) <= ib_budget(kind) and long[0].startswith(ib_template(kind).split("{")[0])
        assert "\nw0 w1 " in long[0] and " w69999" in long[0] and " w35000 " not in long[0]
        assert long[1]["max_gen_toks"] == {"longbook_sum_eng": 1200, "passkey": 6}.get(kind, long[1]["max_gen_toks"])
        assert "short context" in short[0]  # shorter than the budget: whole
        doc = next(iter(task.test_docs()))
        assert task.process_results(doc, [""])["score"] == 0.0


def _module():
    spec = importlib.util.spec_from_file_location("ib", FOLDER / "infinitebench_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ib_budget(task):
    return _module().budget(task, "64k")


def ib_template(task):
    return _module().TASKS[task][0]


def _eval_functions(*names):
    import os
    import re

    source = ROOT / "eval.py"
    nodes = [n for n in ast.parse(source.read_text()).body
             if (isinstance(n, ast.FunctionDef) and n.name in names)
             or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("_LENGTH_NAMED_TASK",
                                                                                   "INFINITEBENCH_FILES"))]
    scope = dict(re=re, os=os, Path=Path, _is_main=lambda: True, EVAL_TASKS_DIR=ROOT / "eval_tasks")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


def test_the_window_must_hold_the_length_and_the_generation_margin():
    check = _eval_functions("_check_task_lengths")["_check_task_lengths"]
    check(["infinitebench_64k", "infinitebench_longbook_sum_eng_64k"], 65536 + 1024)
    for task in ("infinitebench_64k", "infinitebench_math_find_64k"):
        with pytest.raises(ValueError, match=f"{task} needs a model window of 66560"):
            check([task], 65536)


def test_missing_files_leave_the_default_lists_task_out(tmp_path, monkeypatch, capsys, ib):
    scope = _eval_functions("_check_task_lengths", "_skip_optional_tasks", "_infinitebench_files_problem",
                            "_longbench2_files_problem")
    assert set(scope["INFINITEBENCH_FILES"]) == set(ib.TASKS)
    skip = scope["_skip_optional_tasks"]
    monkeypatch.setenv("EVAL_OPTIONAL_TASKS", "infinitebench_128k")
    monkeypatch.setenv("INFINITEBENCH_DIR", str(tmp_path))
    assert skip(["piqa", "infinitebench_128k"], 132096) == ["piqa"]
    assert "InfiniteBench files missing" in capsys.readouterr().out
    _write_data(tmp_path)
    assert skip(["piqa", "infinitebench_128k"], 132096) == ["piqa", "infinitebench_128k"]


@pytest.mark.parametrize("launcher", ["eval.sh", "majob.sh"])
def test_the_main_stage_hands_infinitebench_the_tokenizer(launcher):
    """The tasks cut their prompts with the tokenizer named in --metadata (pretrained)."""
    import subprocess

    source = (ROOT / launcher).read_text()
    start = source.index("MAIN_METADATA_ARGS=()")
    granted = source.index('MAIN_METADATA_ARGS=(--metadata "${META}")', start)
    block = source[start:source.index("fi\n", granted) + 3]
    for tasks, expected in (("infinitebench_64k", "--metadata"), ("piqa", "")):
        done = subprocess.run(["bash", "-c", f'META=m\nBENCHMARKS={tasks}\n{block}\necho "${{MAIN_METADATA_ARGS[0]:-}}"'],
                              capture_output=True, text=True)
        assert done.stdout.strip() == expected, done.stderr


def test_a_named_task_without_its_files_fails_before_the_evaluation(tmp_path, monkeypatch, ib):
    scope = _eval_functions("_check_task_data", "_infinitebench_files_problem")
    monkeypatch.setenv("INFINITEBENCH_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_long_eval_data.py --infinitebench"):
        scope["_check_task_data"](["piqa", "infinitebench_64k"])
    scope["_check_task_data"](["piqa"])  # no InfiniteBench task: nothing to check
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="passkey.jsonl: missing"):
        ib.data_file("passkey")
    _write_data(tmp_path)
    scope["_check_task_data"](["infinitebench_64k"])
    monkeypatch.delenv("INFINITEBENCH_DIR")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    with pytest.raises(FileNotFoundError, match="not on this machine .*refs/main missing.*fetch_long_eval_data.py "
                                                ".*--infinitebench"):
        ib.data_file("math_find")


def test_the_plain_copy_under_hf_home_is_read_first(tmp_path, monkeypatch, ib):
    """The fetch script copies the files to <HF_HOME>/infinitebench: the hub cache's links may not reach the
    machines that evaluate."""
    monkeypatch.delenv("INFINITEBENCH_DIR", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "infinitebench").mkdir()
    _write_data(tmp_path / "infinitebench")
    assert ib.data_file("math_find") == str(tmp_path / "infinitebench" / "math_find.jsonl")
    scope = _eval_functions("_check_task_data", "_infinitebench_files_problem")
    scope["_check_task_data"](["infinitebench_256k"])
    (tmp_path / "infinitebench" / "passkey.jsonl").write_text("../../../../blobs/abc")  # a flattened link
    with pytest.raises(FileNotFoundError, match=r"\['passkey'\].*passkey.jsonl: \d+ bytes; hub cache"):
        scope["_check_task_data"](["infinitebench_256k"])


def test_the_hub_cache_is_read_through_flattened_links(tmp_path, monkeypatch, ib):
    """A copy or sync of the hub cache can turn a snapshot's link into a small file holding the link's text, and
    lose refs/main: the snapshot entry is followed to its blob."""
    monkeypatch.delenv("INFINITEBENCH_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    hub = tmp_path / "hub" / "datasets--xinrongzhang2022--InfiniteBench"
    data = tmp_path / "data"
    data.mkdir()
    _write_data(data)
    (hub / "blobs").mkdir(parents=True)
    snapshot = hub / "snapshots" / "c0ffee"
    snapshot.mkdir(parents=True)
    for i, task in enumerate(ib.TASKS):
        (hub / "blobs" / f"sha{i}").write_bytes((data / f"{task}.jsonl").read_bytes())
        link = f"../../blobs/sha{i}"
        if task == "passkey":
            (snapshot / f"{task}.jsonl").symlink_to(link)  # a live link
        else:
            (snapshot / f"{task}.jsonl").write_text(link)  # flattened
    assert ib.data_file("passkey") == str(snapshot / "passkey.jsonl")
    assert ib.data_file("math_find") == str(hub / "blobs" / f"sha{list(ib.TASKS).index('math_find')}")
    scope = _eval_functions("_check_task_data", "_infinitebench_files_problem")
    scope["_check_task_data"](["infinitebench_64k"])
    (hub / "blobs" / "sha1").unlink()  # number_string's blob did not arrive
    with pytest.raises(FileNotFoundError, match=r"number_string.jsonl: \d+ bytes -> missing"):
        scope["_check_task_data"](["infinitebench_64k"])
