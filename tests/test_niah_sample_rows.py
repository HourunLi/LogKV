"""Tests for eval.py's niah_sample_rows -- the pure-logic flattening of
lm_eval's log_samples output that scripts/analyze_niah_visibility.py depends
on for §7.3's "保存原始输出、标准答案和逐样本分数" requirement.

Uses the same ast.parse-and-exec-one-function technique as
tests/test_demo_launch.py to avoid importing eval.py itself (heavy lm_eval/
torch/numpy imports this pure function doesn't actually need). Not executed
in the environment that wrote this file.
"""

import __future__
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_niah_sample_rows():
    tree = ast.parse((ROOT / "eval.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "niah_sample_rows"
    )
    scope: dict = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "eval.py", "exec", flags=__future__.annotations.compiler_flag),
        scope,
    )
    return scope["niah_sample_rows"]


def test_returns_empty_list_when_no_samples_key():
    niah_sample_rows = _load_niah_sample_rows()
    assert niah_sample_rows({"results": {}}) == []


def test_returns_empty_list_when_samples_present_but_no_niah_task():
    niah_sample_rows = _load_niah_sample_rows()
    results = {"samples": {"piqa": [{"doc_id": 0, "target": "a"}]}}
    assert niah_sample_rows(results) == []


def test_flattens_only_niah_tasks_and_tags_task_name():
    niah_sample_rows = _load_niah_sample_rows()
    results = {
        "samples": {
            "piqa": [{"doc_id": 0, "target": "a"}],
            "niah_single_1": [{"doc_id": 0, "target": "x"}, {"doc_id": 1, "target": "y"}],
            "niah_single_2": [{"doc_id": 0, "target": "z"}],
        }
    }
    rows = niah_sample_rows(results)
    assert len(rows) == 3
    assert {r["task_name"] for r in rows} == {"niah_single_1", "niah_single_2"}
    assert all("doc_id" in r and "target" in r for r in rows)


def test_non_dict_records_are_wrapped_not_dropped():
    niah_sample_rows = _load_niah_sample_rows()
    results = {"samples": {"niah_single_3": ["not-a-dict-record"]}}
    rows = niah_sample_rows(results)
    assert len(rows) == 1
    assert rows[0]["raw"] == "not-a-dict-record"
    assert rows[0]["task_name"] == "niah_single_3"


def test_does_not_mutate_input_records():
    niah_sample_rows = _load_niah_sample_rows()
    original_record = {"doc_id": 0, "target": "x"}
    results = {"samples": {"niah_single_1": [original_record]}}
    niah_sample_rows(results)
    assert "task_name" not in original_record
