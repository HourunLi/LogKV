"""Small CPU-only checks for eval logging; no model or lm_eval import required."""
import ast
import builtins
import sys
import contextlib
import io
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import tqdm


@pytest.fixture
def helpers(monkeypatch):
    for name in ("LOGKV_EVAL_VERBOSE", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    source = Path(__file__).resolve().parents[1] / "eval.py"
    tree = ast.parse(source.read_text())
    names = {"_env_int", "_global_rank", "_world_size", "_local_rank", "_rank_label", "_is_main",
             "_dist_ready", "_eval_verbose", "_eval_print", "_eval_progress", "_harness_logging"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LogKVLM")
    nodes.extend(node for node in model.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_warn_truncation", "all_gather_results"})
    fake_dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False,
                                get_rank=lambda: 0, get_world_size=lambda: 8)
    scope = dict(os=os, sys=sys, builtins=builtins, warnings=__import__('warnings'), torch=torch, logging=logging, contextlib=contextlib, tqdm=tqdm,
                 dist=fake_dist,
                 time=SimpleNamespace(perf_counter=Mock(return_value=0)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


@pytest.mark.parametrize("initialized,rank,env_rank,local_rank,visible", [
    (False, 0, 0, 2, True),
    (False, 0, 8, 0, False),
    (True, 0, 8, 2, True),
    (True, 8, 0, 0, False),
])
def test_only_global_main_prints(helpers, monkeypatch, capsys, initialized, rank, env_rank, local_rank, visible):
    helpers["dist"].is_initialized = lambda: initialized
    helpers["dist"].get_rank = lambda: rank
    monkeypatch.setenv("RANK", str(env_rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    helpers["_eval_print"]("model ready", flush=True)
    assert capsys.readouterr().out == ("model ready\n" if visible else "")


def test_progress_is_one_line_per_tenth_with_the_time_left(helpers, monkeypatch, capsys):
    monkeypatch.setenv("WORLD_SIZE", "8")
    # 25 items: lines at 0, every 3 items (a tenth, rounded up) and at 25; a 400 s item also reports at 4.
    helpers["time"].perf_counter.side_effect = [0] + [10 * i for i in range(1, 4)] + [430] + [430 + 10 * i for i in range(1, 22)]
    assert list(helpers["_eval_progress"](range(25), "Scoring")) == list(range(25))
    output = capsys.readouterr()
    lines = output.out.splitlines()
    counts = [int(line.split(": ")[1].split("/")[0]) for line in lines]
    assert counts == [0, 3, 4, 6, 9, 12, 15, 18, 21, 24, 25]
    assert all("GlobalRank 0/8" in line and "分片" in line for line in lines)
    assert "(12%)" in lines[1] and "约剩 220s" in lines[1] and "约剩" not in lines[-1]
    assert "\r" not in output.out and "\x1b" not in output.out and not output.err


def test_worker_progress_yields_every_item_without_logging(helpers, monkeypatch, capsys):
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    helpers["time"].perf_counter.side_effect = AssertionError("worker must not report progress")
    assert list(helpers["_eval_progress"](range(7), "Scoring")) == list(range(7))
    output = capsys.readouterr()
    assert not output.out and not output.err


def test_verbose_restores_worker_messages_and_tqdm(helpers, monkeypatch, capsys):
    monkeypatch.setenv("LOGKV_EVAL_VERBOSE", "1")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "8")
    original_init = tqdm.tqdm.__init__
    logger = logging.getLogger("lm_eval")
    monkeypatch.setattr(logger, "level", logging.INFO)
    with helpers["_harness_logging"]():
        assert tqdm.tqdm.__init__ is original_init and logger.level == logging.INFO
        helpers["_eval_print"]("worker detail")
        assert list(helpers["_eval_progress"](range(2), "Scoring")) == [0, 1]
    output = capsys.readouterr()
    assert output.out == "worker detail\n"
    assert "Scoring GlobalRank 2/8" in output.err and "100%" in output.err


@pytest.mark.parametrize("fail", [False, True])
def test_harness_keeps_warnings_and_restores_state_after_errors(helpers, monkeypatch, fail):
    log_output, progress_output = io.StringIO(), io.StringIO()
    handler = logging.StreamHandler(log_output)
    configured = {
        "lm_eval": logging.INFO,
        "lm_eval.explicit_child": logging.DEBUG,
        "datasets": logging.ERROR,
        "datasets.inherited_child": logging.NOTSET,
        "huggingface_hub": logging.WARNING,
        "transformers": logging.INFO,
    }
    loggers = {name: logging.getLogger(name) for name in configured}
    for name, logger in loggers.items():
        monkeypatch.setattr(logger, "level", configured[name])
        monkeypatch.setattr(logger, "handlers", [handler])
        monkeypatch.setattr(logger, "propagate", False)
    original_init = tqdm.tqdm.__init__
    imported_tqdm = tqdm.tqdm
    error = RuntimeError("evaluation failed")
    try:
        with helpers["_harness_logging"]():
            for name, logger in loggers.items():
                logger.info(f"{name} info")
                logger.warning(f"{name} warning")
                logger.error(f"{name} error")
            assert list(imported_tqdm(range(3), file=progress_output, disable=False)) == [0, 1, 2]
            if fail:
                raise error
    except RuntimeError as exc:
        assert fail and exc is error
    else:
        assert not fail
    assert tqdm.tqdm.__init__ is original_init
    assert {name: logger.level for name, logger in loggers.items()} == configured
    assert not progress_output.getvalue()
    output = log_output.getvalue()
    assert " info" not in output
    for name in configured:
        assert f"{name} error" in output
        assert (f"{name} warning" in output) == (not name.startswith("datasets"))
    loggers["lm_eval"].info("info restored")
    assert "info restored" in log_output.getvalue()


@pytest.mark.parametrize("verbose,expected_lines", [(False, 2), (True, 101)])
def test_truncation_warns_once_then_summarizes_each_round(helpers, monkeypatch, capsys, verbose, expected_lines):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOGKV_EVAL_VERBOSE", str(int(verbose)))
    model = SimpleNamespace(_truncation_counts={})
    for _ in range(100):
        helpers["_warn_truncation"](model, "generate_until", 128)
    results = ["answer"]
    assert helpers["all_gather_results"](model, results) is results
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == expected_lines
    assert all("GlobalRank 7/" in line for line in lines)
    assert "128" in lines[0] and "100 次" in lines[-1]
    assert model._truncation_counts == {}
    helpers["all_gather_results"](model, results)
    assert not capsys.readouterr().out
    helpers["_warn_truncation"](model, "generate_until", 64)
    assert "64" in capsys.readouterr().out


def test_harness_drops_the_ruler_builders_prints_only(helpers, capsys):
    """lm-eval's RULER sample builders print every sizing step on every rank; other prints stay."""
    # As lm-eval loads it for a task's YAML (!function): module name vt_utils, file under lm_eval/tasks/ruler.
    ruler = {"__name__": "vt_utils"}
    exec(compile("def size():\n    print('Max length 262144 | Current length 117909 | Noises: 4890')",
                 "/site-packages/lm_eval/tasks/ruler/vt_utils.py", "exec"), ruler)
    original = builtins.print
    with helpers["_harness_logging"]():
        ruler["size"]()
        print("eval progress")
    assert builtins.print is original
    out = capsys.readouterr().out
    assert "Noises" not in out and "eval progress" in out
    ruler["size"]()  # outside the evaluation it prints again
    assert "Noises" in capsys.readouterr().out


def test_other_ranks_keep_errors_only_and_rank_0_drops_datasets_cache_notices(helpers, monkeypatch, capsys):
    import warnings
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    loggers = {name: logging.getLogger(name) for name in ("lm_eval", "datasets")}
    for logger in loggers.values():
        monkeypatch.setattr(logger, "level", logging.INFO)
        monkeypatch.setattr(logger, "handlers", [handler])
        monkeypatch.setattr(logger, "propagate", False)
    for rank, expected in (("0", {"lm_eval warning", "lm_eval error", "datasets error"}),
                           ("3", {"lm_eval error", "datasets error"})):
        monkeypatch.setenv("RANK", rank)
        monkeypatch.setenv("WORLD_SIZE", "8")
        output.seek(0), output.truncate()
        with helpers["_harness_logging"]():
            for name, logger in loggers.items():
                logger.warning(f"{name} warning")
                logger.error(f"{name} error")
            with warnings.catch_warnings(record=True) as caught:
                warnings.warn("python warning")
        assert set(output.getvalue().splitlines()) == expected, rank
        assert len(caught) == (1 if rank == "0" else 0)
        assert {n: l.level for n, l in loggers.items()} == {"lm_eval": logging.INFO, "datasets": logging.INFO}
