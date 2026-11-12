"""One evaluation over several lengths (long_eval_plan.py, eval.sh with eval_lengths / EVAL_LENGTHS)."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("long_eval_plan", ROOT / "long_eval_plan.py")
plan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan_module)
plan = plan_module.plan


def test_per_length_passes_take_the_factor_their_length_needs():
    passes = plan("32k,64k,128k,256k", output_dir="/out/", base_overrides={"rope_adjustments": {"factor": 4.0},
                                                                          "n_layer": 28})
    assert [p["name"] for p in passes] == ["32k", "64k", "128k", "256k"]
    assert [p["config_overrides"]["block_size"] for p in passes] == [33792, 66560, 132096, 263168]
    assert "rope_adjustments" not in passes[0]["config_overrides"]  # 32K: the trained length, no YaRN
    assert [p["config_overrides"].get("rope_adjustments", {}).get("factor") for p in passes] == [None, 2., 4., 8.]
    assert passes[3]["config_overrides"]["rope_adjustments"] == dict(factor=8., beta_fast=32, beta_slow=1,
                                                                     original_max_seq_len=32768)
    assert all(p["config_overrides"]["n_layer"] == 28 for p in passes)  # the YAML's other overrides stay
    assert [p["output_dir"] for p in passes] == ["/out/long32k", "/out/long64k", "/out/long128k", "/out/long256k"]
    assert [p["ruler_lengths"] for p in passes] == [[32768], [65536], [131072], [262144]]
    assert len(passes[2]["niah"]) == 13 and len(passes[3]["niah"]) == 12 and "ruler_qa_squad" not in passes[3]["niah"]
    assert passes[1]["main"] == ["babilong_64k", "long_ppl_64k", "long_ppl_books_64k", "niah_turns_64k",
                                 "longbench2_64k", "infinitebench_64k"]
    # Default suites: data may be missing (LongBench v2's or InfiniteBench's files may not be whole).
    assert passes[1]["optional"] == ["long_ppl_64k", "long_ppl_books_64k", "longbench2_64k", "infinitebench_64k"]
    assert passes[0]["main"][-1] == "infinitebench_32k"  # InfiniteBench at 32K too: the reference


def test_fixed_passes_share_the_longest_lengths_factor():
    passes = plan("32k,128k", yarn_mode="fixed", output_dir="/out")
    assert [p["config_overrides"]["block_size"] for p in passes] == [33792, 132096]  # windows still per length
    assert [p["config_overrides"]["rope_adjustments"]["factor"] for p in passes] == [4., 4.]
    assert [p["output_dir"] for p in passes] == ["/out/yarn4/long32k", "/out/yarn4/long128k"]
    assert plan("256k", yarn_mode="fixed")[0]["output_dir"] == "yarn8/long256k"


def test_suites_and_niah_only_narrow_the_tasks():
    p, = plan("128k", suites="turns,ruler,books")
    assert p["main"] == ["long_ppl_books_128k", "niah_turns_128k"] and len(p["niah"]) == 13
    assert p["optional"] == []  # named suites must run
    p, = plan("128k", suites="babilong")
    assert p["niah"] == [] and p["main"] == ["babilong_128k"]
    p, = plan("64k", niah_only=True)
    assert p["main"] == [] and len(p["niah"]) == 13
    for bad in (dict(lengths="48k"), dict(lengths="32k", suites="mmlu"), dict(lengths="32k", yarn_mode="dynamic"),
                dict(lengths="")):
        with pytest.raises(ValueError):
            plan(**bad)


def _launcher(tmp_path, config, env=None, args=()):
    """eval.sh's config reading, task lists, pass plan and stage loop, with run_stages printing what each run gets."""
    source = (ROOT / "eval.sh").read_text()

    def between(start, end_marker):
        i = source.index(start)
        return source[i:source.index(end_marker, i) + len(end_marker)]
    read_config = between('CONFIG_EXPORTS=$("${PYTHON_BIN}" ', 'eval "${CONFIG_EXPORTS}"')
    lengths_block = between("# Several lengths in one job (eval_lengths in the YAML", "    exit 1\nfi\n")
    build_args = between("LOG_KV_ARG_LIST=(", "\nDIAG_ARGS=")
    task_lists = between('DEFAULT_BENCHMARKS="boolq', "    esac\ndone\n")
    plan_block = between("# The length passes (long_eval_plan.py)", '"${LONG_EVAL_PASSES[@]}"\nfi\n')
    loop = between('if [ -z "${EVAL_LENGTHS}" ] || [ "${EVAL_MAIN}" = "1" ]; then', "    run_stages\ndone\n")
    stub = ('run_stages() { echo "PASS|${EVAL_OUTPUT_DIR}|${PASS_ARGS[*]}|${BENCHMARKS}|${NIAH_BENCHMARKS}|'
            '${EVAL_MAX_SEQ_LENGTHS}|${EVAL_OPTIONAL_TASKS}|${LOG_KV_ARG_LIST[*]}"; }')
    shell = "\n".join(("CONFIG_FILE=$1", read_config, lengths_block, build_args, task_lists, plan_block, stub, loop))
    environment = {**os.environ, "PYTHON_BIN": sys.executable, **(env or {}),
                   "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    for name in ("EVAL_LENGTHS", "EVAL_SUITES", "YARN_MODE", "NIAH_ONLY", "EVAL_MAIN", "LIMIT", "LONG_EVAL_LIMIT"):
        if name not in (env or {}):
            environment.pop(name, None)
    done = subprocess.run(["bash", "-c", shell, "bash", str(config), *args], capture_output=True, text=True,
                          cwd=ROOT, env=environment)
    passes = [line.split("|")[1:] for line in done.stdout.splitlines() if line.startswith("PASS|")]
    return done, passes


def _overrides(pass_args):
    """The JSON of "--config_overrides <json> [--limit N]"."""
    return json.loads(pass_args.split(" ", 1)[1].rsplit(" --limit ", 1)[0])


def test_the_launcher_runs_one_pass_per_length(tmp_path):
    config = tmp_path / "long.yaml"
    config.write_text("checkpoint_dir: /ckpt\noutput_path: /out\neval_lengths: [32k, 64k, 128k, 256k]\nlimit: 100\n"
                      "config_overrides:\n  n_layer: 28\n")
    done, passes = _launcher(tmp_path, config)
    assert done.returncode == 0, done.stderr
    assert [p[0] for p in passes] == ["/out/long32k", "/out/long64k", "/out/long128k", "/out/long256k"]
    overrides = [_overrides(p[1]) for p in passes]
    assert [o["block_size"] for o in overrides] == [33792, 66560, 132096, 263168]
    assert [o.get("rope_adjustments", {}).get("factor") for o in overrides] == [None, 2., 4., 8.]
    assert all(o["n_layer"] == 28 for o in overrides)
    assert all(p[1].endswith(" --limit 100") for p in passes)
    assert all("--config_overrides" not in p[6] and "--limit" not in p[6] for p in passes)  # only the pass's own
    assert passes[3][2] == ("babilong_256k,long_ppl_256k,long_ppl_books_256k,niah_turns_256k,longbench2_256k,"
                            "infinitebench_256k")
    assert "ruler_qa_squad" not in passes[3][3] and passes[3][4] == "[262144]"
    assert passes[0][5] == "long_ppl_32k,long_ppl_books_32k,longbench2_32k,infinitebench_32k"

    done, passes = _launcher(tmp_path, config, env=dict(EVAL_LENGTHS="128k", EVAL_SUITES="turns", YARN_MODE="fixed"))
    assert done.returncode == 0, done.stderr
    (out, args, main, niah, lengths, optional, _), = passes
    assert out == "/out/yarn4/long128k" and main == "niah_turns_128k" and niah == "none" and optional == ""
    done, passes = _launcher(tmp_path, config, env=dict(NIAH_ONLY="1", EVAL_LENGTHS="64k"))
    assert [p[2] for p in passes] == ["none"] and passes[0][3].startswith("niah_single_1")
    done, passes = _launcher(tmp_path, config, args=("piqa",))
    assert done.returncode != 0 and "EVAL_SUITES" in done.stdout and not passes  # task lists are not for this mode
    done, passes = _launcher(tmp_path, config, env=dict(EVAL_LENGTHS="48k"))
    assert done.returncode != 0 and not passes


def test_a_single_length_yaml_runs_as_before(tmp_path):
    config = tmp_path / "plain.yaml"
    config.write_text("checkpoint_dir: /ckpt\nlimit: 7\nconfig_overrides:\n  block_size: 66560\n")
    done, passes = _launcher(tmp_path, config)
    assert done.returncode == 0, done.stderr
    (out, args, *_rest, log_kv_args), = passes
    assert args == '--config_overrides {"block_size": 66560} --limit 7' and "--config_overrides" not in log_kv_args
    done, passes = _launcher(tmp_path, config, env=dict(LIMIT="2"))
    assert passes[0][1].endswith("--limit 2")


def test_the_full_evaluation_runs_the_main_line_then_every_length(tmp_path):
    """eval_main: the 32K main line (default lists, the YAML's limit, its overrides) into output_path, then the
    passes with long_limit samples per task, leaving out what the main line measured."""
    config = tmp_path / "full.yaml"
    config.write_text("checkpoint_dir: /ckpt\noutput_path: /out\neval_lengths: [32k, 64k]\neval_main: true\n"
                      "long_limit: 100\nconfig_overrides:\n  n_layer: 28\n")
    done, passes = _launcher(tmp_path, config)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [p[0] for p in passes] == ["/out", "/out/long32k", "/out/long64k"]
    main_line, long32k, long64k = passes
    assert main_line[1] == '--config_overrides {"n_layer": 28}'  # no limit: the whole main line
    assert main_line[2].startswith("boolq,") and main_line[2].endswith(",niah_turns_32k,long_ppl_books_32k")
    assert "niah_single_2_words" in main_line[3] and main_line[4] == "[1024, 2048, 4096, 8192, 16384, 32768]"
    assert main_line[5] == "niah_turns_32k,long_ppl_books_32k"
    # 32K: what the main line did not measure (its turns and book curve, its eight needles at 32768).
    assert long32k[2] == "babilong_32k,long_ppl_32k,longbench2_32k,infinitebench_32k"
    assert long32k[3] == "ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"
    assert long64k[2].startswith("babilong_64k,") and len(long64k[3].split(",")) == 13
    assert all(p[1].endswith(" --limit 100") and _overrides(p[1])["n_layer"] == 28 for p in (long32k, long64k))
    done, passes = _launcher(tmp_path, config, env=dict(LIMIT="2"))  # a quick check: every run at 2
    assert [p[1].rsplit(" ", 1)[-1] for p in passes] == ["2", "2", "2"]
    done, passes = _launcher(tmp_path, config, env=dict(EVAL_LENGTHS="none"))  # the main line only
    assert [p[0] for p in passes] == ["/out"]
    tagged = tmp_path / "tagged.yaml"  # several methods on one checkpoint: one folder each
    tagged.write_text("checkpoint_dir: /ckpt\neval_lengths: [64k]\neval_main: true\neval_tag: full_query\n")
    done, passes = _launcher(tmp_path, tagged)
    assert [p[0] for p in passes] == ["/ckpt/evaluate/full_query", "/ckpt/evaluate/full_query/long64k"]
    done, passes = _launcher(tmp_path, config, env=dict(EVAL_MAIN="0"))  # the lengths only, as eval_long_*.yaml
    assert [p[0] for p in passes] == ["/out/long32k", "/out/long64k"]
    assert passes[0][2].endswith("niah_turns_32k,longbench2_32k,infinitebench_32k")


def _majob_long_eval(tmp_path, context_length, env=None, main=None, niah=None):
    """majob.sh's post-training long-context block, with torchrun printing its arguments."""
    source = (ROOT / "majob.sh").read_text()
    lines = source.splitlines()
    default = {name: subprocess.run(["bash", "-c", "\n".join(l for l in lines if l.startswith(name + "=")) + f'\necho "${name}"'],
                                    capture_output=True, text=True).stdout.strip()
               for name in ("DEFAULT_BENCHMARKS", "DEFAULT_NIAH_BENCHMARKS")}
    main = default["DEFAULT_BENCHMARKS"].rstrip(",") + ",niah_turns_32k,long_ppl_books_32k" if main is None else main
    niah = default["DEFAULT_NIAH_BENCHMARKS"] if niah is None else niah
    block = source[source.index("# 长文本评测（训练后接着跑"):source.index("# 清理 barrier 目录（rank 0 负责）")]
    config = tmp_path / "train.yaml"
    config.write_text(f"save_path: /ckpt\ncontext_length: {context_length}\n")
    stub = f'''fakepython() {{ if [ "$1" = "-m" ]; then printf 'CALL'; printf ' <%s>' "$@"; printf '\\n'; else "{sys.executable}" "$@"; fi; }}
python() {{ "{sys.executable}" "$@"; }}
PYTHON_BIN=fakepython
CONFIG_FILE={config}
SAVE_DIR=/ckpt
EVAL_OUTPUT_DIR=/ckpt/evaluate
NODE_RANK=0; NUM_NODES=1; GPUS_PER_NODE=8; MASTER_ADDR=localhost; EVAL_MASTER_PORT=6001; NIAH_MASTER_PORT=6002
EVAL_LOG_KV_ARGS="--log_kv_B 128"
BENCHMARKS="{main}"
NIAH_BENCHMARKS="{niah}"
EVAL_MAX_SEQ_LENGTHS='[1024, 2048, 4096, 8192, 16384, 32768]'
'''
    environment = {k: v for k, v in os.environ.items()
                   if k not in ("EVAL_LENGTHS", "EVAL_SUITES", "YARN_MODE", "NIAH_ONLY", "LONG_PPL_DATA", "LONG_PPL_BOOKS")}
    done = subprocess.run(["bash", "-c", stub + block + 'echo "OPTIONAL=${EVAL_OPTIONAL_TASKS:-}|BOOKS=${LONG_PPL_BOOKS:-}"'],
                          capture_output=True, text=True, cwd=ROOT, env={**environment, **(env or {})})
    calls = []
    for line in done.stdout.splitlines():
        if line.startswith("CALL"):
            args = line[len("CALL <"):-1].split("> <")
            call = {k: args[args.index(k) + 1] for k in ("--benchmark", "--output_path", "--config_overrides",
                                                         "--limit", "--metadata")}
            call["--master_port"] = next(a.split("=", 1)[1] for a in args if a.startswith("--master_port="))
            calls.append(call)
    return done, calls


def test_majob_runs_the_long_passes_after_training(tmp_path):
    done, calls = _majob_long_eval(tmp_path, 32768, env=dict(LONG_PPL_BOOKS="/books"))
    assert done.returncode == 0, done.stdout + done.stderr
    assert [c["--output_path"] for c in calls] == ["/ckpt/evaluate/long32k"] * 2 + [
        f"/ckpt/evaluate/long{n}" for n in ("64k", "64k", "128k", "128k", "256k", "256k")]
    # 32K: only what the main line did not measure (its main stage ran the turns and the book curve, its NIAH
    # stage the eight needles at 32768).
    assert calls[0]["--benchmark"] == "babilong_32k,long_ppl_32k,longbench2_32k,infinitebench_32k"
    assert calls[1]["--benchmark"] == "ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad,ruler_qa_hotpot"
    assert calls[2]["--benchmark"] == ("babilong_64k,long_ppl_64k,long_ppl_books_64k,niah_turns_64k,longbench2_64k,"
                                       "infinitebench_64k")
    assert len(calls[7]["--benchmark"].split(",")) == 12 and "ruler_qa_squad" not in calls[7]["--benchmark"]
    overrides = [json.loads(c["--config_overrides"]) for c in calls[::2]]
    assert [o["block_size"] for o in overrides] == [33792, 66560, 132096, 263168]
    assert [o.get("rope_adjustments", {}).get("factor") for o in overrides] == [None, 2., 4., 8.]
    assert all(c["--limit"] == "100" for c in calls)
    assert [json.loads(c["--metadata"])["max_seq_lengths"] for c in calls[1::2]] == [[32768], [65536], [131072], [262144]]
    assert [c["--master_port"] for c in calls[:2]] == ["6001", "6002"]
    assert "BOOKS=/books" in done.stdout

    done, calls = _majob_long_eval(tmp_path, 4096)
    assert done.returncode == 0 and not calls and "训练长度 4096 不到 32K" in done.stdout
    done, calls = _majob_long_eval(tmp_path, 32768, env=dict(EVAL_LENGTHS="none"))
    assert done.returncode == 0 and not calls and "跳过长文本评测" in done.stdout
    done, calls = _majob_long_eval(tmp_path, 32768, env=dict(NIAH_ONLY="1", EVAL_LENGTHS="64k"), main="none")
    assert [c["--benchmark"].split(",")[0] for c in calls] == ["niah_single_1"]
    done, calls = _majob_long_eval(tmp_path, 32768, env=dict(EVAL_LENGTHS="128k", YARN_MODE="fixed",
                                                             EVAL_SUITES="turns"))
    assert [c["--output_path"] for c in calls] == ["/ckpt/evaluate/yarn4/long128k"]
