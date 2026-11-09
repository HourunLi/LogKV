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
                                 "longbench2_64k"]
    assert passes[1]["optional"] == ["long_ppl_64k", "long_ppl_books_64k"]  # default suites: data may be missing


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
    """eval.sh's config reading, pass plan and pass loop, with run_stages printing what each pass gets."""
    source = (ROOT / "eval.sh").read_text()

    def between(start, end_marker):
        i = source.index(start)
        return source[i:source.index(end_marker, i) + len(end_marker)]
    read_config = between('CONFIG_EXPORTS=$("${PYTHON_BIN}" ', 'eval "${CONFIG_EXPORTS}"')
    plan_block = between("# Several lengths in one job (eval_lengths in the YAML", '"${LONG_EVAL_PASSES[@]}"\nfi\n')
    build_args = between("LOG_KV_ARG_LIST=(", "\nDIAG_ARGS=")
    loop = between("if [ ${#LONG_EVAL_PASSES[@]} -eq 0 ]; then", "        run_stages\n    done\nfi\n")
    stub = ('run_stages() { echo "PASS|${EVAL_OUTPUT_DIR}|${PASS_ARGS[*]}|${BENCHMARKS}|${NIAH_BENCHMARKS}|'
            '${EVAL_MAX_SEQ_LENGTHS}|${EVAL_OPTIONAL_TASKS}|${LOG_KV_ARG_LIST[*]}"; }')
    shell = "\n".join(("CONFIG_FILE=$1", read_config, plan_block, build_args, stub, loop))
    environment = {**os.environ, "PYTHON_BIN": sys.executable, **(env or {}),
                   "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    for name in ("EVAL_LENGTHS", "EVAL_SUITES", "YARN_MODE", "NIAH_ONLY"):
        if name not in (env or {}):
            environment.pop(name, None)
    done = subprocess.run(["bash", "-c", shell, "bash", str(config), *args], capture_output=True, text=True,
                          cwd=ROOT, env=environment)
    passes = [line.split("|")[1:] for line in done.stdout.splitlines() if line.startswith("PASS|")]
    return done, passes


def test_the_launcher_runs_one_pass_per_length(tmp_path):
    config = tmp_path / "long.yaml"
    config.write_text("checkpoint_dir: /ckpt\noutput_path: /out\neval_lengths: [32k, 64k, 128k, 256k]\nlimit: 100\n"
                      "config_overrides:\n  n_layer: 28\n")
    done, passes = _launcher(tmp_path, config)
    assert done.returncode == 0, done.stderr
    assert [p[0] for p in passes] == ["/out/long32k", "/out/long64k", "/out/long128k", "/out/long256k"]
    overrides = [json.loads(p[1].split(" ", 1)[1]) for p in passes]  # "--config_overrides <json>"
    assert [o["block_size"] for o in overrides] == [33792, 66560, 132096, 263168]
    assert [o.get("rope_adjustments", {}).get("factor") for o in overrides] == [None, 2., 4., 8.]
    assert all(o["n_layer"] == 28 for o in overrides)
    assert all("--config_overrides" not in p[6] and "--limit 100" in p[6] for p in passes)  # only the pass's own
    assert passes[3][2] == "babilong_256k,long_ppl_256k,long_ppl_books_256k,niah_turns_256k,longbench2_256k"
    assert "ruler_qa_squad" not in passes[3][3] and passes[3][4] == "[262144]"
    assert passes[0][5] == "long_ppl_32k,long_ppl_books_32k"

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
    config.write_text("checkpoint_dir: /ckpt\nconfig_overrides:\n  block_size: 66560\n")
    done, passes = _launcher(tmp_path, config)
    assert done.returncode == 0, done.stderr
    (out, args, *_rest, log_kv_args), = passes
    assert args == "" and '--config_overrides {"block_size": 66560}' in log_kv_args
