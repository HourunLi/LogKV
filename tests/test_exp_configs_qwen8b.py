"""8B configurations are self-contained and support 64 distributed training ranks."""

import ast
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

EXP = Path(__file__).resolve().parents[1] / "exp"
QWEN8B = EXP / "qwen8b-32k"
BASE_8B = "/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-8B-Base/"
SAVE_8B = "/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/qwen8b-32k-gamma_k12_b128_p1024_2k"
METHODS = ("baseline", "dense", "word_first")


def resolve(path):
    cfg = yaml.safe_load(path.read_text()) or {}
    if "config" in cfg:
        cfg = {**resolve(path.parent / cfg.pop("config")), **cfg}
    return cfg


@pytest.mark.parametrize("name", sorted(p.name for p in QWEN8B.glob("*.yaml")))
def test_each_8b_config_loads_without_other_model_configs(name, tmp_path):
    isolated = tmp_path / QWEN8B.name
    shutil.copytree(QWEN8B, isolated)
    cfg = resolve(isolated / name)
    dense = name == "dense_niah_base.yaml" or name.endswith("_dense.yaml")
    if not dense:
        assert cfg["fsdp_low_memory"] is True
        assert cfg["global_batch_size"] == 64 and cfg["micro_batch_size"] == 1
        assert cfg["arch_name"] == "Qwen/Qwen3-8B-Base" and cfg["ckpt_dir"] == BASE_8B
    else:
        assert cfg["checkpoint_dir"] == cfg["tokenizer_dir"] == BASE_8B
        assert cfg["log_kv_dense_mode"] is True
    leftovers = [f"{k}={v}" for k, v in cfg.items() if "1.7b" in str(v).lower()]
    assert not leftovers, leftovers


def test_8b_batch_is_valid_for_64_ranks():
    cfg = resolve(QWEN8B / "arc_gamma_k12_b128_p1024_2k.yaml")
    tree = ast.parse((EXP.parent / "demo.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    start = next(index for index, node in enumerate(main.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "launch")
    stop = next(index for index, node in enumerate(main.body)
                if any(isinstance(child, ast.Constant) and child.value == "Tensorboard root:"
                       for child in ast.walk(node)))
    setup = compile(ast.Module(body=main.body[start + 1:stop], type_ignores=[]), "demo.py", "exec")
    scope = dict(fabric=SimpleNamespace(world_size=64, print=lambda *args: None),
                 global_batch_size=cfg["global_batch_size"], micro_batch_size=cfg["micro_batch_size"],
                 use_fsdp=True, fsdp_low_memory=cfg["fsdp_low_memory"])
    exec(setup, scope)
    assert scope["gradient_accumulation_steps"] == 1


@pytest.mark.parametrize("method", sorted(METHODS))
def test_full_evaluations_read_the_8b_checkpoint_and_write_apart(method):
    cfg = resolve(QWEN8B / f"eval_full_{method}.yaml")
    # eval.sh evaluates save_path (a training YAML) or checkpoint_dir, and hands it to RULER as `pretrained`.
    evaluated = cfg.get("save_path") or cfg.get("checkpoint_dir")
    assert evaluated == (BASE_8B if method == "dense" else SAVE_8B)
    if method == "dense":
        assert cfg["log_kv_dense_mode"] is True and cfg["metadata"]["pretrained"] == BASE_8B
        assert cfg["output_path"] == "/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-new/qwen8b_dense_niah_base_eval"
    else:
        assert cfg["arch_name"] == "Qwen/Qwen3-8B-Base" and cfg["ckpt_dir"] == BASE_8B
        assert cfg["eval_tag"] == f"full_{method}" and cfg.get("output_path") is None
    assert cfg["eval_main"] is True and cfg["long_limit"] == 100 and cfg.get("limit") is None


def test_baseline_and_word_first_share_the_checkpoint_and_differ_by_the_admission_score():
    baseline, word_first = (resolve(QWEN8B / f"eval_full_{m}.yaml") for m in ("baseline", "word_first"))
    diff = {k for k in set(baseline) | set(word_first) if baseline.get(k) != word_first.get(k)}
    assert diff == {"log_kv_gamma_surprisal", "eval_tag"}
    assert word_first["log_kv_gamma_surprisal"] == "word_first" and baseline.get("log_kv_gamma_surprisal") is None
