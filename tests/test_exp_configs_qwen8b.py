"""exp/qwen8b-32k: the 1.7B experiments on Qwen3-8B-Base, differing from them only in the model and output paths."""

from pathlib import Path

import pytest
import yaml

EXP = Path(__file__).resolve().parents[1] / "exp"
QWEN8B, QWEN17B = EXP / "qwen8b-32k", EXP / "qwen1.7b-32k"
BASE_8B = "/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-8B-Base/"
SAVE_8B = "/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/qwen8b-32k-gamma_k12_b128_p1024_2k"
# The keys an 8B run changes: the model it starts from and where it writes.
TRAINED = {"arch_name", "ckpt_dir", "expid", "save_path", "merged_state_staging_dir", "tensorboard_root"}
DENSE = {"checkpoint_dir", "tokenizer_dir", "output_path", "metadata"}
METHODS = {"dense": DENSE, "baseline": TRAINED, "word_first": TRAINED}


def resolve(path):
    cfg = yaml.safe_load(path.read_text()) or {}
    if "config" in cfg:
        cfg = {**resolve(path.parent / cfg.pop("config")), **cfg}
    return cfg


def changed(name):
    big, small = resolve(QWEN8B / name), resolve(QWEN17B / name)
    return {k for k in set(big) | set(small) if big.get(k) != small.get(k)}


@pytest.mark.parametrize("name", sorted(p.name for p in QWEN8B.glob("*.yaml")))
def test_each_8b_config_is_its_17b_namesake_on_the_8b_model(name):
    assert (QWEN17B / name).is_file(), name
    dense = name == "dense_niah_base.yaml" or name.endswith("_dense.yaml")
    assert changed(name) == (DENSE if dense else TRAINED), name
    cfg = resolve(QWEN8B / name)
    leftovers = [f"{k}={v}" for k, v in cfg.items() if "1.7b" in str(v).lower()]
    assert not leftovers, leftovers


@pytest.mark.parametrize("method", sorted(METHODS))
def test_full_evaluations_read_the_8b_checkpoint_and_write_apart(method):
    cfg = resolve(QWEN8B / f"eval_full_{method}.yaml")
    # eval.sh evaluates save_path (a training YAML) or checkpoint_dir, and hands it to RULER as `pretrained`.
    evaluated = cfg.get("save_path") or cfg.get("checkpoint_dir")
    assert evaluated == (BASE_8B if method == "dense" else SAVE_8B)
    if method == "dense":
        assert cfg["log_kv_dense_mode"] is True and cfg["metadata"]["pretrained"] == BASE_8B
        assert cfg["output_path"] != resolve(QWEN17B / "eval_full_dense.yaml")["output_path"]
    else:
        assert cfg["arch_name"] == "Qwen/Qwen3-8B-Base" and cfg["ckpt_dir"] == BASE_8B
        assert cfg["eval_tag"] == f"full_{method}" and cfg.get("output_path") is None
    assert cfg["eval_main"] is True and cfg["long_limit"] == 100 and cfg.get("limit") is None


def test_baseline_and_word_first_share_the_checkpoint_and_differ_by_the_admission_score():
    baseline, word_first = (resolve(QWEN8B / f"eval_full_{m}.yaml") for m in ("baseline", "word_first"))
    diff = {k for k in set(baseline) | set(word_first) if baseline.get(k) != word_first.get(k)}
    assert diff == {"log_kv_gamma_surprisal", "eval_tag"}
    assert word_first["log_kv_gamma_surprisal"] == "word_first" and baseline.get("log_kv_gamma_surprisal") is None
