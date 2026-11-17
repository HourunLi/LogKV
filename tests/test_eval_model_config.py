"""Resolve real evaluation architectures without allocating Qwen weights or launching GPUs."""
import ast
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_semantic_eval_flags import _eval_arguments

from litgpt import Config

ROOT = Path(__file__).resolve().parents[1]


def config_loader():
    path = ROOT / "eval.py"
    names = {"_parse_csv_ints", "_filter_config_dict", "_normalize_training_config_dict",
             "_config_from_yaml_and_overrides", "_config_for_evaluation"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(Path=Path, Any=Any, Config=Config, yaml=yaml,
                 _CONFIG_FIELDS={field.name for field in fields(Config)},
                 _is_main=lambda: False, _eval_verbose=lambda: False)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["_config_for_evaluation"]


@pytest.mark.parametrize("folder,arch", [("qwen1.7b-32k", "Qwen/Qwen3-1.7B-Base"),
                                       ("qwen8b-32k", "Qwen/Qwen3-8B-Base")])
def test_detected_launcher_resolves_legacy_experiment_directory(tmp_path, folder, arch):
    args = _eval_arguments(ROOT / "exp" / folder / "eval_full_word_first_query_detected.yaml", "eval.sh")
    assert args[args.index("--arch_name") + 1] == arch
    checkpoint = tmp_path / f"{folder}-gamma_k12_b128_p1024_2k"
    # Reproduce the old failure, and verify both original and trailing-slash paths now work.
    load = config_loader()
    with pytest.raises(ValueError, match="not a supported config name"):
        load(str(checkpoint))
    base = Config.from_name(arch)
    for suffix in ("", "/"):
        cfg = load(str(checkpoint) + suffix, {"block_size": 33792}, arch_name=arch)
        assert (cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_query_groups, cfg.padded_vocab_size) == (
            base.n_layer, base.n_embd, base.n_head, base.n_query_groups, base.padded_vocab_size)
        assert cfg.block_size == 33792


def test_saved_architecture_remains_authoritative_and_overrides_still_apply(tmp_path):
    saved = Config(name="custom-training-model", n_layer=2, n_embd=32, n_head=4,
                   n_query_groups=2, padded_vocab_size=128, block_size=64)
    (tmp_path / "model_config.yaml").write_text(yaml.safe_dump(asdict(saved)))
    cfg = config_loader()(str(tmp_path), {"block_size": 96}, arch_name="Qwen/Qwen3-8B-Base")
    assert (cfg.name, cfg.n_layer, cfg.n_embd, cfg.n_query_groups) == (saved.name, 2, 32, 2)
    assert cfg.block_size == 96


def test_explicit_name_override_and_registered_directory_keep_working(tmp_path):
    overrides = {"name": "pythia-14m", "block_size": 96}
    load = config_loader()
    cfg = load(str(tmp_path / "experiment"), overrides, arch_name="Qwen/Qwen3-8B-Base")
    base = Config.from_name("pythia-14m")
    assert cfg.n_embd == base.n_embd and cfg.block_size == 96
    assert overrides == {"name": "pythia-14m", "block_size": 96}
    assert load(str(tmp_path / "pythia-14m") + "/").n_embd == base.n_embd


def test_unspecified_architecture_does_not_inject_a_default_model(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("checkpoint_dir: /custom-model\n")
    assert "--arch_name" not in _eval_arguments(path, "eval.sh")
