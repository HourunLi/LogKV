"""Experiment YAMLs resolve, and their effective Alpha/Beta/route settings are valid."""

from pathlib import Path

import pytest
import yaml

EXP = Path(__file__).resolve().parents[1] / "exp" / "qwen1.7b-32k"
# Script/CLI fallbacks for keys a YAML leaves unset (majob.sh, eval.sh, demo.py, eval.py).
DEFAULTS = dict(log_kv_alpha_exact_tokens=256, log_kv_beta_novelty=True, log_kv_beta_adaptive_merge=True,
                log_kv_semantic_unified_route=False, log_kv_semantic_anchor_mode="multi",
                log_kv_second_order_scale=1.0, log_kv_cluster_k_max=1)


def resolve(path):
    cfg = yaml.safe_load(path.read_text()) or {}
    if "config" in cfg:
        cfg = {**resolve(path.parent / cfg.pop("config")), **cfg}
    return cfg


def effective(name):
    cfg = resolve(EXP / name)
    out = {key: cfg[key] if cfg.get(key) is not None else value for key, value in DEFAULTS.items()}
    out.update({key: value for key, value in cfg.items() if key not in out})
    if not out.get("log_kv_semantic_clusters"):
        out["log_kv_alpha_exact_tokens"] = 0
    if not out["log_kv_alpha_exact_tokens"]:
        out["log_kv_beta_novelty"] = out["log_kv_beta_adaptive_merge"] = False
    return out


@pytest.mark.parametrize("name", sorted(p.name for p in EXP.glob("*.yaml")))
def test_config_resolves_and_alpha_settings_are_supported(name):
    cfg = effective(name)
    if cfg["log_kv_alpha_exact_tokens"]:
        assert cfg["log_kv_semantic_anchor_mode"] == "mid"
        assert int(cfg["log_kv_cluster_k_max"]) > 1
        assert float(cfg["log_kv_second_order_scale"]) == 0
        assert cfg.get("log_kv_seg_gap_max") is None and not cfg.get("log_kv_seg_block_level")
        assert not cfg.get("log_kv_semantic_legacy_route") and not cfg.get("log_kv_semantic_cluster_chunk_size")
        assert cfg.get("log_kv_semantic_replay_updates") is True


@pytest.mark.parametrize("name", ["arc_semantic_fast.yaml", "arc_attach_alpha_beta_k12_b128_2k.yaml",
                                  "arc_attach_alpha_beta_k12_b128_4k.yaml"])
def test_default_route_is_attach_with_alpha_and_beta(name):
    cfg = effective(name)
    assert cfg["log_kv_semantic_unified_route"] is False
    assert cfg["log_kv_alpha_exact_tokens"] == 256 and cfg["log_kv_alpha_span_max_tokens"] == 64
    assert cfg["log_kv_beta_novelty"] is True and cfg["log_kv_beta_adaptive_merge"] is True


@pytest.mark.parametrize("name, alpha, beta, unified", [
    ("arc_semantic_attach_route_k12_b128_2k.yaml", 0, False, False),
    ("arc_semantic_unified_route_k12_b128_2k.yaml", 0, False, True),
    ("arc_semantic_unified.yaml", 0, False, True),
    ("arc_alpha_k12_b128.yaml", 256, False, True),
    ("arc_beta_cpt100.yaml", 256, True, True),
    ("base.yaml", 0, False, False),
])
def test_earlier_experiments_keep_their_trained_settings(name, alpha, beta, unified):
    cfg = effective(name)
    assert cfg["log_kv_alpha_exact_tokens"] == alpha
    assert cfg["log_kv_beta_novelty"] is beta and cfg["log_kv_beta_adaptive_merge"] is beta
    assert bool(cfg["log_kv_semantic_unified_route"]) is unified
