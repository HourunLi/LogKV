"""Experiment YAMLs resolve, and their effective Alpha/Beta/route settings are valid."""

from pathlib import Path

import pytest
import yaml

EXP = Path(__file__).resolve().parents[1] / "exp" / "qwen1.7b-32k"
# Script/CLI fallbacks for keys a YAML leaves unset (majob.sh, eval.sh, demo.py, eval.py).
DEFAULTS = dict(log_kv_alpha_exact_tokens=256, log_kv_beta_novelty=True, log_kv_beta_adaptive_merge=True,
                log_kv_semantic_unified_route=False, log_kv_semantic_anchor_mode="multi",
                log_kv_second_order_scale=1.0, log_kv_cluster_k_max=1,
                log_kv_gamma_level0_reinsert=True, log_kv_gamma_top_merge="lightest", log_kv_gamma_level_slack=2,
                log_kv_gamma_cue_tokens=64, log_kv_gamma_literals=False, log_kv_gamma_cue_guard=False,
                log_kv_gamma_query_cue=False)
LEGACY_GAMMA = dict(log_kv_gamma_level0_reinsert=False, log_kv_gamma_top_merge="fold", log_kv_gamma_level_slack=2)


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
        out["log_kv_gamma_cue_tokens"] = 0
        out["log_kv_gamma_literals"] = out["log_kv_gamma_query_cue"] = False
    if not (out["log_kv_gamma_cue_tokens"] or out["log_kv_gamma_query_cue"]):
        out["log_kv_gamma_cue_guard"] = False
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


# Eval-only literal runs re-evaluate Gamma checkpoints with the ladder they were trained with.
GAMMA = sorted(p.name for p in EXP.glob("arc_gamma_*.yaml")) + sorted(
    p.name for pattern in ("eval_gamma_literal*.yaml", "eval_gamma_cue_guard*.yaml", "eval_gamma_query*.yaml")
    for p in EXP.glob(pattern))


@pytest.mark.parametrize("name", sorted(p.name for p in EXP.glob("*.yaml")))
def test_only_gamma_experiments_and_the_hub_use_gamma(name):
    """Earlier experiments keep the ladder they were trained with; new ones inherit Gamma."""
    cfg = effective(name)
    gamma = {key: cfg[key] for key in LEGACY_GAMMA}
    # The SWA comparison harness follows the current default route by design.
    if name in GAMMA or name == "arc_semantic_fast.yaml" or name.startswith("compare_niah_"):
        assert gamma["log_kv_gamma_level0_reinsert"] is True and gamma["log_kv_gamma_top_merge"] == "lightest"
    else:
        assert gamma == LEGACY_GAMMA, name


CUE = ["arc_gamma_cue_k12_b128_2k.yaml", "arc_gamma_cue_k12_b128_p1024_2k.yaml",
       "arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml", "eval_gamma_cue.yaml",
       "arc_gamma_cue_literal_k12_b128_2k.yaml", "eval_gamma_literal.yaml", "eval_gamma_literal_p1024_slack0.yaml",
       "eval_gamma_cue_guard.yaml", "eval_gamma_cue_guard_p1024_slack0.yaml",
       "eval_gamma_cue_guard_literal_p1024_slack0.yaml", "eval_gamma_query_p1024_slack0.yaml",
       "eval_gamma_query_literal_p1024_slack0.yaml", "eval_gamma_query_literal.yaml"]
QUERY = ["eval_gamma_query_p1024_slack0.yaml", "eval_gamma_query_literal_p1024_slack0.yaml",
         "eval_gamma_query_literal.yaml"]
LITERAL = ["arc_gamma_cue_literal_k12_b128_2k.yaml", "eval_gamma_literal.yaml", "eval_gamma_literal_nocue.yaml",
           "eval_gamma_literal_p1024_slack0.yaml", "eval_gamma_cue_guard_literal_p1024_slack0.yaml",
           "eval_gamma_query_literal_p1024_slack0.yaml", "eval_gamma_query_literal.yaml"]
GUARD = ["eval_gamma_cue_guard.yaml", "eval_gamma_cue_guard_p1024_slack0.yaml",
         "eval_gamma_cue_guard_literal_p1024_slack0.yaml", *QUERY]


@pytest.mark.parametrize("name", sorted(p.name for p in EXP.glob("*.yaml")))
def test_only_cue_experiments_and_the_hub_use_cues(name):
    """Earlier experiments, the Gamma ladder runs included, keep the span scores they were trained with."""
    cfg = effective(name)
    if name in CUE or name == "arc_semantic_fast.yaml" or name.startswith("compare_niah_"):
        assert cfg["log_kv_gamma_cue_tokens"] == 64, name
    else:
        assert cfg["log_kv_gamma_cue_tokens"] == 0, name


@pytest.mark.parametrize("name", sorted(p.name for p in EXP.glob("*.yaml")))
def test_only_literal_experiments_use_literals(name):
    assert effective(name)["log_kv_gamma_literals"] is (name in LITERAL), name


@pytest.mark.parametrize("name", sorted(p.name for p in EXP.glob("*.yaml")))
def test_only_cue_guard_experiments_use_the_guard(name):
    assert effective(name)["log_kv_gamma_cue_guard"] is (name in GUARD), name


@pytest.mark.parametrize("name", sorted(p.name for p in EXP.glob("*.yaml")))
def test_only_query_cue_evaluations_see_the_question(name):
    assert effective(name)["log_kv_gamma_query_cue"] is (name in QUERY), name


@pytest.mark.parametrize("name, trained, switches", [
    ("eval_gamma_cue_guard.yaml", "arc_gamma_cue_k12_b128_2k.yaml", {"log_kv_gamma_cue_guard"}),
    ("eval_gamma_cue_guard_p1024_slack0.yaml", "arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml",
     {"log_kv_gamma_cue_guard"}),
    ("eval_gamma_cue_guard_literal_p1024_slack0.yaml", "arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml",
     {"log_kv_gamma_cue_guard", "log_kv_gamma_literals"}),
    ("eval_gamma_query_p1024_slack0.yaml", "arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml",
     {"log_kv_gamma_cue_guard", "log_kv_gamma_query_cue"}),
    ("eval_gamma_query_literal_p1024_slack0.yaml", "arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml",
     {"log_kv_gamma_cue_guard", "log_kv_gamma_query_cue", "log_kv_gamma_literals"}),
    ("eval_gamma_query_literal.yaml", "arc_gamma_cue_k12_b128_2k.yaml",
     {"log_kv_gamma_cue_guard", "log_kv_gamma_query_cue", "log_kv_gamma_literals"}),
])
def test_cue_guard_evaluations_reuse_their_checkpoint(name, trained, switches):
    evaluated, base = effective(name), effective(trained)
    assert {k for k in set(evaluated) | set(base) if evaluated.get(k) != base.get(k)} == switches


def test_literal_experiments():
    # Eval-only: each trained checkpoint with its own ladder and cue setting, literals on.
    for name, trained in (("eval_gamma_literal.yaml", "arc_gamma_cue_k12_b128_2k.yaml"),
                          ("eval_gamma_literal_nocue.yaml", "arc_gamma_k12_b128_2k.yaml"),
                          ("eval_gamma_literal_p1024_slack0.yaml", "arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml")):
        evaluated, base = effective(name), effective(trained)
        assert {k for k in set(evaluated) | set(base) if evaluated.get(k) != base.get(k)} == {"log_kv_gamma_literals"}
    assert effective("eval_gamma_literal_nocue.yaml")["log_kv_gamma_cue_tokens"] == 0


def test_cue_experiments():
    run = effective("arc_gamma_cue_k12_b128_2k.yaml")
    ladder = effective("arc_gamma_k12_b128_2k.yaml")
    # The training run differs from the Gamma ladder run only by the cue switch and its names.
    names = ("expid", "save_path", "merged_state_staging_dir", "tensorboard_root", "log_kv_gamma_cue_tokens")
    assert {k: v for k, v in run.items() if k not in names} == {k: v for k, v in ladder.items() if k not in names}
    assert run["expid"] == "qwen1.7b-32k-gamma_cue_k12_b128_2k"
    # Eval-only: the trained Alpha/Beta checkpoint, its ladder, and cues on.
    evaluated, trained = effective("eval_gamma_cue.yaml"), effective("arc_attach_alpha_beta_k12_b128_2k.yaml")
    assert evaluated["save_path"] == trained["save_path"]
    assert {k: evaluated[k] for k in LEGACY_GAMMA} == LEGACY_GAMMA
    assert evaluated["benchmark"] == "niah_single_2,niah_single_3"


@pytest.mark.parametrize("name, exact, slack, budget", [
    ("arc_gamma_k12_b128_2k.yaml", 256, 2, 128),
    ("arc_gamma_k12_b256_2k.yaml", 256, 2, 256),
    ("arc_gamma_k12_b128_p1024_2k.yaml", 1024, 2, 128),
    ("arc_gamma_k12_b128_slack0_2k.yaml", 256, 0, 128),
    ("arc_gamma_cue_k12_b128_p1024_2k.yaml", 1024, 2, 128),
    ("arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml", 1024, 0, 128),
])
def test_gamma_experiments(name, exact, slack, budget):
    cfg = effective(name)
    assert cfg["log_kv_semantic_unified_route"] is False and int(cfg["log_kv_cluster_k_max"]) == 12
    assert cfg["log_kv_beta_novelty"] is True and cfg["log_kv_beta_adaptive_merge"] is True
    assert cfg["log_kv_alpha_exact_tokens"] == exact and cfg["log_kv_gamma_level_slack"] == slack
    assert int(cfg["log_kv_B"]) == budget and int(cfg["log_kv_recent_size"]) == 2048
    assert cfg["expid"] == "qwen1.7b-32k-" + name[len("arc_"):-len(".yaml")]


@pytest.mark.parametrize("name, parent, switches", [
    ("arc_gamma_cue_k12_b128_p1024_2k.yaml", "arc_gamma_cue_k12_b128_2k.yaml", {"log_kv_alpha_exact_tokens"}),
    ("arc_gamma_cue_k12_b128_p1024_2k.yaml", "arc_gamma_k12_b128_p1024_2k.yaml", {"log_kv_gamma_cue_tokens"}),
    ("arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml", "arc_gamma_cue_k12_b128_p1024_2k.yaml",
     {"log_kv_gamma_level_slack"}),
    ("arc_gamma_cue_literal_k12_b128_2k.yaml", "arc_gamma_cue_k12_b128_2k.yaml", {"log_kv_gamma_literals"}),
])
def test_combined_cue_experiments_differ_from_their_parents_only_by_the_named_switches(name, parent, switches):
    names = {"expid", "save_path", "merged_state_staging_dir", "tensorboard_root"}
    run, base = effective(name), effective(parent)
    assert {k for k in set(run) | set(base) if run.get(k) != base.get(k)} - names == switches
