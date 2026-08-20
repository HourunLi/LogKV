import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "unused" / "semantic_s0_sweep.py"
_SPEC = importlib.util.spec_from_file_location("semantic_s0_sweep_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_effective_g_max = _MODULE._effective_g_max
_manifest_sh = _MODULE._manifest_sh
_manifest_vh = _MODULE._manifest_vh

from litgpt.semantic_s0 import route_dpmeans_segments  # noqa: E402


def test_effective_g_max_forces_inf_at_l_block_zero() -> None:
    for g_max in (256, 1024, 8192, math.inf):
        assert _effective_g_max(g_max, 0) == math.inf
    for l_block in (1, 2, 3):
        assert _effective_g_max(256, l_block) == 256
        assert _effective_g_max(math.inf, l_block) == math.inf


def test_l_block_zero_reproduces_true_pure_semantic_routing() -> None:
    # Regression test for the divergence this was written to fix: with eta=0
    # fixed (route_dpmeans_segments has no eta parameter -- structural) and
    # identical data, a finite g_max's gamma-decay still shifts centroid
    # trajectories relative to g_max=inf, which can flip cluster assignment.
    k = np.asarray([[0.0], [0.9], [1.7]], dtype=np.float32)

    route_true_pure_semantic = route_dpmeans_segments(k, lambda_new=1.0, g_max=math.inf, gamma=0.0)
    assert route_true_pure_semantic.cluster_ids.tolist() == [0, 0, 1]

    # Without the fix, l_block=0 would route with the literal (finite) swept
    # g_max and get a *different* answer purely from gamma-decay:
    route_finite_g_max = route_dpmeans_segments(k, lambda_new=1.0, g_max=0, gamma=0.0)
    assert route_finite_g_max.cluster_ids.tolist() == [0, 0, 0]
    assert route_finite_g_max.cluster_ids.tolist() != route_true_pure_semantic.cluster_ids.tolist()

    # With the fix, routing at l_block=0 always uses effective_g_max=inf, so it
    # matches the true pure-semantic reference regardless of the nominal g_max
    # being swept:
    for nominal_g_max in (0, 1, 256, 1024):
        effective = _effective_g_max(nominal_g_max, l_block=0)
        route = route_dpmeans_segments(k, lambda_new=1.0, g_max=effective, gamma=0.0)
        assert route.cluster_ids.tolist() == route_true_pure_semantic.cluster_ids.tolist()


def test_manifest_sh_hard_fails_by_default_when_uncalibrated() -> None:
    manifest = {"key_scale": {}}
    k = np.asarray([[0.0], [1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="no usable calibrated key_scale"):
        _manifest_sh(manifest, layer=0, group=0, k=k, allow_fallback=False)


def test_manifest_sh_falls_back_only_when_explicitly_allowed() -> None:
    manifest = {"key_scale": {}}
    k = np.asarray([[0.0], [2.0]], dtype=np.float32)
    value, source = _manifest_sh(manifest, layer=0, group=0, k=k, allow_fallback=True)
    assert source == "fallback_local"
    assert value > 0


def test_manifest_sh_uses_calibrated_value_when_present() -> None:
    manifest = {"key_scale": {"0": {"s_h": [5.0]}}}
    k = np.asarray([[0.0], [2.0]], dtype=np.float32)
    value, source = _manifest_sh(manifest, layer=0, group=0, k=k, allow_fallback=False)
    assert source == "calibrated"
    assert value == 5.0


def test_manifest_vh_reads_the_independent_value_scale_field_not_key_scale() -> None:
    # key_scale and value_scale are calibrated independently -- a manifest with
    # only key_scale populated must not let _manifest_vh silently borrow it.
    manifest = {"key_scale": {"0": {"s_h": [5.0]}}, "value_scale": {"0": {"s_h": [40.0]}}}
    v = np.asarray([[0.0], [2.0]], dtype=np.float32)
    value, source = _manifest_vh(manifest, layer=0, group=0, v=v, allow_fallback=False)
    assert source == "calibrated"
    assert value == 40.0


def test_manifest_vh_hard_fails_on_old_manifest_missing_value_scale_entirely() -> None:
    # Dumps written before value-scale calibration existed have no "value_scale"
    # key at all -- must be treated the same as a missing per-layer entry (hard
    # fail by default), not crash with an unrelated KeyError/TypeError.
    manifest = {"key_scale": {"0": {"s_h": [5.0]}}}
    v = np.asarray([[0.0], [2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="no usable calibrated value_scale"):
        _manifest_vh(manifest, layer=0, group=0, v=v, allow_fallback=False)
