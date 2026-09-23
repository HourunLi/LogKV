"""Shared, dependency-light YAML config resolution for calibration/analysis
tooling (``measure_cache_bytes.py``, ``calibrate_sinkwindow_w.py``, ...).

Mirrors ``demo.py``'s own ``config:`` base-inheritance chain and
scientific-notation coercion (``demo.py:403-450`` as of this writing) rather
than importing ``demo.py`` directly: ``demo.py`` pulls in ``lightning`` and
``litdata`` at module import time, which these tools don't need and
shouldn't require just to read a YAML's resolved field values -- the whole
point of this class of script is to run without a GPU or even torch
installed. If ``demo.py``'s YAML-merge semantics ever change, update both.
"""

from __future__ import annotations

import os
import re

import yaml

_SCI_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?[eE][-+]?\d+")


def load_yaml_config(yaml_path: str) -> dict:
    """Load a YAML config, recursively resolving a ``config:`` inheritance
    chain: the base is loaded first, then the child's own keys override it
    (``{**base_cfg, **cfg}``), and a ``config:`` path is resolved relative to
    the file that names it -- same semantics as ``demo.py._load_yaml_config``.
    """
    yaml_path = os.path.normpath(os.path.expanduser(yaml_path))
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"config file not found: {yaml_path!r}")
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if "config" in cfg:
        base_name = cfg.pop("config")
        base_path = os.path.join(os.path.dirname(yaml_path), base_name)
        base_cfg = load_yaml_config(base_path)
        return {**base_cfg, **cfg}
    return cfg


def coerce_yaml_sci_floats(cfg: dict) -> dict:
    """PyYAML 1.1 doesn't parse bare scientific notation ("2e-5", no decimal
    point) as float, and this repo's ``exp/*.yaml`` configs use exactly that
    style for fields like ``learning_rate``. Mirrors
    ``demo.py._coerce_yaml_sci_floats`` so a value resolves to the same type
    either loader uses.
    """
    return {
        k: float(v) if isinstance(v, str) and _SCI_FLOAT_RE.fullmatch(v) else v
        for k, v in cfg.items()
    }


def resolve_yaml(yaml_path: str) -> dict:
    """Load + inheritance-resolve + sci-float-coerce in one call."""
    return coerce_yaml_sci_floats(load_yaml_config(yaml_path))
