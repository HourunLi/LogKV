"""Triton kernels on CPU through the interpreter (no GPU needed).

The interpreter must be enabled before Triton decorates the kernels, so each
check runs in its own process.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("triton")


def _run(*args, timeout=3000):
    env = {**os.environ, "TRITON_INTERPRET": "1", "LOGKV_TRITON_CPU": "1",
           "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def test_route_kernels_match_torch_reference():
    result = _run(str(ROOT / "tests" / "interpret_route_check.py"))
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
