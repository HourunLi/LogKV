"""Tests for scripts/calibrate_sinkwindow_w.py.

These import the script directly (unlike demo.py/eval.py's tests, this
script deliberately has no lightning/litdata dependency -- see its own
module docstring -- so a plain import is cheap and doesn't need the
AST-extraction workaround). Not executed in the environment that wrote this
file; run for real before trusting a calibrated window_size.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_sinkwindow_w import calibrate_window_size, sink_window_structural_bytes  # noqa: E402

_ARCH = "Qwen/Qwen3-0.6B-Base"  # any registered small-ish arch works; only used for shape derivation on meta device


def test_structural_bytes_monotonically_increases_with_window_size():
    context_length = 256
    prev = sink_window_structural_bytes(_ARCH, context_length, 1, sink_size=4, window_size=1, dtype=torch.bfloat16)
    for window_size in (2, 4, 8, 16, 32, 64):
        cur = sink_window_structural_bytes(_ARCH, context_length, 1, sink_size=4, window_size=window_size, dtype=torch.bfloat16)
        assert cur > prev, f"expected strictly increasing bytes at window_size={window_size}"
        prev = cur


def test_structural_bytes_does_not_depend_on_context_length():
    """SinkWindow's whole point: storage is O(sink+window), not O(context_length)."""
    b_small_ctx = sink_window_structural_bytes(_ARCH, 512, 1, sink_size=4, window_size=32, dtype=torch.bfloat16)
    b_large_ctx = sink_window_structural_bytes(_ARCH, 32768, 1, sink_size=4, window_size=32, dtype=torch.bfloat16)
    assert b_small_ctx == b_large_ctx


def test_calibrate_window_size_finds_largest_feasible_value():
    context_length = 512
    sink_size = 4
    # Use window_size=16's own byte count as the target -- the calibrator
    # must find exactly 16 (not 15, not 17) since bytes are a simple linear
    # function of window_size for a fixed config.
    target = sink_window_structural_bytes(_ARCH, context_length, 1, sink_size, 16, torch.bfloat16)
    found = calibrate_window_size(
        target_bytes=target, arch_name=_ARCH, context_length=context_length, batch_size=1,
        sink_size=sink_size, dtype=torch.bfloat16,
    )
    assert found == 16


def test_calibrate_window_size_raises_when_even_window_size_one_is_over_budget():
    context_length = 512
    with pytest.raises(ValueError, match="already"):
        calibrate_window_size(
            target_bytes=1,  # 1 byte -- nothing at all fits
            arch_name=_ARCH, context_length=context_length, batch_size=1, sink_size=4, dtype=torch.bfloat16,
        )


def test_calibrate_window_size_respects_batch_size_scaling():
    context_length = 512
    b1 = sink_window_structural_bytes(_ARCH, context_length, 1, sink_size=4, window_size=16, dtype=torch.bfloat16)
    b2 = sink_window_structural_bytes(_ARCH, context_length, 2, sink_size=4, window_size=16, dtype=torch.bfloat16)
    assert b2 == 2 * b1
