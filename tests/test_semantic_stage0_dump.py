import importlib.util
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

from litgpt.config import Config
from litgpt.model import GPT

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "unused" / "semantic_stage0_dump.py"
_SPEC = importlib.util.spec_from_file_location("semantic_stage0_dump_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_assert_layers_are_recordable = _MODULE._assert_layers_are_recordable
_assert_checkpoint_loaded_cleanly = _MODULE._assert_checkpoint_loaded_cleanly


class _FakeLoadResult(NamedTuple):
    # Mirrors torch.nn.modules.module._IncompatibleKeys's shape, which is
    # what model.load_state_dict(..., strict=False) actually returns.
    missing_keys: list[str]
    unexpected_keys: list[str]


def _tiny_config(n_layer: int = 3) -> Config:
    return Config(
        block_size=16,
        padded_vocab_size=16,
        n_layer=n_layer,
        n_head=2,
        n_query_groups=2,
        n_embd=8,
        rotary_percentage=0.5,
    )


def test_valid_layers_pass() -> None:
    config = _tiny_config(n_layer=3)
    model = GPT(config)
    _assert_layers_are_recordable(model, config, {0, 2})


def test_out_of_range_layer_rejected() -> None:
    config = _tiny_config(n_layer=3)
    model = GPT(config)
    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        _assert_layers_are_recordable(model, config, {0, 99})


def test_negative_layer_rejected() -> None:
    config = _tiny_config(n_layer=3)
    model = GPT(config)
    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        _assert_layers_are_recordable(model, config, {-1})


def test_unsupported_attention_module_rejected() -> None:
    # Swap one block's CausalSelfAttention out for a stand-in that isn't it,
    # the way MultiheadLatentAttention would be: it accepts the recorder
    # attribute assignment fine but never calls record_pre_rope, so without
    # this check the dump would silently record nothing for that layer.
    config = _tiny_config(n_layer=3)
    model = GPT(config)

    class _NotCausalSelfAttention(torch.nn.Module):
        block_idx = 1

    model.transformer.h[1].attn = _NotCausalSelfAttention()

    with pytest.raises(NotImplementedError, match=r"\[1\]"):
        _assert_layers_are_recordable(model, config, {0, 1})


def test_clean_load_passes() -> None:
    _assert_checkpoint_loaded_cleanly(_FakeLoadResult([], []), allow_mismatch=False)


def test_missing_keys_hard_fail_by_default() -> None:
    result = _FakeLoadResult(["transformer.h.0.attn.qkv.weight"], [])
    with pytest.raises(RuntimeError, match="checkpoint/config mismatch"):
        _assert_checkpoint_loaded_cleanly(result, allow_mismatch=False)


def test_unexpected_keys_hard_fail_by_default() -> None:
    result = _FakeLoadResult([], ["some.stale.buffer"])
    with pytest.raises(RuntimeError, match="checkpoint/config mismatch"):
        _assert_checkpoint_loaded_cleanly(result, allow_mismatch=False)


def test_mismatch_proceeds_only_when_explicitly_allowed(capsys: pytest.CaptureFixture[str]) -> None:
    result = _FakeLoadResult(["a"], ["b"])
    _assert_checkpoint_loaded_cleanly(result, allow_mismatch=True)
    assert "WARNING" in capsys.readouterr().out
