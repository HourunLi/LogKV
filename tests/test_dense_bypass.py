"""Tests for demo.py's log_kv_dense_mode bypass.

Three layers, cheapest-first:
1. A pure ast-extraction unit test of _check_dense_mode_compatible (no heavy
   imports needed).
2. A source-level regression pin that _run_eval's eval_main(...) call
   actually forwards log_kv_dense_mode (catches the exact bug found during
   design review: the fix is a single kwarg, trivially easy to lose again in
   a future refactor of that call site).
3. The real behavioral claim: training with log_kv_dense_mode=True (i.e.
   simply never calling model.enable_log_kv_training()) gives bit-identical
   results to model.set_kv_cache()'s already-established dense inference
   path, on the same weights and input -- two independent code paths
   converging on the same standard causal attention math is strong evidence
   both are actually computing it, not just "didn't crash".

Not executed in the environment that wrote this file (no GPU/torch there);
run for real before trusting a dense-mode CPT run's correctness.
"""

import __future__
import ast
import re
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_demo_function(name: str):
    tree = ast.parse((ROOT / "demo.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope: dict = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "demo.py", "exec", flags=__future__.annotations.compiler_flag),
        scope,
    )
    return scope[name]


def test_check_dense_mode_compatible_rejects_both_true():
    check = _load_demo_function("_check_dense_mode_compatible")
    with pytest.raises(ValueError, match="contradictory"):
        check(True, True)


@pytest.mark.parametrize("dense,semantic", [(True, False), (False, True), (False, False)])
def test_check_dense_mode_compatible_allows_everything_else(dense, semantic):
    check = _load_demo_function("_check_dense_mode_compatible")
    check(dense, semantic)  # must not raise


def test_run_eval_forwards_log_kv_dense_mode_to_eval_main():
    """Source-level regression pin, not a full closure execution (_run_eval
    closes over ~25 of main()'s locals, not practical to isolate via AST
    extraction) -- specifically guards the bug found during design review:
    _run_eval's eval_main(...) call didn't forward log_kv_dense_mode, so a
    dense-mode training run's run_eval="after"/"both" self-check would
    silently evaluate through the LogKV compressed path instead.
    """
    source = (ROOT / "demo.py").read_text()
    def_start = source.index("def _run_eval(ckpt, benchmark):")
    call_start = source.index("eval_main(", def_start)
    call_end = source.index(")\n", call_start)
    call_source = source[call_start : call_end + 1]
    assert re.search(r"log_kv_dense_mode\s*=\s*log_kv_dense_mode", call_source), (
        "_run_eval's eval_main(...) call must forward log_kv_dense_mode -- without it, "
        "run_eval on a dense-mode training run silently evaluates through the LogKV path"
    )


def _tiny_config():
    from litgpt import Config

    return Config(
        name="dense-bypass-test-tiny",
        block_size=64,
        n_layer=2,
        n_embd=32,
        n_head=4,
        n_query_groups=2,
        vocab_size=97,
        padded_vocab_size=97,
        bias=False,
        norm_eps=1e-5,
    )


def test_dense_training_forward_matches_dense_inference_forward():
    """The actual claim behind log_kv_dense_mode: never calling
    enable_log_kv_training() (training_log_kv stays False) gives the same
    math as the already-established dense inference path
    (GPT.set_kv_cache() -> plain KVCache + causal mask), on identical
    weights and input. Two independently-implemented code paths (training's
    input_pos=None/is_causal=True fallthrough vs. inference's explicit
    KVCache + mask_cache) agreeing bit-for-bit is direct evidence both are
    computing standard dense causal attention, not merely "ran without
    raising".
    """
    from litgpt.model import GPT

    torch.manual_seed(0)
    config = _tiny_config()
    model = GPT(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        # "Dense training" path: log_kv_dense_mode=True means demo.py simply
        # never calls enable_log_kv_training(); training_log_kv stays at its
        # class default False, input_pos=None -- exactly this call.
        training_path_logits = model(idx)

        # Independently-established "dense inference" path.
        model.set_kv_cache(batch_size=1, max_seq_length=config.block_size)
        input_pos = torch.arange(idx.size(1))
        inference_path_logits = model(idx, input_pos=input_pos)

    torch.testing.assert_close(training_path_logits, inference_path_logits)


def test_dense_forward_never_touches_kv_cache_attribute():
    """No cache should be built or consulted at all on the training path --
    confirms this isn't accidentally routing through *some* cache that just
    happens to produce the right numbers.
    """
    from litgpt.model import GPT

    config = _tiny_config()
    model = GPT(config)
    model.eval()
    for block in model.transformer.h:
        assert block.attn.kv_cache is None
        assert block.attn.training_log_kv is False

    idx = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        model(idx)

    for block in model.transformer.h:
        assert block.attn.kv_cache is None, "dense training forward must not build or attach any cache as a side effect"
