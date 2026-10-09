"""SWA must match full local attention across prefill, ring wrap, and decode."""
import ast
from copy import deepcopy
from pathlib import Path

import torch

from litgpt.config import Config
from litgpt.model import GPT


@torch.no_grad()
def test_swa_streaming_matches_local_reference():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    for groups, softcap, window, dtype in [
        (1, None, 5, torch.float32), (2, None, 5, torch.float32),
        (4, 3., 5, torch.float32), (2, None, 1, torch.float32),
        (2, None, 5, torch.bfloat16),
    ]:
        config = Config(block_size=40, n_layer=2, n_embd=32, n_head=4,
                        n_query_groups=groups, padded_vocab_size=64,
                        sliding_window_size=window, rotary_percentage=1.)
        config.attention_logit_softcapping = softcap
        model = GPT(config).eval().to(dtype)
        tolerance = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=.02, rtol=.02)
        tokens = torch.randint(0, 64, (2, 31))
        expected = model(tokens)  # Full matrix with the true local causal mask.
        model.set_kv_cache(batch_size=2, dtype=dtype)
        assert model.mask_cache is None  # No N x N persistent inference mask.
        storage = sum(t.numel() for b in model.transformer.h for t in b.attn.kv_cache.buffers())
        for chunks in [(31,), (12, 8, 7, 1, 1, 1, 1), (1,) * 31]:
            outputs, start = [], 0
            model.reset_kv_cache()
            for size in chunks:
                positions = torch.arange(start, start + size)
                if groups == 2:
                    positions = positions.expand(2, -1) + torch.tensor([[0], [3]])
                outputs.append(model(tokens[:, start:start + size], positions, input_pos_maxp1=start + size))
                start += size
            torch.testing.assert_close(torch.cat(outputs, dim=1), expected, **tolerance)
        # Mutating future tokens must not change earlier outputs.
        altered = tokens.clone()
        altered[:, 14:] = (altered[:, 14:] + 7) % 64
        model.reset_kv_cache()
        actual = model(altered, torch.arange(31))
        torch.testing.assert_close(actual[:, :14], expected[:, :14], **tolerance)
        assert storage == sum(t.numel() for b in model.transformer.h for t in b.attn.kv_cache.buffers())


@torch.no_grad()
def test_swa_auto_budget_and_reset():
    # Exercise the real wrapper methods without importing GPU/HF evaluation dependencies.
    source = Path(__file__).resolve().parents[1] / "eval.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "LogKVLM")
    names = {"_set_eval_cache", "_reset_eval_cache", "_cache_storage_bytes", "_report_cache_budget"}
    cls.bases, cls.decorator_list = [], []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, _is_main=lambda: False)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), scope)
    config = Config(block_size=128, n_layer=2, n_embd=32, n_head=4, n_query_groups=2, padded_vocab_size=64)
    lm = scope["LogKVLM"]()
    lm.model, lm.config, lm._device = GPT(config).eval(), config, "cpu"
    lm._eval_cache_ready, lm.cache_budget = False, {}
    lm.swa_window_size, lm.log_kv_dense_mode = 0, False
    settings = dict(B=2, recent_size=4, prefill_block=4, second_order_scale=0.,
                    importance_pooling=False, importance_pooling_lambda=1., importance_pooling_temperature=1.,
                    semantic_clusters=True, cluster_k_max=2, cluster_lambda_rel=1., seg_eta=1., seg_g0=2048.,
                    seg_gap_max=None, seg_block_level=0, seg_forget=.5, semantic_s_h=None,
                    semantic_flush_granularity=4, semantic_cluster_chunk_size=0, semantic_capacity_beta=0.,
                    semantic_capacity_hard_cap_mult=0., semantic_legacy_route=False, semantic_unified_route=False,
                    semantic_anchor_mode="mid",
                    semantic_pack_backend="torch", semantic_centroid_backend="sequential",
                    semantic_replay_updates=False, semantic_merge_passes=1,
                    alpha_exact_tokens=0, alpha_span_max_tokens=64, beta_novelty=False, beta_adaptive_merge=False)
    for key, value in settings.items():
        setattr(lm, "log_kv_" + key, value)
    lm._set_eval_cache()
    assert lm.swa_window_size > 0
    assert lm.cache_budget["persistent_cache_bytes"] <= lm.cache_budget["reference_logkv_bytes"]
    assert lm.model.mask_cache is None
    reference = deepcopy(lm.model)
    reference.clear_kv_cache()
    tokens = torch.randint(0, 64, (1, 110))
    expected = reference(tokens)
    actual = lm.model(tokens, torch.arange(110))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    lm._set_eval_cache()  # Reset existing SWA; don't rebuild the LogKV reference.
    for block in lm.model.transformer.h:
        assert (block.attn.kv_cache.positions == -1).all()


if __name__ == "__main__":
    test_swa_streaming_matches_local_reference()
    test_swa_auto_budget_and_reset()
    print("SWA streaming and budget checks passed")
