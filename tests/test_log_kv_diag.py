"""Tests for the LogKV score/value oracle diagnostic (``litgpt.log_kv_diag``).

Priority order (most load-bearing first):
  1. exact == dense              — the whole grid is trustworthy only if the
     exact route reproduces an independent dense-attention reference.
  2. baseline == production      — the control must reproduce the production
     path bit-for-bit, else the oracle "improvements" measure the harness.
  3. half-oracles sit below baseline — removing one error term must not raise
     error (coarse sanity for s_oracle / v_oracle).
  4. stats collected per level   — width-1 slots are exact (var==0, mae≈0);
     width>1 slots carry non-zero intra-slot logit variance (D3 pipeline check).
  5. zero-intrusion end to end   — a full prefill under diag_mode("baseline")
     is bit-identical to one that never touches the diagnostic at all.
"""

import pytest
import torch

from litgpt.config import Config
from litgpt.log_kv_cache import (
    CacheAttentionState,
    LogStructuredKVCache,
    append_exact_tokens,
    log_kv_slot_attention,
)
from litgpt.log_kv_diag import DIAG, diag_block_attention, diag_mode, slot_runs
from litgpt.model import CausalSelfAttention


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

N_HEAD = 4
N_GROUPS = 2
K_DIM = 8
V_DIM = 8
SCALE = 0.35


def _build_case(n_prefix: int, blk: int = 6, recent: int = 4, B: int = 4, seed: int = 0):
    """A consistent (slots cover exactly k_prefix) fresh-prefill block scenario.

    Streams ``n_prefix`` tokens into a real LogStructuredKVCache in 2-token
    chunks (the training/inference cadence) so ``get_attention_state()`` returns
    slots that cover exactly ``k[:, :, :n_prefix]``. Returns everything
    ``diag_block_attention`` needs plus the frozen slot state.
    """
    torch.manual_seed(seed)
    max_seq = 512
    cache = LogStructuredKVCache(
        (1, N_GROUPS, max_seq, K_DIM), (1, N_GROUPS, max_seq, V_DIM),
        B=B, recent_size=recent, device=torch.device("cpu"), dtype=torch.float32,
    )
    total = n_prefix + blk
    k = torch.randn(1, N_GROUPS, total, K_DIM)
    v = torch.randn(1, N_GROUPS, total, V_DIM)
    q = torch.randn(1, N_HEAD, blk, K_DIM)

    for s in range(0, n_prefix, 2):
        e = min(s + 2, n_prefix)
        cache.add_recent(k[:, :, s:e, :], v[:, :, s:e, :])
    assert cache.token_count == n_prefix

    state = cache.get_attention_state()
    return {
        "q": q,
        "k_prefix": k[:, :, :n_prefix, :],
        "v_prefix": v[:, :, :n_prefix, :],
        "slot_k": state.slot_k,
        "slot_v": state.slot_v,
        "slot_w": state.slot_w,
        "slot_valid": state.slot_valid,
        "k_tail": k[:, :, n_prefix:total, :],
        "v_tail": v[:, :, n_prefix:total, :],
    }


def _run(mode: str, case: dict, q_chunk: int = 64) -> torch.Tensor:
    with diag_mode(mode, q_chunk=q_chunk):
        return diag_block_attention(
            case["q"], case["k_prefix"], case["v_prefix"],
            case["slot_k"], case["slot_v"], case["slot_w"],
            case["k_tail"], case["v_tail"],
            scale=SCALE, lam=1.0, layer=0, slot_valid=case["slot_valid"],
        )


# ---------------------------------------------------------------------------
# 1. exact == dense  (the load-bearing invariant)
# ---------------------------------------------------------------------------

class TestExactEqualsDense:
    # n_prefix chosen to land at different hierarchy depths (recent=4, B=4):
    # 8 -> level 0 partly filled; 24 -> level 1 present; 40 -> deeper carries.
    @pytest.mark.parametrize("n_prefix", [8, 24, 40])
    def test_exact_equals_dense(self, n_prefix):
        case = _build_case(n_prefix)
        out_exact = _run("exact", case)
        out_dense = _run("dense", case)
        assert torch.allclose(out_exact, out_dense, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("n_prefix", [8, 24, 40])
    def test_exact_equals_dense_q_chunked(self, n_prefix):
        """q_chunk must not change the result (per-query rows are independent)."""
        case = _build_case(n_prefix)
        out_full = _run("exact", case, q_chunk=64)
        out_chunked = _run("exact", case, q_chunk=2)
        assert torch.allclose(out_full, out_chunked, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. baseline == production  (bit-for-bit)
# ---------------------------------------------------------------------------

class TestBaselineMatchesProduction:
    @pytest.mark.parametrize("n_prefix", [8, 24, 40])
    def test_baseline_matches_production(self, n_prefix):
        case = _build_case(n_prefix)
        out_base = _run("baseline", case)

        state = append_exact_tokens(
            CacheAttentionState(
                case["slot_k"], case["slot_v"], case["slot_w"], slot_valid=case["slot_valid"],
            ),
            case["k_tail"],
            case["v_tail"],
        )
        out_prod = log_kv_slot_attention(
            case["q"],
            state.slot_k,
            state.slot_v,
            state.slot_w,
            scale=SCALE,
            causal_tail=case["k_tail"].size(2),
            slot_valid=state.slot_valid,
        )
        assert torch.equal(out_base, out_prod)


# ---------------------------------------------------------------------------
# 3. half-oracles sit strictly below baseline error
# ---------------------------------------------------------------------------

class TestHalfOraclesSitBetween:
    @pytest.mark.parametrize("n_prefix", [24, 40])
    def test_half_oracles_reduce_error(self, n_prefix):
        # Coarse sanity: each half-oracle removes one error term, so its error
        # against the dense reference must fall below baseline. Averaged over a
        # few draws (heavy compression, recent=4) so the strict inequality is
        # not at the mercy of a single random block.
        e_base = e_s = e_v = 0.0
        for seed in range(4):
            case = _build_case(n_prefix, blk=8, seed=seed)
            ref = _run("dense", case)

            def err(mode: str) -> float:
                return (_run(mode, case) - ref).abs().mean().item()

            e_base += err("baseline")
            e_s += err("s_oracle")
            e_v += err("v_oracle")

        assert e_s < e_base, f"s_oracle {e_s} !< baseline {e_base}"
        assert e_v < e_base, f"v_oracle {e_v} !< baseline {e_base}"


# ---------------------------------------------------------------------------
# 4. stats collected per level (D3 pipeline is not all-zero)
# ---------------------------------------------------------------------------

class TestStatsPerLevel:
    def test_stats_are_collected_per_level(self):
        case = _build_case(n_prefix=24)  # yields both width-1 and width>1 runs
        with diag_mode("s_oracle") as st:
            diag_block_attention(
                case["q"], case["k_prefix"], case["v_prefix"],
                case["slot_k"], case["slot_v"], case["slot_w"],
                case["k_tail"], case["v_tail"],
                scale=SCALE, lam=1.0, layer=0, slot_valid=case["slot_valid"],
            )
            summ = st.summary()

        by_level = {e["slot_width"]: e for e in summ["by_level"]}
        widths = set(by_level)
        assert 1 in widths, "recent window (width 1) slots missing from stats"
        assert any(w > 1 for w in widths), "no compressed (width>1) slots collected"

        for w, e in by_level.items():
            if w == 1:
                # Recent tokens are exact: single-token slots have zero intra-slot
                # variance and the approximation reproduces the logit exactly.
                assert e["intra_slot_logit_var"] == 0.0
                assert e["logit_mae"] < 1e-5
            else:
                assert e["intra_slot_logit_var"] > 0.0

        # Peakiness must have been recorded for the layer.
        assert summ["peakiness"] and summ["peakiness"][0]["r2_max"] > 0.0

    def test_slot_runs_rejects_inconsistent_widths(self):
        # Simulate a per-group-divergent width vector.
        slot_w = torch.ones(1, N_GROUPS, 5)
        slot_w[0, 1, 2] = 4.0
        with pytest.raises(ValueError, match="contiguous"):
            slot_runs(slot_w)


# ---------------------------------------------------------------------------
# 5. zero-intrusion end to end
# ---------------------------------------------------------------------------

class TestZeroIntrusion:
    @staticmethod
    def _make_attention(recent: int, prefill_block: int) -> CausalSelfAttention:
        config = Config(
            block_size=64,
            padded_vocab_size=16,
            n_layer=1,
            n_head=2,
            n_query_groups=2,
            n_embd=8,
            rotary_percentage=0.5,
        )
        attn = CausalSelfAttention(config, block_idx=0)
        attn.kv_cache = attn.build_log_kv_cache(
            batch_size=1, max_seq_length=64, B=4, recent_size=recent,
        )
        attn.log_kv_prefill_block = prefill_block
        return attn

    def test_baseline_forward_is_bit_identical_to_production(self):
        recent, prefill_block, T = 4, 8, 32  # even T + block>2 => diag branch fires
        head_size = 4  # n_embd=8 / n_head=2

        torch.manual_seed(0)
        a_plain = self._make_attention(recent, prefill_block)
        torch.manual_seed(0)
        a_diag = self._make_attention(recent, prefill_block)

        torch.manual_seed(7)
        q = torch.randn(1, 2, T, head_size)
        k = torch.randn(1, 2, T, head_size)
        v = torch.randn(1, 2, T, head_size)

        with torch.no_grad():
            y_plain = a_plain._log_kv_training_forward(
                q, k, v, B=1, T=T, reset_cache=True, defer_last_single=True
            )
        assert DIAG.mode == "off"  # untouched by the plain run
        with torch.no_grad(), diag_mode("baseline"):
            y_diag = a_diag._log_kv_training_forward(
                q, k, v, B=1, T=T, reset_cache=True, defer_last_single=True
            )

        assert torch.equal(y_plain, y_diag)
