"""Tests for LogStructuredKVCache with merged-position slots (strict O(B·log N))."""

import math

import pytest
import torch

from litgpt.config import Config
from litgpt.log_kv_cache import (
    CacheAttentionState,
    LogKVStreamTrainingAttention,
    LogStructuredKVCache,
    _pair_rank1_stats,
    _rank1_cross_from_factors,
    _rank1_psd_from_factors,
    _SemanticTreeCluster,
    append_exact_tokens,
    log_kv_chunk_attention,
    log_kv_slot_attention,
)
from litgpt.model import CausalSelfAttention, GPT, build_rope_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cache():
    """A small cache suitable for unit testing the data structure.

    k_dim=8 (full post-RoPE key width), v_dim=8
    B=4 (slots per level), 2:1 compaction
    max_seq_length=64 -> max_levels = max(2, ceil(log2(65/8))+1) = 5
    """
    batch_size = 1
    n_groups = 2
    max_seq_length = 64
    k_dim = 8
    v_dim = 8
    B = 4

    k_shape = (batch_size, n_groups, max_seq_length, k_dim)
    v_shape = (batch_size, n_groups, max_seq_length, v_dim)

    cache = LogStructuredKVCache(
        k_shape, v_shape, B=B,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    return cache


def add_full_kv_in_chunks(cache: LogStructuredKVCache, k: torch.Tensor, v: torch.Tensor, chunk_size: int = 2) -> None:
    """Commit full post-RoPE K/V tensors through the supported explicit cache API."""
    for start in range(0, k.size(2), chunk_size):
        end = min(start + chunk_size, k.size(2))
        cache.add_recent(k[:, :, start:end, :], v[:, :, start:end, :])


def level_count(cache: LogStructuredKVCache, ell: int) -> int:
    return int(cache.level_count[0, 0, 0, ell].item())


def has_compacted_level(cache: LogStructuredKVCache) -> bool:
    return bool((cache.level_count > 0).any().item())


def make_semantic_cache(
    *,
    batch_size: int = 1,
    n_groups: int = 2,
    max_seq_length: int = 64,
    k_dim: int = 8,
    v_dim: int = 8,
    B: int = 4,
    recent_size: int = 2,
    K_max: int = 1,
    cluster_lambda_rel: float = 1.0,
    seg_gap_max: float | None = None,
    seg_block_level: int = 0,
    semantic_s_h: torch.Tensor | float | None = 1.0,
    semantic_flush_granularity: int = 2,
    semantic_cluster_chunk_size: int = 0,
    semantic_capacity_beta: float = 0.0,
    semantic_capacity_hard_cap_mult: float = 0.0,
) -> LogStructuredKVCache:
    cos, sin = build_rope_cache(max_seq_length, k_dim)
    return LogStructuredKVCache(
        (batch_size, n_groups, max_seq_length, k_dim),
        (batch_size, n_groups, max_seq_length, v_dim),
        B=B,
        recent_size=recent_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
        semantic_clusters=True,
        cluster_k_max=K_max,
        cluster_lambda_rel=cluster_lambda_rel,
        seg_gap_max=seg_gap_max,
        seg_block_level=seg_block_level,
        semantic_s_h=semantic_s_h,
        semantic_flush_granularity=semantic_flush_granularity,
        semantic_cluster_chunk_size=semantic_cluster_chunk_size,
        semantic_capacity_beta=semantic_capacity_beta,
        semantic_capacity_hard_cap_mult=semantic_capacity_hard_cap_mult,
        cos_cache=cos,
        sin_cache=sin,
        rope_n_elem=k_dim,
    )


def real_state_tensors(state: CacheAttentionState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.slot_valid is None:
        return state.slot_k, state.slot_v, state.slot_w
    pooled = state.slot_valid.size(-1)
    valid0 = state.slot_valid[0, 0]
    slot_k = torch.cat([state.slot_k[:, :, :pooled, :][:, :, valid0, :], state.slot_k[:, :, pooled:, :]], dim=2)
    slot_v = torch.cat([state.slot_v[:, :, :pooled, :][:, :, valid0, :], state.slot_v[:, :, pooled:, :]], dim=2)
    slot_w = torch.cat([state.slot_w[:, :, :pooled][:, :, valid0], state.slot_w[:, :, pooled:]], dim=2)
    return slot_k, slot_v, slot_w


def assert_cache_states_bit_identical(a: LogStructuredKVCache, b: LogStructuredKVCache) -> None:
    """The full cache state (counters, recent window, all levels) must match bitwise."""
    assert a.token_count == b.token_count
    assert a.recent_count == b.recent_count
    rc = a.recent_count
    assert torch.equal(a.recent_k[:, :, :rc], b.recent_k[:, :, :rc])
    assert torch.equal(a.recent_v[:, :, :rc], b.recent_v[:, :, :rc])
    if a.semantic_clusters:
        assert torch.equal(a.recent_k_raw[:, :, :rc], b.recent_k_raw[:, :, :rc])
        assert torch.equal(a.recent_pos[:, :rc], b.recent_pos[:, :rc])
    assert torch.equal(a.level_count, b.level_count)
    for name in (
        "level_k",
        "level_v",
        "level_w",
        "level_imp",
        "level_sigma_u",
        "level_sigma2",
        "level_gamma_a",
        "level_gamma_b",
        "level_gamma",
        "pad_mask",
    ):
        assert torch.equal(getattr(a, name), getattr(b, name)), f"{name} differ"
    if a.semantic_clusters:
        for name in (
            "level_p_lo",
            "level_p_hi",
            "level_sum_wp",
            "level_order",
            "n_total",
            "p_hi_c",
            "current_segment",
            "level0_phase",
            "alive",
        ):
            assert torch.equal(getattr(a, name), getattr(b, name)), f"{name} differ"
        # centroid/n_eff go through the batched closed-form update in
        # _semantic_join_batch, which is mathematically but not bit-identical
        # to the old per-token recurrence (sub-ULP rounding that varies with
        # run size / flush granularity) -- tolerance compare like the other
        # centroid/n_eff checks in this file.
        torch.testing.assert_close(a.centroid, b.centroid, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a.n_eff, b.n_eff)


# ---------------------------------------------------------------------------
# __init__ / structure tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_basic_attributes(self, small_cache):
        c = small_cache
        assert c.B == 4
        assert c.k_dim == 8
        assert c.v_dim == 8
        assert c.max_seq_length == 64

    def test_max_levels(self, small_cache):
        """max_levels = max(2, ceil(log2((max_seq_length+1)/(B*2))) + 1)."""
        c = small_cache
        expected = max(2, math.ceil(math.log2((64 + 1) / (4 * 2))) + 1)
        assert c.max_levels == expected

    def test_max_levels_for_standard_configs(self):
        """Verify max_levels for 4K, 32K, 64K, 1M with B=1024."""
        for max_seq, expected_levels in [
            (4096, 3),
            (32768, 6),
            (65536, 7),
            (1048576, 11),
        ]:
            c = LogStructuredKVCache(
                (1, 1, max_seq, 1), (1, 1, max_seq, 1),
                B=1024,
            )
            assert c.max_levels == expected_levels, (
                f"max_seq={max_seq}: got {c.max_levels}, expected {expected_levels}"
            )

    def test_level_buffers_registered(self, small_cache):
        c = small_cache
        assert c.K_max == 1
        assert c.L_alloc == c.max_levels
        assert c.B_prime == c.B
        assert c.level_k.shape == (c.batch_size, c.n_groups, c.K_max, c.L_alloc, c.B_prime, c.k_dim)
        assert c.level_v.shape == (c.batch_size, c.n_groups, c.K_max, c.L_alloc, c.B_prime, c.v_dim)
        assert c.level_w.shape == (c.batch_size, c.n_groups, c.K_max, c.L_alloc, c.B_prime)
        assert c.level_imp.shape == c.level_w.shape
        assert c.level_sigma_u.shape == c.level_k.shape
        assert c.level_sigma2.shape == c.level_w.shape
        assert c.level_gamma_a.shape == c.level_k.shape
        assert c.level_gamma_b.shape == c.level_v.shape
        assert c.level_gamma.shape == c.level_w.shape
        assert c.level_count.shape == (c.batch_size, c.n_groups, c.K_max, c.L_alloc)
        assert c.level_count.dtype == torch.int16
        assert c.pad_mask.shape == c.level_w.shape
        assert c._counts == [0] * c.L_alloc
        assert c.level_count.sum() == 0

    def test_semantic_layout_and_cluster_metadata(self):
        c = make_semantic_cache(K_max=3, B=4)

        assert c.semantic_clusters
        assert c.K_max == 3
        assert c.level_k.shape == (1, 2, 3, c.L_alloc, 4, 8)
        assert c.level_count.shape == (1, 2, 3, c.L_alloc)
        assert c.level_count.dtype == torch.int16
        assert c.centroid.shape == (1, 2, 3, 8)
        assert c.n_eff.shape == c.n_total.shape == c.p_hi_c.shape == (1, 2, 3)
        assert c.current_segment.shape == c.level0_phase.shape == c.alive.shape == (1, 2, 3)
        assert c.recent_k.shape[-1] == c.slot_k_dim == 8
        assert c.recent_k_raw.shape[-1] == c.k_dim == 8
        assert c.cos_cache.shape == c.sin_cache.shape == (64, 8)
        assert c.op_log is None
        assert c.op_log_len is None

    def test_initial_state_empty(self, small_cache):
        c = small_cache
        assert c.token_count == 0
        assert c.recent_count == 0
        assert level_count(c, 0) == 0
        assert c.total_slots == 0
        assert c.total_tokens_covered == 0

    def test_buffers_non_persistent(self, small_cache):
        """All cache buffers should be non-persistent (not in state_dict)."""
        sd = small_cache.state_dict()
        assert len(sd) == 0, f"Unexpected persistent keys: {list(sd.keys())}"

    def test_memory_is_logarithmic_in_max_seq_length(self):
        """THE core complexity guarantee: total buffer storage must be
        O(recent_size + B * max_levels) — no term linear in max_seq_length.
        Closed form: recent_size*(k_dim+v_dim)*b*g
                     + max_levels * (B*(3*k_dim+2*v_dim+5)*b*g + b*g).
        The "+5" scalar-per-slot group is level_w/sigma2/gamma/imp/pad_mask."""
        b, g, k_dim, v_dim, B, recent = 1, 2, 8, 8, 4, 8
        for max_seq in (1024, 65536, 1048576):
            c = LogStructuredKVCache(
                (b, g, max_seq, k_dim), (b, g, max_seq, v_dim),
                B=B, recent_size=recent,
            )
            total = sum(buf.numel() for buf in c.buffers())
            expected = (
                recent * (k_dim + v_dim) * b * g
                + c.max_levels * (B * (3 * k_dim + 2 * v_dim + 5) * b * g + b * g)
            )
            assert total == expected, (
                f"max_seq={max_seq}: buffer numel {total} != log-sized {expected} — "
                "an O(N) buffer has crept back in"
            )


# ---------------------------------------------------------------------------
# _compact_tokens tests
# ---------------------------------------------------------------------------

class TestCompactTokens:
    def test_mean_pooling(self):
        """_compact_tokens should mean-pool keys and values."""
        B, G, n, D = 1, 1, 4, 3
        k = torch.arange(n * D, dtype=torch.float32).reshape(B, G, n, D)
        v = torch.arange(n * D, dtype=torch.float32).reshape(B, G, n, D) * 10

        k_entry, v_entry, w_entry = LogStructuredKVCache._compact_tokens(k, v)

        assert k_entry.shape == (B, G, 1, D)
        assert v_entry.shape == (B, G, 1, D)
        assert w_entry.shape == (B, G, 1)
        torch.testing.assert_close(k_entry[0, 0, 0], k[0, 0].mean(dim=0))
        torch.testing.assert_close(v_entry[0, 0, 0], v[0, 0].mean(dim=0))
        assert w_entry[0, 0, 0].item() == float(n)

    def test_partial_chunk(self):
        """_compact_tokens with fewer tokens than expected should use actual count."""
        B, G, D = 1, 1, 3
        k = torch.randn(B, G, 2, D)
        v = torch.randn(B, G, 2, D)

        k_entry, v_entry, w_entry = LogStructuredKVCache._compact_tokens(k, v)

        assert w_entry[0, 0, 0].item() == 2.0
        torch.testing.assert_close(k_entry[0, 0, 0], k[0, 0].mean(dim=0))

    def test_pair_rank1_stats_are_exact(self):
        """A two-token slot has rank-1 key covariance and value-key covariance."""
        B, G, k_dim, v_dim = 1, 1, 4, 3
        k = torch.randn(B, G, 2, k_dim)
        v = torch.randn(B, G, 2, v_dim)

        (
            _k_entry,
            _v_entry,
            _w_entry,
            sigma_u,
            sigma2,
            gamma_a,
            gamma_b,
            gamma,
        ) = LogStructuredKVCache._compact_tokens(k, v, with_stats=True)

        k_c = k - k.mean(dim=2, keepdim=True)
        v_c = v - v.mean(dim=2, keepdim=True)
        cov = torch.einsum("bgnd,bgne->bgde", k_c, k_c) / 2.0
        cross = torch.einsum("bgnc,bgnd->bgcd", v_c, k_c) / 2.0

        cov_rank1 = sigma2[..., 0, None, None] * torch.einsum(
            "bgd,bge->bgde", sigma_u[..., 0, :], sigma_u[..., 0, :]
        )
        cross_rank1 = gamma[..., 0, None, None] * torch.einsum(
            "bgc,bgd->bgcd", gamma_b[..., 0, :], gamma_a[..., 0, :]
        )
        torch.testing.assert_close(cov_rank1, cov, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cross_rank1, cross, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# compact (merge two B-slot blocks) tests
# ---------------------------------------------------------------------------

class TestCompact:
    def test_merge_weighted_average(self):
        """compact should produce weighted average of adjacent pairs.

        Pairing is over the CONCATENATED ``[k1; k2]`` sequence — (slot 2i,
        slot 2i+1) -> slot i (see ``compact()``'s docstring), NOT
        corresponding indices across k1/k2. With ``B_slots=4`` (even), every
        pair here falls entirely within k1 or entirely within k2, so the two
        blocks never actually mix: output slots 0-1 are k1's own two pairs,
        slots 2-3 are k2's own two pairs.
        """
        B_slots = 4
        B, G, D = 1, 1, 2

        k1 = torch.ones(B, G, B_slots, D)
        v1 = torch.ones(B, G, B_slots, D) * 2
        w1 = torch.full((B, G, B_slots), 4.0)

        k2 = torch.ones(B, G, B_slots, D) * 3
        v2 = torch.ones(B, G, B_slots, D) * 6
        w2 = torch.full((B, G, B_slots), 4.0)

        k_out, v_out, w_out = LogStructuredKVCache.compact(k1, v1, w1, k2, v2, w2)

        assert k_out.shape == (B, G, B_slots, D)
        assert v_out.shape == (B, G, B_slots, D)
        assert w_out.shape == (B, G, B_slots)
        # equal weights within each block -> alpha=0.5 -> each pair averages
        # to its own block's (uniform) value: k1's pairs stay 1.0, k2's stay 3.0.
        torch.testing.assert_close(
            k_out, torch.tensor([[[[1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]]]])
        )
        torch.testing.assert_close(
            v_out, torch.tensor([[[[2.0, 2.0], [2.0, 2.0], [6.0, 6.0], [6.0, 6.0]]]])
        )
        torch.testing.assert_close(w_out, torch.full_like(w_out, 8.0))

    def test_merge_unequal_weights(self):
        """compact with unequal weights should use alpha = wa/(wa+wb).

        Pairing is over the CONCATENATED ``[k1; k2]`` sequence, so slot 0
        merges k1's own two entries and slot 1 merges k2's own two entries
        (see ``compact()``'s docstring) — k1 and k2 never mix here since
        ``len(k1)`` is even.
        """
        B, G, D = 1, 1, 1

        k1 = torch.tensor([[[[1.0], [5.0]]]])  # (1,1,2,1)
        v1 = torch.tensor([[[[2.0], [6.0]]]])
        w1 = torch.tensor([[[1.0, 3.0]]])  # (1,1,2)

        k2 = torch.tensor([[[[3.0], [7.0]]]])
        v2 = torch.tensor([[[[4.0], [8.0]]]])
        w2 = torch.tensor([[[3.0, 1.0]]])

        k_out, v_out, w_out = LogStructuredKVCache.compact(k1, v1, w1, k2, v2, w2)

        # slot 0 = merge(k1[0], k1[1]): wa=1, wb=3, alpha=0.25 -> 0.25*1 + 0.75*5 = 4.0
        torch.testing.assert_close(k_out[0, 0, 0, 0], torch.tensor(4.0))
        # slot 1 = merge(k2[0], k2[1]): wa=3, wb=1, alpha=0.75 -> 0.75*3 + 0.25*7 = 4.0
        torch.testing.assert_close(k_out[0, 0, 1, 0], torch.tensor(4.0))
        torch.testing.assert_close(w_out, torch.tensor([[[4.0, 4.0]]]))

    def test_rank1_stats_merge_matches_direct_truncation(self):
        """Chan merge + rank-1 truncation should match direct truncation on raw tokens."""
        torch.manual_seed(123)
        B, G, k_dim, v_dim = 1, 1, 5, 4
        k = torch.randn(B, G, 4, k_dim)
        v = torch.randn(B, G, 4, v_dim)

        left = LogStructuredKVCache._compact_tokens(k[:, :, :2, :], v[:, :, :2, :], with_stats=True)
        right = LogStructuredKVCache._compact_tokens(k[:, :, 2:, :], v[:, :, 2:, :], with_stats=True)
        direct = LogStructuredKVCache._compact_tokens(k, v, with_stats=True)

        merged = LogStructuredKVCache.compact(
            left[0], left[1], left[2],
            right[0], right[1], right[2],
            left[3], left[4], left[5], left[6], left[7],
            right[3], right[4], right[5], right[6], right[7],
        )

        for got, exp in zip(merged[:3], direct[:3]):
            torch.testing.assert_close(got, exp, atol=1e-6, rtol=1e-6)

        got_cov = merged[4][..., None, None] * torch.einsum("bgsd,bgse->bgsde", merged[3], merged[3])
        exp_cov = direct[4][..., None, None] * torch.einsum("bgsd,bgse->bgsde", direct[3], direct[3])
        got_cross = merged[7][..., None, None] * torch.einsum("bgsc,bgsd->bgscd", merged[6], merged[5])
        exp_cross = direct[7][..., None, None] * torch.einsum("bgsc,bgsd->bgscd", direct[6], direct[5])
        torch.testing.assert_close(got_cov, exp_cov, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got_cross, exp_cross, atol=1e-5, rtol=1e-5)

    def test_anchor_fields_are_optional_and_append_when_given(self):
        """Omitting anchors reproduces the prior shape; giving them merges and appends."""
        B, G, D = 1, 1, 1
        k1 = torch.tensor([[[[1.0], [5.0]]]])
        v1 = torch.tensor([[[[2.0], [6.0]]]])
        w1 = torch.tensor([[[1.0, 3.0]]])
        k2 = torch.tensor([[[[3.0], [7.0]]]])
        v2 = torch.tensor([[[[4.0], [8.0]]]])
        w2 = torch.tensor([[[3.0, 1.0]]])

        no_anchors = LogStructuredKVCache.compact(k1, v1, w1, k2, v2, w2)
        assert len(no_anchors) == 3

        p_lo1 = torch.tensor([[[0, 10]]])
        p_hi1 = torch.tensor([[[0, 11]]])
        sum_wp1 = torch.tensor([[[0, 21]]], dtype=torch.int64)
        p_lo2 = torch.tensor([[[20, 30]]])
        p_hi2 = torch.tensor([[[20, 31]]])
        sum_wp2 = torch.tensor([[[60, 61]]], dtype=torch.int64)

        with_anchors = LogStructuredKVCache.compact(
            k1, v1, w1, k2, v2, w2,
            p_lo1=p_lo1, p_hi1=p_hi1, sum_wp1=sum_wp1,
            p_lo2=p_lo2, p_hi2=p_hi2, sum_wp2=sum_wp2,
        )
        for got, exp in zip(with_anchors[:3], no_anchors):
            torch.testing.assert_close(got, exp)
        p_lo_out, p_hi_out, sum_wp_out = with_anchors[3:]
        torch.testing.assert_close(p_lo_out, torch.tensor([[[0, 10]]]))
        torch.testing.assert_close(p_hi_out, torch.tensor([[[20, 31]]]))
        torch.testing.assert_close(sum_wp_out, torch.tensor([[[60, 82]]]))

    def test_anchor_fields_append_after_imp_and_stats(self):
        """Anchor tuple stays last regardless of the imp/rank-1-stats branches."""
        torch.manual_seed(7)
        B, G, k_dim, v_dim = 1, 1, 3, 2
        k1, k2 = torch.randn(B, G, 2, k_dim), torch.randn(B, G, 2, k_dim)
        v1, v2 = torch.randn(B, G, 2, v_dim), torch.randn(B, G, 2, v_dim)
        w1, w2 = torch.full((B, G, 2), 2.0), torch.full((B, G, 2), 2.0)
        left = LogStructuredKVCache._compact_tokens(k1, v1, with_stats=True)
        right = LogStructuredKVCache._compact_tokens(k2, v2, with_stats=True)
        p_lo = torch.zeros(B, G, 2, dtype=torch.int64)
        p_hi = torch.ones(B, G, 2, dtype=torch.int64)
        sum_wp = torch.ones(B, G, 2, dtype=torch.int64)

        out = LogStructuredKVCache.compact(
            left[0], left[1], left[2],
            right[0], right[1], right[2],
            left[3], left[4], left[5], left[6], left[7],
            right[3], right[4], right[5], right[6], right[7],
            p_lo1=p_lo, p_hi1=p_hi, sum_wp1=sum_wp,
            p_lo2=p_lo, p_hi2=p_hi, sum_wp2=sum_wp,
        )
        assert len(out) == 11  # 8 stats fields + 3 anchor fields
        torch.testing.assert_close(out[-3], torch.zeros(B, G, 2, dtype=torch.int64))
        torch.testing.assert_close(out[-2], torch.ones(B, G, 2, dtype=torch.int64))
        torch.testing.assert_close(out[-1], torch.full((B, G, 2), 2, dtype=torch.int64))

    def test_anchor_fields_require_all_or_none(self):
        B, G, D = 1, 1, 1
        k1 = v1 = torch.zeros(B, G, 2, D)
        w1 = torch.ones(B, G, 2)
        with pytest.raises(ValueError, match="anchor"):
            LogStructuredKVCache.compact(k1, v1, w1, k1, v1, w1, p_lo1=torch.zeros(B, G, 2, dtype=torch.int64))


# ---------------------------------------------------------------------------
# Importance-weighted pooling tests
#
# ``importance_pooling`` replaces the uniform pooling weight with a per-slot
# importance mass, independent of ``w`` (token count, which keeps driving the
# log(w) mass bias unchanged) -- see compact()/_compact_tokens() docstrings.
# ---------------------------------------------------------------------------

class TestImportancePooling:
    def test_pair_rank1_stats_weighted_matches_closed_form(self):
        """Weighted 2-point covariance collapses to p_a*p_b*dk dk^T exactly,
        for any (p_a, p_b) summing to 1 -- not just the unweighted 0.5/0.5."""
        B, G, k_dim, v_dim = 1, 1, 4, 3
        ka = torch.randn(B, G, k_dim)
        kb = torch.randn(B, G, k_dim)
        va = torch.randn(B, G, v_dim)
        vb = torch.randn(B, G, v_dim)
        frac_a = torch.tensor([[0.2]])
        frac_b = 1.0 - frac_a

        sigma_u, sigma2, gamma_a, gamma_b, gamma = _pair_rank1_stats(ka, kb, va, vb, frac_a, frac_b)

        m = frac_a.unsqueeze(-1) * ka + frac_b.unsqueeze(-1) * kb
        mv = frac_a.unsqueeze(-1) * va + frac_b.unsqueeze(-1) * vb
        cov = (
            frac_a.unsqueeze(-1) * torch.einsum("bgd,bge->bgde", ka - m, ka - m)
            + frac_b.unsqueeze(-1) * torch.einsum("bgd,bge->bgde", kb - m, kb - m)
        )
        cross = (
            frac_a.unsqueeze(-1) * torch.einsum("bgc,bgd->bgcd", va - mv, ka - m)
            + frac_b.unsqueeze(-1) * torch.einsum("bgc,bgd->bgcd", vb - mv, kb - m)
        )
        cov_rank1 = sigma2[..., None, None] * torch.einsum("bgd,bge->bgde", sigma_u, sigma_u)
        cross_rank1 = gamma[..., None, None] * torch.einsum("bgc,bgd->bgcd", gamma_b, gamma_a)
        torch.testing.assert_close(cov_rank1, cov, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cross_rank1, cross, atol=1e-5, rtol=1e-5)

    def test_pair_rank1_stats_default_matches_unweighted(self):
        """Default frac_a=frac_b=0.5 must exactly reproduce the pre-existing
        (unweighted) formula -- regression safety for the default path."""
        B, G, k_dim, v_dim = 1, 1, 4, 3
        ka = torch.randn(B, G, k_dim)
        kb = torch.randn(B, G, k_dim)
        va = torch.randn(B, G, v_dim)
        vb = torch.randn(B, G, v_dim)
        out_default = _pair_rank1_stats(ka, kb, va, vb)
        out_explicit = _pair_rank1_stats(ka, kb, va, vb, 0.5, 0.5)
        for a, b in zip(out_default, out_explicit):
            torch.testing.assert_close(a, b)

    def test_compact_imp_drives_alpha_not_w(self):
        """compact() with imp1/imp2 uses importance (not w) for the pooling
        alpha, but w_total still comes from w1/w2, unaffected -- mirrors
        TestCompact.test_merge_unequal_weights with w made uninformative
        (equal) and imp carrying the skew instead."""
        k1 = torch.tensor([[[[1.0], [5.0]]]])  # (1,1,2,1)
        v1 = torch.tensor([[[[2.0], [6.0]]]])
        w1 = torch.tensor([[[10.0, 10.0]]])
        k2 = torch.tensor([[[[3.0], [7.0]]]])
        v2 = torch.tensor([[[[4.0], [8.0]]]])
        w2 = torch.tensor([[[10.0, 10.0]]])
        imp1 = torch.tensor([[[1.0, 3.0]]])
        imp2 = torch.tensor([[[3.0, 1.0]]])

        k_out, v_out, w_out, imp_out = LogStructuredKVCache.compact(
            k1, v1, w1, k2, v2, w2, imp1=imp1, imp2=imp2,
        )
        # slot 0 = merge(k1[0], k1[1]): impa=1, impb=3, alpha=0.25 -> 0.25*1+0.75*5=4.0
        torch.testing.assert_close(k_out[0, 0, 0, 0], torch.tensor(4.0))
        # slot 1 = merge(k2[0], k2[1]): impa=3, impb=1, alpha=0.75 -> 0.75*3+0.25*7=4.0
        torch.testing.assert_close(k_out[0, 0, 1, 0], torch.tensor(4.0))
        # w (token count) is unaffected by imp: still count-based 10+10=20.
        torch.testing.assert_close(w_out, torch.full_like(w_out, 20.0))
        torch.testing.assert_close(imp_out, torch.tensor([[[4.0, 4.0]]]))

    def test_compact_without_imp_unchanged_return_shape(self):
        """Omitting imp1/imp2 must reproduce the exact prior 3-tuple return."""
        B_slots = 4
        B, G, D = 1, 1, 2
        k1 = torch.ones(B, G, B_slots, D)
        v1 = torch.ones(B, G, B_slots, D) * 2
        w1 = torch.full((B, G, B_slots), 4.0)
        k2 = torch.ones(B, G, B_slots, D) * 3
        v2 = torch.ones(B, G, B_slots, D) * 6
        w2 = torch.full((B, G, B_slots), 4.0)

        out = LogStructuredKVCache.compact(k1, v1, w1, k2, v2, w2)
        assert len(out) == 3

    def test_compact_tokens_importance_weighted_mean(self):
        """_compact_tokens with imp should produce a p-weighted mean and
        return the raw importance sum as an extra trailing tensor, while w
        (count) stays exactly n regardless of imp."""
        B, G, n, D = 1, 1, 4, 3
        k = torch.randn(B, G, n, D)
        v = torch.randn(B, G, n, D)
        imp = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

        k_entry, v_entry, w_entry, imp_entry = LogStructuredKVCache._compact_tokens(k, v, imp=imp)

        p = (imp / imp.sum(dim=2, keepdim=True)).unsqueeze(-1)
        expected_k = (p * k).sum(dim=2, keepdim=True)
        expected_v = (p * v).sum(dim=2, keepdim=True)
        torch.testing.assert_close(k_entry, expected_k)
        torch.testing.assert_close(v_entry, expected_v)
        assert w_entry[0, 0, 0].item() == float(n)
        torch.testing.assert_close(imp_entry[0, 0, 0], torch.tensor(10.0))

    def test_compact_tokens_uniform_importance_matches_uniform_mean(self):
        """Equal importance weights should degenerate to the plain uniform
        mean, for n > 2 (not just the exact-rank-1 n=2 case)."""
        B, G, n, D = 1, 1, 4, 3
        k = torch.randn(B, G, n, D)
        v = torch.randn(B, G, n, D)
        imp = torch.full((B, G, n), 2.5)

        k_entry, v_entry, _w, _imp = LogStructuredKVCache._compact_tokens(k, v, imp=imp)
        torch.testing.assert_close(k_entry[0, 0, 0], k[0, 0].mean(dim=0))
        torch.testing.assert_close(v_entry[0, 0, 0], v[0, 0].mean(dim=0))

    def test_compact_tokens_weighted_stats_are_exact_for_pairs(self):
        """A weighted 2-token slot's covariance is still exactly rank-1 for
        any (skewed) imp, not just the unweighted case -- generalizes
        TestCompactTokens.test_pair_rank1_stats_are_exact."""
        torch.manual_seed(7)
        B, G, k_dim, v_dim = 1, 1, 4, 3
        k = torch.randn(B, G, 2, k_dim)
        v = torch.randn(B, G, 2, v_dim)
        imp = torch.tensor([[[1.0, 3.0]]])

        (
            _k_entry, _v_entry, _w_entry,
            sigma_u, sigma2, gamma_a, gamma_b, gamma, imp_entry,
        ) = LogStructuredKVCache._compact_tokens(k, v, with_stats=True, imp=imp)

        k_entry = (imp.unsqueeze(-1) * k).sum(dim=2, keepdim=True) / imp.sum(dim=2, keepdim=True).unsqueeze(-1)
        v_entry = (imp.unsqueeze(-1) * v).sum(dim=2, keepdim=True) / imp.sum(dim=2, keepdim=True).unsqueeze(-1)
        p = imp / imp.sum(dim=2, keepdim=True)
        k_c = k - k_entry
        v_c = v - v_entry
        cov = torch.einsum("bgn,bgnd,bgne->bgde", p, k_c, k_c)
        cross = torch.einsum("bgn,bgnc,bgnd->bgcd", p, v_c, k_c)

        cov_rank1 = sigma2[..., 0, None, None] * torch.einsum(
            "bgd,bge->bgde", sigma_u[..., 0, :], sigma_u[..., 0, :]
        )
        cross_rank1 = gamma[..., 0, None, None] * torch.einsum(
            "bgc,bgd->bgcd", gamma_b[..., 0, :], gamma_a[..., 0, :]
        )
        torch.testing.assert_close(cov_rank1, cov, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cross_rank1, cross, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(imp_entry[0, 0, 0], torch.tensor(4.0))

    def test_compact_tokens_uniform_importance_stats_match_unweighted(self):
        """Constant imp over n>2 tokens should exactly reproduce the
        unweighted (imp=None) rank-1 stats path -- both reduce to the same
        inv_sqrt_n scaling, so this is a strong consistency check that
        doesn't depend on rank-1 truncation being exact for n>2 (it generally
        isn't -- see the module's D1 self-check discussion)."""
        torch.manual_seed(11)
        B, G, n, k_dim, v_dim = 1, 1, 4, 5, 3
        k = torch.randn(B, G, n, k_dim)
        v = torch.randn(B, G, n, v_dim)
        imp = torch.full((B, G, n), 7.0)

        unweighted = LogStructuredKVCache._compact_tokens(k, v, with_stats=True)
        weighted = LogStructuredKVCache._compact_tokens(k, v, with_stats=True, imp=imp)

        for got, exp in zip(weighted[:8], unweighted):
            torch.testing.assert_close(got, exp, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(weighted[8], torch.full((B, G, 1), 28.0))

    def test_ingest_chunk_default_off_matches_uniform(self):
        """importance_pooling=False (default) must be bit-identical to the
        pre-existing uniform-pooling behavior, and level_imp stays unused."""
        c = LogStructuredKVCache(
            (1, 1, 64, 8), (1, 1, 64, 8), B=4, device=torch.device("cpu"), dtype=torch.float32,
        )
        assert c.importance_pooling is False
        k = torch.randn(1, 1, 2, 8)
        v = torch.randn(1, 1, 2, 8)
        c.ingest_chunk(k, v)
        torch.testing.assert_close(c.level_k[:, :, 0, 0, 0, :], k.mean(dim=2))
        assert c.level_imp[:, :, 0, 0, 0].abs().sum().item() == 0.0

    def test_ingest_chunk_importance_pooling_on(self):
        """importance_pooling=True should weight by key L2 norm, differing
        from the uniform mean whenever the two tokens' norms differ, while w
        (count) stays untouched."""
        c = LogStructuredKVCache(
            (1, 1, 64, 8), (1, 1, 64, 8), B=4, device=torch.device("cpu"), dtype=torch.float32,
            importance_pooling=True,
        )
        k = torch.stack([torch.ones(8) * 1.0, torch.ones(8) * 5.0]).view(1, 1, 2, 8)
        v = torch.randn(1, 1, 2, 8)
        c.ingest_chunk(k, v)

        imp = k.float().norm(dim=-1)  # (1,1,2)
        p = imp / imp.sum()
        expected_k = p[0, 0, 0] * k[0, 0, 0] + p[0, 0, 1] * k[0, 0, 1]
        torch.testing.assert_close(c.level_k[0, 0, 0, 0, 0], expected_k)
        assert c.level_w[0, 0, 0, 0, 0].item() == 2.0
        torch.testing.assert_close(c.level_imp[0, 0, 0, 0, 0], imp.sum())
        assert not torch.allclose(c.level_k[0, 0, 0, 0, 0], k.mean(dim=2)[0, 0])

    def test_streaming_add_recent_importance_pooling_end_to_end(self):
        """The full add_recent() streaming path (through multiple binary
        carries) with importance_pooling=True should run without error,
        preserve token/slot-count bookkeeping identically to the default
        path, and produce slot content that differs given skewed key norms."""
        B, G, k_dim, v_dim, Bslots = 1, 1, 8, 8, 4
        max_seq_length = 64
        T = 20  # several level-0 fills -> at least one binary carry with B=4

        torch.manual_seed(42)
        base = torch.randn(B, G, T, k_dim)
        # Strongly skew per-token key norms so importance-weighted pooling
        # provably diverges from uniform pooling, not just by noise.
        scale = torch.linspace(0.1, 10.0, T).view(1, 1, T, 1)
        k = base * scale
        v = torch.randn(B, G, T, v_dim)

        c_uniform = LogStructuredKVCache(
            (B, G, max_seq_length, k_dim), (B, G, max_seq_length, v_dim),
            B=Bslots, device=torch.device("cpu"), dtype=torch.float32,
        )
        c_weighted = LogStructuredKVCache(
            (B, G, max_seq_length, k_dim), (B, G, max_seq_length, v_dim),
            B=Bslots, device=torch.device("cpu"), dtype=torch.float32,
            importance_pooling=True,
        )
        add_full_kv_in_chunks(c_uniform, k, v)
        add_full_kv_in_chunks(c_weighted, k, v)

        assert c_weighted.token_count == c_uniform.token_count == T
        assert c_weighted.recent_count == c_uniform.recent_count
        # Carry control flow is a pure function of counts, not content, so
        # bookkeeping must match exactly regardless of importance weighting.
        assert torch.equal(c_weighted.level_count, c_uniform.level_count)
        assert bool((c_weighted.level_count > 0).any())  # a carry actually fired

        state_u = c_uniform.get_attention_state()
        state_w = c_weighted.get_attention_state()
        torch.testing.assert_close(state_u.slot_w, state_w.slot_w)  # w (count) identical
        assert not torch.allclose(state_u.slot_k, state_w.slot_k)  # pooled content differs


class TestImportancePoolingLambda:
    """``importance_pooling_lambda`` blends the importance-driven pooling
    share toward the uniform/count-driven share. Added after the 2026-08-14
    decisive eval showed pure importance pooling (lambda=1.0) regresses niah
    despite improving ACC/LongBench -- see CLAUDE.md 6.5."""

    def test_init_validates_lambda_range(self):
        for bad in (-0.1, 1.1, 2.0):
            with pytest.raises(ValueError):
                LogStructuredKVCache(
                    (1, 1, 64, 8), (1, 1, 64, 8), B=4,
                    device=torch.device("cpu"), dtype=torch.float32,
                    importance_pooling=True, importance_pooling_lambda=bad,
                )

    def test_init_default_lambda_is_one(self):
        c = LogStructuredKVCache(
            (1, 1, 64, 8), (1, 1, 64, 8), B=4, device=torch.device("cpu"), dtype=torch.float32,
        )
        assert c.importance_pooling_lambda == 1.0

    def test_compact_lambda_default_is_bit_identical_to_explicit_one(self):
        """Omitting imp_lambda and passing imp_lambda=1.0 explicitly must take
        the exact same code branch (>=1.0 short-circuit) -- regression safety
        for the pre-existing pure-importance behavior."""
        k1 = torch.tensor([[[[1.0], [5.0]]]])
        v1 = torch.tensor([[[[2.0], [6.0]]]])
        w1 = torch.tensor([[[10.0, 10.0]]])
        k2 = torch.tensor([[[[3.0], [7.0]]]])
        v2 = torch.tensor([[[[4.0], [8.0]]]])
        w2 = torch.tensor([[[10.0, 10.0]]])
        imp1 = torch.tensor([[[1.0, 3.0]]])
        imp2 = torch.tensor([[[3.0, 1.0]]])

        out_default = LogStructuredKVCache.compact(k1, v1, w1, k2, v2, w2, imp1=imp1, imp2=imp2)
        out_lambda1 = LogStructuredKVCache.compact(
            k1, v1, w1, k2, v2, w2, imp1=imp1, imp2=imp2, imp_lambda=1.0,
        )
        for a, b in zip(out_default, out_lambda1):
            assert torch.equal(a, b)

    def test_compact_lambda_zero_matches_count_weighted(self):
        """lambda=0.0 must numerically recover the pure count-weighted alpha
        (same as omitting imp1/imp2 entirely), even though imp is deliberately
        set to the opposite skew of w -- proves lambda=0.0 actually ignores
        imp rather than coincidentally agreeing with it."""
        k1 = torch.tensor([[[[1.0], [5.0]]]])
        v1 = torch.tensor([[[[2.0], [6.0]]]])
        w1 = torch.tensor([[[3.0, 7.0]]])
        k2 = torch.tensor([[[[3.0], [7.0]]]])
        v2 = torch.tensor([[[[4.0], [8.0]]]])
        w2 = torch.tensor([[[2.0, 9.0]]])
        imp1 = torch.tensor([[[100.0, 1.0]]])
        imp2 = torch.tensor([[[1.0, 100.0]]])

        k_uniform, v_uniform, w_uniform = LogStructuredKVCache.compact(k1, v1, w1, k2, v2, w2)
        k_out, v_out, w_out, _imp_out = LogStructuredKVCache.compact(
            k1, v1, w1, k2, v2, w2, imp1=imp1, imp2=imp2, imp_lambda=0.0,
        )
        torch.testing.assert_close(k_out, k_uniform, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v_out, v_uniform, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(w_out, w_uniform)

    def test_compact_lambda_interpolates_exactly(self):
        """For 0 < lambda < 1, alpha must equal
        lambda*alpha_imp + (1-lambda)*alpha_w exactly (closed form)."""
        k1 = torch.tensor([[[[1.0], [5.0]]]])
        w1 = torch.tensor([[[3.0, 7.0]]])
        k2 = torch.tensor([[[[3.0], [7.0]]]])
        w2 = torch.tensor([[[2.0, 9.0]]])
        v1 = torch.zeros_like(k1)
        v2 = torch.zeros_like(k2)
        imp1 = torch.tensor([[[1.0, 3.0]]])
        imp2 = torch.tensor([[[3.0, 1.0]]])
        lam = 0.3

        k_out, _v_out, _w_out, _imp_out = LogStructuredKVCache.compact(
            k1, v1, w1, k2, v2, w2, imp1=imp1, imp2=imp2, imp_lambda=lam,
        )
        alpha_imp = torch.tensor([1.0 / 4.0, 3.0 / 4.0])
        alpha_w = torch.tensor([3.0 / 10.0, 2.0 / 11.0])
        alpha = lam * alpha_imp + (1 - lam) * alpha_w
        expected_k0 = alpha[0] * 1.0 + (1 - alpha[0]) * 5.0
        expected_k1 = alpha[1] * 3.0 + (1 - alpha[1]) * 7.0
        torch.testing.assert_close(k_out[0, 0, 0, 0], expected_k0)
        torch.testing.assert_close(k_out[0, 0, 1, 0], expected_k1)

    def test_compact_tokens_lambda_zero_matches_uniform(self):
        B, G, n, D = 1, 1, 4, 3
        torch.manual_seed(3)
        k = torch.randn(B, G, n, D)
        v = torch.randn(B, G, n, D)
        imp = torch.tensor([[[1.0, 50.0, 2.0, 30.0]]])

        k_uniform, v_uniform, _w = LogStructuredKVCache._compact_tokens(k, v)
        k_out, v_out, _w2, _imp = LogStructuredKVCache._compact_tokens(
            k, v, imp=imp, imp_lambda=0.0,
        )
        torch.testing.assert_close(k_out, k_uniform, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v_out, v_uniform, atol=1e-6, rtol=1e-6)

    def test_compact_tokens_lambda_interpolates_exactly(self):
        B, G, n, D = 1, 1, 4, 2
        torch.manual_seed(5)
        k = torch.randn(B, G, n, D)
        v = torch.randn(B, G, n, D)
        imp = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        lam = 0.4

        k_out, v_out, _w, imp_entry = LogStructuredKVCache._compact_tokens(
            k, v, imp=imp, imp_lambda=lam,
        )
        p_imp = imp / imp.sum(dim=2, keepdim=True)
        p_uniform = torch.full_like(p_imp, 1.0 / n)
        p = (lam * p_imp + (1 - lam) * p_uniform).unsqueeze(-1)
        expected_k = (p * k).sum(dim=2, keepdim=True)
        expected_v = (p * v).sum(dim=2, keepdim=True)
        torch.testing.assert_close(k_out, expected_k)
        torch.testing.assert_close(v_out, expected_v)
        torch.testing.assert_close(imp_entry[0, 0, 0], torch.tensor(10.0))  # imp sum unaffected by lambda

    def test_streaming_lambda_zero_close_to_uniform_pooling(self):
        """importance_pooling=True with lambda=0.0 should be numerically close
        to plain uniform pooling end-to-end through the real streaming path
        (_flush_pairs), even though it's not bit-identical (the imp-carrying
        branch does an fp32 round-trip that the pure uniform branch skips)."""
        B, G, k_dim, v_dim, Bslots = 1, 1, 8, 8, 4
        max_seq_length = 64
        T = 8

        torch.manual_seed(21)
        base = torch.randn(B, G, T, k_dim)
        scale = torch.linspace(0.1, 10.0, T).view(1, 1, T, 1)
        k = base * scale
        v = torch.randn(B, G, T, v_dim)

        c_uniform = LogStructuredKVCache(
            (B, G, max_seq_length, k_dim), (B, G, max_seq_length, v_dim),
            B=Bslots, device=torch.device("cpu"), dtype=torch.float32,
        )
        c_lambda0 = LogStructuredKVCache(
            (B, G, max_seq_length, k_dim), (B, G, max_seq_length, v_dim),
            B=Bslots, device=torch.device("cpu"), dtype=torch.float32,
            importance_pooling=True, importance_pooling_lambda=0.0,
        )
        add_full_kv_in_chunks(c_uniform, k, v)
        add_full_kv_in_chunks(c_lambda0, k, v)

        state_u = c_uniform.get_attention_state()
        state_l0 = c_lambda0.get_attention_state()
        torch.testing.assert_close(state_u.slot_w, state_l0.slot_w)
        torch.testing.assert_close(state_u.slot_k, state_l0.slot_k, atol=1e-5, rtol=1e-5)

    def test_streaming_lambda_half_between_uniform_and_full_importance(self):
        """A partial lambda's pooled slot content should sit strictly between
        the lambda=0 (~uniform) and lambda=1 (full importance) outputs on an
        axis where the two disagree -- sanity check that the blend actually
        interpolates end-to-end, not just in the single-pair closed form."""
        B, G, k_dim, v_dim, Bslots = 1, 1, 8, 8, 4
        max_seq_length = 64
        T = 8

        torch.manual_seed(23)
        base = torch.randn(B, G, T, k_dim)
        scale = torch.linspace(0.1, 10.0, T).view(1, 1, T, 1)
        k = base * scale
        v = torch.randn(B, G, T, v_dim)

        def build(lam):
            c = LogStructuredKVCache(
                (B, G, max_seq_length, k_dim), (B, G, max_seq_length, v_dim),
                B=Bslots, device=torch.device("cpu"), dtype=torch.float32,
                importance_pooling=True, importance_pooling_lambda=lam,
            )
            add_full_kv_in_chunks(c, k, v)
            return c

        c0, chalf, c1 = build(0.0), build(0.5), build(1.0)
        k0 = c0.get_attention_state().slot_k
        khalf = chalf.get_attention_state().slot_k
        k1 = c1.get_attention_state().slot_k

        assert not torch.allclose(k0, k1)  # sanity: lambda actually matters here
        lo = torch.minimum(k0, k1)
        hi = torch.maximum(k0, k1)
        assert bool(((khalf >= lo - 1e-5) & (khalf <= hi + 1e-5)).all())


class TestImportancePoolingTemperature:
    """``importance_pooling_temperature`` reshapes the raw importance
    heuristic's dynamic range before normalization -- orthogonal to
    ``importance_pooling_lambda``, added as a second lever to test the
    "attention-sink outlier" hypothesis for the niah regression (CLAUDE.md
    6.5): compressing the heuristic's own range vs. blending toward uniform."""

    def test_init_validates_temperature_positive(self):
        for bad in (0.0, -0.1, -5.0):
            with pytest.raises(ValueError):
                LogStructuredKVCache(
                    (1, 1, 64, 8), (1, 1, 64, 8), B=4,
                    device=torch.device("cpu"), dtype=torch.float32,
                    importance_pooling=True, importance_pooling_temperature=bad,
                )

    def test_init_default_temperature_is_one(self):
        c = LogStructuredKVCache(
            (1, 1, 64, 8), (1, 1, 64, 8), B=4, device=torch.device("cpu"), dtype=torch.float32,
        )
        assert c.importance_pooling_temperature == 1.0

    def test_ingest_chunk_temperature_default_is_bit_identical(self):
        """Omitting temperature and passing 1.0 explicitly must take the same
        `!= 1.0` short-circuit -- regression safety for the pre-existing
        pure key-norm heuristic."""
        k = torch.stack([torch.ones(8) * 1.0, torch.ones(8) * 5.0]).view(1, 1, 2, 8)
        v = torch.randn(1, 1, 2, 8)

        c_default = LogStructuredKVCache(
            (1, 1, 64, 8), (1, 1, 64, 8), B=4, device=torch.device("cpu"), dtype=torch.float32,
            importance_pooling=True,
        )
        c_temp1 = LogStructuredKVCache(
            (1, 1, 64, 8), (1, 1, 64, 8), B=4, device=torch.device("cpu"), dtype=torch.float32,
            importance_pooling=True, importance_pooling_temperature=1.0,
        )
        c_default.ingest_chunk(k, v)
        c_temp1.ingest_chunk(k, v)
        assert torch.equal(c_default.level_k, c_temp1.level_k)
        assert torch.equal(c_default.level_imp, c_temp1.level_imp)

    def test_ingest_chunk_temperature_matches_manual_pow_of_key_norm(self):
        """ingest_chunk's internal imp = k.norm(dim=-1) ** temperature should
        produce the exact same pooled slot as calling _compact_tokens
        directly with that manually-computed, pre-reshaped imp tensor."""
        B, G, n, D = 1, 1, 4, 3
        torch.manual_seed(9)
        k = torch.randn(B, G, n, D)
        v = torch.randn(B, G, n, D)
        temp = 0.3
        raw_imp = k.float().norm(dim=-1)

        k_expected, v_expected, _w, _imp = LogStructuredKVCache._compact_tokens(
            k, v, imp=raw_imp ** temp,
        )

        c = LogStructuredKVCache(
            (B, G, 64, D), (B, G, 64, D), B=4, device=torch.device("cpu"), dtype=torch.float32,
            importance_pooling=True, importance_pooling_temperature=temp,
        )
        c.ingest_chunk(k, v)
        torch.testing.assert_close(c.level_k[:, :, 0, 0, 0, :], k_expected.squeeze(2))
        torch.testing.assert_close(c.level_v[:, :, 0, 0, 0, :], v_expected.squeeze(2))

    def test_temperature_below_one_moves_toward_uniform(self):
        """A skewed raw importance ratio should become less skewed (closer to
        uniform) after temperature < 1 reshaping -- direct closed-form check
        on the ratio implied by two token norms."""
        imp = torch.tensor([1.0, 9.0])  # 9x skew
        temp = 0.5
        reshaped = imp ** temp  # sqrt: [1, 3] -> 3x skew, strictly less extreme
        p_raw = imp / imp.sum()
        p_reshaped = reshaped / reshaped.sum()
        # both still favor the larger-norm token, but reshaped is closer to 0.5
        assert p_raw[1] > p_reshaped[1] > 0.5

    def test_streaming_temperature_zero_point_five_between_one_and_uniform_lambda(self):
        """temperature=0.5 should sit strictly between temperature=1.0 (raw
        heuristic) and lambda=0.0 (~uniform) on an axis where they disagree --
        sanity check end-to-end through the real streaming path."""
        B, G, k_dim, v_dim, Bslots = 1, 1, 8, 8, 4
        max_seq_length = 64
        T = 8

        torch.manual_seed(31)
        base = torch.randn(B, G, T, k_dim)
        scale = torch.linspace(0.1, 10.0, T).view(1, 1, T, 1)
        k = base * scale
        v = torch.randn(B, G, T, v_dim)

        def build(temp):
            c = LogStructuredKVCache(
                (B, G, max_seq_length, k_dim), (B, G, max_seq_length, v_dim),
                B=Bslots, device=torch.device("cpu"), dtype=torch.float32,
                importance_pooling=True, importance_pooling_temperature=temp,
            )
            add_full_kv_in_chunks(c, k, v)
            return c

        c_uniform = LogStructuredKVCache(
            (B, G, max_seq_length, k_dim), (B, G, max_seq_length, v_dim),
            B=Bslots, device=torch.device("cpu"), dtype=torch.float32,
        )
        add_full_kv_in_chunks(c_uniform, k, v)
        c_raw, c_half = build(1.0), build(0.5)

        k_uniform = c_uniform.get_attention_state().slot_k
        k_raw = c_raw.get_attention_state().slot_k
        k_half = c_half.get_attention_state().slot_k

        assert not torch.allclose(k_uniform, k_raw)  # sanity: heuristic matters here
        lo = torch.minimum(k_uniform, k_raw)
        hi = torch.maximum(k_uniform, k_raw)
        assert bool(((k_half >= lo - 1e-5) & (k_half <= hi + 1e-5)).all())


class TestIngest:
    def test_single_ingest(self, small_cache):
        """Ingesting one 2-token chunk should fill level 0 with 1 entry."""
        c = small_cache
        B, G, D, v_dim = 1, 2, 8, 8

        c.ingest_chunk(torch.randn(B, G, 2, D), torch.randn(B, G, 2, v_dim))

        assert level_count(c, 0) == 1
        assert c.token_count == 2
        assert c.recent_count == 0
        assert c.total_slots == 1  # level 0 only
        assert c.total_tokens_covered == 2

    def test_ingest_fills_accumulator_and_carries(self, small_cache):
        """Ingesting B chunks should fill level 0 and carry to level 1."""
        c = small_cache
        B_batch, G, D, v_dim = 1, 2, 8, 8

        for i in range(c.B):  # 4 ingests
            c.ingest_chunk(torch.randn(B_batch, G, 2, D), torch.randn(B_batch, G, 2, v_dim))

        # After B=4 ingests, level 0 should carry to level 1
        assert level_count(c, 0) == 0
        assert level_count(c, 1) > 0
        assert level_count(c, 2) == 0
        assert c.token_count == c.B * 2  # 4 * 2 = 8
        assert c.total_slots == c.B  # level 1 has B slots

    def test_ingest_triggers_binary_carry(self):
        """Ingesting enough chunks to fill multiple levels."""
        B_slots = 2
        max_seq = 64
        D = 4
        v_dim = 4

        c = LogStructuredKVCache(
            (1, 1, max_seq, D), (1, 1, max_seq, v_dim),
            B=B_slots,
        )

        # Fill level 0 (B=2 ingests), then fill again -> carry to level 1
        for i in range(B_slots):
            c.ingest_chunk(torch.randn(1, 1, 2, D), torch.randn(1, 1, 2, v_dim))
        assert level_count(c, 1) > 0
        assert level_count(c, 0) == 0

        for i in range(B_slots):
            c.ingest_chunk(torch.randn(1, 1, 2, D), torch.randn(1, 1, 2, v_dim))
        # Now level 1 should be cleared, level 2 should be occupied
        assert level_count(c, 1) == 0
        assert level_count(c, 2) > 0
        assert level_count(c, 0) == 0

    def test_indexed_level_storage_survives_multiple_carries(self):
        """Unified entry storage keeps counts/content correct across binary carry."""
        c = LogStructuredKVCache(
            (1, 1, 128, 1), (1, 1, 128, 1),
            B=2,
            recent_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for i in range(8):
            values = torch.tensor([2.0 * i, 2.0 * i + 1.0]).view(1, 1, 2, 1)
            c.ingest_chunk(values, values * 10.0)

        assert c.level_count.shape == (1, 1, 1, c.L_alloc)
        assert [level_count(c, ell) for ell in range(4)] == [0, 0, 0, 2]
        torch.testing.assert_close(c.level_w[0, 0, 0, 3, :2], torch.tensor([8.0, 8.0]))
        torch.testing.assert_close(c.level_k[0, 0, 0, 3, :2, 0], torch.tensor([3.5, 11.5]))
        torch.testing.assert_close(c.level_v[0, 0, 0, 3, :2, 0], torch.tensor([35.0, 115.0]))
        assert not c.pad_mask[0, 0, 0, 3, :2].any()


# ---------------------------------------------------------------------------
# disabled forward API / explicit update tests
# ---------------------------------------------------------------------------

class TestDisabledForwardApi:
    @pytest.mark.parametrize("T", (1, 3))
    def test_forward_is_disabled(self, small_cache, T):
        """Direct cache calls must fail before mutating state."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8

        k = torch.randn(B, G, T, k_dim)
        v = torch.randn(B, G, T, k_dim)
        input_pos = torch.arange(T)

        with pytest.raises(RuntimeError, match="disabled/deprecated"):
            c(input_pos, k, v)

        assert c.token_count == 0
        assert c.recent_count == 0


class TestExplicitCacheUpdates:
    def test_add_recent_compacts_into_hierarchy(self, small_cache):
        """Explicit commits should build compact levels and recent state."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        T = 10

        k = torch.randn(B, G, T, k_dim)
        v = torch.randn(B, G, T, k_dim)

        add_full_kv_in_chunks(c, k, v)

        # 10 tokens -> 5 flushes. B=4 -> 1 carry (4 entries) + 1 in level 0.
        # But last chunk (tokens 9-10) stays in recent buffer.
        # So: 4 flushes -> 1 carry to level 1. Recent has 2 tokens.
        # Tokens: [0,1] flush, [2,3] flush, [4,5] flush, [6,7] flush -> 4 compact entries
        # -> carry to level 1. [8,9] stays in recent (last chunk, not flushed).
        assert level_count(c, 1) > 0
        assert c.recent_count == 2  # last chunk stays
        assert c.token_count == T

    def test_add_recent_stores_single_token(self, small_cache):
        """Single-token explicit commits should store token in recent buffer."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8

        k = torch.randn(B, G, 1, k_dim)
        v = torch.randn(B, G, 1, k_dim)

        add_full_kv_in_chunks(c, k, v, chunk_size=1)

        assert c.recent_count == 1
        assert c.token_count == 1

    def test_add_recent_after_prefill(self, small_cache):
        """Explicit commits after an existing prompt should extend recent state."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        T_prefill = 9  # leaves 1 token in recent (9 % 2 = 1)

        k_pre = torch.randn(B, G, T_prefill, k_dim)
        v_pre = torch.randn(B, G, T_prefill, k_dim)
        add_full_kv_in_chunks(c, k_pre, v_pre)

        recent_after_prefill = c.recent_count
        tokens_after_prefill = c.token_count

        k_dec = torch.randn(B, G, 1, k_dim)
        v_dec = torch.randn(B, G, 1, k_dim)
        add_full_kv_in_chunks(c, k_dec, v_dec, chunk_size=1)

        assert c.token_count == tokens_after_prefill + 1
        assert c.recent_count == recent_after_prefill + 1

    def test_add_recent_splits_chunk_after_single_token(self, small_cache):
        """A 2-token commit after one recent token should compact the first pair and keep order."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8

        k = torch.randn(B, G, 3, k_dim)
        v = torch.randn(B, G, 3, k_dim)
        add_full_kv_in_chunks(c, k[:, :, :1, :], v[:, :, :1, :], chunk_size=1)
        add_full_kv_in_chunks(c, k[:, :, 1:, :], v[:, :, 1:, :], chunk_size=2)

        assert c.token_count == 3
        assert level_count(c, 0) == 1
        assert c.recent_count == 1

        state = c.get_attention_state()
        slot_k, slot_v, slot_w = real_state_tensors(state)
        # slot 0 = mean of tokens 0-1, slot 1 = exact token 2 (w=1)
        torch.testing.assert_close(slot_k[:, :, 0, :], k[:, :, :2, :].mean(dim=2))
        torch.testing.assert_close(slot_k[:, :, 1, :], k[:, :, 2, :])
        torch.testing.assert_close(slot_w[0, 0], torch.tensor([2.0, 1.0]))

    def test_add_recent_flushes_when_recent_overflows(self, small_cache):
        """When recent buffer overflows, the oldest 2 tokens should compact."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        t = 2  # compaction chunk size (fixed)

        for i in range(t):
            k = torch.randn(B, G, 1, k_dim)
            v = torch.randn(B, G, 1, k_dim)
            add_full_kv_in_chunks(c, k, v, chunk_size=1)

        assert c.recent_count == t
        assert level_count(c, 0) == 0
        assert c.token_count == t

        k = torch.randn(B, G, 1, k_dim)
        v = torch.randn(B, G, 1, k_dim)
        add_full_kv_in_chunks(c, k, v, chunk_size=1)

        assert c.recent_count == 1
        assert level_count(c, 0) == 1
        assert c.token_count == t + 1


# ---------------------------------------------------------------------------
# get_attention_state tests
# ---------------------------------------------------------------------------

class TestGetAttentionState:
    def test_empty_cache_state(self, small_cache):
        """Empty non-semantic cache returns no slots."""
        state = small_cache.get_attention_state()
        assert isinstance(state, CacheAttentionState)
        assert state.slot_k.size(2) == 0
        assert state.slot_v.size(2) == 0
        assert state.slot_w.size(2) == 0
        assert state.slot_valid is None
        assert state.M_s is None

    def test_state_after_ingest(self, small_cache):
        """After ingest, one compact slot of weight 2, no recent tokens."""
        c = small_cache
        c.ingest_chunk(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))

        state = c.get_attention_state()
        _slot_k, _slot_v, slot_w = real_state_tensors(state)

        assert state.slot_valid is None
        assert slot_w.size(2) == 1
        assert slot_w[0, 0, 0].item() == 2.0

    def test_state_with_rank1_stats(self, small_cache):
        """with_stats=True should append zero stats for exact recent tokens."""
        c = small_cache
        k = torch.randn(1, 2, 3, 8)
        v = torch.randn(1, 2, 3, 8)
        add_full_kv_in_chunks(c, k, v, chunk_size=1)

        state = c.get_attention_state(with_stats=True)

        assert state.slot_k.shape == state.slot_sigma_u.shape == state.slot_gamma_a.shape
        assert state.slot_v.shape == state.slot_gamma_b.shape
        assert state.slot_w.shape == state.slot_sigma2.shape == state.slot_gamma.shape
        _slot_k, _slot_v, slot_w = real_state_tensors(state)
        assert slot_w[0, 0].tolist() == [2.0, 1.0]
        assert state.slot_sigma2[0, 0, 0] > 0.0
        assert state.slot_gamma[0, 0, 0] >= 0.0
        assert state.slot_sigma2[0, 0, 1] == 0.0
        assert state.slot_gamma[0, 0, 1] == 0.0

    def test_state_after_prefill(self, small_cache):
        """After prefill, compact slots + recent w=1 tokens should be present."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        T = 9  # 4 flushes (8 tokens) + 1 in recent

        k = torch.randn(B, G, T, k_dim)
        v = torch.randn(B, G, T, k_dim)
        add_full_kv_in_chunks(c, k, v)

        state = c.get_attention_state()
        _slot_k, _slot_v, slot_w = real_state_tensors(state)

        # 4 compact entries (after carry to level 1) + 1 recent token
        assert slot_w.size(2) == c.B + 1
        torch.testing.assert_close(slot_w[0, 0], torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0]))
        # Weights must account for every committed token
        assert slot_w[0, 0].sum().item() == T

    def test_slot_exactness_weighted_mean(self, small_cache):
        """THE merge-exactness invariant: every slot's key/value equals the
        brute-force weighted mean of the tokens it covers, no matter how many
        binary-carry merges produced it. Spans are reconstructed from
        cumsum(slot_w) (slots are contiguous and time-ordered)."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        T = 20  # several flushes + one carry through level 1

        k = torch.randn(B, G, T, k_dim)
        v = torch.randn(B, G, T, k_dim)
        add_full_kv_in_chunks(c, k, v)

        state = c.get_attention_state()
        slot_k, slot_v, slot_w = real_state_tensors(state)
        w = slot_w[0, 0]
        assert w.sum().item() == T  # all tokens accounted for

        ends = torch.cumsum(w, dim=0).long()
        starts = ends - w.long()
        for s in range(slot_k.size(2)):
            span_k = k[:, :, starts[s]:ends[s], :].mean(dim=2)
            span_v = v[:, :, starts[s]:ends[s], :].mean(dim=2)
            torch.testing.assert_close(slot_k[:, :, s, :], span_k, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(slot_v[:, :, s, :], span_v, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# SemanticLogKV routing / anchors
# ---------------------------------------------------------------------------

class TestSemanticLogKV:
    def test_kmax1_flushes_single_token_entries_and_materializes_anchors(self):
        torch.manual_seed(0)
        c = make_semantic_cache(K_max=1, recent_size=2)
        k_raw = torch.randn(1, 2, 5, 8)
        v = torch.randn(1, 2, 5, 8)

        for i in range(5):
            c.add_recent(k_raw[:, :, i:i + 1], v[:, :, i:i + 1], k_raw=k_raw[:, :, i:i + 1], input_pos=torch.tensor([i]))

        assert c.token_count == 5
        assert c.recent_count == 1
        assert int(c.level_count[0, 0, 0, 0].item()) == 4
        assert c._semantic_counts[0][0][0][0] == 4
        torch.testing.assert_close(c.level_w[0, 0, 0, 0, :4], torch.ones(4))
        torch.testing.assert_close(c.level_p_lo[0, 0, 0, 0, :4], torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(c.level_p_hi[0, 0, 0, 0, :4], torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(c.level_sum_wp[0, 0, 0, 0, :4], torch.tensor([0, 1, 2, 3]))

        state = c.get_attention_state(with_stats=True)
        assert state.slot_valid is not None
        assert state.M_s is not None
        slot_k, _slot_v, slot_w = real_state_tensors(state)
        assert slot_k.size(2) == 5
        assert slot_w[0, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
        assert int(state.slot_valid[0, 0].sum().item()) == 4
        assert torch.equal(state.M_s[0, 0][state.slot_valid[0, 0]], torch.ones(4, dtype=torch.long))

        c.reset_parameters()
        assert int(c.level_count.sum().item()) == 0
        assert all(
            count == 0
            for b_counts in c._semantic_counts
            for g_counts in b_counts
            for c_counts in g_counts
            for count in c_counts
        )

    def test_new_segment_inserts_zero_pad_before_boundary_token(self):
        c = make_semantic_cache(K_max=1, n_groups=1, seg_gap_max=0.0, seg_block_level=1)
        k_raw = torch.randn(1, 1, 2, 8)
        v = torch.randn(1, 1, 2, 8)

        c.route_and_flush_batch(k_raw, v, torch.tensor([0, 3]))

        assert int(c.level_count[0, 0, 0, 0].item()) == 3
        torch.testing.assert_close(c.level_w[0, 0, 0, 0, :3], torch.tensor([1.0, 0.0, 1.0]))
        assert c.pad_mask[0, 0, 0, 0, 1]
        assert not c.pad_mask[0, 0, 0, 0, 0]
        assert not c.pad_mask[0, 0, 0, 0, 2]
        assert torch.isfinite(c.level_k[0, 0, 0, 0, 1]).all()
        assert torch.isfinite(c.level_v[0, 0, 0, 0, 1]).all()
        state = c.get_attention_state()
        assert state.slot_valid is not None
        assert int(state.slot_valid[0, 0].sum().item()) == 2

    def test_semantic_attention_state_gathers_only_valid_pooled_slots(self):
        c = make_semantic_cache(K_max=1, n_groups=1, seg_gap_max=0.0, seg_block_level=1)
        c.route_and_flush_batch(torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 8), torch.tensor([0, 3]))

        state = c.get_attention_state(with_stats=True)

        assert state.slot_valid is not None
        assert state.slot_k.size(2) == 2
        assert state.slot_v.size(2) == 2
        assert state.slot_w.size(2) == 2
        assert state.slot_sigma_u.size(2) == 2
        assert bool(state.slot_valid.all().item())

    def test_direct_phase_batches_same_cluster_like_sequential_join(self):
        torch.manual_seed(42)
        seq = make_semantic_cache(K_max=2, n_groups=1, B=3, recent_size=8, cluster_lambda_rel=1e9)
        batched = make_semantic_cache(K_max=2, n_groups=1, B=3, recent_size=8, cluster_lambda_rel=1e9)
        k0 = torch.randn(8)
        v0 = torch.randn(8)
        for cache in (seq, batched):
            cache._semantic_new_cluster(0, 0, 0, 0, k0, v0, torch.tensor(0), record=False)
        k_raw = torch.randn(1, 1, 8, 8)
        v = torch.randn(1, 1, 8, 8)
        pos = torch.arange(1, 9)

        for i in range(8):
            seq._semantic_join_or_segment(0, 0, 0, int(pos[i]), k_raw[0, 0, i], v[0, 0, i], pos[i], record=False)
        batched.route_and_flush_batch(k_raw, v, pos)

        for name in (
            "level_k",
            "level_v",
            "level_w",
            "level_sigma_u",
            "level_sigma2",
            "level_gamma_a",
            "level_gamma_b",
            "level_gamma",
            "level_p_lo",
            "level_p_hi",
            "level_sum_wp",
            "level_order",
            "pad_mask",
            "level_count",
            "n_total",
            "p_hi_c",
            "current_segment",
            "level0_phase",
            "alive",
        ):
            assert torch.equal(getattr(seq, name), getattr(batched, name)), f"{name} differ"
        torch.testing.assert_close(seq.centroid, batched.centroid, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(seq.n_eff, batched.n_eff)

    def test_kmax1_batch_route_matches_single_token_route_with_segments(self):
        torch.manual_seed(7)
        seq = make_semantic_cache(K_max=1, n_groups=1, B=3, recent_size=8, seg_gap_max=3.0, seg_block_level=1)
        batched = make_semantic_cache(K_max=1, n_groups=1, B=3, recent_size=8, seg_gap_max=3.0, seg_block_level=1)
        pos = torch.tensor([0, 1, 2, 8, 9, 16, 17, 18])
        k_raw = torch.randn(1, 1, pos.numel(), 8)
        v = torch.randn(1, 1, pos.numel(), 8)

        for i in range(pos.numel()):
            seq.route_and_flush_batch(k_raw[:, :, i:i + 1], v[:, :, i:i + 1], pos[i:i + 1])
        batched.route_and_flush_batch(k_raw, v, pos)

        for name in (
            "level_k",
            "level_v",
            "level_w",
            "level_sigma_u",
            "level_sigma2",
            "level_gamma_a",
            "level_gamma_b",
            "level_gamma",
            "level_p_lo",
            "level_p_hi",
            "level_sum_wp",
            "level_order",
            "pad_mask",
            "level_count",
            "n_total",
            "p_hi_c",
            "current_segment",
            "level0_phase",
            "alive",
        ):
            assert torch.equal(getattr(seq, name), getattr(batched, name)), f"{name} differ"
        torch.testing.assert_close(seq.centroid, batched.centroid, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(seq.n_eff, batched.n_eff)

    def test_semantic_chunk_tree_merges_near_local_clusters_across_chunks(self):
        c = make_semantic_cache(
            K_max=4,
            n_groups=1,
            B=4,
            recent_size=8,
            cluster_lambda_rel=0.25,
            semantic_flush_granularity=4,
            semantic_cluster_chunk_size=2,
        )
        k_raw = torch.zeros(1, 1, 4, 8)
        k_raw[0, 0, :, 0] = torch.tensor([0.0, 10.0, 0.1, 10.1])
        v = torch.randn(1, 1, 4, 8)

        c.route_and_flush_batch(k_raw, v, torch.arange(4))

        live = torch.nonzero(c.alive[0, 0], as_tuple=False).flatten().tolist()
        assert len(live) == 2
        assert sorted(int(c.n_total[0, 0, idx].item()) for idx in live) == [2, 2]
        assert sorted(round(float(c.centroid[0, 0, idx, 0].item()), 2) for idx in live) == [0.05, 10.05]
        assert sorted(
            (int(c.level_p_lo[0, 0, idx, 0, 0].item()), int(c.level_p_lo[0, 0, idx, 0, 1].item()))
            for idx in live
        ) == [(0, 2), (1, 3)]

    def test_semantic_chunk_tree_local_chunk_uses_noncausal_components(self):
        c = make_semantic_cache(
            K_max=4, n_groups=1, recent_size=8, cluster_lambda_rel=0.25, semantic_cluster_chunk_size=3
        )
        k_raw = torch.zeros(1, 1, 3, 8)
        k_raw[0, 0, :, 0] = torch.tensor([0.0, 10.0, 0.1])

        clusters = c._semantic_tree_local_chunks(0, 0, k_raw, 0, 3, [[0, 1, 2]])

        assert sorted(node.tokens for node in clusters) == [(0, 2), (1,)]
        assert sorted(round(float(node.centroid[0].item()), 2) for node in clusters) == [0.05, 10.0]

    def test_tree_candidate_ranking_uses_temporal_tiebreak(self):
        # algorithm-spec.md §5.3: eta only reorders candidates, it must not
        # affect the accept/reject threshold. Two candidates tied on semantic
        # distance (0.01 either way) but far apart in p_hi -- the temporally
        # closer one must win.
        c = make_semantic_cache(K_max=4, n_groups=1, cluster_lambda_rel=1.0)
        older = torch.zeros(8)
        recent = torch.zeros(8)
        recent[0] = 0.2
        pool = [
            _SemanticTreeCluster(older, 1, 0, (0,), ()),
            _SemanticTreeCluster(recent, 1, 100, (1,), ()),
        ]
        node_vec = torch.zeros(8)
        node_vec[0] = 0.1
        node = _SemanticTreeCluster(node_vec, 1, 101, (), (0,))

        best = c._semantic_tree_best_candidate(0, 0, pool, [0, 1], node)
        assert best == 1

    def test_tree_candidate_ranking_uses_capacity_penalty(self):
        c = make_semantic_cache(K_max=4, n_groups=1, cluster_lambda_rel=1.0, semantic_capacity_beta=1.0)
        target = c._semantic_capacity_target()
        pool = [
            _SemanticTreeCluster(torch.zeros(8), 1, 0, (0,), ()),
            _SemanticTreeCluster(torch.zeros(8), int(target * 3), 0, (1,), ()),
        ]
        node = _SemanticTreeCluster(torch.zeros(8), 1, 0, (), (0,))

        # Identical centroid and p_hi for both candidates -> semantic and
        # temporal terms are tied at 0; only the capacity penalty differs.
        best = c._semantic_tree_best_candidate(0, 0, pool, [0, 1], node)
        assert best == 0

    def test_semantic_chunk_tree_consolidates_existing_clusters_via_ward_reduce(self):
        c = make_semantic_cache(
            K_max=2, n_groups=1, B=4, recent_size=8,
            cluster_lambda_rel=0.25, semantic_cluster_chunk_size=2,
        )
        k0 = torch.zeros(8)
        k1 = torch.zeros(8)
        k1[0] = 0.01
        c._semantic_new_cluster(0, 0, 0, 0, k0, torch.randn(8), torch.tensor(0), record=False)
        c._semantic_new_cluster(0, 0, 1, 1, k1, torch.randn(8), torch.tensor(1), record=False)

        k_raw = torch.zeros(1, 1, 4, 8)
        k_raw[0, 0, :, 0] = 100.0  # far from both existing clusters
        v = torch.randn(1, 1, 4, 8)
        c.route_and_flush_batch(k_raw, v, torch.arange(2, 6))

        live = torch.nonzero(c.alive[0, 0], as_tuple=False).flatten().tolist()
        assert len(live) == 2  # K_max respected: the two close existing clusters merged
        n_totals = sorted(int(c.n_total[0, 0, idx].item()) for idx in live)
        assert n_totals == [2, 4]
        merged_idx = [idx for idx in live if int(c.n_total[0, 0, idx].item()) == 2][0]
        assert round(float(c.centroid[0, 0, merged_idx, 0].item()), 4) == 0.005

    def test_semantic_chunk_tree_local_ward_merge_when_chunk_exceeds_k_max(self):
        c = make_semantic_cache(
            K_max=2, n_groups=1, B=4, recent_size=8,
            cluster_lambda_rel=0.25, semantic_cluster_chunk_size=3,
        )
        k_raw = torch.zeros(1, 1, 4, 8)
        k_raw[0, 0, :, 0] = torch.tensor([0.0, 100.0, 200.0, -100.0])
        v = torch.randn(1, 1, 4, 8)

        c.route_and_flush_batch(k_raw, v, torch.arange(4))

        live = torch.nonzero(c.alive[0, 0], as_tuple=False).flatten().tolist()
        assert len(live) <= 2  # K_max respected even though the first chunk
                               # alone holds 3 mutually distant tokens
        assert sum(int(c.n_total[0, 0, idx].item()) for idx in live) == 4  # no tokens lost

    def test_semantic_chunk_tree_ward_selects_multiple_disjoint_pairs(self):
        c = make_semantic_cache(K_max=3, n_groups=1)
        clusters = []
        for i, x in enumerate([0.0, 0.1, 10.0, 10.1, 100.0]):
            vec = torch.zeros(8)
            vec[0] = x
            clusters.append(_SemanticTreeCluster(vec, 1, i, (), (i,)))

        assert c._semantic_tree_ward_pairs(clusters, 2) == [(0, 1), (2, 3)]

    def test_op_log_replay_rebuilds_multicluster_state(self):
        c = make_semantic_cache(K_max=2, n_groups=1, B=3, recent_size=2, cluster_lambda_rel=0.25, seg_gap_max=8.0)
        r = make_semantic_cache(K_max=2, n_groups=1, B=3, recent_size=2, cluster_lambda_rel=0.25, seg_gap_max=8.0)
        k_raw = torch.tensor(
            [[[
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [8.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]]]
        )
        v = torch.randn(1, 1, 5, 8)
        pos = torch.tensor([0, 1, 10, 11, 30])

        c.begin_op_log()
        c.route_and_flush_batch(k_raw, v, pos, record_op_log=True)
        op_log, op_log_len = c.take_op_log()
        assert op_log is not None
        assert op_log_len is not None
        assert op_log_len[0, 0] >= 6
        assert len(c._last_op_log_host[0][0]) == int(op_log_len[0, 0].item())

        r.route_and_flush_batch(
            k_raw,
            v,
            pos,
            replay_op_log=op_log,
            replay_op_log_len=op_log_len,
            replay_op_log_host=c._last_op_log_host,
        )

        assert_cache_states_bit_identical(c, r)

    def test_ward_merge_rebuild_phase_tracks_replayed_entries(self):
        c = make_semantic_cache(K_max=2, n_groups=1, B=2, recent_size=2, seg_block_level=2)
        k_raw = torch.randn(9, 8)
        v = torch.randn(9, 8)

        c._semantic_new_cluster(0, 0, 0, 0, k_raw[0], v[0], torch.tensor(0), record=False)
        for i in range(1, 5):
            c._semantic_join(0, 0, 0, 0, i, k_raw[i], v[i], torch.tensor(i), record=False)
        c._semantic_new_cluster(0, 0, 1, 10, k_raw[5], v[5], torch.tensor(10), record=False)
        for i in range(6, 9):
            c._semantic_join(0, 0, 1, 0, i + 10, k_raw[i], v[i], torch.tensor(i + 10), record=False)

        expected = len(c._semantic_collect_entries(0, 0, 0)) + len(c._semantic_collect_entries(0, 0, 1))
        assert expected != int((c.level0_phase[0, 0, 0] + c.level0_phase[0, 0, 1]).item())

        c._semantic_ward_merge(0, 0, 0, 1, record=False)

        assert int(c.level0_phase[0, 0, 0].item()) == expected

    def test_capacity_penalty_prefers_less_loaded_cluster(self):
        c = make_semantic_cache(K_max=2, n_groups=1, max_seq_length=8, cluster_lambda_rel=10.0, semantic_capacity_beta=10.0)
        c.alive[0, 0, :2] = True
        c.centroid[0, 0, 0].zero_()
        c.centroid[0, 0, 1].zero_()
        c.centroid[0, 0, 1, 0] = 0.5
        c.n_total[0, 0] = torch.tensor([8, 0], dtype=torch.int32)
        c.p_hi_c[0, 0] = torch.tensor([0, 0])

        winner, s_winner, direct = c._semantic_existing_assignments(
            torch.zeros(1, 1, 1, 8),
            torch.tensor([[1]]),
        )

        assert int(winner[0, 0, 0].item()) == 1
        assert bool(direct[0, 0, 0].item())
        assert s_winner[0, 0, 0].item() == pytest.approx(0.25)

    def test_hard_cap_forces_split_before_joining_overfull_cluster(self):
        c = make_semantic_cache(
            K_max=2,
            n_groups=1,
            max_seq_length=8,
            cluster_lambda_rel=10.0,
            semantic_capacity_hard_cap_mult=1.0,
        )
        c._set_semantic_alive(0, 0, 0, True)
        c.centroid[0, 0, 0].zero_()
        c.n_eff[0, 0, 0] = 4.0
        c._set_semantic_n_total(0, 0, 0, 4)
        c._set_semantic_p_hi(0, 0, 0, 3)

        c.route_and_flush_batch(torch.zeros(1, 1, 1, 8), torch.randn(1, 1, 1, 8), torch.tensor([4]))

        assert int(c.n_total[0, 0, 0].item()) == 4
        assert bool(c.alive[0, 0, 1].item())
        assert int(c.n_total[0, 0, 1].item()) == 1

    def test_ward_pair_avoids_merges_over_hard_cap_when_possible(self):
        c = make_semantic_cache(K_max=3, n_groups=1, max_seq_length=12, semantic_capacity_hard_cap_mult=1.0)
        for c_idx, n in enumerate([4, 1, 1]):
            c._set_semantic_alive(0, 0, c_idx, True)
            c._set_semantic_n_total(0, 0, c_idx, n)
        c.centroid[0, 0].zero_()
        c.centroid[0, 0, 1, 0] = 0.01
        c.centroid[0, 0, 2, 0] = 10.0

        assert c._semantic_ward_pair(0, 0) == (1, 2)

    def test_ward_pair_falls_back_when_every_merge_exceeds_hard_cap(self):
        c = make_semantic_cache(K_max=2, n_groups=1, max_seq_length=8, semantic_capacity_hard_cap_mult=1.0)
        for c_idx in range(2):
            c._set_semantic_alive(0, 0, c_idx, True)
            c._set_semantic_n_total(0, 0, c_idx, 4)
        c.centroid[0, 0].zero_()

        assert c._semantic_ward_pair(0, 0) == (0, 1)

    def test_ward_cost_incremental_update_matches_full_rebuild(self):
        c = make_semantic_cache(K_max=3, n_groups=1, B=3)
        for idx, offset in enumerate([0.0, 1.0, 4.0]):
            k = torch.zeros(8)
            k[0] = offset
            c._semantic_new_cluster(0, 0, idx, idx, k, torch.randn(8), torch.tensor(idx), record=False)
        assert c._semantic_ward_pair(0, 0) == (0, 1)

        c._semantic_ward_merge(0, 0, 0, 1, record=False)
        incremental = c.ward_cost.clone()
        c._semantic_ward_dirty[0][0] = True
        c._semantic_rebuild_ward_cost(0, 0)

        torch.testing.assert_close(incremental, c.ward_cost, atol=1e-6, rtol=1e-6)

    def test_semantic_flush_schedule_is_independent_of_commit_chunking(self):
        torch.manual_seed(123)
        a = make_semantic_cache(K_max=2, n_groups=1, B=3, recent_size=4, cluster_lambda_rel=0.25)
        b = make_semantic_cache(K_max=2, n_groups=1, B=3, recent_size=4, cluster_lambda_rel=0.25)
        k_raw = torch.randn(1, 1, 8, 8)
        v = torch.randn(1, 1, 8, 8)

        for start in range(0, 8, 2):
            end = start + 2
            a.add_recent(k_raw[:, :, start:end], v[:, :, start:end], k_raw=k_raw[:, :, start:end], input_pos=torch.arange(start, end))
        for start in range(0, 8, 4):
            end = start + 4
            b.add_recent(k_raw[:, :, start:end], v[:, :, start:end], k_raw=k_raw[:, :, start:end], input_pos=torch.arange(start, end))

        assert_cache_states_bit_identical(a, b)

    def test_semantic_flush_granularity_keeps_kmax1_state(self):
        torch.manual_seed(321)
        a = make_semantic_cache(K_max=1, n_groups=1, B=4, recent_size=4)
        b = make_semantic_cache(K_max=1, n_groups=1, B=4, recent_size=4, semantic_flush_granularity=4)
        k_raw = torch.randn(1, 1, 8, 8)
        v = torch.randn(1, 1, 8, 8)

        for cache in (a, b):
            cache.add_recent(k_raw[:, :, :4], v[:, :, :4], k_raw=k_raw[:, :, :4], input_pos=torch.arange(4))
            cache.add_recent(k_raw[:, :, 4:], v[:, :, 4:], k_raw=k_raw[:, :, 4:], input_pos=torch.arange(4, 8))

        assert_cache_states_bit_identical(a, b)


# ---------------------------------------------------------------------------
# log_kv_slot_attention tests
# ---------------------------------------------------------------------------

class TestSlotAttention:
    @staticmethod
    def _make_state(n_compact, n_recent, B=1, G=2, k_dim=8, v_dim=8):
        """Build a (slot_k, slot_v, slot_w) state: w=2 compact slots then w=1 recent."""
        n = n_compact + n_recent
        slot_k = torch.randn(B, G, n, k_dim)
        slot_v = torch.randn(B, G, n, v_dim)
        slot_w = torch.cat(
            [torch.full((B, G, n_compact), 2.0), torch.ones(B, G, n_recent)], dim=-1
        )
        return slot_k, slot_v, slot_w

    def test_output_shape(self):
        """Output should have shape (B, nh, T_q, v_dim)."""
        B, nh, T_q, k_dim, v_dim = 1, 4, 1, 8, 8

        slot_k, slot_v, slot_w = self._make_state(n_compact=3, n_recent=2)
        q = torch.randn(B, nh, T_q, k_dim)

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)

    def test_gqa_expansion(self):
        """When nh != n_groups, k/v/w should be expanded via repeat_interleave."""
        B, nh, G, T_q, k_dim, v_dim = 1, 4, 2, 1, 8, 8  # nh=4, G=2 -> repeat factor 2

        slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=1, G=G)
        q = torch.randn(B, nh, T_q, k_dim)

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)
        assert not torch.isnan(out).any()

    def test_empty_state(self):
        """With no slots to attend to, output should be zeros."""
        B, nh, T_q, k_dim, v_dim = 1, 4, 1, 8, 8

        slot_k, slot_v, slot_w = self._make_state(n_compact=0, n_recent=0)
        q = torch.randn(B, nh, T_q, k_dim)

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_only_recent(self):
        """With only recent tokens (no compact slots), should still work."""
        B, nh, G, T_q, v_dim = 1, 2, 2, 1, 8

        slot_k, slot_v, slot_w = self._make_state(n_compact=0, n_recent=3, G=G)
        q = torch.randn(B, nh, T_q, 8)

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)
        assert not torch.isnan(out).any()

    def test_mask_applied(self):
        """Mask should prevent attention to certain slots."""
        B, nh, G, T_q, v_dim = 1, 2, 2, 3, 8

        slot_k, slot_v, slot_w = self._make_state(n_compact=1, n_recent=2, G=G)
        q = torch.randn(B, nh, T_q, 8)

        n_slots = 3
        mask = torch.ones(T_q, n_slots, dtype=torch.bool)
        for i in range(T_q):
            mask[i, i + 1:] = False

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1, mask=mask)

        assert out.shape == (B, nh, T_q, v_dim)
        assert not torch.isnan(out).any()

        # Query 0 sees only slot 0 -> its output must be exactly slot 0's value.
        v_exp = slot_v.repeat_interleave(nh // G, dim=1)
        torch.testing.assert_close(out[:, :, 0, :], v_exp[:, :, 0, :])

    def test_scale_applied(self):
        """Different scales should produce different outputs."""
        B, nh, G, T_q = 1, 2, 2, 1

        slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=1, G=G)
        q = torch.randn(B, nh, T_q, 8)

        out1 = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.01)
        out2 = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=1.0)

        assert not torch.allclose(out1, out2)

    def test_equivalent_to_standard_attention_when_all_weight_1(self):
        """When all slots have weight=1 (no compaction), the mass bias is 0 and
        slot attention must match standard scaled dot-product attention."""
        B, nh, G, T_q, k_dim, v_dim = 1, 2, 2, 1, 8, 8
        scale = 1.0 / math.sqrt(k_dim)

        slot_k, slot_v, slot_w = self._make_state(
            n_compact=0, n_recent=3, G=G, k_dim=k_dim, v_dim=v_dim
        )
        q = torch.randn(B, nh, T_q, k_dim)

        # Standard attention with GQA expansion
        q_per_kv = nh // G
        k_exp = slot_k.repeat_interleave(q_per_kv, dim=1)
        v_exp = slot_v.repeat_interleave(q_per_kv, dim=1)
        scores = torch.matmul(q, k_exp.mT) * scale
        out_std = torch.matmul(torch.softmax(scores, dim=-1), v_exp)

        out_slot = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=scale)

        torch.testing.assert_close(out_slot, out_std, atol=1e-5, rtol=1e-5)

    def test_log_w_mass_bias_matches_duplicated_tokens(self):
        """THE mass-bias invariant: a slot with weight w and key x must draw
        exactly the same softmax mass as w separate w=1 slots with key x
        (log-sum-exp of equal logits = logit + log w). Needs a distractor slot
        so the mass split is actually observable."""
        B, G, nh, k_dim, v_dim = 1, 2, 2, 8, 8
        scale = 1.0 / math.sqrt(k_dim)
        q = torch.randn(B, nh, 1, k_dim)

        x_k = torch.randn(B, G, 1, k_dim)
        x_v = torch.randn(B, G, 1, v_dim)
        y_k = torch.randn(B, G, 1, k_dim)  # distractor
        y_v = torch.randn(B, G, 1, v_dim)

        # State A: token x twice (two w=1 slots) + distractor y
        kA = torch.cat([x_k, x_k, y_k], dim=2)
        vA = torch.cat([x_v, x_v, y_v], dim=2)
        wA = torch.ones(B, G, 3)

        # State B: one merged slot (k=x, w=2) + distractor y
        kB = torch.cat([x_k, y_k], dim=2)
        vB = torch.cat([x_v, y_v], dim=2)
        wB = torch.cat([torch.full((B, G, 1), 2.0), torch.ones(B, G, 1)], dim=-1)

        out_A = log_kv_slot_attention(q, kA, vA, wA, scale=scale)
        out_B = log_kv_slot_attention(q, kB, vB, wB, scale=scale)

        torch.testing.assert_close(out_B, out_A, atol=1e-5, rtol=1e-5)

    def test_lam_zero_disables_mass_bias(self):
        """lam=0 must ignore slot weights entirely (∝1/w forgetting ablation)."""
        B, G, nh, k_dim = 1, 2, 2, 8
        q = torch.randn(B, nh, 1, k_dim)
        slot_k, slot_v, _ = self._make_state(n_compact=2, n_recent=1, G=G)

        w_real = torch.tensor([[[8.0, 2.0, 1.0]]]).expand(B, G, 3)
        w_ones = torch.ones(B, G, 3)

        out_biased = log_kv_slot_attention(q, slot_k, slot_v, w_real, scale=0.5)
        out_lam0 = log_kv_slot_attention(q, slot_k, slot_v, w_real, scale=0.5, lam=0.0)
        out_unweighted = log_kv_slot_attention(q, slot_k, slot_v, w_ones, scale=0.5)

        torch.testing.assert_close(out_lam0, out_unweighted)
        assert not torch.allclose(out_biased, out_lam0)

    def test_slot_m_none_reproduces_prior_mass_bias(self):
        """Omitting slot_M must reproduce the exact lam*log(w) formula."""
        B, G, nh, k_dim = 1, 2, 2, 8
        q = torch.randn(B, nh, 1, k_dim)
        slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=1, G=G)

        out_default = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.3)
        out_m_none = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.3, slot_M=None)

        torch.testing.assert_close(out_default, out_m_none)

    def test_slot_m_matches_manual_mass_bias(self):
        """slot_M given must match lam*log(w) - log(M), unconditionally on lam."""
        B, G, nh, k_dim = 1, 2, 4, 8  # nh != G -> GQA branch
        q = torch.randn(B, nh, 1, k_dim)
        slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=1, G=G)
        S = slot_w.size(-1)
        slot_M = torch.tensor([[1, 3, 2]] * G, dtype=torch.float32).unsqueeze(0)
        assert slot_M.shape == (B, G, S)

        for lam in (0.0, 0.5, 1.0):
            out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.3, lam=lam, slot_M=slot_M)

            k_exp = slot_k.repeat_interleave(nh // G, dim=1)
            v_exp = slot_v.repeat_interleave(nh // G, dim=1)
            bias = (lam * slot_w.log() - slot_M.log()).repeat_interleave(nh // G, dim=1)
            scores = torch.matmul(q, k_exp.mT) * 0.3 + bias.unsqueeze(-2)
            expected = torch.matmul(torch.softmax(scores, dim=-1), v_exp)

            torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_slot_valid_masks_invalid_pooled_slots(self):
        """Invalid pooled slots must draw zero softmax mass, MHA and GQA alike."""
        for nh in (2, 4):  # nh==G (MHA) and nh!=G (GQA)
            B, G, k_dim = 1, 2, 8
            q = torch.randn(B, nh, 1, k_dim)
            slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=1, G=G, k_dim=k_dim, v_dim=k_dim)
            slot_valid = torch.tensor([[True, False]] * G).unsqueeze(0)  # mask the 2nd of 2 pooled slots

            out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.3, slot_valid=slot_valid)

            k_exp = slot_k.repeat_interleave(nh // G, dim=1)
            v_exp = slot_v.repeat_interleave(nh // G, dim=1)
            scores = torch.matmul(q, k_exp.mT) * 0.3 + slot_w.log().repeat_interleave(nh // G, dim=1).unsqueeze(-2)
            scores[..., 1] = float("-inf")
            expected = torch.matmul(torch.softmax(scores, dim=-1), v_exp)

            torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_slot_valid_all_invalid_row_raises(self):
        B, G, nh, k_dim = 1, 2, 2, 8
        q = torch.randn(B, nh, 1, k_dim)
        slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=0, G=G)
        slot_valid = torch.zeros(B, G, 2, dtype=torch.bool)

        with pytest.raises(ValueError, match="no valid slot"):
            log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.3, slot_valid=slot_valid)

    def test_rank1_score_correction_affects_softmax_mass(self):
        """Sigma stats add the second-order score term before softmax."""
        q = torch.tensor([[[[1.0, 0.0]]]])  # (B=1, nh=1, T=1, D=2)
        slot_k = torch.zeros(1, 1, 2, 2)
        slot_v = torch.tensor([[[[1.0], [0.0]]]])
        slot_w = torch.ones(1, 1, 2)

        sigma_u = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        sigma2 = torch.tensor([[[2.0, 0.0]]])
        gamma_a = torch.zeros_like(slot_k)
        gamma_b = torch.zeros_like(slot_v)
        gamma = torch.zeros_like(slot_w)

        out = log_kv_slot_attention(
            q,
            slot_k,
            slot_v,
            slot_w,
            scale=1.0,
            slot_sigma_u=sigma_u,
            slot_sigma2=sigma2,
            slot_gamma_a=gamma_a,
            slot_gamma_b=gamma_b,
            slot_gamma=gamma,
        )

        torch.testing.assert_close(out[0, 0, 0, 0], torch.sigmoid(torch.tensor(1.0)))

    def test_rank1_value_correction_adds_gamma_readout(self):
        """Gamma stats add scale * gamma * (q·a) * b after slot softmax."""
        q = torch.tensor([[[[3.0, 0.0]]]])
        slot_k = torch.zeros(1, 1, 1, 2)
        slot_v = torch.zeros(1, 1, 1, 2)
        slot_w = torch.ones(1, 1, 1)
        sigma_u = torch.zeros_like(slot_k)
        sigma2 = torch.zeros_like(slot_w)
        gamma_a = torch.tensor([[[[1.0, 0.0]]]])
        gamma_b = torch.tensor([[[[0.0, 1.0]]]])
        gamma = torch.tensor([[[2.0]]])

        out = log_kv_slot_attention(
            q,
            slot_k,
            slot_v,
            slot_w,
            scale=0.5,
            slot_sigma_u=sigma_u,
            slot_sigma2=sigma2,
            slot_gamma_a=gamma_a,
            slot_gamma_b=gamma_b,
            slot_gamma=gamma,
        )

        torch.testing.assert_close(out, torch.tensor([[[[0.0, 3.0]]]]))

    def test_rank1_value_correction_accepts_bf16_gamma_b_with_fp32_gamma(self):
        """Gamma's scalar slot factor is fp32; gamma_b follows the activation dtype."""
        B, G, nh, T_q, S, D, Dv = 1, 2, 4, 1, 3, 8, 8
        q = torch.randn(B, nh, T_q, D, dtype=torch.bfloat16)
        slot_k = torch.randn(B, G, S, D, dtype=torch.bfloat16)
        slot_v = torch.randn(B, G, S, Dv, dtype=torch.bfloat16)
        slot_w = torch.ones(B, G, S)
        sigma_u = torch.zeros_like(slot_k)
        sigma2 = torch.zeros(B, G, S)
        gamma_a = torch.randn(B, G, S, D, dtype=torch.bfloat16)
        gamma_b = torch.randn(B, G, S, Dv, dtype=torch.bfloat16)
        gamma = torch.ones(B, G, S)

        out = log_kv_slot_attention(
            q,
            slot_k,
            slot_v,
            slot_w,
            scale=0.5,
            slot_sigma_u=sigma_u,
            slot_sigma2=sigma2,
            slot_gamma_a=gamma_a,
            slot_gamma_b=gamma_b,
            slot_gamma=gamma,
        )

        assert out.dtype == torch.bfloat16

    def test_append_exact_tokens(self):
        """Helper must append w=1 entries after the cached slots, in order."""
        B, G, k_dim, v_dim = 1, 2, 8, 8
        slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=1, G=G)
        k_new = torch.randn(B, G, 2, k_dim)
        v_new = torch.randn(B, G, 2, v_dim)

        out = append_exact_tokens(CacheAttentionState(slot_k, slot_v, slot_w), k_new, v_new)

        assert out.slot_k.size(2) == 5
        torch.testing.assert_close(out.slot_k[:, :, 3:, :], k_new)
        torch.testing.assert_close(out.slot_v[:, :, 3:, :], v_new)
        torch.testing.assert_close(out.slot_w[0, 0], torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0]))

    def test_append_exact_tokens_with_rank1_stats(self):
        """Appended exact tokens should receive zero second-order stats."""
        B, G, k_dim, v_dim = 1, 2, 8, 4
        slot_k = torch.randn(B, G, 2, k_dim)
        slot_v = torch.randn(B, G, 2, v_dim)
        slot_w = torch.full((B, G, 2), 2.0)
        sigma_u = torch.randn_like(slot_k)
        sigma2 = torch.rand_like(slot_w)
        gamma_a = torch.randn_like(slot_k)
        gamma_b = torch.randn_like(slot_v)
        gamma = torch.rand_like(slot_w)
        k_new = torch.randn(B, G, 3, k_dim)
        v_new = torch.randn(B, G, 3, v_dim)

        out = append_exact_tokens(
            CacheAttentionState(
                slot_k=slot_k,
                slot_v=slot_v,
                slot_w=slot_w,
                slot_sigma_u=sigma_u,
                slot_sigma2=sigma2,
                slot_gamma_a=gamma_a,
                slot_gamma_b=gamma_b,
                slot_gamma=gamma,
            ),
            k_new,
            v_new,
        )

        assert len(out) == 10
        assert out.slot_sigma_u.size(2) == 5
        torch.testing.assert_close(out.slot_sigma_u[:, :, :2, :], sigma_u)
        torch.testing.assert_close(out.slot_sigma2[:, :, :2], sigma2)
        torch.testing.assert_close(out.slot_gamma_a[:, :, :2, :], gamma_a)
        torch.testing.assert_close(out.slot_gamma_b[:, :, :2, :], gamma_b)
        torch.testing.assert_close(out.slot_gamma[:, :, :2], gamma)
        torch.testing.assert_close(out.slot_sigma_u[:, :, 2:, :], torch.zeros_like(k_new))
        torch.testing.assert_close(out.slot_sigma2[:, :, 2:], torch.zeros(B, G, 3))
        torch.testing.assert_close(out.slot_gamma_b[:, :, 2:, :], torch.zeros_like(v_new))

    def test_append_exact_tokens_preserves_prefix_anchor_fields(self):
        slot_k = torch.zeros(1, 1, 2, 1)
        slot_v = torch.tensor([[[[0.0], [10.0]]]])
        slot_w = torch.tensor([[[2.0, 2.0]]])
        state = CacheAttentionState(
            slot_k,
            slot_v,
            slot_w,
            slot_valid=torch.tensor([[[True, False]]]),
            M_s=torch.tensor([[[1, 2]]]),
        )

        out = append_exact_tokens(state, torch.zeros(1, 1, 1, 1), torch.tensor([[[[100.0]]]]))

        assert torch.equal(out.slot_valid, state.slot_valid)
        assert torch.equal(out.M_s, state.M_s)
        assert out.slot_w.size(-1) == 3
        y = log_kv_slot_attention(
            torch.zeros(1, 1, 1, 1),
            out.slot_k,
            out.slot_v,
            out.slot_w,
            scale=1.0,
            lam=0.0,
            slot_M=out.M_s,
            slot_valid=out.slot_valid,
        )
        torch.testing.assert_close(y, torch.tensor([[[[50.0]]]]))


# ---------------------------------------------------------------------------
# reset_parameters tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_state(self, small_cache):
        c = small_cache
        # Put some data in
        c.ingest_chunk(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))
        assert c.token_count > 0
        assert level_count(c, 0) > 0

        c.reset_parameters()

        assert c.token_count == 0
        assert c.recent_count == 0
        assert c.level_count.sum() == 0
        assert c.total_slots == 0
        assert c.total_tokens_covered == 0
        assert c.recent_k.abs().sum() == 0
        assert c.level_k.abs().sum() == 0


# ---------------------------------------------------------------------------
# Integration: explicit commit + get_attention_state round-trip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_explicit_commit_roundtrip(self):
        """End-to-end: commit prompt/chunks and verify state shapes throughout."""
        B_batch, G = 1, 2
        k_dim = 8
        v_dim = 8
        max_seq = 64

        c = LogStructuredKVCache(
            (B_batch, G, max_seq, k_dim),
            (B_batch, G, max_seq, v_dim),
            B=4,
        )

        # Prefill 7 tokens (3 flushes of 2 + 1 in recent)
        T_prefill = 7
        k = torch.randn(B_batch, G, T_prefill, k_dim)
        v = torch.randn(B_batch, G, T_prefill, v_dim)
        add_full_kv_in_chunks(c, k, v)

        assert c.token_count == T_prefill
        assert c.recent_count == 1  # 7 % 2 = 1

        # Decode 3 more tokens
        for i in range(3):
            k_dec = torch.randn(B_batch, G, 1, k_dim)
            v_dec = torch.randn(B_batch, G, 1, v_dim)
            add_full_kv_in_chunks(c, k_dec, v_dec, chunk_size=1)

        assert c.token_count == T_prefill + 3

        # Verify we can compute attention with the final state
        nh = 4
        state = c.get_attention_state()
        assert state.slot_w[0, 0].sum().item() == T_prefill + 3
        q = torch.randn(B_batch, nh, 1, k_dim)
        out = log_kv_slot_attention(
            q, state.slot_k, state.slot_v, state.slot_w, scale=0.1, slot_valid=state.slot_valid,
        )
        assert out.shape == (B_batch, nh, 1, v_dim)
        assert not torch.isnan(out).any()

    def test_cache_attention_matches_dense_for_pairwise_identical_tokens(self):
        """End-to-end EXACTNESS check of the whole design: when the two tokens
        inside every compacted pair are identical, the merged key/value equal
        the shared token and +log(2) equals the log-sum-exp of the two equal
        logits — so slot attention over the cache must reproduce dense
        attention over all tokens exactly (up to fp tolerance)."""
        B_batch, G, nh = 1, 2, 4
        k_dim = 8
        v_dim = 8
        T = 8  # 3 flushes (tokens 0-5 -> 3 slots, no carry with B=4) + 2 recent

        c = LogStructuredKVCache(
            (B_batch, G, 64, k_dim), (B_batch, G, 64, v_dim), B=4,
        )

        base_k = torch.randn(B_batch, G, T // 2, k_dim)
        base_v = torch.randn(B_batch, G, T // 2, v_dim)
        k = base_k.repeat_interleave(2, dim=2)  # pairs of identical tokens
        v = base_v.repeat_interleave(2, dim=2)
        add_full_kv_in_chunks(c, k, v)  # 2-chunks align with the pairs
        assert level_count(c, 0) == 3
        assert c.recent_count == 2

        scale = 1.0 / math.sqrt(k_dim)
        q = torch.randn(B_batch, nh, 1, k_dim)

        state = c.get_attention_state()
        out_slot = log_kv_slot_attention(
            q, state.slot_k, state.slot_v, state.slot_w, scale=scale, slot_valid=state.slot_valid,
        )

        # Dense attention over all T tokens
        q_per_kv = nh // G
        k_exp = k.repeat_interleave(q_per_kv, dim=1)
        v_exp = v.repeat_interleave(q_per_kv, dim=1)
        scores = torch.matmul(q, k_exp.mT) * scale
        out_dense = torch.matmul(torch.softmax(scores, dim=-1), v_exp)

        torch.testing.assert_close(out_slot, out_dense, atol=1e-5, rtol=1e-5)

    def test_capacity_for_4k(self):
        """Verify the cache can handle 4K tokens without overflow (B=1024, 2:1 compaction)."""
        B_batch, G = 1, 1
        k_dim = 8
        v_dim = 8
        max_seq = 4096

        c = LogStructuredKVCache(
            (B_batch, G, max_seq, k_dim),
            (B_batch, G, max_seq, v_dim),
            B=1024,
        )

        # Simulate ingest of all 4096/2 = 2048 compact entries.
        # This fills level 0 twice and carries through level 1, with no overflow.
        for i in range(2048):
            c.ingest_chunk(
                torch.randn(B_batch, G, 2, k_dim),
                torch.randn(B_batch, G, 2, v_dim),
            )

        assert level_count(c, 2) > 0
        assert level_count(c, 0) == 0
        assert level_count(c, 1) == 0
        assert c.token_count == 4096


# ---------------------------------------------------------------------------
# Integration: attention-layer LogKV streaming boundaries
# ---------------------------------------------------------------------------

class TestAttentionStreamingBoundaries:
    @staticmethod
    def _make_attention() -> CausalSelfAttention:
        config = Config(
            block_size=16,
            padded_vocab_size=16,
            n_layer=1,
            n_head=2,
            n_query_groups=2,
            n_embd=8,
            rotary_percentage=0.5,
        )
        attn = CausalSelfAttention(config, block_idx=0)
        attn.kv_cache = attn.build_log_kv_cache(
            batch_size=1,
            max_seq_length=16,
            B=4,
            recent_size=2,
        )
        return attn

    @staticmethod
    def _random_qkv(attn: CausalSelfAttention, T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = attn.config
        q = torch.randn(1, config.n_head, T, config.head_size)
        k = torch.randn(1, config.n_query_groups, T, config.head_size)
        v = torch.randn(1, config.n_query_groups, T, config.head_size)
        return q, k, v

    def test_odd_prefill_tail_pairs_with_first_decode(self):
        """Inference prefill should defer a trailing singleton until the next streamed token."""
        attn = self._make_attention()
        cache = attn.kv_cache
        assert isinstance(cache, LogStructuredKVCache)

        q, k, v = self._random_qkv(attn, T=3)
        out = attn._log_kv_training_forward(q, k, v, B=1, T=3, reset_cache=True, defer_last_single=True)

        assert out.shape == (1, 3, attn.config.n_embd)
        assert cache.token_count == 2
        assert cache.recent_count == 2
        assert attn._log_kv_pending is not None

        q, k, v = self._random_qkv(attn, T=1)
        out = attn._log_kv_training_forward(q, k, v, B=1, T=1, reset_cache=False, defer_last_single=True)

        assert out.shape == (1, 1, attn.config.n_embd)
        assert attn._log_kv_pending is None
        assert cache.token_count == 4
        assert level_count(cache, 0) == 1
        assert cache.recent_count == 2

    def test_streaming_chunk_consumes_pending_before_new_pairs(self):
        """A later multi-token inference call should consume pending before grouping new tokens."""
        attn = self._make_attention()
        cache = attn.kv_cache
        assert isinstance(cache, LogStructuredKVCache)

        q, k, v = self._random_qkv(attn, T=3)
        attn._log_kv_training_forward(q, k, v, B=1, T=3, reset_cache=True, defer_last_single=True)
        assert attn._log_kv_pending is not None

        q, k, v = self._random_qkv(attn, T=3)
        attn._log_kv_training_forward(q, k, v, B=1, T=3, reset_cache=False, defer_last_single=True)

        assert attn._log_kv_pending is None
        assert cache.token_count == 6
        assert level_count(cache, 0) == 2
        assert cache.recent_count == 2


# ---------------------------------------------------------------------------
# Batched flush: chunked add_recent must equal strict 2-token streaming
# ---------------------------------------------------------------------------

class TestBatchedFlushEquivalence:
    """add_recent() flushes window overflow in one batch (one shift, vectorized
    pair merges). The resulting cache state must be bit-identical to feeding the
    same token stream in the strict 2-token cadence."""

    @staticmethod
    def _new_cache(recent: int = 16, B: int = 4, max_seq: int = 2048) -> LogStructuredKVCache:
        return LogStructuredKVCache(
            (1, 2, max_seq, 8), (1, 2, max_seq, 8), B=B, recent_size=recent,
            device=torch.device("cpu"), dtype=torch.float32,
        )

    @pytest.mark.parametrize("chunk_sizes", ([16], [2, 4, 14, 16, 6, 12]))
    def test_even_chunked_add_recent_matches_streaming(self, chunk_sizes):
        torch.manual_seed(0)
        T = 700  # even; drives many flushes and several binary carries (B=4)
        k = torch.randn(1, 2, T, 8)
        v = torch.randn(1, 2, T, 8)

        ref = self._new_cache()
        add_full_kv_in_chunks(ref, k, v, chunk_size=2)

        c = self._new_cache()
        i, si = 0, 0
        while i < T:
            n = min(chunk_sizes[si % len(chunk_sizes)], T - i)
            c.add_recent(k[:, :, i:i + n], v[:, :, i:i + n])
            i += n
            si += 1

        assert_cache_states_bit_identical(ref, c)

    def test_full_window_replacement_chunk(self):
        """n == recent_size arriving at a full window flushes the entire old
        window — the boundary case of the batched path."""
        torch.manual_seed(1)
        recent = 8
        k = torch.randn(1, 2, 2 * recent, 8)
        v = torch.randn(1, 2, 2 * recent, 8)

        ref = self._new_cache(recent=recent)
        add_full_kv_in_chunks(ref, k, v, chunk_size=2)

        c = self._new_cache(recent=recent)
        c.add_recent(k[:, :, :recent], v[:, :, :recent])
        c.add_recent(k[:, :, recent:], v[:, :, recent:])

        assert_cache_states_bit_identical(ref, c)


# ---------------------------------------------------------------------------
# Vectorized block prefill (inference): exact in-window, state-exact always
# ---------------------------------------------------------------------------

class TestBlockPrefill:
    """log_kv_prefill_block > 2 batches prefill attention. Within the recent
    window the outputs must equal strict streaming (block=2); past compaction
    the outputs may deviate boundedly, but the cache state trajectory — and
    therefore every subsequent decode step — must stay bit-identical."""

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
            batch_size=1,
            max_seq_length=64,
            B=4,
            recent_size=recent,
        )
        attn.log_kv_prefill_block = prefill_block
        return attn

    def _run_pair(self, recent: int, T: int) -> tuple:
        # Identical weights: re-seed before each construction so qkv/proj match.
        torch.manual_seed(0)
        a_stream = self._make_attention(recent, prefill_block=2)
        torch.manual_seed(0)
        a_block = self._make_attention(recent, prefill_block=64)

        torch.manual_seed(42)
        head_size = 4  # n_embd=8 / n_head=2
        q = torch.randn(1, 2, T, head_size)
        k = torch.randn(1, 2, T, head_size)
        v = torch.randn(1, 2, T, head_size)
        with torch.no_grad():
            y_stream = a_stream._log_kv_training_forward(
                q, k, v, B=1, T=T, reset_cache=True, defer_last_single=True)
            y_block = a_block._log_kv_training_forward(
                q, k, v, B=1, T=T, reset_cache=True, defer_last_single=True)
        return a_stream, a_block, y_stream, y_block

    def test_in_window_outputs_match_streaming(self):
        """No flush inside the sequence -> block prefill is the same math as
        streaming (differences only from fp32 matmul tiling)."""
        a_s, a_b, y_s, y_b = self._run_pair(recent=32, T=20)
        torch.testing.assert_close(y_b, y_s, atol=1e-5, rtol=1e-5)
        assert_cache_states_bit_identical(a_s.kv_cache, a_b.kv_cache)

    def test_post_compaction_state_and_decode_match_streaming(self):
        """Past compaction the prefill OUTPUTS deviate (bounded, documented),
        but cache state and subsequent decode logits must not."""
        a_s, a_b, y_s, y_b = self._run_pair(recent=8, T=40)
        assert torch.isfinite(y_b).all()
        assert_cache_states_bit_identical(a_s.kv_cache, a_b.kv_cache)

        torch.manual_seed(1)
        head_size = 4
        q = torch.randn(1, 2, 2, head_size)
        k = torch.randn(1, 2, 2, head_size)
        v = torch.randn(1, 2, 2, head_size)
        with torch.no_grad():
            d_s = a_s._log_kv_training_forward(q, k, v, B=1, T=2, reset_cache=False, defer_last_single=True)
            d_b = a_b._log_kv_training_forward(q, k, v, B=1, T=2, reset_cache=False, defer_last_single=True)
        assert torch.equal(d_s, d_b)


# ---------------------------------------------------------------------------
# Low-memory training Function: must equal the naive differentiable reference
# ---------------------------------------------------------------------------

class TestLowMemTrainingEquivalence:
    """``_log_kv_train_lowmem_forward`` (LogKVStreamTrainingAttention:
    graph-free forward stream + chunk-by-chunk backward replay) must reproduce
    the naive per-chunk-graph reference
    ``_log_kv_training_forward(reset_cache=True, defer_last_single=False)``
    exactly: outputs, input gradients, parameter gradients, cache end state."""

    @staticmethod
    def _make_attention() -> CausalSelfAttention:
        config = Config(
            block_size=64,
            padded_vocab_size=16,
            n_layer=1,
            n_head=4,
            n_query_groups=2,  # GQA: exercises the broadcast attention branch
            n_embd=16,
            rotary_percentage=0.5,
        )
        attn = CausalSelfAttention(config, block_idx=0)
        attn.kv_cache = attn.build_log_kv_cache(
            batch_size=1,
            max_seq_length=64,
            B=4,
            recent_size=4,  # small: compaction kicks in almost immediately
        )
        return attn

    def _make_pair_with_inputs(self, T: int, seed: int):
        # Identical weights: re-seed before each construction so qkv/proj match.
        torch.manual_seed(0)
        a_ref = self._make_attention()
        torch.manual_seed(0)
        a_new = self._make_attention()

        torch.manual_seed(seed)
        hs = a_ref.config.head_size
        q0 = torch.randn(1, 4, T, hs)
        k0 = torch.randn(1, 2, T, hs)
        v0 = torch.randn(1, 2, T, hs)
        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        return a_ref, a_new, (q1, k1, v1), (q2, k2, v2)

    @pytest.mark.parametrize("T", [1, 2, 5, 12, 33])
    def test_matches_reference_forward_and_grads(self, T):
        a_ref, a_new, (q1, k1, v1), (q2, k2, v2) = self._make_pair_with_inputs(T, seed=7)

        y_ref = a_ref._log_kv_training_forward(q1, k1, v1, B=1, T=T)
        y_new = a_new._log_kv_train_lowmem_forward(q2, k2, v2, B=1, T=T)

        # Same op sequence (both chunk through log_kv_chunk_attention semantics
        # with causal_tail), so outputs and cache trajectories are identical.
        assert torch.equal(y_new, y_ref)
        assert_cache_states_bit_identical(a_ref.kv_cache, a_new.kv_cache)

        # Position-dependent loss so gradient errors cannot cancel.
        torch.manual_seed(99)
        w = torch.randn_like(y_ref)
        (y_ref * w).sum().backward()
        (y_new * w).sum().backward()

        torch.testing.assert_close(q2.grad, q1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(k2.grad, k1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(v2.grad, v1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            a_new.proj.weight.grad, a_ref.proj.weight.grad, rtol=1e-6, atol=1e-6
        )

    def test_backward_ignores_forwards_cache_end_state(self):
        """Backward replays the stream from a reset cache, so mutating the
        cache between forward and backward (as interleaved micro-batches or
        activation-checkpoint recompute would) must not change gradients."""
        T = 12
        a_ref, a_new, (q1, k1, v1), (q2, k2, v2) = self._make_pair_with_inputs(T, seed=3)

        y_ref = a_ref._log_kv_training_forward(q1, k1, v1, B=1, T=T)
        y_new = a_new._log_kv_train_lowmem_forward(q2, k2, v2, B=1, T=T)

        # Trash the low-mem module's cache state before its backward runs.
        hs = a_new.config.head_size
        a_new.kv_cache.reset_parameters()
        a_new.kv_cache.add_recent(torch.randn(1, 2, 2, hs), torch.randn(1, 2, 2, hs))

        y_ref.sum().backward()
        y_new.sum().backward()
        torch.testing.assert_close(q2.grad, q1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(k2.grad, k1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(v2.grad, v1.grad, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Rank-1 truncation quality vs the exact eigh/QR/SVD it replaced
# ---------------------------------------------------------------------------

class TestRank1Approximation:
    """``_rank1_psd_from_factors`` / ``_rank1_cross_from_factors`` find their
    direction by iteration, not by solving, so they are near-optimal rather than
    optimal. That approximation feeds the Sigma/Gamma corrections and therefore
    the training objective whenever the second-order gate is open, so the error
    is pinned here against the exact decompositions it replaced — otherwise a
    drift in truncation quality is indistinguishable from a modelling problem.

    The metric is the RECONSTRUCTION error, not agreement of the directions:
    inside a degenerate eigenspace the direction is arbitrary and two equally
    optimal answers can be far apart, while the rank-1 approximation they induce
    is equally good. What must not regress is how well ``gamma * u v^T``
    reproduces the matrix being truncated.
    """

    # Budget for how much worse than the exact decomposition the truncation may
    # be, NOT a pin on the iteration count. Worst observed across these cases and
    # a dozen seeds is ~3.8e-4 (psd) / ~2e-4 (cross), so this leaves ~5x
    # headroom while still failing if _RANK1_SQUARINGS drops to 5 or below, or
    # if the reduction to the r x r core is wrong. Tighten rather than loosen.
    MAX_EXCESS_PSD = 2e-3
    MAX_EXCESS_CROSS = 2e-3

    N, D = 1024, 64

    @staticmethod
    def _exact_psd(factors):
        """The eigh path this replaced: exact top-eigen truncation."""
        ff = factors.float()
        gram = torch.matmul(ff, ff.mT)
        eigvals, eigvecs = torch.linalg.eigh(gram)
        top = eigvals[..., -1].clamp_min(0.0)
        coeff = eigvecs[..., -1]
        d = torch.matmul(coeff.unsqueeze(-2), ff).squeeze(-2)
        d = d / top.sqrt().clamp_min(1e-12).unsqueeze(-1)
        d = torch.where(top.unsqueeze(-1) > 1e-12, d, torch.zeros_like(d))
        return d, top

    @staticmethod
    def _exact_cross(lf, rf):
        """The QR+SVD path this replaced: exact top-singular truncation."""
        left, right = lf.float().mT, rf.float().mT
        q_l, r_l = torch.linalg.qr(left, mode="reduced")
        q_r, r_r = torch.linalg.qr(right, mode="reduced")
        u_c, s, vh_c = torch.linalg.svd(torch.matmul(r_l, r_r.mT), full_matrices=False)
        gamma = s[..., 0]
        lu = torch.matmul(q_l, u_c[..., :, :1]).squeeze(-1)
        ru = torch.matmul(q_r, vh_c.mT[..., :, :1]).squeeze(-1)
        keep = gamma.unsqueeze(-1) > 1e-12
        return torch.where(keep, lu, torch.zeros_like(lu)), torch.where(keep, ru, torch.zeros_like(ru)), gamma

    @staticmethod
    def _rel_err(M, gamma, u, v):
        approx = gamma[..., None, None] * u.unsqueeze(-1) * v.unsqueeze(-2)
        return (M - approx).flatten(-2).norm(dim=-1) / M.flatten(-2).norm(dim=-1).clamp_min(1e-30)

    def _factors(self, case, r):
        """Cases chosen to stress iteration: a degenerate top pair is the slowest
        to separate, and rank-deficient / vanishing input exercises the guards."""
        torch.manual_seed(1000 + r)
        g = torch.randn(self.N, r, self.D)
        if case == "generic":
            return g
        if case == "one_dominant":
            return g * torch.logspace(2, -2, r)[:, None]
        if case == "near_degenerate":
            return torch.nn.functional.normalize(g, dim=-1)
        if case == "rank_deficient":
            return torch.stack([g[:, 0, :]] * (r - 1) + [g[:, r - 1, :]], 1)
        if case == "tiny":
            return g * 1e-8
        if case == "zeros":
            return torch.zeros(self.N, r, self.D)
        raise AssertionError(case)

    CASES = ["generic", "one_dominant", "near_degenerate", "rank_deficient", "tiny", "zeros"]

    @pytest.mark.parametrize("r", [2, 3])
    @pytest.mark.parametrize("case", CASES)
    def test_psd_reconstruction_matches_exact_eigh(self, case, r):
        F = self._factors(case, r)
        M = torch.matmul(F.float().mT, F.float())

        u, top = _rank1_psd_from_factors(F)
        u_ex, top_ex = self._exact_psd(F)

        assert torch.isfinite(u).all() and torch.isfinite(top).all()
        assert (top >= 0).all(), "a top eigenvalue of a PSD matrix cannot be negative"
        # Direction is a unit vector, or exactly zero where the input vanished.
        norms = u.float().norm(dim=-1)
        assert torch.all(((norms - 1.0).abs() < 1e-4) | (norms == 0)), norms

        excess = (self._rel_err(M, top.float(), u.float(), u.float())
                  - self._rel_err(M, top_ex, u_ex, u_ex)).max().item()
        assert excess < self.MAX_EXCESS_PSD, (
            f"{case} r={r}: rank-1 reconstruction is {excess:.3e} worse than exact eigh"
        )

    @pytest.mark.parametrize("r", [2, 3])
    @pytest.mark.parametrize("case", CASES)
    def test_cross_reconstruction_matches_exact_svd(self, case, r):
        R = self._factors(case, r)
        L = self._factors(case, r) if case in ("zeros", "tiny") else torch.randn(self.N, r, self.D)
        M = torch.matmul(L.float().mT, R.float())

        lu, ru, gamma = _rank1_cross_from_factors(L, R)
        lu_ex, ru_ex, gamma_ex = self._exact_cross(L, R)

        assert torch.isfinite(lu).all() and torch.isfinite(ru).all() and torch.isfinite(gamma).all()
        assert (gamma >= 0).all(), "a singular value cannot be negative"

        excess = (self._rel_err(M, gamma.float(), lu.float(), ru.float())
                  - self._rel_err(M, gamma_ex, lu_ex, ru_ex)).max().item()
        assert excess < self.MAX_EXCESS_CROSS, (
            f"{case} r={r}: rank-1 reconstruction is {excess:.3e} worse than exact QR/SVD"
        )

    def test_exactly_recovers_a_true_rank1_input(self):
        """When the input really is rank 1 there is no approximation to make and
        the iteration has nothing to trade off — it must land on the answer."""
        torch.manual_seed(5)
        f = torch.nn.functional.normalize(torch.randn(self.N, self.D), dim=-1)
        scale = torch.rand(self.N, 1) * 3 + 0.5
        F = (scale * f).unsqueeze(1)  # (N, 1, D): a single factor
        u, top = _rank1_psd_from_factors(F)
        torch.testing.assert_close(top, scale.squeeze(-1).square(), rtol=1e-5, atol=1e-5)
        # Direction matches up to sign.
        assert (torch.einsum("nd,nd->n", u, f).abs() - 1.0).abs().max() < 1e-5


# ---------------------------------------------------------------------------
# Second-order gate: scale == 0 must SKIP the statistics, not multiply by zero
# ---------------------------------------------------------------------------

class TestSecondOrderGate:
    """A zero second-order scale makes every Sigma/Gamma term vanish, so the
    cache must not build the statistics that feed them — that is what makes the
    warmup steps cost the first-order price instead of the full price times
    zero. The skip has to be free of observable effects: same outputs, same
    compaction trajectory, same gradients."""

    @staticmethod
    def _make_attention(scale: float) -> CausalSelfAttention:
        torch.manual_seed(0)
        config = Config(
            block_size=64,
            padded_vocab_size=16,
            n_layer=1,
            n_head=4,
            n_query_groups=2,  # GQA: exercises the broadcast attention branch
            n_embd=16,
            rotary_percentage=0.5,
        )
        attn = CausalSelfAttention(config, block_idx=0)
        attn.kv_cache = attn.build_log_kv_cache(
            batch_size=1, max_seq_length=64, B=4, recent_size=4
        )
        attn.log_kv_second_order_scale = scale
        return attn

    @staticmethod
    def _inputs(T: int, seed: int):
        torch.manual_seed(seed)
        q = torch.randn(1, 4, T, 4)
        k = torch.randn(1, 2, T, 4)
        v = torch.randn(1, 2, T, 4)
        return tuple(t.clone().requires_grad_(True) for t in (q, k, v))

    @pytest.mark.parametrize("T", [5, 12, 33])
    def test_gate_zero_leaves_stats_unbuilt(self, T):
        attn = self._make_attention(0.0)
        q, k, v = self._inputs(T, seed=11)
        attn._log_kv_train_lowmem_forward(q, k, v, B=1, T=T)

        cache = attn.kv_cache
        assert cache.second_order is False
        for ell in range(cache.max_levels):
            for name in ("level_sigma_u", "level_sigma2", "level_gamma_a",
                         "level_gamma_b", "level_gamma"):
                buf = getattr(cache, name)[:, :, 0, ell]
                assert not buf.any(), f"{name}_{ell} was built despite a zero gate"

    def test_gate_zero_matches_stats_present_but_gated(self):
        """Skipping the statistics must be indistinguishable from building them
        and multiplying by a zero gate — the behaviour before the skip existed.

        Driven at the cache level because the attention entry points re-derive
        the flag from the scale, so this is the only place both sides can be
        held at a zero gate while differing in whether the statistics exist.
        """
        torch.manual_seed(3)
        kd = vd = 8

        def build():
            return LogStructuredKVCache(
                (1, 2, 128, kd), (1, 2, 128, vd), B=4, recent_size=4, dtype=torch.float32
            )

        c_built, c_skipped = build(), build()
        c_skipped.second_order = False
        for _ in range(12):
            k = torch.randn(1, 2, 4, kd)
            v = torch.randn(1, 2, 4, vd)
            c_built.add_recent(k, v)
            c_skipped.add_recent(k, v)

        # The two caches really do differ in whether the statistics were built.
        assert c_built.level_sigma2.any()
        assert not c_skipped.level_sigma2.any()

        q = torch.randn(1, 4, 4, kd)
        kb = torch.randn(1, 2, 4, kd)
        vb = torch.randn(1, 2, 4, vd)
        y_gated = log_kv_chunk_attention(c_built, q, kb, vb, kd ** -0.5, 0.0)
        y_skipped = log_kv_chunk_attention(c_skipped, q, kb, vb, kd ** -0.5, 0.0)
        assert torch.equal(y_gated, y_skipped)

    @pytest.mark.parametrize("T", [5, 12, 33])
    def test_gate_zero_still_trains(self, T):
        """Gradients must still flow to q/k/v and the projection with the gate
        closed — the warmup steps are real training steps."""
        attn = self._make_attention(0.0)
        q, k, v = self._inputs(T, seed=13)
        y = attn._log_kv_train_lowmem_forward(q, k, v, B=1, T=T)
        torch.manual_seed(5)
        (y * torch.randn_like(y)).sum().backward()
        for name, g in (("q", q.grad), ("k", k.grad), ("v", v.grad),
                        ("proj", attn.proj.weight.grad)):
            assert g is not None, f"no gradient reached {name}"
            assert torch.isfinite(g).all(), f"non-finite gradient at {name}"
            assert g.abs().sum() > 0, f"gradient at {name} is all zero"

    def test_flag_guard_blocks_the_external_tamper(self):
        """The original coupling repro flipped ``cache.second_order`` on a cache
        already holding compacted entries between forward and backward. The
        guarded property must refuse that outright rather than let it corrupt
        the next carry — this is the public-API half of the fix."""
        T = 33
        a = self._make_attention(1.0)
        q, k, v = self._inputs(T, seed=7)
        a._log_kv_train_lowmem_forward(q, k, v, B=1, T=T)
        assert has_compacted_level(a.kv_cache), "need a non-empty level for this test to mean anything"
        with pytest.raises(RuntimeError, match="cannot change `second_order`"):
            a.kv_cache.second_order = False
        # Refused, not partially applied.
        assert a.kv_cache.second_order is True

    def test_function_ignores_a_bypassed_flag_between_forward_and_backward(self):
        """Belt and suspenders: even if something reaches past the property
        guard by writing the private backing field directly, the streaming
        Function must still re-derive the flag itself before replaying rather
        than trusting whatever the cache carries — this is what makes the
        guard a hardening, not the only thing standing between here and the
        original 81%-gradient-error repro."""
        T = 33
        torch.manual_seed(99)

        def run(tamper):
            a = self._make_attention(1.0)
            q, k, v = self._inputs(T, seed=7)
            y = a._log_kv_train_lowmem_forward(q, k, v, B=1, T=T)
            if tamper:
                a.kv_cache._second_order = False  # bypasses the guard on purpose
            torch.manual_seed(99)
            (y * torch.randn_like(y)).sum().backward()
            return y, (q.grad, k.grad, v.grad)

        y_clean, g_clean = run(tamper=False)
        y_tampered, g_tampered = run(tamper=True)
        assert torch.equal(y_clean, y_tampered)
        for name, a_, b_ in zip("qkv", g_clean, g_tampered):
            assert torch.equal(a_, b_), f"d{name} drifted despite the Function owning the flag"

    def test_function_ignores_a_stale_flag_from_the_caller(self):
        """A caller that never set the flag (or left it from another layer at a
        different gate) must not silently get the first-order result."""
        T = 12
        scale = 1.0 / math.sqrt(self._make_attention(1.0).config.head_size)

        def run(preset):
            a = self._make_attention(1.0)
            a.kv_cache.second_order = preset  # stale/ambient value
            q, k, v = self._inputs(T, seed=7)
            return LogKVStreamTrainingAttention.apply(q, k, v, a.kv_cache, scale, 2, 1.0)

        assert torch.equal(run(preset=False), run(preset=True))

    def test_nonsemantic_level_count_keeps_host_mirror(self):
        """The hot non-semantic path keeps host counts to avoid per-chunk device sync."""
        attn = self._make_attention(1.0)
        cache = attn.kv_cache
        T = 40
        q, k, v = self._inputs(T, seed=17)
        attn._log_kv_train_lowmem_forward(q, k, v, B=1, T=T)
        assert cache._counts == [level_count(cache, ell) for ell in range(cache.L_alloc)]
        assert cache.level_count.shape == (cache.batch_size, cache.n_groups, cache.K_max, cache.L_alloc)
        assert cache.level_count.dtype == torch.int16
        assert has_compacted_level(cache)
        cache.reset_parameters()
        assert cache.level_count.sum().item() == 0
        assert cache._counts == [0] * cache.L_alloc

    def _built_cache(self, second_order=True):
        c = LogStructuredKVCache(
            (1, 2, 128, 8), (1, 2, 128, 8), B=4, recent_size=4, dtype=torch.float32
        )
        c.second_order = second_order
        return c

    def test_guard_permits_same_value_writes_on_a_live_cache(self):
        c = self._built_cache(second_order=True)
        for _ in range(8):
            c.add_recent(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
        assert has_compacted_level(c) or c.recent_count > 0
        c.second_order = True  # no-op: same value, must never raise regardless of cache state
        assert c.second_order is True

    def test_guard_permits_a_change_before_any_level_is_populated(self):
        """Only compacted LEVEL entries are at risk — a cache holding tokens
        purely in the recent window (no compaction has fired yet) has no
        stats regime to straddle, so switching must still be allowed."""
        c = self._built_cache(second_order=True)
        c.add_recent(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))
        assert not has_compacted_level(c), "test setup needs recent-only state, no compacted levels"
        c.second_order = False  # must not raise
        assert c.second_order is False

    def test_guard_blocks_both_directions_on_a_populated_level(self):
        for start in (True, False):
            c = self._built_cache(second_order=start)
            for _ in range(8):  # enough to fill level 0 (B=4) at least once
                c.add_recent(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
            assert has_compacted_level(c), "test setup needs a populated level"
            with pytest.raises(RuntimeError, match="cannot change `second_order`"):
                c.second_order = not start

    def test_guard_releases_after_reset(self):
        c = self._built_cache(second_order=True)
        for _ in range(8):
            c.add_recent(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
        with pytest.raises(RuntimeError):
            c.second_order = False
        c.reset_parameters()
        c.second_order = False  # allowed again: nothing left to straddle
        assert c.second_order is False


# ---------------------------------------------------------------------------
# Salience pinning (SnapKV-style observation window)
# ---------------------------------------------------------------------------

class TestSaliencePinning:
    """Observation-window pinning: at a FRESH inference prefill the prompt-tail
    queries score the prefix and the top tokens per KV group are kept as exact
    w=1 slots. Pins are DUPLICATES — the hierarchy/recent trajectory must stay
    bit-identical with pinning on or off — and pin_size=0 is a strict no-op."""

    T = 40
    NEEDLE = 2  # early position: compacted into the hierarchy long before the tail

    @staticmethod
    def _make_attention(pin_size: int) -> CausalSelfAttention:
        config = Config(
            block_size=64,
            padded_vocab_size=16,
            n_layer=1,
            n_head=4,
            n_query_groups=2,  # GQA: groups select independently
            n_embd=16,
            rotary_percentage=0.5,
        )
        attn = CausalSelfAttention(config, block_idx=0)
        attn.kv_cache = attn.build_log_kv_cache(
            batch_size=1, max_seq_length=64, B=4, recent_size=4,
            pin_size=pin_size,
        )
        attn.log_kv_pin_obs_window = 4
        return attn

    def _needle_qkv(self, attn: CausalSelfAttention):
        """Random stream plus one distinctive needle key that the observation
        window (the last log_kv_pin_obs_window queries — 'the question') is
        aligned with. Everything else is low-energy noise."""
        torch.manual_seed(5)
        hs = attn.config.head_size
        T = self.T
        q = torch.randn(1, 4, T, hs) * 0.1
        k = torch.randn(1, 2, T, hs) * 0.1
        v = torch.randn(1, 2, T, hs)
        direction = torch.zeros(hs)
        direction[0] = 1.0
        k[:, :, self.NEEDLE, :] = 8.0 * direction
        q[:, :, T - 4:, :] = 8.0 * direction
        return q, k, v

    def test_needle_gets_pinned_exactly(self):
        # pin_size 8 >= the 7-wide max-pool tie span around the needle, so the
        # needle index itself is guaranteed into the top-k regardless of how
        # topk breaks ties among equal pooled scores.
        attn = self._make_attention(pin_size=8)
        cache = attn.kv_cache
        q, k, v = self._needle_qkv(attn)

        with torch.no_grad():
            attn._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)

        assert cache.pin_count == 8
        idx = attn._log_kv_pin_indices  # (1, groups, n_pin), time-ordered
        assert idx is not None
        # Candidates end at T - recent_size: recent tokens are never pinned
        # (they are already exact; duplicating them would distort mass).
        assert int(idx.max()) < self.T - cache.recent_size
        # Every group pinned the needle, and its K/V are the EXACT originals.
        for g in range(2):
            row = (idx[0, g] == self.NEEDLE).nonzero()
            assert row.numel() == 1, f"group {g} did not pin the needle: {idx[0, g].tolist()}"
            j = int(row[0, 0])
            assert torch.equal(cache.pin_k[0, g, j], k[0, g, self.NEEDLE])
            assert torch.equal(cache.pin_v[0, g, j], v[0, g, self.NEEDLE])
        # Pins surface in the attention state as extra exact w=1 slots.
        state = cache.get_attention_state()
        _, _, real_w = real_state_tensors(state)
        assert real_w.size(2) == cache.total_slots
        assert int((real_w[0, 0] == 1).sum()) == cache.pin_count + cache.recent_count

    def test_hierarchy_trajectory_unchanged_by_pinning(self):
        torch.manual_seed(0)
        a_off = self._make_attention(pin_size=0)
        torch.manual_seed(0)
        a_on = self._make_attention(pin_size=8)
        q, k, v = self._needle_qkv(a_on)

        with torch.no_grad():
            a_off._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)
            a_on._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)

        # Pins DUPLICATE tokens: compaction/recent must not change at all.
        assert_cache_states_bit_identical(a_off.kv_cache, a_on.kv_cache)
        assert a_off.kv_cache.pin_count == 0
        assert a_on.kv_cache.pin_count == 8

    def test_disabled_or_short_prompt_is_noop(self):
        # pin_size=0: selection never runs.
        attn = self._make_attention(pin_size=0)
        q, k, v = self._needle_qkv(attn)
        with torch.no_grad():
            attn._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)
        assert attn.kv_cache.pin_count == 0
        assert attn._log_kv_pin_indices is None

        # Prompt no longer than recent_size: no candidates, no pins.
        attn2 = self._make_attention(pin_size=8)
        torch.manual_seed(1)
        hs = attn2.config.head_size
        with torch.no_grad():
            attn2._log_kv_training_forward(
                torch.randn(1, 4, 4, hs), torch.randn(1, 2, 4, hs), torch.randn(1, 2, 4, hs),
                B=1, T=4, reset_cache=True, defer_last_single=True,
            )
        assert attn2.kv_cache.pin_count == 0

    def test_decode_keeps_pins_and_reset_clears_them(self):
        attn = self._make_attention(pin_size=8)
        cache = attn.kv_cache
        q, k, v = self._needle_qkv(attn)
        with torch.no_grad():
            attn._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)
        assert cache.pin_count == 8

        # Decode steps never re-select (token_count > 0) and keep pins visible.
        torch.manual_seed(2)
        hs = attn.config.head_size
        with torch.no_grad():
            out = attn._log_kv_training_forward(
                torch.randn(1, 4, 1, hs), torch.randn(1, 2, 1, hs), torch.randn(1, 2, 1, hs),
                B=1, T=1, reset_cache=False, defer_last_single=True,
            )
        assert out.shape == (1, 1, attn.config.n_embd)
        assert cache.pin_count == 8

        cache.reset_parameters()
        assert cache.pin_count == 0
        assert torch.equal(cache.pin_k, torch.zeros_like(cache.pin_k))


# ---------------------------------------------------------------------------
# Integration: GPT.forward LogKV mask handling
# ---------------------------------------------------------------------------

class TestGPTForwardLogKVMask:
    def test_all_log_kv_blocks_do_not_require_mask_cache(self):
        config = Config(
            block_size=16,
            padded_vocab_size=16,
            n_layer=1,
            n_head=2,
            n_query_groups=2,
            n_embd=8,
            rotary_percentage=0.5,
        )
        model = GPT(config)
        model.set_log_kv_cache(batch_size=1, max_seq_length=16, B=4, recent_size=2)
        model.mask_cache = None

        idx = torch.randint(0, config.padded_vocab_size, (1, 3))
        input_pos = torch.arange(3)

        logits = model(idx, input_pos)

        assert logits.shape == (1, 3, config.padded_vocab_size)


# ---------------------------------------------------------------------------
# Integration: GPT.forward LogKV input_pos contract
# ---------------------------------------------------------------------------

class TestGPTForwardLogKVInputPos:
    @staticmethod
    def _make_model(batch_size: int = 1) -> GPT:
        config = Config(
            block_size=16,
            padded_vocab_size=16,
            n_layer=1,
            n_head=2,
            n_query_groups=2,
            n_embd=8,
            rotary_percentage=0.5,
        )
        model = GPT(config)
        model.set_log_kv_cache(batch_size=batch_size, max_seq_length=16, B=4, recent_size=2)
        return model

    def test_rejects_non_contiguous_input_pos(self):
        model = self._make_model()
        idx = torch.randint(0, model.config.padded_vocab_size, (1, 3))

        with pytest.raises(ValueError, match="append-only contiguous input_pos"):
            model(idx, torch.tensor([0, 2, 3]))

    def test_pending_token_counts_toward_next_expected_input_pos(self):
        model = self._make_model()

        idx = torch.randint(0, model.config.padded_vocab_size, (1, 3))
        model(idx, torch.arange(3))

        attn = model.transformer.h[0].attn
        cache = attn.kv_cache
        assert isinstance(cache, LogStructuredKVCache)
        assert cache.token_count == 2
        assert attn._log_kv_pending is not None

        idx_next = torch.randint(0, model.config.padded_vocab_size, (1, 1))
        with pytest.raises(ValueError, match="Expected 3..3"):
            model(idx_next, torch.tensor([2]))

        logits = model(idx_next, torch.tensor([3]))
        assert logits.shape == (1, 1, model.config.padded_vocab_size)

    def test_rejects_per_sample_batched_input_pos(self):
        model = self._make_model(batch_size=2)
        idx = torch.randint(0, model.config.padded_vocab_size, (2, 3))
        input_pos = torch.tensor([[0, 1, 2], [1, 2, 3]])

        with pytest.raises(ValueError, match="per-sample input_pos"):
            model(idx, input_pos)


# ---------------------------------------------------------------------------
# Regression: LogKV cache dtype reconciliation (bf16 model, fp32-default cache)
# ---------------------------------------------------------------------------

class TestLogKVDtypeReconcile:
    """A bf16 model must run LogKV inference even when the cache was allocated at
    the process-default dtype (fp32) — the eval path calls set_log_kv_cache()
    and the model is cast to bf16 separately. Without reconciliation the second
    prefill chunk would cat/matmul fp32 buffers against bf16 activations."""

    @staticmethod
    def _make_bf16_model() -> GPT:
        config = Config(
            block_size=16,
            padded_vocab_size=16,
            n_layer=1,
            n_head=2,
            n_query_groups=2,
            n_embd=8,
            rotary_percentage=0.5,
        )
        model = GPT(config).to(torch.bfloat16)
        # No dtype passed -> buffers use the process default (fp32 in the test
        # suite), deliberately mismatched against the bf16 model.
        model.set_log_kv_cache(batch_size=1, max_seq_length=16, B=4, recent_size=2)
        return model

    def test_bf16_inference_with_model_dtype_cache(self):
        """Cache now defaults to the model's dtype (bf16), not the process default.
        This avoids dtype mismatches that previously required runtime reconciliation."""
        model = self._make_bf16_model()
        cache = model.transformer.h[0].attn.kv_cache
        assert isinstance(cache, LogStructuredKVCache)
        # set_log_kv_cache now defaults dtype to next(model.parameters()).dtype
        assert cache.recent_k.dtype == torch.bfloat16  # allocated at model dtype

        # Prefill 5 tokens (crosses several 2-token chunks) then a decode step.
        idx = torch.randint(0, model.config.padded_vocab_size, (1, 5))
        logits = model(idx, torch.arange(5))
        assert logits.shape == (1, 5, model.config.padded_vocab_size)
        assert logits.dtype == torch.bfloat16

        # Buffers were reconciled to the activation dtype.
        assert cache.recent_k.dtype == torch.bfloat16
        assert cache.level_k.dtype == torch.bfloat16

        idx_next = torch.randint(0, model.config.padded_vocab_size, (1, 1))
        logits = model(idx_next, torch.tensor([5]))
        assert logits.shape == (1, 1, model.config.padded_vocab_size)
        assert logits.dtype == torch.bfloat16

    def test_convert_dtype_is_idempotent_noop(self):
        model = self._make_bf16_model()
        cache = model.transformer.h[0].attn.kv_cache

        cache._convert_dtype(torch.bfloat16)
        buf_before = cache.recent_k
        cache._convert_dtype(torch.bfloat16)  # already bf16 -> must be a no-op
        assert cache.recent_k is buf_before  # no new allocation
        assert cache.level_count.dtype == torch.int16  # counts stay indexed integer tensor


# ---------------------------------------------------------------------------
# Integration: LogKV inference equals dense attention within the recent window
# ---------------------------------------------------------------------------

class TestLogKVMatchesDense:
    """Within recent_size no compaction occurs, so LogKV inference must reproduce
    dense causal attention exactly. This validates the prefill fast path that
    underpins eval.py's loglikelihood scoring through the logKV cache."""

    @staticmethod
    def _make_model() -> GPT:
        config = Config(
            block_size=32,
            padded_vocab_size=16,
            n_layer=2,
            n_head=4,
            n_query_groups=2,
            n_embd=16,
            rotary_percentage=0.5,
        )
        torch.manual_seed(0)
        model = GPT(config)
        model.eval()
        return model

    @pytest.mark.parametrize("seq_len", (1, 2, 6, 7))
    def test_prefill_within_recent_matches_dense(self, seq_len):
        model = self._make_model()
        idx = torch.randint(0, model.config.padded_vocab_size, (1, seq_len))

        with torch.no_grad():
            dense = model(idx)  # dense causal attention (no cache)
            # recent_size >= seq_len -> entire prefill is exact (fast path, no compaction)
            model.set_log_kv_cache(batch_size=1, max_seq_length=seq_len, B=4, recent_size=16)
            logkv = model(idx, input_pos=torch.arange(seq_len))
            model.clear_kv_cache()

        torch.testing.assert_close(logkv, dense, atol=1e-4, rtol=1e-4)

    def test_semantic_prefill_within_recent_matches_dense(self):
        model = self._make_model()
        idx = torch.randint(0, model.config.padded_vocab_size, (1, 6))

        with torch.no_grad():
            dense = model(idx)
            model.set_log_kv_cache(
                batch_size=1,
                max_seq_length=6,
                B=4,
                recent_size=16,
                semantic_clusters=True,
                cluster_k_max=1,
                pin_size=0,
            )
            logkv = model(idx, input_pos=torch.arange(6))
            model.clear_kv_cache()

        torch.testing.assert_close(logkv, dense, atol=1e-4, rtol=1e-4)

    def test_streamed_decode_within_recent_matches_dense(self):
        """Prefill + one-token decodes must match a single dense forward over the
        whole sequence, as long as everything stays within recent_size."""
        model = self._make_model()
        full = torch.randint(0, model.config.padded_vocab_size, (1, 6))

        with torch.no_grad():
            dense = model(full)

            model.set_log_kv_cache(batch_size=1, max_seq_length=6, B=4, recent_size=16)
            out = []
            out.append(model(full[:, :4], input_pos=torch.arange(4)))  # prefill
            for pos in range(4, 6):  # decode token by token
                out.append(model(full[:, pos:pos + 1], input_pos=torch.tensor([pos])))
            model.clear_kv_cache()

        streamed = torch.cat(out, dim=1)
        torch.testing.assert_close(streamed, dense, atol=1e-4, rtol=1e-4)
