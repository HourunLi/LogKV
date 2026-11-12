"""Gamma: delayed Alpha evictions re-sort only level 0; equal-bytes spare levels."""

import numpy as np
import pytest
import torch

from litgpt.log_kv_cache import LogStructuredKVCache


def _stream(alpha, gamma, route="attach", N=2048, T=64, B=16, K=4, G=2, D=16, seed=3):
    torch.manual_seed(seed)
    c = LogStructuredKVCache(
        (1, G, N, D), (1, G, N, D), B=B, recent_size=T, semantic_flush_granularity=T, semantic_clusters=True,
        cluster_k_max=K, semantic_unified_route=route == "unified", semantic_anchor_mode="mid",
        allocate_second_order=False, semantic_replay_updates=True, alpha_exact_tokens=alpha, alpha_span_max_tokens=8,
        gamma_level0_reinsert=gamma, cos_cache=torch.ones(N, D), sin_cache=torch.zeros(N, D), rope_n_elem=D,
    )
    protos = torch.randn(K, D) * 2
    k = (protos[torch.randint(K, (N,))] + torch.randn(N, D))[None, None].expand(1, G, N, D).clone()
    k = k + 0.3 * torch.randn_like(k)
    v = torch.randn_like(k)
    rng = np.random.default_rng(seed)
    ends = np.zeros(N, dtype=bool)
    cuts = np.cumsum(rng.integers(3, 12, N))
    ends[cuts[cuts < N]] = True
    ends = ends.tolist()
    with torch.no_grad():
        for s in range(0, N, T):
            rows = slice(s, s + T)
            c._semantic_attention_plan()
            c.add_recent(k[:, :, rows], v[:, :, rows], k_raw=k[:, :, rows], span_ends=[ends[rows]] if alpha else None)
    w = c.level_w[0]
    live = w > 0
    span = float(((c.level_p_hi[0] - c.level_p_lo[0] + 1).float() * w)[live].sum() / w[live].sum())
    return c, int(live.sum()), span, float(w[live].max())


@pytest.mark.parametrize("route", ["attach", "unified"])
def test_level0_reinsert_keeps_the_ladder_of_a_run_without_alpha(route):
    """Evicted exact spans no longer re-merge whole clusters on every flush."""
    _, entries, span, heaviest = _stream(0, False, route)
    _, legacy_entries, legacy_span, legacy_heaviest = _stream(32, False, route)
    _, gamma_entries, gamma_span, gamma_heaviest = _stream(32, True, route)
    # The legacy re-ladder halves the live entries and smears positions.
    assert legacy_entries < 0.6 * entries and legacy_span > 10 * span and legacy_heaviest > 4 * heaviest
    # Gamma keeps the ladder of the run without Alpha (some late tokens still merge).
    assert gamma_entries >= 0.85 * entries and gamma_span <= 4 * span and gamma_heaviest <= heaviest


def _token_cache(gamma):
    return LogStructuredKVCache(
        (1, 1, 512, 8), (1, 1, 512, 8), B=8, recent_size=8, semantic_clusters=True, cluster_k_max=2,
        semantic_anchor_mode="mid", allocate_second_order=False, semantic_replay_updates=True,
        semantic_flush_granularity=8, alpha_exact_tokens=4, alpha_span_max_tokens=4, gamma_level0_reinsert=gamma,
        cos_cache=torch.ones(512, 8), sin_cache=torch.zeros(512, 8), rope_n_elem=8,
    )


def _commit(cache, k, pos, new=False):
    """Route tokens `pos` (with keys `k`, one row) into cluster 0 through the Alpha commit."""
    n = len(pos)
    kr = k.view(1, 1, n, -1)
    host = [list(pos)]
    positions = torch.tensor(host)
    if new:
        cache._semantic_new_clusters([(0, 0, 0, 0)], kr, kr, positions, host, record=False)
        if n > 1:
            cache._alpha_commit_joins([(0, 0, 0, list(range(1, n)))], kr, kr, positions, host, record=False)
    else:
        cache._alpha_commit_joins([(0, 0, 0, list(range(n)))], kr, kr, positions, host, record=False)


@pytest.mark.parametrize("gamma", [False, True])
def test_level0_reinsert_leaves_upper_levels_untouched(gamma):
    torch.manual_seed(5)
    c = _token_cache(gamma)
    B = c.B
    position = 100
    _commit(c, torch.randn(1, 8), [position], new=True)
    # One token at a time until level 0 is full, level 1 has one free slot and level 2 is used.
    for _ in range(4000):
        counts = c._semantic_counts[0][0][0]
        if counts[0] == B and counts[1] == B - 1 and counts[2] > 0:
            break
        position += 2
        _commit(c, torch.randn(1, 8), [position])
    else:
        pytest.fail("ladder state not reached")
    before = {name: value.clone() for name, value in c.named_buffers()}
    counts = list(c._semantic_counts[0][0][0])
    mass = float(c.level_w.sum())
    # Two evicted exact tokens, older than everything in the cluster.
    _commit(c, torch.randn(2, 8), [1, 3])
    assert float(c.level_w.sum()) == mass + 2
    assert c._semantic_n_total[0][0][0] == mass + 2 and c._semantic_p_hi_c[0][0][0] == position
    upper = slice(2, None)
    if gamma:
        # Level 0 overflows by two rows: their one carry fills level 1's free slot.
        assert c._semantic_counts[0][0][0][:3] == [B, B, counts[2]]
        for name in ("level_k", "level_v", "level_w", "level_p_lo", "level_p_hi", "level_sum_wp"):
            after, old = c.get_buffer(name)[0, 0, 0], before[name][0, 0, 0]
            assert torch.equal(after[upper], old[upper]), name
            assert torch.equal(after[1, :B - 1], old[1, :B - 1]), name
        # The carry is the two late tokens, the oldest rows of the re-sorted level 0.
        assert c.level_w[0, 0, 0, 1, B - 1] == 2 and c.level_p_lo[0, 0, 0, 1, B - 1] == 1
        assert c.level_p_hi[0, 0, 0, 1, B - 1] == 3
    else:
        # Legacy: the whole cluster is re-appended, so already merged entries merge again.
        assert not torch.equal(c.level_w[0, 0, 0, upper], before["level_w"][0, 0, 0, upper])


@pytest.mark.parametrize("alpha", [0, 256])
def test_level_slack_spends_spare_levels_on_width_at_equal_bytes(alpha):
    def cache(slack):
        return LogStructuredKVCache(
            (1, 1, 32768, 32), (1, 1, 32768, 32), B=128, recent_size=2048, semantic_flush_granularity=2048,
            semantic_clusters=True, cluster_k_max=12, semantic_anchor_mode="mid", allocate_second_order=False,
            semantic_replay_updates=True, alpha_exact_tokens=alpha, alpha_span_max_tokens=64,
            gamma_level_slack=slack, cos_cache=torch.ones(32768, 32), sin_cache=torch.zeros(32768, 32),
            rope_n_elem=32, dtype=torch.bfloat16,
        )

    classic = cache(2)
    if not alpha:
        assert (classic.B, classic.L_alloc) == (128, 7)  # Two spare levels: the classic ladder, unchanged.
    for slack in (0, 1, 3):
        c = cache(slack)
        assert c.alpha_budget_bytes[1] <= c.alpha_budget_bytes[0]
        assert c.alpha_baseline_B == 128
        if slack < 2:
            assert c.L_alloc < classic.L_alloc and c.B > classic.B
        else:
            assert c.L_alloc >= classic.L_alloc and c.B < classic.B


def test_gamma_switches_are_validated():
    args = dict(B=8, recent_size=8, semantic_clusters=True, cluster_k_max=2, allocate_second_order=False,
                cos_cache=torch.ones(8, 8), sin_cache=torch.zeros(8, 8), rope_n_elem=8)
    with pytest.raises(ValueError, match="gamma_top_merge"):
        LogStructuredKVCache((1, 1, 64, 8), (1, 1, 64, 8), gamma_top_merge="oldest", **args)
    with pytest.raises(ValueError, match="gamma_level_slack"):
        LogStructuredKVCache((1, 1, 64, 8), (1, 1, 64, 8), gamma_level_slack=-1, **args)
