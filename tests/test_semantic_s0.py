import math

import numpy as np
import pytest
import torch

from litgpt.log_kv_cache import LogStructuredKVCache
from litgpt.semantic_s0 import (
    OfflineEntry,
    RouteResult,
    RunningKeyScale,
    Stage0DumpRecorder,
    SweepAccumulator,
    _merge_ladders_for_ward,
    route_dpmeans_segments,
    route_single_cluster_bprime_ladder,
    simulate_segment_ladders,
    summarize_entries,
    vanilla_logkv_compressed_entries,
    vanilla_logkv_full_cache_entries,
)


def test_route_dpmeans_uses_pure_semantic_nearest_cluster() -> None:
    k = np.asarray([[0.0], [0.1], [10.0], [10.1], [0.2]], dtype=np.float32)

    route = route_dpmeans_segments(k, lambda_new=1.0, g_max=float("inf"), gamma=0.5)

    assert route.cluster_count == 2
    assert route.cluster_ids.tolist() == [0, 0, 1, 1, 0]
    assert route.segment_ids.tolist() == [0, 0, 0, 0, 0]


def test_g_max_opens_new_segment_without_new_cluster() -> None:
    k = np.asarray([[0.0], [10.0], [0.1]], dtype=np.float32)

    route = route_dpmeans_segments(k, lambda_new=1.0, g_max=1, gamma=0.5)

    assert route.cluster_count == 2
    assert route.segment_count == 3
    assert route.cluster_ids.tolist() == [0, 1, 0]
    assert route.segment_ids.tolist() == [0, 0, 1]


def test_route_dpmeans_rejects_nonpositive_lambda_new() -> None:
    # A non-positive threshold forces (almost) every token into its own
    # singleton cluster (d2 >= 0 > lambda_new holds for virtually every
    # comparison) without ever raising -- a silently degenerate sweep cell
    # rather than a meaningful one.
    k = np.asarray([[0.0], [0.1], [0.2]], dtype=np.float32)
    with pytest.raises(ValueError, match="lambda_new"):
        route_dpmeans_segments(k, lambda_new=-1.0, g_max=float("inf"), gamma=0.5)
    with pytest.raises(ValueError, match="lambda_new"):
        route_dpmeans_segments(k, lambda_new=0.0, g_max=float("inf"), gamma=0.5)
    with pytest.raises(ValueError, match="lambda_new"):
        route_dpmeans_segments(k, lambda_new=float("nan"), g_max=float("inf"), gamma=0.5)


def test_route_dpmeans_rejects_negative_g_max() -> None:
    # A negative (or -inf) g_max forces (almost) every same-cluster arrival to
    # open a new segment (p - p_hi[c] >= 1 > g_max holds for virtually every
    # comparison) without ever raising -- pathologically frequent segment
    # breaks that still look like a normal, valid sweep result.
    k = np.asarray([[0.0], [0.1], [0.2]], dtype=np.float32)
    with pytest.raises(ValueError, match="g_max"):
        route_dpmeans_segments(k, lambda_new=1.0, g_max=-1.0, gamma=0.5)
    with pytest.raises(ValueError, match="g_max"):
        # math.isinf(-inf) is True, so a naive "isinf or >= 0" check would let
        # this slip through -- must be rejected explicitly.
        route_dpmeans_segments(k, lambda_new=1.0, g_max=-math.inf, gamma=0.5)


def test_route_dpmeans_accepts_boundary_g_max_values() -> None:
    # g_max=0 and g_max=+inf are legitimate degenerate endpoints (used
    # elsewhere in this file/the S0.0 sweep itself) and must not be rejected
    # by the new validation.
    k = np.asarray([[0.0], [0.1]], dtype=np.float32)
    route_dpmeans_segments(k, lambda_new=1.0, g_max=0.0, gamma=0.5)
    route_dpmeans_segments(k, lambda_new=1.0, g_max=math.inf, gamma=0.5)


def test_route_dpmeans_kmax_triggers_ward_before_new_cluster() -> None:
    k = np.asarray([[0.0], [10.0], [20.0]], dtype=np.float32)

    route = route_dpmeans_segments(k, lambda_new=1.0, g_max=math.inf, gamma=0.5, k_max=2)

    assert route.cluster_count == 2
    assert route.cluster_ids.tolist() == [0, 0, 1]
    assert route.cluster_sizes == [2, 1]
    assert route.new_cluster_attempt_count == 3
    assert route.k_max_binding_count == 1
    assert route.ward_merge_count == 1
    assert route.k_max_binding_rate == pytest.approx(1 / 3)
    assert route.ward_merged_token_mask.tolist() == [False, True, False]
    assert route.ward_touched_token_mask.tolist() == [True, True, False]


def test_route_dpmeans_kmax_one_is_single_cluster_degenerate_boundary() -> None:
    k = np.asarray([[0.0], [10.0], [20.0]], dtype=np.float32)

    route = route_dpmeans_segments(k, lambda_new=1.0, g_max=math.inf, gamma=0.5, k_max=1)

    assert route.cluster_count == 1
    assert route.cluster_ids.tolist() == [0, 0, 0]
    assert route.cluster_sizes == [3]
    assert route.ward_merge_count == 0
    assert route.k_max_binding_count == 0
    assert route.novelty_suppressed_count == 2


def test_route_single_cluster_bprime_ladder_is_one_sequential_cluster() -> None:
    # A single-cluster, same-B' position-order CONTROL for isolating what
    # semantic clustering contributes -- NOT the vanilla/existing LogKV
    # reference (that is vanilla_logkv_compressed_entries/vanilla_logkv_
    # full_cache_entries, tested below): this has no recent-window carve-out
    # and inserts single raw tokens (w=1) at level 0, matching the *new*
    # semantic-cluster ladder's mechanics, not vanilla's real w=2 pairing.
    # See route_single_cluster_bprime_ladder's docstring.
    route = route_single_cluster_bprime_ladder(5)
    assert route.cluster_ids.tolist() == [0, 0, 0, 0, 0]
    assert route.segment_ids.tolist() == [0, 0, 0, 0, 0]
    assert route.cluster_count == 1
    assert route.segment_count == 1
    assert route.cluster_sizes == [5]


def test_route_single_cluster_bprime_ladder_empty_input() -> None:
    route = route_single_cluster_bprime_ladder(0)
    assert route.cluster_ids.tolist() == []
    assert route.segment_ids.tolist() == []
    assert route.cluster_count == 0
    assert route.segment_count == 0
    assert route.cluster_sizes == []


def test_route_single_cluster_bprime_ladder_matches_binary_carry_reference() -> None:
    # Feeding the single-cluster route through simulate_segment_ladders must
    # reduce to exactly the same b_prime binary-carry construction as the
    # dedicated LogStructuredKVCache-equivalence test below (see
    # test_b_prime_members_relocate_unmerged_before_second_batch_merges):
    # exactly b_prime members land as b_prime still-unmerged, single-member
    # entries relocated to level 1, not merged pairs.
    b_prime = 4
    route = route_single_cluster_bprime_ladder(b_prime)
    entries, _ = simulate_segment_ladders(route, b_prime=b_prime, l_block=0)
    assert sorted(e.members for e in entries) == [[0], [1], [2], [3]]


def test_vanilla_logkv_compressed_entries_no_compaction_below_recent_size() -> None:
    entries, meta = vanilla_logkv_compressed_entries(3, b=4, recent_size=4)
    assert entries == []
    assert meta["recent_count"] == 3
    assert meta["compactable_token_count"] == 0


def test_vanilla_logkv_compressed_entries_even_overflow_pairs_oldest_first() -> None:
    # T=12, recent_size=4 -> overflow=8 (even): tokens 0..7 compacted as 4
    # consecutive pairs, tokens 8..11 stay exact in the recent window.
    entries, meta = vanilla_logkv_compressed_entries(12, b=4, recent_size=4)
    assert meta["recent_count"] == 4
    assert meta["compactable_token_count"] == 8
    assert sorted(e.members for e in entries) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert all(len(e.members) == 2 for e in entries)  # vanilla's real w=2 level-0 granularity


def test_vanilla_logkv_compressed_entries_odd_overflow_compacts_one_extra_token() -> None:
    # T=13, recent_size=4 -> raw overflow=9 (odd). log_kv_cache.py's
    # _flush_pairs always flushes a complete number of pairs, rounding UP --
    # so one extra token (10 total, not 9) gets compacted and the final
    # recent window ends up holding recent_size-1=3 tokens, not a full 4.
    entries, meta = vanilla_logkv_compressed_entries(13, b=4, recent_size=4)
    assert meta["recent_count"] == 3
    assert meta["compactable_token_count"] == 10
    assert sorted(e.members for e in entries) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def test_vanilla_logkv_compressed_entries_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="token_count"):
        vanilla_logkv_compressed_entries(-1, b=4, recent_size=4)
    with pytest.raises(ValueError, match="b must be"):
        vanilla_logkv_compressed_entries(10, b=0, recent_size=4)
    with pytest.raises(ValueError, match="b must be"):
        vanilla_logkv_compressed_entries(10, b=-1, recent_size=4)
    with pytest.raises(ValueError, match="recent_size"):
        vanilla_logkv_compressed_entries(10, b=4, recent_size=1)


def test_vanilla_logkv_compressed_entries_accepts_odd_b() -> None:
    # Unlike simulate_segment_ladders' b_prime, b here must only be positive
    # -- the real LogStructuredKVCache constructor places no evenness
    # requirement on B, and compact()'s pairwise merge never needed it either
    # (it always pairs up a 2*b-length concatenated sequence, even regardless
    # of whether b itself is). Odd b must not raise.
    entries, meta = vanilla_logkv_compressed_entries(30, b=3, recent_size=4)
    assert meta["compactable_token_count"] == 26
    assert sum(len(e.members) for e in entries) == 26


def test_vanilla_logkv_full_cache_entries_covers_every_token_below_recent_size() -> None:
    # T=3 <= recent_size=4: compressed_entries alone reports 0 entries/0
    # tokens (correct for "compressed slot variance", undefined so far -- see
    # vanilla_logkv_compressed_entries's docstring), but the *full* attention
    # state still has 3 real, w=1 exact slots. full_cache_entries must
    # represent all of them, each as its own trivial width-1 entry.
    entries, meta = vanilla_logkv_full_cache_entries(3, b=4, recent_size=4)
    assert sorted(e.members for e in entries) == [[0], [1], [2]]
    assert all(len(e.members) == 1 for e in entries)
    assert meta["compressed_entry_count"] == 0
    assert meta["recent_entry_count"] == 3
    assert meta["entry_count"] == 3
    assert meta["coverage_token_count"] == 3


def test_vanilla_logkv_full_cache_entries_appends_recent_tokens_after_compressed() -> None:
    # T=12, recent_size=4 -> compressed_entries covers [0,8) as 4 pairs (see
    # test_vanilla_logkv_compressed_entries_even_overflow_pairs_oldest_first);
    # full_cache_entries must add the remaining [8,12) back as individual
    # width-1 entries, so every position in [0, 12) is covered exactly once.
    compressed_entries, compressed_meta = vanilla_logkv_compressed_entries(12, b=4, recent_size=4)
    entries, meta = vanilla_logkv_full_cache_entries(12, b=4, recent_size=4)

    covered = sorted(pos for e in entries for pos in e.members)
    assert covered == list(range(12))  # every token accounted for exactly once
    assert meta["compressed_entry_count"] == len(compressed_entries) == 4
    assert meta["recent_entry_count"] == 4
    assert meta["entry_count"] == 8
    assert meta["coverage_token_count"] == 12
    # The width-1 recent entries contribute exactly 0 variance -- no special
    # casing needed in summarize_entries, a single-member entry's mean is
    # itself, so its SSE is definitionally 0.
    summary = summarize_entries(np.arange(12, dtype=np.float32).reshape(-1, 1), None, entries)
    assert summary["real_token_count"] == 12


def test_vanilla_logkv_full_cache_entries_level_counts_only_covers_compressed_prefix() -> None:
    # meta["level_counts"] from vanilla_logkv_compressed_entries is renamed to
    # "compressed_level_counts" in the full-cache meta (not carried over under
    # its original key): it only ever describes the 4 compressed-prefix
    # entries (see test_vanilla_logkv_full_cache_entries_appends_recent_tokens_
    # after_compressed for why T=12, b=4, recent_size=4 -> 4 compressed
    # entries), never the 4 appended width-1 recent entries, so it must NOT
    # sum to full-cache entry_count (8) -- a stale "level_counts" key next to
    # a full-cache entry_count would invite slicing full_entries by it as if
    # it covered every returned entry, which it does not.
    entries, meta = vanilla_logkv_full_cache_entries(12, b=4, recent_size=4)

    assert "level_counts" not in meta
    assert meta["entry_count"] == len(entries) == 8
    assert sum(meta["compressed_level_counts"].values()) == meta["compressed_entry_count"] == 4
    assert sum(meta["compressed_level_counts"].values()) != meta["entry_count"]


def _run_real_cache_add_recent(
    token_count: int, b: int, recent_size: int, chunk_size: int
) -> LogStructuredKVCache:
    """Feed 0..token_count-1 (as 1-D keys equal to their own position) through
    the real LogStructuredKVCache via add_recent(), in chunks of chunk_size.
    k[i] = float(i) makes every merged mean independently checkable: the true
    weighted mean over any set of member positions is just their arithmetic
    mean, and compact()'s weighted-mean merge is documented to reproduce
    exactly that regardless of merge hierarchy (log_kv_cache.py:685-691).
    """
    k_shape = (1, 1, max(token_count, 8), 1)
    v_shape = (1, 1, max(token_count, 8), 1)
    cache = LogStructuredKVCache(k_shape, v_shape, B=b, recent_size=recent_size, device=torch.device("cpu"), dtype=torch.float32)
    k = torch.arange(token_count, dtype=torch.float32).view(1, 1, token_count, 1)
    v = k.clone()
    offset = 0
    while offset < token_count:
        take = min(chunk_size, recent_size, token_count - offset)
        cache.add_recent(k[:, :, offset : offset + take, :], v[:, :, offset : offset + take, :])
        offset += take
    return cache


@pytest.mark.parametrize(
    ("token_count", "b", "recent_size"),
    [
        (12, 4, 4),   # even overflow
        (13, 4, 4),   # odd overflow (the parity edge case)
        (21, 4, 6),   # odd overflow, different recent_size
        (37, 8, 5),   # larger B, odd overflow
        (100, 8, 17), # a level-1+ carry actually fires
        (4, 4, 4),    # T == recent_size exactly: nothing compacted
        (3, 4, 4),    # T < recent_size: nothing compacted
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 3, None])  # None -> recent_size (largest legal single add_recent() call)
def test_vanilla_logkv_compressed_entries_matches_real_cache(token_count: int, b: int, recent_size: int, chunk_size: int | None) -> None:
    # Validates vanilla_logkv_compressed_entries' combinatorial derivation (recent-window
    # carve-out incl. odd-overflow parity, and w=2 level-0 pre-pairing)
    # against the actual LogStructuredKVCache -- not just the shared carry
    # primitive (_append_entry), which test_b_prime_members_relocate_
    # unmerged_before_second_batch_merges already covers independently.
    # Checked per-level *and* per-slot (weight and mean, not just aggregate
    # counts), across several add_recent() chunkings, since the real cache's
    # docstring claims (and this confirms) the final state is chunk-invariant.
    resolved_chunk_size = recent_size if chunk_size is None else chunk_size
    cache = _run_real_cache_add_recent(token_count, b, recent_size, resolved_chunk_size)
    entries, meta = vanilla_logkv_compressed_entries(token_count, b=b, recent_size=recent_size)

    assert cache.recent_count == meta["recent_count"]

    # vanilla_logkv_compressed_entries returns a flat list; re-slice it back into
    # per-level groups using meta["level_counts"], which records entries in
    # the same level-by-level order they were appended in (entry width alone
    # cannot recover level assignment once carries land at different depths).
    levels: list[list] = []
    cursor = 0
    for level in sorted(int(lvl_str) for lvl_str in meta["level_counts"]):
        count = meta["level_counts"][str(level)]
        levels.append(entries[cursor : cursor + count])
        cursor += count

    for level, level_entries in enumerate(levels):
        real_count = int(cache.level_count[level].item())
        assert real_count == len(level_entries), f"level {level}: count mismatch"
        real_w = getattr(cache, f"level_w_{level}")[0, 0, :real_count]
        real_k = getattr(cache, f"level_k_{level}")[0, 0, :real_count, 0]
        for slot_i, entry in enumerate(level_entries):
            assert real_w[slot_i].item() == pytest.approx(len(entry.members))
            assert real_k[slot_i].item() == pytest.approx(float(np.mean(entry.members)), abs=1e-4)
    for level in range(len(levels), cache.max_levels):
        assert int(cache.level_count[level].item()) == 0, f"unexpected occupied level {level}"


def test_b_prime_members_relocate_unmerged_before_second_batch_merges() -> None:
    # Mirrors LogStructuredKVCache._add_compact_entry/_binary_carry exactly
    # (verified empirically against the real cache, not just read off the
    # docstring): a level that is *empty* when a full b_prime-wide block
    # arrives absorbs it unmerged (LogStructuredKVCache._binary_carry's
    # "if self._counts[ell] == 0: self._set_level(...); return" -- no
    # compact() call). Only a level that *already* holds a full block merges
    # the two b_prime-wide blocks pairwise into one new b_prime-wide block
    # and keeps propagating. So exactly b_prime members must land as
    # b_prime still-separate, single-member entries (just relocated to
    # level 1), and it takes a *second* full batch (2*b_prime members total)
    # before the first real pairwise merge happens.
    b_prime = 4
    route_one_batch = RouteResult(
        cluster_ids=np.zeros(b_prime, dtype=np.int32),
        segment_ids=np.zeros(b_prime, dtype=np.int32),
        cluster_count=1,
        segment_count=1,
        cluster_sizes=[b_prime],
    )
    entries_one_batch, _ = simulate_segment_ladders(route_one_batch, b_prime=b_prime, l_block=0)
    assert sorted(e.members for e in entries_one_batch) == [[0], [1], [2], [3]]

    route_two_batches = RouteResult(
        cluster_ids=np.zeros(2 * b_prime, dtype=np.int32),
        segment_ids=np.zeros(2 * b_prime, dtype=np.int32),
        cluster_count=1,
        segment_count=1,
        cluster_sizes=[2 * b_prime],
    )
    entries_two_batches, _ = simulate_segment_ladders(route_two_batches, b_prime=b_prime, l_block=0)
    assert sorted(e.members for e in entries_two_batches) == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_l_block_padding_blocks_low_level_cross_segment_merge() -> None:
    # b_prime=4 needs a *second* full batch to trigger any real merge (see
    # test_b_prime_members_relocate_unmerged_before_second_batch_merges), so
    # this needs 8 members, not 4, to exercise an actual straddling pair.
    # Segment 0 has an odd length (5) so the natural (0,1)(2,3)(4,5)(6,7)
    # pairing at that merge event lands the segment boundary *inside* the
    # (4,5) pair when unblocked.
    route = RouteResult(
        cluster_ids=np.zeros(8, dtype=np.int32),
        segment_ids=np.asarray([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
        cluster_count=1,
        segment_count=2,
        cluster_sizes=[8],
    )
    k = np.asarray([[0.0]] * 5 + [[10.0]] * 3, dtype=np.float32)

    entries_unblocked, _ = simulate_segment_ladders(route, b_prime=4, l_block=0)
    summary_unblocked = summarize_entries(k, None, entries_unblocked)

    entries_blocked, meta_blocked = simulate_segment_ladders(route, b_prime=4, l_block=1)
    summary_blocked = summarize_entries(k, None, entries_blocked)

    assert any(entry.members == [4, 5] for entry in entries_unblocked)
    assert not any(4 in entry.members and 5 in entry.members for entry in entries_blocked)
    assert meta_blocked["pad_entry_count"] == 1
    assert summary_blocked["token_weighted_key_var"] < summary_unblocked["token_weighted_key_var"]


def test_kmax_ward_ladder_replay_uses_online_merge_events_not_final_labels() -> None:
    k = (np.arange(9, dtype=np.float32) * 10.0).reshape(-1, 1)
    route = route_dpmeans_segments(k, lambda_new=1.0, g_max=math.inf, gamma=0.5, k_max=2)

    event_entries, event_meta = simulate_segment_ladders(route, b_prime=4, l_block=0)
    final_label_route = RouteResult(
        cluster_ids=route.cluster_ids.copy(),
        segment_ids=route.segment_ids.copy(),
        cluster_count=route.cluster_count,
        segment_count=route.segment_count,
        cluster_sizes=route.cluster_sizes,
    )
    static_entries, static_meta = simulate_segment_ladders(final_label_route, b_prime=4, l_block=0)

    assert sorted(pos for entry in event_entries for pos in entry.members) == list(range(9))
    assert event_meta["entry_count"] == 7
    assert event_meta["entry_count"] != static_meta["entry_count"]
    assert sorted(entry.members for entry in event_entries) != sorted(entry.members for entry in static_entries)


def test_merge_ladders_for_ward_keeps_each_level_order_sorted() -> None:
    # Regression test for a real bug: keep and free grow as two *independent*
    # ladders before a Ward merge, so keep's own pre-existing content at some
    # level is not guaranteed to be chronologically newer than whatever just
    # cascaded up from the level below *in this merge* -- e.g. an active
    # cluster's level-1 entry (order=900) can be newer than a quiet cluster's
    # level-0 entry (order=5) that only now got folded into `ejected` on its
    # way up. Concatenating native (sorted) with ejected (also individually
    # sorted, but not merged back in) without a final sort would leave a
    # level's resident list out of order and let two chronologically-distant
    # entries get compacted together while a genuinely older entry sits
    # unmerged right next to them -- inflating entry spans/E[M] for no reason.
    keep_levels: list[list[OfflineEntry]] = [
        [OfflineEntry(members=[950], order=950), OfflineEntry(members=[951], order=951)],
        [OfflineEntry(members=[900], order=900)],
    ]
    free_levels: list[list[OfflineEntry]] = [
        [OfflineEntry(members=[5], order=5)],
    ]

    _merge_ladders_for_ward(keep_levels, free_levels, b_prime=2)

    for level_entries in keep_levels:
        orders = [entry.order for entry in level_entries]
        assert orders == sorted(orders)
    all_members = sorted(pos for level in keep_levels for entry in level for pos in entry.members)
    assert all_members == [5, 900, 950, 951]


def test_running_key_scale_matches_global_variance() -> None:
    scale = RunningKeyScale()
    scale.update(0, np.asarray([[[0.0], [2.0]]], dtype=np.float32))
    scale.update(0, np.asarray([[[4.0]]], dtype=np.float32))

    payload = scale.to_manifest()

    assert payload["0"]["count"] == [3]
    assert np.isclose(payload["0"]["s_h"][0], 8.0 / 3.0)


def test_cluster_sizes_are_not_decayed_by_segment_forgetting() -> None:
    # n_total (exposed as RouteResult.cluster_sizes) must count every real token
    # ever assigned to a cluster even though gamma repeatedly resets the centroid
    # mixing weight n_eff. algorithm-spec.md S5.5/S5.6 call out conflating the two
    # as a historical bug: Ward-merge cost and the O(log n) space bound both need
    # the true, undecayed size, and using the decayed one makes long-lived,
    # content-rich clusters look artificially "small" and cheap to merge away.
    k = np.asarray([[0.0], [0.01], [0.02], [0.03], [0.04]], dtype=np.float32)

    # g_max=0 forces every token after the first into a brand new segment (and,
    # with gamma=0.0, a full centroid reset) while staying one semantic cluster.
    route = route_dpmeans_segments(k, lambda_new=1.0, g_max=0, gamma=0.0)

    assert route.cluster_count == 1
    assert route.segment_count == 5
    assert route.cluster_sizes == [5]


def test_sweep_accumulator_relative_variance_is_scale_invariant() -> None:
    # Two (layer, KV-group) contributors with identical *relative* dispersion
    # (key_sse / s_h / token_count == 0.5) but s_h differing by 1e6x, matching
    # algorithm-spec.md S5.2's "k 的范数在不同层、不同 KV group 之间差好几个
    # 数量级". Combining raw SSE across such contributors -- which is what the
    # sweep script's "overall" accumulator does across layers/groups -- lets the
    # large-scale one swamp the small-scale one and hides its trend, violating
    # the CLAUDE.md S2.5 requirement to report Stage-0 statistics per layer x
    # head. The *_relative fields normalize by each contributor's own s_h before
    # combining, so they must stay comparable to what each contributor reports
    # on its own.
    small = SweepAccumulator()
    small.add(
        route=RouteResult(
            cluster_ids=np.zeros(0, dtype=np.int32),
            segment_ids=np.zeros(0, dtype=np.int32),
            cluster_count=1,
            segment_count=1,
            cluster_sizes=[10],
        ),
        ladder_meta={"entry_count": 1, "pad_entry_count": 0},
        summary={"nonpad_entry_count": 1, "real_token_count": 10, "key_sse": 50.0, "value_sse": 0.0},
        sh=10.0,
    )
    large = SweepAccumulator()
    large.add(
        route=RouteResult(
            cluster_ids=np.zeros(0, dtype=np.int32),
            segment_ids=np.zeros(0, dtype=np.int32),
            cluster_count=1,
            segment_count=1,
            cluster_sizes=[10],
        ),
        ladder_meta={"entry_count": 1, "pad_entry_count": 0},
        summary={"nonpad_entry_count": 1, "real_token_count": 10, "key_sse": 50_000_000.0, "value_sse": 0.0},
        sh=10_000_000.0,
    )
    assert np.isclose(small.finalize()["token_weighted_key_var_relative"], 0.5)
    assert np.isclose(large.finalize()["token_weighted_key_var_relative"], 0.5)

    combined = SweepAccumulator()
    for contributor_sse, contributor_sh in ((50.0, 10.0), (50_000_000.0, 10_000_000.0)):
        combined.add(
            route=RouteResult(
                cluster_ids=np.zeros(0, dtype=np.int32),
                segment_ids=np.zeros(0, dtype=np.int32),
                cluster_count=1,
                segment_count=1,
                cluster_sizes=[10],
            ),
            ladder_meta={"entry_count": 1, "pad_entry_count": 0},
            summary={"nonpad_entry_count": 1, "real_token_count": 10, "key_sse": contributor_sse, "value_sse": 0.0},
            sh=contributor_sh,
        )
    finalized = combined.finalize()

    # The scale-normalized field agrees with what each contributor reports alone.
    assert np.isclose(finalized["token_weighted_key_var_relative"], 0.5)
    # The raw absolute field, in contrast, is dominated by the large-scale
    # contributor and no longer resembles either individual contributor's value
    # (small=5.0, large=5_000_000.0) -- this is exactly the trap the relative
    # field exists to avoid when combining across layers/KV-groups.
    assert finalized["token_weighted_key_var"] > 1_000_000.0


def test_summarize_entries_reports_raw_widths_and_spans() -> None:
    # 8 same-cluster tokens with b_prime=4 land as 4 merged entries at level 2
    # (see test_b_prime_members_relocate_unmerged_before_second_batch_merges):
    # [0,1],[2,3],[4,5],[6,7], each covering 2 members with span 1. The raw
    # per-entry lists SweepAccumulator pools must match summarize_entries' own
    # aggregate fields, not just be present.
    route = RouteResult(
        cluster_ids=np.zeros(8, dtype=np.int32),
        segment_ids=np.zeros(8, dtype=np.int32),
        cluster_count=1,
        segment_count=1,
        cluster_sizes=[8],
    )
    k = np.arange(8, dtype=np.float32).reshape(-1, 1)
    entries, _ = simulate_segment_ladders(route, b_prime=4, l_block=0)
    summary = summarize_entries(k, None, entries)

    assert summary["entry_widths"] == [2.0, 2.0, 2.0, 2.0]
    assert summary["entry_spans"] == [1.0, 1.0, 1.0, 1.0]
    assert np.isclose(np.mean(summary["entry_widths"]), summary["entry_width_mean"])
    assert np.isclose(np.mean(summary["entry_spans"]), summary["entry_span_mean"])


def test_sweep_accumulator_global_quantiles_do_not_hide_outlier_entries() -> None:
    # A per-sample mean already dilutes a rare huge-span entry (9 tiny entries
    # + 1 span=30000 entry averages to 3000 within that one sample), and
    # quantiles taken over many samples' means dilutes it further. The
    # entry_*_global_* fields pool every individual entry's raw span/width
    # across all samples instead, so the outlier survives into the reported
    # tail undiluted.
    dummy_route = RouteResult(
        cluster_ids=np.zeros(0, dtype=np.int32),
        segment_ids=np.zeros(0, dtype=np.int32),
        cluster_count=1,
        segment_count=1,
        cluster_sizes=[1],
    )
    dummy_meta = {"entry_count": 1, "pad_entry_count": 0}

    acc = SweepAccumulator()
    for _ in range(9):
        acc.add(
            route=dummy_route,
            ladder_meta=dummy_meta,
            sh=1.0,
            summary={
                "nonpad_entry_count": 10,
                "real_token_count": 10,
                "key_sse": 0.0,
                "entry_width_mean": 1.0,
                "entry_span_mean": 0.0,
                "entry_widths": [1.0] * 10,
                "entry_spans": [0.0] * 10,
            },
        )
    acc.add(
        route=dummy_route,
        ladder_meta=dummy_meta,
        sh=1.0,
        summary={
            "nonpad_entry_count": 10,
            "real_token_count": 10,
            "key_sse": 0.0,
            "entry_width_mean": 1.0,
            "entry_span_mean": 3000.0,
            "entry_widths": [1.0] * 10,
            "entry_spans": [0.0] * 9 + [30000.0],
        },
    )

    result = acc.finalize()
    # Neither per-sample-mean field ever gets anywhere near the true outlier:
    # the mean-of-means caps out at 300 (a 100x understatement), and even its
    # own p99 (taken over only 10 per-sample means) can't exceed the single
    # already-diluted 3000 data point.
    assert result["entry_span_mean_mean"] == 300.0
    assert result["entry_span_mean_quantiles"]["p99"] <= 3000.0
    # The pooled, entry-level field reports the raw outlier undiluted.
    assert result["entry_global_count"] == 100
    assert result["entry_span_global_max"] == 30000.0


def _dummy_route(n: int = 1) -> RouteResult:
    return RouteResult(
        cluster_ids=np.zeros(0, dtype=np.int32),
        segment_ids=np.zeros(0, dtype=np.int32),
        cluster_count=1,
        segment_count=1,
        cluster_sizes=[n],
    )


_DUMMY_LADDER_META = {"entry_count": 1, "pad_entry_count": 0}


def test_sweep_accumulator_omits_coverage_meta_when_ladder_meta_lacks_it() -> None:
    # simulate_segment_ladders's meta (used by the semantic sweep cells and the
    # single-cluster-b'-budget baseline) never carries compactable_token_count/
    # recent_count/coverage_token_count/compressed_entry_count/recent_entry_count
    # -- every token there always lands in some entry by construction. finalize()
    # must leave the corresponding "{key}_mean"/covered_token_fraction fields out
    # entirely (not default them to 0, which would misleadingly read as "0 tokens
    # covered").
    acc = SweepAccumulator()
    acc.add(
        route=_dummy_route(10),
        ladder_meta=_DUMMY_LADDER_META,
        sh=1.0,
        summary={"nonpad_entry_count": 1, "real_token_count": 10, "key_sse": 0.0},
    )
    result = acc.finalize()

    for key in SweepAccumulator._META_MEAN_KEYS:
        assert f"{key}_mean" not in result
    assert "covered_token_fraction" not in result


def test_sweep_accumulator_coverage_meta_matches_vanilla_compressed_prefix() -> None:
    # Two samples fed through the real vanilla_logkv_compressed_entries meta
    # (mirroring how the sweep script's vanilla_logkv_compressed_prefix_baseline
    # accumulator is populated): T=12,recent_size=4 -> compactable=8, recent=4,
    # coverage=8 (see test_vanilla_logkv_full_cache_entries_appends_recent_
    # tokens_after_compressed); T=3,recent_size=4 -> compactable=0, recent=3,
    # coverage=0 (nothing compacted yet, see test_vanilla_logkv_full_cache_
    # entries_covers_every_token_below_recent_size). compressed_entry_count/
    # recent_entry_count are full-cache-only fields, absent from this meta, and
    # must stay absent from the aggregate too.
    acc = SweepAccumulator()
    for token_count in (12, 3):
        _, meta = vanilla_logkv_compressed_entries(token_count, b=4, recent_size=4)
        acc.add(
            route=_dummy_route(token_count),
            ladder_meta=meta,
            sh=1.0,
            summary={
                "nonpad_entry_count": meta["entry_count"],
                "real_token_count": meta["compactable_token_count"],
                "key_sse": 0.0,
            },
        )
    result = acc.finalize()

    assert result["compactable_token_count_mean"] == pytest.approx((8 + 0) / 2)
    assert result["recent_count_mean"] == pytest.approx((4 + 3) / 2)
    assert result["coverage_token_count_mean"] == pytest.approx((8 + 0) / 2)
    # 8 covered out of 12+3=15 source tokens -- well below 1.0, correctly
    # reflecting that the exact recent window is not represented at all here.
    assert result["covered_token_fraction"] == pytest.approx(8 / 15)
    assert "compressed_entry_count_mean" not in result
    assert "recent_entry_count_mean" not in result


def test_sweep_accumulator_coverage_meta_matches_vanilla_full_cache() -> None:
    # vanilla_logkv_full_cache_entries's meta covers every source token by
    # construction, so the aggregate's covered_token_fraction should read ~1.0
    # -- the complement of the compressed-prefix case above.
    acc = SweepAccumulator()
    token_count = 12
    _, meta = vanilla_logkv_full_cache_entries(token_count, b=4, recent_size=4)
    acc.add(
        route=_dummy_route(token_count),
        ladder_meta=meta,
        sh=1.0,
        summary={"nonpad_entry_count": meta["entry_count"], "real_token_count": token_count, "key_sse": 0.0},
    )
    result = acc.finalize()

    assert result["coverage_token_count_mean"] == pytest.approx(12.0)
    assert result["covered_token_fraction"] == pytest.approx(1.0)
    assert result["compressed_entry_count_mean"] == pytest.approx(4.0)
    assert result["recent_entry_count_mean"] == pytest.approx(4.0)


def test_sweep_accumulator_reports_genuinely_zero_coverage_not_absent() -> None:
    # Every sample has token_count <= recent_size, so vanilla_logkv_compressed_
    # entries reports coverage_token_count=0 for all of them (nothing compacted
    # yet -- a real, meaningful "0% covered" measurement, not "not computed").
    # covered_token_fraction must still be present and read 0.0: gating its
    # presence on "coverage_token_sum > 0" (rather than on "did any sample
    # supply coverage_token_count at all") would wrongly suppress this
    # legitimate zero, indistinguishable from the "never measured" case that
    # test_sweep_accumulator_omits_coverage_meta_when_ladder_meta_lacks_it
    # covers.
    acc = SweepAccumulator()
    for token_count in (3, 4):
        _, meta = vanilla_logkv_compressed_entries(token_count, b=4, recent_size=4)
        assert meta["coverage_token_count"] == 0  # sanity: both below/at recent_size
        acc.add(
            route=_dummy_route(token_count),
            ladder_meta=meta,
            sh=1.0,
            summary={"nonpad_entry_count": meta["entry_count"], "real_token_count": 0, "key_sse": 0.0},
        )
    result = acc.finalize()

    assert result["coverage_token_count_mean"] == pytest.approx(0.0)
    assert "covered_token_fraction" in result
    assert result["covered_token_fraction"] == pytest.approx(0.0)


def test_sweep_accumulator_covered_token_fraction_denominator_excludes_uncovered_samples() -> None:
    # SweepAccumulator.add()'s contract does not forbid mixing coverage-
    # bearing ladder_meta (vanilla_logkv_compressed_entries's) with
    # coverage-less ladder_meta (simulate_segment_ladders's, used by the
    # semantic sweep cells) within one accumulator -- the sweep script
    # happens not to do this today (each of its three accumulators is fed a
    # single consistent ladder_meta shape throughout), but the class itself
    # must not silently fold a coverage-less call's route.cluster_sizes into
    # covered_token_fraction's denominator, or that call's tokens would drag
    # the fraction toward 0 despite never having their coverage measured.
    acc = SweepAccumulator()
    # A large coverage-less contributor: if its 1000 source tokens leaked
    # into the denominator, covered_token_fraction would be ~8/1012 (~0.008)
    # instead of the correct 8/12.
    acc.add(
        route=_dummy_route(1000),
        ladder_meta=_DUMMY_LADDER_META,
        sh=1.0,
        summary={"nonpad_entry_count": 1, "real_token_count": 1000, "key_sse": 0.0},
    )
    _, meta = vanilla_logkv_compressed_entries(12, b=4, recent_size=4)
    # sanity, see test_vanilla_logkv_full_cache_entries_appends_recent_tokens_after_compressed
    assert meta["coverage_token_count"] == 8
    acc.add(
        route=_dummy_route(12),
        ladder_meta=meta,
        sh=1.0,
        summary={
            "nonpad_entry_count": meta["entry_count"],
            "real_token_count": meta["compactable_token_count"],
            "key_sse": 0.0,
        },
    )
    result = acc.finalize()

    assert result["covered_token_fraction"] == pytest.approx(8 / 12)


def test_value_var_reported_as_none_not_zero_when_any_sample_skipped_it() -> None:
    # --skip_value_var means summarize_entries never puts "value_sse" in summary
    # at all for that call. Reporting a fake 0.0 there reads as "value variance
    # genuinely measured to be zero" (values identical within every entry) --
    # a much stronger, false claim -- instead of "not computed".
    acc = SweepAccumulator()
    acc.add(
        route=_dummy_route(),
        ladder_meta=_DUMMY_LADDER_META,
        sh=1.0,
        summary={"nonpad_entry_count": 1, "real_token_count": 10, "key_sse": 0.0, "value_sse": 5.0},
        vh=1.0,
    )
    result_with_value = acc.finalize()
    assert result_with_value["value_var_available"] is True
    assert result_with_value["token_weighted_value_var"] == pytest.approx(0.5)

    # A second accumulator where one sample never had value data at all (the
    # --skip_value_var case): the whole aggregate must be flagged unavailable,
    # not silently averaged over only the samples that happened to have it
    # (that would mix a token_count_sum denominator that includes the skipped
    # sample's tokens with a value_sse_sum numerator that doesn't).
    acc2 = SweepAccumulator()
    acc2.add(
        route=_dummy_route(),
        ladder_meta=_DUMMY_LADDER_META,
        sh=1.0,
        summary={"nonpad_entry_count": 1, "real_token_count": 10, "key_sse": 0.0, "value_sse": 5.0},
        vh=1.0,
    )
    acc2.add(
        route=_dummy_route(),
        ladder_meta=_DUMMY_LADDER_META,
        sh=1.0,
        summary={"nonpad_entry_count": 1, "real_token_count": 10, "key_sse": 0.0},  # no value_sse
    )
    result_mixed = acc2.finalize()
    assert result_mixed["value_var_available"] is False
    assert result_mixed["token_weighted_value_var"] is None
    assert result_mixed["token_weighted_value_var_relative"] is None


def test_value_var_relative_normalizes_by_vh_not_sh() -> None:
    # token_weighted_value_var_relative must divide value SSE by a value-space
    # calibrated scale (vh), not the key-space s_h -- key and value vectors
    # have no reason to share a norm scale. sh=1.0 and vh=10.0 here: if the
    # bug regressed (dividing by sh instead), the result would be 10x too big.
    acc = SweepAccumulator()
    acc.add(
        route=_dummy_route(),
        ladder_meta=_DUMMY_LADDER_META,
        sh=1.0,
        sh_source="calibrated",
        summary={"nonpad_entry_count": 1, "real_token_count": 10, "key_sse": 0.0, "value_sse": 50.0},
        vh=10.0,
        vh_source="calibrated",
    )
    result = acc.finalize()
    assert result["token_weighted_value_var_relative"] == pytest.approx(0.5)  # 50/10/10, not 50/1/10
    assert result["vh_source"] == ["calibrated"]


def test_value_var_relative_unavailable_without_vh() -> None:
    # Value data present but no vh supplied (e.g. an old manifest without
    # value_scale calibration, used without --allow_fallback_sh) must not
    # silently normalize by sh instead -- it should report unavailable.
    acc = SweepAccumulator()
    acc.add(
        route=_dummy_route(),
        ladder_meta=_DUMMY_LADDER_META,
        sh=1.0,
        summary={"nonpad_entry_count": 1, "real_token_count": 10, "key_sse": 0.0, "value_sse": 50.0},
    )
    result = acc.finalize()
    assert result["value_var_available"] is True
    assert result["token_weighted_value_var"] == pytest.approx(5.0)
    assert result["token_weighted_value_var_relative"] is None
    assert result["vh_source"] == []


def _make_recorder(save_dtype: str) -> Stage0DumpRecorder:
    return Stage0DumpRecorder(
        output_dir=".",
        sample_index=0,
        sample_id="s0",
        layers={0},
        key_scale=RunningKeyScale(),
        value_scale=RunningKeyScale(),
        save_dtype=save_dtype,
    )


def test_stage0_dump_recorder_accepts_float32_and_float16() -> None:
    assert _make_recorder("float32").save_dtype is np.float32
    assert _make_recorder("float16").save_dtype is np.float16


def test_stage0_dump_recorder_rejects_invalid_save_dtype() -> None:
    # A typo like "fp32" used to silently fall through to float16 (the `else`
    # branch of a two-way ternary) instead of raising -- routing would then
    # quietly use a lower-precision k/v than the caller intended, without any
    # error to signal it. Direct instantiation (bypassing the CLI's `choices`
    # guard) must still catch this.
    with pytest.raises(ValueError, match="save_dtype"):
        _make_recorder("fp32")
