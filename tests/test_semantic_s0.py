import numpy as np
import pytest

from litgpt.semantic_s0 import (
    RouteResult,
    RunningKeyScale,
    SweepAccumulator,
    route_dpmeans_segments,
    simulate_segment_ladders,
    summarize_entries,
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

