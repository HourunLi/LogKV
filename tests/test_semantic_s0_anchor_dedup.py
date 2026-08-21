import importlib.util
import math
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "unused" / "semantic_s0_anchor_dedup.py"
_SPEC = importlib.util.spec_from_file_location("semantic_s0_anchor_dedup_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

AnchorDedupAccumulator = _MODULE.AnchorDedupAccumulator
OfflineEntry = _MODULE.OfflineEntry
SCHEME_SEMANTIC = _MODULE.SCHEME_SEMANTIC
SCHEME_SINGLE_CLUSTER = _MODULE.SCHEME_SINGLE_CLUSTER
SCHEME_VANILLA_FULL = _MODULE.SCHEME_VANILLA_FULL
_add_ratios = _MODULE._add_ratios
_add_ratios_by_layer_group = _MODULE._add_ratios_by_layer_group
_attach_scheme_physical_width = _MODULE._attach_scheme_physical_width
_effective_g_max = _MODULE._effective_g_max
_entry_anchor_stats = _MODULE._entry_anchor_stats
_mid_anchor = _MODULE._mid_anchor


def test_mid_anchor_uses_round_half_up_integer_mean() -> None:
    assert _mid_anchor([0]) == 0
    assert _mid_anchor([0, 1]) == 1
    assert _mid_anchor([0, 2]) == 1
    assert _mid_anchor([2, 3, 4]) == 3


def test_entry_anchor_stats_deduplicates_lo_mid_hi() -> None:
    singleton = _entry_anchor_stats(OfflineEntry(members=[5]))
    adjacent_pair = _entry_anchor_stats(OfflineEntry(members=[0, 1]))
    sparse_pair = _entry_anchor_stats(OfflineEntry(members=[0, 2]))

    assert singleton["m"] == 1
    assert singleton["lo_hi_m"] == 1
    assert adjacent_pair["m"] == 2
    assert adjacent_pair["lo_hi_m"] == 2
    assert sparse_pair["m"] == 3
    assert sparse_pair["mid_is_new"] is True


def test_accumulator_skips_pad_entries_and_reports_e_m_distribution() -> None:
    acc = AnchorDedupAccumulator()
    entries = [
        OfflineEntry(members=[]),
        OfflineEntry(members=[5]),
        OfflineEntry(members=[0, 1]),
        OfflineEntry(members=[0, 2]),
    ]

    acc.add(entries=entries, ladder_meta={"entry_count": 4, "pad_entry_count": 1}, sh_source="calibrated")
    row = acc.finalize()

    assert row["sample_groups"] == 1
    assert row["entry_count_mean"] == 4
    assert row["real_entry_count_mean"] == 3
    assert row["pad_entry_count_mean"] == 1
    assert row["logical_anchor_count_mean"] == 6
    assert row["fixed3_anchor_count_mean"] == 12
    assert row["fixed3_entry_anchor_count_mean"] == 12
    assert row["fixed3_real_anchor_count_mean"] == 9
    assert row["lo_hi_anchor_count_mean"] == 5
    assert row["E_M"] == 2
    assert row["E_M_lo_hi"] == 5 / 3
    assert row["m_counts"] == {"1": 1, "2": 1, "3": 1}
    assert row["mid_new_count"] == 1
    assert math.isclose(row["gather_savings_fraction_vs_fixed3"], 1 / 2)
    assert math.isclose(row["gather_savings_fraction_vs_fixed3_real"], 1 / 3)


def test_add_ratios_attaches_semantic_vs_baseline_ratios() -> None:
    rows = [
        {
            "scheme": SCHEME_SINGLE_CLUSTER,
            "g_max": None,
            "l_block": None,
            "entry_count_mean": 10.0,
            "real_entry_count_mean": 10.0,
            "logical_anchor_count_mean": 20.0,
            "fixed3_anchor_count_mean": 30.0,
            "current_scheme_physical_slot_count_mean": 30.0,
            "lo_hi_anchor_count_mean": 15.0,
        },
        {
            "scheme": SCHEME_SEMANTIC,
            "g_max": "256",
            "l_block": 1,
            "entry_count_mean": 25.0,
            "real_entry_count_mean": 20.0,
            "logical_anchor_count_mean": 50.0,
            "fixed3_anchor_count_mean": 60.0,
            "current_scheme_physical_slot_count_mean": 60.0,
            "lo_hi_anchor_count_mean": 35.0,
        },
    ]

    _add_ratios(rows)

    semantic = rows[1]
    assert semantic["entry_count_mean_ratio_vs_single_cluster"] == 2.5
    assert semantic["logical_anchor_count_mean_ratio_vs_single_cluster"] == 2.5
    assert semantic["fixed3_anchor_count_mean_ratio_vs_single_cluster"] == 2.0
    assert semantic["current_scheme_physical_slot_count_mean_ratio_vs_single_cluster"] == 2.0


def test_add_ratios_by_layer_group_does_not_cross_contaminate_baselines() -> None:
    # Regression test for a real bug: calling _add_ratios directly on
    # by_layer_group rows (which have one row per (scheme, layer, group), not
    # one per scheme) would let its {scheme: row} dict comprehension silently
    # keep only the *last* (layer, group)'s baseline row per scheme and divide
    # every other (layer, group)'s semantic numbers by it. Deliberately order
    # layer 1's baseline last so a naive (ungrouped) _add_ratios call would
    # divide layer 0's semantic row by layer 1's baseline (10.0/100.0=0.1)
    # instead of its own (10.0/2.0=5.0) -- _add_ratios_by_layer_group must not
    # do that.
    rows = [
        {
            "scheme": SCHEME_SEMANTIC,
            "g_max": "256",
            "l_block": 1,
            "layer": 0,
            "group": 0,
            "entry_count_mean": 10.0,
        },
        {
            "scheme": SCHEME_SINGLE_CLUSTER,
            "g_max": None,
            "l_block": None,
            "layer": 0,
            "group": 0,
            "entry_count_mean": 2.0,
        },
        {
            "scheme": SCHEME_SEMANTIC,
            "g_max": "256",
            "l_block": 1,
            "layer": 1,
            "group": 0,
            "entry_count_mean": 150.0,
        },
        {
            "scheme": SCHEME_SINGLE_CLUSTER,
            "g_max": None,
            "l_block": None,
            "layer": 1,
            "group": 0,
            "entry_count_mean": 100.0,
        },
    ]

    _add_ratios_by_layer_group(rows)

    layer0_semantic = rows[0]
    layer1_semantic = rows[2]
    assert layer0_semantic["entry_count_mean_ratio_vs_single_cluster"] == 5.0
    assert layer1_semantic["entry_count_mean_ratio_vs_single_cluster"] == 1.5


def test_scheme_physical_width_uses_anchor_x3_for_semantic_and_entries_for_vanilla() -> None:
    semantic = {
        "scheme": SCHEME_SEMANTIC,
        "entry_count_mean": 20.0,
        "fixed3_anchor_count_mean": 60.0,
    }
    vanilla = {
        "scheme": SCHEME_VANILLA_FULL,
        "entry_count_mean": 30.0,
        "fixed3_anchor_count_mean": 90.0,
    }

    _attach_scheme_physical_width(semantic)
    _attach_scheme_physical_width(vanilla)

    assert semantic["current_scheme_physical_slot_count_mean"] == 60.0
    assert semantic["current_scheme_physical_slot_note"] == "anchor_entry_x3_no_gather"
    assert vanilla["current_scheme_physical_slot_count_mean"] == 30.0
    assert vanilla["current_scheme_physical_slot_note"] == "vanilla_full_cache_compressed_prefix_plus_exact_recent"


def test_effective_g_max_forces_inf_at_l_block_zero() -> None:
    assert _effective_g_max(256, 0) == float("inf")
    assert _effective_g_max(256, 1) == 256
