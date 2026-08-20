"""Offline Stage-0 helpers for the SemanticLogKV S0.0 sweep.

The code in this module is intentionally CPU-oriented and independent from the
production LogKV cache. It consumes dumped pre-RoPE keys/values and simulates
only the routing and entry grouping needed by the S0.0 question: does the gain
come from semantic grouping itself, or from better temporal segment boundaries?
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_int_list(text: str | Iterable[int]) -> list[int]:
    if isinstance(text, str):
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    return [int(x) for x in text]


def parse_g_max_list(text: str | Iterable[int | float]) -> list[float]:
    if not isinstance(text, str):
        return [float(x) for x in text]
    values: list[float] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(math.inf if raw.lower() in {"inf", "infty", "infinite", "∞"} else float(raw))
    return values


def format_g_max(value: float) -> str:
    return "inf" if math.isinf(float(value)) else f"{float(value):g}"


def quantiles(values: list[float], qs: Iterable[float] = (0.5, 0.9, 0.99)) -> dict[str, float]:
    if not values:
        return {f"p{int(q * 100):02d}": 0.0 for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


class RunningKeyScale:
    """Accumulate per-(layer, KV group) ``E||x-xbar||^2`` for a vector stream.

    Despite the name (it started out key-only), ``update()`` is a generic
    two-moment accumulator -- ``unused/semantic_stage0_dump.py`` also uses a
    second instance of this same class to calibrate a *value*-space scale
    (``s_h`` computed on ``v`` instead of ``k``), since key-space and
    value-space vectors have no reason to share a norm scale and normalizing
    value SSE by the key-space ``s_h`` would not actually make it comparable
    across layers/groups (see ``SweepAccumulator``'s docstring).
    """

    def __init__(self) -> None:
        self._sum: dict[int, np.ndarray] = {}
        self._sumsq: dict[int, np.ndarray] = {}
        self._count: dict[int, np.ndarray] = {}

    def update(self, layer: int, k_raw: np.ndarray) -> None:
        # k_raw: (G, T, D)
        k = np.asarray(k_raw, dtype=np.float32)
        if k.ndim != 3:
            raise ValueError(f"k_raw must have shape (G,T,D), got {k_raw.shape}")
        layer = int(layer)
        sums = k.sum(axis=1, dtype=np.float64)
        sumsq = np.square(k, dtype=np.float32).sum(axis=(1, 2), dtype=np.float64)
        counts = np.full(k.shape[0], k.shape[1], dtype=np.int64)
        if layer not in self._sum:
            self._sum[layer] = sums
            self._sumsq[layer] = sumsq
            self._count[layer] = counts
            return
        self._sum[layer] += sums
        self._sumsq[layer] += sumsq
        self._count[layer] += counts

    def to_manifest(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for layer in sorted(self._sum):
            sums = self._sum[layer]
            sumsq = self._sumsq[layer]
            counts = self._count[layer].astype(np.float64)
            means = sums / counts[:, None]
            sh = (sumsq / counts) - np.square(means).sum(axis=1)
            sh = np.maximum(sh, 0.0)
            out[str(layer)] = {
                "count": self._count[layer].astype(int).tolist(),
                "s_h": sh.astype(float).tolist(),
                "mean": means.astype(float).tolist(),
            }
        return out


class Stage0DumpRecorder:
    """Duck-typed recorder attached to selected attention layers during dump.

    ``save_dtype`` defaults to ``"float32"``: ``key_scale``/``value_scale``
    (``RunningKeyScale``, above) always accumulate ``s_h``/``vh`` in fp32 from
    the pre-quantization tensor, before ``record_pre_rope`` ever casts to
    ``save_dtype`` for storage. Saving at ``float16`` would make routing (which
    reads the saved, quantized k/v back off disk) diverge from the scale the
    threshold ``lambda_new = lambda_rel * s_h`` was calibrated against -- fp16
    rounding noise near the threshold boundary can flip a cluster assignment
    that the fp32-calibrated threshold never saw. ``float16`` is still
    supported for callers that explicitly accept that risk (e.g. a quick
    smoke test where storage size matters more than exact routing).
    """

    def __init__(
        self,
        *,
        output_dir: str | Path,
        sample_index: int,
        sample_id: str,
        layers: set[int],
        key_scale: RunningKeyScale,
        value_scale: RunningKeyScale,
        save_dtype: str = "float32",
        compressed: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.sample_index = int(sample_index)
        self.sample_id = str(sample_id)
        self.layers = {int(x) for x in layers}
        self.key_scale = key_scale
        self.value_scale = value_scale
        if save_dtype == "float32":
            self.save_dtype = np.float32
        elif save_dtype == "float16":
            self.save_dtype = np.float16
        else:
            raise ValueError(f"save_dtype must be 'float32' or 'float16', got {save_dtype!r}")
        self.compressed = bool(compressed)
        self.records: list[dict[str, Any]] = []

    def record_pre_rope(self, layer: int, k: Any, v: Any) -> None:
        layer = int(layer)
        if layer not in self.layers:
            return
        if int(k.shape[0]) != 1:
            raise ValueError(f"Stage0 dump requires batch size 1, got k shape {tuple(k.shape)}")
        k_f32 = k[0].detach().to("cpu").float().numpy()
        v_f32 = v[0].detach().to("cpu").float().numpy()
        self.key_scale.update(layer, k_f32)
        self.value_scale.update(layer, v_f32)
        k_np = k_f32.astype(self.save_dtype)
        v_np = v_f32.astype(self.save_dtype)

        name = f"sample_{self.sample_index:04d}_layer_{layer:02d}.npz"
        path = self.output_dir / name
        if self.compressed:
            np.savez_compressed(path, k_raw=k_np, v=v_np)
        else:
            np.savez(path, k_raw=k_np, v=v_np)
        self.records.append(
            {
                "layer": layer,
                "path": name,
                "shape": {"k_raw": list(k_np.shape), "v": list(v_np.shape)},
                "dtype": str(np.dtype(self.save_dtype)),
            }
        )


@dataclass
class RouteResult:
    cluster_ids: np.ndarray
    segment_ids: np.ndarray
    cluster_count: int
    segment_count: int
    cluster_sizes: list[int]


def route_single_cluster_bprime_ladder(token_count: int) -> RouteResult:
    """Single-cluster, same-B' position-order control -- NOT a vanilla-LogKV baseline.

    Every token lands in cluster 0, segment 0, in arrival order (no semantic
    clustering at all), so feeding this through simulate_segment_ladders/
    _append_entry isolates what semantic clustering itself contributes when
    the B' ladder budget and mechanics are held fixed to whatever the sweep is
    already using for its semantic cells. That is a real, useful control --
    but it must not be read as "the existing/vanilla LogKV" reference:

    1. It runs at the sweep's ``b_prime`` (default 8), not the deployed
       vanilla config's ``log_kv_B`` (512, exp/qwen1.7b-32k/base.yaml).
    2. It has no recent-window carve-out at all (every token, including the
       most recent ones, goes through compaction), unlike vanilla's
       ``recent_size`` (1024) tokens that stay exact and are never compacted.
    3. Level 0 here holds single raw tokens (w=1) per slot -- the *new*
       semantic-cluster design's ladder convention (CLAUDE.md S2.1: "ladder
       的 level 0 存单 token（w=1），不是现有方案的'2 token 合并'"). Vanilla's
       real level 0 (log_kv_cache.py's ``_flush_pairs``) pre-merges 2 raw
       tokens into one w=2 entry before it ever reaches the shared
       binary-carry mechanism -- a different base granularity, not just a
       different B'/recent_size.

    For the actual vanilla-LogKV S0.4 reference point, use
    ``vanilla_logkv_compressed_entries()``/``vanilla_logkv_full_cache_
    entries()`` instead, which reproduce all three of the above faithfully
    (validated against the real LogStructuredKVCache, see
    tests/test_semantic_s0.py) -- see the former's docstring for which of
    the two to prefer for the S0.4 decision itself.
    """
    token_count = int(token_count)
    if token_count == 0:
        return RouteResult(
            cluster_ids=np.zeros(0, dtype=np.int32),
            segment_ids=np.zeros(0, dtype=np.int32),
            cluster_count=0,
            segment_count=0,
            cluster_sizes=[],
        )
    return RouteResult(
        cluster_ids=np.zeros(token_count, dtype=np.int32),
        segment_ids=np.zeros(token_count, dtype=np.int32),
        cluster_count=1,
        segment_count=1,
        cluster_sizes=[token_count],
    )


def vanilla_logkv_compressed_entries(
    token_count: int, *, b: int, recent_size: int
) -> tuple[list[OfflineEntry], dict[str, Any]]:
    """Faithful, position-only reconstruction of the *real* vanilla LogKV's compressed-prefix entries.

    Returns only the entries that have actually been flushed into the
    compression ladder -- the last ``recent_size`` tokens (or fewer, see the
    parity note below) that ``add_recent()`` keeps exact are *not* included
    here at all, not even as trivial width-1 entries. That is deliberate, not
    an oversight: this function answers "of the tokens vanilla has actually
    compressed, how much variance sits in each compression slot", which is
    the S0.4-relevant question when the goal is to judge slot quality without
    diluting it by ~``recent_size`` structurally-zero-variance exact tokens
    (see the S0.4 primary/secondary-metric discussion in
    ``vanilla_logkv_full_cache_entries``'s docstring and ``docs/
    experiments.md``). When ``token_count <= recent_size`` this returns an
    empty list -- correctly: vanilla has compressed nothing yet, so
    "compressed-slot variance" is undefined, not zero. Use
    ``vanilla_logkv_full_cache_entries`` when you need every token in the
    prompt accounted for (e.g. as a sanity check that total token coverage
    matches ``token_count``).

    This -- either this function or ``vanilla_logkv_full_cache_entries`` --
    is the actual S0.4 "现有位置槽内方差" reference point -- unlike
    ``route_single_cluster_bprime_ladder`` (a single-cluster control built
    from the *new* semantic-cluster ladder's mechanics), this reproduces two
    structural properties specific to the deployed ``LogStructuredKVCache``
    (``log_kv_cache.py``), both load-bearing for getting a comparable number:

    1. **Recent window.** The last ``recent_size`` tokens are never compacted
       at all -- ``add_recent()`` keeps them exact in a sliding window and
       only flushes older tokens into the ladder on overflow. When
       ``token_count <= recent_size``, nothing is compacted.

       Parity subtlety: ``_flush_pairs()`` always flushes a *complete* number
       of pairs, rounding the raw overflow up to the next even count
       (log_kv_cache.py:1186, ``flush_len = 2 * ((overflow + 1) // 2)``). So
       when ``token_count - recent_size`` is odd, one extra token beyond the
       strict minimum gets compacted and the final recent window ends up
       holding ``recent_size - 1`` tokens, not a full ``recent_size``. This
       holds regardless of how the input is chunked into ``add_recent()``
       calls (verified empirically, not just reasoned about -- see the test).

    2. **Level-0 granularity.** Vanilla pre-merges 2 raw tokens into one
       ``w=2`` entry (log_kv_cache.py's ``_flush_pairs``, lines ~990-1018)
       *before* that entry ever reaches the shared binary-carry mechanism --
       unlike the new semantic design (and ``route_single_cluster_bprime_
       ladder``), whose level 0 holds single raw tokens (``w=1``). So the
       oldest-first compactable tokens here are paired up (0,1), (2,3), ...
       *before* being handed to ``_append_entry``, not one at a time.

    The binary-carry promotion itself (``_append_entry``) is untouched and
    reused as-is: it is a generic "insert one entry, cascade-merge on carry"
    primitive that does not care whether what it is handed is a single token
    or a pre-paired block, and it already matches ``LogStructuredKVCache.
    _add_compact_entry``/``_binary_carry`` exactly (see ``_append_entry``'s
    docstring). Only what gets fed to it, and the recent-window exclusion
    before that, differ between vanilla and the semantic design.

    ``b`` corresponds to ``LogStructuredKVCache``'s ``B`` constructor
    argument / the ``--log_kv_B`` CLI flag (512 in the deployed
    exp/qwen1.7b-32k/base.yaml config) -- named lowercase here, not ``B``,
    per glossary.md's note that bare ``B`` is overloaded across this
    codebase (batch size in tensor-shape comments vs. this per-level entry
    budget). Only required to be a positive integer, matching the real
    ``LogStructuredKVCache`` constructor, which places no evenness
    requirement on ``B`` -- unlike ``simulate_segment_ladders``'s
    ``b_prime``, which *does* require an even value, but for an unrelated
    reason specific to the semantic design (``PAD_INSERT`` alignment to
    ``2**l_block`` segment boundaries, a concept vanilla has no equivalent
    of). ``compact()``'s pairwise merge itself never needed evenness either:
    it always pairs up a concatenated ``2*b``-length sequence, which is even
    regardless of whether ``b`` itself is (verified against odd ``b`` in
    tests/test_semantic_s0.py, not just reasoned about). ``recent_size``
    corresponds to ``--log_kv_recent_size`` (1024 in that same config).

    Validated end-to-end (per-level occupancy *and* per-slot weight/mean, not
    just aggregate counts) against a real ``LogStructuredKVCache`` fed
    through ``add_recent()`` at several chunk sizes and both even/odd ``b``
    -- see tests/test_semantic_s0.py's ``test_vanilla_logkv_compressed_
    entries_matches_real_cache`` tests.
    """
    token_count = int(token_count)
    if token_count < 0:
        raise ValueError(f"token_count must be non-negative, got {token_count}")
    b = int(b)
    if b <= 0:
        raise ValueError(f"b must be a positive integer, got {b}")
    recent_size = int(recent_size)
    if recent_size < 2:
        raise ValueError(f"recent_size must be >= 2, got {recent_size} (matches LogStructuredKVCache's own bound)")

    if token_count <= recent_size:
        compactable = 0
    else:
        overflow = token_count - recent_size
        compactable = overflow + (overflow % 2)  # round up to an even count -- see docstring's parity note
    recent_count = token_count - compactable

    levels: list[list[OfflineEntry]] = [[]]
    for pair_start in range(0, compactable, 2):
        _append_entry(levels, OfflineEntry(members=[pair_start, pair_start + 1]), b)

    entries: list[OfflineEntry] = []
    level_counts: dict[int, int] = {}
    for level, level_entries in enumerate(levels):
        level_counts[level] = len(level_entries)
        entries.extend(level_entries)

    meta = {
        "entry_count": len(entries),
        "pad_entry_count": 0,  # vanilla is a single flat ladder -- no segment boundaries to pad
        "compactable_token_count": compactable,
        "recent_count": recent_count,
        "coverage_token_count": compactable,  # how many of token_count are represented by these entries
        "level_counts": {str(k): int(v) for k, v in sorted(level_counts.items())},
    }
    return entries, meta


def vanilla_logkv_full_cache_entries(
    token_count: int, *, b: int, recent_size: int
) -> tuple[list[OfflineEntry], dict[str, Any]]:
    """The full vanilla-LogKV attention state: compressed prefix + exact recent window.

    ``vanilla_logkv_compressed_entries`` alone omits the ``recent_size``
    tokens ``add_recent()`` keeps exact -- correct for asking "how much
    variance sits inside each compression slot", but if read as "the entire
    thing a query would attend to under deployed vanilla LogKV" it silently
    undercounts: at ``token_count <= recent_size`` the compressed-only view
    reports ``real_token_count=0``/variance 0 even though the real attention
    state has ``token_count`` exact, ``w=1`` slots. This function appends
    those recent tokens back in, each as its own trivial ``OfflineEntry``
    (width 1, therefore exactly 0 internal variance -- ``summarize_entries``
    needs no special-casing for this, a width-1 entry's SSE is definitionally
    0), so every position in ``[0, token_count)`` is covered by exactly one
    entry.

    **Which of the two functions to use for the S0.4 decision gate**: prefer
    ``vanilla_logkv_compressed_entries`` as the primary metric. The
    ``recent_size`` (1024 by default) exact slots this function adds are
    structurally zero-variance regardless of content, so folding them into a
    token-weighted average dilutes the "is compression itself doing well"
    signal -- more so the smaller ``token_count`` is relative to
    ``recent_size``. Use this full-cache version as a secondary sanity check
    (e.g. confirming total token coverage equals ``token_count``, or as an
    end-to-end reference if what is actually needed is "the whole state a
    query attends to"), and always report which one a given number came
    from -- they answer different questions and are not interchangeable.

    Meta adds ``compressed_entry_count``/``recent_entry_count`` (the latter
    numerically equal to ``recent_count`` here, since every recent token
    becomes exactly one width-1 entry, but conceptually distinct: one counts
    tokens still held exact, the other counts ``OfflineEntry`` objects
    created for them) on top of ``vanilla_logkv_compressed_entries``'s
    fields; ``entry_count``/``coverage_token_count`` are overwritten to
    reflect the full (compressed + recent) picture rather than the
    compressed-only one.

    ``level_counts`` is renamed to ``compressed_level_counts`` in the
    returned meta (rather than carried over under its original key): it only
    ever describes the compressed-prefix ladder levels (from the inherited
    ``vanilla_logkv_compressed_entries`` meta) and never included the
    appended recent-window entries, so ``sum(compressed_level_counts.values())
    == compressed_entry_count``, not ``entry_count`` -- leaving it named
    ``level_counts`` next to a full-cache ``entry_count`` invites slicing
    ``full_entries`` by it as if it covered every returned entry, which it
    does not (see ``test_vanilla_logkv_full_cache_entries_level_counts_only_
    covers_compressed_prefix`` in tests/test_semantic_s0.py).
    """
    entries, meta = vanilla_logkv_compressed_entries(token_count, b=b, recent_size=recent_size)
    recent_start = meta["compactable_token_count"]
    full_entries = list(entries)
    for pos in range(recent_start, int(token_count)):
        full_entries.append(OfflineEntry(members=[pos]))

    full_meta = dict(meta)
    full_meta["compressed_level_counts"] = full_meta.pop("level_counts")
    full_meta.update(
        {
            "compressed_entry_count": len(entries),
            "recent_entry_count": len(full_entries) - len(entries),
            "entry_count": len(full_entries),
            "coverage_token_count": int(token_count),
        }
    )
    return full_entries, full_meta


def route_dpmeans_segments(
    k_raw: np.ndarray,
    *,
    lambda_new: float,
    g_max: float,
    gamma: float = 0.5,
) -> RouteResult:
    """Strict serial DP-means routing with ``eta=0``.

    ``g_max`` controls segment creation inside the winning semantic cluster.
    ``gamma`` is the centroid count forgetting factor applied when a new segment
    opens, matching the algorithm spec. No ``K_max`` clipping or Ward merge is
    applied; this is the Stage-0 offline scientific measurement path.
    """

    k = np.asarray(k_raw, dtype=np.float32)
    if k.ndim != 2:
        raise ValueError(f"k_raw must have shape (T,D), got {k_raw.shape}")
    lambda_new = float(lambda_new)
    if not (math.isfinite(lambda_new) and lambda_new > 0.0):
        raise ValueError(
            f"lambda_new must be finite and > 0, got {lambda_new} -- a non-positive threshold "
            f"forces (almost) every token into its own cluster (d2 >= 0 > lambda_new is true for "
            f"virtually every comparison) without ever raising an error, silently producing a "
            f"degenerate all-singleton routing instead of a meaningful sweep cell."
        )
    g_max = float(g_max)
    if not (g_max == math.inf or (math.isfinite(g_max) and g_max >= 0.0)):
        raise ValueError(
            f"g_max must be finite and >= 0, or +inf, got {g_max} -- note -inf is deliberately "
            f"rejected too (math.isinf(-inf) is True, so a naive isinf check would let it slip "
            f"through): a negative/-inf g_max forces (almost) every same-cluster arrival to open a "
            f"new segment (p - p_hi[c] >= 1 > g_max is true for virtually every comparison) without "
            f"ever raising an error, silently producing pathologically frequent segment breaks."
        )
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0,1], got {gamma}")
    t_total, dim = k.shape
    if t_total == 0:
        return RouteResult(
            cluster_ids=np.zeros(0, dtype=np.int32),
            segment_ids=np.zeros(0, dtype=np.int32),
            cluster_count=0,
            segment_count=0,
            cluster_sizes=[],
        )

    centroids = np.empty((max(1, t_total), dim), dtype=np.float32)
    n_eff = np.zeros(t_total, dtype=np.float32)
    n_total = np.zeros(t_total, dtype=np.int64)
    p_hi = np.zeros(t_total, dtype=np.int64)
    current_segment = np.zeros(t_total, dtype=np.int32)

    cluster_ids = np.empty(t_total, dtype=np.int32)
    segment_ids = np.empty(t_total, dtype=np.int32)
    cluster_count = 0
    segment_count = 0

    for p in range(t_total):
        x = k[p]
        if cluster_count == 0:
            c = 0
            centroids[c] = x
            n_eff[c] = 1.0
            n_total[c] = 1
            p_hi[c] = p
            current_segment[c] = 0
            cluster_count = 1
            segment_count = 1
        else:
            diff = centroids[:cluster_count] - x
            d2 = np.einsum("kd,kd->k", diff, diff, optimize=True)
            c = int(np.argmin(d2))
            if float(d2[c]) > float(lambda_new):
                c = cluster_count
                centroids[c] = x
                n_eff[c] = 1.0
                n_total[c] = 1
                p_hi[c] = p
                current_segment[c] = 0
                cluster_count += 1
                segment_count += 1
            else:
                if not math.isinf(float(g_max)) and p - int(p_hi[c]) > float(g_max):
                    current_segment[c] += 1
                    segment_count += 1
                    n_eff[c] *= float(gamma)

                n_eff_pre = float(n_eff[c])
                n_eff[c] = n_eff_pre + 1.0
                if n_eff_pre == 0.0:
                    centroids[c] = x
                else:
                    centroids[c] = (n_eff_pre * centroids[c] + x) / float(n_eff[c])
                n_total[c] += 1
                p_hi[c] = p

        cluster_ids[p] = c
        segment_ids[p] = int(current_segment[c])

    return RouteResult(
        cluster_ids=cluster_ids,
        segment_ids=segment_ids,
        cluster_count=int(cluster_count),
        segment_count=int(segment_count),
        cluster_sizes=n_total[:cluster_count].astype(int).tolist(),
    )


@dataclass
class OfflineEntry:
    members: list[int]
    pad_count: int = 0

    @property
    def is_pad_only(self) -> bool:
        return not self.members


def _merge_entries(a: OfflineEntry, b: OfflineEntry) -> OfflineEntry:
    return OfflineEntry(members=a.members + b.members, pad_count=a.pad_count + b.pad_count)


def _append_entry(levels: list[list[OfflineEntry]], entry: OfflineEntry, b_prime: int) -> None:
    """Binary-counter carry, matching ``LogStructuredKVCache._add_compact_entry``/
    ``_binary_carry`` (``log_kv_cache.py``) exactly rather than an independent
    "pair up whenever this level reaches capacity" scheme.

    Level 0 accumulates single entries one at a time; once it reaches
    ``b_prime`` it hands the *whole* block up as one carry unit (matching
    ``_add_compact_entry``'s ``if self._counts[0] >= self.B: ... _binary_carry``).
    A level that is empty when a block arrives just absorbs it **unmerged**
    (``_binary_carry``'s ``if self._counts[ell] == 0: self._set_level(...); return``
    -- no ``compact()`` call, entries keep whatever width they arrived with).
    Only a level that is *already holding a full block* merges the two
    ``b_prime``-wide blocks pairwise into one new ``b_prime``-wide block
    (``compact()``'s "two B-slot blocks -> one B-slot block", halving entry
    count and roughly doubling each entry's width) and keeps propagating that
    merged result to the next level. So a cluster with exactly ``b_prime``
    members ends up as ``b_prime`` still-unmerged, single-member entries
    (relocated to level 1, not merged into ``b_prime/2`` pairs) -- verified
    empirically against ``LogStructuredKVCache`` directly: 4 single-token
    inserts with ``B=4`` leave ``level_count == [0, 4, 0, ...]`` with every
    level-1 entry at its original width, and only the *second* batch of 4
    triggers an actual pairwise merge, landing 4 width-2 entries at level 2.
    """
    levels[0].append(entry)
    if len(levels[0]) < b_prime:
        return
    block = levels[0][:b_prime]
    levels[0].clear()

    level = 1
    while True:
        while level >= len(levels):
            levels.append([])
        if not levels[level]:
            levels[level] = block
            return
        combined = levels[level] + block
        levels[level] = []
        block = [_merge_entries(combined[2 * i], combined[2 * i + 1]) for i in range(b_prime)]
        level += 1


def simulate_segment_ladders(
    route: RouteResult,
    *,
    b_prime: int = 8,
    l_block: int = 1,
) -> tuple[list[OfflineEntry], dict[str, Any]]:
    if b_prime <= 0 or b_prime % 2 != 0:
        raise ValueError(f"b_prime must be a positive even integer, got {b_prime}")
    if l_block < 0:
        raise ValueError(f"l_block must be non-negative, got {l_block}")
    align = 1 << int(l_block)

    ladders: dict[int, list[list[OfflineEntry]]] = {}
    active_segment: dict[int, int] = {}
    logical_count: dict[int, int] = {}
    pad_entries = 0

    for pos, (cluster, segment) in enumerate(zip(route.cluster_ids.tolist(), route.segment_ids.tolist())):
        cluster = int(cluster)
        segment = int(segment)
        if cluster not in ladders:
            ladders[cluster] = [[]]
            active_segment[cluster] = segment
            logical_count[cluster] = 0
        elif segment != active_segment[cluster]:
            count = (-logical_count[cluster]) % align
            for _ in range(count):
                _append_entry(ladders[cluster], OfflineEntry(members=[], pad_count=1), b_prime)
                pad_entries += 1
                logical_count[cluster] += 1
            active_segment[cluster] = segment
        _append_entry(ladders[cluster], OfflineEntry(members=[pos]), b_prime)
        logical_count[cluster] += 1

    entries: list[OfflineEntry] = []
    level_counts: dict[int, int] = {}
    for levels in ladders.values():
        for level, level_entries in enumerate(levels):
            level_counts[level] = level_counts.get(level, 0) + len(level_entries)
            entries.extend(level_entries)

    meta = {
        "entry_count": len(entries),
        "real_entry_count": sum(1 for e in entries if e.members),
        "pad_entry_count": pad_entries,
        "level_counts": {str(k): int(v) for k, v in sorted(level_counts.items())},
    }
    return entries, meta


def summarize_entries(k_raw: np.ndarray, v: np.ndarray | None, entries: list[OfflineEntry]) -> dict[str, Any]:
    k = np.asarray(k_raw, dtype=np.float32)
    vv = None if v is None else np.asarray(v, dtype=np.float32)
    token_count = 0
    key_sse = 0.0
    value_sse = 0.0
    widths: list[float] = []
    spans: list[float] = []
    entry_key_vars: list[float] = []
    entry_value_vars: list[float] = []

    for entry in entries:
        if not entry.members:
            continue
        idx = np.asarray(entry.members, dtype=np.int64)
        kk = k[idx]
        k_centered = kk - kk.mean(axis=0, keepdims=True)
        k_entry_sse = float(np.square(k_centered, dtype=np.float32).sum(dtype=np.float64))
        n = int(idx.size)
        key_sse += k_entry_sse
        token_count += n
        widths.append(float(n))
        spans.append(float(int(idx.max()) - int(idx.min())))
        entry_key_vars.append(k_entry_sse / max(n, 1))
        if vv is not None:
            val = vv[idx]
            v_centered = val - val.mean(axis=0, keepdims=True)
            v_entry_sse = float(np.square(v_centered, dtype=np.float32).sum(dtype=np.float64))
            value_sse += v_entry_sse
            entry_value_vars.append(v_entry_sse / max(n, 1))

    out = {
        "real_token_count": int(token_count),
        "nonpad_entry_count": int(len(widths)),
        "token_weighted_key_var": float(key_sse / max(token_count, 1)),
        "entry_mean_key_var": float(np.mean(entry_key_vars)) if entry_key_vars else 0.0,
        "key_sse": float(key_sse),
        "entry_width_mean": float(np.mean(widths)) if widths else 0.0,
        "entry_width_max": float(max(widths)) if widths else 0.0,
        "entry_span_mean": float(np.mean(spans)) if spans else 0.0,
        "entry_span_max": float(max(spans)) if spans else 0.0,
        "entry_width_quantiles": quantiles(widths),
        "entry_span_quantiles": quantiles(spans),
        # Raw per-entry values, not just this call's own aggregates -- so callers
        # that combine many summarize_entries() calls (SweepAccumulator) can pool
        # them into true global quantiles instead of taking quantiles of per-call
        # means, which dilutes/hides rare huge-span entries (see SweepAccumulator).
        "entry_widths": widths,
        "entry_spans": spans,
    }
    if vv is not None:
        out.update(
            {
                "token_weighted_value_var": float(value_sse / max(token_count, 1)),
                "entry_mean_value_var": float(np.mean(entry_value_vars)) if entry_value_vars else 0.0,
                "value_sse": float(value_sse),
            }
        )
    return out


class SweepAccumulator:
    """Accumulates S0.0 statistics for one ``(g_max, l_block)`` config.

    ``token_weighted_key_var``/``token_weighted_value_var`` are absolute,
    un-normalized units (raw squared distance in that layer/KV-group's key or
    value space). Summing these across *different* (layer, group) pairs -- as
    the "overall" accumulator in the sweep script does -- is only meaningful
    if every contributor shares the same intrinsic norm scale, which
    ``algorithm-spec.md`` S5.2 explicitly says is false for keys ("k 的范数在
    不同层、不同 KV group 之间差好几个数量级") and there is no reason to
    expect it to be true for values either -- key-space and value-space are
    different projections with no shared scale. A single high-magnitude layer
    can completely dominate an absolute cross-layer sum and hide a real
    signal in every other layer, which directly conflicts with the hard
    requirement in CLAUDE.md S2.5 / experiments.md that Stage 0 statistics be
    reported per layer x head. ``token_weighted_key_var_relative`` divides
    each contributor's key SSE by its own calibrated ``s_h`` (pass ``sh`` to
    ``add()``) before accumulating, so it stays comparable (and safely
    combinable) across layers/groups. ``token_weighted_value_var_relative``
    does the analogous thing for value SSE, but needs its *own* calibrated
    scale -- pass ``vh`` (the value-space ``E||v-vbar||^2``, from
    ``manifest["value_scale"]``, not the key-space ``s_h``) to populate it;
    dividing value SSE by the key-space ``s_h`` instead would not actually
    make it comparable across layers/groups.

    ``--skip_value_var`` means no value data was even read, so ``summary``
    has no ``"value_sse"`` key at all for that call -- treating that as
    "value_sse=0.0" (the previous behavior) reports a genuine-looking zero
    variance instead of "not computed", which reads as "values are identical
    within every entry", a much stronger and false claim. ``value_var_
    available``/``token_weighted_value_var`` (and the ``_relative``
    counterpart) reflect this: they go ``False``/``None`` if *any*
    contributing sample lacked value data (an all-or-nothing signal, since
    averaging only the samples that happened to have it would silently mix
    the token-count denominator across samples that were included in
    ``token_count_sum`` but excluded from ``value_sse_sum``).

    ``entry_width_mean_mean``/``entry_span_mean_mean`` (and their
    ``_quantiles``) are quantiles of each *sample's own mean* width/span, not
    quantiles of the individual entries themselves -- a single sample with
    999 tiny entries and 1 pathologically huge-span entry reports a per-sample
    mean that's already diluted 1/1000, so that outlier entry barely moves
    these fields even before being combined with other samples. S0.0/S0.5
    care about the tail of the true entry-level distribution (do a handful of
    entries end up with huge spans?), so ``add()`` also pools every entry's
    raw width/span (via ``summarize_entries``'s ``entry_widths``/
    ``entry_spans``) into ``entry_width_global_*``/``entry_span_global_*``,
    which are quantiles/max over that flat, entry-level pool and do not
    dilute outliers the way the per-sample-mean fields do.

    ``{key}_mean`` (for ``key`` in ``_META_MEAN_KEYS`` --
    ``compactable_token_count``, ``recent_count``, ``coverage_token_count``,
    ``compressed_entry_count``, ``recent_entry_count``) and
    ``covered_token_fraction`` surface the ``vanilla_logkv_compressed_
    entries``/``vanilla_logkv_full_cache_entries`` meta fields that ``add()``
    would otherwise silently drop. Without these, the sweep script's JSON
    output can only *say* in a free-text note that the compressed-prefix
    baseline reports 0 entries whenever ``token_count <= recent_size``, or
    that the full-cache baseline covers every token -- a reader has no field
    in the row itself to check either claim against. ``add()`` only
    populates a given ``{key}_mean`` when ``ladder_meta`` actually contains
    that key: ``simulate_segment_ladders``'s meta (used by the semantic
    sweep cells and the single-cluster-b'-budget baseline) has none of them,
    since every token there always lands in some entry, so these fields are
    simply absent (not zero) from accumulators that never saw them.
    ``covered_token_fraction`` is ``coverage_token_sum / total_source_tokens``
    across every ``add()`` call that supplied ``coverage_token_count``, where
    ``total_source_tokens`` comes from ``sum(route.cluster_sizes)`` (present
    on every call regardless of whether ``ladder_meta`` carries coverage
    fields). It should read close to 1.0 for ``vanilla_logkv_full_cache_
    baseline`` (every token is represented, by construction) and well below
    1.0 for ``vanilla_logkv_compressed_prefix_baseline`` whenever
    ``token_count`` is not much larger than ``recent_size`` (the exact
    recent window is not represented there at all -- see
    ``vanilla_logkv_compressed_entries``'s docstring).
    """

    _META_MEAN_KEYS = (
        "compactable_token_count",
        "recent_count",
        "coverage_token_count",
        "compressed_entry_count",
        "recent_entry_count",
    )

    def __init__(self) -> None:
        self.sample_groups = 0
        self.cluster_count_sum = 0.0
        self.segment_count_sum = 0.0
        self.entry_count_sum = 0.0
        self.nonpad_entry_count_sum = 0.0
        self.pad_entry_count_sum = 0.0
        self.token_count_sum = 0.0
        self.key_sse_sum = 0.0
        self.value_sse_sum = 0.0
        self.key_sse_norm_sum = 0.0
        self.value_sse_norm_sum = 0.0
        self.widths: list[float] = []
        self.spans: list[float] = []
        self.all_widths: list[float] = []
        self.all_spans: list[float] = []
        self.sh_sources: set[str] = set()
        self.vh_sources: set[str] = set()
        self.value_var_available: bool = True
        self.value_var_relative_available: bool = True
        self.meta_sums: dict[str, float] = {k: 0.0 for k in self._META_MEAN_KEYS}
        self.meta_counts: dict[str, int] = {k: 0 for k in self._META_MEAN_KEYS}
        self.total_source_tokens: float = 0.0
        self.coverage_token_sum: float = 0.0

    def add(
        self,
        *,
        route: RouteResult,
        ladder_meta: dict[str, Any],
        summary: dict[str, Any],
        sh: float,
        sh_source: str = "calibrated",
        vh: float | None = None,
        vh_source: str | None = None,
    ) -> None:
        self.sample_groups += 1
        self.cluster_count_sum += route.cluster_count
        self.segment_count_sum += route.segment_count
        self.entry_count_sum += float(ladder_meta.get("entry_count", 0))
        self.nonpad_entry_count_sum += float(summary.get("nonpad_entry_count", 0))
        self.pad_entry_count_sum += float(ladder_meta.get("pad_entry_count", 0))
        self.token_count_sum += float(summary.get("real_token_count", 0))
        self.key_sse_sum += float(summary.get("key_sse", 0.0))
        sh_safe = max(float(sh), 1e-12)
        self.key_sse_norm_sum += float(summary.get("key_sse", 0.0)) / sh_safe
        if "value_sse" in summary:
            self.value_sse_sum += float(summary["value_sse"])
            if vh is not None:
                self.value_sse_norm_sum += float(summary["value_sse"]) / max(float(vh), 1e-12)
                self.vh_sources.add(vh_source or "calibrated")
            else:
                self.value_var_relative_available = False
        else:
            self.value_var_available = False
            self.value_var_relative_available = False
        self.widths.append(float(summary.get("entry_width_mean", 0.0)))
        self.spans.append(float(summary.get("entry_span_mean", 0.0)))
        self.all_widths.extend(summary.get("entry_widths", []))
        self.all_spans.extend(summary.get("entry_spans", []))
        self.sh_sources.add(sh_source)

        # coverage/recent meta (only present on vanilla_logkv_compressed_
        # entries/vanilla_logkv_full_cache_entries's ladder_meta -- see the
        # class docstring's _META_MEAN_KEYS paragraph for why absence here is
        # not treated as zero).
        self.total_source_tokens += float(sum(route.cluster_sizes))
        for key in self._META_MEAN_KEYS:
            if key in ladder_meta:
                self.meta_sums[key] += float(ladder_meta[key])
                self.meta_counts[key] += 1
        if "coverage_token_count" in ladder_meta:
            self.coverage_token_sum += float(ladder_meta["coverage_token_count"])

    def finalize(self) -> dict[str, Any]:
        denom = max(self.sample_groups, 1)
        out = {
            "sample_groups": int(self.sample_groups),
            "cluster_count_mean": self.cluster_count_sum / denom,
            "segment_count_mean": self.segment_count_sum / denom,
            "entry_count_mean": self.entry_count_sum / denom,
            "nonpad_entry_count_mean": self.nonpad_entry_count_sum / denom,
            "pad_entry_count_mean": self.pad_entry_count_sum / denom,
            "token_weighted_key_var": self.key_sse_sum / max(self.token_count_sum, 1.0),
            # None (not 0.0) when any contributing sample had no value data at all
            # (--skip_value_var) -- see the class docstring for why reporting a
            # fake zero instead is actively misleading, not just "not computed".
            "token_weighted_value_var": (
                self.value_sse_sum / max(self.token_count_sum, 1.0) if self.value_var_available else None
            ),
            "value_var_available": self.value_var_available,
            # Scale-comparable version of token_weighted_key_var: each sample's key
            # SSE is divided by its own (layer, KV-group) s_h before accumulating, so
            # summing across layers/groups with different intrinsic key-norm scales
            # (as the sweep script's "overall" accumulator does) does not let one
            # high-magnitude layer dominate the result. See the class docstring.
            "token_weighted_key_var_relative": self.key_sse_norm_sum / max(self.token_count_sum, 1.0),
            # Same idea for value SSE, normalized by vh (a *value*-space calibrated
            # scale -- see the class docstring for why the key-space s_h would not
            # do). None if value data was unavailable, or if any contributing sample
            # provided value data without a usable vh (e.g. dump manifests written
            # before value-scale calibration existed, without --allow_fallback_sh).
            "token_weighted_value_var_relative": (
                self.value_sse_norm_sum / max(self.token_count_sum, 1.0)
                if self.value_var_relative_available
                else None
            ),
            "entry_width_mean_mean": float(np.mean(self.widths)) if self.widths else 0.0,
            "entry_span_mean_mean": float(np.mean(self.spans)) if self.spans else 0.0,
            "entry_width_mean_quantiles": quantiles(self.widths),
            "entry_span_mean_quantiles": quantiles(self.spans),
            # True entry-level distribution (pooled across every entry from every
            # sample), not quantiles of per-sample means -- see the class docstring
            # for why the _mean_* fields above can hide rare huge-span entries.
            "entry_global_count": len(self.all_spans),
            "entry_width_global_max": float(max(self.all_widths)) if self.all_widths else 0.0,
            "entry_span_global_max": float(max(self.all_spans)) if self.all_spans else 0.0,
            "entry_width_global_quantiles": quantiles(self.all_widths),
            "entry_span_global_quantiles": quantiles(self.all_spans),
            # "calibrated" (algorithm-spec.md S5.2's whole-calibration-set s_h) unless
            # any contributing sample had to fall back to this one prompt's own local
            # variance -- see _manifest_sh in unused/semantic_s0_sweep.py.
            "sh_source": sorted(self.sh_sources),
            "vh_source": sorted(self.vh_sources) if self.value_var_relative_available else [],
        }
        # Coverage/recent meta -- absent (not defaulted to 0) for accumulators
        # whose ladder_meta never carried a given key (e.g. the semantic sweep
        # cells and the single-cluster-b'-budget baseline, whose entries always
        # cover 100% of their input tokens by construction) -- see the class
        # docstring's _META_MEAN_KEYS paragraph.
        for key in self._META_MEAN_KEYS:
            if self.meta_counts[key]:
                out[f"{key}_mean"] = self.meta_sums[key] / self.meta_counts[key]
        # Gated on whether any sample supplied coverage_token_count at all, not
        # on coverage_token_sum > 0: a baseline where every sample has
        # token_count <= recent_size genuinely covers 0 tokens (nothing has
        # been compacted yet, per vanilla_logkv_compressed_entries's
        # docstring) -- that 0.0 is a real, reportable measurement, not a
        # "not computed" sentinel, so it must not be suppressed here.
        if self.meta_counts["coverage_token_count"] and self.total_source_tokens > 0:
            out["covered_token_fraction"] = self.coverage_token_sum / self.total_source_tokens
        return out


def load_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "manifest.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def manifest_base_dir(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_dir() else p.parent
