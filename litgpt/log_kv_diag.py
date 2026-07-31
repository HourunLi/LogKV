"""Diagnostic decomposition for LogKV slot attention — score/value oracle grid.

For ANY bucketing of the tokens into slots ``s`` (each slot covering a token
set ``S_s``), attention factors EXACTLY:

    L_s  = logsumexp_{j in S_s}(z_j)              # slot log-mass, z_j = scale·q·k_j
    V_s  = sum_{j in S_s} softmax_{j in S_s}(z_j)·v_j
    out  = softmax_s(L_s)·V_s   ==  dense attention          (algebraic identity)

The production path (``log_kv_slot_attention``) replaces both sides with a
first-order approximation:

    L_s ≈ scale·q·mean_j(k_j) + λ·log(w_s)        (score side)
    V_s ≈ mean_j(v_j)                              (value side)

This module swaps the two sides in and out independently to build a 2×2
attribution grid, so a benchmark drop can be split into "scoring error" vs
"read-out error":

    mode        L_s (score)              V_s (value)          answers
    --------    ---------------------    -----------------    -------------------
    baseline    approx (production)      mean(v)              control (== prod)
    s_oracle    exact logsumexp          mean(v)              how much is scoring
    v_oracle    approx (production)      exact softmax·v      how much is read-out
    exact       exact                    exact                self-check (== dense)
    dense       no bucketing — softmax over every raw token   independent oracle

``exact`` and ``dense`` must agree (two independent routes to the same identity);
``baseline`` must reproduce production bit-for-bit (it delegates to it).

``summary()`` reduces the accumulated stats to three tables, labeled with the
mode they were collected under:
  * ``by_level`` — per (layer, slot_width): score side ``logit_mae/logit_rmse``
    (approx vs exact slot logit) + ``intra_slot_logit_var``, and value side
    ``value_mae/value_rel`` (mean-pooled vs exact within-slot read-out). These are
    measurements of the CURRENT cache state, taken identically under every mode —
    they do not collapse to zero in oracle modes; across-mode dumps differ only
    through hidden-state drift caused by the substituted outputs.
  * ``by_layer_output`` — per layer: relative L2 error of the baseline / s_oracle
    / v_oracle outputs vs ``exact``, all four corners evaluated on the SAME
    hidden states, so one run yields a drift-free D2 attribution.
  * ``peakiness`` — per layer: running max + p99.9 of |scale·q·k| over a
    deterministic sample buffer, with the pool size ``n``.

Scope / preconditions (enforced; also see ``diag_block_attention``):
  * Diagnostics run with ``pin_size == 0`` and on a FRESH prefill only. The
    slot→token span mapping assumes every slot covers a contiguous, time-ordered
    run of the exact prefix tokens (``get_attention_state`` invariant). Salience
    pins break that (scattered duplicates), and a decode continuation / pending
    tail breaks ``token_count == len(prefix)``. Both are rejected.
  * ``mode == "off"`` (default) is fully inert: ``diag_block_attention`` is never
    reached from ``model.py`` unless a ``diag_mode(...)`` context is active, so
    production training/eval keeps its exact numerics and performance.
"""

import contextlib

import torch

from litgpt.log_kv_cache import append_exact_tokens, log_kv_slot_attention

_MODES = ("off", "baseline", "s_oracle", "v_oracle", "exact", "dense")
# Per-layer cap on the deterministic |scale·q·k| sample buffer backing the
# summary()-time p99.9 (see DiagState.add_peak). ~1 MB fp32 per layer.
_PEAK_BUF_CAP = 262_144
# Modes whose slot logit is the exact within-slot logsumexp (score oracle).
_L_EXACT = ("s_oracle", "exact")
# Modes whose read-out is the exact within-slot softmax over v (value oracle).
_V_EXACT = ("v_oracle", "exact")


# ======================================================================
# Global diagnostic state
# ======================================================================


class DiagState:
    """Global singleton holding the active diagnostic mode + accumulated stats.

    ``mode`` gates everything: ``enabled`` is ``mode != "off"`` and the model
    only routes through the diagnostic branch when ``enabled`` is true. Stats
    accumulate across chunks / blocks / layers / samples into ``stats`` (keyed by
    ``(layer, slot_width)``) and ``peak`` (keyed by ``layer``); ``summary()``
    reduces them to a plot-ready dict. Use ``diag_mode(...)`` to flip the mode.
    """

    def __init__(self) -> None:
        self.mode: str = "off"
        self.q_chunk: int = 64
        self.collect: bool = True
        # Mode the accumulated stats were collected under. Unlike ``mode`` this is
        # NOT restored when a ``diag_mode`` context exits (mirroring the stats
        # themselves), so a ``summary()`` dump written after the ``with`` block —
        # eval.py writes there — is labeled with the collecting mode, not "off".
        self.stats_mode: str = "off"
        # (layer, slot_width) -> running logit-error / intra-slot-variance /
        # value-readout-error sums.
        self.stats: dict[tuple[int, int], dict[str, float]] = {}
        # layer -> peak |scale·q·k| running max + deterministic sample buffer.
        self.peak: dict[int, dict] = {}
        # layer -> running squared output error of each grid corner vs ``exact``.
        self.out_stats: dict[int, dict[str, float]] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def reset_stats(self) -> None:
        self.stats = {}
        self.peak = {}
        self.out_stats = {}

    def add_slot_stats(
        self, layer: int, width: int, err: torch.Tensor, intra_var: torch.Tensor
    ) -> None:
        """Accumulate approx-vs-exact logit error + intra-slot logit variance.

        ``err = L_approx - L_exact`` and ``intra_var = var_{j in slot}(z_j)`` are
        both (B, G, rf, T_c, n_slots); everything is summed so D3's rate-distortion
        curve (variance vs slot width) is exact per (layer, width).
        """
        acc = self._level_acc(layer, width)
        acc["count"] += float(err.numel())
        acc["sum_abs"] += err.abs().sum().item()
        acc["sum_sq"] += err.double().square().sum().item()
        acc["var_sum"] += intra_var.sum().item()

    def add_value_stats(
        self, layer: int, width: int, v_err: torch.Tensor, v_ref: torch.Tensor
    ) -> None:
        """Accumulate read-out error ``‖mean(v) − V_exact(q, slot)‖`` per (query, slot).

        ``v_err`` / ``v_ref`` are (B, G, rf, T_c, n) per-dim RMS norms (L2 ÷ √Dv) of
        the mean-vs-exact read-out gap and of the exact read-out itself. Lands in
        the same (layer, width) cells as ``add_slot_stats``, so D2's score and
        value sides share one rate-distortion table: ``value_mae`` is absolute
        (per-dim RMS units), ``value_rel`` is relative to ``‖V_exact‖``.
        """
        acc = self._level_acc(layer, width)
        acc["v_count"] += float(v_err.numel())
        acc["v_abs"] += v_err.sum().item()
        acc["v_rel"] += (v_err / v_ref.clamp_min(1e-8)).sum().item()

    def add_out_stats(
        self,
        layer: int,
        err_base: float,
        err_s: float,
        err_v: float,
        ref: float,
        n_queries: int,
    ) -> None:
        """Accumulate per-layer squared output error of each grid corner vs ``exact``.

        One diagnostic run therefore yields the full D2 attribution per layer
        (``by_layer_output`` in ``summary()``) with all four corners evaluated on
        the SAME hidden states — no cross-run drift confound.
        """
        acc = self.out_stats.setdefault(
            int(layer), {"base": 0.0, "s": 0.0, "v": 0.0, "ref": 0.0, "n": 0}
        )
        acc["base"] += err_base
        acc["s"] += err_s
        acc["v"] += err_v
        acc["ref"] += ref
        acc["n"] += n_queries

    def _level_acc(self, layer: int, width: int) -> dict[str, float]:
        return self.stats.setdefault(
            (int(layer), int(width)),
            {
                "count": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "var_sum": 0.0,
                "v_count": 0.0, "v_abs": 0.0, "v_rel": 0.0,
            },
        )

    def add_peak(self, layer: int, abs_z: torch.Tensor) -> None:
        """Record D5 peakiness: running max + a sample buffer of |scale·q·k|.

        The buffer holds a deterministic strided subsample per layer (capped at
        ``_PEAK_BUF_CAP``; thinned 2× on overflow), and ``summary()`` takes ONE
        quantile over the whole run. The previous max-of-per-call-p99.9 proxy was
        not comparable across modes — slot paths call once per width-run, dense
        once per prefix chunk, so identical layer-0 populations still produced
        different p99.9 (the spurious exact-vs-dense gaps in early dumps).
        """
        if abs_z.numel() == 0:
            return
        flat = abs_z.detach().reshape(-1).float()
        acc = self.peak.setdefault(
            int(layer), {"r2_max": 0.0, "buf": [], "buf_n": 0, "stride": 1, "n_seen": 0}
        )
        acc["r2_max"] = max(acc["r2_max"], flat.max().item())
        acc["n_seen"] += flat.numel()
        take = flat[:: acc["stride"]]
        if take.numel() > _PEAK_BUF_CAP:  # one huge call: pre-thin it directly
            take = take[:: -(-take.numel() // _PEAK_BUF_CAP)]
        acc["buf"].append(take.cpu())
        acc["buf_n"] += take.numel()
        while acc["buf_n"] > _PEAK_BUF_CAP:
            thinned = torch.cat(acc["buf"])[::2]
            acc["buf"] = [thinned]
            acc["buf_n"] = thinned.numel()
            acc["stride"] *= 2

    def summary(self) -> dict:
        """Plot-ready reduction of the accumulated stats (see module docstring)."""
        by_level = []
        for (layer, width), acc in sorted(self.stats.items()):
            n = max(acc["count"], 1.0)
            nv = max(acc.get("v_count", 0.0), 1.0)
            by_level.append(
                {
                    "layer": layer,
                    "slot_width": width,
                    "logit_mae": acc["sum_abs"] / n,
                    "logit_rmse": (acc["sum_sq"] / n) ** 0.5,
                    "intra_slot_logit_var": acc["var_sum"] / n,
                    "value_mae": acc.get("v_abs", 0.0) / nv,
                    "value_rel": acc.get("v_rel", 0.0) / nv,
                }
            )
        by_layer_output = []
        for layer, acc in sorted(self.out_stats.items()):
            ref = max(acc["ref"], 1e-30)
            by_layer_output.append(
                {
                    "layer": layer,
                    "err_baseline": (acc["base"] / ref) ** 0.5,
                    "err_s_oracle": (acc["s"] / ref) ** 0.5,
                    "err_v_oracle": (acc["v"] / ref) ** 0.5,
                    "n_queries": acc["n"],
                }
            )
        peakiness = []
        for layer, acc in sorted(self.peak.items()):
            buf = torch.cat(acc["buf"]) if acc["buf"] else torch.zeros(1)
            peakiness.append(
                {
                    "layer": layer,
                    "r2_max": acc["r2_max"],
                    "r2_p999": torch.quantile(buf, 0.999).item(),
                    "n": acc["n_seen"],
                }
            )
        return {
            "mode": self.stats_mode,
            "by_level": by_level,
            "by_layer_output": by_layer_output,
            "peakiness": peakiness,
        }


DIAG = DiagState()


@contextlib.contextmanager
def diag_mode(mode: str, q_chunk: int = 64, collect: bool = True, reset: bool = True):
    """Temporarily switch the global diagnostic mode; restore on exit.

    Args:
        mode: one of ``off / baseline / s_oracle / v_oracle / exact / dense``.
        q_chunk: query-axis chunk for the memory-heavy oracle paths (they
            materialize a per-token fp32 score; see ``diag_block_attention``).
        collect: gather per-level stats + peakiness while running.
        reset: clear previously accumulated stats on entry (default). Stats are
            deliberately NOT cleared on exit, so ``DIAG.summary()`` can be read
            after the ``with`` block.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown diag mode {mode!r}; expected one of {_MODES}")
    prev = (DIAG.mode, DIAG.q_chunk, DIAG.collect)
    if reset:
        DIAG.reset_stats()
    DIAG.mode, DIAG.q_chunk, DIAG.collect = mode, q_chunk, collect
    if mode != "off":
        # Not restored on exit (mirrors the stats themselves): eval.py dumps
        # summary() after this context closes, and the label must name the mode
        # the stats were collected under — restoring it produced mode="off" in
        # every early dump.
        DIAG.stats_mode = mode
    try:
        yield DIAG
    finally:
        DIAG.mode, DIAG.q_chunk, DIAG.collect = prev


# ======================================================================
# Slot -> token-span mapping
# ======================================================================


def slot_runs(slot_w: torch.Tensor) -> tuple[list[tuple[int, int, int]], int]:
    """Group time-ordered slots into runs of equal width.

    LogKV produces slots in level order (every slot in a hierarchy level shares a
    width — a power of two — and the recent window is all width 1), so the slot
    sequence collapses into a few maximal runs of constant width. Each run of
    ``n`` slots of ``width`` covers a contiguous block of ``n * width`` exact
    tokens, so ``reshape(..., n, width)`` recovers the per-slot token spans without
    any scatter.

    Requires ``slot_w`` identical across batch/group: with salience pins active
    the slots are scattered duplicates rather than one contiguous token run, so
    the mapping is invalid (see ``diag_block_attention`` — diagnostics need
    ``pin_size == 0``).

    Returns:
        runs: list of ``(slot_offset, n_slots, width)`` in time order.
        total_tokens: sum of ``n_slots * width`` over all runs.
    """
    if not torch.all(slot_w == slot_w[0, 0]):
        raise ValueError(
            "slot widths differ across batch/group — diagnostics require pin_size=0 "
            "(salience pins scatter duplicate slots and break the contiguous "
            "slot->token span mapping)"
        )
    widths = slot_w[0, 0].round().to(torch.long).tolist()
    runs: list[tuple[int, int, int]] = []
    total = 0
    i, s = 0, len(widths)
    while i < s:
        w = widths[i]
        j = i
        while j < s and widths[j] == w:
            j += 1
        n = j - i
        runs.append((i, n, w))
        total += n * w
        i = j
    return runs, total


# ======================================================================
# Attention decompositions
# ======================================================================


def _production_block(
    q: torch.Tensor,
    slot_k: torch.Tensor,
    slot_v: torch.Tensor,
    slot_w: torch.Tensor,
    k_tail: torch.Tensor,
    v_tail: torch.Tensor,
    scale: float,
    lam: float,
) -> torch.Tensor:
    """Exactly the production block-prefill call — the ``baseline`` output.

    Delegates to ``append_exact_tokens`` + ``log_kv_slot_attention`` with the same
    argument order the model uses, so ``baseline`` reproduces production bit-for-bit
    (the zero-intrusion guarantee).
    """
    k_all, v_all, w_all = append_exact_tokens(slot_k, slot_v, slot_w, k_tail, v_tail)
    return log_kv_slot_attention(
        q, k_all, v_all, w_all, scale=scale, lam=lam, causal_tail=k_tail.size(2)
    )


def _causal_tail_mask(c0: int, t_c: int, blk: int, device: torch.device) -> torch.Tensor:
    """(T_c, blk) bool, True where query at block-offset c0+i must NOT see tail j.

    The in-flight block is causal against itself: query i sees tail token j iff
    ``j <= i`` (query i == tail token i).
    """
    q_pos = torch.arange(c0, c0 + t_c, device=device)
    j_pos = torch.arange(blk, device=device)
    return j_pos[None, :] > q_pos[:, None]


def _diag_slot_core(
    mode: str,
    q: torch.Tensor,          # (B, nh, T_q, D)
    k_prefix: torch.Tensor,   # (B, G, N, D) exact tokens the slots cover
    v_prefix: torch.Tensor,   # (B, G, N, Dv)
    slot_k: torch.Tensor,     # (B, G, S, D) frozen slot state
    slot_v: torch.Tensor,     # (B, G, S, Dv)
    slot_w: torch.Tensor,     # (B, G, S)
    k_tail: torch.Tensor,     # (B, G, blk, D) causal in-flight block
    v_tail: torch.Tensor,     # (B, G, blk, Dv)
    runs: list[tuple[int, int, int]],
    scale: float,
    lam: float,
    layer: int,
) -> torch.Tensor:
    """Slot-bucketed attention with per-side (score/value) oracle swaps + stats.

    Handles ``baseline`` (approx/approx — kept for stats; the returned output is
    overridden by ``_production_block`` for bit-exactness), ``s_oracle``,
    ``v_oracle`` and ``exact``. Everything runs in fp32; the query axis is chunked
    (``DIAG.q_chunk``) because the oracle sides materialize a per-token
    (B, G, rf, T_c, N) score, unlike the O(S) production path.

    With ``DIAG.collect`` the within-slot softmax + token values are retained per
    chunk and ALL FOUR grid corners are evaluated (value-side stats + drift-free
    per-layer output attribution), so every collecting run — including
    ``baseline`` — carries the value-oracle memory envelope. The returned output
    for the active mode is built from the same expressions as the stats-off path,
    so enabling ``collect`` never changes what the model sees.
    """
    B, nh, T_q, D = q.shape
    G = slot_k.size(1)
    rf = nh // G
    Dv = slot_v.size(-1)
    S = slot_k.size(2)
    blk = k_tail.size(2)
    collect = DIAG.collect
    l_exact = mode in _L_EXACT
    v_exact = mode in _V_EXACT

    qf = q.float()
    kp = k_prefix.float()
    vp = v_prefix.float()
    skf = slot_k.float()
    svf = slot_v.float()
    kt = k_tail.float()
    vt = v_tail.float()
    logw = lam * slot_w.float().log()  # (B, G, S); w=1 slots contribute 0

    outs: list[torch.Tensor] = []
    q_chunk = max(1, DIAG.q_chunk)
    for c0 in range(0, T_q, q_chunk):
        c1 = min(T_q, c0 + q_chunk)
        t_c = c1 - c0
        qg = qf[:, :, c0:c1, :].reshape(B, G, rf, t_c, D)

        # Per-run slot logits, both sides. Under ``collect`` both sides plus the
        # within-slot softmax/values are always materialized (the D2 grid needs
        # all four corners); with stats off, only what the active mode returns,
        # so a stats-off oracle run stays as lean as before.
        l_ex_parts: list[torch.Tensor | None] = []
        l_ap_parts: list[torch.Tensor | None] = []
        w_keep: list[torch.Tensor | None] = []  # per-run within-slot softmax(z)
        v_keep: list[torch.Tensor | None] = []  # per-run token values
        keep_tokens = v_exact or collect        # exact-V read-out needs both
        tok = 0
        for soff, n, width in runs:
            kr = kp[:, :, tok:tok + n * width, :].reshape(B, G, n, width, D)
            # z is the per-token score; every reachable mode needs it (exact-L
            # logsumexp, exact-V within-softmax, or stats), so it is unconditional.
            z = scale * torch.einsum("bgrtd,bgnwd->bgrtnw", qg, kr)  # (B,G,rf,t_c,n,width)
            l_exact_run = torch.logsumexp(z, dim=-1) if (l_exact or collect) else None
            if not l_exact or collect:
                skr = skf[:, :, soff:soff + n, :]                   # (B,G,n,D)
                l_approx_run = (
                    scale * torch.einsum("bgrtd,bgnd->bgrtn", qg, skr)
                    + logw[:, :, None, None, soff:soff + n]
                )
            else:
                l_approx_run = None
            l_ex_parts.append(l_exact_run)
            l_ap_parts.append(l_approx_run)

            within = torch.softmax(z, dim=-1) if keep_tokens else None
            vr = (
                vp[:, :, tok:tok + n * width, :].reshape(B, G, n, width, Dv)
                if keep_tokens
                else None
            )
            w_keep.append(within)
            v_keep.append(vr)

            if collect:
                DIAG.add_slot_stats(
                    layer, width, l_approx_run - l_exact_run, z.var(dim=-1, unbiased=False)
                )
                DIAG.add_peak(layer, z.abs())
                # Value side of the D2 grid: what a width-w mean-pooled read-out
                # actually loses, ‖mean(v) − softmax_within(z)·v‖ per (query, slot).
                v_ex_run = torch.einsum("bgrtnw,bgnwc->bgrtnc", within, vr)
                diff = v_ex_run - svf[:, :, None, None, soff:soff + n, :]
                dv_sqrt = Dv ** 0.5
                DIAG.add_value_stats(
                    layer, width, diff.norm(dim=-1) / dv_sqrt, v_ex_run.norm(dim=-1) / dv_sqrt
                )
            tok += n * width

        l_tail = scale * torch.einsum("bgrtd,bgsd->bgrts", qg, kt)  # (B,G,rf,t_c,blk)
        l_tail = l_tail.masked_fill(
            _causal_tail_mask(c0, t_c, blk, q.device).view(1, 1, 1, t_c, blk), float("-inf")
        )

        def _mix(l_parts: list[torch.Tensor], exact_v: bool) -> torch.Tensor:
            """Softmax over [slot logits | causal tail] + the chosen read-out."""
            p = torch.softmax(torch.cat(l_parts + [l_tail], dim=-1), dim=-1)
            p_prefix = p[..., :S]
            p_tail = p[..., S:]
            if exact_v:
                o = qg.new_zeros(B, G, rf, t_c, Dv)
                for (soff, n, _width), within, vr in zip(runs, w_keep, v_keep):
                    tw = p_prefix[..., soff:soff + n].unsqueeze(-1) * within
                    o = o + torch.einsum("bgrtnw,bgnwc->bgrtc", tw, vr)
            else:
                o = torch.einsum("bgrts,bgsc->bgrtc", p_prefix, svf)  # p·mean(v) == p·slot_v
            return o + torch.einsum("bgrts,bgsc->bgrtc", p_tail, vt)

        if collect:
            # All four grid corners on the SAME hidden states — drift-free
            # per-layer D2 attribution from a single run (see add_out_stats).
            out_exact_c = _mix(l_ex_parts, True)
            out_base_c = _mix(l_ap_parts, False)
            out_s_c = _mix(l_ex_parts, False)
            out_v_c = _mix(l_ap_parts, True)
            DIAG.add_out_stats(
                layer,
                (out_base_c - out_exact_c).square().sum().item(),
                (out_s_c - out_exact_c).square().sum().item(),
                (out_v_c - out_exact_c).square().sum().item(),
                out_exact_c.square().sum().item(),
                B * nh * t_c,
            )
            out = (
                out_exact_c
                if (l_exact and v_exact)
                else out_s_c if l_exact else out_v_c if v_exact else out_base_c
            )
        else:
            out = _mix(l_ex_parts if l_exact else l_ap_parts, v_exact)
        outs.append(out.reshape(B, nh, t_c, Dv))

    return torch.cat(outs, dim=2).to(q.dtype)


def _diag_dense(
    q: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_tail: torch.Tensor,
    v_tail: torch.Tensor,
    scale: float,
    layer: int,
) -> torch.Tensor:
    """Dense causal attention over every raw token — independent of any bucketing.

    Prefix fully visible (every prefix token precedes the block), in-flight block
    causal against itself. This is the ground-truth oracle that ``exact`` must
    match, computed by a completely separate route (flat softmax, no slots).
    """
    B, nh, T_q, D = q.shape
    G = k_prefix.size(1)
    rf = nh // G
    Dv = v_prefix.size(-1)
    N = k_prefix.size(2)
    blk = k_tail.size(2)
    collect = DIAG.collect

    qf = q.float()
    kp = k_prefix.float()
    vp = v_prefix.float()
    kt = k_tail.float()
    vt = v_tail.float()

    outs: list[torch.Tensor] = []
    q_chunk = max(1, DIAG.q_chunk)
    for c0 in range(0, T_q, q_chunk):
        c1 = min(T_q, c0 + q_chunk)
        t_c = c1 - c0
        qg = qf[:, :, c0:c1, :].reshape(B, G, rf, t_c, D)

        z_pre = scale * torch.einsum("bgrtd,bgnd->bgrtn", qg, kp)   # (B,G,rf,t_c,N)
        z_tail = scale * torch.einsum("bgrtd,bgsd->bgrts", qg, kt)  # (B,G,rf,t_c,blk)
        z_tail = z_tail.masked_fill(
            _causal_tail_mask(c0, t_c, blk, q.device).view(1, 1, 1, t_c, blk), float("-inf")
        )
        if collect:
            DIAG.add_peak(layer, z_pre.abs())

        p = torch.softmax(torch.cat([z_pre, z_tail], dim=-1), dim=-1)
        out = torch.einsum("bgrtn,bgnc->bgrtc", p[..., :N], vp)
        out = out + torch.einsum("bgrts,bgsc->bgrtc", p[..., N:], vt)
        outs.append(out.reshape(B, nh, t_c, Dv))

    return torch.cat(outs, dim=2).to(q.dtype)


def diag_block_attention(
    q: torch.Tensor,          # (B, nh, T_q, D) current-block queries, post-RoPE
    k_prefix: torch.Tensor,   # (B, G, N, D) exact tokens the frozen slots cover
    v_prefix: torch.Tensor,   # (B, G, N, Dv)
    slot_k: torch.Tensor,     # (B, G, S, D) frozen slot state (get_attention_state)
    slot_v: torch.Tensor,     # (B, G, S, Dv)
    slot_w: torch.Tensor,     # (B, G, S)
    k_tail: torch.Tensor,     # (B, G, T_q, D) in-flight block, causal vs queries
    v_tail: torch.Tensor,     # (B, G, T_q, Dv)
    scale: float,
    lam: float = 1.0,
    layer: int = 0,
) -> torch.Tensor:
    """Diagnostic replacement for one production block-prefill attention step.

    Parallels ``log_kv_slot_attention`` but additionally receives the EXACT prefix
    tokens (``k_prefix``/``v_prefix``) the frozen slots cover, so the score and
    value sides can each be swapped between the production first-order approximation
    and the exact within-slot computation (see module docstring for the 2×2 grid).

    Preconditions (the caller in ``model.py`` guards them; do NOT relax):
      * fresh prefill only — ``k_prefix`` must equal the tokens the slots cover,
        i.e. ``cache.token_count == start`` with no decode continuation / pending
        tail. Not valid during decode.
      * ``pin_size == 0`` — salience pins scatter duplicate slots and break the
        contiguous slot→token mapping (rejected by ``slot_runs`` /the token-count
        check below).

    Returns (B, nh, T_q, Dv), matching the production output shape.
    """
    mode = DIAG.mode
    if mode in ("off", "baseline"):
        # off: defensive (never reached with a live diag context). baseline:
        # bit-exact production, optionally with a stats-only oracle pass.
        out = _production_block(q, slot_k, slot_v, slot_w, k_tail, v_tail, scale, lam)
        if mode == "baseline" and DIAG.collect:
            runs, total = slot_runs(slot_w)
            if total != k_prefix.size(2):
                raise ValueError(
                    f"slot span ({total}) != prefix length ({k_prefix.size(2)}): "
                    "diagnostics require a fresh prefill with pin_size=0"
                )
            _diag_slot_core(
                "baseline", q, k_prefix, v_prefix, slot_k, slot_v, slot_w,
                k_tail, v_tail, runs, scale, lam, layer,
            )
        return out

    if mode == "dense":
        return _diag_dense(q, k_prefix, v_prefix, k_tail, v_tail, scale, layer)

    runs, total = slot_runs(slot_w)
    if total != k_prefix.size(2):
        raise ValueError(
            f"slot span ({total}) != prefix length ({k_prefix.size(2)}): "
            "diagnostics require a fresh prefill with pin_size=0"
        )
    return _diag_slot_core(
        mode, q, k_prefix, v_prefix, slot_k, slot_v, slot_w,
        k_tail, v_tail, runs, scale, lam, layer,
    )
