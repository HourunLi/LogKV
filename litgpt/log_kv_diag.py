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
        # (layer, slot_width) -> running logit-error / intra-slot-variance sums.
        self.stats: dict[tuple[int, int], dict[str, float]] = {}
        # layer -> peak |scale·q·k| running max + p99.9.
        self.peak: dict[int, dict[str, float]] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def reset_stats(self) -> None:
        self.stats = {}
        self.peak = {}

    def add_slot_stats(
        self, layer: int, width: int, err: torch.Tensor, intra_var: torch.Tensor
    ) -> None:
        """Accumulate approx-vs-exact logit error + intra-slot logit variance.

        ``err = L_approx - L_exact`` and ``intra_var = var_{j in slot}(z_j)`` are
        both (B, G, rf, T_c, n_slots); everything is summed so D3's rate-distortion
        curve (variance vs slot width) is exact per (layer, width).
        """
        key = (int(layer), int(width))
        acc = self.stats.setdefault(
            key, {"count": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "var_sum": 0.0}
        )
        acc["count"] += float(err.numel())
        acc["sum_abs"] += err.abs().sum().item()
        acc["sum_sq"] += err.double().square().sum().item()
        acc["var_sum"] += intra_var.sum().item()

    def add_peak(self, layer: int, abs_z: torch.Tensor) -> None:
        """Record D5 peakiness: running max + p99.9 of |scale·q·k| for a layer.

        p99.9 is aggregated as the max of per-call p99.9 estimates (a diagnostic
        runs on a handful of samples, so this is a fine, cheap proxy). Large score
        tensors are subsampled before the quantile — ``torch.quantile`` caps its
        input size, and exact percentiles are not needed here.
        """
        if abs_z.numel() == 0:
            return
        flat = abs_z.detach().flatten().float()
        m = flat.max().item()
        if flat.numel() > 1_000_000:
            idx = torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)
            flat = flat[idx]
        p = torch.quantile(flat, 0.999).item()
        acc = self.peak.setdefault(int(layer), {"r2_max": 0.0, "r2_p999": 0.0})
        acc["r2_max"] = max(acc["r2_max"], m)
        acc["r2_p999"] = max(acc["r2_p999"], p)

    def summary(self) -> dict:
        """Plot-ready reduction of the accumulated stats (see module docstring)."""
        by_level = []
        for (layer, width), acc in sorted(self.stats.items()):
            n = max(acc["count"], 1.0)
            by_level.append(
                {
                    "layer": layer,
                    "slot_width": width,
                    "logit_mae": acc["sum_abs"] / n,
                    "logit_rmse": (acc["sum_sq"] / n) ** 0.5,
                    "intra_slot_logit_var": acc["var_sum"] / n,
                }
            )
        peakiness = [
            {"layer": layer, "r2_max": acc["r2_max"], "r2_p999": acc["r2_p999"]}
            for layer, acc in sorted(self.peak.items())
        ]
        return {"mode": self.mode, "by_level": by_level, "peakiness": peakiness}


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

        l_parts: list[torch.Tensor] = []       # per-run slot logits, (B,G,rf,t_c,n)
        z_keep: list[torch.Tensor | None] = []  # per-run token scores (value oracle)
        v_keep: list[torch.Tensor | None] = []  # per-run token values  (value oracle)
        tok = 0
        for soff, n, width in runs:
            kr = kp[:, :, tok:tok + n * width, :].reshape(B, G, n, width, D)
            # z is the per-token score; every reachable mode needs it (exact-L
            # logsumexp, exact-V within-softmax, or stats), so it is unconditional.
            z = scale * torch.einsum("bgrtd,bgnwd->bgrtnw", qg, kr)  # (B,G,rf,t_c,n,width)
            # The other logit is only materialized when it is actually used
            # (as the slot logit or for the approx-vs-exact stat), so on a 32K
            # sequence a pure oracle run never pays for the unused side.
            l_exact_run = torch.logsumexp(z, dim=-1) if (l_exact or collect) else None
            if not l_exact or collect:
                skr = skf[:, :, soff:soff + n, :]                   # (B,G,n,D)
                l_approx_run = (
                    scale * torch.einsum("bgrtd,bgnd->bgrtn", qg, skr)
                    + logw[:, :, None, None, soff:soff + n]
                )
            else:
                l_approx_run = None
            l_parts.append(l_exact_run if l_exact else l_approx_run)

            if collect:
                DIAG.add_slot_stats(
                    layer, width, l_approx_run - l_exact_run, z.var(dim=-1, unbiased=False)
                )
                DIAG.add_peak(layer, z.abs())

            if v_exact:
                z_keep.append(z)
                v_keep.append(vp[:, :, tok:tok + n * width, :].reshape(B, G, n, width, Dv))
            else:
                z_keep.append(None)
                v_keep.append(None)
            tok += n * width

        l_prefix = torch.cat(l_parts, dim=-1)  # (B,G,rf,t_c,S), slot-aligned
        l_tail = scale * torch.einsum("bgrtd,bgsd->bgrts", qg, kt)  # (B,G,rf,t_c,blk)
        l_tail = l_tail.masked_fill(
            _causal_tail_mask(c0, t_c, blk, q.device).view(1, 1, 1, t_c, blk), float("-inf")
        )

        p = torch.softmax(torch.cat([l_prefix, l_tail], dim=-1), dim=-1)
        p_prefix = p[..., :S]
        p_tail = p[..., S:]

        if v_exact:
            out = qg.new_zeros(B, G, rf, t_c, Dv)
            for (soff, n, _width), z, vr in zip(runs, z_keep, v_keep):
                within = torch.softmax(z, dim=-1)                    # (B,G,rf,t_c,n,width)
                tw = p_prefix[..., soff:soff + n].unsqueeze(-1) * within
                out = out + torch.einsum("bgrtnw,bgnwc->bgrtc", tw, vr)
        else:
            out = torch.einsum("bgrts,bgsc->bgrtc", p_prefix, svf)   # p·mean(v) == p·slot_v
        out = out + torch.einsum("bgrts,bgsc->bgrtc", p_tail, vt)
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
