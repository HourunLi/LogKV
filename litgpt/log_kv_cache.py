"""Log-Structured KV Cache with merged-position slots — strict O(B·log N).

Architecture (uniform 2:1 compaction at every level):
- Buffer (sliding window): recent_size raw tokens, no compression (exact keys).
- Level 0 (write level):   B entries, each = 2 tokens merged. Partially fillable
                            (0 to B entries). Filled one entry at a time from buffer
                            flush. When full, carries to level 1.
- Level 1+ (carry levels):  B entries per level, each = 2 entries from prev level
                            merged. All-or-nothing (0 or B entries). Binary carry
                            promotes full blocks upward.

Position handling ("expected RoPE", merged into the compressed state):
- Keys enter the cache as FULL post-RoPE vectors
      k_j = [ R(p_j)·k_pos_j ; k_content_j ]
  (partial rotary: only the leading rope_n_elem dims are rotated; any remaining
  content channel is position-free; full rotary means the content channel is
  empty — both are supported).
- Merging a block mean-pools the WHOLE key. Mean commutes with concatenation, so
  the merged key is [ mean_j R(p_j)·k_pos_j ; mean_j k_content_j ]: the block's
  position embedding is the weighted mean of its tokens' rotated position keys —
  the exact expectation of the rotation over the span. Per RoPE frequency θ over
  a width-w span centred at c this equals R(c)·sin(wθ/2)/(w·sin(θ/2)): RoPE at
  the span centre, damped per-frequency by a Dirichlet factor. High frequencies
  fade as spans widen, so positional resolution decays ∝ span width ∝ distance
  (constant relative error). The merged key is deliberately NOT renormalized:
  the norm shrinkage encodes the span's positional uncertainty.
- NO per-token state of any kind survives outside the recent window.

Slot attention (``log_kv_slot_attention``), when rank-1 stats are supplied:
    score_s = scale·(q · k_s) + 0.5·scale²·sigma2_s·(q · sigma_u_s)² + λ·log(w_s)
    read_s  = v_s + scale·gamma_s·(q · gamma_a_s)·gamma_b_s
    out     = softmax(score) · read
The +λ·log(w_s) mass bias gives a w-token slot softmax mass ≈ w·exp(score),
while the Sigma/Gamma terms add the second-order score correction and first-order
value read-out correction. Exact recent / in-flight tokens carry zero stats, so
they reduce to ordinary token attention. Calling ``log_kv_slot_attention`` without
stats preserves the original first-order compatibility path.

Memory:  O(recent_size + B·log(N/2)) slots — no Θ(N) term.
Compute: O(recent_size + B·log(N/2)) per query — no Θ(N) term.
"""

from __future__ import annotations

import math
from typing import NamedTuple, NoReturn

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from litgpt.log_kv_pin_score_diag import DIAG as LOG_KV_PIN_SCORE_DIAG
from litgpt.log_kv_position import anchor_mass_bias, merge_anchors


_RANK1_EPS = 1e-12

# Squarings used to extract the dominant eigenvector of the tiny (r x r) cores
# below. Each round squares the eigenvalue gap, so k rounds separate the top
# eigenpair as well as 2^k power iterations would, at k batched GEMMs.
#
# 8 rounds because they are almost free and the accuracy is not: the cost of
# these helpers sits in the one (batch, r, D) projection, not in the r x r
# squarings, so 5 -> 8 rounds costs ~4% of the helper while cutting the
# worst-case truncation error ~8x (3.1e-3 -> 3.8e-4 excess relative Frobenius
# error over an exact eigh, measured on the near-degenerate case that converges
# slowest). TestRank1Approximation pins the resulting quality.
_RANK1_SQUARINGS = 8


class CacheAttentionState(NamedTuple):
    slot_k: torch.Tensor
    slot_v: torch.Tensor
    slot_w: torch.Tensor
    slot_valid: torch.Tensor | None = None
    M_s: torch.Tensor | None = None
    slot_sigma_u: torch.Tensor | None = None
    slot_sigma2: torch.Tensor | None = None
    slot_gamma_a: torch.Tensor | None = None
    slot_gamma_b: torch.Tensor | None = None
    slot_gamma: torch.Tensor | None = None


def _normalize(x: torch.Tensor, eps: float = _RANK1_EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a unit direction and the original norm, computed in fp32."""
    xf = x.float()
    norm = xf.norm(dim=-1, keepdim=True)
    unit = torch.where(norm > eps, xf / norm.clamp_min(eps), torch.zeros_like(xf))
    return unit.to(x.dtype), norm.squeeze(-1).to(x.dtype)


def _dominant_eigvec_small(core: torch.Tensor) -> torch.Tensor:
    """Unit dominant right eigenvector of a batch of tiny ``(..., r, r)`` cores.

    ``core`` must have a real non-negative spectrum: either symmetric PSD, or a
    product of two PSD Gram matrices (similar to the PSD ``G^(1/2) H G^(1/2)``).

    Repeated squaring drives ``core^(2^k)`` towards the rank-1 dominant
    projector, so every column ends up parallel to the top eigenvector and the
    largest-norm column is the best-conditioned representative. Convergence is
    ``(lam2/lam1)^(2^k)``; the slowest case is a near-degenerate top pair, which
    is exactly the case where any vector of that eigenspace is an equally good
    rank-1 direction, so the resulting approximation error stays small either
    way.

    Why not ``torch.linalg.eigh``/``qr``/``svd``: those dispatch to cuSOLVER,
    whose batched small-matrix paths carry a large fixed cost (and, for tall-thin
    QR, degrade towards a per-matrix loop). LogKV calls this once per merged slot
    per compaction — thousands of matrices per call, thousands of calls per
    training step — where that fixed cost dominated everything else. This path is
    batched GEMM only.
    """
    r = core.size(-1)
    if r == 1:
        return torch.ones_like(core[..., 0])
    m = core
    for _ in range(_RANK1_SQUARINGS):
        # Renormalize by the trace before squaring: raw powers reach lam1^(2^k)
        # and overflow fp32 within a few rounds, and the scale is irrelevant to
        # the direction. trace > 0 for any matrix with non-negative spectrum.
        trace = m.diagonal(dim1=-2, dim2=-1).sum(-1).abs().clamp_min(_RANK1_EPS)
        m = m / trace[..., None, None]
        m = torch.matmul(m, m)
    # Columns of m^(2^k) are all parallel to the top eigenvector; pick the
    # longest one (zero everywhere iff the input was the zero matrix).
    col_norms = m.norm(dim=-2)                       # (..., r)
    pick = col_norms.argmax(dim=-1, keepdim=True)    # (..., 1)
    vec = torch.gather(m, -1, pick.unsqueeze(-2).expand(*m.shape[:-1], 1)).squeeze(-1)
    norm = vec.norm(dim=-1, keepdim=True)
    return torch.where(norm > _RANK1_EPS, vec / norm.clamp_min(_RANK1_EPS), torch.zeros_like(vec))


def _rank1_psd_from_factors(factors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Near-optimal rank-1 PSD approximation to ``sum_i f_i f_i^T``.

    ``factors`` is ``(..., r, D)`` with small ``r`` for merged slots (3 during
    carry; 2 for raw pairs). The non-zero spectrum of the D x D matrix lives in
    the r x r Gram matrix, so the truncation runs there, without ever
    materializing a full covariance matrix.

    NOT an exact top-eigen truncation: the direction comes from
    ``_dominant_eigvec_small``, which iterates rather than solves. The
    eigenvalue is then an exact Rayleigh quotient of that direction, so it is
    the best scale for whatever direction was found. ``TestRank1Approximation``
    pins the resulting reconstruction error against an ``eigh`` reference.
    """
    if factors.size(-2) == 0:
        return factors[..., :0, :].sum(dim=-2), factors.new_zeros(factors.shape[:-2])

    ff = factors.float()
    gram = torch.matmul(ff, ff.mT)
    coeff = _dominant_eigvec_small(gram)
    # raw = F^T c, so ||raw||^2 = c^T G c is the Rayleigh quotient — the top
    # eigenvalue, and second-order accurate in any eigenvector error. It also
    # comes for free from the projection we need anyway.
    raw = torch.matmul(coeff.unsqueeze(-2), ff).squeeze(-2)
    top = raw.square().sum(-1)
    direction = raw / top.sqrt().clamp_min(_RANK1_EPS).unsqueeze(-1)
    direction = torch.where(top.unsqueeze(-1) > _RANK1_EPS, direction, torch.zeros_like(direction))
    return direction.to(factors.dtype), top.to(factors.dtype)


def _rank1_cross_from_factors(
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Near-optimal rank-1 approximation to ``M = sum_i l_i r_i^T``.

    Like ``_rank1_psd_from_factors`` this is iterative, not an exact SVD
    truncation; ``TestRank1Approximation`` pins the error against a QR/SVD
    reference. The singular value is computed from the directions actually
    found, so the returned triplet is always self-consistent.

    Returns ``left_unit, right_unit, singular_value``. The full value-key cross
    covariance is never materialized, and neither are the thin bases of its
    factors. With ``L`` / ``R`` the ``(..., r, D)`` factor stacks, ``M = L^T R``
    and ``M M^T = L^T (R R^T) L``, so the top left singular vector lies in the
    row space of ``L``: writing it as ``L^T c`` reduces the eigenproblem to the
    r x r core ``(R R^T)(L L^T)``. From its dominant eigenvector ``c``,
    ``u ∝ L^T c`` and ``v ∝ M^T u = R^T (L L^T) c``, and the singular value is
    ``u^T M v = <L u, R v>``. Everything is r x r or a single (r, D) projection.
    """
    left = left_factors.float()    # (..., r, Dv)
    right = right_factors.float()  # (..., r, Dk)
    gram_left = torch.matmul(left, left.mT)     # (..., r, r)
    gram_right = torch.matmul(right, right.mT)  # (..., r, r)
    coeff = _dominant_eigvec_small(torch.matmul(gram_right, gram_left))  # (..., r)

    left_raw = torch.matmul(coeff.unsqueeze(-2), left).squeeze(-2)  # (..., Dv) = L^T c
    # R^T (L L^T) c — positively aligned with the true right vector for this u,
    # so the singular value below comes out non-negative without a sign fix.
    right_raw = torch.matmul(
        torch.matmul(coeff.unsqueeze(-2), gram_left), right
    ).squeeze(-2)  # (..., Dk)

    left_norm = left_raw.norm(dim=-1, keepdim=True)
    right_norm = right_raw.norm(dim=-1, keepdim=True)
    left_unit = torch.where(
        left_norm > _RANK1_EPS, left_raw / left_norm.clamp_min(_RANK1_EPS), torch.zeros_like(left_raw)
    )
    right_unit = torch.where(
        right_norm > _RANK1_EPS, right_raw / right_norm.clamp_min(_RANK1_EPS), torch.zeros_like(right_raw)
    )

    gamma = (
        torch.matmul(left, left_unit.unsqueeze(-1)) * torch.matmul(right, right_unit.unsqueeze(-1))
    ).sum(dim=(-2, -1)).clamp_min(0.0)
    left_unit = torch.where(gamma.unsqueeze(-1) > _RANK1_EPS, left_unit, torch.zeros_like(left_unit))
    right_unit = torch.where(gamma.unsqueeze(-1) > _RANK1_EPS, right_unit, torch.zeros_like(right_unit))
    return (
        left_unit.to(left_factors.dtype),
        right_unit.to(right_factors.dtype),
        gamma.to(left_factors.dtype),
    )


def _pair_rank1_stats(
    ka: torch.Tensor,
    kb: torch.Tensor,
    va: torch.Tensor,
    vb: torch.Tensor,
    frac_a: torch.Tensor | float = 0.5,
    frac_b: torch.Tensor | float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact rank-1 covariance / cross-covariance stats for a 2-token slot.

    ``frac_a``/``frac_b`` are the pooling weights used for the slot mean
    (default 0.5/0.5, i.e. an unweighted pair). For a 2-point set the
    weighted covariance ``p_a(ka-m)(ka-m)^T + p_b(kb-m)(kb-m)^T`` collapses
    to exactly ``p_a*p_b * dk dk^T`` (``m = p_a*ka + p_b*kb``, ``dk = ka-kb``)
    regardless of ``p_a``/``p_b`` — rank-1 with no truncation error, same as
    the unweighted case (which is this formula at ``p_a=p_b=0.5``).
    """
    dk = ka - kb
    dv = va - vb
    sigma_u, dk_norm = _normalize(dk)
    gamma_b, dv_norm = _normalize(dv)
    cross = frac_a * frac_b
    sigma2 = dk_norm.square() * cross
    gamma = dk_norm * dv_norm * cross
    gamma_a = sigma_u
    return sigma_u, sigma2, gamma_a, gamma_b, gamma


class LogStructuredKVCache(nn.Module):
    """Log-Structured KV Cache storing full post-RoPE keys in merged slots.

    Args:
        k_shape: (batch_size, n_groups, max_seq_length, k_dim) — k_dim is the
            FULL post-RoPE key width (RoPE'd position channel + content channel).
            max_seq_length only sizes the level hierarchy and the append-only
            token counter; no O(N) buffer is allocated.
        v_shape: (batch_size, n_groups, max_seq_length, v_dim)
        B: slots per level (default 512)
        recent_size: sliding window size (default 0 = 2)
        device / dtype: torch device / dtype
        importance_pooling: if True, merges are weighted by a per-token
            importance mass (tracked separately from the token-count weight
            ``w``, which keeps driving the ``log(w)`` mass bias unchanged)
            instead of a uniform mean. See ``compact()``/``_compact_tokens()``.
        importance_pooling_lambda: only consulted when ``importance_pooling``
            is True. Linearly blends the importance-driven pooling share
            toward the uniform/count-driven share — ``1.0`` (default) is pure
            importance pooling, ``0.0`` is numerically equivalent to plain
            uniform pooling, values in between interpolate. Added after the
            first decisive eval (2026-08-14) showed importance pooling
            improves ACC/LongBench but regresses niah — the working
            hypothesis is that the pure key-norm heuristic over-weights
            high-norm non-needle tokens, and a partial blend may recover
            niah while keeping most of the LongBench gain (see CLAUDE.md 6.5).
        importance_pooling_temperature: only consulted when
            ``importance_pooling`` is True. Exponent applied to the raw
            per-token importance heuristic (key L2 norm) before it is
            normalized into pooling weights: ``imp = raw_imp ** temperature``.
            ``1.0`` (default) is the original heuristic, unchanged (bit-
            identical short-circuit). Values ``< 1.0`` compress the dynamic
            range, damping outlier tokens (e.g. attention-sink-style high-
            norm tokens) relative to the bulk, without fully discarding the
            salience signal the way blending the whole distribution toward
            uniform (``importance_pooling_lambda``) does; ``temperature -> 0``
            approaches uniform in the limit, ``> 1.0`` sharpens further.
            Orthogonal to ``importance_pooling_lambda`` — both may be set at
            once (temperature reshapes first, lambda blends the result).
    """

    def __init__(
        self,
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        B: int = 512,
        recent_size: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        pin_size: int = 0,
        importance_pooling: bool = False,
        importance_pooling_lambda: float = 1.0,
        importance_pooling_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        batch_size, n_groups, max_seq_length, k_dim = k_shape
        _, _, _, v_dim = v_shape

        self.batch_size = batch_size
        self.n_groups = n_groups
        self.max_seq_length = max_seq_length
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.B = B
        self.K_max = 1
        self.recent_size = recent_size if recent_size > 0 else 2
        # Explicit raise (not assert): must survive `python -O`.
        if self.recent_size < 2:
            raise ValueError(f"recent_size ({self.recent_size}) must be >= 2")
        self.importance_pooling = bool(importance_pooling)
        self.importance_pooling_lambda = float(importance_pooling_lambda)
        if not 0.0 <= self.importance_pooling_lambda <= 1.0:
            raise ValueError(
                f"importance_pooling_lambda must be in [0, 1], got {self.importance_pooling_lambda}"
            )
        self.importance_pooling_temperature = float(importance_pooling_temperature)
        if self.importance_pooling_temperature <= 0.0:
            raise ValueError(
                f"importance_pooling_temperature must be > 0, got {self.importance_pooling_temperature}"
            )

        denom = B * 2
        # +1 for the write level (level 0). The formula gives the number of carry
        # levels needed; total levels = carry + 1 write level.
        self.max_levels = max(2, math.ceil(math.log2(max((max_seq_length + 1) / denom, 1))) + 1)
        self.L_alloc = self.max_levels
        self.B_prime = B

        # Total tokens ever committed (scalar bookkeeping only — replaces the
        # former Θ(N) per-token position-key buffer).
        self.token_count: int = 0

        # ---- Sliding window: last < recent_size tokens (exact keys + values) ----
        self.register_buffer(
            "recent_k",
            torch.zeros(batch_size, n_groups, self.recent_size, k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "recent_v",
            torch.zeros(batch_size, n_groups, self.recent_size, v_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.recent_count: int = 0

        # ---- Hierarchical entries: (B, G, K_max, L_alloc, B', ·) ----
        # K_max=1 is the current single-cluster path; the indexed layout is the
        # Stage 1 storage contract that later semantic routing will populate.
        self.register_buffer(
            "level_k",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "level_v",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, v_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "level_w",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, device=device, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "level_imp",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, device=device, dtype=torch.float32),
            persistent=False,
        )
        # Rank-1 key covariance:
        #   Sigma_s ~= sigma2_s * sigma_u_s sigma_u_s^T
        self.register_buffer(
            "level_sigma_u",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "level_sigma2",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, device=device, dtype=dtype),
            persistent=False,
        )
        # Rank-1 value-key cross covariance:
        #   Gamma_s ~= gamma_s * gamma_b_s gamma_a_s^T
        # where gamma_a lives in key/query space and gamma_b in value space.
        self.register_buffer(
            "level_gamma_a",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "level_gamma_b",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, v_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "level_gamma",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "level_count",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, dtype=torch.int16, device=device),
            persistent=False,
        )
        self.register_buffer(
            "pad_mask",
            torch.zeros(batch_size, n_groups, self.K_max, self.L_alloc, B, dtype=torch.bool, device=device),
            persistent=False,
        )

        # When False, compaction skips the rank-1 second-order statistics
        # entirely (they stay zero) and the levels behave like the first-order
        # cache. Driven by the second-order gate: a scale of 0 makes every
        # correction term vanish, so computing and merging the statistics that
        # feed it is pure overhead. See ``log_kv_chunk_attention``. Guarded by
        # the ``second_order`` property below — set the backing field directly
        # here since a fresh cache has nothing for the guard to protect.
        self._second_order: bool = True

        # ---- Salience pins: up to pin_size exact tokens kept OUTSIDE the
        # hierarchy (SnapKV-style observation-window selection at prefill; see
        # CausalSelfAttention._log_kv_select_pins). Rationale: uniform mean-
        # pooling dilutes a distant low-redundancy fact (a "needle") by 1/w, and
        # retrospective salience cannot save it — but at prefill time the
        # query IS in the prompt tail, so the trailing observation window can
        # score the whole prefix and pin what it will need before compaction
        # buries it. Pinned tokens are DUPLICATES: the hierarchy still pools
        # them (state trajectory is bit-identical with pins on or off), the pin
        # buffer just re-exposes them as exact w=1 slots. The double-counted
        # softmax mass is one token out of the covering slot's w — negligible.
        # pin_size=0 disables everything (buffers stay empty). Memory stays
        # O(recent + pin + B*logN).
        self.pin_size = pin_size
        self.register_buffer(
            "pin_k",
            torch.zeros(batch_size, n_groups, pin_size, k_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "pin_v",
            torch.zeros(batch_size, n_groups, pin_size, v_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.pin_count: int = 0

    # ------------------------------------------------------------------
    # Second-order gate (guarded: cannot change regime on a live cache)
    # ------------------------------------------------------------------

    @property
    def second_order(self) -> bool:
        """Whether compaction builds and merges the rank-1 Sigma/Gamma stats.

        Guarded rather than a plain flag: flipping it while a level already
        holds compacted entries would leave the hierarchy straddling two
        regimes, silently.

        - True -> False mid-stream: the next binary carry takes the no-stats
          branch of ``compact()``, which ignores whatever stats the existing
          level already had and produces a merged entry with none — those
          real, previously-built stats are gone.
        - False -> True mid-stream: the next carry merges genuine new stats
          against a sibling that has structural zero stats (built while this
          was False) via Chan's formula. That reads as "this half of the slot
          had exactly zero variance", not "unknown" — an understated, wrong
          covariance for the merged slot, not a conservative one.

        Both keep producing finite numbers, so nothing downstream would flag
        it. Call ``reset_parameters()`` before switching regimes on a cache
        that has ever run ``add_recent()``; a fresh cache and same-value writes
        are always fine.
        """
        return self._second_order

    @second_order.setter
    def second_order(self, value: bool) -> None:
        value = bool(value)
        if value != self._second_order and bool((self.level_count > 0).any().item()):
            raise RuntimeError(
                f"LogStructuredKVCache: cannot change `second_order` "
                f"({self._second_order} -> {value}) while a compacted level "
                "already holds entries. Call reset_parameters() first -- see "
                "the `second_order` property docstring for why switching "
                "regimes in place is unsafe."
            )
        self._second_order = value

    # ------------------------------------------------------------------
    # Salience pins
    # ------------------------------------------------------------------

    def set_pinned(self, k_sel: torch.Tensor, v_sel: torch.Tensor) -> None:
        """Install the pinned exact tokens (post-RoPE K/V), replacing any prior set.

        Args:
            k_sel: (batch, n_groups, n_pin, k_dim), n_pin <= pin_size. Each
                group carries its own selection (GQA groups score independently).
            v_sel: (batch, n_groups, n_pin, v_dim)
        """
        n_pin = k_sel.size(2)
        # Explicit raises (not asserts): must survive `python -O`.
        if n_pin > self.pin_size:
            raise ValueError(f"n_pin ({n_pin}) exceeds pin_size ({self.pin_size})")
        if v_sel.size(2) != n_pin:
            raise ValueError(f"k_sel/v_sel pin counts differ: {n_pin} vs {v_sel.size(2)}")
        self.pin_k[:, :, :n_pin, :] = k_sel
        self.pin_v[:, :, :n_pin, :] = v_sel
        if n_pin < self.pin_count:
            self.pin_k[:, :, n_pin:self.pin_count, :].zero_()
            self.pin_v[:, :, n_pin:self.pin_count, :].zero_()
        self.pin_count = n_pin

    # ------------------------------------------------------------------
    # Level accessors
    # ------------------------------------------------------------------

    def _get_level(self, ell: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.level_k[:, :, 0, ell],
            self.level_v[:, :, 0, ell],
            self.level_w[:, :, 0, ell],
        )

    def _get_level_stats(
        self, ell: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.level_sigma_u[:, :, 0, ell],
            self.level_sigma2[:, :, 0, ell],
            self.level_gamma_a[:, :, 0, ell],
            self.level_gamma_b[:, :, 0, ell],
            self.level_gamma[:, :, 0, ell],
        )

    def _get_level_imp(self, ell: int) -> torch.Tensor:
        return self.level_imp[:, :, 0, ell]

    def _level_count(self, ell: int) -> int:
        # ponytail: K_max=1/aligned-batch path; switch to per-row counts only
        # when semantic routing creates genuinely divergent level occupancy.
        return int(self.level_count[0, 0, 0, ell].item())

    def _set_level(
        self,
        ell: int,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        sigma_u: torch.Tensor | None = None,
        sigma2: torch.Tensor | None = None,
        gamma_a: torch.Tensor | None = None,
        gamma_b: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
        imp: torch.Tensor | None = None,
    ) -> None:
        self.level_k[:, :, 0, ell].copy_(k)
        self.level_v[:, :, 0, ell].copy_(v)
        self.level_w[:, :, 0, ell].copy_(w)
        if sigma_u is None:
            self.level_sigma_u[:, :, 0, ell].zero_()
            self.level_sigma2[:, :, 0, ell].zero_()
            self.level_gamma_a[:, :, 0, ell].zero_()
            self.level_gamma_b[:, :, 0, ell].zero_()
            self.level_gamma[:, :, 0, ell].zero_()
        else:
            self.level_sigma_u[:, :, 0, ell].copy_(sigma_u)
            self.level_sigma2[:, :, 0, ell].copy_(sigma2)
            self.level_gamma_a[:, :, 0, ell].copy_(gamma_a)
            self.level_gamma_b[:, :, 0, ell].copy_(gamma_b)
            self.level_gamma[:, :, 0, ell].copy_(gamma)
        if imp is None:
            self.level_imp[:, :, 0, ell].zero_()
        else:
            self.level_imp[:, :, 0, ell].copy_(imp)
        self.level_count[:, :, 0, ell].fill_(self.B)
        self.pad_mask[:, :, 0, ell].zero_()

    def _clear_level(self, ell: int) -> None:
        self.level_k[:, :, 0, ell].zero_()
        self.level_v[:, :, 0, ell].zero_()
        self.level_w[:, :, 0, ell].zero_()
        self.level_imp[:, :, 0, ell].zero_()
        self.level_sigma_u[:, :, 0, ell].zero_()
        self.level_sigma2[:, :, 0, ell].zero_()
        self.level_gamma_a[:, :, 0, ell].zero_()
        self.level_gamma_b[:, :, 0, ell].zero_()
        self.level_gamma[:, :, 0, ell].zero_()
        self.level_count[:, :, 0, ell].zero_()
        self.pad_mask[:, :, 0, ell].zero_()

    # ------------------------------------------------------------------
    # Compact: compress tokens -> 1 entry via mean pooling (2:1 by default)
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_tokens(
        k: torch.Tensor,  # (B, G, n, k_dim) full post-RoPE keys
        v: torch.Tensor,  # (B, G, n, v_dim)
        with_stats: bool = False,
        imp: torch.Tensor | None = None,  # (B, G, n) optional per-token importance weights
        imp_lambda: float = 1.0,
    ) -> tuple[torch.Tensor, ...]:
        """Compress n tokens into a single compact entry via mean pooling.

        Mean-pooling the full key merges content and position in one step:
        the position sub-channel becomes the expected rotation over the span
        (see module docstring). No renormalization — the per-frequency norm
        shrinkage encodes the span's positional uncertainty.
        Returns k_entry (B,G,1,k_dim), v_entry (B,G,1,v_dim), w_entry (B,G,1).
        With ``with_stats=True`` also returns rank-1 approximations to the
        within-slot key covariance and value-key cross covariance.

        ``imp``, when given, replaces the uniform 1/n pooling weight with a
        per-token importance-weighted mean (need not be pre-normalized); the
        raw sum of ``imp`` is returned as one extra trailing fp32 tensor (the
        slot's cumulative importance mass — independent of ``w_entry``, the
        token count, which is always uniform regardless of ``imp``). Omitting
        ``imp`` reproduces the exact prior uniform-mean behavior and return
        shape. ``imp_lambda`` (only consulted when ``imp`` is given) linearly
        blends the importance-normalized weights toward the uniform ``1/n``
        weights — ``1.0`` (default) is pure importance pooling (unchanged
        expression, bit-identical to the pre-``imp_lambda`` code), ``0.0`` is
        numerically equivalent to uniform pooling (same shares as ``imp=None``,
        via a different, imp-carrying code path so not bit-identical).
        """
        n = k.size(2)
        has_imp = imp is not None
        if has_imp:
            imp = imp.float()
            imp_entry = imp.sum(dim=2, keepdim=True)  # (B, G, 1), fp32
            if imp_lambda >= 1.0:
                p = (imp / imp_entry.clamp_min(_RANK1_EPS)).unsqueeze(-1)  # (B, G, n, 1)
            else:
                p_imp = imp / imp_entry.clamp_min(_RANK1_EPS)
                p = (imp_lambda * p_imp + (1.0 - imp_lambda) * (1.0 / n)).unsqueeze(-1)
            k_entry = (p * k.float()).sum(dim=2, keepdim=True).to(k.dtype)
            v_entry = (p * v.float()).sum(dim=2, keepdim=True).to(v.dtype)
        else:
            k_entry = k.mean(dim=2, keepdim=True)
            v_entry = v.mean(dim=2, keepdim=True)
        w_entry = torch.full(
            (k.size(0), k.size(1), 1),
            float(n),
            device=k.device,
            dtype=k.dtype,
        )
        if not with_stats:
            return (k_entry, v_entry, w_entry) if not has_imp else (k_entry, v_entry, w_entry, imp_entry)

        if has_imp:
            sqrt_p = p.sqrt()  # (B, G, n, 1), fp32
            k_centered = (k - k_entry).float() * sqrt_p
            v_centered = (v - v_entry).float() * sqrt_p
        else:
            inv_sqrt_n = float(n) ** -0.5
            k_centered = (k - k_entry).float() * inv_sqrt_n
            v_centered = (v - v_entry).float() * inv_sqrt_n
        sigma_u, sigma2 = _rank1_psd_from_factors(k_centered)
        gamma_b, gamma_a, gamma = _rank1_cross_from_factors(v_centered, k_centered)
        stats_out = (
            k_entry,
            v_entry,
            w_entry,
            sigma_u.unsqueeze(2).to(k.dtype),
            sigma2.unsqueeze(2).to(k.dtype),
            gamma_a.unsqueeze(2).to(k.dtype),
            gamma_b.unsqueeze(2).to(k.dtype),
            gamma.unsqueeze(2).to(k.dtype),
        )
        return stats_out if not has_imp else stats_out + (imp_entry,)

    # ------------------------------------------------------------------
    # Compact operation: merge two B-slot blocks -> one B-slot block (adjacent pairs)
    # ------------------------------------------------------------------

    @staticmethod
    def compact(
        k1: torch.Tensor, v1: torch.Tensor, w1: torch.Tensor,
        k2: torch.Tensor, v2: torch.Tensor, w2: torch.Tensor,
        sigma_u1: torch.Tensor | None = None,
        sigma2_1: torch.Tensor | None = None,
        gamma_a1: torch.Tensor | None = None,
        gamma_b1: torch.Tensor | None = None,
        gamma1: torch.Tensor | None = None,
        sigma_u2: torch.Tensor | None = None,
        sigma2_2: torch.Tensor | None = None,
        gamma_a2: torch.Tensor | None = None,
        gamma_b2: torch.Tensor | None = None,
        gamma2: torch.Tensor | None = None,
        imp1: torch.Tensor | None = None,
        imp2: torch.Tensor | None = None,
        imp_lambda: float = 1.0,
        p_lo1: torch.Tensor | None = None,
        p_hi1: torch.Tensor | None = None,
        sum_wp1: torch.Tensor | None = None,
        p_lo2: torch.Tensor | None = None,
        p_hi2: torch.Tensor | None = None,
        sum_wp2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Merge two B-slot blocks into one B-slot block.

        Concatenates the two blocks in time order (k1=older, k2=newer) and pairs
        ADJACENT slots: (slot 2i, slot 2i+1) -> slot i. Every merged slot covers
        a contiguous span, and the weighted mean keeps each slot's key equal to
        the true weighted mean over all tokens it covers (mean-merge is
        associative), so no error accumulates across levels.

        If the rank-1 stats are provided for both input blocks, they are merged
        with Chan's parallel covariance formula and truncated back to rank 1 via
        the iterative small-matrix routines above (near-optimal, not exact — see
        their docstrings). Without stats, this preserves the old three-tensor
        return for tests and diagnostic callers.

        ``imp1``/``imp2`` are optional per-slot importance-mass tensors (same
        shape as ``w1``/``w2``), independent of the token-count weights: when
        given, they (not ``w1``/``w2``) drive the pooling ``alpha`` and the
        Chan-merge fractions, and their sum is returned as one extra trailing
        tensor. ``w_total`` (token count, used only for the ``log(w)`` mass
        bias) is always computed from ``w1``/``w2`` regardless. Omitting them
        reproduces the exact prior count-weighted behavior and return shape.
        ``imp_lambda`` (only consulted when ``imp1``/``imp2`` are given) linearly
        blends the importance share toward the count share ``wa/w_total`` —
        ``1.0`` (default) is pure importance (unchanged expression, bit-identical
        to the pre-``imp_lambda`` code), ``0.0`` is numerically equivalent to the
        count-weighted path.

        ``p_lo*``/``p_hi*``/``sum_wp*`` are optional per-slot semantic-cluster
        anchor fields (see ``log_kv_position.merge_anchors``); when given for
        both inputs, the merged ``(p_lo, p_hi, sum_wp)`` is appended as a
        trailing 3-tuple, independent of the ``imp``/rank-1-stats branches.
        Omitting them reproduces the exact prior return shape.
        """
        k_cat = torch.cat([k1, k2], dim=-2)  # (B, G, 2B, D)
        v_cat = torch.cat([v1, v2], dim=-2)
        w_cat = torch.cat([w1, w2], dim=-1)  # (B, G, 2B)

        ka = k_cat[..., 0::2, :]  # even slots (older half of each adjacent pair)
        kb = k_cat[..., 1::2, :]  # odd slots  (newer half of each adjacent pair)
        va = v_cat[..., 0::2, :]
        vb = v_cat[..., 1::2, :]
        wa = w_cat[..., 0::2]
        wb = w_cat[..., 1::2]

        w_total = wa + wb  # (B, G, B) -- token count, always count-based

        has_imp = imp1 is not None
        if has_imp:
            imp_cat = torch.cat([imp1, imp2], dim=-1)
            impa, impb = imp_cat[..., 0::2], imp_cat[..., 1::2]
            imp_total = impa + impb
            if imp_lambda >= 1.0:
                alpha = (impa / imp_total.clamp(min=1e-8)).unsqueeze(-1)
            else:
                alpha_imp = impa / imp_total.clamp(min=1e-8)
                alpha_w = wa / w_total.clamp(min=1e-8)
                alpha = (imp_lambda * alpha_imp + (1.0 - imp_lambda) * alpha_w).unsqueeze(-1)
        else:
            alpha = (wa / w_total.clamp(min=1e-8)).unsqueeze(-1)  # (B, G, B, 1)

        k_out = alpha * ka + (1 - alpha) * kb
        v_out = alpha * va + (1 - alpha) * vb

        has_anchors = p_lo1 is not None
        if has_anchors:
            if any(x is None for x in (p_hi1, sum_wp1, p_lo2, p_hi2, sum_wp2)):
                raise ValueError("compact() requires either all anchor fields or none")
            anchors_out = merge_anchors(p_lo1, p_hi1, sum_wp1, p_lo2, p_hi2, sum_wp2)

        def _with_anchors(out: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            return out + anchors_out if has_anchors else out

        if sigma_u1 is None:
            return _with_anchors((k_out, v_out, w_total) if not has_imp else (k_out, v_out, w_total, imp_total))

        if any(x is None for x in (
            sigma2_1, gamma_a1, gamma_b1, gamma1,
            sigma_u2, sigma2_2, gamma_a2, gamma_b2, gamma2,
        )):
            raise ValueError("compact() requires either all rank-1 stats or none")

        su_cat = torch.cat([sigma_u1, sigma_u2], dim=-2)
        s2_cat = torch.cat([sigma2_1, sigma2_2], dim=-1)
        ga_cat = torch.cat([gamma_a1, gamma_a2], dim=-2)
        gb_cat = torch.cat([gamma_b1, gamma_b2], dim=-2)
        gm_cat = torch.cat([gamma1, gamma2], dim=-1)

        sua, sub = su_cat[..., 0::2, :], su_cat[..., 1::2, :]
        s2a, s2b = s2_cat[..., 0::2], s2_cat[..., 1::2]
        gaa, gab = ga_cat[..., 0::2, :], ga_cat[..., 1::2, :]
        gba, gbb = gb_cat[..., 0::2, :], gb_cat[..., 1::2, :]
        gma, gmb = gm_cat[..., 0::2], gm_cat[..., 1::2]

        if has_imp:
            frac_a_imp = impa.float() / imp_total.float().clamp_min(1e-8)
            frac_b_imp = impb.float() / imp_total.float().clamp_min(1e-8)
            if imp_lambda >= 1.0:
                frac_a, frac_b = frac_a_imp, frac_b_imp
            else:
                frac_a_w = wa.float() / w_total.float().clamp_min(1e-8)
                frac_b_w = wb.float() / w_total.float().clamp_min(1e-8)
                frac_a = imp_lambda * frac_a_imp + (1.0 - imp_lambda) * frac_a_w
                frac_b = imp_lambda * frac_b_imp + (1.0 - imp_lambda) * frac_b_w
        else:
            frac_a = wa.float() / w_total.float().clamp_min(1e-8)
            frac_b = wb.float() / w_total.float().clamp_min(1e-8)
        cross_frac = frac_a * frac_b

        dk = (ka - kb).float()
        dv = (va - vb).float()
        sigma_factors = torch.stack(
            [
                (frac_a * s2a.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * sua.float(),
                (frac_b * s2b.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * sub.float(),
                cross_frac.clamp_min(0.0).sqrt().unsqueeze(-1) * dk,
            ],
            dim=-2,
        )
        sigma_u, sigma2 = _rank1_psd_from_factors(sigma_factors)

        left_factors = torch.stack(
            [
                (frac_a * gma.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * gba.float(),
                (frac_b * gmb.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * gbb.float(),
                cross_frac.clamp_min(0.0).sqrt().unsqueeze(-1) * dv,
            ],
            dim=-2,
        )
        right_factors = torch.stack(
            [
                (frac_a * gma.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * gaa.float(),
                (frac_b * gmb.float()).clamp_min(0.0).sqrt().unsqueeze(-1) * gab.float(),
                cross_frac.clamp_min(0.0).sqrt().unsqueeze(-1) * dk,
            ],
            dim=-2,
        )
        gamma_b, gamma_a, gamma = _rank1_cross_from_factors(left_factors, right_factors)
        stats_out = (
            k_out,
            v_out,
            w_total,
            sigma_u.to(k_out.dtype),
            sigma2.to(k_out.dtype),
            gamma_a.to(k_out.dtype),
            gamma_b.to(v_out.dtype),
            gamma.to(k_out.dtype),
        )
        return _with_anchors(stats_out if not has_imp else stats_out + (imp_total,))

    # ------------------------------------------------------------------
    # Add compact entry to level 0; carry to level 1+ when full
    # ------------------------------------------------------------------

    def _add_compact_entry(
        self,
        k_entry: torch.Tensor,
        v_entry: torch.Tensor,
        w_entry: torch.Tensor,
        sigma_u_entry: torch.Tensor | None = None,
        sigma2_entry: torch.Tensor | None = None,
        gamma_a_entry: torch.Tensor | None = None,
        gamma_b_entry: torch.Tensor | None = None,
        gamma_entry: torch.Tensor | None = None,
        imp_entry: torch.Tensor | None = None,
    ) -> None:
        """Add one compact entry to level 0. If level 0 is full, binary carry to levels 1+."""
        self._append_level0(
            k_entry.unsqueeze(2),
            v_entry.unsqueeze(2),
            w_entry.unsqueeze(2),
            None if sigma_u_entry is None else sigma_u_entry.unsqueeze(2),
            None if sigma2_entry is None else sigma2_entry.unsqueeze(2),
            None if gamma_a_entry is None else gamma_a_entry.unsqueeze(2),
            None if gamma_b_entry is None else gamma_b_entry.unsqueeze(2),
            None if gamma_entry is None else gamma_entry.unsqueeze(2),
            None if imp_entry is None else imp_entry.unsqueeze(2),
        )

    # ------------------------------------------------------------------
    # Binary carry: promote B entries through levels 1+
    # ------------------------------------------------------------------

    def _binary_carry(
        self,
        block_k: torch.Tensor,
        block_v: torch.Tensor,
        block_w: torch.Tensor,
        block_sigma_u: torch.Tensor | None = None,
        block_sigma2: torch.Tensor | None = None,
        block_gamma_a: torch.Tensor | None = None,
        block_gamma_b: torch.Tensor | None = None,
        block_gamma: torch.Tensor | None = None,
        block_imp: torch.Tensor | None = None,
    ) -> None:
        new_k, new_v, new_w = block_k, block_v, block_w
        new_su, new_s2 = block_sigma_u, block_sigma2
        new_ga, new_gb, new_gm = block_gamma_a, block_gamma_b, block_gamma
        new_imp = block_imp
        for ell in range(1, self.max_levels):
            ek, ev, ew = self._get_level(ell)
            if self._level_count(ell) == 0:
                self._set_level(ell, new_k, new_v, new_w, new_su, new_s2, new_ga, new_gb, new_gm, imp=new_imp)
                return

            eimp = self._get_level_imp(ell) if new_imp is not None else None
            if new_su is None:
                if new_imp is None:
                    new_k, new_v, new_w = self.compact(ek, ev, ew, new_k, new_v, new_w)
                else:
                    new_k, new_v, new_w, new_imp = self.compact(
                        ek, ev, ew, new_k, new_v, new_w,
                        imp1=eimp, imp2=new_imp, imp_lambda=self.importance_pooling_lambda,
                    )
            else:
                esu, es2, ega, egb, egm = self._get_level_stats(ell)
                if new_imp is None:
                    (
                        new_k,
                        new_v,
                        new_w,
                        new_su,
                        new_s2,
                        new_ga,
                        new_gb,
                        new_gm,
                    ) = self.compact(
                        ek, ev, ew,
                        new_k, new_v, new_w,
                        esu, es2, ega, egb, egm,
                        new_su, new_s2, new_ga, new_gb, new_gm,
                    )
                else:
                    (
                        new_k,
                        new_v,
                        new_w,
                        new_su,
                        new_s2,
                        new_ga,
                        new_gb,
                        new_gm,
                        new_imp,
                    ) = self.compact(
                        ek, ev, ew,
                        new_k, new_v, new_w,
                        esu, es2, ega, egb, egm,
                        new_su, new_s2, new_ga, new_gb, new_gm,
                        imp1=eimp, imp2=new_imp, imp_lambda=self.importance_pooling_lambda,
            )
            self._clear_level(ell)
        raise RuntimeError("LogStructuredKVCache: binary carry overflow; max_levels formula needs review.")

    # ------------------------------------------------------------------
    # Token accounting (append-only contract)
    # ------------------------------------------------------------------

    def _count_tokens(self, n: int) -> None:
        """Advance the committed-token counter, enforcing the sizing contract."""
        if self.token_count + n > self.max_seq_length:
            raise RuntimeError(
                f"LogStructuredKVCache: token overflow! "
                f"token_count={self.token_count}, n={n}, max_seq_length={self.max_seq_length}."
            )
        self.token_count += n

    # ------------------------------------------------------------------
    # Flush: compact oldest 2 tokens from buffer -> level 0
    # ------------------------------------------------------------------

    def _flush_recent(self) -> None:
        """Compact the oldest 2 tokens from the sliding window into level 0."""
        if self.recent_count < 2:
            return
        self._flush_pairs(2)

    def _flush_pairs(self, flush_len: int) -> None:
        """Batched equivalent of ``flush_len // 2`` sequential ``_flush_recent``
        calls: pairwise-merge the oldest ``flush_len`` window tokens into level-0
        entries in one shot, then shift the window once.

        The state trajectory is identical to the sequential version — the same
        (2i, 2i+1) pairs are merged with the same mean, entries reach level 0 in
        the same order, and carries fire at the same counts — but the O(window)
        clone+shift happens once instead of once per pair.
        """
        f = flush_len // 2
        rk = self.recent_k[:, :, :flush_len, :]
        rv = self.recent_v[:, :, :flush_len, :]
        B_, G_, _, kd = rk.shape
        vd = rv.size(-1)
        # Same arithmetic as _compact_tokens on each 2-token pair (mean over a
        # size-2 dim in the storage dtype), just for all f pairs at once.
        rk_pairs = rk.reshape(B_, G_, f, 2, kd)
        rv_pairs = rv.reshape(B_, G_, f, 2, vd)
        if self.importance_pooling:
            # Heuristic per-token importance: post-RoPE key L2 norm (no new
            # learnable params; a pure function of already-detached k, so no
            # backward-path change is needed -- see module docstring).
            imp_tok = rk.float().norm(dim=-1)  # (B, G, flush_len)
            temp = self.importance_pooling_temperature
            if temp != 1.0:
                imp_tok = imp_tok.clamp_min(_RANK1_EPS) ** temp
            imp_pairs = imp_tok.reshape(B_, G_, f, 2)
            imp_a, imp_b = imp_pairs[..., 0], imp_pairs[..., 1]
            pimp = imp_a + imp_b  # (B, G, f), raw (unclamped) cumulative mass
            frac_a_imp = imp_a / pimp.clamp_min(_RANK1_EPS)
            lam = self.importance_pooling_lambda
            # Raw tokens each carry w=1, so the uniform share of a pair is
            # exactly 0.5 -- blending toward it is `lam * frac_a_imp + (1-lam) * 0.5`.
            frac_a = (frac_a_imp if lam >= 1.0 else (lam * frac_a_imp + (1.0 - lam) * 0.5)).unsqueeze(-1)
            frac_b = 1.0 - frac_a
            pk = frac_a * rk_pairs[:, :, :, 0, :] + frac_b * rk_pairs[:, :, :, 1, :]
            pv = frac_a * rv_pairs[:, :, :, 0, :] + frac_b * rv_pairs[:, :, :, 1, :]
        else:
            pk = rk_pairs.mean(dim=3)
            pv = rv_pairs.mean(dim=3)
            pimp = None
            frac_a = frac_b = 0.5
        pw = torch.full((B_, G_, f), 2.0, device=rk.device, dtype=rk.dtype)
        if self.second_order:
            psu, ps2, pga, pgb, pgm = _pair_rank1_stats(
                rk_pairs[:, :, :, 0, :],
                rk_pairs[:, :, :, 1, :],
                rv_pairs[:, :, :, 0, :],
                rv_pairs[:, :, :, 1, :],
                frac_a.squeeze(-1) if self.importance_pooling else frac_a,
                frac_b.squeeze(-1) if self.importance_pooling else frac_b,
            )
            self._append_level0(pk, pv, pw, psu, ps2, pga, pgb, pgm, pimp=pimp)
        else:
            self._append_level0(pk, pv, pw, pimp=pimp)

        # Shift the survivors to the front. Source/destination overlap in the
        # same storage; PyTorch copy_ with overlapping src/dst is undefined (may
        # corrupt silently on CUDA), so the source must be cloned first.
        remaining = self.recent_count - flush_len
        if remaining > 0:
            self.recent_k[:, :, :remaining, :] = self.recent_k[:, :, flush_len:self.recent_count, :].clone()
            self.recent_v[:, :, :remaining, :] = self.recent_v[:, :, flush_len:self.recent_count, :].clone()
        self.recent_k[:, :, remaining:self.recent_count, :].zero_()
        self.recent_v[:, :, remaining:self.recent_count, :].zero_()
        self.recent_count = remaining

    def _append_level0(
        self,
        pk: torch.Tensor,
        pv: torch.Tensor,
        pw: torch.Tensor,
        psu: torch.Tensor | None = None,
        ps2: torch.Tensor | None = None,
        pga: torch.Tensor | None = None,
        pgb: torch.Tensor | None = None,
        pgm: torch.Tensor | None = None,
        pimp: torch.Tensor | None = None,
    ) -> None:
        """Append f compact entries to level 0 in order, carrying when it fills.

        Trajectory-identical to f sequential ``_add_compact_entry`` calls: the
        binary carry fires exactly when the count reaches B, between the same
        two entries as in the sequential version.
        """
        f = pk.size(2)
        if f == 0:
            return
        off = 0
        while off < f:
            idx = self._level_count(0)
            take = min(self.B - idx, f - off)
            dst = slice(idx, idx + take)
            src = slice(off, off + take)
            self.level_k[:, :, 0, 0, dst, :].copy_(pk[:, :, src, :])
            self.level_v[:, :, 0, 0, dst, :].copy_(pv[:, :, src, :])
            self.level_w[:, :, 0, 0, dst].copy_(pw[:, :, src])
            if psu is None:
                self.level_sigma_u[:, :, 0, 0, dst, :].zero_()
                self.level_sigma2[:, :, 0, 0, dst].zero_()
                self.level_gamma_a[:, :, 0, 0, dst, :].zero_()
                self.level_gamma_b[:, :, 0, 0, dst, :].zero_()
                self.level_gamma[:, :, 0, 0, dst].zero_()
            else:
                self.level_sigma_u[:, :, 0, 0, dst, :].copy_(psu[:, :, src, :])
                self.level_sigma2[:, :, 0, 0, dst].copy_(ps2[:, :, src])
                self.level_gamma_a[:, :, 0, 0, dst, :].copy_(pga[:, :, src, :])
                self.level_gamma_b[:, :, 0, 0, dst, :].copy_(pgb[:, :, src, :])
                self.level_gamma[:, :, 0, 0, dst].copy_(pgm[:, :, src])
            if pimp is None:
                self.level_imp[:, :, 0, 0, dst].zero_()
            else:
                self.level_imp[:, :, 0, 0, dst].copy_(pimp[:, :, src])
            self.pad_mask[:, :, 0, 0, dst].zero_()
            idx += take
            off += take
            self.level_count[:, :, 0, 0].fill_(idx)
            if idx == self.B:
                lk, lv, lw = self._get_level(0)
                limp = self._get_level_imp(0) if pimp is not None else None
                if self.second_order:
                    lsu, ls2, lga, lgb, lgm = self._get_level_stats(0)
                    self._binary_carry(
                        lk, lv, lw,
                        lsu, ls2, lga, lgb, lgm,
                        block_imp=limp,
                    )
                else:
                    self._binary_carry(lk, lv, lw, block_imp=limp)
                self._clear_level(0)

    # ------------------------------------------------------------------
    # Ingest chunk (testing): direct compact into level 0, bypass buffer
    # ------------------------------------------------------------------

    def ingest_chunk(
        self,
        k: torch.Tensor,   # (B, G, t_chunk, k_dim) full post-RoPE keys, detached
        v: torch.Tensor,   # (B, G, t_chunk, v_dim) detached
    ) -> None:
        """Compact a chunk of full keys + values straight into level 0.
        Bypasses the buffer. Intended for testing.

        When ``self.importance_pooling`` is set, uses the same heuristic
        per-token importance (post-RoPE key L2 norm) as the real streaming
        path (``_flush_pairs``) -- note this pools the whole chunk in one
        flat weighted average, not the real path's pairwise 2:1 hierarchy,
        so for ``t_chunk > 2`` this is a testing convenience, not a
        trajectory-identical stand-in (same caveat already applies to the
        unweighted default).
        """
        self._count_tokens(k.size(2))
        imp = k.float().norm(dim=-1) if self.importance_pooling else None
        if imp is not None and self.importance_pooling_temperature != 1.0:
            imp = imp.clamp_min(_RANK1_EPS) ** self.importance_pooling_temperature

        entries = self._compact_tokens(
            k, v, with_stats=True, imp=imp, imp_lambda=self.importance_pooling_lambda,
        )
        if self.importance_pooling:
            (
                k_entry, v_entry, w_entry,
                sigma_u_entry, sigma2_entry, gamma_a_entry, gamma_b_entry, gamma_entry,
                imp_entry,
            ) = entries
            imp_entry = imp_entry.squeeze(2)
        else:
            (
                k_entry, v_entry, w_entry,
                sigma_u_entry, sigma2_entry, gamma_a_entry, gamma_b_entry, gamma_entry,
            ) = entries
            imp_entry = None
        k_entry = k_entry.squeeze(2)
        v_entry = v_entry.squeeze(2)
        w_entry = w_entry.squeeze(2)
        sigma_u_entry = sigma_u_entry.squeeze(2)
        sigma2_entry = sigma2_entry.squeeze(2)
        gamma_a_entry = gamma_a_entry.squeeze(2)
        gamma_b_entry = gamma_b_entry.squeeze(2)
        gamma_entry = gamma_entry.squeeze(2)

        self._add_compact_entry(
            k_entry, v_entry, w_entry,
            sigma_u_entry, sigma2_entry, gamma_a_entry, gamma_b_entry, gamma_entry,
            imp_entry=imp_entry,
        )

    # ------------------------------------------------------------------
    # Add to recent (training): sliding window entry point
    # ------------------------------------------------------------------

    def add_recent(
        self,
        k: torch.Tensor,   # (B, G, n, k_dim) full post-RoPE keys, detached
        v: torch.Tensor,   # (B, G, n, v_dim) detached
    ) -> None:
        """Add tokens to the sliding window. When the window overflows, the
        oldest 2 tokens are flushed (compacted) into level 0.
        """
        n = k.size(2)
        # Explicit raise (not assert): must survive `python -O`.
        if n > self.recent_size:
            raise ValueError(f"chunk size {n} exceeds recent_size {self.recent_size}")

        self._count_tokens(n)

        # Batched flush: streaming lazily flushes the oldest pair each time the
        # window refills, so over this whole chunk it flushes ceil(overflow/2)
        # pairs — all taken from the CURRENT window front (appends only ever go
        # to the right). Flushing them up front in one batch reaches the exact
        # same final state with one shift instead of one per pair.
        overflow = self.recent_count + n - self.recent_size
        if overflow > 0:
            flush_len = 2 * ((overflow + 1) // 2)
            if flush_len <= self.recent_count:
                self._flush_pairs(flush_len)
            else:
                # Degenerate corner (odd recent_count with n == recent_size):
                # fall back to the lazy interleaved order.
                offset = 0
                while offset < n:
                    if self.recent_count == self.recent_size:
                        self._flush_recent()
                    capacity = self.recent_size - self.recent_count
                    if capacity <= 0:
                        raise RuntimeError(
                            "LogStructuredKVCache.add_recent() could not make room in the recent buffer; "
                            f"recent_count={self.recent_count}, recent_size={self.recent_size}."
                        )
                    take = min(capacity, n - offset)
                    self.recent_k[:, :, self.recent_count:self.recent_count + take, :] = k[:, :, offset:offset + take, :]
                    self.recent_v[:, :, self.recent_count:self.recent_count + take, :] = v[:, :, offset:offset + take, :]
                    self.recent_count += take
                    offset += take
                return

        self.recent_k[:, :, self.recent_count:self.recent_count + n, :] = k
        self.recent_v[:, :, self.recent_count:self.recent_count + n, :] = v
        self.recent_count += n

    # ------------------------------------------------------------------
    # nn.Module forward intentionally disabled
    # ------------------------------------------------------------------

    def forward(
        self,
        input_pos: torch.Tensor,
        k: torch.Tensor,   # (B, G, T, k_dim) post-RoPE
        v: torch.Tensor,   # (B, G, T, v_dim)
    ) -> NoReturn:
        """Disabled direct update API.

        LogKV attention must build a temporary state that includes the current
        token(s), compute slot attention from that state, and only then commit
        the token(s) to the cache. Calling the cache directly used to implement
        an incompatible raw-prefill / write-before-decode path, so it is
        intentionally disabled. ``input_pos`` is accepted only to keep the
        stale call site fail-fast and self-explanatory. Use
        ``GPT.set_log_kv_cache()`` and let ``CausalSelfAttention`` manage the
        update order, or call
        ``get_attention_state()`` and ``add_recent()`` explicitly.
        """
        raise RuntimeError(
            "LogStructuredKVCache.forward() is disabled/deprecated: direct cache calls cannot implement "
            "LogKV correctly because attention must include the current token(s) before committing them. "
            "Use GPT.set_log_kv_cache() so CausalSelfAttention owns the update order, or use the explicit "
            "get_attention_state()/add_recent() helpers in tests."
        )

    # ------------------------------------------------------------------
    # Build attention state: slot-granular, O(recent + B*log N) entries
    # ------------------------------------------------------------------

    def get_attention_state(self, with_stats: bool = False) -> CacheAttentionState:
        """Assemble the cache state for ``log_kv_slot_attention``.

        Returns:
            slot_k: (B, G, n_slots, k_dim) — time-ordered slot keys: compact
                    levels oldest (highest level) first down to level 0 as a
                    fixed-size pooled prefix, then exact pin/recent slots.
            slot_v: (B, G, n_slots, v_dim)
            slot_w: (B, G, n_slots) — token count per slot (0 for invalid
                    pooled slots, 1 for recent/exact pins).
            Returns a ``CacheAttentionState`` with ``slot_valid`` masking the
            fixed pooled prefix. If ``with_stats=True``, also fills:
            slot_sigma_u / slot_sigma2: rank-1 key covariance stats;
            slot_gamma_a / slot_gamma_b / slot_gamma: rank-1 value-key cross
                covariance stats. Exact recent and pinned tokens have zero stats.

        Valid pooled slots cover contiguous, time-ordered spans after filtering
        by ``slot_valid``. With salience pins active (pin_count > 0) that
        contiguity invariant no longer holds past the pooled prefix: pins are
        scattered duplicates of tokens the hierarchy also covers. Attention
        itself is order-agnostic over the state (the whole state is fully
        visible; only the appended in-flight chunk is causal).
        """
        pooled_k = self.level_k[:, :, 0].flip(2).reshape(
            self.batch_size, self.n_groups, self.L_alloc * self.B_prime, self.k_dim
        )
        pooled_v = self.level_v[:, :, 0].flip(2).reshape(
            self.batch_size, self.n_groups, self.L_alloc * self.B_prime, self.v_dim
        )
        pooled_w = self.level_w[:, :, 0].flip(2).reshape(
            self.batch_size, self.n_groups, self.L_alloc * self.B_prime
        )
        counts = self.level_count[:, :, 0].flip(2).to(torch.long)
        slot_ids = torch.arange(self.B_prime, device=self.level_count.device).view(1, 1, 1, self.B_prime)
        pooled_valid = (slot_ids < counts.unsqueeze(-1)).reshape(
            self.batch_size, self.n_groups, self.L_alloc * self.B_prime
        )

        k_parts: list[torch.Tensor] = [pooled_k]
        v_parts: list[torch.Tensor] = [pooled_v]
        w_parts: list[torch.Tensor] = [pooled_w]
        sigma_u_parts: list[torch.Tensor] = []
        sigma2_parts: list[torch.Tensor] = []
        gamma_a_parts: list[torch.Tensor] = []
        gamma_b_parts: list[torch.Tensor] = []
        gamma_parts: list[torch.Tensor] = []
        if with_stats:
            sigma_u_parts.append(self.level_sigma_u[:, :, 0].flip(2).reshape(
                self.batch_size, self.n_groups, self.L_alloc * self.B_prime, self.k_dim
            ))
            sigma2_parts.append(self.level_sigma2[:, :, 0].flip(2).reshape(
                self.batch_size, self.n_groups, self.L_alloc * self.B_prime
            ))
            gamma_a_parts.append(self.level_gamma_a[:, :, 0].flip(2).reshape(
                self.batch_size, self.n_groups, self.L_alloc * self.B_prime, self.k_dim
            ))
            gamma_b_parts.append(self.level_gamma_b[:, :, 0].flip(2).reshape(
                self.batch_size, self.n_groups, self.L_alloc * self.B_prime, self.v_dim
            ))
            gamma_parts.append(self.level_gamma[:, :, 0].flip(2).reshape(
                self.batch_size, self.n_groups, self.L_alloc * self.B_prime
            ))

        # Salience pins: exact w=1 duplicates from the compressed region,
        # placed between the levels and the recent window (they are older than
        # everything in recent by construction).
        if self.pin_count > 0:
            k_parts.append(self.pin_k[:, :, :self.pin_count, :])
            v_parts.append(self.pin_v[:, :, :self.pin_count, :])
            w_parts.append(
                self.level_w.new_ones(self.batch_size, self.n_groups, self.pin_count)
            )
            if with_stats:
                sigma_u_parts.append(self.pin_k[:, :, :self.pin_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.pin_count, self.k_dim
                ))
                sigma2_parts.append(self.pin_k.new_zeros(self.batch_size, self.n_groups, self.pin_count))
                gamma_a_parts.append(self.pin_k[:, :, :self.pin_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.pin_count, self.k_dim
                ))
                gamma_b_parts.append(self.pin_v[:, :, :self.pin_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.pin_count, self.v_dim
                ))
                gamma_parts.append(self.pin_k.new_zeros(self.batch_size, self.n_groups, self.pin_count))

        # Recent window: exact tokens, weight 1 each
        if self.recent_count > 0:
            k_parts.append(self.recent_k[:, :, :self.recent_count, :])
            v_parts.append(self.recent_v[:, :, :self.recent_count, :])
            w_parts.append(
                self.level_w.new_ones(self.batch_size, self.n_groups, self.recent_count)
            )
            if with_stats:
                sigma_u_parts.append(self.recent_k[:, :, :self.recent_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.recent_count, self.k_dim
                ))
                sigma2_parts.append(self.recent_k.new_zeros(self.batch_size, self.n_groups, self.recent_count))
                gamma_a_parts.append(self.recent_k[:, :, :self.recent_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.recent_count, self.k_dim
                ))
                gamma_b_parts.append(self.recent_v[:, :, :self.recent_count, :].new_zeros(
                    self.batch_size, self.n_groups, self.recent_count, self.v_dim
                ))
                gamma_parts.append(self.recent_k.new_zeros(self.batch_size, self.n_groups, self.recent_count))

        slot_k = torch.cat(k_parts, dim=-2)
        slot_v = torch.cat(v_parts, dim=-2)
        slot_w = torch.cat(w_parts, dim=-1)
        if not with_stats:
            return CacheAttentionState(slot_k, slot_v, slot_w, slot_valid=pooled_valid)
        return CacheAttentionState(
            slot_k=slot_k,
            slot_v=slot_v,
            slot_w=slot_w,
            slot_valid=pooled_valid,
            slot_sigma_u=torch.cat(sigma_u_parts, dim=-2),
            slot_sigma2=torch.cat(sigma2_parts, dim=-1),
            slot_gamma_a=torch.cat(gamma_a_parts, dim=-2),
            slot_gamma_b=torch.cat(gamma_b_parts, dim=-2),
            slot_gamma=torch.cat(gamma_parts, dim=-1),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _convert_dtype(self, dtype: torch.dtype) -> None:
        """Reconcile all data buffers with the activation dtype.

        Needed when the cache was allocated at a different (default) dtype than
        the running activations — e.g. bf16 inference where ``set_log_kv_cache``
        received no explicit ``dtype`` and the process default is fp32. Without
        this, ``get_attention_state`` / ``add_recent`` would try to ``cat`` /
        ``matmul`` fp32 buffers against bf16 activations and raise.

        Idempotent and cheap: a no-op once the buffers already match, so it is
        safe to call on every forward (mirrors ``KVCache.forward``'s reconcile).
        ``level_count``/``pad_mask`` stay integer/bool and are deliberately
        untouched. ``level_w``/``level_imp`` stay fp32 per the indexed storage
        contract: they feed scalar merge/bias math, not activation matmuls.
        """
        if self.recent_k.dtype == dtype:
            return
        self.recent_k = self.recent_k.to(dtype)
        self.recent_v = self.recent_v.to(dtype)
        self.pin_k = self.pin_k.to(dtype)
        self.pin_v = self.pin_v.to(dtype)
        self.level_k = self.level_k.to(dtype)
        self.level_v = self.level_v.to(dtype)
        self.level_sigma_u = self.level_sigma_u.to(dtype)
        self.level_sigma2 = self.level_sigma2.to(dtype)
        self.level_gamma_a = self.level_gamma_a.to(dtype)
        self.level_gamma_b = self.level_gamma_b.to(dtype)
        self.level_gamma = self.level_gamma.to(dtype)

    def reset_parameters(self) -> None:
        """Reset all buffers to zero."""
        self.token_count = 0
        self.recent_k.zero_()
        self.recent_v.zero_()
        self.recent_count = 0
        self.pin_k.zero_()
        self.pin_v.zero_()
        self.pin_count = 0
        self.level_k.zero_()
        self.level_v.zero_()
        self.level_w.zero_()
        self.level_imp.zero_()
        self.level_sigma_u.zero_()
        self.level_sigma2.zero_()
        self.level_gamma_a.zero_()
        self.level_gamma_b.zero_()
        self.level_gamma.zero_()
        self.level_count.zero_()
        self.pad_mask.zero_()

    @property
    def total_slots(self) -> int:
        return self.recent_count + self.pin_count + int(self.level_count[0, 0, 0].sum().item())

    @property
    def total_tokens_covered(self) -> int:
        return self.token_count


# ======================================================================
# Slot attention: merged-position slots + log-multiplicity mass bias
# ======================================================================


def append_exact_tokens(
    state: CacheAttentionState,
    k_new: torch.Tensor,    # (B, G, n, k_dim) exact tokens, appended in time order
    v_new: torch.Tensor,    # (B, G, n, v_dim)
) -> CacheAttentionState:
    """Append exact per-token entries (w=1 slots) after the cached slots.

    Used by the attention layer to make the current chunk (and any pending
    token) visible to attention BEFORE it is committed to the cache. Gradient
    flows through ``k_new``/``v_new``; the ones-weights are constants.
    ``slot_valid``/``M_s`` cover only the pooled prefix and are passed through.
    If slot rank-1 stats are provided, appended exact tokens receive zero stats.
    """
    n_new = k_new.size(2)
    ones = state.slot_w.new_ones(state.slot_w.size(0), state.slot_w.size(1), n_new)
    k_all = torch.cat([state.slot_k, k_new], dim=2)
    v_all = torch.cat([state.slot_v, v_new], dim=2)
    w_all = torch.cat([state.slot_w, ones], dim=-1)
    if state.slot_sigma_u is None:
        if any(x is not None for x in (state.slot_sigma2, state.slot_gamma_a, state.slot_gamma_b, state.slot_gamma)):
            raise ValueError("append_exact_tokens() requires either all rank-1 stats or none")
        return state._replace(slot_k=k_all, slot_v=v_all, slot_w=w_all)

    if any(x is None for x in (state.slot_sigma2, state.slot_gamma_a, state.slot_gamma_b, state.slot_gamma)):
        raise ValueError("append_exact_tokens() requires either all rank-1 stats or none")
    return state._replace(
        slot_k=k_all,
        slot_v=v_all,
        slot_w=w_all,
        slot_sigma_u=torch.cat([state.slot_sigma_u, torch.zeros_like(k_new)], dim=2),
        slot_sigma2=torch.cat([state.slot_sigma2, state.slot_sigma2.new_zeros(ones.shape)], dim=-1),
        slot_gamma_a=torch.cat([state.slot_gamma_a, torch.zeros_like(k_new)], dim=2),
        slot_gamma_b=torch.cat([state.slot_gamma_b, torch.zeros_like(v_new)], dim=2),
        slot_gamma=torch.cat([state.slot_gamma, state.slot_gamma.new_zeros(ones.shape)], dim=-1),
    )


def _slot_mass_bias(slot_w: torch.Tensor, slot_M: torch.Tensor | None, lam: float) -> torch.Tensor | None:
    if slot_M is None:
        return lam * slot_w.to(torch.float32).log() if lam != 0.0 else None
    if slot_M.size(-1) > slot_w.size(-1):
        raise ValueError("slot_M cannot be wider than slot_w")
    bias = lam * slot_w.to(torch.float32).log() if lam != 0.0 else torch.zeros_like(slot_w, dtype=torch.float32)
    s_m = slot_M.size(-1)
    bias[..., :s_m] = anchor_mass_bias(slot_w[..., :s_m], slot_M, lam)
    return bias


def log_kv_slot_attention(
    q: torch.Tensor,        # (B, nh, T_q, k_dim) full post-RoPE queries
    slot_k: torch.Tensor,   # (B, G, S, k_dim) merged slot keys (position included)
    slot_v: torch.Tensor,   # (B, G, S, v_dim)
    slot_w: torch.Tensor,   # (B, G, S) token count per slot (>= 1)
    scale: float,
    mask: torch.Tensor | None = None,  # (T_q, S) bool, True = attend
    lam: float = 1.0,
    causal_tail: int = 0,
    slot_M: torch.Tensor | None = None,       # (B, G, S) distinct-anchor count per slot
    slot_valid: torch.Tensor | None = None,   # (B, G, S_pooled) bool; pooled-prefix only
    slot_sigma_u: torch.Tensor | None = None,  # (B, G, S, k_dim)
    slot_sigma2: torch.Tensor | None = None,   # (B, G, S)
    slot_gamma_a: torch.Tensor | None = None,  # (B, G, S, k_dim)
    slot_gamma_b: torch.Tensor | None = None,  # (B, G, S, v_dim)
    slot_gamma: torch.Tensor | None = None,    # (B, G, S)
    second_order_scale: float = 1.0,
    pin_slot_range: tuple[int, int] | None = None,
    pooled_slot_range: tuple[int, int] | None = None,
    pin_score_diag_layer: int | None = None,
    pin_score_diag_q_offset: int = 0,
    pin_score_diag_q_slice: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Slot-granular attention over merged-position entries.

        score_s = scale·(q · k_s) + 0.5·scale²·sigma2_s·(q · sigma_u_s)² + λ·log(w_s)
        read_s  = v_s + scale·gamma_s·(q · gamma_a_s)·gamma_b_s
        out     = softmax(score) · read

    One logit per SLOT — compute and memory are O(S) = O(recent + B·log N),
    never O(N). The position information lives inside ``slot_k`` (expected-RoPE
    sub-channel, see module docstring), so a single dot product covers both the
    content and the position score. The +λ·log(w_s) bias restores the softmax
    mass a w-token span would have contributed token-by-token; it is exact when
    the span's tokens are identical and first-order otherwise. When the optional
    rank-1 stats are present, the Sigma/Gamma terms add the cheap second-order
    slot corrections. λ=1 preserves mass ∝ token count; λ=0 yields a built-in
    ∝1/w long-range forgetting curve (ablation knob).

    Note the bias is added AFTER ``scale``: it is a multiplicity correction on
    the logits, not a similarity, so it must not be shrunk by 1/sqrt(d).

    Preconditions (enforced by construction in the callers):
      - ``slot_w`` entries are >= 1 (log is finite);
      - every ``mask`` row has at least one True (softmax row is finite) —
        chunk-causal masks always allow the diagonal;
      - if ``slot_valid`` is given, every row must keep >= 1 valid slot after
        all masking — checked at runtime (raises) since this one isn't
        guaranteed by construction the way ``mask``/``causal_tail`` are.

    Args:
        q: (B, nh, T_q, k_dim)
        slot_k / slot_v / slot_w: from ``get_attention_state()`` (+ optionally
            ``append_exact_tokens`` for the in-flight chunk)
        slot_sigma_u / slot_sigma2 / slot_gamma_a / slot_gamma_b / slot_gamma:
            optional rank-1 second-order slot statistics. When present, score
            receives ``0.5 * scale**2 * sigma2 * (q·sigma_u)**2`` and value
            read-out receives ``scale * gamma * (q·gamma_a) * gamma_b``. Both
            corrections are multiplied by ``second_order_scale`` as a single
            coupled gate, so CPT can warm them up together.
        scale: attention scale (applied to the dot product only)
        mask: optional (T_q, S) bool. True = allowed. General-purpose escape
            hatch (tests); the model uses ``causal_tail`` instead. None (and
            causal_tail == 0) = attend all.
        lam: weight of the log-multiplicity mass bias (default 1.0)
        slot_M: distinct-anchor count per slot (see ``log_kv_position.
            dedup_anchors``). When given, the mass bias becomes
            ``lam*log(w) - log(M)`` (the ``-log(M)`` term is unconditional,
            not gated by ``lam``) so anchor-expanded entries don't leak
            ``M``-fold extra softmax mass. None reproduces the exact prior
            ``lam*log(w)`` formula.
        slot_valid: (B, G, S_pooled) bool, True = real anchor. Covers only
            the pooled-entry prefix of the slot axis (S_pooled <= S); the
            exact/in-flight suffix is never invalid by construction. Masked
            slots get score -inf, same as ``mask``/``causal_tail`` but
            orthogonal to both (can combine with either).
        causal_tail: if > 0, the LAST ``causal_tail`` slots are the in-flight
            chunk appended behind the frozen cache state (see
            ``append_exact_tokens``); they are masked causally against the
            queries (query i sees appended entry j iff j <= i; requires
            ``causal_tail == T_q``), while everything before them stays fully
            visible. Memory: this replaces a (T_q, S) bool mask + a
            masked_fill copy of the full fp32 score tensor with one in-place
            fill on the (T_q, causal_tail) tail slice — per streaming chunk,
            per layer. Mutually exclusive with ``mask``.

    Returns:
        (B, nh, T_q, v_dim)
    """
    B, nh, T_q, k_dim = q.shape
    v_dim = slot_v.size(-1)
    S = slot_k.size(2)
    if S == 0:
        return torch.zeros(B, nh, T_q, v_dim, device=q.device, dtype=q.dtype)

    has_rank1_stats = slot_sigma_u is not None
    if has_rank1_stats and any(x is None for x in (slot_sigma2, slot_gamma_a, slot_gamma_b, slot_gamma)):
        raise ValueError("log_kv_slot_attention() requires either all rank-1 stats or none")
    use_rank1_stats = has_rank1_stats and second_order_scale != 0.0
    pin_score_diag_active = (
        LOG_KV_PIN_SCORE_DIAG.enabled
        and pin_slot_range is not None
        and pooled_slot_range is not None
        and pin_score_diag_layer is not None
        and pin_score_diag_q_slice is not None
    )

    if causal_tail:
        # Explicit raises (not asserts): survive `python -O`.
        if mask is not None:
            raise ValueError("causal_tail and mask are mutually exclusive")
        if causal_tail != T_q:
            raise ValueError(
                f"causal_tail ({causal_tail}) must equal T_q ({T_q}): the tail entries "
                "are the appended in-flight chunk, aligned one-to-one with the queries"
            )
        # (T_q, T_q) bool, True above the diagonal = blocked. Tiny (chunk-sized,
        # not S-sized) and the only allocation masking costs on this path.
        tail_blocked = torch.ones(T_q, causal_tail, dtype=torch.bool, device=q.device).triu_(1)

    # Scores in fp32: the log-w bias shifts logits by up to ~log(N) and bf16
    # resolution degrades with magnitude; fp32 keeps cross-level logit
    # differences intact. S is O(log N), so the fp32 buffer is small.
    #
    # In-place ops (mul_/add_/masked_fill_) are deliberate: none of these
    # intermediates is a saved tensor for backward (matmul saves its operands,
    # softmax saves its output, the scalar/bias ops save nothing), so mutating
    # the score buffer is autograd-safe and avoids three full-size fp32
    # temporaries per call — once per streaming chunk, per layer.
    nkv = slot_k.size(1)
    if nh != nkv:
        # --- GQA without materializing an rf× copy of the slots ---
        # repeat_interleave(rf, dim=1) on slot_k/v/w would allocate a fresh
        # (B, nh, S, ·) copy AND, in training, save that expanded tensor for
        # backward — rf× the slot memory for every one of the O(T) streaming
        # chunks, which is what runs the GPU out of memory. Instead fold the rf
        # query heads that share a KV group into their own axis and let matmul
        # broadcast the (B, nkv, 1, …) group across them: the K/V operands stay
        # (B, nkv, S, ·), nothing is duplicated, and the tensor autograd saves is
        # rf× smaller. Mathematically identical to the repeat_interleave path
        # (query head h ↔ KV group h // rf, matching model.py's GQA convention);
        # any difference is kernel-level FP reduction order, ULP-scale.
        rf = nh // nkv
        qg = q.reshape(B, nkv, rf, T_q, k_dim)               # (B, nkv, rf, T_q, k_dim)
        scores = torch.matmul(qg, slot_k.unsqueeze(2).mT)    # (B, nkv, rf, T_q, S)
        scores = scores.to(torch.float32)
        scores.mul_(scale)
        dot_stats = (
            LOG_KV_PIN_SCORE_DIAG.capture_score_stats(
                scores,
                q_slice=pin_score_diag_q_slice,
                pin_slot_range=pin_slot_range,
                pooled_slot_range=pooled_slot_range,
            )
            if pin_score_diag_active
            else None
        )
        if use_rank1_stats:
            sigma_dot = torch.matmul(qg, slot_sigma_u.unsqueeze(2).mT).to(torch.float32)
            scores.add_(
                second_order_scale * 0.5 * scale * scale
                * sigma_dot.square()
                * slot_sigma2.to(torch.float32)[:, :, None, None, :]
            )
        mass_bias = _slot_mass_bias(slot_w, slot_M, lam)
        if mass_bias is not None:
            scores.add_(mass_bias[:, :, None, None, :])
        if mask is not None:
            scores.masked_fill_(~mask.view(1, 1, 1, T_q, S), float("-inf"))
        elif causal_tail:
            scores[..., S - causal_tail:].masked_fill_(tail_blocked, float("-inf"))
        if slot_valid is not None:
            s_pooled = slot_valid.size(-1)
            scores[..., :s_pooled].masked_fill_(
                (~slot_valid)[:, :, None, None, :], float("-inf")
            )
            if not torch.isfinite(scores).any(dim=-1).all():
                raise ValueError("log_kv_slot_attention(): a query row has no valid slot")
        attn = torch.softmax(scores, dim=-1).to(q.dtype)     # (B, nkv, rf, T_q, S)
        if pin_score_diag_active:
            LOG_KV_PIN_SCORE_DIAG.record(
                layer=int(pin_score_diag_layer),
                branch="gqa",
                q_offset=int(pin_score_diag_q_offset),
                q_slice=pin_score_diag_q_slice,
                pin_slot_range=pin_slot_range,
                pooled_slot_range=pooled_slot_range,
                dot_stats=dot_stats,
                final_scores=scores,
                attn=attn,
            )
        out = torch.matmul(attn, slot_v.unsqueeze(2))        # (B, nkv, rf, T_q, v_dim)
        if use_rank1_stats:
            # Activation dtype, matching the ``attn @ slot_v`` matmul above —
            # see the MHA branch for why fp32 here bought nothing but cost the
            # tensor cores.
            gamma_dot = torch.matmul(qg, slot_gamma_a.unsqueeze(2).mT)
            gamma_weight = attn * gamma_dot * slot_gamma[:, :, None, None, :]
            corr = torch.matmul(gamma_weight, slot_gamma_b.unsqueeze(2))
            out = out + corr * (second_order_scale * scale)
        return out.reshape(B, nh, T_q, v_dim)

    # MHA (nh == nkv): one logit per slot, no head expansion needed.
    scores = torch.matmul(q, slot_k.mT).to(torch.float32)  # (B, nh, T_q, S)
    scores.mul_(scale)
    dot_stats = (
        LOG_KV_PIN_SCORE_DIAG.capture_score_stats(
            scores,
            q_slice=pin_score_diag_q_slice,
            pin_slot_range=pin_slot_range,
            pooled_slot_range=pooled_slot_range,
        )
        if pin_score_diag_active
        else None
    )
    if use_rank1_stats:
        sigma_dot = torch.matmul(q, slot_sigma_u.mT).to(torch.float32)
        scores.add_(
            second_order_scale * 0.5 * scale * scale
            * sigma_dot.square()
            * slot_sigma2.to(torch.float32).unsqueeze(-2)
        )
    mass_bias = _slot_mass_bias(slot_w, slot_M, lam)
    if mass_bias is not None:
        scores.add_(mass_bias.unsqueeze(-2))
    if mask is not None:
        scores.masked_fill_(~mask.view(1, 1, T_q, S), float("-inf"))
    elif causal_tail:
        scores[..., S - causal_tail:].masked_fill_(tail_blocked, float("-inf"))
    if slot_valid is not None:
        s_pooled = slot_valid.size(-1)
        scores[..., :s_pooled].masked_fill_((~slot_valid).unsqueeze(-2), float("-inf"))
        if not torch.isfinite(scores).any(dim=-1).all():
            raise ValueError("log_kv_slot_attention(): a query row has no valid slot")
    attn = torch.softmax(scores, dim=-1).to(q.dtype)  # (B, nh, T_q, S)
    if pin_score_diag_active:
        LOG_KV_PIN_SCORE_DIAG.record(
            layer=int(pin_score_diag_layer),
            branch="mha",
            q_offset=int(pin_score_diag_q_offset),
            q_slice=pin_score_diag_q_slice,
            pin_slot_range=pin_slot_range,
            pooled_slot_range=pooled_slot_range,
            dot_stats=dot_stats,
            final_scores=scores,
            attn=attn,
        )
    out = torch.matmul(attn, slot_v)                  # (B, nh, T_q, v_dim)
    if use_rank1_stats:
        # Value read-out correction in the activation dtype, matching the
        # ``attn @ slot_v`` matmul right above it. Promoting this one to fp32
        # bought no accuracy — its largest factor, ``attn``, has already been
        # rounded to the activation dtype, and matmul accumulates in fp32
        # regardless — while running a same-shaped matmul off the tensor cores.
        gamma_dot = torch.matmul(q, slot_gamma_a.mT)
        gamma_weight = attn * gamma_dot * slot_gamma.unsqueeze(-2)
        corr = torch.matmul(gamma_weight, slot_gamma_b)
        out = out + corr * (second_order_scale * scale)
    return out


# ======================================================================
# Low-memory training autograd: stream without a graph, replay in backward
# ======================================================================


def log_kv_chunk_attention(
    cache: LogStructuredKVCache,
    q_b: torch.Tensor,   # (B, nh, t, k_dim) current-chunk queries, post-RoPE
    k_b: torch.Tensor,   # (B, G, t, k_dim) current-chunk keys, post-RoPE
    v_b: torch.Tensor,   # (B, G, t, v_dim)
    scale: float,
    second_order_scale: float = 1.0,
) -> torch.Tensor:
    """One streaming attention step, WITHOUT committing the chunk.

    [frozen cache state (fully visible)] + [current chunk (causal)]. This is
    the shared building block of ``LogKVStreamTrainingAttention``: its forward
    stream and its backward replay must both compute chunks through this exact
    function so the recomputed graphs match the streamed outputs.

    ``second_order_scale == 0`` skips the rank-1 statistics end to end — they are
    not gathered, not concatenated, and not multiplied into the scores — so a
    zero gate costs exactly the first-order path rather than the full
    second-order path multiplied by zero.
    """
    if second_order_scale == 0.0:
        state = append_exact_tokens(cache.get_attention_state(with_stats=False), k_b, v_b)
        return log_kv_slot_attention(
            q_b, state.slot_k, state.slot_v, state.slot_w,
            scale=scale,
            causal_tail=q_b.size(2),
            slot_M=state.M_s,
            slot_valid=state.slot_valid,
        )
    state = append_exact_tokens(cache.get_attention_state(with_stats=True), k_b, v_b)
    return log_kv_slot_attention(
        q_b, state.slot_k, state.slot_v, state.slot_w,
        scale=scale,
        causal_tail=q_b.size(2),
        slot_M=state.M_s,
        slot_valid=state.slot_valid,
        slot_sigma_u=state.slot_sigma_u,
        slot_sigma2=state.slot_sigma2,
        slot_gamma_a=state.slot_gamma_a,
        slot_gamma_b=state.slot_gamma_b,
        slot_gamma=state.slot_gamma,
        second_order_scale=second_order_scale,
    )


def _sample_training_pin_positions(
    k: torch.Tensor,
    cache: LogStructuredKVCache,
    T: int,
    train_block: int,
    pin_train_max: int,
    pin_train_prob: float,
) -> tuple[int, torch.Tensor | None]:
    """Sample one chunk-boundary injection point and random historical pins."""
    pin_train_max = min(int(pin_train_max), cache.pin_size)
    pin_train_prob = float(pin_train_prob)
    if cache.pin_size <= 0 or pin_train_max <= 0 or pin_train_prob <= 0.0:
        return -1, None
    if T <= train_block:
        return -1, None
    if torch.rand((), device=k.device).item() >= pin_train_prob:
        return -1, None

    # Loop starts are 0, train_block, 2*train_block, ... < T. Injection must
    # happen at a nonzero start so the pins are drawn from already-streamed K/V.
    n_inject_points = (T - 1) // train_block
    if n_inject_points <= 0:
        return -1, None
    inject_start = int(torch.randint(1, n_inject_points + 1, (), device=k.device).item()) * train_block
    max_pin = min(pin_train_max, inject_start)
    if max_pin <= 0:
        return -1, None

    # Random dose in [0, max_pin] so the model still sees pin-free forwards even
    # when the injection Bernoulli fires.
    n_pin = int(torch.randint(0, max_pin + 1, (), device=k.device).item())
    if n_pin <= 0:
        return -1, None

    # Per batch/KV-group sampling without replacement. ``topk`` over random
    # scores is cheap at the target batch/group sizes and keeps every group free
    # to receive a different random pin set, matching eval-time per-group pins.
    rand = torch.rand(k.size(0), k.size(1), inject_start, device=k.device)
    positions = rand.topk(n_pin, dim=-1).indices.sort(dim=-1).values
    return inject_start, positions


def _install_training_pins(
    cache: LogStructuredKVCache,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor | None,
) -> None:
    """Install detached exact K/V pins selected from the full sequence tensors."""
    if positions is None or positions.numel() == 0:
        return
    positions = positions.to(device=k.device, dtype=torch.long)
    k_sel = torch.gather(k, 2, positions.unsqueeze(-1).expand(-1, -1, -1, k.size(-1))).detach()
    v_sel = torch.gather(v, 2, positions.unsqueeze(-1).expand(-1, -1, -1, v.size(-1))).detach()
    cache.set_pinned(k_sel, v_sel)


class LogKVStreamTrainingAttention(torch.autograd.Function):
    """Constant-in-T-times-S memory autograd for logKV streaming training.

    Problem: the naive training graph saves, for every 2-token chunk, the
    concatenated [detached prefix + chunk] keys/values (O(S) each) that its
    two matmuls need for backward. Per layer that is O(T/2 * S) saved tensors
    — hundreds of GB for a 32K stream at B=512/recent=1024 — the long-context
    training OOM (the failing allocation is merely whichever op runs next
    once the accumulated graph has eaten the device).

    Forward here streams the whole sequence with NO recorded graph and returns
    the attention output. Backward resets the cache and REPLAYS the same stream:
    each block's attention is recomputed with a throwaway local graph that
    ``torch.autograd.grad`` consumes immediately, so at most one block graph is
    alive at any time. Peak memory: O(T + train_block*S) instead of O(T*S).
    Cost: one extra streaming pass plus the per-block backwards.

    Correctness:
      - For a fixed train_block, gradients are EXACT for that block-streaming
        objective: in the naive graph, gradient reaches q/k/v only through each
        token's own block (the cache commits detached copies), so per-block
        grads are complete and disjoint, and the replayed per-block
        ``autograd.grad`` reproduces them one-to-one.
      - The replay is deterministic: compaction is pure mean-pooling with
        count-based binary carries — no RNG, identical shapes take identical
        kernels — so the rebuilt per-block prefix states equal forward's.
      - Backward does not depend on any cache state forward left behind: it
        resets the buffers AND re-derives ``cache.second_order`` from its own
        ``second_order_scale`` before replaying, so neither interleaved
        forward/backward across micro-batches, nor activation-checkpoint
        recompute ordering, nor another layer running at a different gate can
        make the replay rebuild a cache that forward never attended over.

    ``train_block=2`` is the strict 2-token streaming reference. Larger blocks
    freeze the prefix state at block start and use causal exact attention within
    the block, matching the inference prefill speed/memory tradeoff while
    preserving the exact final cache state. First-order autograd only
    (``once_differentiable``) — CPT never needs grad-of-grad.
    """

    @staticmethod
    def forward(ctx, q, k, v, cache, scale, train_block, second_order_scale, *pin_args):
        if len(pin_args) > 2:
            raise TypeError(
                "LogKVStreamTrainingAttention accepts at most two pin args: "
                "pin_train_max and pin_train_prob"
            )
        ctx._num_inputs = 7 + len(pin_args)
        pin_train_max = int(pin_args[0]) if len(pin_args) >= 1 else 0
        pin_train_prob = float(pin_args[1]) if len(pin_args) >= 2 else 0.0

        T = q.size(2)
        train_block = int(train_block)
        if train_block < 2:
            raise ValueError(f"logKV train_block must be >= 2, got {train_block}")
        if train_block > cache.recent_size:
            raise ValueError(
                f"logKV train_block ({train_block}) must be <= recent_size ({cache.recent_size})"
            )
        outputs: list[torch.Tensor] = []
        # Explicit no_grad: the memory guarantee of this whole scheme rests on
        # this pass recording nothing (Function.forward already runs detached;
        # this makes the invariant local and future-proof).
        with torch.no_grad():
            cache.reset_parameters()
            # Own the flag rather than trusting the caller, and set it AFTER
            # the reset above so the cache is always empty when this runs (the
            # guarded setter would otherwise raise on a cache left non-empty by
            # a previous call at a different gate). It decides whether
            # add_recent() builds the second-order statistics, so leaving it to
            # ambient state breaks this Function two ways: a caller that never
            # sets it runs the attention math against statistics that were
            # never built (silently first-order), and anything that flips it
            # between forward and backward makes the replay rebuild a
            # different cache than the one forward attended over, so the
            # gradients belong to a different function than the output.
            # Deriving it here from the gate keeps the pair consistent no
            # matter who calls.
            cache.second_order = second_order_scale != 0.0
            pin_inject_start, pin_positions = _sample_training_pin_positions(
                k, cache, T, train_block, pin_train_max, pin_train_prob
            )
            start = 0
            while start < T:  # mirrored in backward() — keep in sync
                if start == pin_inject_start:
                    _install_training_pins(cache, k, v, pin_positions)
                end = min(start + train_block, T)
                outputs.append(
                    log_kv_chunk_attention(
                        cache,
                        q[:, :, start:end], k[:, :, start:end], v[:, :, start:end],
                        scale,
                        second_order_scale,
                    )
                )
                cache.add_recent(k[:, :, start:end], v[:, :, start:end])
                start = end
        ctx.save_for_backward(q, k, v)
        ctx.cache = cache
        ctx.scale = scale
        ctx.train_block = train_block
        ctx.second_order_scale = second_order_scale
        ctx.pin_inject_start = pin_inject_start
        ctx.pin_positions = pin_positions
        return torch.cat(outputs, dim=2)  # (B, nh, T, v_dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        q, k, v = ctx.saved_tensors
        cache = ctx.cache
        scale = ctx.scale
        train_block = ctx.train_block
        second_order_scale = ctx.second_order_scale
        pin_inject_start = ctx.pin_inject_start
        pin_positions = ctx.pin_positions
        T = q.size(2)
        # Blocks partition [0, T) and each position's grad comes from exactly
        # its own block, so the empty buffers are fully overwritten.
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        cache.reset_parameters()
        # Re-derive AFTER the reset, same reasoning and ordering as forward():
        # the replay must rebuild the cache exactly as forward built it, the
        # flag is mutable state anyone could have changed in between, and the
        # guarded setter needs the cache empty to accept either value.
        cache.second_order = second_order_scale != 0.0
        start = 0
        while start < T:  # mirrors forward() — keep in sync
            if start == pin_inject_start:
                with torch.no_grad():
                    _install_training_pins(cache, k, v, pin_positions)
            end = min(start + train_block, T)
            q_b = q[:, :, start:end].detach().requires_grad_(True)
            k_b = k[:, :, start:end].detach().requires_grad_(True)
            v_b = v[:, :, start:end].detach().requires_grad_(True)
            with torch.enable_grad():
                y_b = log_kv_chunk_attention(cache, q_b, k_b, v_b, scale, second_order_scale)
            g_q, g_k, g_v = torch.autograd.grad(y_b, (q_b, k_b, v_b), grad_y[:, :, start:end])
            dq[:, :, start:end] = g_q
            dk[:, :, start:end] = g_k
            dv[:, :, start:end] = g_v
            with torch.no_grad():
                cache.add_recent(k[:, :, start:end], v[:, :, start:end])
            start = end
        grad_inputs = (dq, dk, dv, None, None, None, None)
        return grad_inputs + (None,) * (ctx._num_inputs - len(grad_inputs))
