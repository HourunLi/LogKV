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

import math
from typing import NoReturn

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from litgpt.log_kv_pin_score_diag import DIAG as LOG_KV_PIN_SCORE_DIAG


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact rank-1 covariance / cross-covariance stats for a 2-token slot."""
    dk = ka - kb
    dv = va - vb
    sigma_u, dk_norm = _normalize(dk)
    gamma_b, dv_norm = _normalize(dv)
    sigma2 = dk_norm.square() * 0.25
    gamma = dk_norm * dv_norm * 0.25
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
        self.recent_size = recent_size if recent_size > 0 else 2
        # Explicit raise (not assert): must survive `python -O`.
        if self.recent_size < 2:
            raise ValueError(f"recent_size ({self.recent_size}) must be >= 2")

        denom = B * 2
        # +1 for the write level (level 0). The formula gives the number of carry
        # levels needed; total levels = carry + 1 write level.
        self.max_levels = max(2, math.ceil(math.log2(max((max_seq_length + 1) / denom, 1))) + 1)

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

        # ---- Hierarchical levels: B slots each (k, v, w) ----
        # Level 0: partially fillable (0 to B entries), written one entry at a time.
        # Level 1+: all-or-nothing (0 or B entries), written via binary carry.
        for ell in range(self.max_levels):
            self.register_buffer(
                f"level_k_{ell}",
                torch.zeros(batch_size, n_groups, B, k_dim, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"level_v_{ell}",
                torch.zeros(batch_size, n_groups, B, v_dim, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"level_w_{ell}",
                torch.zeros(batch_size, n_groups, B, device=device, dtype=dtype),
                persistent=False,
            )
            # Rank-1 key covariance:
            #   Sigma_s ~= sigma2_s * sigma_u_s sigma_u_s^T
            self.register_buffer(
                f"level_sigma_u_{ell}",
                torch.zeros(batch_size, n_groups, B, k_dim, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"level_sigma2_{ell}",
                torch.zeros(batch_size, n_groups, B, device=device, dtype=dtype),
                persistent=False,
            )
            # Rank-1 value-key cross covariance:
            #   Gamma_s ~= gamma_s * gamma_b_s gamma_a_s^T
            # where gamma_a lives in key/query space and gamma_b in value space.
            self.register_buffer(
                f"level_gamma_a_{ell}",
                torch.zeros(batch_size, n_groups, B, k_dim, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"level_gamma_b_{ell}",
                torch.zeros(batch_size, n_groups, B, v_dim, device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"level_gamma_{ell}",
                torch.zeros(batch_size, n_groups, B, device=device, dtype=dtype),
                persistent=False,
            )
        self.register_buffer(
            "level_count",
            torch.zeros(self.max_levels, dtype=torch.long, device=device),
            persistent=False,
        )
        # Host-side mirror of ``level_count``. Every control-flow decision in the
        # streaming path (how many slots a level holds, whether a carry fires)
        # reads a level count, and reading the device tensor means a
        # device-to-host sync per read — O(max_levels) stalls per streaming
        # chunk, per layer, on both the forward stream and the backward replay.
        # The mirror keeps those decisions on the host; the buffer stays
        # authoritative for state_dict/tests and is written (never read) in step.
        self._counts: list[int] = [0] * self.max_levels

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
        if value != self._second_order and any(c > 0 for c in self._counts):
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
            getattr(self, f"level_k_{ell}"),
            getattr(self, f"level_v_{ell}"),
            getattr(self, f"level_w_{ell}"),
        )

    def _get_level_stats(
        self, ell: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            getattr(self, f"level_sigma_u_{ell}"),
            getattr(self, f"level_sigma2_{ell}"),
            getattr(self, f"level_gamma_a_{ell}"),
            getattr(self, f"level_gamma_b_{ell}"),
            getattr(self, f"level_gamma_{ell}"),
        )

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
    ) -> None:
        getattr(self, f"level_k_{ell}").copy_(k)
        getattr(self, f"level_v_{ell}").copy_(v)
        getattr(self, f"level_w_{ell}").copy_(w)
        if sigma_u is None:
            getattr(self, f"level_sigma_u_{ell}").zero_()
            getattr(self, f"level_sigma2_{ell}").zero_()
            getattr(self, f"level_gamma_a_{ell}").zero_()
            getattr(self, f"level_gamma_b_{ell}").zero_()
            getattr(self, f"level_gamma_{ell}").zero_()
        else:
            getattr(self, f"level_sigma_u_{ell}").copy_(sigma_u)
            getattr(self, f"level_sigma2_{ell}").copy_(sigma2)
            getattr(self, f"level_gamma_a_{ell}").copy_(gamma_a)
            getattr(self, f"level_gamma_b_{ell}").copy_(gamma_b)
            getattr(self, f"level_gamma_{ell}").copy_(gamma)
        self.level_count[ell] = self.B
        self._counts[ell] = self.B

    def _clear_level(self, ell: int) -> None:
        getattr(self, f"level_k_{ell}").zero_()
        getattr(self, f"level_v_{ell}").zero_()
        getattr(self, f"level_w_{ell}").zero_()
        getattr(self, f"level_sigma_u_{ell}").zero_()
        getattr(self, f"level_sigma2_{ell}").zero_()
        getattr(self, f"level_gamma_a_{ell}").zero_()
        getattr(self, f"level_gamma_b_{ell}").zero_()
        getattr(self, f"level_gamma_{ell}").zero_()
        self.level_count[ell] = 0
        self._counts[ell] = 0

    # ------------------------------------------------------------------
    # Compact: compress tokens -> 1 entry via mean pooling (2:1 by default)
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_tokens(
        k: torch.Tensor,  # (B, G, n, k_dim) full post-RoPE keys
        v: torch.Tensor,  # (B, G, n, v_dim)
        with_stats: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Compress n tokens into a single compact entry via mean pooling.

        Mean-pooling the full key merges content and position in one step:
        the position sub-channel becomes the expected rotation over the span
        (see module docstring). No renormalization — the per-frequency norm
        shrinkage encodes the span's positional uncertainty.
        Returns k_entry (B,G,1,k_dim), v_entry (B,G,1,v_dim), w_entry (B,G,1).
        With ``with_stats=True`` also returns rank-1 approximations to the
        within-slot key covariance and value-key cross covariance.
        """
        n = k.size(2)
        k_entry = k.mean(dim=2, keepdim=True)
        v_entry = v.mean(dim=2, keepdim=True)
        w_entry = torch.full(
            (k.size(0), k.size(1), 1),
            float(n),
            device=k.device,
            dtype=k.dtype,
        )
        if not with_stats:
            return k_entry, v_entry, w_entry

        inv_sqrt_n = float(n) ** -0.5
        k_centered = (k - k_entry).float() * inv_sqrt_n
        v_centered = (v - v_entry).float() * inv_sqrt_n
        sigma_u, sigma2 = _rank1_psd_from_factors(k_centered)
        gamma_b, gamma_a, gamma = _rank1_cross_from_factors(v_centered, k_centered)
        return (
            k_entry,
            v_entry,
            w_entry,
            sigma_u.unsqueeze(2).to(k.dtype),
            sigma2.unsqueeze(2).to(k.dtype),
            gamma_a.unsqueeze(2).to(k.dtype),
            gamma_b.unsqueeze(2).to(k.dtype),
            gamma.unsqueeze(2).to(k.dtype),
        )

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

        w_total = wa + wb  # (B, G, B)
        alpha = (wa / w_total.clamp(min=1e-8)).unsqueeze(-1)  # (B, G, B, 1)

        k_out = alpha * ka + (1 - alpha) * kb
        v_out = alpha * va + (1 - alpha) * vb
        if sigma_u1 is None:
            return k_out, v_out, w_total

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
        return (
            k_out,
            v_out,
            w_total,
            sigma_u.to(k_out.dtype),
            sigma2.to(k_out.dtype),
            gamma_a.to(k_out.dtype),
            gamma_b.to(v_out.dtype),
            gamma.to(k_out.dtype),
        )

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
    ) -> None:
        """Add one compact entry to level 0. If level 0 is full, binary carry to levels 1+."""
        idx = self._counts[0]
        getattr(self, "level_k_0")[:, :, idx, :] = k_entry
        getattr(self, "level_v_0")[:, :, idx, :] = v_entry
        getattr(self, "level_w_0")[:, :, idx] = w_entry
        if sigma_u_entry is None:
            # Zero the slot in place rather than materializing zero tensors to
            # copy from: this runs per compacted entry.
            getattr(self, "level_sigma_u_0")[:, :, idx, :].zero_()
            getattr(self, "level_sigma2_0")[:, :, idx].zero_()
            getattr(self, "level_gamma_a_0")[:, :, idx, :].zero_()
            getattr(self, "level_gamma_b_0")[:, :, idx, :].zero_()
            getattr(self, "level_gamma_0")[:, :, idx].zero_()
        else:
            getattr(self, "level_sigma_u_0")[:, :, idx, :] = sigma_u_entry
            getattr(self, "level_sigma2_0")[:, :, idx] = sigma2_entry
            getattr(self, "level_gamma_a_0")[:, :, idx, :] = gamma_a_entry
            getattr(self, "level_gamma_b_0")[:, :, idx, :] = gamma_b_entry
            getattr(self, "level_gamma_0")[:, :, idx] = gamma_entry
        self.level_count[0] = idx + 1
        self._counts[0] = idx + 1

        if self._counts[0] >= self.B:
            lk, lv, lw = self._get_level(0)
            if self.second_order:
                lsu, ls2, lga, lgb, lgm = self._get_level_stats(0)
                self._binary_carry(
                    lk.clone(), lv.clone(), lw.clone(),
                    lsu.clone(), ls2.clone(), lga.clone(), lgb.clone(), lgm.clone(),
                )
            else:
                self._binary_carry(lk.clone(), lv.clone(), lw.clone())
            self._clear_level(0)

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
    ) -> None:
        new_k, new_v, new_w = block_k, block_v, block_w
        new_su, new_s2 = block_sigma_u, block_sigma2
        new_ga, new_gb, new_gm = block_gamma_a, block_gamma_b, block_gamma
        for ell in range(1, self.max_levels):
            if self._counts[ell] == 0:
                self._set_level(ell, new_k, new_v, new_w, new_su, new_s2, new_ga, new_gb, new_gm)
                return
            ek, ev, ew = self._get_level(ell)
            if new_su is None:
                new_k, new_v, new_w = self.compact(ek, ev, ew, new_k, new_v, new_w)
            else:
                esu, es2, ega, egb, egm = self._get_level_stats(ell)
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
            self._clear_level(ell)
        raise RuntimeError(
            f"LogStructuredKVCache: binary carry overflow! "
            f"All {self.max_levels} levels occupied. "
            f"max_seq_length={self.max_seq_length}, B={self.B}. "
            f"This is a bug — max_levels formula needs review."
        )

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
        pk = rk_pairs.mean(dim=3)
        pv = rv_pairs.mean(dim=3)
        pw = torch.full((B_, G_, f), 2.0, device=rk.device, dtype=rk.dtype)
        if self.second_order:
            psu, ps2, pga, pgb, pgm = _pair_rank1_stats(
                rk_pairs[:, :, :, 0, :],
                rk_pairs[:, :, :, 1, :],
                rv_pairs[:, :, :, 0, :],
                rv_pairs[:, :, :, 1, :],
            )
            self._append_level0(pk, pv, pw, psu, ps2, pga, pgb, pgm)
        else:
            self._append_level0(pk, pv, pw)

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
    ) -> None:
        """Append f compact entries to level 0 in order, carrying when it fills.

        Trajectory-identical to f sequential ``_add_compact_entry`` calls: the
        binary carry fires exactly when the count reaches B, between the same
        two entries as in the sequential version.
        """
        f = pk.size(2)
        off = 0
        while off < f:
            idx = self._counts[0]
            take = min(self.B - idx, f - off)
            getattr(self, "level_k_0")[:, :, idx:idx + take, :] = pk[:, :, off:off + take, :]
            getattr(self, "level_v_0")[:, :, idx:idx + take, :] = pv[:, :, off:off + take, :]
            getattr(self, "level_w_0")[:, :, idx:idx + take] = pw[:, :, off:off + take]
            if psu is None:
                # Zero in place instead of materializing zero tensors to copy
                # from: this runs on every window flush.
                getattr(self, "level_sigma_u_0")[:, :, idx:idx + take, :].zero_()
                getattr(self, "level_sigma2_0")[:, :, idx:idx + take].zero_()
                getattr(self, "level_gamma_a_0")[:, :, idx:idx + take, :].zero_()
                getattr(self, "level_gamma_b_0")[:, :, idx:idx + take, :].zero_()
                getattr(self, "level_gamma_0")[:, :, idx:idx + take].zero_()
            else:
                getattr(self, "level_sigma_u_0")[:, :, idx:idx + take, :] = psu[:, :, off:off + take, :]
                getattr(self, "level_sigma2_0")[:, :, idx:idx + take] = ps2[:, :, off:off + take]
                getattr(self, "level_gamma_a_0")[:, :, idx:idx + take, :] = pga[:, :, off:off + take, :]
                getattr(self, "level_gamma_b_0")[:, :, idx:idx + take, :] = pgb[:, :, off:off + take, :]
                getattr(self, "level_gamma_0")[:, :, idx:idx + take] = pgm[:, :, off:off + take]
            self.level_count[0] = idx + take
            self._counts[0] = idx + take
            off += take
            if self._counts[0] >= self.B:
                lk, lv, lw = self._get_level(0)
                if self.second_order:
                    lsu, ls2, lga, lgb, lgm = self._get_level_stats(0)
                    self._binary_carry(
                        lk.clone(), lv.clone(), lw.clone(),
                        lsu.clone(), ls2.clone(), lga.clone(), lgb.clone(), lgm.clone(),
                    )
                else:
                    self._binary_carry(lk.clone(), lv.clone(), lw.clone())
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
        """
        self._count_tokens(k.size(2))

        (
            k_entry,
            v_entry,
            w_entry,
            sigma_u_entry,
            sigma2_entry,
            gamma_a_entry,
            gamma_b_entry,
            gamma_entry,
        ) = self._compact_tokens(k, v, with_stats=True)
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

    def get_attention_state(self, with_stats: bool = False) -> tuple[torch.Tensor, ...]:
        """Assemble the cache state for ``log_kv_slot_attention``.

        Returns:
            slot_k: (B, G, n_slots, k_dim) — time-ordered slot keys: compact
                    levels oldest (highest level) first down to level 0, then
                    the recent-window tokens as exact w=1 slots.
            slot_v: (B, G, n_slots, v_dim)
            slot_w: (B, G, n_slots) — token count per slot (1 for recent).
            If ``with_stats=True``, also returns:
            slot_sigma_u / slot_sigma2: rank-1 key covariance stats;
            slot_gamma_a / slot_gamma_b / slot_gamma: rank-1 value-key cross
                covariance stats. Exact recent and pinned tokens have zero stats.

        Slots cover contiguous, time-ordered spans, so ``cumsum(slot_w)`` gives
        the token boundaries of every slot (used by exactness tests). With
        salience pins active (pin_count > 0) that contiguity invariant no
        longer holds: pins are scattered duplicates of tokens the hierarchy
        also covers. Attention itself is order-agnostic over the state (the
        whole state is fully visible; only the appended in-flight chunk is
        causal), so ordering matters only to those boundary-based tests, which
        run with pins disabled.
        """
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        w_parts: list[torch.Tensor] = []
        sigma_u_parts: list[torch.Tensor] = []
        sigma2_parts: list[torch.Tensor] = []
        gamma_a_parts: list[torch.Tensor] = []
        gamma_b_parts: list[torch.Tensor] = []
        gamma_parts: list[torch.Tensor] = []

        # Compact levels: oldest (highest level) first, down to level 0. Counts
        # come from the host mirror — reading the device tensor here would sync
        # once per level, per streaming chunk, per layer.
        for ell in range(self.max_levels - 1, -1, -1):
            count = self._counts[ell]
            if count > 0:
                lk, lv, lw = self._get_level(ell)
                k_parts.append(lk[:, :, :count, :])
                v_parts.append(lv[:, :, :count, :])
                w_parts.append(lw[:, :, :count])
                if with_stats:
                    lsu, ls2, lga, lgb, lgm = self._get_level_stats(ell)
                    sigma_u_parts.append(lsu[:, :, :count, :])
                    sigma2_parts.append(ls2[:, :, :count])
                    gamma_a_parts.append(lga[:, :, :count, :])
                    gamma_b_parts.append(lgb[:, :, :count, :])
                    gamma_parts.append(lgm[:, :, :count])

        # Salience pins: exact w=1 duplicates from the compressed region,
        # placed between the levels and the recent window (they are older than
        # everything in recent by construction).
        if self.pin_count > 0:
            k_parts.append(self.pin_k[:, :, :self.pin_count, :])
            v_parts.append(self.pin_v[:, :, :self.pin_count, :])
            w_parts.append(
                self.pin_k.new_ones(self.batch_size, self.n_groups, self.pin_count)
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
                self.recent_k.new_ones(self.batch_size, self.n_groups, self.recent_count)
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

        if w_parts:
            slot_k = torch.cat(k_parts, dim=-2)
            slot_v = torch.cat(v_parts, dim=-2)
            slot_w = torch.cat(w_parts, dim=-1)
            if not with_stats:
                return slot_k, slot_v, slot_w
            return (
                slot_k,
                slot_v,
                slot_w,
                torch.cat(sigma_u_parts, dim=-2),
                torch.cat(sigma2_parts, dim=-1),
                torch.cat(gamma_a_parts, dim=-2),
                torch.cat(gamma_b_parts, dim=-2),
                torch.cat(gamma_parts, dim=-1),
            )

        slot_k = self.recent_k[:, :, :0, :]
        slot_v = self.recent_v[:, :, :0, :]
        slot_w = getattr(self, "level_w_0")[:, :, :0]
        if not with_stats:
            return slot_k, slot_v, slot_w
        return (
            slot_k,
            slot_v,
            slot_w,
            self.recent_k[:, :, :0, :],
            slot_w,
            self.recent_k[:, :, :0, :],
            self.recent_v[:, :, :0, :],
            slot_w,
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
        ``level_count`` stays ``long`` and is deliberately untouched. Compaction
        weights are always powers of two (uniform 2:1), hence exact in bf16/fp16,
        so converting ``level_w`` does not corrupt token counts.
        """
        if self.recent_k.dtype == dtype:
            return
        self.recent_k = self.recent_k.to(dtype)
        self.recent_v = self.recent_v.to(dtype)
        self.pin_k = self.pin_k.to(dtype)
        self.pin_v = self.pin_v.to(dtype)
        for ell in range(self.max_levels):
            setattr(self, f"level_k_{ell}", getattr(self, f"level_k_{ell}").to(dtype))
            setattr(self, f"level_v_{ell}", getattr(self, f"level_v_{ell}").to(dtype))
            setattr(self, f"level_w_{ell}", getattr(self, f"level_w_{ell}").to(dtype))
            setattr(self, f"level_sigma_u_{ell}", getattr(self, f"level_sigma_u_{ell}").to(dtype))
            setattr(self, f"level_sigma2_{ell}", getattr(self, f"level_sigma2_{ell}").to(dtype))
            setattr(self, f"level_gamma_a_{ell}", getattr(self, f"level_gamma_a_{ell}").to(dtype))
            setattr(self, f"level_gamma_b_{ell}", getattr(self, f"level_gamma_b_{ell}").to(dtype))
            setattr(self, f"level_gamma_{ell}", getattr(self, f"level_gamma_{ell}").to(dtype))

    def reset_parameters(self) -> None:
        """Reset all buffers to zero."""
        self.token_count = 0
        self.recent_k.zero_()
        self.recent_v.zero_()
        self.recent_count = 0
        self.pin_k.zero_()
        self.pin_v.zero_()
        self.pin_count = 0
        for ell in range(self.max_levels):
            self._clear_level(ell)

    @property
    def total_slots(self) -> int:
        return self.recent_count + self.pin_count + sum(self._counts)

    @property
    def total_tokens_covered(self) -> int:
        return self.token_count


# ======================================================================
# Slot attention: merged-position slots + log-multiplicity mass bias
# ======================================================================


def append_exact_tokens(
    slot_k: torch.Tensor,   # (B, G, S, k_dim)
    slot_v: torch.Tensor,   # (B, G, S, v_dim)
    slot_w: torch.Tensor,   # (B, G, S)
    k_new: torch.Tensor,    # (B, G, n, k_dim) exact tokens, appended in time order
    v_new: torch.Tensor,    # (B, G, n, v_dim)
    slot_sigma_u: torch.Tensor | None = None,
    slot_sigma2: torch.Tensor | None = None,
    slot_gamma_a: torch.Tensor | None = None,
    slot_gamma_b: torch.Tensor | None = None,
    slot_gamma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Append exact per-token entries (w=1 slots) after the cached slots.

    Used by the attention layer to make the current chunk (and any pending
    token) visible to attention BEFORE it is committed to the cache. Gradient
    flows through ``k_new``/``v_new``; the ones-weights are constants. If slot
    rank-1 stats are provided, the appended exact tokens receive zero
    covariance / cross-covariance stats.
    """
    n_new = k_new.size(2)
    ones = slot_w.new_ones(slot_w.size(0), slot_w.size(1), n_new)
    k_all = torch.cat([slot_k, k_new], dim=2)
    v_all = torch.cat([slot_v, v_new], dim=2)
    w_all = torch.cat([slot_w, ones], dim=-1)
    if slot_sigma_u is None:
        return k_all, v_all, w_all

    if any(x is None for x in (slot_sigma2, slot_gamma_a, slot_gamma_b, slot_gamma)):
        raise ValueError("append_exact_tokens() requires either all rank-1 stats or none")
    return (
        k_all,
        v_all,
        w_all,
        torch.cat([slot_sigma_u, torch.zeros_like(k_new)], dim=2),
        torch.cat([slot_sigma2, torch.zeros_like(ones)], dim=-1),
        torch.cat([slot_gamma_a, torch.zeros_like(k_new)], dim=2),
        torch.cat([slot_gamma_b, torch.zeros_like(v_new)], dim=2),
        torch.cat([slot_gamma, torch.zeros_like(ones)], dim=-1),
    )


def log_kv_slot_attention(
    q: torch.Tensor,        # (B, nh, T_q, k_dim) full post-RoPE queries
    slot_k: torch.Tensor,   # (B, G, S, k_dim) merged slot keys (position included)
    slot_v: torch.Tensor,   # (B, G, S, v_dim)
    slot_w: torch.Tensor,   # (B, G, S) token count per slot (>= 1)
    scale: float,
    mask: torch.Tensor | None = None,  # (T_q, S) bool, True = attend
    lam: float = 1.0,
    causal_tail: int = 0,
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
        chunk-causal masks always allow the diagonal.

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
        if lam != 0.0:
            # slot_w >= 1 by construction; guarded via lam gate so lam=0 can
            # never produce 0 * log(0) = NaN even on malformed input.
            scores.add_(lam * slot_w.to(torch.float32).log()[:, :, None, None, :])
        if mask is not None:
            scores.masked_fill_(~mask.view(1, 1, 1, T_q, S), float("-inf"))
        elif causal_tail:
            scores[..., S - causal_tail:].masked_fill_(tail_blocked, float("-inf"))
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
    if lam != 0.0:
        # slot_w >= 1 by construction; guarded via lam gate so lam=0 can never
        # produce 0 * log(0) = NaN even on malformed input.
        scores.add_(lam * slot_w.to(torch.float32).log().unsqueeze(-2))
    if mask is not None:
        scores.masked_fill_(~mask.view(1, 1, T_q, S), float("-inf"))
    elif causal_tail:
        scores[..., S - causal_tail:].masked_fill_(tail_blocked, float("-inf"))
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
        slot_k, slot_v, slot_w = cache.get_attention_state(with_stats=False)
        k_all, v_all, w_all = append_exact_tokens(slot_k, slot_v, slot_w, k_b, v_b)
        return log_kv_slot_attention(
            q_b, k_all, v_all, w_all,
            scale=scale,
            causal_tail=q_b.size(2),
        )
    (
        slot_k,
        slot_v,
        slot_w,
        slot_sigma_u,
        slot_sigma2,
        slot_gamma_a,
        slot_gamma_b,
        slot_gamma,
    ) = cache.get_attention_state(with_stats=True)
    (
        k_all,
        v_all,
        w_all,
        sigma_u_all,
        sigma2_all,
        gamma_a_all,
        gamma_b_all,
        gamma_all,
    ) = append_exact_tokens(
        slot_k, slot_v, slot_w, k_b, v_b,
        slot_sigma_u, slot_sigma2, slot_gamma_a, slot_gamma_b, slot_gamma,
    )
    return log_kv_slot_attention(
        q_b, k_all, v_all, w_all,
        scale=scale,
        causal_tail=q_b.size(2),
        slot_sigma_u=sigma_u_all,
        slot_sigma2=sigma2_all,
        slot_gamma_a=gamma_a_all,
        slot_gamma_b=gamma_b_all,
        slot_gamma=gamma_all,
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
