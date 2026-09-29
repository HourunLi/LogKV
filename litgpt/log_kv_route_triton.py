"""Fused Ward row minima on CUDA, including compactness/capacity masks.

One program handles a row of one group. Only O(groups * clusters) results
leave the kernel; the caller applies the unchanged mutual-nearest pairing.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _minimum(cost, row, cols, M):
    best_cost = tl.min(cost, axis=0)
    # Exclude padded columns even when the entire row is inf. An all-inf row
    # selects itself, as does the PyTorch XOR argmin, and stays invalid by cost.
    tie = tl.where((cols < M) & (cost == best_cost), row ^ cols, 2 * M)
    best = tl.min(tie, axis=0) ^ row
    # Keep gathers in bounds even if non-finite input makes every tie invalid.
    best = tl.where(best_cost < float("inf"), best, row)
    return best, best_cost


@triton.jit(do_not_specialize=["M", "CAP"])
def _nearest(DATA, MASS, RADIUS, LIMITS, NORM, BEST, COST, FULL_BEST, FULL_COST,
             M, CAP, HAVE_LIMITS: tl.constexpr, HAVE_CAP: tl.constexpr,
             HAVE_NORM: tl.constexpr, SQRT_RN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    offset = lane * M
    mi = tl.load(MASS + offset + row)
    mj = tl.load(MASS + offset + cols, cols < M, 0)
    total = mi + mj
    denom = tl.maximum(total, 1.0)
    distance2 = tl.load(DATA + (offset + row) * M + cols, cols < M, 0)
    if HAVE_NORM:
        ni = tl.load(NORM + offset + row)
        nj = tl.load(NORM + offset + cols, cols < M, 0)
        distance2 = tl.maximum(ni + nj - 2.0 * distance2, 0.0)
    cost = distance2 * tl.div_rn(mi * mj, denom)
    valid = (cols < M) & (cols != row) & (mi > 0) & (mj > 0)
    if HAVE_LIMITS:
        ri = tl.load(RADIUS + offset + row)
        rj = tl.load(RADIUS + offset + cols, cols < M, 0)
        limit = tl.load(LIMITS + lane)
        if SQRT_RN:
            distance = tl.sqrt_rn(distance2)
        else:
            distance = tl.sqrt(distance2)
        bound = tl.maximum(ri + tl.div_rn(mj, denom) * distance,
                           rj + tl.div_rn(mi, denom) * distance)
        valid = valid & (bound <= limit)
    cost = tl.where(valid, cost, float("inf"))
    if HAVE_CAP:
        full_best, full_cost = _minimum(cost, row, cols, M)
        tl.store(FULL_BEST + offset + row, full_best.to(tl.int64))
        tl.store(FULL_COST + offset + row, full_cost)
        cost = tl.where(total <= CAP, cost, float("inf"))
    best, best_cost = _minimum(cost, row, cols, M)
    tl.store(BEST + offset + row, best.to(tl.int64))
    tl.store(COST + offset + row, best_cost)


def nearest(data, mass, radius, limits, cap, norm=None):
    """Return capped row minima and optional uncapped minima, without fallback.

    ``data`` is contiguous FP32 [groups, clusters, clusters]: squared distances
    when norm is absent, otherwise Gram products with FP32 norm [groups, clusters].
    Capacity fallback belongs to the caller and must apply to an entire group.
    """
    groups, size = mass.shape
    best = torch.empty_like(mass, dtype=torch.int64)
    cost = torch.empty_like(mass, dtype=torch.float32)
    have_cap = cap != math.inf
    full = (torch.empty_like(best), torch.empty_like(cost)) if have_cap else None
    if groups and size:
        _nearest[(size, groups)](
            data, mass, radius, limits if limits is not None else mass,
            norm if norm is not None else mass, best, cost,
            full[0] if have_cap else best, full[1] if have_cap else cost,
            size, cap, limits is not None, have_cap, norm is not None,
            hasattr(tl, "sqrt_rn"), triton.next_power_of_2(size),
            num_warps=4, enable_fp_fusion=False,
        )
    return (best, cost), full
