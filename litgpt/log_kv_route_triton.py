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
def _nearest(DATA, MASS, RADIUS, LIMITS, NORM, MATCHED, BEST, COST, FULL_BEST, FULL_COST,
             M, CAP, HAVE_LIMITS: tl.constexpr, HAVE_CAP: tl.constexpr,
             HAVE_NORM: tl.constexpr, HAVE_MATCHED: tl.constexpr,
             SQRT_RN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    offset = lane * M
    mi = tl.load(MASS + offset + row)
    mj = tl.load(MASS + offset + cols, cols < M, 0)
    if HAVE_MATCHED:
        mi = tl.where(tl.load(MATCHED + offset + row), 0.0, mi)
        mj = tl.where(tl.load(MATCHED + offset + cols, cols < M, 1), 0.0, mj)
    total = mi + mj
    denom = tl.maximum(total, 1.0)
    valid = (cols < M) & (cols != row) & (mi > 0) & (mj > 0)
    # Later sweeps only need the unmatched submatrix; masked loads avoid
    # rereading rows/columns already consumed by earlier independent pairs.
    distance2 = tl.load(DATA + (offset + row) * M + cols, valid, 0)
    if HAVE_NORM:
        ni = tl.load(NORM + offset + row)
        nj = tl.load(NORM + offset + cols, cols < M, 0)
        distance2 = tl.maximum(ni + nj - 2.0 * distance2, 0.0)
    cost = distance2 * tl.div_rn(mi * mj, denom)
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
            norm if norm is not None else mass, mass, best, cost,
            full[0] if have_cap else best, full[1] if have_cap else cost,
            size, cap, limits is not None, have_cap, norm is not None, False,
            hasattr(tl, "sqrt_rn"), triton.next_power_of_2(size),
            num_warps=4, enable_fp_fusion=False,
        )
    return (best, cost), full


@triton.jit(do_not_specialize=["M"])
def _accept_mutual(BEST, COST, MATCHED, PARTNER, PAIR_COST, M, BLOCK: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK)
    at = lane * M + rows
    valid = rows < M
    peer = tl.load(BEST + at, valid, 0)
    back = tl.load(BEST + lane * M + peer, valid, 0)
    cost = tl.load(COST + at, valid, float("inf"))
    used = tl.load(MATCHED + at, valid, 1)
    accept = valid & ~used & (rows != peer) & (back == rows) & (cost < float("inf"))
    tl.store(MATCHED + at, used | accept, valid)
    tl.store(PARTNER + at, peer, accept)
    tl.store(PAIR_COST + at, cost, accept)


def matching_sweeps(data, mass, norm, passes):
    """Several disjoint MNN sweeps with frozen centers; no host reads between sweeps.

    Reuse Gram products and scratch buffers. Successive kernels on the same
    stream ensure no row sees a partially updated matching mask.
    """
    groups, size = mass.shape
    matched = torch.zeros_like(mass, dtype=torch.bool)
    best = torch.empty_like(mass, dtype=torch.int64)
    cost = torch.empty_like(mass)
    partner = torch.arange(size, device=mass.device).expand(groups, -1).contiguous()
    pair_cost = torch.full_like(mass, float("inf"))
    block = triton.next_power_of_2(size)
    for _ in range(passes):
        _nearest[(size, groups)](
            data, mass, mass, mass, norm if norm is not None else mass,
            matched, best, cost, best, cost, size, math.inf,
            False, False, norm is not None, True, hasattr(tl, "sqrt_rn"), block,
            num_warps=4, enable_fp_fusion=False,
        )
        _accept_mutual[(groups,)](best, cost, matched, partner, pair_cost, size, block, num_warps=4)
    return partner, pair_cost


@triton.jit(do_not_specialize=["ROWS"])
def _merge_pack(MU, MASS, INDICES, OUT_MU, OUT_MASS, OUT_RADIUS, ROWS,
                D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    dims = tl.arange(0, BD)
    live = row < ROWS
    src = tl.load(INDICES + row, live, -1).to(tl.int64)
    present = live & (src >= 0)
    partner = tl.load(INDICES + ROWS + row, present, -1).to(tl.int64)
    merge = present & (partner >= 0)
    ma = tl.load(MASS + src, present, 0)
    mb = tl.load(MASS + partner, merge, 0)
    a = tl.load(MU + src[:, None] * D + dims[None, :],
                present[:, None] & (dims < D)[None, :], 0)
    b = tl.load(MU + partner[:, None] * D + dims[None, :],
                merge[:, None] & (dims < D)[None, :], 0)
    total = ma + mb
    merged = tl.div_rn(ma[:, None] * a + mb[:, None] * b,
                       tl.where(merge, total, 1.0)[:, None])
    # Copy untouched centers directly: recomputing ma*a/ma changes rounding.
    value = tl.where(merge[:, None], merged, a)
    tl.store(OUT_MU + row[:, None] * D + dims[None, :], value,
             live[:, None] & (dims < D)[None, :])
    tl.store(OUT_MASS + row, tl.where(merge, total, ma), live)
    tl.store(OUT_RADIUS + row, 0.0, live)


def merge_pack(mu, mass, indices):
    """Merge and pack a global routing round into independent FP32 buffers.

    indices is contiguous int64 [2, groups_out, clusters_out]. Its two planes
    contain flattened source/partner rows; -1 denotes padding/no partner.
    Radius is zero because only the global stage uses this kernel.
    """
    groups, size = indices.shape[1:]
    dims = mu.size(-1)
    out_mu = mu.new_empty((groups, size, dims))
    out_mass = mass.new_empty((groups, size))
    out_radius = mass.new_empty((groups, size))
    rows = groups * size
    if rows:
        _merge_pack[(triton.cdiv(rows, 8),)](
            mu, mass, indices, out_mu, out_mass, out_radius, rows,
            dims, triton.next_power_of_2(dims), 8,
            num_warps=4, enable_fp_fusion=False,
        )
    return out_mu, out_mass, out_radius
