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


# ---------------------------------------------------------------------------
# Incremental rounds: persistent squared distances, Lance-Williams row updates.
#
# Row minima are packed as (cost bits << 32) | (row ^ col). Costs are >= 0, so
# their IEEE bits order like the floats and one int64 minimum reproduces the
# XOR tie rule above. Keys are symmetric, so a merged row's own entries double
# as pushes into the rows that may now prefer it.
# ---------------------------------------------------------------------------

@triton.jit
def _pack(cost, key):
    cost = tl.where(cost == 0, 0.0, cost)  # -0.0 would sort after every cost.
    return (cost.to(tl.int32, bitcast=True).to(tl.int64) << 32) | key.to(tl.int64)


@triton.jit
def _unpack_best(packed, row):
    return (packed - ((packed >> 32) << 32)) ^ row


@triton.jit
def _cost(d2, mi, mj, valid):
    denom = tl.maximum(mi + mj, 1.0)
    valid = valid & (mi > 0) & (mj > 0)
    return tl.where(valid, d2 * tl.div_rn(mi * mj, denom), float("inf")), valid


@triton.jit
def _cost_limited(d2, mi, mj, ri, rj, limit, valid, SQRT_RN: tl.constexpr):
    denom = tl.maximum(mi + mj, 1.0)
    if SQRT_RN:
        distance = tl.sqrt_rn(d2)
    else:
        distance = tl.sqrt(d2)
    bound = tl.maximum(ri + tl.div_rn(mj, denom) * distance, rj + tl.div_rn(mi, denom) * distance)
    valid = valid & (mi > 0) & (mj > 0) & (bound <= limit)
    return tl.where(valid, d2 * tl.div_rn(mi * mj, denom), float("inf")), valid


@triton.jit(do_not_specialize=["M", "ROUND"])
def _scan(DIST, MASS, RADIUS, LIMITS, ALIVE, KEEP, MERGED, REJECT, BEST, PACKED, M, ROUND,
          FULL: tl.constexpr, HAVE_LIMITS: tl.constexpr, SQRT_RN: tl.constexpr,
          ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """Full row minimum for rows whose previous partner merged, or all rows.

    Most rows are clean after a round, so each program checks ROWS rows and
    rescans only the dirty ones instead of launching one program per row.
    """
    lane = tl.program_id(1).to(tl.int64)
    base = lane * M
    for i in range(ROWS):
        row = tl.program_id(0) * ROWS + i
        dirty = row < M
        if dirty:
            dirty = tl.load(ALIVE + base + row) != 0
        if not FULL:
            if dirty:
                best = tl.minimum(tl.maximum(tl.load(BEST + base + row), 0), M - 1)
                dirty = (tl.load(KEEP + base + row) != ROUND) & (
                    (tl.load(MERGED + base + best) == ROUND) | (tl.load(REJECT + base + row) == ROUND))
        if dirty:
            mi = tl.load(MASS + base + row)
            result = tl.min(tl.full([BLOCK], 9223372036854775807, tl.int64), axis=0)
            for start in range(0, M, BLOCK):
                cols = start + tl.arange(0, BLOCK)
                inb = cols < M
                mj = tl.load(MASS + base + cols, inb, 0.0)
                d2 = tl.load(DIST + (base + row) * M + cols, inb, 0.0)
                if HAVE_LIMITS:
                    rj = tl.load(RADIUS + base + cols, inb, 0.0)
                    cost, _ = _cost_limited(d2, mi, mj, tl.load(RADIUS + base + row), rj,
                                            tl.load(LIMITS + lane), inb & (cols != row), SQRT_RN)
                else:
                    cost, _ = _cost(d2, mi, mj, inb & (cols != row))
                packed = tl.where(inb, _pack(cost, row ^ cols), 9223372036854775807)
                result = tl.minimum(result, tl.min(packed, axis=0))
            tl.store(PACKED + base + row, result)


@triton.jit(do_not_specialize=["M", "ROUND"])
def _select(PACKED, ALIVE, MASS, RADIUS, LIMITS, MU, DIST, BEST, PAIR_COST, BOUND, REJECT, NPROP, M, ROUND,
            DIM: tl.constexpr, HAVE_LIMITS: tl.constexpr, SQRT_RN: tl.constexpr,
            BS: tl.constexpr, BD: tl.constexpr):
    """Mutual pairs from packed minima; exact compactness check before budgeting."""
    lane = tl.program_id(1).to(tl.int64)
    base = lane * M
    rows = (tl.program_id(0) * BS + tl.arange(0, BS)).to(tl.int64)
    live = rows < M
    packed = tl.load(PACKED + base + rows, live, 0)
    best = tl.minimum(_unpack_best(packed, rows), M - 1)
    cost = (packed >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    alive = tl.load(ALIVE + base + rows, live, 0) != 0
    mutual = _unpack_best(tl.load(PACKED + base + best, live, 0), best) == rows
    valid = live & alive & (rows < best) & mutual & (cost < float("inf"))
    tl.atomic_add(NPROP + lane, tl.sum(valid.to(tl.int32), axis=0))
    if HAVE_LIMITS:
        # GEMM cancellation must not widen a candidate: verify with differences.
        acc = tl.zeros([BS], tl.float32)
        for start in tl.static_range(0, DIM, BD):
            d = start + tl.arange(0, BD)
            mask = valid[:, None] & (d < DIM)[None, :]
            xa = tl.load(MU + (base + rows)[:, None] * DIM + d[None, :], mask, 0.0)
            xb = tl.load(MU + (base + best)[:, None] * DIM + d[None, :], mask, 0.0)
            diff = xa - xb
            acc += tl.sum(diff * diff, axis=1)
        if SQRT_RN:
            exact = tl.sqrt_rn(acc)
        else:
            exact = tl.sqrt(acc)
        ma = tl.load(MASS + base + rows, valid, 0.0)
        mb = tl.load(MASS + base + best, valid, 0.0)
        denom = tl.maximum(ma + mb, 1.0)
        bound = tl.maximum(tl.load(RADIUS + base + rows, valid, 0.0) + tl.div_rn(mb, denom) * exact,
                           tl.load(RADIUS + base + best, valid, 0.0) + tl.div_rn(ma, denom) * exact)
        rejected = valid & (bound > tl.load(LIMITS + lane))
        tl.store(BOUND + base + rows, bound, valid & ~rejected)
        tl.store(REJECT + base + rows, ROUND, rejected)
        tl.store(REJECT + base + best, ROUND, rejected)
        # Keep the exact entry: its rescan reproduces this bound and rejects it.
        tl.store(DIST + (base + rows) * M + best, acc, rejected)
        tl.store(DIST + (base + best) * M + rows, acc, rejected)
        valid = valid & ~rejected
    tl.store(BEST + base + rows, best, live)
    tl.store(PAIR_COST + base + rows, tl.where(valid, cost, float("inf")), live)


@triton.jit(do_not_specialize=["M", "H", "ROUND", "TARGET"])
def _plan(ORDER, PAIR_COST, BEST, MASS, RADIUS, BOUND, DIST, ALIVE, KEEP, MERGED, PARTNER, PMA, PMB, PDAB,
          PAIR_A, COUNT, MERGES, NPROP, ACTIVE, STUCK, TRACE, M, H, ROUND, TARGET,
          HAVE_LIMITS: tl.constexpr, BLOCK: tl.constexpr):
    """Cheapest pairs within the lane budget; commit masses and the merge trace."""
    lane = tl.program_id(0).to(tl.int64)
    base = lane * M
    t = tl.arange(0, BLOCK)
    inb = t < H
    left = tl.load(ORDER + base + t, inb, 0)
    ok = inb & (tl.load(PAIR_COST + base + left, inb, float("inf")) < float("inf"))
    count = tl.load(COUNT + lane)
    nprop = tl.load(NPROP + lane)
    rank = tl.cumsum(ok.to(tl.int32), axis=0)
    ok = ok & (rank <= count - TARGET)
    taken = tl.sum(ok.to(tl.int32), axis=0)
    right = tl.load(BEST + base + left, ok, 0)
    ma = tl.load(MASS + base + left, ok, 0.0)
    mb = tl.load(MASS + base + right, ok, 0.0)
    dab = tl.load(DIST + (base + left) * M + right, ok, 0.0)
    merges = tl.load(MERGES + lane)
    stuck = tl.load(STUCK + lane)
    tl.debug_barrier()
    tl.store(NPROP + lane, 0)
    tl.store(PAIR_A + lane * H + t, tl.where(ok, left, -1).to(tl.int32), inb)
    tl.store(PARTNER + base + left, right, ok)
    tl.store(PMA + base + left, ma, ok)
    tl.store(PMB + base + left, mb, ok)
    tl.store(PDAB + base + left, dab, ok)
    tl.store(KEEP + base + left, ROUND, ok)
    tl.store(MERGED + base + left, ROUND, ok)
    tl.store(MERGED + base + right, ROUND, ok)
    tl.store(MASS + base + left, ma + mb, ok)
    tl.store(MASS + base + right, 0.0, ok)
    tl.store(ALIVE + base + right, 0, ok)
    if HAVE_LIMITS:
        tl.store(RADIUS + base + left, tl.load(BOUND + base + left, ok, 0.0), ok)
    slot = (base + merges + rank - 1) * 2
    tl.store(TRACE + slot, left, ok)
    tl.store(TRACE + slot + 1, right, ok)
    remaining = count - taken
    tl.store(COUNT + lane, remaining)
    tl.store(MERGES + lane, merges + taken)
    tl.store(ACTIVE + lane, ((remaining > TARGET) & (nprop > 0)).to(tl.int32))
    tl.store(STUCK + lane, stuck | ((remaining > TARGET) & (nprop == 0)).to(tl.int32))


@triton.jit(do_not_specialize=["M", "H", "ROUND"])
def _lance_williams(DIST, MU, MASS, RADIUS, LIMITS, ALIVE, KEEP, PARTNER, PMA, PMB, PDAB, PAIR_A, PACKED,
                    M, H, ROUND, DIM: tl.constexpr, HAVE_LIMITS: tl.constexpr, SQRT_RN: tl.constexpr,
                    BLOCK: tl.constexpr, BD: tl.constexpr):
    """Merged centroid, updated distance row/column, row minimum and pushes.

    Row a is written only by its own program; columns are written only into
    live rows that did not merge this round. Dead rows b and dead columns are
    never written, so every read of rows a/b and gathered columns b_q sees the
    previous round.
    """
    t = tl.program_id(0)
    lane = tl.program_id(1).to(tl.int64)
    base = lane * M
    a = tl.load(PAIR_A + lane * H + t).to(tl.int64)
    if a >= 0:
        b = tl.load(PARTNER + base + a)
        ma = tl.load(PMA + base + a)
        mb = tl.load(PMB + base + a)
        total = ma + mb
        alpha = tl.div_rn(ma, total)
        beta = tl.div_rn(mb, total)
        shift = alpha * beta * tl.load(PDAB + base + a)
        denom = tl.maximum(total, 1.0)
        for start in tl.static_range(0, DIM, BD):
            d = start + tl.arange(0, BD)
            dm = d < DIM
            xa = tl.load(MU + (base + a) * DIM + d, dm, 0.0)
            xb = tl.load(MU + (base + b) * DIM + d, dm, 0.0)
            tl.store(MU + (base + a) * DIM + d, tl.div_rn(ma * xa + mb * xb, denom), dm)
        mi = tl.load(MASS + base + a)
        result = tl.min(tl.full([BLOCK], 9223372036854775807, tl.int64), axis=0)
        for start in range(0, M, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            inb = cols < M
            x_aa = tl.load(DIST + (base + a) * M + cols, inb, 0.0)
            x_ba = tl.load(DIST + (base + b) * M + cols, inb, 0.0)
            new = alpha * x_aa + beta * x_ba - shift
            keep = inb & (tl.load(KEEP + base + cols, inb, -1) == ROUND)
            q = tl.load(PARTNER + base + cols, keep, 0)
            x_ab = tl.load(DIST + (base + a) * M + q, keep, 0.0)
            x_bb = tl.load(DIST + (base + b) * M + q, keep, 0.0)
            qa = tl.load(PMA + base + cols, keep, 1.0)
            qb = tl.load(PMB + base + cols, keep, 0.0)
            qalpha = tl.div_rn(qa, qa + qb)
            qbeta = tl.div_rn(qb, qa + qb)
            qshift = qalpha * qbeta * tl.load(PDAB + base + cols, keep, 0.0)
            # Both clusters merged this round: expand the lower keep first, so
            # the program owning the other row computes the same bits and D
            # stays exactly symmetric (asymmetry can leave no mutual pair).
            own_first = qalpha * new + qbeta * (alpha * x_ab + beta * x_bb - shift) - qshift
            other_first = (alpha * (qalpha * x_aa + qbeta * x_ab - qshift)
                           + beta * (qalpha * x_ba + qbeta * x_bb - qshift) - shift)
            new = tl.where(keep, tl.where(a < cols, own_first, other_first), new)
            new = tl.where(cols == a, 0.0, tl.maximum(new, 0.0))
            alive = inb & (tl.load(ALIVE + base + cols, inb, 0) != 0)
            tl.store(DIST + (base + a) * M + cols, new, alive)
            tl.store(DIST + (base + cols) * M + a, new, alive & ~keep)
            mj = tl.load(MASS + base + cols, inb, 0.0)
            if HAVE_LIMITS:
                rj = tl.load(RADIUS + base + cols, inb, 0.0)
                cost, valid = _cost_limited(new, mi, mj, tl.load(RADIUS + base + a), rj,
                                            tl.load(LIMITS + lane), alive & (cols != a), SQRT_RN)
            else:
                cost, valid = _cost(new, mi, mj, alive & (cols != a))
            packed = tl.where(inb, _pack(cost, cols ^ a), 9223372036854775807)
            result = tl.minimum(result, tl.min(packed, axis=0))
            tl.atomic_min(PACKED + base + cols, packed, mask=valid & ~keep)
        tl.store(PACKED + base + a, result)


@triton.jit(do_not_specialize=["M"])
def _mirror(DIST, M, BLOCK: tl.constexpr):
    """Copy the upper triangle onto the lower one, in place and exactly."""
    bi = tl.program_id(0)
    bj = tl.program_id(1)
    if bj <= bi:
        base = tl.program_id(2).to(tl.int64) * M * M
        rows = bi * BLOCK + tl.arange(0, BLOCK)
        cols = bj * BLOCK + tl.arange(0, BLOCK)
        # Read the transposed tile row-contiguously; lower entries are never sources.
        upper = tl.load(DIST + base + cols[:, None].to(tl.int64) * M + rows[None, :],
                        (cols < M)[:, None] & (rows < M)[None, :], 0.0)
        lower = (rows[:, None] > cols[None, :]) & (rows < M)[:, None] & (cols < M)[None, :]
        tl.store(DIST + base + rows[:, None].to(tl.int64) * M + cols[None, :], tl.trans(upper), lower)


def mirror(dist):
    """Make [L, M, M] exactly symmetric; the incremental rounds rely on it."""
    lanes, size = dist.shape[:2]
    if lanes and size > 1:
        tiles = triton.cdiv(size, 32)
        _mirror[(tiles, tiles, lanes)](dist, size, 32, num_warps=4)
    return dist


class UnifiedReduce:
    """Device state for incremental rounds; the host reads one flag per lane.

    Shapes never shrink, so rounds need no host sizes. Traces are written in
    round order and, within a round, in (cost, left) order.
    """

    INF_KEY = 0x7F800000 << 32

    def __init__(self, dist, mu, mass, count, limits, target):
        lanes, size = mass.shape
        dev = mass.device
        self.dist, self.mu, self.mass, self.limits = dist, mu, mass, limits
        self.size, self.half, self.target = size, size // 2, int(target)
        self.dim = mu.size(-1)
        ids = torch.arange(size, device=dev)
        self.alive = (ids[None, :] < count[:, None]).to(torch.int8)
        self.radius = torch.zeros_like(mass)
        self.bound = torch.zeros_like(mass)
        self.count = count.to(torch.int32)
        self.merges = torch.zeros(lanes, dtype=torch.int32, device=dev)
        self.nprop = torch.zeros_like(self.merges)
        self.active = torch.zeros_like(self.merges)
        self.stuck = torch.zeros_like(self.merges)
        self.keep = torch.full((lanes, size), -1, dtype=torch.int32, device=dev)
        self.merged = torch.full_like(self.keep, -1)
        self.reject = torch.full_like(self.keep, -1)
        self.partner = torch.zeros((lanes, size), dtype=torch.int64, device=dev)
        self.best = torch.zeros_like(self.partner)
        self.pma = torch.ones_like(mass)
        self.pmb = torch.zeros_like(mass)
        self.pdab = torch.zeros_like(mass)
        self.pair_cost = torch.empty_like(mass)
        self.pair_a = torch.empty((lanes, max(self.half, 1)), dtype=torch.int32, device=dev)
        self.trace = torch.zeros((lanes, size, 2), dtype=torch.int64, device=dev)
        self.packed = torch.full((lanes, size), self.INF_KEY, dtype=torch.int64, device=dev)
        self.sqrt_rn = hasattr(tl, "sqrt_rn")
        self.row_block = min(1024, triton.next_power_of_2(size))
        self.dim_block = min(32, triton.next_power_of_2(self.dim))
        self._scan(0, full=True)

    def _limits(self):
        return self.limits if self.limits is not None else self.mass

    def _scan(self, round_, *, full=False):
        lanes, size = self.mass.shape
        _scan[(triton.cdiv(size, 4), lanes)](
            self.dist, self.mass, self.radius, self._limits(), self.alive, self.keep, self.merged,
            self.reject, self.best, self.packed, size, round_, full, self.limits is not None,
            self.sqrt_rn, 4, self.row_block, num_warps=4, enable_fp_fusion=False,
        )

    def step(self, round_):
        lanes, size = self.mass.shape
        limited = self.limits is not None
        _select[(triton.cdiv(size, 64), lanes)](
            self.packed, self.alive, self.mass, self.radius, self._limits(), self.mu, self.dist,
            self.best, self.pair_cost, self.bound, self.reject, self.nprop, size, round_,
            self.dim, limited, self.sqrt_rn, 64, self.dim_block, num_warps=4, enable_fp_fusion=False,
        )
        order = self.pair_cost.argsort(dim=1, stable=True)
        plan_block = triton.next_power_of_2(max(self.half, 1))
        _plan[(lanes,)](
            order, self.pair_cost, self.best, self.mass, self.radius, self.bound, self.dist, self.alive,
            self.keep, self.merged, self.partner, self.pma, self.pmb, self.pdab, self.pair_a,
            self.count, self.merges, self.nprop, self.active, self.stuck, self.trace,
            size, self.half, round_, self.target, limited, plan_block,
            num_warps=8 if plan_block > 1024 else 4, enable_fp_fusion=False,
        )
        _lance_williams[(max(self.half, 1), lanes)](
            self.dist, self.mu, self.mass, self.radius, self._limits(), self.alive, self.keep,
            self.partner, self.pma, self.pmb, self.pdab, self.pair_a, self.packed,
            size, self.half, round_, self.dim, limited, self.sqrt_rn, self.row_block,
            triton.next_power_of_2(self.dim) if self.dim <= 256 else 256,
            num_warps=4, enable_fp_fusion=False,
        )
        self._scan(round_)
        return self.active

    def finish(self):
        """One readback: per-lane traces, merge counts, survivors and stuck flags."""
        lanes, size = self.alive.shape
        packed = torch.cat((self.merges.long(), self.stuck.long(), self.alive.long().view(-1),
                            self.trace.view(-1))).cpu().numpy()
        merges, stuck = packed[:lanes], packed[lanes:2 * lanes].astype(bool)
        alive = packed[2 * lanes:2 * lanes + lanes * size].reshape(lanes, size).astype(bool)
        trace = packed[2 * lanes + lanes * size:].reshape(lanes, size, 2)
        return [trace[lane, :n] for lane, n in enumerate(merges)], alive, stuck
