"""First-order ladder updates and ordered centroid updates on the current stream.

Merge reads finish before survivor writes launch: an in-place cross-CTA merge
would race with readers of the old slots. No atomics or host readbacks are used.
"""

import torch
import triton
import triton.language as tl


_FIELDS = (0, 1, 2, 8, 9, 10, 11, 12)


@triton.jit
def _read(F, S, SI, GI, index, stored_rows, d, mask, DIM: tl.constexpr):
    stored = index < stored_rows
    fi = tl.load(SI + index, mask & stored, 0).to(tl.int64)
    si = tl.load(GI + index - stored_rows, mask & ~stored, 0).to(tl.int64)
    a = tl.load(F + fi[:, None] * DIM + d[None, :], mask[:, None] & stored[:, None] & (d < DIM)[None, :], 0)
    b = tl.load(S + si[:, None] * DIM + d[None, :], mask[:, None] & ~stored[:, None] & (d < DIM)[None, :], 0)
    # Reference pool construction casts staging rows to the storage dtype.
    return tl.where(stored[:, None], a, b.to(F.dtype.element_ty))


@triton.jit(do_not_specialize=["NS", "NP", "NV"])
def _merge(F, S, O, SI, GI, PI, VI, NS, NP, NV,
           DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    d = tl.arange(0, BD).to(tl.int64)
    live, pair = r < NP + NV, r < NP
    a_pair = tl.load(PI + 2 * r, pair, 0).to(tl.int64)
    a_copy = tl.load(VI + r - NP, live & ~pair, 0).to(tl.int64)
    ai = tl.where(pair, a_pair, a_copy)
    bi = tl.load(PI + 2 * r + 1, pair, 0).to(tl.int64)
    zero = tl.arange(0, 1).to(tl.int64)
    wa = _read(F[2], S[2], SI, GI, ai, NS, zero, live, 1).to(tl.float32)
    wb = _read(F[2], S[2], SI, GI, bi, NS, zero, pair, 1).to(tl.float32)
    alpha = tl.div_rn(wa, tl.maximum(wa + wb, 1.e-8))
    for f in tl.static_range(8):
        dim = DK if f == 0 else DV if f == 1 else 1
        a = _read(F[f], S[f], SI, GI, ai, NS, d, live, dim)
        b = _read(F[f], S[f], SI, GI, bi, NS, d, pair, dim)
        if f < 2:
            value = alpha * a.to(tl.float32) + (1.0 - alpha) * b.to(tl.float32)
        elif f == 2 or f == 5:
            value = a + b
        elif f == 3:
            value = tl.where((wa > 0) & (wb > 0), tl.minimum(a, b), tl.where(wa > 0, a, b))
        elif f == 4:
            value = tl.where((wa > 0) & (wb > 0), tl.maximum(a, b), tl.where(wa > 0, a, b))
        elif f == 6:
            value = tl.minimum(a, b)
        else:
            value = a & b
        value = tl.where(pair[:, None], value, a)
        tl.store(O[f] + r[:, None] * dim + d[None, :], value, live[:, None] & (d < dim)[None, :])


@triton.jit(do_not_specialize=["N", "NC", "OFFSET"])
def _scatter(F, S, SRC, DST, CLEAR, N, NC, OFFSET, DIRECT: tl.constexpr,
             DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    d = tl.arange(0, BD).to(tl.int64)
    copy = r < N
    clear = (r >= N) & (r < N + NC)
    if DIRECT:
        src = OFFSET + r
    else:
        src = tl.load(SRC + r, copy, 0).to(tl.int64)
    dst = tl.where(copy, tl.load(DST + r, copy, 0), tl.load(CLEAR + r - N, clear, 0)).to(tl.int64)
    for f in tl.static_range(8):
        dim = DK if f == 0 else DV if f == 1 else 1
        value = tl.load(S[f] + src[:, None] * dim + d[None, :], copy[:, None] & (d < dim)[None, :], 0)
        tl.store(F[f] + dst[:, None] * dim + d[None, :], value,
                 (copy | clear)[:, None] & (d < dim)[None, :])


def fill(fields, stage, src, dst):
    if dst.numel():
        f, s = tuple(fields[i] for i in _FIELDS), tuple(stage[i] for i in _FIELDS)
        _scatter[(triton.cdiv(dst.numel(), 8),)](
            f, s, src, dst, dst, dst.numel(), 0, 0, False,
            fields[0].size(1), fields[1].size(1), 8,
            triton.next_power_of_2(max(fields[0].size(1), fields[1].size(1))),
            num_warps=4, enable_fp_fusion=False,
        )


def merge_scatter(fields, stage, si, gi, pi, vi, dst, clear):
    f, s = tuple(fields[i] for i in _FIELDS), tuple(stage[i] for i in _FIELDS)
    pairs, survivors = pi.numel() // 2, vi.numel()
    n = pairs + survivors
    # Carry K/V remain fp32, as in compact(); each next-level load rounds to
    # storage dtype. Survivor scratch prevents writes racing with merge reads.
    out = tuple(torch.empty((n, *x.shape[1:]), device=x.device,
                            dtype=torch.float32 if i < 2 else x.dtype) for i, x in enumerate(f))
    dk, dv = fields[0].size(1), fields[1].size(1)
    bd = triton.next_power_of_2(max(dk, dv))
    _merge[(triton.cdiv(n, 8),)](f, s, out, si, gi, pi, vi, si.numel(), pairs, survivors,
                               dk, dv, 8, bd, num_warps=4, enable_fp_fusion=False)
    if survivors + clear.numel():
        _scatter[(triton.cdiv(survivors + clear.numel(), 8),)](
            f, out, vi, dst, clear, survivors, clear.numel(), pairs, True,
            dk, dv, 8, bd, num_warps=4, enable_fp_fusion=False,
        )
    carry = [None] * 13
    for i, x in zip(_FIELDS, out):
        carry[i] = x[:pairs]
    return tuple(carry)


@triton.jit(do_not_specialize=["NR", "START"])
def _centroid(K, MU, NE, META, NR, START, D: tl.constexpr, BD: tl.constexpr):
    r = START + tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BD).to(tl.int64)
    cluster = tl.load(META + r).to(tl.int64)
    begin = tl.load(META + 2 * NR + r).to(tl.int64)
    length = tl.load(META + 3 * NR + r)
    forget = tl.load(META + 4 * NR + r).to(tl.int32).to(tl.float32, bitcast=True)
    acc = tl.full((BD,), 0., tl.float32)
    # A tree reduction or atomic sum changes cancellation/rounding. Keep the
    # same increasing-token accumulation order as torch.segment_reduce.
    for i in range(length):
        acc = acc + tl.load(K + (begin + i) * D + d, d < D, 0).to(tl.float32)
    pre = tl.load(NE + cluster) * forget
    n = length.to(tl.float32)
    denom = pre + n
    mu = tl.load(MU + cluster * D + d, d < D, 0)
    value = tl.where(pre > 0, tl.div_rn(pre * mu + acc, denom), tl.div_rn(acc, n))
    tl.store(MU + cluster * D + d, value, d < D)
    # NE is shared by all feature warps. A fast warp must not overwrite it
    # while another warp is still loading the old count after its sum loop.
    tl.debug_barrier()
    tl.store(NE + cluster, denom)


def centroid(k, mu, n_eff, metadata, start, count):
    _centroid[(count,)](k, mu, n_eff, metadata, metadata.size(1), start, k.size(1),
                        triton.next_power_of_2(k.size(1)), num_warps=4, enable_fp_fusion=False)
