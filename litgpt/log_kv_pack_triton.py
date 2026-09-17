"""One kernel writes pooled RoPE keys, mass bias, recent and current K/V.

No attention implementation lives here. The result goes to PyTorch Flash SDPA.
Compile with enable_fp_fusion=False to match the reference RoPE rounding order.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["P", "R", "T", "OCAP", "START"])
def _pack(
    LK, LV, W, IDX, POS, VALID, COS, SIN, RK, RV, CK, CV, OK, OV,
    N: tl.constexpr, P, R, T, START,
    DK: tl.constexpr, DV: tl.constexpr, DR: tl.constexpr, DA: tl.constexpr,
    GROUPS: tl.constexpr, RCAP: tl.constexpr, OCAP,
    CKB: tl.constexpr, CKG: tl.constexpr, CKT: tl.constexpr, CKD: tl.constexpr,
    CVB: tl.constexpr, CVG: tl.constexpr, CVT: tl.constexpr, CVD: tl.constexpr,
    COS0: tl.constexpr, COS1: tl.constexpr, SIN0: tl.constexpr, SIN1: tl.constexpr,
    ROPE_FP32: tl.constexpr,
    BT: tl.constexpr, BD: tl.constexpr,
):
    lane = tl.program_id(0)
    s = START + tl.program_id(1) * BT + tl.arange(0, BT)
    d = tl.arange(0, BD)
    live = s < P + R + T
    in_pool = live & (s < P)
    ix = tl.load(IDX + lane * P + s, in_pool, 0)
    pos = tl.load(POS + lane * P + s, in_pool, 0)
    valid = tl.load(VALID + lane * P + s, in_pool, 0)
    kmask = (in_pool & valid)[:, None] & (d < DK)[None, :]
    raw = tl.load(LK + (lane * N + ix[:, None]) * DK + d[None, :], kmask, 0).to(tl.float32)
    if DR > 0:
        other = tl.where(d < DR // 2, d + DR // 2, d - DR // 2)
        rotated = tl.load(
            LK + (lane * N + ix[:, None]) * DK + other[None, :],
            (in_pool & valid)[:, None] & (d < DR)[None, :], 0,
        ).to(tl.float32)
        rotated = tl.where((d < DR // 2)[None, :], -rotated, rotated)
        cos = tl.load(COS + pos[:, None] * COS0 + d[None, :] * COS1,
                      (in_pool & valid)[:, None] & (d < DR)[None, :], 0).to(tl.float32)
        sin = tl.load(SIN + pos[:, None] * SIN0 + d[None, :] * SIN1,
                      (in_pool & valid)[:, None] & (d < DR)[None, :], 0).to(tl.float32)
        a, b = raw * cos, rotated * sin
        if not ROPE_FP32:
            a = a.to(LK.dtype.element_ty).to(tl.float32)
            b = b.to(LK.dtype.element_ty).to(tl.float32)
        raw = tl.where((d < DR)[None, :], a + b, raw)
    kval = raw
    vval = tl.load(LV + (lane * N + ix[:, None]) * DV + d[None, :],
                   (in_pool & valid)[:, None] & (d < DV)[None, :], 0).to(tl.float32)
    mass = tl.load(W + lane * N + ix, in_pool & valid, 1).to(tl.float32)
    bias = tl.where(valid, tl.log(tl.maximum(mass, 1.0)), -10000.0)
    kval = tl.where(in_pool[:, None] & (d == DK)[None, :], bias[:, None], kval)

    in_recent = live & (s >= P) & (s < P + R)
    kr = tl.load(RK + (lane * RCAP + (s - P)[:, None]) * DK + d[None, :],
                 in_recent[:, None] & (d < DK)[None, :], 0)
    vr = tl.load(RV + (lane * RCAP + (s - P)[:, None]) * DV + d[None, :],
                 in_recent[:, None] & (d < DV)[None, :], 0)
    kval = tl.where(in_recent[:, None], kr.to(tl.float32), kval)
    vval = tl.where(in_recent[:, None], vr.to(tl.float32), vval)

    in_current = live & (s >= P + R)
    cb, cg = lane // GROUPS, lane % GROUPS
    kc = tl.load(CK + cb * CKB + cg * CKG + (s - P - R)[:, None] * CKT + d[None, :] * CKD,
                 in_current[:, None] & (d < DK)[None, :], 0)
    vc = tl.load(CV + cb * CVB + cg * CVG + (s - P - R)[:, None] * CVT + d[None, :] * CVD,
                 in_current[:, None] & (d < DV)[None, :], 0)
    kval = tl.where(in_current[:, None], kc.to(tl.float32), kval)
    vval = tl.where(in_current[:, None], vc.to(tl.float32), vval)
    out = (lane * OCAP + s[:, None]) * DA + d[None, :]
    mask = live[:, None] & (d < DA)[None, :]
    tl.store(OK + out, kval, mask)
    tl.store(OV + out, vval, mask)


def pack(cache, plan, k, v, dim, buffers=None, skip_pooled=False, recent_start=0):
    indices, positions, _, valid = plan
    pooled, recent, current = indices.size(-1), cache.recent_count, k.size(2)
    length = pooled + recent + current
    if buffers is None:
        ka = k.new_empty((*k.shape[:2], length, dim))
        va = v.new_empty((*v.shape[:2], length, dim))
    else:
        ka, va = buffers
    capacity = ka.size(2)
    start = pooled + recent_start if skip_pooled else 0
    work = length - start
    if work:
        _pack[(k.size(0) * k.size(1), triton.cdiv(work, 16))](
            cache.level_k, cache.level_v, cache.level_w, indices, positions, valid,
            cache.cos_cache, cache.sin_cache, cache.recent_k, cache.recent_v, k, v, ka, va,
            N=cache.K_max * cache.L_alloc * cache.B_prime, P=pooled, R=recent, T=current, START=start,
            DK=k.size(-1), DV=v.size(-1), DR=cache.rope_n_elem, DA=dim,
            GROUPS=k.size(1), RCAP=cache.recent_capacity, OCAP=capacity,
            CKB=k.stride(0), CKG=k.stride(1), CKT=k.stride(2), CKD=k.stride(3),
            CVB=v.stride(0), CVG=v.stride(1), CVT=v.stride(2), CVD=v.stride(3),
            COS0=cache.cos_cache.stride(0), COS1=cache.cos_cache.stride(1),
            SIN0=cache.sin_cache.stride(0), SIN1=cache.sin_cache.stride(1),
            ROPE_FP32=cache.cos_cache.dtype == torch.float32 or k.dtype == torch.float32,
            BT=16, BD=triton.next_power_of_2(dim),
            num_warps=4, enable_fp_fusion=False,
        )
    return ka[:, :, :length], va[:, :, :length]
