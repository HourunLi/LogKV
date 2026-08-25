import torch

from litgpt.log_kv_position import (
    anchor_mass_bias,
    dedup_anchors,
    materialize_anchor_directions,
    materialize_anchor_keys,
    merge_anchors,
    mid_anchor,
)
from litgpt.model import apply_rope, build_rope_cache


def _expected_apply_rope_at(
    content: torch.Tensor,
    anchors: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_n_elem: int,
) -> torch.Tensor:
    expanded = content.unsqueeze(-2).expand(*anchors.shape, content.size(-1))
    flat = expanded.reshape(1, -1, content.size(-1))
    pos = anchors.reshape(-1)
    roped = apply_rope(flat[..., :rope_n_elem], cos[pos].unsqueeze(0), sin[pos].unsqueeze(0))
    return torch.cat((roped, flat[..., rope_n_elem:]), dim=-1).reshape(*anchors.shape, -1)


def test_merge_anchors_is_associative_and_pad_identity() -> None:
    pad_lo = torch.tensor([torch.iinfo(torch.int64).max])
    pad_hi = torch.tensor([-1])
    zero = torch.tensor([0])
    real = (torch.tensor([3]), torch.tensor([7]), torch.tensor([20]))

    for got, exp in zip(merge_anchors(*real, pad_lo, pad_hi, zero), real):
        torch.testing.assert_close(got, exp)
    for got, exp in zip(merge_anchors(pad_lo, pad_hi, zero, *real), real):
        torch.testing.assert_close(got, exp)

    a = (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    b = (torch.tensor([3]), torch.tensor([3]), torch.tensor([3]))
    c = (torch.tensor([8]), torch.tensor([8]), torch.tensor([8]))
    left = merge_anchors(*merge_anchors(*a, *b), *c)
    right = merge_anchors(*a, *merge_anchors(*b, *c))

    for got, exp in zip(left, right):
        torch.testing.assert_close(got, exp)
    torch.testing.assert_close(mid_anchor(left[0], left[1], left[2], torch.tensor([3])), torch.tensor([4]))


def test_mid_anchor_uses_integer_round_half_up_and_clamps() -> None:
    lo = torch.tensor([0, 0, 10, 1_000_000])
    hi = torch.tensor([1, 2, 12, 1_000_001])
    sum_wp = torch.tensor([1, 2, 150, 2_000_001], dtype=torch.int64)
    w = torch.tensor([2, 2, 10, 2])

    torch.testing.assert_close(mid_anchor(lo, hi, sum_wp, w), torch.tensor([1, 1, 12, 1_000_001]))


def test_dedup_anchors_keeps_first_occurrence_and_sanitizes_invalid_entries() -> None:
    lo = torch.tensor([5, 0, 0, torch.iinfo(torch.int64).max])
    mid = torch.tensor([5, 1, 0, -1])
    hi = torch.tensor([5, 1, 2, -1])
    w = torch.tensor([1.0, 2.0, 3.0, 0.0])

    anchors, slot_valid, M = dedup_anchors(lo, hi, mid, w)

    torch.testing.assert_close(anchors, torch.tensor([[5, 5, 5], [0, 1, 1], [0, 0, 2], [0, 0, 0]]))
    assert torch.equal(
        slot_valid,
        torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, False, True],
                [False, False, False],
            ]
        ),
    )
    torch.testing.assert_close(M, torch.tensor([1, 2, 2, 1]))


def test_materialize_anchor_keys_matches_model_apply_rope() -> None:
    torch.manual_seed(123)
    rope_n_elem = 4
    cos, sin = build_rope_cache(seq_len=8, n_elem=rope_n_elem)
    k_raw = torch.randn(1, 2, 2, 6)
    anchors = torch.tensor([[[[0, 2, 5], [1, 1, 4]], [[3, 6, 7], [0, 0, 0]]]])

    got = materialize_anchor_keys(k_raw, anchors, cos, sin, rope_n_elem)
    expected = _expected_apply_rope_at(k_raw, anchors, cos, sin, rope_n_elem)

    torch.testing.assert_close(got, expected)


def test_materialize_anchor_keys_handles_rope_one_cache_widening() -> None:
    torch.manual_seed(321)
    rope_n_elem = 1
    cos, sin = build_rope_cache(seq_len=4, n_elem=rope_n_elem)
    k_raw = torch.randn(1, 1, 2, 3)
    anchors = torch.tensor([[[[0], [3]]]])

    got = materialize_anchor_keys(k_raw, anchors, cos, sin, rope_n_elem)
    expected = _expected_apply_rope_at(k_raw, anchors, cos, sin, rope_n_elem)

    assert got.shape[-1] == 4
    torch.testing.assert_close(got, expected)


def test_materialize_anchor_directions_rotates_key_space_only() -> None:
    torch.manual_seed(456)
    rope_n_elem = 4
    cos, sin = build_rope_cache(seq_len=8, n_elem=rope_n_elem)
    anchors = torch.tensor([[[[5]]]])
    sigma_u = torch.randn(1, 1, 1, 6)
    gamma_a = torch.randn(1, 1, 1, 6)

    sigma_eff, gamma_eff = materialize_anchor_directions(sigma_u, gamma_a, anchors, cos, sin, rope_n_elem)

    torch.testing.assert_close(sigma_eff, _expected_apply_rope_at(sigma_u, anchors, cos, sin, rope_n_elem))
    torch.testing.assert_close(gamma_eff, _expected_apply_rope_at(gamma_a, anchors, cos, sin, rope_n_elem))

    q_raw = torch.randn(1, 1, 1, 6)
    q = _expected_apply_rope_at(q_raw, torch.tensor([[[[3]]]]), cos, sin, rope_n_elem).squeeze(-2)
    assert not torch.allclose((q * sigma_eff.squeeze(-2)).sum(-1), (q * sigma_u).sum(-1))


def test_anchor_mass_bias_conserves_count_factor_for_any_lambda() -> None:
    lo = torch.tensor([0, 0, 5, torch.iinfo(torch.int64).max])
    mid = torch.tensor([1, 0, 5, -1])
    hi = torch.tensor([2, 2, 5, -1])
    w = torch.tensor([8.0, 4.0, 1.0, 0.0])
    _anchors, slot_valid, M = dedup_anchors(lo, hi, mid, w)
    w_3 = w.unsqueeze(-1).expand_as(slot_valid)
    M_3 = M.unsqueeze(-1).expand_as(slot_valid)

    for lam in (0.0, 0.5, 1.0):
        factor = torch.exp(anchor_mass_bias(w_3, M_3, lam=lam)) * slot_valid
        torch.testing.assert_close(factor.sum(dim=-1)[:3], w[:3].pow(lam))

    assert torch.isfinite(anchor_mass_bias(torch.tensor([0.0]), torch.tensor([0]), lam=0.0)).all()
