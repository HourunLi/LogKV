"""Fused Ward rounds vs the Torch reference on CPU through the Triton interpreter.

Run with TRITON_INTERPRET=1 (tests/test_log_kv_triton_interpret.py does).
"""
import torch

from litgpt import log_kv_route_triton as fused
from litgpt.log_kv_cache import LogStructuredKVCache, _UnifiedReduceTorch


def run(reducer_type, mu, mass, count, limits, target, passes, narrow):
    state = mu.clone()
    dist = LogStructuredKVCache._semantic_unified_pair_matrix(state)
    reducer = reducer_type(dist, state, mass.clone(), count, limits, target, passes)
    for round_ in range(1, 2 * mu.size(1) + 8):
        flag = reducer.step(round_)
        if narrow and hasattr(reducer, "limit_live") and round_ % 2 == 0:
            reducer.limit_live(int(reducer.count.max()))
        if not bool(flag.any()):
            break
    assert torch.equal(dist, dist.transpose(1, 2)), "incremental distance matrix lost symmetry"
    return reducer.finish(), state


def main():
    for sort in (True, False):
        # In-kernel bitonic sort and the preallocated argsort fallback.
        fused.UnifiedReduce.sort_ok = sort
        check_rounds()
    check_pair_distances()
    print("route kernels match the Torch reference")


def check_rounds():
    generator = torch.Generator().manual_seed(95)
    for size in (9, 40, 70):
        mu = torch.randn(4, size, 8, generator=generator)
        count = torch.tensor([size, size - 3, 5, 1])
        mask = torch.arange(size)[None, :] < count[:, None]
        mu = mu * mask[..., None]
        mass = torch.randint(1, 4, (4, size), generator=generator).float() * mask
        cases = ((None, 3, 1), (None, 3, 4), (None, 1, 2), (torch.tensor([1., 2., 30., .1]), 1, 1))
        for limits, target, passes in cases:
            for narrow in (False, True):
                (expected, expected_mu) = run(_UnifiedReduceTorch, mu, mass, count, limits, target, passes, False)
                (actual, actual_mu) = run(fused.UnifiedReduce, mu, mass, count, limits, target, passes, narrow)
                label = f"size={size} limits={limits is not None} passes={passes} narrow={narrow}"
                if limits is None:
                    for a, b in zip(actual[0], expected[0]):
                        assert (a == b).all(), f"fused merge trace differs from Torch ({label})"
                    assert (actual[1] == expected[1]).all() and (actual[2] == expected[2]).all(), label
                    torch.testing.assert_close(actual_mu, expected_mu, rtol=0, atol=0, msg=label)
                else:
                    # The candidate pass sums exact squared differences in another
                    # order, so compare partitions.
                    for lane, (a, b) in enumerate(zip(actual[0], expected[0])):
                        assert sorted(map(tuple, a.tolist())) == sorted(map(tuple, b.tolist())), \
                            f"fused candidate partition differs from Torch (lane {lane}, {label})"
                    assert (actual[1] == expected[1]).all(), label


def check_pair_distances():
    # Fused distance epilogue against the unfused expression on the same GEMM.
    generator = torch.Generator().manual_seed(96)
    for lanes, size, dim in ((3, 33, 7), (2, 130, 16)):
        mu = torch.randn(lanes, size, dim, generator=generator)
        centered = mu - mu[:, :1]
        norm = centered.square().sum(-1)
        gram = torch.bmm(centered, centered.transpose(1, 2))
        dist = ((norm[:, :, None] + norm[:, None, :]) - 2 * gram).clamp_min(0)
        expected = dist.triu() + dist.triu(1).transpose(1, 2)
        assert torch.equal(fused.pair_distances(gram.clone(), norm), expected), "fused pair distances differ"


if __name__ == "__main__":
    main()
