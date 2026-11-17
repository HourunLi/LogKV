import numpy as np
import pytest
import torch

from litgpt.config import Config
from litgpt.dual_exact import choose_dual_spans, cut_dual_spans
from litgpt.model import GPT


def test_fixed_quotas_keep_duplicates_and_archive_a_true_token_only_once():
    # Both pools pick the same first two-token span; each also picks its own second span.
    cut = cut_dual_spans([[[]], [[]]], [[]], [list(range(8))], [[False, True] * 4], 0, 2)

    class Stream:
        def __init__(self, values):
            self.values = values

        def span_values(self, row, positions, lengths, stop):
            return np.array(self.values)

    selected, state = choose_dual_spans(cut, [0.] * 8, 8, 4,
                                       Stream([100., 90., 0., 0.]), Stream([100., 0., 0., 90.]), [8])
    assert selected.positions == [[0, 1, 2, 3, 0, 1, 6, 7]]
    assert selected.archive_positions[0].tolist() == [4, 5]
    assert sum(n for n, _ in state[0][0]) == sum(n for n, _ in state[1][0]) == 4
    # An old duplicate drops from both pools: only one copy may enter the ladder.
    new = cut_dual_spans(state, selected.positions, [list(range(8, 12))], [[False, True] * 2], 8, 2)
    selected, _ = choose_dual_spans(new, [0.] * 12, 8, 4,
                                   Stream([0., 0., 100., 100.]), Stream([0., 0., 100., 100.]), [12])
    assert len(selected.archive_positions[0]) == len(set(selected.archive_positions[0]))
    assert set(selected.archive_positions[0]) == {0, 1, 2, 3, 6, 7}


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_streaming_dual_pool_conserves_source_mass_positions_and_resets(prefetch, batch_size):
    torch.manual_seed(19)
    model = GPT(Config(block_size=128, n_layer=2, n_embd=16, n_head=2, n_query_groups=1,
                       vocab_size=41, padding_multiple=1)).eval()
    model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7],
                                      literal_texts=[f" w{i}" for i in range(41)])
    model.set_log_kv_cache(batch_size=batch_size, B=8, recent_size=8, prefill_block=8, second_order_scale=0.,
                           semantic_clusters=True, cluster_k_max=2, semantic_flush_granularity=8,
                           semantic_anchor_mode="mid", alpha_exact_tokens=8, alpha_span_max_tokens=2,
                           gamma_query_cue=True, gamma_query_pool_tokens=4, gamma_surprisal="word_first")
    model.set_gamma_queries([[11, 13]] * batch_size)
    for block in model.transformer.h:
        block.attn.kv_cache._alpha_prefetch_on_cpu = prefetch
    ids = torch.randint(0, 41, (batch_size, 72))
    with torch.no_grad():
        for start in range(0, 72, 8):
            logits = model(ids[:, start:start + 8], input_pos=torch.arange(start, start + 8))
            assert torch.isfinite(logits).all()
            for block in model.transformer.h:
                cache = block.attn.kv_cache
                unique = torch.tensor([len(set(row)) for row in cache._alpha_positions])[:, None]
                mass = cache.level_w.sum((2, 3, 4)) + unique + cache.recent_count
                torch.testing.assert_close(mass, torch.full_like(mass, start + 8))
                assert cache.alpha_count <= 8
                assert all(sum(n for n, _ in row) <= 4 for pool in cache._alpha_dual_spans for row in pool)
    model.reset_log_kv_cache()
    for block in model.transformer.h:
        cache = block.attn.kv_cache
        assert cache.alpha_count == 0 and cache._alpha_dual_spans == [[[]] * batch_size, [[]] * batch_size]


def test_duplicate_attention_is_exactly_a_log_two_bias():
    q, k, v = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    duplicated = torch.nn.functional.scaled_dot_product_attention(
        q, torch.cat((k, k[:, :, 1:2]), 2), torch.cat((v, v[:, :, 1:2]), 2))
    bias = torch.tensor([0., np.log(2.), 0.], dtype=q.dtype).reshape(1, 1, 1, 3)
    equivalent = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    torch.testing.assert_close(duplicated, equivalent)
