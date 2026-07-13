"""Tests for LogStructuredKVCache with DeepSeek V4-style Content-Position Decoupling."""

import math

import pytest
import torch

from litgpt.config import Config
from litgpt.log_kv_cache import LogStructuredKVCache, log_kv_decoupled_attention
from litgpt.model import CausalSelfAttention, GPT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cache():
    """A small cache suitable for unit testing the data structure.

    head_size=8, d_pos=2, content_dim=6, v_dim=8
    B=4 (slots per level), 2:1 compaction
    max_seq_length=64 -> max_levels = max(2, ceil(log2(65/8))+1) = 5
    """
    batch_size = 1
    n_groups = 2
    max_seq_length = 64
    d_pos = 2
    content_dim = 6
    v_dim = 8
    B = 4

    k_content_shape = (batch_size, n_groups, max_seq_length, content_dim)
    v_shape = (batch_size, n_groups, max_seq_length, v_dim)

    cache = LogStructuredKVCache(
        k_content_shape, v_shape,
        d_pos=d_pos, B=B,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    return cache


def add_full_kv_in_chunks(cache: LogStructuredKVCache, k: torch.Tensor, v: torch.Tensor, chunk_size: int = 2) -> None:
    """Commit full post-RoPE K/V tensors through the supported explicit cache API."""
    for start in range(0, k.size(2), chunk_size):
        end = min(start + chunk_size, k.size(2))
        k_chunk = k[:, :, start:end, :]
        cache.add_recent(k_chunk[..., cache.d_pos:], v[:, :, start:end, :], k_chunk[..., :cache.d_pos])


# ---------------------------------------------------------------------------
# __init__ / structure tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_basic_attributes(self, small_cache):
        c = small_cache
        assert c.B == 4
        assert c.d_pos == 2
        assert c.content_dim == 6
        assert c.v_dim == 8
        assert c.max_seq_length == 64

    def test_max_levels(self, small_cache):
        """max_levels = max(2, ceil(log2((max_seq_length+1)/(B*2))) + 1)."""
        c = small_cache
        expected = max(2, math.ceil(math.log2((64 + 1) / (4 * 2))) + 1)
        assert c.max_levels == expected

    def test_max_levels_for_standard_configs(self):
        """Verify max_levels for 4K, 32K, 64K, 1M with B=1024."""
        for max_seq, expected_levels in [
            (4096, 3),
            (32768, 6),
            (65536, 7),
            (1048576, 11),
        ]:
            c = LogStructuredKVCache(
                (1, 1, max_seq, 1), (1, 1, max_seq, 1),
                d_pos=1, B=1024,
            )
            assert c.max_levels == expected_levels, (
                f"max_seq={max_seq}: got {c.max_levels}, expected {expected_levels}"
            )

    def test_level_buffers_registered(self, small_cache):
        c = small_cache
        for ell in range(c.max_levels):
            assert hasattr(c, f"level_k_{ell}")
            assert hasattr(c, f"level_v_{ell}")
            assert hasattr(c, f"level_w_{ell}")
        assert c.level_count.shape == (c.max_levels,)
        assert c.level_count.sum() == 0

    def test_initial_state_empty(self, small_cache):
        c = small_cache
        assert c.pos_count == 0
        assert c.recent_count == 0
        assert c.level_count[0].item() == 0
        assert c.total_slots == 0
        assert c.total_tokens_covered == 0

    def test_buffers_non_persistent(self, small_cache):
        """All cache buffers should be non-persistent (not in state_dict)."""
        sd = small_cache.state_dict()
        assert len(sd) == 0, f"Unexpected persistent keys: {list(sd.keys())}"


# ---------------------------------------------------------------------------
# _compact_tokens tests
# ---------------------------------------------------------------------------

class TestCompactTokens:
    def test_mean_pooling(self):
        """_compact_tokens should mean-pool content and value."""
        B, G, n, D = 1, 1, 4, 3
        k = torch.arange(n * D, dtype=torch.float32).reshape(B, G, n, D)
        v = torch.arange(n * D, dtype=torch.float32).reshape(B, G, n, D) * 10

        k_entry, v_entry, w_entry = LogStructuredKVCache._compact_tokens(k, v)

        assert k_entry.shape == (B, G, 1, D)
        assert v_entry.shape == (B, G, 1, D)
        assert w_entry.shape == (B, G, 1)
        torch.testing.assert_close(k_entry[0, 0, 0], k[0, 0].mean(dim=0))
        torch.testing.assert_close(v_entry[0, 0, 0], v[0, 0].mean(dim=0))
        assert w_entry[0, 0, 0].item() == float(n)

    def test_partial_chunk(self):
        """_compact_tokens with fewer tokens than expected should use actual count."""
        B, G, D = 1, 1, 3
        k = torch.randn(B, G, 2, D)
        v = torch.randn(B, G, 2, D)

        k_entry, v_entry, w_entry = LogStructuredKVCache._compact_tokens(k, v)

        assert w_entry[0, 0, 0].item() == 2.0
        torch.testing.assert_close(k_entry[0, 0, 0], k[0, 0].mean(dim=0))


# ---------------------------------------------------------------------------
# compact (merge two B-slot blocks) tests
# ---------------------------------------------------------------------------

class TestCompact:
    def test_merge_weighted_average(self):
        """compact should produce weighted average of adjacent pairs."""
        B_slots = 4
        B, G, D = 1, 1, 2

        k1 = torch.ones(B, G, B_slots, D)
        v1 = torch.ones(B, G, B_slots, D) * 2
        w1 = torch.full((B, G, B_slots), 4.0)

        k2 = torch.ones(B, G, B_slots, D) * 3
        v2 = torch.ones(B, G, B_slots, D) * 6
        w2 = torch.full((B, G, B_slots), 4.0)

        k_out, v_out, w_out = LogStructuredKVCache.compact(k1, v1, w1, k2, v2, w2)

        assert k_out.shape == (B, G, B_slots, D)
        assert v_out.shape == (B, G, B_slots, D)
        assert w_out.shape == (B, G, B_slots)
        # equal weights -> alpha=0.5 -> 0.5*1 + 0.5*3 = 2
        torch.testing.assert_close(k_out, torch.full_like(k_out, 2.0))
        torch.testing.assert_close(v_out, torch.full_like(v_out, 4.0))
        torch.testing.assert_close(w_out, torch.full_like(w_out, 8.0))

    def test_merge_unequal_weights(self):
        """compact with unequal weights should use alpha = wa/(wa+wb)."""
        B_slots = 2
        B, G, D = 1, 1, 1

        k1 = torch.tensor([[[[1.0], [5.0]]]])  # (1,1,2,1)
        v1 = torch.tensor([[[[2.0], [6.0]]]])
        w1 = torch.tensor([[[1.0, 3.0]]])  # (1,1,2)

        k2 = torch.tensor([[[[3.0], [7.0]]]])
        v2 = torch.tensor([[[[4.0], [8.0]]]])
        w2 = torch.tensor([[[3.0, 1.0]]])

        k_out, v_out, w_out = LogStructuredKVCache.compact(k1, v1, w1, k2, v2, w2)

        # pair 0: wa=1, wb=3, alpha=0.25 -> 0.25*1 + 0.75*3 = 2.5
        torch.testing.assert_close(k_out[0, 0, 0, 0], torch.tensor(2.5))
        # pair 1: wa=3, wb=1, alpha=0.75 -> 0.75*5 + 0.25*7 = 5.5
        torch.testing.assert_close(k_out[0, 0, 1, 0], torch.tensor(5.5))
        torch.testing.assert_close(w_out, torch.tensor([[[4.0, 4.0]]]))


# ---------------------------------------------------------------------------
# _flush_recent / ingest_chunk tests
# ---------------------------------------------------------------------------

class TestIngest:
    def test_single_ingest(self, small_cache):
        """Ingesting one 2-token chunk should fill level 0 with 1 entry."""
        c = small_cache
        B, G, D = 1, 2, 6
        v_dim = 8
        d_pos = 2

        kc = torch.randn(B, G, 2, D)
        kv = torch.randn(B, G, 2, v_dim)
        kp = torch.randn(B, G, 2, d_pos)

        c.ingest_chunk(kc, kv, kp)

        assert c.level_count[0].item() == 1
        assert c.pos_count == 2
        assert c.recent_count == 0
        assert c.total_slots == 1  # level 0 only
        assert c.total_tokens_covered == 2

    def test_ingest_fills_accumulator_and_carries(self, small_cache):
        """Ingesting B chunks should fill level 0 and carry to level 1."""
        c = small_cache
        B_batch, G, D = 1, 2, 6
        v_dim = 8
        d_pos = 2

        for i in range(c.B):  # 4 ingests
            kc = torch.randn(B_batch, G, 2, D)
            kv = torch.randn(B_batch, G, 2, v_dim)
            kp = torch.randn(B_batch, G, 2, d_pos)
            c.ingest_chunk(kc, kv, kp)

        # After B=4 ingests, level 0 should carry to level 1
        assert c.level_count[0].item() == 0
        assert c.level_count[1].item() > 0
        assert c.level_count[2].item() == 0
        assert c.pos_count == c.B * 2  # 4 * 2 = 8
        assert c.total_slots == c.B  # level 1 has B slots

    def test_ingest_triggers_binary_carry(self):
        """Ingesting enough chunks to fill multiple levels."""
        B_slots = 2
        max_seq = 64
        d_pos = 2
        D = 4
        v_dim = 4

        c = LogStructuredKVCache(
            (1, 1, max_seq, D), (1, 1, max_seq, v_dim),
            d_pos=d_pos, B=B_slots,
        )

        # Fill level 0 (B=2 ingests), then fill again -> carry to level 1
        for i in range(B_slots):
            c.ingest_chunk(
                torch.randn(1, 1, 2, D),
                torch.randn(1, 1, 2, v_dim),
                torch.randn(1, 1, 2, d_pos),
            )
        assert c.level_count[1].item() > 0
        assert c.level_count[0].item() == 0

        for i in range(B_slots):
            c.ingest_chunk(
                torch.randn(1, 1, 2, D),
                torch.randn(1, 1, 2, v_dim),
                torch.randn(1, 1, 2, d_pos),
            )
        # Now level 1 should be cleared, level 2 should be occupied
        assert c.level_count[1].item() == 0
        assert c.level_count[2].item() > 0
        assert c.level_count[0].item() == 0


# ---------------------------------------------------------------------------
# disabled forward API / explicit update tests
# ---------------------------------------------------------------------------

class TestDisabledForwardApi:
    @pytest.mark.parametrize("T", (1, 3))
    def test_forward_is_disabled(self, small_cache, T):
        """Direct cache calls must fail before mutating state."""
        c = small_cache
        B, G = 1, 2
        head_size = 8

        k = torch.randn(B, G, T, head_size)
        v = torch.randn(B, G, T, head_size)
        input_pos = torch.arange(T)

        with pytest.raises(RuntimeError, match="disabled/deprecated"):
            c(input_pos, k, v)

        assert c.pos_count == 0
        assert c.recent_count == 0


class TestExplicitCacheUpdates:
    def test_add_recent_compacts_into_hierarchy(self, small_cache):
        """Explicit commits should build compact levels and recent state."""
        c = small_cache
        B, G = 1, 2
        head_size = 8
        T = 10

        k = torch.randn(B, G, T, head_size)
        v = torch.randn(B, G, T, head_size)

        add_full_kv_in_chunks(c, k, v)

        # 10 tokens -> 5 flushes. B=4 -> 1 carry (4 entries) + 1 in level 0.
        # But last chunk (tokens 9-10) stays in recent buffer.
        # So: 4 flushes -> 1 carry to level 1. Recent has 2 tokens.
        # Tokens: [0,1] flush, [2,3] flush, [4,5] flush, [6,7] flush -> 4 compact entries
        # -> carry to level 1. [8,9] stays in recent (last chunk, not flushed).
        assert c.level_count[1].item() > 0
        assert c.recent_count == 2  # last chunk stays
        assert c.pos_count == T

    def test_add_recent_stores_single_token(self, small_cache):
        """Single-token explicit commits should store token in recent buffer."""
        c = small_cache
        B, G = 1, 2
        head_size = 8

        k = torch.randn(B, G, 1, head_size)
        v = torch.randn(B, G, 1, head_size)

        add_full_kv_in_chunks(c, k, v, chunk_size=1)

        assert c.recent_count == 1
        assert c.pos_count == 1

    def test_add_recent_after_prefill(self, small_cache):
        """Explicit commits after an existing prompt should extend recent state."""
        c = small_cache
        B, G = 1, 2
        head_size = 8
        T_prefill = 9  # leaves 1 token in recent (9 % 2 = 1)

        k_pre = torch.randn(B, G, T_prefill, head_size)
        v_pre = torch.randn(B, G, T_prefill, head_size)
        add_full_kv_in_chunks(c, k_pre, v_pre)

        recent_after_prefill = c.recent_count
        pos_after_prefill = c.pos_count

        k_dec = torch.randn(B, G, 1, head_size)
        v_dec = torch.randn(B, G, 1, head_size)
        add_full_kv_in_chunks(c, k_dec, v_dec, chunk_size=1)

        assert c.pos_count == pos_after_prefill + 1
        assert c.recent_count == recent_after_prefill + 1

    def test_add_recent_splits_chunk_after_single_token(self, small_cache):
        """A 2-token commit after one recent token should compact the first pair and keep order."""
        c = small_cache
        B, G = 1, 2
        head_size = 8

        k_first = torch.randn(B, G, 1, head_size)
        v_first = torch.randn(B, G, 1, head_size)
        add_full_kv_in_chunks(c, k_first, v_first, chunk_size=1)

        k_pair = torch.randn(B, G, 2, head_size)
        v_pair = torch.randn(B, G, 2, head_size)
        add_full_kv_in_chunks(c, k_pair, v_pair, chunk_size=2)

        state = c.get_attention_state()
        _, _, _, compact_pos_keys, _, _, recent_pos_keys = state

        assert c.pos_count == 3
        assert c.level_count[0].item() == 1
        assert c.recent_count == 1
        torch.testing.assert_close(compact_pos_keys, c.pos_keys[:, :, :2, :])
        torch.testing.assert_close(recent_pos_keys, c.pos_keys[:, :, 2:3, :])

    def test_add_recent_flushes_when_recent_overflows(self, small_cache):
        """When recent buffer overflows, the oldest 2 tokens should compact."""
        c = small_cache
        B, G = 1, 2
        head_size = 8
        t = 2  # compaction chunk size (fixed)

        for i in range(t):
            k = torch.randn(B, G, 1, head_size)
            v = torch.randn(B, G, 1, head_size)
            add_full_kv_in_chunks(c, k, v, chunk_size=1)

        assert c.recent_count == t
        assert c.level_count[0].item() == 0
        assert c.pos_count == t

        k = torch.randn(B, G, 1, head_size)
        v = torch.randn(B, G, 1, head_size)
        add_full_kv_in_chunks(c, k, v, chunk_size=1)

        assert c.recent_count == 1
        assert c.level_count[0].item() == 1
        assert c.pos_count == t + 1


# ---------------------------------------------------------------------------
# get_attention_state tests
# ---------------------------------------------------------------------------

class TestGetAttentionState:
    def test_empty_cache_state(self, small_cache):
        """Empty cache should return zero-size tensors."""
        c = small_cache
        state = c.get_attention_state()
        compact_k, compact_v, compact_w, compact_pos_keys, \
            recent_content, recent_values, recent_pos_keys = state

        assert compact_k.size(2) == 0
        assert compact_v.size(2) == 0
        assert compact_w.size(2) == 0
        assert compact_pos_keys.size(2) == 0
        assert recent_content.size(2) == 0
        assert recent_values.size(2) == 0
        assert recent_pos_keys.size(2) == 0

    def test_state_after_ingest(self, small_cache):
        """After ingest, compact slots should be in level 0, recent should be empty."""
        c = small_cache
        B, G = 1, 2
        D, v_dim, d_pos = 6, 8, 2

        c.ingest_chunk(
            torch.randn(B, G, 2, D),
            torch.randn(B, G, 2, v_dim),
            torch.randn(B, G, 2, d_pos),
        )

        state = c.get_attention_state()
        compact_k, compact_v, compact_w, compact_pos_keys, \
            recent_content, recent_values, recent_pos_keys = state

        assert compact_k.size(2) == 1  # 1 slot in level 0
        assert compact_w[0, 0, 0].item() == 2.0
        assert compact_pos_keys.size(2) == 2  # 2 position keys
        assert recent_content.size(2) == 0  # nothing in recent (ingest bypasses recent)

    def test_state_after_prefill(self, small_cache):
        """After prefill, compact slots + recent tokens should be present."""
        c = small_cache
        B, G = 1, 2
        head_size = 8
        T = 9  # 4 flushes (8 tokens) + 1 in recent

        k = torch.randn(B, G, T, head_size)
        v = torch.randn(B, G, T, head_size)
        add_full_kv_in_chunks(c, k, v)

        state = c.get_attention_state()
        compact_k, compact_v, compact_w, compact_pos_keys, \
            recent_content, recent_values, recent_pos_keys = state

        # 4 compact entries in level 0 (after carry), 1 token in recent
        assert compact_k.size(2) == c.B  # level 0 has B slots
        assert compact_pos_keys.size(2) == T - c.recent_count  # 8 compressed tokens
        assert recent_content.size(2) == c.recent_count  # 1 recent token

    def test_pos_keys_alignment(self, small_cache):
        """compact_pos_keys should cover compressed tokens, recent_pos_keys the rest."""
        c = small_cache
        B, G = 1, 2
        head_size = 8
        T = 5  # 2 flushes (4 tokens) + 1 in recent

        k = torch.randn(B, G, T, head_size)
        v = torch.randn(B, G, T, head_size)
        add_full_kv_in_chunks(c, k, v)

        state = c.get_attention_state()
        _, _, _, compact_pos_keys, _, _, recent_pos_keys = state

        # compact_pos_keys should be the first 4 tokens' position keys
        expected_compact = c.pos_keys[:, :, :4, :]
        torch.testing.assert_close(compact_pos_keys, expected_compact)

        # recent_pos_keys should be the 5th token's position key
        expected_recent = c.pos_keys[:, :, 4:5, :]
        torch.testing.assert_close(recent_pos_keys, expected_recent)


# ---------------------------------------------------------------------------
# log_kv_decoupled_attention tests
# ---------------------------------------------------------------------------

class TestDecoupledAttention:
    @staticmethod
    def _make_state(n_compact, n_recent, B=1, G=2, content_dim=6, v_dim=8, d_pos=2):
        """Helper to build a cache_state tuple."""
        compact_k = torch.randn(B, G, n_compact, content_dim)
        compact_v = torch.randn(B, G, n_compact, v_dim)
        # Each compact slot covers 2 tokens
        compact_w = torch.full((B, G, n_compact), 2.0)
        compact_pos_keys = torch.randn(B, G, n_compact * 2, d_pos)

        recent_content = torch.randn(B, G, n_recent, content_dim)
        recent_values = torch.randn(B, G, n_recent, v_dim)
        recent_pos_keys = torch.randn(B, G, n_recent, d_pos)

        return (compact_k, compact_v, compact_w, compact_pos_keys,
                recent_content, recent_values, recent_pos_keys)

    def test_output_shape(self):
        """Output should have shape (B, nh, T_q, v_dim)."""
        B, nh, T_q = 1, 4, 1
        content_dim, d_pos, v_dim = 6, 2, 8

        state = self._make_state(n_compact=3, n_recent=2)
        q_content = torch.randn(B, nh, T_q, content_dim)
        q_pos = torch.randn(B, nh, T_q, d_pos)

        out = log_kv_decoupled_attention(q_content, q_pos, state, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)

    def test_gqa_expansion(self):
        """When nh != n_groups, k/v should be expanded via repeat_interleave."""
        B, nh, G = 1, 4, 2  # nh=4, G=2 -> repeat factor 2
        T_q = 1
        content_dim, d_pos, v_dim = 6, 2, 8

        state = self._make_state(n_compact=2, n_recent=1, G=G)
        q_content = torch.randn(B, nh, T_q, content_dim)
        q_pos = torch.randn(B, nh, T_q, d_pos)

        out = log_kv_decoupled_attention(q_content, q_pos, state, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)
        assert not torch.isnan(out).any()

    def test_empty_state(self):
        """With no tokens to attend to, output should be zeros."""
        B, nh, T_q = 1, 4, 1
        content_dim, d_pos, v_dim = 6, 2, 8

        state = self._make_state(n_compact=0, n_recent=0)
        q_content = torch.randn(B, nh, T_q, content_dim)
        q_pos = torch.randn(B, nh, T_q, d_pos)

        out = log_kv_decoupled_attention(q_content, q_pos, state, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_only_recent(self):
        """With only recent tokens (no compact), should still work."""
        B, nh, G = 1, 2, 2
        T_q = 1
        content_dim, d_pos, v_dim = 6, 2, 8

        state = self._make_state(n_compact=0, n_recent=3, G=G)
        q_content = torch.randn(B, nh, T_q, content_dim)
        q_pos = torch.randn(B, nh, T_q, d_pos)

        out = log_kv_decoupled_attention(q_content, q_pos, state, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)
        assert not torch.isnan(out).any()

    def test_mask_applied(self):
        """Mask should prevent attention to certain tokens."""
        B, nh, G = 1, 2, 2
        T_q = 3
        content_dim, d_pos, v_dim = 6, 2, 8

        state = self._make_state(n_compact=1, n_recent=2, G=G)
        # n_compact_tokens = 1*2 = 2, n_recent = 2, n_total = 4
        q_content = torch.randn(B, nh, T_q, content_dim)
        q_pos = torch.randn(B, nh, T_q, d_pos)

        # Causal mask: query i can only attend to tokens 0..i
        n_total = 2 + 2  # compact_tokens + recent
        mask = torch.ones(T_q, n_total, dtype=torch.bool)
        for i in range(T_q):
            mask[i, i + 1:] = False

        out = log_kv_decoupled_attention(q_content, q_pos, state, scale=0.1, mask=mask)

        assert out.shape == (B, nh, T_q, v_dim)
        assert not torch.isnan(out).any()

    def test_scale_applied(self):
        """Different scales should produce different outputs."""
        B, nh, G = 1, 2, 2
        T_q = 1
        content_dim, d_pos, v_dim = 6, 2, 8

        state = self._make_state(n_compact=2, n_recent=1, G=G)
        q_content = torch.randn(B, nh, T_q, content_dim)
        q_pos = torch.randn(B, nh, T_q, d_pos)

        out1 = log_kv_decoupled_attention(q_content, q_pos, state, scale=0.01)
        out2 = log_kv_decoupled_attention(q_content, q_pos, state, scale=1.0)

        # Different scales should produce different attention distributions
        assert not torch.allclose(out1, out2)

    def test_equivalent_to_standard_attention_when_all_weight_1(self):
        """When all slots have weight=1 (no compaction), decoupled attention
        should match standard scaled dot-product attention on the combined
        content+position keys."""
        B, nh, G = 1, 2, 2
        T_q = 1
        content_dim, d_pos, v_dim = 4, 4, 8
        head_size = content_dim + d_pos  # 8
        scale = 1.0 / math.sqrt(head_size)

        # No compact slots, 3 recent tokens (each weight=1, no compaction)
        state = self._make_state(n_compact=0, n_recent=3, G=G,
                                  content_dim=content_dim, v_dim=v_dim, d_pos=d_pos)

        q_content = torch.randn(B, nh, T_q, content_dim)
        q_pos = torch.randn(B, nh, T_q, d_pos)
        q_full = torch.cat([q_pos, q_content], dim=-1)  # (B, nh, T_q, head_size)

        # For standard attention: k_full = cat(pos, content), v = values
        recent_content = state[4]  # (B, G, 3, content_dim)
        recent_pos_keys = state[6]  # (B, G, 3, d_pos)
        recent_values = state[5]  # (B, G, 3, v_dim)

        # Expand GQA for standard attention
        q_per_kv = nh // G
        k_full = torch.cat([recent_pos_keys, recent_content], dim=-1)  # (B, G, 3, head_size)
        k_full = k_full.repeat_interleave(q_per_kv, dim=1)  # (B, nh, 3, head_size)
        v_expanded = recent_values.repeat_interleave(q_per_kv, dim=1)  # (B, nh, 3, v_dim)

        # Standard attention
        scores = torch.matmul(q_full, k_full.mT) * scale  # (B, nh, T_q, 3)
        attn_std = torch.softmax(scores, dim=-1)
        out_std = torch.matmul(attn_std, v_expanded)  # (B, nh, T_q, v_dim)

        # Decoupled attention
        out_decoupled = log_kv_decoupled_attention(
            q_content, q_pos, state, scale=scale
        )

        torch.testing.assert_close(out_decoupled, out_std, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# reset_parameters tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_state(self, small_cache):
        c = small_cache
        # Put some data in
        c.ingest_chunk(
            torch.randn(1, 2, 2, 6),
            torch.randn(1, 2, 2, 8),
            torch.randn(1, 2, 2, 2),
        )
        assert c.pos_count > 0
        assert c.level_count[0].item() > 0

        c.reset_parameters()

        assert c.pos_count == 0
        assert c.recent_count == 0
        assert c.level_count.sum() == 0
        assert c.total_slots == 0
        assert c.total_tokens_covered == 0
        assert c.pos_keys.abs().sum() == 0


# ---------------------------------------------------------------------------
# Integration: explicit commit + get_attention_state round-trip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_explicit_commit_roundtrip(self):
        """End-to-end: commit prompt/chunks and verify state shapes throughout."""
        B_batch, G = 1, 2
        head_size = 8
        d_pos = 2
        content_dim = head_size - d_pos  # 6
        v_dim = head_size
        max_seq = 64

        c = LogStructuredKVCache(
            (B_batch, G, max_seq, content_dim),
            (B_batch, G, max_seq, v_dim),
            d_pos=d_pos, B=4,
        )

        # Prefill 7 tokens (3 flushes of 2 + 1 in recent)
        T_prefill = 7
        k = torch.randn(B_batch, G, T_prefill, head_size)
        v = torch.randn(B_batch, G, T_prefill, head_size)
        add_full_kv_in_chunks(c, k, v)

        # Verify state after prefill
        state = c.get_attention_state()
        compact_k, compact_v, compact_w, compact_pos_keys, \
            recent_content, recent_values, recent_pos_keys = state
        assert c.pos_count == T_prefill
        assert c.recent_count == 1  # 7 % 2 = 1

        # Decode 3 more tokens
        for i in range(3):
            k_dec = torch.randn(B_batch, G, 1, head_size)
            v_dec = torch.randn(B_batch, G, 1, head_size)
            add_full_kv_in_chunks(c, k_dec, v_dec, chunk_size=1)

        # After 3 more decode steps: recent_count should cycle
        # Start: 1 -> 2 (flush) -> 0 + 1 acc -> 1 -> 2 (flush) -> 0 + 2 acc
        assert c.pos_count == T_prefill + 3

        # Verify we can compute attention with the final state
        nh = 4
        state = c.get_attention_state()
        q_content = torch.randn(B_batch, nh, 1, content_dim)
        q_pos = torch.randn(B_batch, nh, 1, d_pos)
        out = log_kv_decoupled_attention(q_content, q_pos, state, scale=0.1)
        assert out.shape == (B_batch, nh, 1, v_dim)
        assert not torch.isnan(out).any()

    def test_capacity_for_4k(self):
        """Verify the cache can handle 4K tokens without overflow (B=1024, 2:1 compaction)."""
        B_batch, G = 1, 1
        head_size = 8
        d_pos = 2
        content_dim = head_size - d_pos
        v_dim = head_size
        max_seq = 4096

        c = LogStructuredKVCache(
            (B_batch, G, max_seq, content_dim),
            (B_batch, G, max_seq, v_dim),
            d_pos=d_pos, B=1024,
        )

        # Simulate ingest of all 4096/2 = 2048 compact entries.
        # This fills level 0 twice and carries through level 1, with no overflow.
        for i in range(2048):
            c.ingest_chunk(
                torch.randn(B_batch, G, 2, content_dim),
                torch.randn(B_batch, G, 2, v_dim),
                torch.randn(B_batch, G, 2, d_pos),
            )

        assert c.level_count[2].item() > 0
        assert c.level_count[0].item() == 0
        assert c.level_count[1].item() == 0
        assert c.pos_count == 4096


# ---------------------------------------------------------------------------
# Integration: attention-layer LogKV streaming boundaries
# ---------------------------------------------------------------------------

class TestAttentionStreamingBoundaries:
    @staticmethod
    def _make_attention() -> CausalSelfAttention:
        config = Config(
            block_size=16,
            padded_vocab_size=16,
            n_layer=1,
            n_head=2,
            n_query_groups=2,
            n_embd=8,
            rotary_percentage=0.5,
        )
        attn = CausalSelfAttention(config, block_idx=0)
        attn.kv_cache = attn.build_log_kv_cache(
            batch_size=1,
            max_seq_length=16,
            B=4,
            recent_size=2,
        )
        return attn

    @staticmethod
    def _random_qkv(attn: CausalSelfAttention, T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = attn.config
        q = torch.randn(1, config.n_head, T, config.head_size)
        k = torch.randn(1, config.n_query_groups, T, config.head_size)
        v = torch.randn(1, config.n_query_groups, T, config.head_size)
        return q, k, v

    def test_odd_prefill_tail_pairs_with_first_decode(self):
        """Inference prefill should defer a trailing singleton until the next streamed token."""
        attn = self._make_attention()
        cache = attn.kv_cache
        assert isinstance(cache, LogStructuredKVCache)

        q, k, v = self._random_qkv(attn, T=3)
        out = attn._log_kv_training_forward(q, k, v, B=1, T=3, reset_cache=True, defer_last_single=True)

        assert out.shape == (1, 3, attn.config.n_embd)
        assert cache.pos_count == 2
        assert cache.recent_count == 2
        assert attn._log_kv_pending is not None

        q, k, v = self._random_qkv(attn, T=1)
        out = attn._log_kv_training_forward(q, k, v, B=1, T=1, reset_cache=False, defer_last_single=True)

        assert out.shape == (1, 1, attn.config.n_embd)
        assert attn._log_kv_pending is None
        assert cache.pos_count == 4
        assert cache.level_count[0].item() == 1
        assert cache.recent_count == 2

    def test_streaming_chunk_consumes_pending_before_new_pairs(self):
        """A later multi-token inference call should consume pending before grouping new tokens."""
        attn = self._make_attention()
        cache = attn.kv_cache
        assert isinstance(cache, LogStructuredKVCache)

        q, k, v = self._random_qkv(attn, T=3)
        attn._log_kv_training_forward(q, k, v, B=1, T=3, reset_cache=True, defer_last_single=True)
        assert attn._log_kv_pending is not None

        q, k, v = self._random_qkv(attn, T=3)
        attn._log_kv_training_forward(q, k, v, B=1, T=3, reset_cache=False, defer_last_single=True)

        assert attn._log_kv_pending is None
        assert cache.pos_count == 6
        assert cache.level_count[0].item() == 2
        assert cache.recent_count == 2


# ---------------------------------------------------------------------------
# Integration: GPT.forward LogKV mask handling
# ---------------------------------------------------------------------------

class TestGPTForwardLogKVMask:
    def test_all_log_kv_blocks_do_not_require_mask_cache(self):
        config = Config(
            block_size=16,
            padded_vocab_size=16,
            n_layer=1,
            n_head=2,
            n_query_groups=2,
            n_embd=8,
            rotary_percentage=0.5,
        )
        model = GPT(config)
        model.set_log_kv_cache(batch_size=1, max_seq_length=16, B=4, recent_size=2)
        model.mask_cache = None

        idx = torch.randint(0, config.padded_vocab_size, (1, 3))
        input_pos = torch.arange(3)

        logits = model(idx, input_pos)

        assert logits.shape == (1, 3, config.padded_vocab_size)


# ---------------------------------------------------------------------------
# Integration: GPT.forward LogKV input_pos contract
# ---------------------------------------------------------------------------

class TestGPTForwardLogKVInputPos:
    @staticmethod
    def _make_model(batch_size: int = 1) -> GPT:
        config = Config(
            block_size=16,
            padded_vocab_size=16,
            n_layer=1,
            n_head=2,
            n_query_groups=2,
            n_embd=8,
            rotary_percentage=0.5,
        )
        model = GPT(config)
        model.set_log_kv_cache(batch_size=batch_size, max_seq_length=16, B=4, recent_size=2)
        return model

    def test_rejects_non_contiguous_input_pos(self):
        model = self._make_model()
        idx = torch.randint(0, model.config.padded_vocab_size, (1, 3))

        with pytest.raises(ValueError, match="append-only contiguous input_pos"):
            model(idx, torch.tensor([0, 2, 3]))

    def test_pending_token_counts_toward_next_expected_input_pos(self):
        model = self._make_model()

        idx = torch.randint(0, model.config.padded_vocab_size, (1, 3))
        model(idx, torch.arange(3))

        attn = model.transformer.h[0].attn
        cache = attn.kv_cache
        assert isinstance(cache, LogStructuredKVCache)
        assert cache.pos_count == 2
        assert attn._log_kv_pending is not None

        idx_next = torch.randint(0, model.config.padded_vocab_size, (1, 1))
        with pytest.raises(ValueError, match="Expected 3..3"):
            model(idx_next, torch.tensor([2]))

        logits = model(idx_next, torch.tensor([3]))
        assert logits.shape == (1, 1, model.config.padded_vocab_size)

    def test_rejects_per_sample_batched_input_pos(self):
        model = self._make_model(batch_size=2)
        idx = torch.randint(0, model.config.padded_vocab_size, (2, 3))
        input_pos = torch.tensor([[0, 1, 2], [1, 2, 3]])

        with pytest.raises(ValueError, match="per-sample input_pos"):
            model(idx, input_pos)


# ---------------------------------------------------------------------------
# Regression: LogKV cache dtype reconciliation (bf16 model, fp32-default cache)
# ---------------------------------------------------------------------------

class TestLogKVDtypeReconcile:
    """A bf16 model must run LogKV inference even when the cache was allocated at
    the process-default dtype (fp32) — the eval path calls set_log_kv_cache()
    and the model is cast to bf16 separately. Without reconciliation the second
    prefill chunk would cat/matmul fp32 buffers against bf16 activations."""

    @staticmethod
    def _make_bf16_model() -> GPT:
        config = Config(
            block_size=16,
            padded_vocab_size=16,
            n_layer=1,
            n_head=2,
            n_query_groups=2,
            n_embd=8,
            rotary_percentage=0.5,
        )
        model = GPT(config).to(torch.bfloat16)
        # No dtype passed -> buffers use the process default (fp32 in the test
        # suite), deliberately mismatched against the bf16 model.
        model.set_log_kv_cache(batch_size=1, max_seq_length=16, B=4, recent_size=2)
        return model

    def test_bf16_inference_with_model_dtype_cache(self):
        """Cache now defaults to the model's dtype (bf16), not the process default.
        This avoids dtype mismatches that previously required runtime reconciliation."""
        model = self._make_bf16_model()
        cache = model.transformer.h[0].attn.kv_cache
        assert isinstance(cache, LogStructuredKVCache)
        # set_log_kv_cache now defaults dtype to next(model.parameters()).dtype
        assert cache.recent_content.dtype == torch.bfloat16  # allocated at model dtype

        # Prefill 5 tokens (crosses several 2-token chunks) then a decode step.
        idx = torch.randint(0, model.config.padded_vocab_size, (1, 5))
        logits = model(idx, torch.arange(5))
        assert logits.shape == (1, 5, model.config.padded_vocab_size)
        assert logits.dtype == torch.bfloat16

        # Buffers were reconciled to the activation dtype.
        assert cache.recent_content.dtype == torch.bfloat16
        assert cache.pos_keys.dtype == torch.bfloat16
        assert getattr(cache, "level_k_0").dtype == torch.bfloat16

        idx_next = torch.randint(0, model.config.padded_vocab_size, (1, 1))
        logits = model(idx_next, torch.tensor([5]))
        assert logits.shape == (1, 1, model.config.padded_vocab_size)
        assert logits.dtype == torch.bfloat16

    def test_convert_dtype_is_idempotent_noop(self):
        model = self._make_bf16_model()
        cache = model.transformer.h[0].attn.kv_cache

        cache._convert_dtype(torch.bfloat16)
        buf_before = cache.recent_content
        cache._convert_dtype(torch.bfloat16)  # already bf16 -> must be a no-op
        assert cache.recent_content is buf_before  # no new allocation
        assert cache.level_count.dtype == torch.long  # counts stay integer


# ---------------------------------------------------------------------------
# Integration: LogKV inference equals dense attention within the recent window
# ---------------------------------------------------------------------------

class TestLogKVMatchesDense:
    """Within recent_size no compaction occurs, so LogKV inference must reproduce
    dense causal attention exactly. This validates the prefill fast path and
    underpins routing loglikelihood scoring through LogKV (eval.use_log_kv)."""

    @staticmethod
    def _make_model() -> GPT:
        config = Config(
            block_size=32,
            padded_vocab_size=16,
            n_layer=2,
            n_head=4,
            n_query_groups=2,
            n_embd=16,
            rotary_percentage=0.5,
        )
        torch.manual_seed(0)
        model = GPT(config)
        model.eval()
        return model

    @pytest.mark.parametrize("seq_len", (1, 2, 6, 7))
    def test_prefill_within_recent_matches_dense(self, seq_len):
        model = self._make_model()
        idx = torch.randint(0, model.config.padded_vocab_size, (1, seq_len))

        with torch.no_grad():
            dense = model(idx)  # dense causal attention (no cache)
            # recent_size >= seq_len -> entire prefill is exact (fast path, no compaction)
            model.set_log_kv_cache(batch_size=1, max_seq_length=seq_len, B=4, recent_size=16)
            logkv = model(idx, input_pos=torch.arange(seq_len))
            model.clear_kv_cache()

        torch.testing.assert_close(logkv, dense, atol=1e-4, rtol=1e-4)

    def test_streamed_decode_within_recent_matches_dense(self):
        """Prefill + one-token decodes must match a single dense forward over the
        whole sequence, as long as everything stays within recent_size."""
        model = self._make_model()
        full = torch.randint(0, model.config.padded_vocab_size, (1, 6))

        with torch.no_grad():
            dense = model(full)

            model.set_log_kv_cache(batch_size=1, max_seq_length=6, B=4, recent_size=16)
            out = []
            out.append(model(full[:, :4], input_pos=torch.arange(4)))  # prefill
            for pos in range(4, 6):  # decode token by token
                out.append(model(full[:, pos:pos + 1], input_pos=torch.tensor([pos])))
            model.clear_kv_cache()

        streamed = torch.cat(out, dim=1)
        torch.testing.assert_close(streamed, dense, atol=1e-4, rtol=1e-4)
