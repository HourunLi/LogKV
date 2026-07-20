"""Tests for LogStructuredKVCache with merged-position slots (strict O(B·log N))."""

import math

import pytest
import torch

from litgpt.config import Config
from litgpt.log_kv_cache import LogStructuredKVCache, append_exact_tokens, log_kv_slot_attention
from litgpt.model import CausalSelfAttention, GPT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cache():
    """A small cache suitable for unit testing the data structure.

    k_dim=8 (full post-RoPE key width), v_dim=8
    B=4 (slots per level), 2:1 compaction
    max_seq_length=64 -> max_levels = max(2, ceil(log2(65/8))+1) = 5
    """
    batch_size = 1
    n_groups = 2
    max_seq_length = 64
    k_dim = 8
    v_dim = 8
    B = 4

    k_shape = (batch_size, n_groups, max_seq_length, k_dim)
    v_shape = (batch_size, n_groups, max_seq_length, v_dim)

    cache = LogStructuredKVCache(
        k_shape, v_shape, B=B,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    return cache


def add_full_kv_in_chunks(cache: LogStructuredKVCache, k: torch.Tensor, v: torch.Tensor, chunk_size: int = 2) -> None:
    """Commit full post-RoPE K/V tensors through the supported explicit cache API."""
    for start in range(0, k.size(2), chunk_size):
        end = min(start + chunk_size, k.size(2))
        cache.add_recent(k[:, :, start:end, :], v[:, :, start:end, :])


def assert_cache_states_bit_identical(a: LogStructuredKVCache, b: LogStructuredKVCache) -> None:
    """The full cache state (counters, recent window, all levels) must match bitwise."""
    assert a.token_count == b.token_count
    assert a.recent_count == b.recent_count
    rc = a.recent_count
    assert torch.equal(a.recent_k[:, :, :rc], b.recent_k[:, :, :rc])
    assert torch.equal(a.recent_v[:, :, :rc], b.recent_v[:, :, :rc])
    assert torch.equal(a.level_count, b.level_count)
    for ell in range(a.max_levels):
        for name in ("level_k_", "level_v_", "level_w_"):
            assert torch.equal(getattr(a, f"{name}{ell}"), getattr(b, f"{name}{ell}")), f"{name}{ell} differ"


# ---------------------------------------------------------------------------
# __init__ / structure tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_basic_attributes(self, small_cache):
        c = small_cache
        assert c.B == 4
        assert c.k_dim == 8
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
                B=1024,
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
        assert c.token_count == 0
        assert c.recent_count == 0
        assert c.level_count[0].item() == 0
        assert c.total_slots == 0
        assert c.total_tokens_covered == 0

    def test_buffers_non_persistent(self, small_cache):
        """All cache buffers should be non-persistent (not in state_dict)."""
        sd = small_cache.state_dict()
        assert len(sd) == 0, f"Unexpected persistent keys: {list(sd.keys())}"

    def test_memory_is_logarithmic_in_max_seq_length(self):
        """THE core complexity guarantee: total buffer storage must be
        O(recent_size + B * max_levels) — no term linear in max_seq_length.
        Closed form: recent_size*(k_dim+v_dim)*b*g
                     + max_levels * (B*(k_dim+v_dim+1)*b*g + 1)."""
        b, g, k_dim, v_dim, B, recent = 1, 2, 8, 8, 4, 8
        for max_seq in (1024, 65536, 1048576):
            c = LogStructuredKVCache(
                (b, g, max_seq, k_dim), (b, g, max_seq, v_dim),
                B=B, recent_size=recent,
            )
            total = sum(buf.numel() for buf in c.buffers())
            expected = (
                recent * (k_dim + v_dim) * b * g
                + c.max_levels * (B * (k_dim + v_dim + 1) * b * g + 1)
            )
            assert total == expected, (
                f"max_seq={max_seq}: buffer numel {total} != log-sized {expected} — "
                "an O(N) buffer has crept back in"
            )


# ---------------------------------------------------------------------------
# _compact_tokens tests
# ---------------------------------------------------------------------------

class TestCompactTokens:
    def test_mean_pooling(self):
        """_compact_tokens should mean-pool keys and values."""
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
        B, G, D, v_dim = 1, 2, 8, 8

        c.ingest_chunk(torch.randn(B, G, 2, D), torch.randn(B, G, 2, v_dim))

        assert c.level_count[0].item() == 1
        assert c.token_count == 2
        assert c.recent_count == 0
        assert c.total_slots == 1  # level 0 only
        assert c.total_tokens_covered == 2

    def test_ingest_fills_accumulator_and_carries(self, small_cache):
        """Ingesting B chunks should fill level 0 and carry to level 1."""
        c = small_cache
        B_batch, G, D, v_dim = 1, 2, 8, 8

        for i in range(c.B):  # 4 ingests
            c.ingest_chunk(torch.randn(B_batch, G, 2, D), torch.randn(B_batch, G, 2, v_dim))

        # After B=4 ingests, level 0 should carry to level 1
        assert c.level_count[0].item() == 0
        assert c.level_count[1].item() > 0
        assert c.level_count[2].item() == 0
        assert c.token_count == c.B * 2  # 4 * 2 = 8
        assert c.total_slots == c.B  # level 1 has B slots

    def test_ingest_triggers_binary_carry(self):
        """Ingesting enough chunks to fill multiple levels."""
        B_slots = 2
        max_seq = 64
        D = 4
        v_dim = 4

        c = LogStructuredKVCache(
            (1, 1, max_seq, D), (1, 1, max_seq, v_dim),
            B=B_slots,
        )

        # Fill level 0 (B=2 ingests), then fill again -> carry to level 1
        for i in range(B_slots):
            c.ingest_chunk(torch.randn(1, 1, 2, D), torch.randn(1, 1, 2, v_dim))
        assert c.level_count[1].item() > 0
        assert c.level_count[0].item() == 0

        for i in range(B_slots):
            c.ingest_chunk(torch.randn(1, 1, 2, D), torch.randn(1, 1, 2, v_dim))
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
        k_dim = 8

        k = torch.randn(B, G, T, k_dim)
        v = torch.randn(B, G, T, k_dim)
        input_pos = torch.arange(T)

        with pytest.raises(RuntimeError, match="disabled/deprecated"):
            c(input_pos, k, v)

        assert c.token_count == 0
        assert c.recent_count == 0


class TestExplicitCacheUpdates:
    def test_add_recent_compacts_into_hierarchy(self, small_cache):
        """Explicit commits should build compact levels and recent state."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        T = 10

        k = torch.randn(B, G, T, k_dim)
        v = torch.randn(B, G, T, k_dim)

        add_full_kv_in_chunks(c, k, v)

        # 10 tokens -> 5 flushes. B=4 -> 1 carry (4 entries) + 1 in level 0.
        # But last chunk (tokens 9-10) stays in recent buffer.
        # So: 4 flushes -> 1 carry to level 1. Recent has 2 tokens.
        # Tokens: [0,1] flush, [2,3] flush, [4,5] flush, [6,7] flush -> 4 compact entries
        # -> carry to level 1. [8,9] stays in recent (last chunk, not flushed).
        assert c.level_count[1].item() > 0
        assert c.recent_count == 2  # last chunk stays
        assert c.token_count == T

    def test_add_recent_stores_single_token(self, small_cache):
        """Single-token explicit commits should store token in recent buffer."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8

        k = torch.randn(B, G, 1, k_dim)
        v = torch.randn(B, G, 1, k_dim)

        add_full_kv_in_chunks(c, k, v, chunk_size=1)

        assert c.recent_count == 1
        assert c.token_count == 1

    def test_add_recent_after_prefill(self, small_cache):
        """Explicit commits after an existing prompt should extend recent state."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        T_prefill = 9  # leaves 1 token in recent (9 % 2 = 1)

        k_pre = torch.randn(B, G, T_prefill, k_dim)
        v_pre = torch.randn(B, G, T_prefill, k_dim)
        add_full_kv_in_chunks(c, k_pre, v_pre)

        recent_after_prefill = c.recent_count
        tokens_after_prefill = c.token_count

        k_dec = torch.randn(B, G, 1, k_dim)
        v_dec = torch.randn(B, G, 1, k_dim)
        add_full_kv_in_chunks(c, k_dec, v_dec, chunk_size=1)

        assert c.token_count == tokens_after_prefill + 1
        assert c.recent_count == recent_after_prefill + 1

    def test_add_recent_splits_chunk_after_single_token(self, small_cache):
        """A 2-token commit after one recent token should compact the first pair and keep order."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8

        k = torch.randn(B, G, 3, k_dim)
        v = torch.randn(B, G, 3, k_dim)
        add_full_kv_in_chunks(c, k[:, :, :1, :], v[:, :, :1, :], chunk_size=1)
        add_full_kv_in_chunks(c, k[:, :, 1:, :], v[:, :, 1:, :], chunk_size=2)

        assert c.token_count == 3
        assert c.level_count[0].item() == 1
        assert c.recent_count == 1

        slot_k, slot_v, slot_w = c.get_attention_state()
        # slot 0 = mean of tokens 0-1, slot 1 = exact token 2 (w=1)
        torch.testing.assert_close(slot_k[:, :, 0, :], k[:, :, :2, :].mean(dim=2))
        torch.testing.assert_close(slot_k[:, :, 1, :], k[:, :, 2, :])
        torch.testing.assert_close(slot_w[0, 0], torch.tensor([2.0, 1.0]))

    def test_add_recent_flushes_when_recent_overflows(self, small_cache):
        """When recent buffer overflows, the oldest 2 tokens should compact."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        t = 2  # compaction chunk size (fixed)

        for i in range(t):
            k = torch.randn(B, G, 1, k_dim)
            v = torch.randn(B, G, 1, k_dim)
            add_full_kv_in_chunks(c, k, v, chunk_size=1)

        assert c.recent_count == t
        assert c.level_count[0].item() == 0
        assert c.token_count == t

        k = torch.randn(B, G, 1, k_dim)
        v = torch.randn(B, G, 1, k_dim)
        add_full_kv_in_chunks(c, k, v, chunk_size=1)

        assert c.recent_count == 1
        assert c.level_count[0].item() == 1
        assert c.token_count == t + 1


# ---------------------------------------------------------------------------
# get_attention_state tests
# ---------------------------------------------------------------------------

class TestGetAttentionState:
    def test_empty_cache_state(self, small_cache):
        """Empty cache should return zero-size tensors."""
        slot_k, slot_v, slot_w = small_cache.get_attention_state()
        assert slot_k.size(2) == 0
        assert slot_v.size(2) == 0
        assert slot_w.size(2) == 0

    def test_state_after_ingest(self, small_cache):
        """After ingest, one compact slot of weight 2, no recent tokens."""
        c = small_cache
        c.ingest_chunk(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))

        slot_k, slot_v, slot_w = c.get_attention_state()

        assert slot_k.size(2) == 1
        assert slot_w[0, 0, 0].item() == 2.0

    def test_state_after_prefill(self, small_cache):
        """After prefill, compact slots + recent w=1 tokens should be present."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        T = 9  # 4 flushes (8 tokens) + 1 in recent

        k = torch.randn(B, G, T, k_dim)
        v = torch.randn(B, G, T, k_dim)
        add_full_kv_in_chunks(c, k, v)

        slot_k, slot_v, slot_w = c.get_attention_state()

        # 4 compact entries (after carry to level 1) + 1 recent token
        assert slot_k.size(2) == c.B + 1
        torch.testing.assert_close(slot_w[0, 0], torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0]))
        # Weights must account for every committed token
        assert slot_w[0, 0].sum().item() == T

    def test_slot_exactness_weighted_mean(self, small_cache):
        """THE merge-exactness invariant: every slot's key/value equals the
        brute-force weighted mean of the tokens it covers, no matter how many
        binary-carry merges produced it. Spans are reconstructed from
        cumsum(slot_w) (slots are contiguous and time-ordered)."""
        c = small_cache
        B, G = 1, 2
        k_dim = 8
        T = 20  # several flushes + one carry through level 1

        k = torch.randn(B, G, T, k_dim)
        v = torch.randn(B, G, T, k_dim)
        add_full_kv_in_chunks(c, k, v)

        slot_k, slot_v, slot_w = c.get_attention_state()
        w = slot_w[0, 0]
        assert w.sum().item() == T  # all tokens accounted for

        ends = torch.cumsum(w, dim=0).long()
        starts = ends - w.long()
        for s in range(slot_k.size(2)):
            span_k = k[:, :, starts[s]:ends[s], :].mean(dim=2)
            span_v = v[:, :, starts[s]:ends[s], :].mean(dim=2)
            torch.testing.assert_close(slot_k[:, :, s, :], span_k, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(slot_v[:, :, s, :], span_v, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# log_kv_slot_attention tests
# ---------------------------------------------------------------------------

class TestSlotAttention:
    @staticmethod
    def _make_state(n_compact, n_recent, B=1, G=2, k_dim=8, v_dim=8):
        """Build a (slot_k, slot_v, slot_w) state: w=2 compact slots then w=1 recent."""
        n = n_compact + n_recent
        slot_k = torch.randn(B, G, n, k_dim)
        slot_v = torch.randn(B, G, n, v_dim)
        slot_w = torch.cat(
            [torch.full((B, G, n_compact), 2.0), torch.ones(B, G, n_recent)], dim=-1
        )
        return slot_k, slot_v, slot_w

    def test_output_shape(self):
        """Output should have shape (B, nh, T_q, v_dim)."""
        B, nh, T_q, k_dim, v_dim = 1, 4, 1, 8, 8

        slot_k, slot_v, slot_w = self._make_state(n_compact=3, n_recent=2)
        q = torch.randn(B, nh, T_q, k_dim)

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)

    def test_gqa_expansion(self):
        """When nh != n_groups, k/v/w should be expanded via repeat_interleave."""
        B, nh, G, T_q, k_dim, v_dim = 1, 4, 2, 1, 8, 8  # nh=4, G=2 -> repeat factor 2

        slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=1, G=G)
        q = torch.randn(B, nh, T_q, k_dim)

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)
        assert not torch.isnan(out).any()

    def test_empty_state(self):
        """With no slots to attend to, output should be zeros."""
        B, nh, T_q, k_dim, v_dim = 1, 4, 1, 8, 8

        slot_k, slot_v, slot_w = self._make_state(n_compact=0, n_recent=0)
        q = torch.randn(B, nh, T_q, k_dim)

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_only_recent(self):
        """With only recent tokens (no compact slots), should still work."""
        B, nh, G, T_q, v_dim = 1, 2, 2, 1, 8

        slot_k, slot_v, slot_w = self._make_state(n_compact=0, n_recent=3, G=G)
        q = torch.randn(B, nh, T_q, 8)

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1)

        assert out.shape == (B, nh, T_q, v_dim)
        assert not torch.isnan(out).any()

    def test_mask_applied(self):
        """Mask should prevent attention to certain slots."""
        B, nh, G, T_q, v_dim = 1, 2, 2, 3, 8

        slot_k, slot_v, slot_w = self._make_state(n_compact=1, n_recent=2, G=G)
        q = torch.randn(B, nh, T_q, 8)

        n_slots = 3
        mask = torch.ones(T_q, n_slots, dtype=torch.bool)
        for i in range(T_q):
            mask[i, i + 1:] = False

        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1, mask=mask)

        assert out.shape == (B, nh, T_q, v_dim)
        assert not torch.isnan(out).any()

        # Query 0 sees only slot 0 -> its output must be exactly slot 0's value.
        v_exp = slot_v.repeat_interleave(nh // G, dim=1)
        torch.testing.assert_close(out[:, :, 0, :], v_exp[:, :, 0, :])

    def test_scale_applied(self):
        """Different scales should produce different outputs."""
        B, nh, G, T_q = 1, 2, 2, 1

        slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=1, G=G)
        q = torch.randn(B, nh, T_q, 8)

        out1 = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.01)
        out2 = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=1.0)

        assert not torch.allclose(out1, out2)

    def test_equivalent_to_standard_attention_when_all_weight_1(self):
        """When all slots have weight=1 (no compaction), the mass bias is 0 and
        slot attention must match standard scaled dot-product attention."""
        B, nh, G, T_q, k_dim, v_dim = 1, 2, 2, 1, 8, 8
        scale = 1.0 / math.sqrt(k_dim)

        slot_k, slot_v, slot_w = self._make_state(
            n_compact=0, n_recent=3, G=G, k_dim=k_dim, v_dim=v_dim
        )
        q = torch.randn(B, nh, T_q, k_dim)

        # Standard attention with GQA expansion
        q_per_kv = nh // G
        k_exp = slot_k.repeat_interleave(q_per_kv, dim=1)
        v_exp = slot_v.repeat_interleave(q_per_kv, dim=1)
        scores = torch.matmul(q, k_exp.mT) * scale
        out_std = torch.matmul(torch.softmax(scores, dim=-1), v_exp)

        out_slot = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=scale)

        torch.testing.assert_close(out_slot, out_std, atol=1e-5, rtol=1e-5)

    def test_log_w_mass_bias_matches_duplicated_tokens(self):
        """THE mass-bias invariant: a slot with weight w and key x must draw
        exactly the same softmax mass as w separate w=1 slots with key x
        (log-sum-exp of equal logits = logit + log w). Needs a distractor slot
        so the mass split is actually observable."""
        B, G, nh, k_dim, v_dim = 1, 2, 2, 8, 8
        scale = 1.0 / math.sqrt(k_dim)
        q = torch.randn(B, nh, 1, k_dim)

        x_k = torch.randn(B, G, 1, k_dim)
        x_v = torch.randn(B, G, 1, v_dim)
        y_k = torch.randn(B, G, 1, k_dim)  # distractor
        y_v = torch.randn(B, G, 1, v_dim)

        # State A: token x twice (two w=1 slots) + distractor y
        kA = torch.cat([x_k, x_k, y_k], dim=2)
        vA = torch.cat([x_v, x_v, y_v], dim=2)
        wA = torch.ones(B, G, 3)

        # State B: one merged slot (k=x, w=2) + distractor y
        kB = torch.cat([x_k, y_k], dim=2)
        vB = torch.cat([x_v, y_v], dim=2)
        wB = torch.cat([torch.full((B, G, 1), 2.0), torch.ones(B, G, 1)], dim=-1)

        out_A = log_kv_slot_attention(q, kA, vA, wA, scale=scale)
        out_B = log_kv_slot_attention(q, kB, vB, wB, scale=scale)

        torch.testing.assert_close(out_B, out_A, atol=1e-5, rtol=1e-5)

    def test_lam_zero_disables_mass_bias(self):
        """lam=0 must ignore slot weights entirely (∝1/w forgetting ablation)."""
        B, G, nh, k_dim = 1, 2, 2, 8
        q = torch.randn(B, nh, 1, k_dim)
        slot_k, slot_v, _ = self._make_state(n_compact=2, n_recent=1, G=G)

        w_real = torch.tensor([[[8.0, 2.0, 1.0]]]).expand(B, G, 3)
        w_ones = torch.ones(B, G, 3)

        out_biased = log_kv_slot_attention(q, slot_k, slot_v, w_real, scale=0.5)
        out_lam0 = log_kv_slot_attention(q, slot_k, slot_v, w_real, scale=0.5, lam=0.0)
        out_unweighted = log_kv_slot_attention(q, slot_k, slot_v, w_ones, scale=0.5)

        torch.testing.assert_close(out_lam0, out_unweighted)
        assert not torch.allclose(out_biased, out_lam0)

    def test_append_exact_tokens(self):
        """Helper must append w=1 entries after the cached slots, in order."""
        B, G, k_dim, v_dim = 1, 2, 8, 8
        slot_k, slot_v, slot_w = self._make_state(n_compact=2, n_recent=1, G=G)
        k_new = torch.randn(B, G, 2, k_dim)
        v_new = torch.randn(B, G, 2, v_dim)

        k_all, v_all, w_all = append_exact_tokens(slot_k, slot_v, slot_w, k_new, v_new)

        assert k_all.size(2) == 5
        torch.testing.assert_close(k_all[:, :, 3:, :], k_new)
        torch.testing.assert_close(v_all[:, :, 3:, :], v_new)
        torch.testing.assert_close(w_all[0, 0], torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# reset_parameters tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_state(self, small_cache):
        c = small_cache
        # Put some data in
        c.ingest_chunk(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8))
        assert c.token_count > 0
        assert c.level_count[0].item() > 0

        c.reset_parameters()

        assert c.token_count == 0
        assert c.recent_count == 0
        assert c.level_count.sum() == 0
        assert c.total_slots == 0
        assert c.total_tokens_covered == 0
        assert c.recent_k.abs().sum() == 0
        assert getattr(c, "level_k_0").abs().sum() == 0


# ---------------------------------------------------------------------------
# Integration: explicit commit + get_attention_state round-trip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_explicit_commit_roundtrip(self):
        """End-to-end: commit prompt/chunks and verify state shapes throughout."""
        B_batch, G = 1, 2
        k_dim = 8
        v_dim = 8
        max_seq = 64

        c = LogStructuredKVCache(
            (B_batch, G, max_seq, k_dim),
            (B_batch, G, max_seq, v_dim),
            B=4,
        )

        # Prefill 7 tokens (3 flushes of 2 + 1 in recent)
        T_prefill = 7
        k = torch.randn(B_batch, G, T_prefill, k_dim)
        v = torch.randn(B_batch, G, T_prefill, v_dim)
        add_full_kv_in_chunks(c, k, v)

        assert c.token_count == T_prefill
        assert c.recent_count == 1  # 7 % 2 = 1

        # Decode 3 more tokens
        for i in range(3):
            k_dec = torch.randn(B_batch, G, 1, k_dim)
            v_dec = torch.randn(B_batch, G, 1, v_dim)
            add_full_kv_in_chunks(c, k_dec, v_dec, chunk_size=1)

        assert c.token_count == T_prefill + 3

        # Verify we can compute attention with the final state
        nh = 4
        slot_k, slot_v, slot_w = c.get_attention_state()
        assert slot_w[0, 0].sum().item() == T_prefill + 3
        q = torch.randn(B_batch, nh, 1, k_dim)
        out = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=0.1)
        assert out.shape == (B_batch, nh, 1, v_dim)
        assert not torch.isnan(out).any()

    def test_cache_attention_matches_dense_for_pairwise_identical_tokens(self):
        """End-to-end EXACTNESS check of the whole design: when the two tokens
        inside every compacted pair are identical, the merged key/value equal
        the shared token and +log(2) equals the log-sum-exp of the two equal
        logits — so slot attention over the cache must reproduce dense
        attention over all tokens exactly (up to fp tolerance)."""
        B_batch, G, nh = 1, 2, 4
        k_dim = 8
        v_dim = 8
        T = 8  # 3 flushes (tokens 0-5 -> 3 slots, no carry with B=4) + 2 recent

        c = LogStructuredKVCache(
            (B_batch, G, 64, k_dim), (B_batch, G, 64, v_dim), B=4,
        )

        base_k = torch.randn(B_batch, G, T // 2, k_dim)
        base_v = torch.randn(B_batch, G, T // 2, v_dim)
        k = base_k.repeat_interleave(2, dim=2)  # pairs of identical tokens
        v = base_v.repeat_interleave(2, dim=2)
        add_full_kv_in_chunks(c, k, v)  # 2-chunks align with the pairs
        assert c.level_count[0].item() == 3
        assert c.recent_count == 2

        scale = 1.0 / math.sqrt(k_dim)
        q = torch.randn(B_batch, nh, 1, k_dim)

        slot_k, slot_v, slot_w = c.get_attention_state()
        out_slot = log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale=scale)

        # Dense attention over all T tokens
        q_per_kv = nh // G
        k_exp = k.repeat_interleave(q_per_kv, dim=1)
        v_exp = v.repeat_interleave(q_per_kv, dim=1)
        scores = torch.matmul(q, k_exp.mT) * scale
        out_dense = torch.matmul(torch.softmax(scores, dim=-1), v_exp)

        torch.testing.assert_close(out_slot, out_dense, atol=1e-5, rtol=1e-5)

    def test_capacity_for_4k(self):
        """Verify the cache can handle 4K tokens without overflow (B=1024, 2:1 compaction)."""
        B_batch, G = 1, 1
        k_dim = 8
        v_dim = 8
        max_seq = 4096

        c = LogStructuredKVCache(
            (B_batch, G, max_seq, k_dim),
            (B_batch, G, max_seq, v_dim),
            B=1024,
        )

        # Simulate ingest of all 4096/2 = 2048 compact entries.
        # This fills level 0 twice and carries through level 1, with no overflow.
        for i in range(2048):
            c.ingest_chunk(
                torch.randn(B_batch, G, 2, k_dim),
                torch.randn(B_batch, G, 2, v_dim),
            )

        assert c.level_count[2].item() > 0
        assert c.level_count[0].item() == 0
        assert c.level_count[1].item() == 0
        assert c.token_count == 4096


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
        assert cache.token_count == 2
        assert cache.recent_count == 2
        assert attn._log_kv_pending is not None

        q, k, v = self._random_qkv(attn, T=1)
        out = attn._log_kv_training_forward(q, k, v, B=1, T=1, reset_cache=False, defer_last_single=True)

        assert out.shape == (1, 1, attn.config.n_embd)
        assert attn._log_kv_pending is None
        assert cache.token_count == 4
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
        assert cache.token_count == 6
        assert cache.level_count[0].item() == 2
        assert cache.recent_count == 2


# ---------------------------------------------------------------------------
# Batched flush: chunked add_recent must equal strict 2-token streaming
# ---------------------------------------------------------------------------

class TestBatchedFlushEquivalence:
    """add_recent() flushes window overflow in one batch (one shift, vectorized
    pair merges). The resulting cache state must be bit-identical to feeding the
    same token stream in the strict 2-token cadence."""

    @staticmethod
    def _new_cache(recent: int = 16, B: int = 4, max_seq: int = 2048) -> LogStructuredKVCache:
        return LogStructuredKVCache(
            (1, 2, max_seq, 8), (1, 2, max_seq, 8), B=B, recent_size=recent,
            device=torch.device("cpu"), dtype=torch.float32,
        )

    @pytest.mark.parametrize("chunk_sizes", ([16], [2, 4, 14, 16, 6, 12]))
    def test_even_chunked_add_recent_matches_streaming(self, chunk_sizes):
        torch.manual_seed(0)
        T = 700  # even; drives many flushes and several binary carries (B=4)
        k = torch.randn(1, 2, T, 8)
        v = torch.randn(1, 2, T, 8)

        ref = self._new_cache()
        add_full_kv_in_chunks(ref, k, v, chunk_size=2)

        c = self._new_cache()
        i, si = 0, 0
        while i < T:
            n = min(chunk_sizes[si % len(chunk_sizes)], T - i)
            c.add_recent(k[:, :, i:i + n], v[:, :, i:i + n])
            i += n
            si += 1

        assert_cache_states_bit_identical(ref, c)

    def test_full_window_replacement_chunk(self):
        """n == recent_size arriving at a full window flushes the entire old
        window — the boundary case of the batched path."""
        torch.manual_seed(1)
        recent = 8
        k = torch.randn(1, 2, 2 * recent, 8)
        v = torch.randn(1, 2, 2 * recent, 8)

        ref = self._new_cache(recent=recent)
        add_full_kv_in_chunks(ref, k, v, chunk_size=2)

        c = self._new_cache(recent=recent)
        c.add_recent(k[:, :, :recent], v[:, :, :recent])
        c.add_recent(k[:, :, recent:], v[:, :, recent:])

        assert_cache_states_bit_identical(ref, c)


# ---------------------------------------------------------------------------
# Vectorized block prefill (inference): exact in-window, state-exact always
# ---------------------------------------------------------------------------

class TestBlockPrefill:
    """log_kv_prefill_block > 2 batches prefill attention. Within the recent
    window the outputs must equal strict streaming (block=2); past compaction
    the outputs may deviate boundedly, but the cache state trajectory — and
    therefore every subsequent decode step — must stay bit-identical."""

    @staticmethod
    def _make_attention(recent: int, prefill_block: int) -> CausalSelfAttention:
        config = Config(
            block_size=64,
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
            max_seq_length=64,
            B=4,
            recent_size=recent,
        )
        attn.log_kv_prefill_block = prefill_block
        return attn

    def _run_pair(self, recent: int, T: int) -> tuple:
        # Identical weights: re-seed before each construction so qkv/proj match.
        torch.manual_seed(0)
        a_stream = self._make_attention(recent, prefill_block=2)
        torch.manual_seed(0)
        a_block = self._make_attention(recent, prefill_block=64)

        torch.manual_seed(42)
        head_size = 4  # n_embd=8 / n_head=2
        q = torch.randn(1, 2, T, head_size)
        k = torch.randn(1, 2, T, head_size)
        v = torch.randn(1, 2, T, head_size)
        with torch.no_grad():
            y_stream = a_stream._log_kv_training_forward(
                q, k, v, B=1, T=T, reset_cache=True, defer_last_single=True)
            y_block = a_block._log_kv_training_forward(
                q, k, v, B=1, T=T, reset_cache=True, defer_last_single=True)
        return a_stream, a_block, y_stream, y_block

    def test_in_window_outputs_match_streaming(self):
        """No flush inside the sequence -> block prefill is the same math as
        streaming (differences only from fp32 matmul tiling)."""
        a_s, a_b, y_s, y_b = self._run_pair(recent=32, T=20)
        torch.testing.assert_close(y_b, y_s, atol=1e-5, rtol=1e-5)
        assert_cache_states_bit_identical(a_s.kv_cache, a_b.kv_cache)

    def test_post_compaction_state_and_decode_match_streaming(self):
        """Past compaction the prefill OUTPUTS deviate (bounded, documented),
        but cache state and subsequent decode logits must not."""
        a_s, a_b, y_s, y_b = self._run_pair(recent=8, T=40)
        assert torch.isfinite(y_b).all()
        assert_cache_states_bit_identical(a_s.kv_cache, a_b.kv_cache)

        torch.manual_seed(1)
        head_size = 4
        q = torch.randn(1, 2, 2, head_size)
        k = torch.randn(1, 2, 2, head_size)
        v = torch.randn(1, 2, 2, head_size)
        with torch.no_grad():
            d_s = a_s._log_kv_training_forward(q, k, v, B=1, T=2, reset_cache=False, defer_last_single=True)
            d_b = a_b._log_kv_training_forward(q, k, v, B=1, T=2, reset_cache=False, defer_last_single=True)
        assert torch.equal(d_s, d_b)


# ---------------------------------------------------------------------------
# Low-memory training Function: must equal the naive differentiable reference
# ---------------------------------------------------------------------------

class TestLowMemTrainingEquivalence:
    """``_log_kv_train_lowmem_forward`` (LogKVStreamTrainingAttention:
    graph-free forward stream + chunk-by-chunk backward replay) must reproduce
    the naive per-chunk-graph reference
    ``_log_kv_training_forward(reset_cache=True, defer_last_single=False)``
    exactly: outputs, input gradients, parameter gradients, cache end state."""

    @staticmethod
    def _make_attention() -> CausalSelfAttention:
        config = Config(
            block_size=64,
            padded_vocab_size=16,
            n_layer=1,
            n_head=4,
            n_query_groups=2,  # GQA: exercises the broadcast attention branch
            n_embd=16,
            rotary_percentage=0.5,
        )
        attn = CausalSelfAttention(config, block_idx=0)
        attn.kv_cache = attn.build_log_kv_cache(
            batch_size=1,
            max_seq_length=64,
            B=4,
            recent_size=4,  # small: compaction kicks in almost immediately
        )
        return attn

    def _make_pair_with_inputs(self, T: int, seed: int):
        # Identical weights: re-seed before each construction so qkv/proj match.
        torch.manual_seed(0)
        a_ref = self._make_attention()
        torch.manual_seed(0)
        a_new = self._make_attention()

        torch.manual_seed(seed)
        hs = a_ref.config.head_size
        q0 = torch.randn(1, 4, T, hs)
        k0 = torch.randn(1, 2, T, hs)
        v0 = torch.randn(1, 2, T, hs)
        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        return a_ref, a_new, (q1, k1, v1), (q2, k2, v2)

    @pytest.mark.parametrize("T", [1, 2, 5, 12, 33])
    def test_matches_reference_forward_and_grads(self, T):
        a_ref, a_new, (q1, k1, v1), (q2, k2, v2) = self._make_pair_with_inputs(T, seed=7)

        y_ref = a_ref._log_kv_training_forward(q1, k1, v1, B=1, T=T)
        y_new = a_new._log_kv_train_lowmem_forward(q2, k2, v2, B=1, T=T)

        # Same op sequence (both chunk through log_kv_chunk_attention semantics
        # with causal_tail), so outputs and cache trajectories are identical.
        assert torch.equal(y_new, y_ref)
        assert_cache_states_bit_identical(a_ref.kv_cache, a_new.kv_cache)

        # Position-dependent loss so gradient errors cannot cancel.
        torch.manual_seed(99)
        w = torch.randn_like(y_ref)
        (y_ref * w).sum().backward()
        (y_new * w).sum().backward()

        torch.testing.assert_close(q2.grad, q1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(k2.grad, k1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(v2.grad, v1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            a_new.proj.weight.grad, a_ref.proj.weight.grad, rtol=1e-6, atol=1e-6
        )

    def test_backward_ignores_forwards_cache_end_state(self):
        """Backward replays the stream from a reset cache, so mutating the
        cache between forward and backward (as interleaved micro-batches or
        activation-checkpoint recompute would) must not change gradients."""
        T = 12
        a_ref, a_new, (q1, k1, v1), (q2, k2, v2) = self._make_pair_with_inputs(T, seed=3)

        y_ref = a_ref._log_kv_training_forward(q1, k1, v1, B=1, T=T)
        y_new = a_new._log_kv_train_lowmem_forward(q2, k2, v2, B=1, T=T)

        # Trash the low-mem module's cache state before its backward runs.
        hs = a_new.config.head_size
        a_new.kv_cache.reset_parameters()
        a_new.kv_cache.add_recent(torch.randn(1, 2, 2, hs), torch.randn(1, 2, 2, hs))

        y_ref.sum().backward()
        y_new.sum().backward()
        torch.testing.assert_close(q2.grad, q1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(k2.grad, k1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(v2.grad, v1.grad, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Salience pinning (SnapKV-style observation window)
# ---------------------------------------------------------------------------

class TestSaliencePinning:
    """Observation-window pinning: at a FRESH inference prefill the prompt-tail
    queries score the prefix and the top tokens per KV group are kept as exact
    w=1 slots. Pins are DUPLICATES — the hierarchy/recent trajectory must stay
    bit-identical with pinning on or off — and pin_size=0 is a strict no-op."""

    T = 40
    NEEDLE = 2  # early position: compacted into the hierarchy long before the tail

    @staticmethod
    def _make_attention(pin_size: int) -> CausalSelfAttention:
        config = Config(
            block_size=64,
            padded_vocab_size=16,
            n_layer=1,
            n_head=4,
            n_query_groups=2,  # GQA: groups select independently
            n_embd=16,
            rotary_percentage=0.5,
        )
        attn = CausalSelfAttention(config, block_idx=0)
        attn.kv_cache = attn.build_log_kv_cache(
            batch_size=1, max_seq_length=64, B=4, recent_size=4,
            pin_size=pin_size,
        )
        attn.log_kv_pin_obs_window = 4
        return attn

    def _needle_qkv(self, attn: CausalSelfAttention):
        """Random stream plus one distinctive needle key that the observation
        window (the last log_kv_pin_obs_window queries — 'the question') is
        aligned with. Everything else is low-energy noise."""
        torch.manual_seed(5)
        hs = attn.config.head_size
        T = self.T
        q = torch.randn(1, 4, T, hs) * 0.1
        k = torch.randn(1, 2, T, hs) * 0.1
        v = torch.randn(1, 2, T, hs)
        direction = torch.zeros(hs)
        direction[0] = 1.0
        k[:, :, self.NEEDLE, :] = 8.0 * direction
        q[:, :, T - 4:, :] = 8.0 * direction
        return q, k, v

    def test_needle_gets_pinned_exactly(self):
        # pin_size 8 >= the 7-wide max-pool tie span around the needle, so the
        # needle index itself is guaranteed into the top-k regardless of how
        # topk breaks ties among equal pooled scores.
        attn = self._make_attention(pin_size=8)
        cache = attn.kv_cache
        q, k, v = self._needle_qkv(attn)

        with torch.no_grad():
            attn._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)

        assert cache.pin_count == 8
        idx = attn._log_kv_pin_indices  # (1, groups, n_pin), time-ordered
        assert idx is not None
        # Candidates end at T - recent_size: recent tokens are never pinned
        # (they are already exact; duplicating them would distort mass).
        assert int(idx.max()) < self.T - cache.recent_size
        # Every group pinned the needle, and its K/V are the EXACT originals.
        for g in range(2):
            row = (idx[0, g] == self.NEEDLE).nonzero()
            assert row.numel() == 1, f"group {g} did not pin the needle: {idx[0, g].tolist()}"
            j = int(row[0, 0])
            assert torch.equal(cache.pin_k[0, g, j], k[0, g, self.NEEDLE])
            assert torch.equal(cache.pin_v[0, g, j], v[0, g, self.NEEDLE])
        # Pins surface in the attention state as extra exact w=1 slots.
        sk, _, sw = cache.get_attention_state()
        assert sk.size(2) == cache.total_slots
        assert int((sw[0, 0] == 1).sum()) == cache.pin_count + cache.recent_count

    def test_hierarchy_trajectory_unchanged_by_pinning(self):
        torch.manual_seed(0)
        a_off = self._make_attention(pin_size=0)
        torch.manual_seed(0)
        a_on = self._make_attention(pin_size=8)
        q, k, v = self._needle_qkv(a_on)

        with torch.no_grad():
            a_off._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)
            a_on._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)

        # Pins DUPLICATE tokens: compaction/recent must not change at all.
        assert_cache_states_bit_identical(a_off.kv_cache, a_on.kv_cache)
        assert a_off.kv_cache.pin_count == 0
        assert a_on.kv_cache.pin_count == 8

    def test_disabled_or_short_prompt_is_noop(self):
        # pin_size=0: selection never runs.
        attn = self._make_attention(pin_size=0)
        q, k, v = self._needle_qkv(attn)
        with torch.no_grad():
            attn._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)
        assert attn.kv_cache.pin_count == 0
        assert attn._log_kv_pin_indices is None

        # Prompt no longer than recent_size: no candidates, no pins.
        attn2 = self._make_attention(pin_size=8)
        torch.manual_seed(1)
        hs = attn2.config.head_size
        with torch.no_grad():
            attn2._log_kv_training_forward(
                torch.randn(1, 4, 4, hs), torch.randn(1, 2, 4, hs), torch.randn(1, 2, 4, hs),
                B=1, T=4, reset_cache=True, defer_last_single=True,
            )
        assert attn2.kv_cache.pin_count == 0

    def test_decode_keeps_pins_and_reset_clears_them(self):
        attn = self._make_attention(pin_size=8)
        cache = attn.kv_cache
        q, k, v = self._needle_qkv(attn)
        with torch.no_grad():
            attn._log_kv_training_forward(q, k, v, B=1, T=self.T, reset_cache=True, defer_last_single=True)
        assert cache.pin_count == 8

        # Decode steps never re-select (token_count > 0) and keep pins visible.
        torch.manual_seed(2)
        hs = attn.config.head_size
        with torch.no_grad():
            out = attn._log_kv_training_forward(
                torch.randn(1, 4, 1, hs), torch.randn(1, 2, 1, hs), torch.randn(1, 2, 1, hs),
                B=1, T=1, reset_cache=False, defer_last_single=True,
            )
        assert out.shape == (1, 1, attn.config.n_embd)
        assert cache.pin_count == 8

        cache.reset_parameters()
        assert cache.pin_count == 0
        assert torch.equal(cache.pin_k, torch.zeros_like(cache.pin_k))


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
        assert cache.token_count == 2
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
        assert cache.recent_k.dtype == torch.bfloat16  # allocated at model dtype

        # Prefill 5 tokens (crosses several 2-token chunks) then a decode step.
        idx = torch.randint(0, model.config.padded_vocab_size, (1, 5))
        logits = model(idx, torch.arange(5))
        assert logits.shape == (1, 5, model.config.padded_vocab_size)
        assert logits.dtype == torch.bfloat16

        # Buffers were reconciled to the activation dtype.
        assert cache.recent_k.dtype == torch.bfloat16
        assert getattr(cache, "level_k_0").dtype == torch.bfloat16

        idx_next = torch.randint(0, model.config.padded_vocab_size, (1, 1))
        logits = model(idx_next, torch.tensor([5]))
        assert logits.shape == (1, 1, model.config.padded_vocab_size)
        assert logits.dtype == torch.bfloat16

    def test_convert_dtype_is_idempotent_noop(self):
        model = self._make_bf16_model()
        cache = model.transformer.h[0].attn.kv_cache

        cache._convert_dtype(torch.bfloat16)
        buf_before = cache.recent_k
        cache._convert_dtype(torch.bfloat16)  # already bf16 -> must be a no-op
        assert cache.recent_k is buf_before  # no new allocation
        assert cache.level_count.dtype == torch.long  # counts stay integer


# ---------------------------------------------------------------------------
# Integration: LogKV inference equals dense attention within the recent window
# ---------------------------------------------------------------------------

class TestLogKVMatchesDense:
    """Within recent_size no compaction occurs, so LogKV inference must reproduce
    dense causal attention exactly. This validates the prefill fast path that
    underpins eval.py's loglikelihood scoring through the logKV cache."""

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
