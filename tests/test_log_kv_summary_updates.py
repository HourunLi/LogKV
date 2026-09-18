"""Summary mass/position invariants and graph-owned update replay."""
from copy import deepcopy
from contextlib import nullcontext
import gc
import weakref
from unittest.mock import patch

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

from litgpt.config import Config
from litgpt.model import GPT, Block
from litgpt.log_kv_cache import LogStructuredKVCache, _SemanticReplayUpdates
from litgpt.log_kv_checkpoint import enable_logkv_checkpoint_replay

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA'))]


def cache_for(device='cpu', summary=8, **kwargs):
    shape = (2, 2, 256, 8)
    return LogStructuredKVCache(
        shape, shape, B=3, recent_size=16, semantic_flush_granularity=16,
        semantic_clusters=True, cluster_k_max=8, semantic_summary_size=summary,
        semantic_centroid_backend='parallel', semantic_anchor_mode='mid',
        allocate_second_order=False, device=device,
        cos_cache=torch.ones(256, 8, device=device), sin_cache=torch.zeros(256, 8, device=device),
        rope_n_elem=8, **kwargs)


def state(cache):
    return {name: x.clone() for name, x in cache.named_buffers() if not name.startswith('op_log')}


def assert_state(cache, expected):
    for name, x in state(cache).items():
        torch.testing.assert_close(x, expected[name], atol=0, rtol=0, msg=name)


@pytest.mark.parametrize('device', DEVICES)
def test_summary_preserves_mass_moments_and_lane_boundaries(device):
    cache = cache_for(device)
    counts = [19, 1, 5]
    k = torch.arange(25 * 8, device=device, dtype=torch.float32).reshape(25, 8)
    v = k * 2
    p = torch.arange(25, device=device, dtype=torch.int64) * 11 + 2**34
    block = (k, v, torch.ones(25, device=device), *(None,) * 5,
             p, p, p, torch.arange(25, device=device), torch.zeros(25, device=device, dtype=torch.bool))
    actual, compressed = cache._semantic_summarize_block(block, counts)
    assert compressed == [3, 1, 1]
    offset, row = 0, 0
    for count in counts:
        for start in range(offset, offset + count, 8):
            end = min(start + 8, offset + count)
            torch.testing.assert_close(actual[0][row], k[start:end].mean(0), atol=0, rtol=0)
            torch.testing.assert_close(actual[1][row], v[start:end].mean(0), atol=0, rtol=0)
            assert actual[2][row] == end - start
            assert actual[8][row] == p[start] and actual[9][row] == p[end - 1]
            assert actual[10][row] == p[start:end].sum()
            row += 1
        offset += count
    torch.testing.assert_close((actual[0] * actual[2][:, None]).sum(0), k.sum(0))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('summary', [1, 8])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_update_replay_skips_gather_centroid_and_parsing_and_restores_all_state(device, summary, dtype):
    torch.manual_seed(11)
    cache = cache_for(device, summary, semantic_replay_updates=True, dtype=dtype)
    tape = _SemanticReplayUpdates()
    cache.begin_op_log()
    inputs, expected = [], []
    for start in range(0, 128, 16):
        # Noncontiguous payload, with enough appends to exercise carries.
        k = torch.randn(2, 2, 32, 8, device=device, dtype=dtype)[:, :, ::2]
        v = torch.randn_like(k)
        p = torch.arange(start, start + 16, device=device).expand(2, -1)
        host = [list(range(start, start + 16))] * 2
        inputs.append((k, v, p, host))
        with torch.no_grad(), cache._semantic_update_context(tape):
            cache.route_and_flush_batch(k, v, p, positions_host=host, record_op_log=True)
        expected.append((state(cache), deepcopy(cache._semantic_counts)))
    log, lengths = cache.take_op_log()
    assert tape.tensors and all(not x.requires_grad for x in tape.tensors)
    for _ in range(2):
        cache.reset_parameters()
        with patch.object(cache, '_route_and_flush_batch', side_effect=AssertionError('parsed/rerouted')):
            for (k, v, p, host), (buffers, counts) in zip(inputs, expected):
                with torch.no_grad(), cache._semantic_update_context(tape, replay=True):
                    cache.route_and_flush_batch(k, v, p, positions_host=host,
                                               replay_op_log=log, replay_op_log_len=lengths)
                assert_state(cache, buffers)
                assert cache._semantic_counts == counts
    assert cache._active_updates is None and cache._update_actions is None
    with cache._semantic_update_context(tape, replay=True), pytest.raises(RuntimeError, match='missing'):
        cache.route_and_flush_batch(k, v, p, positions_host=[[999] * 16] * 2, replay_op_log=log)


@pytest.mark.parametrize('device', DEVICES)
def test_summary_oplog_replay_without_saved_updates_is_exact(device):
    torch.manual_seed(21)
    cache = cache_for(device)
    inputs = [(torch.randn(2, 2, 16, 8, device=device), torch.randn(2, 2, 16, 8, device=device))
              for _ in range(6)]
    cache.begin_op_log()
    states = []
    for i, (k, v) in enumerate(inputs):
        pos = torch.arange(i * 16, (i + 1) * 16, device=device)
        cache.route_and_flush_batch(k, v, pos, record_op_log=True)
        states.append(state(cache))
    log, lengths = cache.take_op_log()
    host_log = cache._last_op_log_host
    cache.reset_parameters()
    for i, (k, v) in enumerate(inputs):
        pos = torch.arange(i * 16, (i + 1) * 16, device=device)
        cache.route_and_flush_batch(k, v, pos, replay_op_log=log, replay_op_log_len=lengths,
                                   replay_op_log_host=host_log)
        assert_state(cache, states[i])


@pytest.mark.parametrize('device', DEVICES)
def test_checkpoint_summary_updates_match_reference_gradients_and_release(device):
    torch.manual_seed(14)
    config = Config(block_size=64, n_layer=2, n_embd=32, n_head=4, n_query_groups=2,
                    vocab_size=41, padding_multiple=1, rotary_percentage=1.)
    original = GPT(config).to(device)
    models = [deepcopy(original), deepcopy(original)]
    for i, model in enumerate(models):
        model.enable_log_kv_training(
            batch_size=1, B=3, recent_size=8, train_block=8, second_order_scale=0.,
            semantic_clusters=True, cluster_k_max=8, semantic_flush_granularity=8,
            semantic_anchor_mode='mid', semantic_pack_backend='torch', allocate_second_order=False,
            semantic_summary_size=8, semantic_replay_updates=bool(i), semantic_centroid_backend='parallel',
            device=torch.device(device))
        apply_activation_checkpointing(model, check_fn=lambda m: isinstance(m, Block))
        enable_logkv_checkpoint_replay(model, Block)
    xs = [torch.randint(41, (1, n), device=device) for n in (40, 48)]
    targets = [torch.randint(41, x.shape, device=device) for x in xs]
    outputs, grads, refs = [], [], []
    save = _SemanticReplayUpdates.save

    def watch(tape, tensor):
        idx = save(tape, tensor)
        if idx is not None:
            refs.append(weakref.ref(tape.tensors[idx]))
        return idx

    for i, model in enumerate(models):
        with patch.object(_SemanticReplayUpdates, 'save', watch):
            losses = [model(x, targets=t, loss_chunk_size=7) for x, t in zip(xs, targets)]
        # Two outstanding graphs, reverse traversal, then retain_graph reuse.
        guard = patch.object(LogStructuredKVCache, '_route_and_flush_batch',
                             side_effect=AssertionError('update replay reran routing/centroid')) if i else nullcontext()
        with guard:
            losses[0].backward(retain_graph=True)
            losses[1].backward()
            losses[0].backward()
        outputs.append([x.detach() for x in losses])
        grads.append([p.grad.clone() for p in model.parameters()])
        del losses
    for a, b in zip(outputs[0] + grads[0], outputs[1] + grads[1]):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    gc.collect()
    assert refs and all(ref() is None for ref in refs)


@pytest.mark.parametrize('kwargs', [dict(semantic_legacy_route=True), dict(semantic_cluster_chunk_size=4),
                                     dict(seg_gap_max=1)])
def test_unsupported_summary_combinations_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        cache_for(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/Triton')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_parallel_centroid_uneven_long_runs_match_fp64_reference(dtype):
    from litgpt.log_kv_updates_triton import centroid_parallel
    lengths = [1, 31, 32, 33, 257, 2049]
    torch.manual_seed(5)
    k = torch.randn(sum(lengths), 128, device='cuda', dtype=dtype)
    mu = torch.randn(9, 128, device='cuda')
    ne = torch.arange(9, device='cuda').float() * 61
    clusters = torch.tensor([7, 2, 0, 8, 1, 4], device='cuda')
    forget = torch.tensor([.5, 1., 0., .5, 1., 1.], device='cuda')
    meta = torch.stack((clusters, torch.arange(6, device='cuda'),
                        torch.tensor([0] + lengths[:-1], device='cuda').cumsum(0),
                        torch.tensor(lengths, device='cuda'), forget.view(torch.int32).long()))
    old_mu, old_ne = mu.clone(), ne.clone()
    expected_mu, expected_ne = mu.clone(), ne.clone()
    offset = 0
    for row, length in enumerate(lengths):
        c = int(clusters[row])
        pre = old_ne[c].double() * forget[row].double()
        expected_mu[c] = ((pre * old_mu[c].double() + k[offset:offset+length].double().sum(0)) / (pre + length)).float()
        expected_ne[c] = pre + length
        offset += length
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        centroid_parallel(k, mu, ne, meta, 0, 6)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(mu, expected_mu, atol=5e-7, rtol=3e-6)
    torch.testing.assert_close(ne, expected_ne, atol=0, rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_summary_inference_flush_partition_and_causality(device):
    from litgpt.log_kv_cache import LogKVStreamTrainingAttention
    torch.manual_seed(98)
    caches = [cache_for(device, semantic_replay_updates=True) for _ in range(2)]
    k = torch.randn(2, 2, 64, 8, device=device)
    v = torch.randn_like(k)
    for cache, chunk in zip(caches, (4, 16)):
        for start in range(0, 64, chunk):
            cache.add_recent(k[:, :, start:start+chunk], v[:, :, start:start+chunk],
                             k_raw=k[:, :, start:start+chunk])
        assert cache._active_updates is None and cache._update_actions is None
        torch.testing.assert_close(cache.level_w.sum((2, 3, 4)),
                                   torch.full((2, 2), 48., device=device))
    assert_state(caches[0], state(caches[1]))
    q = torch.randn_like(k)
    original = LogKVStreamTrainingAttention.apply(q, k, v, caches[0], .3, 16, 0., k)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, :, 25:] *= 10
    changed_v[:, :, 25:] *= -7
    changed = LogKVStreamTrainingAttention.apply(q, changed_k, changed_v, caches[0], .3, 16, 0., changed_k)
    torch.testing.assert_close(original[:, :, :25], changed[:, :, :25], atol=1e-6, rtol=1e-5)


def test_summary_rejects_second_order_before_updates():
    cache = cache_for()
    with pytest.raises(ValueError, match='second_order_scale=0'):
        cache.second_order = True


def test_saved_updates_replay_from_prefilled_cache_without_reset():
    torch.manual_seed(79)
    cache = cache_for()
    k, v = torch.randn(2, 2, 32, 8), torch.randn(2, 2, 32, 8)
    cache.route_and_flush_batch(k[:, :, :16], v[:, :, :16], torch.arange(16))
    expected = deepcopy(cache)
    tape = _SemanticReplayUpdates()
    host = [list(range(16, 32))] * 2
    with expected._semantic_update_context(tape):
        expected.route_and_flush_batch(k[:, :, 16:], v[:, :, 16:], torch.arange(16, 32),
                                       positions_host=host, record_op_log=True)
    log, lengths = expected.take_op_log()
    with cache._semantic_update_context(tape, replay=True):
        cache.route_and_flush_batch(k[:, :, 16:], v[:, :, 16:], torch.arange(16, 32),
                                   positions_host=host, replay_op_log=log, replay_op_log_len=lengths)
    assert_state(cache, state(expected))
    # The same update cannot be applied twice or at a different cursor.
    with cache._semantic_update_context(tape, replay=True), pytest.raises(RuntimeError, match='out of order'):
        cache.route_and_flush_batch(k[:, :, 16:], v[:, :, 16:], torch.arange(16, 32),
                                   positions_host=host, replay_op_log=log, replay_op_log_len=lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_update_tape_cross_stream_wait_before_reset_and_replay():
    torch.manual_seed(83)
    device = torch.device('cuda')
    cache = cache_for(device, semantic_replay_updates=True, dtype=torch.bfloat16)
    tape = _SemanticReplayUpdates()
    k = torch.randn(2, 2, 64, 8, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    pos = torch.arange(64, device=device).reshape(4, 16)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer), torch.no_grad():
        cache.begin_op_log()
        for i in range(4):
            host = [list(range(i * 16, (i + 1) * 16))] * 2
            with cache._semantic_update_context(tape):
                cache.route_and_flush_batch(k[:, :, i*16:(i+1)*16], v[:, :, i*16:(i+1)*16], pos[i],
                                           positions_host=host, record_op_log=True)
        log, lengths = cache.take_op_log()
        expected = state(cache)
        ready = torch.cuda.Event()
        ready.record(producer)
    with torch.cuda.stream(consumer), torch.no_grad():
        # No host synchronize. Waiting before reset protects the capture's reads
        # of mutable buffers; each replay also waits before consuming its data.
        tape.wait(device)
        consumer.wait_event(ready)  # expected-state clones and op-log copies
        cache.reset_parameters()
        for i in range(4):
            host = [list(range(i * 16, (i + 1) * 16))] * 2
            with cache._semantic_update_context(tape, replay=True):
                cache.route_and_flush_batch(k[:, :, i*16:(i+1)*16], v[:, :, i*16:(i+1)*16], pos[i],
                                           positions_host=host, replay_op_log=log, replay_op_log_len=lengths)
    torch.cuda.current_stream().wait_stream(consumer)
    assert_state(cache, expected)


def test_update_replay_retains_segment_pads_when_summaries_disabled():
    torch.manual_seed(85)
    cache = cache_for(summary=1, semantic_replay_updates=True, seg_gap_max=1, seg_block_level=2)
    k, v = torch.randn(2, 2, 64, 8), torch.randn(2, 2, 64, 8)
    tape = _SemanticReplayUpdates()
    cache.begin_op_log()
    for i in range(4):
        host = [list(range(i*32, (i+1)*32, 2))] * 2
        with cache._semantic_update_context(tape):
            cache.route_and_flush_batch(k[:, :, i*16:(i+1)*16], v[:, :, i*16:(i+1)*16], torch.tensor(host),
                                       positions_host=host, record_op_log=True)
    expected = state(cache)
    log, lengths = cache.take_op_log()
    cache.reset_parameters()
    for i in range(4):
        host = [list(range(i*32, (i+1)*32, 2))] * 2
        with cache._semantic_update_context(tape, replay=True):
            cache.route_and_flush_batch(k[:, :, i*16:(i+1)*16], v[:, :, i*16:(i+1)*16], torch.tensor(host),
                                       positions_host=host, replay_op_log=log, replay_op_log_len=lengths)
    assert_state(cache, expected)
