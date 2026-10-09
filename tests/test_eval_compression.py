"""Exercise compression reporting in the actual eval request methods on CPU."""

import ast
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from litgpt.config import Config
from litgpt.generate.base import generate
from litgpt.kv_compression import cache_compression_snapshot
from litgpt.model import GPT


class Tokenizer:
    bos_id = 1
    eos_id = None

    def encode(self, text, device="cpu", bos=False):
        return torch.tensor([int(token) for token in text.split()], device=device)

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens.tolist())


@pytest.fixture(scope="module")
def wrapper_class():
    # Import just the real methods, avoiding eval.py's GPU/HF runtime setup.
    source = Path(__file__).resolve().parents[1] / "eval.py"
    cls = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "LogKVLM")
    names = {"_score_tokens", "generate_until", "loglikelihood_rolling", "_reset_eval_cache"}
    cls.bases, cls.decorator_list = [], []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(
        torch=torch, time=time, litgpt_generate=generate,
        cache_compression_snapshot=cache_compression_snapshot,
        tqdm=SimpleNamespace(tqdm=lambda items, **kwargs: items),
        _global_rank=lambda: 0, _world_size=lambda: 1, _local_rank=lambda: 0,
        _rank_label=lambda: "CPU", _tqdm_position=lambda: 0,
        _request_sample_id=lambda req, **kwargs: kwargs["global_request_index"],
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), scope)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield scope["LogKVLM"]
    torch.set_num_threads(old_threads)


def make_lm(wrapper_class, backend, block_size=32):
    torch.manual_seed(17)
    config = Config(block_size=block_size, n_layer=2, n_embd=16, n_head=2,
                    n_query_groups=1, padded_vocab_size=32)
    lm = wrapper_class()
    lm.model = GPT(config).eval()
    lm._device, lm.tokenizer = "cpu", Tokenizer()
    lm.log_kv_dense_mode, lm.swa_window_size = backend == "dense", None
    if lm.log_kv_dense_mode:
        lm.model.set_kv_cache(batch_size=1)
    else:
        lm.model.set_log_kv_cache(batch_size=1, B=2, recent_size=8, prefill_block=2,
                                 second_order_scale=0.)
    lm._set_eval_cache = lm._reset_eval_cache
    lm.all_gather_results = lambda items, tag: items
    lm.compression_metrics, lm.ppl_metrics, lm.gen_metrics = [], [], []
    lm.max_gen_toks = 3
    return lm


def assert_cache_reset(lm):
    for block in lm.model.transformer.h:
        cache = block.attn.kv_cache
        if lm.log_kv_dense_mode:
            assert not cache.k.any()
        else:
            assert cache.token_count == cache.recent_count == 0
            assert block.attn._log_kv_pending is None


@torch.no_grad()
@pytest.mark.parametrize("backend,seq_len", [("dense", 5), ("logkv", 5), ("logkv", 28)])
def test_score_snapshot_precedes_reset_and_preserves_score(wrapper_class, backend, seq_len):
    lm = make_lm(wrapper_class, backend)
    tokens = torch.arange(1, seq_len + 1).unsqueeze(0)
    logits = lm.model(tokens, input_pos=torch.arange(seq_len))
    expected_logits = logits[0, 1:-1].float()
    targets = tokens[0, 2:]
    expected_score = expected_logits.log_softmax(-1).gather(-1, targets[:, None]).sum().item()
    expected_greedy = bool((expected_logits.argmax(-1) == targets).all())
    expected_snapshot = cache_compression_snapshot(lm.model, seq_len)

    score, greedy = lm._score_tokens(tokens[0, :2].tolist(), targets.tolist())

    assert score == pytest.approx(expected_score, abs=1e-5)
    assert greedy == expected_greedy
    record, = lm.compression_metrics
    assert record["request_type"] == "loglikelihood"
    assert {key: record[key] for key in expected_snapshot} == expected_snapshot
    assert record["processed_tokens"] == seq_len
    if seq_len == 5:
        assert record["kv_retention_ratio"] == record["kv_compression_factor"] == 1
    else:
        assert 0 < record["kv_retention_ratio"] < 1
    assert_cache_reset(lm)


@torch.no_grad()
@pytest.mark.parametrize("backend", ["dense", "logkv"])
@pytest.mark.parametrize("eos_id,generated_count", [(0, 1), (None, 3)])
def test_generation_excludes_unforwarded_last_token(wrapper_class, backend, eos_id, generated_count):
    lm = make_lm(wrapper_class, backend)
    lm.model.lm_head.weight.zero_()  # Greedy generation deterministically returns token 0.
    lm.tokenizer.eos_id = eos_id
    prompt = torch.tensor([1, 2, 3])
    expected = generate(lm.model, prompt, max_returned_tokens=6,
                        temperature=0., top_p=0., eos_id=eos_id)
    request = SimpleNamespace(args=("1 2 3", {"max_gen_toks": 3}))

    result = lm.generate_until([request])

    assert result == [lm.tokenizer.decode(expected[3:])]
    record, = lm.compression_metrics
    assert record["request_type"] == "generate_until"
    assert record["generated_tokens"] == generated_count
    assert record["processed_tokens"] == len(prompt) + generated_count - 1
    assert record["dense_slots"] == record["processed_tokens"] * 2
    assert record["retained_slots"] == record["dense_slots"]
    assert record["kv_retention_ratio"] == record["kv_compression_factor"] == 1
    assert_cache_reset(lm)


def test_rolling_reports_each_window_with_rolling_request_type(wrapper_class):
    lm = make_lm(wrapper_class, "dense", block_size=8)
    request = SimpleNamespace(args=(" ".join(str(token) for token in range(2, 21)),))

    score, = lm.loglikelihood_rolling([request])

    assert math.isfinite(score)
    assert [record["processed_tokens"] for record in lm.compression_metrics] == [8, 8, 6]
    assert all(record["request_type"] == "loglikelihood_rolling" for record in lm.compression_metrics)
    assert all(record["kv_retention_ratio"] == 1 for record in lm.compression_metrics)
    assert_cache_reset(lm)
