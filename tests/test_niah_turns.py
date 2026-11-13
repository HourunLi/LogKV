"""Several questions on one cache (eval.py's LogKVLM._generate_turns, eval_tasks/niah_turns): on CPU, the
answers equal greedy decoding of the whole transcript without a cache wherever the cache is exact."""

import ast
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from litgpt.config import Config
from litgpt.generate.base import generate
from litgpt.kv_compression import cache_compression_snapshot
from litgpt.model import GPT

ROOT = Path(__file__).resolve().parents[1]
STOP = 5  # the line end of the test vocabulary
SEP = " | "


class Tokenizer:
    bos_id, eos_id, alpha_line_end_ids = 1, None, (STOP,)

    def encode(self, text, device="cpu", bos=None):
        return torch.tensor([int(token) for token in text.split()], device=device, dtype=torch.long)

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens.tolist())


@pytest.fixture(scope="module")
def wrapper_class():
    tree = ast.parse((ROOT / "eval.py").read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "LogKVLM")
    names = {"_generate_turns", "generate_until", "_reset_eval_cache", "_warn_truncation", "_prefill_chunk", "_forward",
             "_generate_chunked"}
    cls.bases, cls.decorator_list = [], []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(os=__import__("os"), torch=torch, time=time, litgpt_generate=generate,
                 cache_compression_snapshot=cache_compression_snapshot,
                 _eval_progress=lambda items, desc: items, _eval_verbose=lambda: False, _global_rank=lambda: 0,
                 _world_size=lambda: 1, _rank_label=lambda: "CPU",
                 _request_sample_id=lambda req, **kwargs: kwargs["global_request_index"])
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
        "_synchronize_inference_device", "_reset_inference_peak_memory", "_inference_peak_memory",
        "summarize_inference_metrics",
    }]
    exec(compile(ast.Module(body=helpers + [cls], type_ignores=[]), str(ROOT / "eval.py"), "exec"), scope)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield scope["LogKVLM"]
    torch.set_num_threads(old_threads)


def make_lm(wrapper_class, backend, recent_size=64, block_size=64):
    torch.manual_seed(23)
    config = Config(block_size=block_size, n_layer=2, n_embd=16, n_head=2, n_query_groups=1, padded_vocab_size=32)
    lm = wrapper_class()
    lm.model = GPT(config).eval()
    reference = GPT(config).eval()
    reference.load_state_dict(lm.model.state_dict())
    lm._device, lm.tokenizer = "cpu", Tokenizer()
    lm.log_kv_dense_mode, lm.swa_window_size = backend == "dense", None
    if backend == "dense":
        lm.model.set_kv_cache(batch_size=1)
    else:
        lm.model.set_log_kv_cache(batch_size=1, B=2, recent_size=recent_size, prefill_block=2, second_order_scale=0.)
    lm._set_eval_cache = lm._reset_eval_cache
    lm.all_gather_results = lambda items, tag: items
    lm.compression_metrics, lm.gen_metrics, lm._truncation_counts = [], [], {}
    lm.max_gen_toks = 4
    return lm, reference


@torch.no_grad()
def uncached_turns(model, context, questions, max_new):
    """Greedy answers when the whole transcript is run again for every token."""
    transcript, answers = list(context), []
    for question in questions:
        transcript += question
        answer = []
        for _ in range(max_new):
            token = int(model(torch.tensor([transcript]))[0, -1].argmax())
            answer.append(token)
            transcript.append(token)
            if token == STOP:
                break
        answers.append(answer)
    return answers


@pytest.mark.parametrize("backend,recent_size", [("dense", None), ("logkv", 64)])
def test_turns_match_uncached_greedy_decoding_where_the_cache_is_exact(wrapper_class, backend, recent_size):
    lm, reference = make_lm(wrapper_class, backend, recent_size=recent_size or 64)
    context, questions = [2, 9, 4, 11, 7, 3, 8, 12, 6, 10], [[13, 14], [15, 16, 17], [18]]
    expected = uncached_turns(reference, context, questions, max_new=4)
    prompt = SEP.join([" ".join(map(str, context))] + [" ".join(map(str, q)) for q in questions])
    request = SimpleNamespace(args=(prompt, {"max_gen_toks": 4, "turn_separator": SEP}))

    result, = lm.generate_until([request])

    assert result.split(SEP) == [" ".join(map(str, answer)) for answer in expected]
    record, = lm.compression_metrics
    # Every answer but the last is forwarded with the next question.
    processed = len(context) + sum(map(len, questions)) + sum(map(len, expected)) - 1
    assert record["request_type"] == "generate_turns" and record["turns"] == 3
    assert record["processed_tokens"] == processed


def test_turns_run_on_a_compressing_cache_and_keep_the_context_query_free(wrapper_class):
    lm, _ = make_lm(wrapper_class, "logkv", recent_size=8)
    context = list(range(2, 31))
    prompt = SEP.join([" ".join(map(str, context)), "13 14", "15 16"])
    calls = []
    lm._query_line_ends = object()
    lm.model.log_kv_cue_stream = object()
    lm.model.set_gamma_queries = calls.append

    result, = lm.generate_until([SimpleNamespace(args=(prompt, {"max_gen_toks": 3, "turn_separator": SEP}))])

    assert len(result.split(SEP)) == 2 and calls == [[None]]  # no question is known when the context is cached
    record, = lm.compression_metrics
    assert 0 < record["kv_retention_ratio"] < 1  # the context was compressed before the questions came


def test_a_long_context_is_cached_in_chunks_with_the_same_answers(wrapper_class, monkeypatch):
    answers = []
    for chunk in ("0", "4"):
        monkeypatch.setenv("LOGKV_EVAL_PREFILL_CHUNK", chunk)
        lm, _ = make_lm(wrapper_class, "logkv", recent_size=8)
        prompt = SEP.join([" ".join(str(t % 30 + 2) for t in range(29)), "13 14", "15 16"])
        answers += lm.generate_until([SimpleNamespace(args=(prompt, {"max_gen_toks": 3, "turn_separator": SEP}))])
    assert answers[0] == answers[1]


def test_a_context_longer_than_the_window_loses_its_start(wrapper_class):
    lm, _ = make_lm(wrapper_class, "dense", block_size=24)
    prompt = SEP.join([" ".join(map(str, range(2, 32))), "13 14", "15"])
    result, = lm.generate_until([SimpleNamespace(args=(prompt, {"max_gen_toks": 2, "turn_separator": SEP}))])
    assert len(result.split(SEP)) == 2
    assert lm._truncation_counts  # warned
    assert lm.compression_metrics[0]["prompt_len"] == 24 - 3 - 2 * 2


@pytest.mark.parametrize("backend", ["dense", "logkv"])
@pytest.mark.parametrize("stop_after_first", [False, True])
def test_full_eval_can_summarize_ordinary_and_multiturn_generation(wrapper_class, backend, stop_after_first):
    lm, _ = make_lm(wrapper_class, backend)
    with torch.no_grad():
        lm.model.lm_head.weight.zero_()  # All answers deterministically contain token 0.
    if stop_after_first:
        lm.tokenizer.eos_id = 0
        lm.tokenizer.alpha_line_end_ids = (0,)
    requests = [
        SimpleNamespace(args=("1 2 3", {"max_gen_toks": 3})),
        SimpleNamespace(args=(SEP.join(["1 2 3", "4", "5 6"]),
                              {"max_gen_toks": 3, "turn_separator": SEP})),
    ]

    results = lm.generate_until(requests)
    summarize = lm._generate_turns.__globals__["summarize_inference_metrics"]
    summary = summarize(lm.gen_metrics, [])["generation"]

    tokens_per_answer = 1 if stop_after_first else 3
    assert results == [" ".join(["0"] * tokens_per_answer),
                       SEP.join([" ".join(["0"] * tokens_per_answer)] * 2)]
    assert summary["requests"] == 3  # One ordinary request plus two answer turns.
    assert summary["prompt_tokens"] == 3 + 4 + 3  # Later turns also forward the previous answer's last token.
    assert summary["generated_tokens"] == 3 * tokens_per_answer
    assert summary["decode_tokens"] == 3 * (tokens_per_answer - 1)
    assert summary["total_time_s"] == pytest.approx(summary["prefill_time_s"] + summary["decode_time_s"])
    assert summary["avg_ttft_ms"] == pytest.approx(1000 * summary["prefill_time_s"] / 3)
    if stop_after_first:
        assert summary["decode_time_s"] == 0
        assert summary["decode_tokens_per_sec"] is summary["decode_ms_per_token"] is None
    turns = lm.gen_metrics[1:]
    assert [row["turn_index"] for row in turns] == [1, 2]
    assert all(row["request_type"] == "generate_turns" and row["sample_id"] == 1
               and row["rank"] == 0 and row["context_len"] == 3 for row in turns)
    assert sum(row["prompt_len"] + row["decode_tokens"] for row in turns) == lm.compression_metrics[-1]["processed_tokens"]
    assert all(row["peak_memory_allocated_bytes"] is row["peak_memory_reserved_bytes"] is None
               for row in lm.gen_metrics)


@pytest.mark.parametrize("backend", ["dense", "logkv"])
@pytest.mark.parametrize("stop_after_first", [False, True])
def test_multiturn_timing_excludes_text_decoding_and_separates_each_first_token(
    wrapper_class, monkeypatch, backend, stop_after_first,
):
    lm, _ = make_lm(wrapper_class, backend)
    with torch.no_grad():
        lm.model.lm_head.weight.zero_()
    if stop_after_first:
        lm.tokenizer.alpha_line_end_ids = (0,)
    clock = [10.]
    forward, decode = lm._forward, lm.tokenizer.decode

    def timed_forward(tokens, *args):
        logits = forward(tokens, *args)
        clock[0] += 4. if tokens.numel() > 1 else 2.
        return logits

    def slow_text_decode(tokens):
        clock[0] += 100.  # Postprocessing must not inflate model timing.
        return decode(tokens)

    lm._forward, lm.tokenizer.decode = timed_forward, slow_text_decode
    monkeypatch.setitem(lm._generate_turns.__globals__, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    lm._generate_turns(SEP.join(["1 2 3", "4", "5 6"]), SEP, 3, sample_id="timed")

    assert len(lm.gen_metrics) == 2
    for row in lm.gen_metrics:
        count = 1 if stop_after_first else 3
        assert row["generated_tokens"] == count
        assert row["decode_tokens"] == count - 1
        assert row["prefill_time_s"] == 4.
        assert row["ttft_ms"] == 4000.
        assert row["decode_time_s"] == 2. * (count - 1)
        assert row["total_time_s"] == 4. + 2. * (count - 1)
        assert row["prefill_tokens_per_sec"] == row["prompt_len"] / 4.


class WordCounter:
    """What RULER's generator needs of a tokenizer: the token count of a text."""

    def __call__(self, text):
        return SimpleNamespace(input_ids=text.split())


@pytest.fixture
def turns_utils():
    pytest.importorskip("lm_eval")
    import importlib.util

    spec = importlib.util.spec_from_file_location("niah_turns_utils", ROOT / "eval_tasks" / "niah_turns" / "niah_turns_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multiquery_samples_become_a_context_and_one_question_per_key(turns_utils):
    from lm_eval.tasks.ruler.niah_utils import TEMPLATE
    from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

    samples = generate_samples(get_haystack(type_haystack="repeat"), max_seq_length=2048, template=TEMPLATE,
                               type_haystack="repeat", type_needle_k="words", type_needle_v="numbers",
                               num_needle_q=4, num_samples=3, tokens_to_generate=384, TOKENIZER=WordCounter())
    for sample in samples:
        doc = turns_utils.turns_of(sample)
        context, *questions = doc["input"].split(turns_utils.SEPARATOR)
        assert len(questions) == 4 and "What are all" not in context and context.startswith("Some special magic")
        for key, value, question in zip(doc["keys"], doc["outputs"], questions):
            assert f"special magic numbers for {key} is: {value}." in context  # RULER's needle
            assert question.startswith(f"\nWhat is the special magic number for {key} ") and value not in question
        assert sorted(doc["outputs"]) == sorted(sample["outputs"])
    assert turns_utils.keys_of("The special magic numbers for a-b, c, d-e, and f mentioned in the provided "
                               "text are") == ["a-b", "c", "d-e", "f"]
    doc = dict(outputs=["111", "222", "333", "444"])
    sep = turns_utils.SEPARATOR
    assert turns_utils.process_results(doc, [sep.join([" 111.\n", " 999", " 333.\n"])]) == dict(
        acc=0.5, turn1=1.0, turn2=0.0, turn3=1.0, turn4=0.0)  # a missing answer scores 0


def test_the_tasks_ask_eval_py_for_turns(turns_utils, monkeypatch):
    import functools
    import inspect

    import lm_eval.tasks.ruler.common_utils as common
    import lm_eval.tasks.ruler.prepare_niah as prepare
    from lm_eval.tasks import get_task_dict

    real, filler = prepare.generate_samples, prepare.get_haystack(type_haystack="repeat")
    calls = []

    def small(haystack, TOKENIZER=None, **kwargs):  # the real generator on a short filler haystack
        calls.append(kwargs)
        return real(filler, TOKENIZER, **dict(
            kwargs, max_seq_length=2048, type_haystack="repeat", num_samples=2))
    # niah_turns_utils binds these names when lm-eval loads it for the task.
    monkeypatch.setattr(prepare, "generate_samples", small)
    monkeypatch.setattr(prepare, "get_haystack", lambda type_haystack: None)
    # lm-eval's get_tokenizer is cached by its arguments, so they must be hashable.
    monkeypatch.setattr(common, "get_tokenizer", functools.cache(lambda tokenizer=None, pretrained=None: WordCounter()))
    source = ROOT / "eval.py"
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_task_manager"]
    scope = dict(inspect=inspect, EVAL_TASKS_DIR=ROOT / "eval_tasks")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    # The main stage's metadata, as eval.sh / majob.sh pass it: max_seq_lengths is a list.
    manager = scope["_task_manager"]({"pretrained": "unused", "max_seq_lengths": [262144]})
    assert {f"niah_turns_{n}" for n in ("32k", "64k", "128k", "256k")} <= set(manager.all_tasks)
    task = get_task_dict(["niah_turns_256k"], manager)["niah_turns_256k"]
    assert calls[-1]["max_seq_length"] == 262144 and calls[-1]["num_needle_q"] == 4
    task.build_all_requests(limit=1, rank=0, world_size=1)
    request, = task.instances
    prompt, gen_kwargs = request.args
    doc = next(iter(task.test_docs()))
    assert prompt == doc["input"] and len(prompt.split(turns_utils.SEPARATOR)) == 5
    assert gen_kwargs["turn_separator"] == turns_utils.SEPARATOR and gen_kwargs["max_gen_toks"] == 32
