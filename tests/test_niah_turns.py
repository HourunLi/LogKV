"""Several questions on one cache (eval.py's LogKVLM._generate_turns, eval_tasks/niah_turns): on CPU, the
answers equal greedy decoding of the whole transcript without a cache wherever the cache is exact."""

import ast
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from litgpt.config import Config
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
    cls = next(node for node in ast.parse((ROOT / "eval.py").read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "LogKVLM")
    names = {"_generate_turns", "generate_until", "_reset_eval_cache", "_warn_truncation"}
    cls.bases, cls.decorator_list = [], []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(torch=torch, time=time, cache_compression_snapshot=cache_compression_snapshot,
                 _eval_progress=lambda items, desc: items, _eval_verbose=lambda: False, _global_rank=lambda: 0,
                 _world_size=lambda: 1, _rank_label=lambda: "CPU",
                 _request_sample_id=lambda req, **kwargs: kwargs["global_request_index"])
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / "eval.py"), "exec"), scope)
    return scope["LogKVLM"]


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


def test_a_context_longer_than_the_window_loses_its_start(wrapper_class):
    lm, _ = make_lm(wrapper_class, "dense", block_size=24)
    prompt = SEP.join([" ".join(map(str, range(2, 32))), "13 14", "15"])
    result, = lm.generate_until([SimpleNamespace(args=(prompt, {"max_gen_toks": 2, "turn_separator": SEP}))])
    assert len(result.split(SEP)) == 2
    assert lm._truncation_counts  # warned
    assert lm.compression_metrics[0]["prompt_len"] == 24 - 3 - 2 * 2


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
    monkeypatch.setattr(common, "get_tokenizer", lambda **kwargs: WordCounter())
    source = ROOT / "eval.py"
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_task_manager"]
    scope = dict(inspect=inspect, EVAL_TASKS_DIR=ROOT / "eval_tasks")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    manager = scope["_task_manager"]({"pretrained": "unused"})
    assert {f"niah_turns_{n}" for n in ("32k", "64k", "128k", "256k")} <= set(manager.all_tasks)
    task = get_task_dict(["niah_turns_256k"], manager)["niah_turns_256k"]
    assert calls[-1]["max_seq_length"] == 262144 and calls[-1]["num_needle_q"] == 4
    task.build_all_requests(limit=1, rank=0, world_size=1)
    request, = task.instances
    prompt, gen_kwargs = request.args
    doc = next(iter(task.test_docs()))
    assert prompt == doc["input"] and len(prompt.split(turns_utils.SEPARATOR)) == 5
    assert gen_kwargs["turn_separator"] == turns_utils.SEPARATOR and gen_kwargs["max_gen_toks"] == 32
