"""Long-context evaluation: YaRN through the LogKV cache, the launcher's length and override settings,
and the per-length BABILong tasks."""
import ast
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from litgpt.config import Config
from litgpt.model import GPT, build_rope_cache

ROOT = Path(__file__).resolve().parents[1]
YARN = dict(factor=4.0, beta_fast=32, beta_slow=1, original_max_seq_len=64)
KWARGS = dict(batch_size=1, B=8, prefill_block=8, second_order_scale=0., semantic_clusters=True, cluster_k_max=4,
              semantic_flush_granularity=8, semantic_anchor_mode="mid", semantic_replay_updates=True,
              alpha_exact_tokens=8, alpha_span_max_tokens=4)


def test_yarn_rope_reaches_the_logkv_cache():
    torch.manual_seed(3)
    config = Config(block_size=256, n_layer=2, n_embd=32, n_head=4, n_query_groups=2, vocab_size=41,
                    padding_multiple=1, rotary_percentage=1., rope_adjustments=YARN)
    model = GPT(config).eval()
    model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7])
    scaled = build_rope_cache(256, config.rope_n_elem, base=config.rope_base, extra_config=YARN)[0]
    assert torch.allclose(model.cos, scaled) and not torch.allclose(model.cos, build_rope_cache(256, config.rope_n_elem)[0])
    ids = torch.randint(0, 41, (48,))
    with torch.no_grad():
        full = model(ids[None]).float()
        model.set_log_kv_cache(recent_size=64, **KWARGS)
        assert torch.allclose(model.transformer.h[0].attn.kv_cache.cos_cache, model.cos)
        cached = model(ids[None], input_pos=torch.arange(48)).float()
    assert torch.allclose(cached, full, atol=1e-4)  # inside the recent window the cache is exact


def _eval_settings(config):
    source = (ROOT / "eval.sh").read_text()
    start = source.index('CONFIG_EXPORTS=$("${PYTHON_BIN}" ')
    read_config = source[start:source.index('eval "${CONFIG_EXPORTS}"', start) + len('eval "${CONFIG_EXPORTS}"')]
    start = source.index('LOG_KV_ARG_LIST=(')
    build_args = source[start:source.index('\nDIAG_ARGS=', start)]
    start = source.index('DEFAULT_NIAH_BENCHMARKS="niah_single_1')
    niah = source[start:source.index('\n', source.index('NIAH_BENCHMARKS=${3:-', start))]
    start = source.index("# RULER / NIAH context lengths")
    meta = source[start:source.index('\n', source.index("META=", start))]
    shell = '\n'.join(('set -e', 'CONFIG_FILE=$1', 'set --', read_config, build_args, niah, meta,
                       'printf "%s\\n" "${LOG_KV_ARG_LIST[@]}"', 'echo "NIAH=${NIAH_BENCHMARKS}"', 'echo "META=${META}"'))
    env = {**os.environ, "PYTHON_BIN": sys.executable,
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    shell += '\necho "LONG_PPL=${CONFIG_LONG_PPL_DATA}"'
    out = subprocess.run(["bash", "-c", shell, "bash", str(config)], capture_output=True, text=True, check=True,
                         env=env).stdout.splitlines()
    niah_list = next(line[5:] for line in out if line.startswith("NIAH="))
    meta = json.loads(next(line[5:] for line in out if line.startswith("META=")))
    return out, niah_list, meta


def test_the_launcher_reads_lengths_tasks_and_overrides_from_the_yaml(tmp_path):
    plain = tmp_path / "plain.yaml"
    plain.write_text("save_path: /tmp/unused-checkpoint\n")
    args, niah, meta = _eval_settings(plain)
    assert meta["max_seq_lengths"] == [1024, 2048, 4096, 8192, 16384, 32768] and "--config_overrides" not in args
    assert niah.startswith("niah_single_1,niah_single_2,niah_single_3,niah_multikey_1")
    long = tmp_path / "long.yaml"
    long.write_text("save_path: /tmp/unused-checkpoint\neval_max_seq_lengths: [65536]\n"
                    "niah_benchmark: [niah_single_2, niah_multikey_1]\n"
                    "config_overrides:\n  block_size: 135168\n  rope_adjustments: {factor: 4.0, beta_fast: 32}\n"
                    "limit: 100\nlong_ppl_data: /data/prolong-heldout\n")
    args, niah, meta = _eval_settings(long)
    assert meta["max_seq_lengths"] == [65536] and niah == "niah_single_2,niah_multikey_1"
    assert args[args.index("--limit") + 1] == "100" and "LONG_PPL=/data/prolong-heldout" in args
    overrides = json.loads(args[args.index("--config_overrides") + 1])
    assert overrides == {"block_size": 135168, "rope_adjustments": {"factor": 4.0, "beta_fast": 32}}


def test_per_length_babilong_tasks_load_their_length(monkeypatch):
    pytest.importorskip("lm_eval")
    import datasets

    calls = []

    def fake(path, name=None, split=None):
        calls.append((path, name, split))
        return datasets.Dataset.from_dict({"input": ["Mary went to the kitchen."], "question": ["Where is Mary?"],
                                           "target": ["kitchen"]})
    monkeypatch.setattr(datasets, "load_dataset", fake)
    source = ROOT / "eval.py"
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_task_manager"]
    scope = dict(inspect=inspect, EVAL_TASKS_DIR=ROOT / "eval_tasks")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    manager = scope["_task_manager"]({"pretrained": "unused"})
    names = {f"babilong_qa{i}_{n}" for i in range(1, 6) for n in ("32k", "64k", "128k")}
    assert names <= set(manager.all_tasks) and {"babilong_64k", "babilong_128k"} <= set(manager.all_groups)

    from lm_eval.tasks import get_task_dict
    task = get_task_dict(["babilong_qa2_128k"], manager)["babilong_qa2_128k"]
    doc = list(task.test_docs())[0]
    assert calls[-1] == ("RMT-team/babilong-1k-samples", "128k", "qa2")
    assert task.doc_to_text(doc) == "Mary went to the kitchen.\nWhere is Mary?" and task.config.num_fewshot == 2
    assert task.process_results(doc, [" The most recent location of Mary is kitchen."]) == {"acc": 1.0}


def test_the_dense_kv_cache_needs_no_square_mask_and_matches_the_uncached_model():
    torch.manual_seed(5)
    config = Config(block_size=128, n_layer=2, n_embd=32, n_head=4, n_query_groups=2, vocab_size=41,
                    padding_multiple=1, rotary_percentage=1.)
    model = GPT(config).eval()
    ids = torch.randint(0, 41, (2, 24))
    with torch.no_grad():
        full = model(ids).float()
        model.set_kv_cache(batch_size=2)
        assert model.mask_cache is None  # no max_seq_length x max_seq_length mask
        prefill = model(ids[:, :16], input_pos=torch.arange(16)).float()  # causal from position 0
        steps = [model(ids[:, t:t + 1], input_pos=torch.tensor([t]), input_pos_maxp1=t + 1).float()
                 for t in range(16, 20)]
        chunk = model(ids[:, 20:24], input_pos=torch.arange(20, 24)).float()  # a later chunk: its own mask
    assert torch.allclose(prefill, full[:, :16], atol=1e-5)
    assert torch.allclose(torch.cat(steps, 1), full[:, 16:20], atol=1e-5)
    assert torch.allclose(chunk, full[:, 20:24], atol=1e-5)
    with torch.no_grad():  # per-row positions (2-D input_pos)
        model.set_kv_cache(batch_size=2)
        model(ids[:, :16], input_pos=torch.arange(16))
        rows = model(ids[:, 16:18], input_pos=torch.arange(16, 18).expand(2, -1)).float()
    assert torch.allclose(rows, full[:, 16:18], atol=1e-5)


@pytest.fixture
def word_tokenizer(tmp_path):
    """A whitespace word-level tokenizer saved where AutoTokenizer can load it."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"[UNK]": 0, **{f"w{i}": i + 1 for i in range(50)}}
    model = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    model.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    model.decoder = tokenizers.decoders.WordPiece(prefix="##")  # joins words with spaces
    path = tmp_path / "tok"
    transformers.PreTrainedTokenizerFast(tokenizer_object=model, unk_token="[UNK]").save_pretrained(path)
    return str(path)


def test_long_ppl_scores_the_tail_of_the_same_long_documents(tmp_path, monkeypatch, word_tokenizer):
    pytest.importorskip("lm_eval")
    import importlib.util

    import pyarrow as pa
    import pyarrow.parquet as pq

    spec = importlib.util.spec_from_file_location("long_ppl_utils", ROOT / "eval_tasks" / "long_ppl" / "long_ppl_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    monkeypatch.setattr(utils, "LONGEST", 64)
    monkeypatch.setattr(utils, "TAIL", 8)
    words = [f"w{i % 50}" for i in range(200)]
    # A short row, then two long ones; rows of "text" as in ProLong's parquet shards.
    pq.write_table(pa.table({"text": [" ".join(words[:10]), " ".join(words[:100]), " ".join(words[50:200])]}),
                   tmp_path / "shard-0.parquet")
    monkeypatch.setenv("LONG_PPL_DATA", str(tmp_path))
    monkeypatch.setenv("LONG_PPL_CACHE", str(tmp_path / "cache"))
    rows = utils.build(32, {"pretrained": word_tokenizer})["test"]
    assert len(rows) == 2 and rows[0]["context"].split() == words[:24] and rows[0]["continuation"].split() == words[24:32]
    assert rows[1]["context"].split()[0] == "w0" and rows[0]["tail_tokens"] == 8  # the second row starts at w50
    longer = utils.build(48, {"pretrained": word_tokenizer})["test"]
    assert longer[0]["continuation"].split() == words[40:48]  # the same document, more context before the tail
    assert list((tmp_path / "cache").iterdir())  # the selection is cached
    # A bucket mount refuses the rename: the cache is copied into place instead.
    monkeypatch.setenv("LONG_PPL_CACHE", str(tmp_path / "bucket"))

    def refuse(src, dst):
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(utils.os, "replace", refuse)
    assert utils.build(32, {"pretrained": word_tokenizer})["test"]["continuation"] == rows["continuation"]
    assert [f.suffix for f in (tmp_path / "bucket").iterdir()] == [".json"]
    monkeypatch.undo()
    monkeypatch.setattr(utils, "LONGEST", 64)
    monkeypatch.setattr(utils, "TAIL", 8)
    monkeypatch.setenv("LONG_PPL_DATA", str(tmp_path))
    monkeypatch.setenv("LONG_PPL_CACHE", str(tmp_path / "cache"))
    scores = [utils.process_results(row, [(-8. * math.log(4.), False)]) for row in rows]
    assert utils.perplexity([s["ppl"] for s in scores]) == pytest.approx(4.)
    monkeypatch.setenv("LONG_PPL_DATA", str(tmp_path / "missing"))
    monkeypatch.setenv("LONG_PPL_CACHE", str(tmp_path / "cache2"))
    with pytest.raises(FileNotFoundError):
        utils.build(32, {"pretrained": word_tokenizer})


def test_long_ppl_tasks_are_found_by_eval_py():
    pytest.importorskip("lm_eval")
    source = ROOT / "eval.py"
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_task_manager"]
    scope = dict(inspect=inspect, EVAL_TASKS_DIR=ROOT / "eval_tasks")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    manager = scope["_task_manager"]({"pretrained": "unused"})
    assert {"long_ppl_32k", "long_ppl_64k", "long_ppl_128k"} <= set(manager.all_tasks)


def test_the_fetch_script_caches_the_ppl_documents_the_tasks_read(tmp_path, monkeypatch, word_tokenizer):
    pytest.importorskip("lm_eval")
    import importlib.util

    import pyarrow as pa
    import pyarrow.parquet as pq

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    fetch = load("fetch_long_eval_data", ROOT / "unused" / "fetch_long_eval_data.py")
    utils = load("long_ppl_utils_check", ROOT / "eval_tasks" / "long_ppl" / "long_ppl_utils.py")
    assert "academic_single" in fetch.longbench2_subsets() and len(fetch.longbench2_subsets()) == 20
    words = " ".join(f"w{i % 50}" for i in range(140000))
    pq.write_table(pa.table({"text": [words]}), tmp_path / "heldout.parquet")
    monkeypatch.setenv("LONG_PPL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LONG_PPL_DOCS", "1")
    lines = []
    fetch.cache_ppl(str(tmp_path / "heldout.parquet"), word_tokenizer, lines)
    assert "1 documents of >= 131072 tokens" in lines[0]
    stats = []
    fetch.ppl_data_stats(str(tmp_path / "heldout.parquet"), word_tokenizer, utils, stats)
    assert "1 rows in 1 file(s)" in stats[0] and stats[1].endswith("[140000]")
    # The tasks find the same cache whatever path names the same tokenizer files.
    copy = tmp_path / "tok-copy"
    copy.mkdir()
    for f in Path(word_tokenizer).iterdir():
        (copy / f.name).write_bytes(f.read_bytes())
    key = utils.cache_key(str(tmp_path / "heldout.parquet"), str(copy), 1, False)
    assert (tmp_path / "cache" / f"{key}.json").is_file()


def test_longbench2_subsets_come_from_the_snapshot_files_when_the_card_has_no_configs(tmp_path, monkeypatch):
    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq

    from litgpt import longbench2_patch as patch

    files = ["academic_single/train-00000.parquet", "data/academic_multi-00000-of-00001.parquet",
             "legal_single.jsonl", "README.md"]
    assert patch.subset_files(files, "academic_single") == ["academic_single/train-00000.parquet"]
    assert patch.subset_files(files, "academic_multi") == ["data/academic_multi-00000-of-00001.parquet"]
    assert patch.subset_files(files, "legal_single") == ["legal_single.jsonl"] and not patch.subset_files(files, "legal")
    (tmp_path / "academic_single").mkdir()
    pq.write_table(pa.table({"question": ["q1", "q2"], "answer": [0, 1]}), tmp_path / "academic_single" / "train-0.parquet")
    pq.write_table(pa.table({"question": ["other"], "answer": [2]}), tmp_path / "multinews.parquet")
    monkeypatch.setattr(patch, "snapshot", lambda: (str(tmp_path), ["academic_single/train-0.parquet", "multinews.parquet"]))
    original = datasets.load_dataset

    def no_configs(path=None, name=None, *args, **kwargs):
        if path == patch.REPO:
            raise ValueError(f"BuilderConfig '{name}' not found. Available: ['default']")
        return original(path, name, *args, **kwargs)
    monkeypatch.setattr(datasets, "load_dataset", no_configs)
    patch.apply_patch()
    loaded = datasets.load_dataset(path=patch.REPO, name="academic_single")  # as lm-eval's task calls it
    assert loaded["train"]["question"] == ["q1", "q2"]
    assert datasets.load_dataset(patch.REPO, name="multinews", split="train")["question"] == ["other"]
    with pytest.raises(ValueError, match="no file"):
        datasets.load_dataset(patch.REPO, name="table_qa")


def test_the_fetch_script_skips_certificate_checks_and_waits_out_rate_limits(monkeypatch):
    import importlib.util
    import io

    import requests

    calls, waits = [], []

    def record(self, method, url, **kwargs):
        calls.append(kwargs)
        response = requests.Response()
        response.status_code, response.raw = (429 if len(calls) < 3 else 200), io.BytesIO()
        response.headers["Retry-After"] = "7"
        return response

    # Restored after the test: Session.request (with the script's wrapper), the variables it sets, time.sleep.
    monkeypatch.setattr(requests.Session, "request", record)
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if not k.startswith("HF_HUB_")})
    monkeypatch.setattr("time.sleep", waits.append)
    path = ROOT / "unused" / "fetch_long_eval_data.py"
    spec = importlib.util.spec_from_file_location("fetch_long_eval_data_ssl", path)
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    fetch.patch_requests(verify_ssl=False)
    # huggingface_hub's sessions go through Session.request too.
    assert requests.get("https://hf-mirror.com/x", timeout=5).status_code == 200
    assert waits == [7, 7] and all(c["verify"] is False and c["timeout"] == 5 for c in calls)
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0" and os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "300"


def test_the_fetch_script_spots_files_that_are_not_parquet(tmp_path):
    import importlib.util

    import pyarrow as pa
    import pyarrow.parquet as pq

    spec = importlib.util.spec_from_file_location("fetch_long_eval_data_pq", ROOT / "unused" / "fetch_long_eval_data.py")
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    good, pointer = tmp_path / "good.parquet", tmp_path / "pointer.parquet"
    pq.write_table(pa.table({"x": [1]}), good)
    pointer.write_text("version https://git-lfs.github.com/spec/v1\\noid sha256:abc\\nsize 123\\n")
    assert fetch.is_parquet(good) and not fetch.is_parquet(pointer) and not fetch.is_parquet(tmp_path / "missing")
    assert "git-lfs" in fetch.describe(pointer) and fetch.describe(tmp_path / "missing") == "missing"
