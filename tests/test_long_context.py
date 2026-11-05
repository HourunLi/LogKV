"""Long-context evaluation: YaRN through the LogKV cache, the launcher's length and override settings,
and the per-length BABILong tasks."""
import ast
import inspect
import json
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
                    "limit: 100\n")
    args, niah, meta = _eval_settings(long)
    assert meta["max_seq_lengths"] == [65536] and niah == "niah_single_2,niah_multikey_1"
    assert args[args.index("--limit") + 1] == "100"
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
