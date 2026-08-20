#!/usr/bin/env python
"""Export real RULER NIAH docs to the JSONL schema `semantic_stage0_dump.py`'s
``--samples`` expects.

There is no existing exporter for this in the repo: ``eval.py`` builds and
scores NIAH prompts on the fly through lm-eval-harness's RULER task, it never
writes them out as a standalone file. This script drives the *same* task
construction path ``eval.py``/``majob.sh`` already use for real NIAH evals
(``metadata={"pretrained": <tokenizer dir>, "max_seq_lengths": [...]}`` ->
``TaskManager`` -> ``get_task_dict`` -- see ``majob.sh``'s ``NIAH_BENCHMARKS``
block and ``exp/qwen1.7b-32k/dense_niah_base.yaml``'s ``metadata:`` field) and
just writes the resulting docs to disk instead of scoring them, so the same
RULER cache / offline patch already required for this project's real NIAH
evals is required here too.

Requires lm-eval>=0.4.9 (pyproject.toml's floor): ``TaskManager.__init__``
only gained its ``metadata`` kwarg in that release (absent through 0.4.2-0.4.8,
confirmed against upstream source) -- the same constraint ``eval.py`` already
has via its own unconditional ``simple_evaluate(..., metadata=metadata, ...)``
call, which internally builds ``TaskManager(metadata=metadata)`` the same way
this script does directly.

Example:
    python unused/semantic_s0_export_niah_samples.py \
      --tokenizer_dir ckpt/qwen1.7b-32k-warmup \
      --tasks niah_single_1,niah_single_2,niah_single_3 \
      --max_seq_lengths 32768 \
      --limit_per_task 8 \
      --output niah_prompts.jsonl

Then feed the result straight into the Stage-0 dump:
    python unused/semantic_stage0_dump.py \
      --checkpoint_dir ckpt/qwen1.7b-32k-warmup \
      --samples niah_prompts.jsonl \
      --output_dir stage0_dump \
      --layers 0,7,14,21,27
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Same fallback HF/RULER cache setup as eval.py's module-level block (only
# applied when the launch environment hasn't already configured these caches
# itself). Must run before lm_eval/datasets get imported, otherwise RULER's
# local-cache patch below has nothing to redirect and falls through to the
# network.
if "HF_DATASETS_CACHE" not in os.environ and "PKU" not in os.environ:
    _BASE = "/home/ma-user/work/bucket-wulan-green/wubohan/data/hf_cache"
    os.environ["HF_HOME"] = _BASE
    os.environ["HF_DATASETS_CACHE"] = f"{_BASE}/hf_cache"
    os.environ["HF_EVALUATE_CACHE"] = f"{_BASE}/evaluate"
    os.environ["HF_MODULES_CACHE"] = f"{_BASE}/modules"
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_BASE}/hub"
    os.environ["HF_HUB_CACHE"] = f"{_BASE}/hub"
    os.environ["RULER_CACHE_DIR"] = f"{_BASE}/ruler_cache"
    os.environ["NLTK_DATA"] = f"{_BASE}/nltk_data"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from lm_eval.tasks import TaskManager, get_task_dict  # noqa: E402

from litgpt.ruler_patch import apply_patch  # noqa: E402
from litgpt.semantic_s0 import parse_int_list  # noqa: E402

apply_patch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tokenizer_dir",
        required=True,
        help="HF-format tokenizer dir -- same thing eval.py's --metadata "
        "'{\"pretrained\": ...}' points at (the checkpoint dir, if it has "
        "tokenizer.json/tokenizer_config.json copied next to lit_model.pth).",
    )
    parser.add_argument("--tasks", default="niah_single_1,niah_single_2,niah_single_3")
    parser.add_argument(
        "--max_seq_lengths",
        default="32768",
        help="Comma-separated context lengths RULER should build prompts for. "
        "Stage 0 only needs long prompts, so this defaults to a single length "
        "instead of majob.sh's full sweep (which also builds short buckets "
        "used for other benchmark points).",
    )
    parser.add_argument(
        "--limit_per_task",
        type=int,
        default=8,
        help="Keep at most this many docs per task. RULER always generates "
        "500 per length internally regardless of this flag -- it only "
        "controls how many of those get written out.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Subsampling seed when limit_per_task < generated count")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    max_seq_lengths = parse_int_list(args.max_seq_lengths)

    metadata = {"pretrained": args.tokenizer_dir, "max_seq_lengths": max_seq_lengths}
    task_manager = TaskManager(metadata=metadata)
    task_dict = get_task_dict(tasks, task_manager)

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for task_name in tasks:
            task = task_dict[task_name]
            all_docs = list(task.eval_docs)
            docs = (
                all_docs
                if args.limit_per_task is None or len(all_docs) <= args.limit_per_task
                else rng.sample(all_docs, args.limit_per_task)
            )
            for i, doc in enumerate(docs):
                record = dict(doc)
                record["prompt"] = task.doc_to_text(doc)
                record["sample_id"] = f"{task_name}_{i:04d}"
                record["task_name"] = task_name
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            print(f"[export] {task_name}: wrote {len(docs)}/{len(all_docs)} generated docs", flush=True)

    print(f"[export] wrote {written} total samples to {args.output}")


if __name__ == "__main__":
    main()
