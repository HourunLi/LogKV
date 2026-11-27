#!/usr/bin/env python
"""Export a frozen, fully-annotated RULER NIAH sample set for the 32K
Dense / SinkWindow / SemanticLogKV comparison.

Why this exists as a standalone export step, run once, rather than having
each branch's eval call the RULER generator on demand: the task's own
protocol requires "生成后落盘，由三个分支读取同一份数据" (generate once,
persist to disk, all three branches read the *same* file) and explicitly
warns "不能假设三次独立调用数据生成器一定会得到相同样本" (don't assume three
independent generator calls produce the same samples). This script drives
the same real task-construction path ``eval.py``/``majob.sh`` already use for
production NIAH evals (``metadata={"pretrained": ..., "max_seq_lengths": [...]}``
-> ``TaskManager`` -> ``get_task_dict``, see ``majob.sh``'s ``NIAH_BENCHMARKS``
block), so the exported prompts are the real eval prompts, not a
reimplementation -- it just also writes them to disk with full metadata
instead of only scoring them in-process.

This supersedes ``unused/semantic_s0_export_niah_samples.py`` *for this
experiment's frozen evaluation set specifically* (that script's own docstring
says it exists to feed Stage-0 k/v dumps, a separate, still-active research
track this repo's CLAUDE.md lists under "已实现" -- it is intentionally left
untouched here rather than moved, since other Stage-0 workflows may still
reference its exact path). The extension beyond that script: real needle/
answer span annotation via ``find_needle_spans`` (including the
``answer_exact`` span added to ``litgpt/needle_spans.py`` alongside this
script, specifically so SinkWindow-visibility bucketing can key off where the
answer *content* sits rather than the sentence it's embedded in), the actual
tokenized prompt length, and task/package version metadata.

Known limitation, stated rather than hidden: RULER's own internal
document-generation seed is not independently exposed by this script (or by
``eval.py``/``majob.sh``, which don't control it either) -- reproducibility
across the three branches comes from running this export exactly once and
having every branch's eval read the resulting file, not from a guaranteed
bit-identical regeneration on a second run. Do not re-run this script and
expect the same samples; keep the one output file this produces.

Behavior with more than one ``--max_seq_lengths`` value in one call is
unverified in this environment (no ``lm_eval`` installed here to inspect
RULER's internals against) -- this experiment only needs 32768 (the task's
own scope: "只测 32K"), so the default is a single length; pass multiple only
for building a separate calibration set, and check the printed per-task doc
count looks sane before trusting the result.

Usage:
    python scripts/export_niah_samples.py \\
      --tokenizer_dir <base checkpoint dir with tokenizer.json> \\
      --tasks niah_single_1,niah_single_2,niah_single_3 \\
      --max_seq_lengths 32768 \\
      --output niah_samples_32k.jsonl
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Same fallback HF/RULER offline-cache setup as eval.py's module-level block
# and unused/semantic_s0_export_niah_samples.py -- must run before
# lm_eval/datasets get imported, or RULER's local-cache patch below has
# nothing to redirect and falls through to the network. Only applied when the
# launch environment hasn't already configured these caches itself.
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

from litgpt.needle_spans import find_needle_spans  # noqa: E402
from litgpt.ruler_patch import apply_patch  # noqa: E402
from litgpt.semantic_s0 import parse_int_list  # noqa: E402
from litgpt.tokenizer import Tokenizer  # noqa: E402

apply_patch()


def _lm_eval_version() -> str:
    try:
        return importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _prompt_token_count(tokenizer: Tokenizer, prompt: str) -> int:
    encoded = tokenizer.encode(prompt)
    return int(encoded.numel()) if hasattr(encoded, "numel") else len(encoded)


def _answer_values(doc: dict) -> list[str]:
    for key in ("outputs", "output", "answer", "answers", "target", "targets"):
        v = doc.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return [str(v)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tokenizer_dir",
        required=True,
        help="HF-format tokenizer dir (tokenizer.json/tokenizer_config.json) -- "
        "same thing eval.py's --metadata '{\"pretrained\": ...}' points at.",
    )
    parser.add_argument("--tasks", default="niah_single_1,niah_single_2,niah_single_3")
    parser.add_argument(
        "--max_seq_lengths",
        default="32768",
        help="Comma-separated context lengths RULER should build prompts for. "
        "This experiment's scope is 32K only; see the module docstring before passing more than one.",
    )
    parser.add_argument(
        "--limit_per_task",
        type=int,
        default=None,
        help="Keep at most this many docs per task. Default (unset) keeps all -- "
        "RULER generates 500/task/length internally regardless of this flag, and the task spec "
        "prefers keeping that existing 500-per-task setting.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Subsampling seed, only consulted when --limit_per_task subsamples")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError(f"--tasks must name at least one task, got {args.tasks!r}")
    max_seq_lengths = parse_int_list(args.max_seq_lengths)
    if not max_seq_lengths:
        raise ValueError(f"--max_seq_lengths must list at least one length, got {args.max_seq_lengths!r}")
    if args.limit_per_task is not None and args.limit_per_task < 1:
        raise ValueError(f"--limit_per_task must be >= 1 (or omitted), got {args.limit_per_task}")

    tokenizer = Tokenizer(args.tokenizer_dir)
    metadata = {"pretrained": args.tokenizer_dir, "max_seq_lengths": max_seq_lengths}
    task_manager = TaskManager(metadata=metadata)
    task_dict = get_task_dict(tasks, task_manager)

    rng = random.Random(args.seed)
    lm_eval_version = _lm_eval_version()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    span_source_hits: dict[str, int] = {}
    samples_with_no_span = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for task_name in tasks:
            task = task_dict[task_name]
            all_docs = list(task.eval_docs)
            subsampled = args.limit_per_task is not None and len(all_docs) > args.limit_per_task
            docs = rng.sample(all_docs, args.limit_per_task) if subsampled else all_docs
            for i, doc in enumerate(docs):
                doc_dict: dict[str, Any] = dict(doc)
                prompt = task.doc_to_text(doc)
                sample_id = f"{task_name}_{i:04d}"
                answer_values = _answer_values(doc_dict)
                spans = [s.to_dict() for s in find_needle_spans(tokenizer, prompt, doc=doc_dict)]
                for s in spans:
                    span_source_hits[s["source"]] = span_source_hits.get(s["source"], 0) + 1
                if not spans:
                    samples_with_no_span += 1
                record = {
                    "sample_id": sample_id,
                    "task_name": task_name,
                    "task_version": getattr(task, "VERSION", None),
                    "lm_eval_version": lm_eval_version,
                    "prompt": prompt,
                    "answer": answer_values,
                    "prompt_token_count": _prompt_token_count(tokenizer, prompt),
                    # Every span find_needle_spans found, tagged by source --
                    # "answer_exact" is the tight span visibility bucketing
                    # must use; "answer_sentence"/"*_regex" are the wider
                    # needle-sentence spans kept for diagnostics per the task
                    # spec ("另保留整个 needle 句子的坐标供诊断").
                    "needle_answer_spans": spans,
                    "raw_doc": doc_dict,
                    "export_subsample_seed": args.seed if subsampled else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            print(f"[export] {task_name}: wrote {len(docs)}/{len(all_docs)} docs", flush=True)

    if written == 0:
        raise RuntimeError(
            f"wrote 0 samples to {args.output} -- every task in {tasks} produced an empty eval_docs "
            "set; downstream eval would silently see no samples, refusing to write an unusable JSONL"
        )

    print(f"[export] wrote {written} total samples to {args.output}")
    print(
        "[export] span source hit counts (sanity check -- a source dropping to 0 across a real "
        "export likely means a RULER prompt-template change broke that regex/heuristic in "
        f"litgpt/needle_spans.py, not that this task genuinely has no needle): {span_source_hits}"
    )
    if samples_with_no_span:
        print(
            f"[export] WARNING: {samples_with_no_span}/{written} samples got NO span at all from "
            "find_needle_spans -- SinkWindow-visibility bucketing has no answer_exact span to key "
            "off for these. Investigate before treating this export as usable for the formal comparison."
        )


if __name__ == "__main__":
    main()
