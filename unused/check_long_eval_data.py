"""Check, offline and read-only, that the long-context evaluation data is complete (what fetch_long_eval_data.py
put into eval.sh's caches). Run it the way eval.sh runs, with diag.sh's offline environment:

    bash diag.sh unused/check_long_eval_data.py [--tokenizer <Qwen3-1.7B-Base dir>]

with LONG_PPL_DATA / LONG_PPL_BOOKS exported as for eval.sh. Each line says OK or MISSING / BAD:
    BABILong      RMT-team/babilong-1k-samples 32k / 64k / 128k and RMT-team/babilong 256k, qa1 ... qa5 (rows)
    LongBench v2  the hub snapshot's files are whole parquet, and every subset lm-eval's longbench2 reads loads
                  (503 questions in all)
    RULER QA      SQuAD v2 dev and HotpotQA dev in RULER_CACHE_DIR
    InfiniteBench the ten tasks' jsonl files (INFINITEBENCH_DIR, or the hub cache) are whole, and every sample
                  makes a prompt and an answer
    long_ppl      LONG_PPL_DATA: documents of >= 131072 and >= 262144 tokens
    books         LONG_PPL_BOOKS: books of >= 262144 tokens (want LONG_PPL_DOCS, 100)
The ppl lines need --tokenizer (they read the cached selection, or select again when it is missing).
Exits 1 when anything is missing.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BABILONG = [("RMT-team/babilong-1k-samples", n) for n in ("32k", "64k", "128k")] + [("RMT-team/babilong", "256k")]
RULER_QA = ("dev-v2.0.json", "hotpot_dev_distractor_v1.json")


def whole_parquet(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            return path.stat().st_size >= 12 and head == b"PAR1" and f.read(4) == b"PAR1"
    except OSError:
        return False


def check_babilong(report) -> None:
    import datasets

    for path, length in BABILONG:
        for split in (f"qa{i}" for i in range(1, 6)):
            try:
                rows = len(datasets.load_dataset(path, name=length, split=split))
                report(rows > 0, f"BABILong {path} {length} {split}: {rows} rows")
            except Exception as error:  # noqa: BLE001
                report(False, f"BABILong {path} {length} {split}: {type(error).__name__}: {str(error)[:200]}")


def check_longbench2(report) -> None:
    import datasets
    import lm_eval

    from litgpt.longbench2_patch import REPO, apply_patch, snapshot

    try:
        root, files = snapshot()
    except Exception as error:  # noqa: BLE001
        report(False, f"LongBench v2: no snapshot in the hub cache ({type(error).__name__}: {str(error)[:200]})")
        return
    bad = [f for f in files if f.endswith(".parquet") and not whole_parquet(Path(root) / f)]
    report(bool(files) and not bad, f"LongBench v2 snapshot {root}: {len(files)} files"
           + (f", not whole parquet: {bad}" if bad else ", all whole parquet"))
    apply_patch()  # as eval.py
    folder = Path(lm_eval.__file__).parent / "tasks" / "longbench2"
    subsets = sorted({line.split(":", 1)[1].strip() for p in folder.glob("*.yaml")
                      for line in p.read_text().splitlines() if line.startswith("dataset_name:")})
    total = 0
    for name in subsets:
        try:
            rows = len(datasets.load_dataset(REPO, name, split="train"))
            total += rows
            report(rows > 0, f"LongBench v2 {name}: {rows} questions")
        except Exception as error:  # noqa: BLE001
            report(False, f"LongBench v2 {name}: {type(error).__name__}: {str(error)[:200]}")
    report(total == 503, f"LongBench v2: {total} questions in {len(subsets)} subsets (want 503)")


def check_ruler_qa(report) -> None:
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    folder = Path(os.environ.get("RULER_CACHE_DIR", os.path.join(home, "ruler_cache")))
    for name in RULER_QA:
        path = folder / name
        report(path.is_file() and path.stat().st_size > 1e6,
               f"RULER QA {path}: " + (f"{path.stat().st_size / 1e6:.1f} MB" if path.is_file() else "missing"))


def check_infinitebench(report) -> None:
    import json

    spec = importlib.util.spec_from_file_location(
        "infinitebench_utils", ROOT / "eval_tasks" / "infinitebench" / "infinitebench_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    for task in utils.TASKS:
        try:
            path = utils.data_file(task)
            rows = 0
            with open(path, encoding="utf8") as f:
                for line in f:
                    if line.strip():
                        eg = json.loads(line)
                        utils.prompt_of(task, eg), utils.answer_of(task, eg)
                        rows += 1
            report(rows > 0, f"InfiniteBench {task}: {rows} samples in {path}")
        except Exception as error:  # noqa: BLE001
            report(False, f"InfiniteBench {task}: {type(error).__name__}: {str(error)[:800]}")


def check_ppl(report, tokenizer: str | None) -> None:
    spec = importlib.util.spec_from_file_location("long_ppl_utils", ROOT / "eval_tasks" / "long_ppl" / "long_ppl_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    want = int(os.environ.get("LONG_PPL_DOCS", "100"))
    for env, longest, label, need in (("LONG_PPL_DATA", utils.LONGEST, "long_ppl 32k-128k", 1),
                                      ("LONG_PPL_DATA", utils.LONGEST_256K, "long_ppl_256k", 1),
                                      ("LONG_PPL_BOOKS", utils.LONGEST_256K, "long_ppl_books", want)):
        if not os.environ.get(env):
            report(False, f"{label}: {env} is not set")
            continue
        if not tokenizer:
            report(False, f"{label}: give --tokenizer to count the documents in {os.environ[env]}")
            continue
        try:
            docs = utils._documents({"pretrained": tokenizer}, longest, env=env)
            report(len(docs) >= need, f"{label}: {len(docs)} documents of >= {longest} tokens in {os.environ[env]}"
                   + ("" if len(docs) >= want else f" (want {want})"))
        except Exception as error:  # noqa: BLE001
            report(False, f"{label}: {type(error).__name__}: {str(error)[:200]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default=None, help="tokenizer dir, for the long_ppl / books document counts")
    args = p.parse_args()
    if os.environ.get("HF_DATASETS_OFFLINE") != "1":
        print("note: HF_DATASETS_OFFLINE is not 1; run through diag.sh to check what eval.sh will see")
    failures = []

    def report(ok, text):
        print(("OK       " if ok else "MISSING  ") + text, flush=True)
        if not ok:
            failures.append(text)
    for check in (check_babilong, check_longbench2, check_ruler_qa, check_infinitebench):
        check(report)
    check_ppl(report, args.tokenizer)
    print(f"\n{len(failures)} problem(s)" + ("" if failures else ": the long-context data is complete"))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
