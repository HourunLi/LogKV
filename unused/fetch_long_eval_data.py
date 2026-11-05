"""Fetch the data of the long-context evaluations into the caches eval.sh reads (run once, online).

eval.sh runs offline (HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1) with HF_HOME, HF_DATASETS_CACHE and
RULER_CACHE_DIR from its environment header; diag.sh sets the same paths. This script turns the offline
switches off for itself only, then:

    BABILong      RMT-team/babilong-1k-samples, configs 32k / 64k / 128k, splits qa1 ... qa5
    RULER QA      SQuAD v2 dev and HotpotQA dev (distractor) JSON, into RULER_CACHE_DIR, where
                  litgpt/ruler_patch.py looks first
    LongBench v2  recursal/longbench-v2, every subset lm-eval's longbench2 tasks load (--longbench2)
    long_ppl      --ppl-data: finds the documents of at least 131072 tokens (with --tokenizer) and caches
                  the selection the long_ppl tasks use, so eval ranks do not tokenize it again

and finally loads every dataset again in an offline subprocess, as eval.py will. A summary goes to
<--out>.summary.txt beside --out (default fetch_long_eval_data.json in the current directory).

    bash diag.sh unused/fetch_long_eval_data.py --longbench2 \\
        --ppl-data <held-out ProLong parquet file or directory> --tokenizer <Qwen3-1.7B-Base dir>
"""
from __future__ import annotations

import os

# Before any Hugging Face import: these are read when the libraries load.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BABILONG = "RMT-team/babilong-1k-samples"
BABILONG_LENGTHS = ("32k", "64k", "128k")
QA_SPLITS = tuple(f"qa{i}" for i in range(1, 6))
LONGBENCH2 = "recursal/longbench-v2"
RULER_QA = {
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json": "dev-v2.0.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json": "hotpot_dev_distractor_v1.json",
}


def longbench2_subsets() -> list[str]:
    import lm_eval

    folder = Path(lm_eval.__file__).parent / "tasks" / "longbench2"
    names = set()
    for path in folder.glob("*.yaml"):
        for line in path.read_text().splitlines():
            if line.startswith("dataset_name:"):
                names.add(line.split(":", 1)[1].strip())
    return sorted(names)


def ruler_cache_dir() -> Path:
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(os.environ.get("RULER_CACHE_DIR", os.path.join(home, "ruler_cache")))


def fetch_ruler_qa(lines: list[str]) -> None:
    import requests

    folder = ruler_cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for url, name in RULER_QA.items():
        path = folder / name
        if not path.is_file():
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            path.write_text(json.dumps(response.json(), ensure_ascii=False), encoding="utf-8")
        lines.append(f"RULER QA {name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def fetch_datasets(babilong: bool, longbench2: bool, lines: list[str]) -> list[tuple[str, str, str]]:
    import datasets

    loads = [(BABILONG, n, s) for n in BABILONG_LENGTHS for s in QA_SPLITS] if babilong else []
    if longbench2:
        loads += [(LONGBENCH2, name, "train") for name in longbench2_subsets()]
    for path, name, split in loads:
        rows = len(datasets.load_dataset(path, name=name, split=split))
        lines.append(f"{path} {name} {split}: {rows} rows")
    return loads


def check_offline(loads, lines: list[str]) -> bool:
    """Load every dataset again with the offline switches eval.sh sets, in a fresh process."""
    script = ("import datasets, json, sys\n"
              "for path, name, split in json.loads(sys.argv[1]):\n"
              "    datasets.load_dataset(path, name=name, split=split)\n"
              "print('ok')\n")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    done = subprocess.run([sys.executable, "-c", script, json.dumps(loads)], env=env, capture_output=True, text=True)
    ok = done.returncode == 0 and done.stdout.strip().endswith("ok")
    lines.append("offline reload: " + ("ok" if ok else "FAILED\n" + done.stderr[-2000:]))
    return ok


def cache_ppl(data: str, tokenizer: str, lines: list[str]) -> None:
    """Select the long_ppl documents now (the tasks read the same cache)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("long_ppl_utils", ROOT / "eval_tasks" / "long_ppl" / "long_ppl_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    os.environ["LONG_PPL_DATA"] = data
    docs = utils._documents({"pretrained": tokenizer})
    want = int(os.environ.get("LONG_PPL_DOCS", "100"))
    lines.append(f"long_ppl: {len(docs)} documents of >= {utils.LONGEST} tokens from {data} (want {want}); "
                 f"cache {os.environ.get('LONG_PPL_CACHE', '~/.cache/logkv_long_ppl')}")
    if len(docs) < want:
        lines.append("  fewer than wanted: point --ppl-data at more shards, or if the rows are pieces of longer "
                     "texts, rerun with LONG_PPL_CONCAT=1 (and run eval.sh with it too)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--longbench2", action="store_true", help="also fetch LongBench v2 (every subset)")
    p.add_argument("--no-babilong", action="store_true")
    p.add_argument("--no-ruler-qa", action="store_true")
    p.add_argument("--ppl-data", default=None, help="held-out ProLong parquet / jsonl file or directory")
    p.add_argument("--tokenizer", default=None, help="tokenizer dir for --ppl-data (e.g. the Qwen3-1.7B-Base dir)")
    p.add_argument("--out", default="fetch_long_eval_data.json")
    args = p.parse_args()
    lines = [f"HF_HOME={os.environ.get('HF_HOME')}", f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}",
             f"RULER_CACHE_DIR={ruler_cache_dir()}"]
    ok = True
    if not args.no_ruler_qa:
        fetch_ruler_qa(lines)
    loads = fetch_datasets(not args.no_babilong, args.longbench2, lines)
    if loads:
        ok = check_offline(loads, lines)
    if args.ppl_data:
        if not args.tokenizer:
            raise SystemExit("--ppl-data needs --tokenizer (the dir with tokenizer.json)")
        cache_ppl(args.ppl_data, args.tokenizer, lines)
    text = "\n".join(lines)
    out = Path(args.out).expanduser()
    out.with_name(out.stem + ".summary.txt").write_text(text + "\n")
    out.write_text(json.dumps(dict(lines=lines, ok=ok), indent=1))
    print(text)
    print(f"summary written to {out.with_name(out.stem + '.summary.txt')}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
