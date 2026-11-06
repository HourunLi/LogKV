"""Fetch the data of the long-context evaluations into the caches eval.sh reads (run once, online).

eval.sh runs offline (HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1) with HF_HOME, HF_DATASETS_CACHE and
RULER_CACHE_DIR from its environment header; diag.sh sets the same paths. This script turns the offline
switches off for itself only, then:

    BABILong      RMT-team/babilong-1k-samples, configs 32k / 64k / 128k, splits qa1 ... qa5
    RULER QA      SQuAD v2 dev and HotpotQA dev (distractor) JSON, into RULER_CACHE_DIR, where
                  litgpt/ruler_patch.py looks first
    LongBench v2  recursal/longbench-v2, every subset lm-eval's longbench2 tasks load (--longbench2)
    long_ppl      --ppl-data: finds the documents of at least 131072 tokens (with --tokenizer) and caches
                  the selection the long_ppl tasks use, so eval ranks do not tokenize it again;
                  --ppl-hub <dataset repo> (e.g. emozilla/pg19-test, PG19's 100 test books) downloads
                  the repository first and uses its folder as --ppl-data (export the printed path as
                  LONG_PPL_DATA for eval.sh)

and finally loads every dataset again in an offline subprocess, as eval.py will. A summary goes to
<--out>.summary.txt beside --out (default fetch_long_eval_data.json in the current directory).

The platform's proxy re-signs HTTPS with a self-signed CA, so certificate checks are off unless --verify-ssl
is given (as litgpt/ruler_patch.py does for RULER); downloads then stay on requests, without hf_transfer or
hf_xet, which check certificates themselves. hf-mirror answers 429 when the Hub quota it shares across its users
runs out; every request waits and retries then.

    bash diag.sh unused/fetch_long_eval_data.py --longbench2 \\
        --ppl-data <held-out ProLong parquet file or directory> --tokenizer <Qwen3-1.7B-Base dir>
"""
from __future__ import annotations

import os

# Before any Hugging Face import: these are read when the libraries load.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
# The mirror can pause mid-file for longer than the Hub's 10 s default read timeout.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BABILONG = "RMT-team/babilong-1k-samples"
BABILONG_LENGTHS = ("32k", "64k", "128k")
QA_SPLITS = tuple(f"qa{i}" for i in range(1, 6))
LONGBENCH2 = "recursal/longbench-v2"
RULER_QA = {
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json": "dev-v2.0.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json": "hotpot_dev_distractor_v1.json",
}


def patch_requests(verify_ssl: bool, retries: int = 10) -> None:
    """Every requests call (RULER QA, and the Hub API calls and files of datasets) retries a 429 after its
    Retry-After (60 s without one, at most 300 s), and skips certificate checks unless verify_ssl. Call before
    any Hugging Face import: without the checks, hf_transfer and hf_xet are switched off when the Hub loads."""
    import time

    import requests
    import urllib3

    if not verify_ssl:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    request = requests.Session.request

    def patched_request(self, *args, **kwargs):
        if not verify_ssl:
            kwargs["verify"] = False
        for attempt in range(retries + 1):
            response = request(self, *args, **kwargs)
            if response.status_code != 429 or attempt == retries:
                return response
            wait = response.headers.get("Retry-After", "")
            wait = min(int(wait), 300) if wait.isdigit() else 60
            print(f"429 rate limit on {response.url}; retry {attempt + 1}/{retries} in {wait} s", file=sys.stderr)
            response.close()
            time.sleep(wait)

    requests.Session.request = patched_request


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


def fetch_datasets(babilong: bool, longbench2: bool, lines: list[str]) -> tuple[list[tuple[str, str, str]], bool]:
    """Each dataset group in turn; a group that fails is reported and the others still run."""
    import datasets

    from litgpt.longbench2_patch import apply_patch, snapshot

    apply_patch()  # as eval.py: LongBench v2 subsets from the snapshot's files when the card has no configs
    groups = []
    if babilong:
        groups.append(("BABILong", [(BABILONG, n, s) for n in BABILONG_LENGTHS for s in QA_SPLITS]))
    if longbench2:
        groups.append(("LongBench v2", [(LONGBENCH2, name, "train") for name in longbench2_subsets()]))
    loads, ok = [], True
    for label, group in groups:
        try:
            if label == "LongBench v2":
                root, files = snapshot()
                lines.append(f"LongBench v2 snapshot {root}: {len(files)} data files, e.g. {files[:4]}")
                if not repair_parquet(root, files, lines):
                    raise ValueError("some LongBench v2 files are still not parquet after a fresh download (above)")
            for path, name, split in group:
                rows = len(datasets.load_dataset(path, name=name, split=split))
                lines.append(f"{path} {name} {split}: {rows} rows")
            loads += group
        except Exception as error:  # noqa: BLE001 - report and go on with the next group
            ok = False
            lines.append(f"{label}: FAILED {type(error).__name__}: {str(error)[:1500]}")
    return loads, ok


def is_parquet(path: Path) -> bool:
    """A parquet file starts and ends with the magic bytes PAR1."""
    if not path.is_file() or path.stat().st_size < 12:
        return False
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, os.SEEK_END)
        return head == b"PAR1" and f.read(4) == b"PAR1"


def describe(path: Path) -> str:
    """Size and first bytes of a file that should be parquet (a Git LFS pointer, an error page, a cut file)."""
    if not path.exists():
        return "missing"
    with open(path, "rb") as f:
        head = f.read(120)
    return f"{path.stat().st_size} bytes, starts {head!r}"


def repair_parquet(root: str, files: list[str], lines: list[str], repo: str | None = None) -> bool:
    """Download again, fresh, every snapshot parquet file that is not a whole parquet file."""
    from huggingface_hub import hf_hub_download

    from litgpt.longbench2_patch import REPO

    repo = repo or REPO
    bad = [f for f in files if f.endswith(".parquet") and not is_parquet(Path(root) / f)]
    ok = True
    for f in bad:
        before = describe(Path(root) / f)
        fresh = Path(hf_hub_download(repo, f, repo_type="dataset", force_download=True))
        good = is_parquet(fresh)
        ok = ok and good
        lines.append(f"{repo} {f}: was {before}; downloaded again: " + ("ok" if good else describe(fresh)))
    if not bad:
        lines.append(f"{repo} files: all whole parquet")
    return ok


def fetch_ppl_hub(repo: str, lines: list[str]) -> str:
    """Download a dataset repository for long_ppl and return its snapshot folder (the --ppl-data)."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo, repo_type="dataset")
    files = sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.suffix in (".parquet", ".jsonl"))
    lines.append(f"long_ppl data {repo}: {root} ({len(files)} files, e.g. {files[:3]})")
    if not repair_parquet(root, files, lines, repo):
        raise ValueError(f"some {repo} files are still not parquet after a fresh download (above)")
    lines.append(f"  for eval.sh: export LONG_PPL_DATA={root}")
    return root


def check_offline(loads, lines: list[str]) -> bool:
    """Load every dataset again with the offline switches eval.sh sets (and its LongBench v2 patch), in a
    fresh process."""
    script = ("import datasets, json, sys\n"
              f"sys.path.insert(0, {str(ROOT)!r})\n"
              "from litgpt.longbench2_patch import apply_patch\n"
              "apply_patch()\n"
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
    want = int(os.environ.get("LONG_PPL_DOCS", "100"))
    try:
        docs = utils._documents({"pretrained": tokenizer})
    except ValueError:
        ppl_data_stats(data, tokenizer, utils, lines)
        raise
    lines.append(f"long_ppl: {len(docs)} documents of >= {utils.LONGEST} tokens from {data} (want {want}); "
                 f"cache {utils.cache_dir()}")
    if len(docs) < want:
        ppl_data_stats(data, tokenizer, utils, lines)
        lines.append("  fewer than wanted: point --ppl-data at more shards, or if the rows are pieces of longer "
                     "texts, rerun with LONG_PPL_CONCAT=1 (and run eval.sh with it too)")


def ppl_data_stats(data: str, tokenizer: str, utils, lines: list[str], files: int = 2, show: int = 5) -> None:
    """How long the rows are: characters over every row of the first files, tokens of the first and the
    longest rows (rows of one length throughout are packed sequences, not documents)."""
    import heapq

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    paths = utils._files(data)[:files]
    lengths, first, longest = [], [], []
    for path in paths:
        for i, text in enumerate(utils._texts(path)):
            lengths.append(len(text))
            if len(first) < show:
                first.append(text)
            heapq.heappush(longest, (len(text), len(lengths), text))
            if len(longest) > show:
                heapq.heappop(longest)
    if not lengths:
        lines.append(f"  long_ppl data: no text rows in {[str(p) for p in paths]}")
        return
    chars = np.asarray(lengths)

    def tokens(texts):
        return [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    lines.append(f"  long_ppl data: {len(chars)} rows in {len(paths)} file(s); characters p50 {int(np.median(chars))}, "
                 f"p90 {int(np.percentile(chars, 90))}, max {int(chars.max())}")
    lines.append(f"  tokens of the first {len(first)} rows: {tokens(first)}")
    lines.append(f"  tokens of the {len(longest)} longest rows: {sorted(tokens([t for _, _, t in longest]), reverse=True)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--longbench2", action="store_true", help="also fetch LongBench v2 (every subset)")
    p.add_argument("--no-babilong", action="store_true")
    p.add_argument("--no-ruler-qa", action="store_true")
    p.add_argument("--ppl-data", default=None, help="held-out ProLong parquet / jsonl file or directory")
    p.add_argument("--ppl-hub", default=None,
                   help="dataset repository to download and use as --ppl-data (e.g. emozilla/pg19-test)")
    p.add_argument("--tokenizer", default=None, help="tokenizer dir for --ppl-data (e.g. the Qwen3-1.7B-Base dir)")
    p.add_argument("--out", default="fetch_long_eval_data.json")
    p.add_argument("--verify-ssl", action="store_true", help="keep HTTPS certificate checks (off by default)")
    args = p.parse_args()
    patch_requests(args.verify_ssl)
    lines = [f"HF_HOME={os.environ.get('HF_HOME')}", f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}",
             f"RULER_CACHE_DIR={ruler_cache_dir()}",
             f"HTTPS certificate checks: {'on' if args.verify_ssl else 'off (--verify-ssl turns them on)'}"]
    ok = True
    if (args.ppl_data or args.ppl_hub) and not args.tokenizer:
        raise SystemExit("--ppl-data / --ppl-hub need --tokenizer (the dir with tokenizer.json)")
    if not args.no_ruler_qa:
        try:
            fetch_ruler_qa(lines)
        except Exception as error:  # noqa: BLE001
            ok = False
            lines.append(f"RULER QA: FAILED {type(error).__name__}: {error}")
    loads, fetched = fetch_datasets(not args.no_babilong, args.longbench2, lines)
    ok = ok and fetched
    if loads:
        ok = check_offline(loads, lines) and ok
    if args.ppl_hub:
        try:
            args.ppl_data = fetch_ppl_hub(args.ppl_hub, lines)
        except Exception as error:  # noqa: BLE001
            ok = False
            lines.append(f"long_ppl data {args.ppl_hub}: FAILED {type(error).__name__}: {str(error)[:1500]}")
    if args.ppl_data:
        try:
            cache_ppl(args.ppl_data, args.tokenizer, lines)
        except Exception as error:  # noqa: BLE001
            ok = False
            lines.append(f"long_ppl: FAILED {type(error).__name__}: {error}")
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
