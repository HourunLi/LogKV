"""LongBench v2 by subset when recursal/longbench-v2 exposes no per-subset configs.

lm-eval's longbench2 tasks load ``datasets.load_dataset("recursal/longbench-v2", "<subset>")``; when the
dataset card defines no configs, datasets only offers "default" (all 20 files) and every task fails with
"BuilderConfig '<subset>' not found". This patch keeps the normal call and, when it fails, reads the
subset's files from the repository snapshot instead (downloaded online, or from the hub cache when
HF_HUB_OFFLINE=1): a file belongs to the subset when its stem or its folder is the subset's name, or its
stem starts with the name and "-" or ".". Applied by eval.py; unused/fetch_long_eval_data.py uses it to
fetch and check the same files.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO = "recursal/longbench-v2"
BUILDERS = {".parquet": "parquet", ".jsonl": "json", ".json": "json", ".csv": "csv", ".arrow": "arrow"}
TASKS_DIR = Path(__file__).resolve().parents[1] / "eval_tasks" / "longbench2"


def subsets() -> list[str]:
    """The subsets evaluated by our pinned task definitions, including on lm-eval 0.4.9."""
    names = {(yaml.safe_load(path.read_text()) or {}).get("dataset_name") for path in TASKS_DIR.glob("*.yaml")}
    names.discard(None)
    if not names:
        raise FileNotFoundError(f"LongBench v2 task definitions missing from {TASKS_DIR}")
    return sorted(names)


def subset_files(files, name: str) -> list[str]:
    """The data files (paths relative to the snapshot) of subset ``name``."""
    chosen = []
    for f in files:
        path = Path(f)
        if path.suffix not in BUILDERS:
            continue
        stem = path.name[:-len(path.suffix)]
        if stem == name or path.parent.name == name or stem.startswith((name + "-", name + ".")):
            chosen.append(f)
    return sorted(chosen)


def real_file(path) -> str:
    """``path``, or the blob it points to when a copy or sync of the hub cache turned the snapshot's link into a small
    file holding the link's text (``../../blobs/<sha256>``), which is what the machines that evaluate may see."""
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 512:
        return str(p)
    text = p.read_bytes().decode(errors="ignore").strip()
    if "/blobs/" in text and "\n" not in text:
        blob = os.path.normpath(p.parent / text)
        if os.path.isfile(blob):
            return blob
    return str(p)


def _cached_snapshot() -> str | None:
    """A snapshot folder of the hub cache when refs/main is lost (the newest one)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = Path(os.environ.get("HF_HUB_CACHE") or HF_HUB_CACHE) / ("datasets--" + REPO.replace("/", "--")) / "snapshots"
    shots = sorted((p for p in folder.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
    return str(shots[-1]) if shots else None


def snapshot() -> tuple[str, list[str]]:
    """The repository snapshot's folder and data files (offline: the hub cache only, any snapshot there when
    refs/main is lost). A data file may be a flattened link: read it through real_file."""
    from huggingface_hub import snapshot_download

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    try:
        root = snapshot_download(REPO, repo_type="dataset", local_files_only=offline)
    except Exception:
        root = _cached_snapshot() if offline else None
        if root is None:
            raise
    files = sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file() and p.suffix in BUILDERS)
    return root, files


def data_problem() -> str | None:
    """Why the offline data is incomplete; use the same subset list for fetching and evaluation."""
    repair = ("Run bash diag.sh unused/fetch_long_eval_data.py --longbench2 --no-babilong "
              "with the evaluation job's HF_HOME, then retry.")
    try:
        root, files = snapshot()
    except Exception as error:
        return f"no LongBench v2 snapshot in the hub cache ({type(error).__name__}: {str(error)[:200]}). {repair}"
    missing = [name for name in subsets() if not subset_files(files, name)]
    if missing:
        return f"LongBench v2 snapshot {root} is missing subsets: {missing}. {repair}"
    bad = []
    for name in files:
        if not name.endswith(".parquet"):
            continue
        path = Path(real_file(Path(root) / name))
        try:
            with path.open("rb") as stream:
                head = stream.read(4)
                stream.seek(-4, os.SEEK_END)
                whole = path.stat().st_size >= 12 and head == b"PAR1" and stream.read(4) == b"PAR1"
        except OSError:
            whole = False
        if not whole:
            bad.append(name)
    if bad:
        return f"LongBench v2 files in {root} are not whole parquet: {bad}. {repair}"
    return None


def load_subset(load, name: str, split=None):
    root, files = snapshot()
    chosen = subset_files(files, name)
    if not chosen:
        raise ValueError(f"no file of {REPO} matches subset {name!r}; its data files: {files}")
    builder = BUILDERS[Path(chosen[0]).suffix]
    data = load(builder, data_files={"train": [real_file(os.path.join(root, f)) for f in chosen]})
    return data[split] if split else data


def apply_patch() -> None:
    import datasets

    original = datasets.load_dataset
    if getattr(original, "_longbench2_patched", False):
        return

    def load_dataset(path=None, name=None, *args, **kwargs):
        if path != REPO or not name:
            return original(path, name, *args, **kwargs)
        try:
            return original(path, name, *args, **kwargs)
        except Exception as error:  # no such config (online), or no cache for it (offline)
            try:
                return load_subset(original, name, kwargs.get("split"))
            except Exception as fallback:
                raise fallback from error

    load_dataset._longbench2_patched = True
    datasets.load_dataset = load_dataset
