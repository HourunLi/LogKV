"""Ordinary-file preparation for the 512K/1M evaluations; no symlinks or rename-based publishing."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@cache
def hub_info(repo):
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo, files_metadata=True)


def download_plain(repo, name, folder, lines):
    """Fetch a pinned resolve URL directly; verify Hub size and SHA256 before using the file."""
    import requests
    from huggingface_hub import hf_hub_url

    info = hub_info(repo)
    item = next((entry for entry in info.siblings if entry.rfilename == name), None)
    if item is None:
        raise FileNotFoundError(f"{repo}@{info.sha} has no {name}")
    lfs = getattr(item, "lfs", None)
    expected_hash = (lfs.get("sha256") or lfs.get("oid")) if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    size = getattr(item, "size", None)
    target = Path(folder) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()

    def complete():
        return (target.is_file() and (size is None or target.stat().st_size == size)
                and (not expected_hash or sha256(target) == expected_hash))

    if not complete():
        url = hf_hub_url(repo, name, repo_type="dataset", revision=info.sha)
        with requests.get(url, stream=True, timeout=(60, 300)) as response:
            response.raise_for_status()
            with target.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        if not complete():
            raise ValueError(f"Incomplete download or SHA256 mismatch: {target}")
    lines.append(f"ordinary file {repo}@{info.sha}/{name}: {target.stat().st_size} bytes, SHA256={sha256(target)}")
    return target


def babi_utils():
    spec = importlib.util.spec_from_file_location("babilong_plain_utils", ROOT / "eval_tasks/babilong_long/babilong_long_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_babilong(length, lines):
    """Keep official JSON plus memory-mappable Arrow streams so eval never builds a shared cache."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    utils = babi_utils()
    for split in (f"qa{i}" for i in range(1, 6)):
        arrow, target = utils.plain_paths(length, split)
        name = str(target.relative_to(utils.plain_folder()))
        download_plain("RMT-team/babilong", name, utils.plain_folder(), lines)
        source_hash = sha256(target)
        manifest = arrow.with_suffix(".manifest.json")
        valid = False
        try:
            record = json.loads(manifest.read_text())
            valid = (not arrow.is_symlink() and record["source_sha256"] == source_hash
                     and record["arrow_sha256"] == sha256(arrow) and len(utils.load_plain(length, split)) == 100)
        except (OSError, ValueError, KeyError):
            pass
        if not valid:
            with target.open(encoding="utf-8") as stream:
                rows = json.load(stream)
            if (not isinstance(rows, list) or len(rows) != 100
                    or any(not isinstance(row, dict) or not {"input", "question", "target"} <= row.keys() for row in rows)):
                raise ValueError(f"BABILong {target} must contain 100 input/question/target records")
            table = pa.Table.from_pylist(rows)
            if arrow.is_symlink():
                arrow.unlink()
            with pa.OSFile(str(arrow), "wb") as sink:
                with ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)
            manifest.write_text(json.dumps(dict(source_sha256=source_hash, arrow_sha256=sha256(arrow), rows=100)))
        data = utils.load_plain(length, split)
        if len(data) != 100:
            raise ValueError(f"BABILong {arrow}: expected 100 rows, found {len(data)}")
        lines.append(f"BABILong {length} {split}: {len(data)} rows, ordinary Arrow, offline reload: ok ({arrow})")


def prepare_ruler_essays(lines):
    import pyarrow.parquet as pq
    from litgpt.ruler_patch import _essay_file

    target = _essay_file()
    source = download_plain("baber/paul_graham_essays", "essays.parquet", target.parent / "source", lines)
    table = pq.read_table(source, columns=["text"], partitioning=None)
    text = " ".join(str(value) for value in table["text"].to_pylist() if value)
    if not text.strip():
        raise ValueError(f"Empty RULER essay source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    target.write_text(text, encoding="utf-8")
    lines.append(f"RULER ordinary essays: {target}, {len(text)} characters, SHA256={sha256(target)}; "
                 "512K/1M generation cycles the background and inserts each needle once (extended-v1)")
    prepare_nltk(lines)


def prepare_nltk(lines):
    """Use the same requests SSL policy as the downloader; no urllib certificate dependency."""
    import io
    import zipfile
    import requests

    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    root = Path(os.environ.get("NLTK_DATA", str(home / "nltk_data")).split(os.pathsep)[0]).expanduser()
    if not (root / "tokenizers/punkt_tab/english/collocations.tab").is_file():
        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                path = root / "tokenizers" / item.filename
                if Path(item.filename).is_absolute() or ".." in Path(item.filename).parts:
                    raise ValueError(f"Invalid NLTK archive entry: {item.filename}")
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink():
                    path.unlink()
                with archive.open(item) as source, path.open("wb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
    lines.append(f"RULER NLTK punkt_tab ordinary files: {root / 'tokenizers/punkt_tab'}")
