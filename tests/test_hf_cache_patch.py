"""Offline loads skip the <hash>.incomplete folder an interrupted download leaves (litgpt/hf_cache_patch.py).

Needs datasets (written against 4.8.5); no model or GPU. Each load runs in a fresh offline process, so the
unpatched behaviour is seen as it is."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("datasets")
PATCH = Path(__file__).resolve().parents[1] / "litgpt" / "hf_cache_patch.py"
DATASET, CONFIG, HASH = "RMT-team/babilong-1k-samples", "128k", "fc4d1a584dfc498c37578753bee4cdd91b987ae2"
LOAD = """
import importlib.util, json, sys
if sys.argv[2] == "patched":  # by file: the package itself needs torch
    spec = importlib.util.spec_from_file_location("hf_cache_patch", sys.argv[1])
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    patch.apply_patch()
import datasets
try:
    rows = datasets.load_dataset("{dataset}", name="{config}", split="qa1")
    print(json.dumps(dict(rows=list(rows["target"]))))
except Exception as error:
    print(json.dumps(dict(error=type(error).__name__, message=str(error))))
""".format(dataset=DATASET, config=CONFIG)


def _write_config(folder):
    """A finished builder cache of the config: one arrow file per split and its dataset_info.json."""
    from datasets import DatasetInfo, Features, SplitDict, SplitInfo, Value
    from datasets.arrow_writer import ArrowWriter

    folder.mkdir(parents=True)
    features, splits = Features(target=Value("string")), SplitDict()
    for split in ("qa1", "qa2"):
        with ArrowWriter(path=str(folder / f"babilong-1k-samples-{split}.arrow"), features=features) as writer:
            writer.write_batch(dict(target=[f"{split}-a", f"{split}-b"]))
            writer.finalize()
        splits.add(SplitInfo(name=split, num_examples=2, num_bytes=0))
    DatasetInfo(dataset_name="babilong-1k-samples", config_name=CONFIG, version="0.0.0",
                features=features, splits=splits).write_to_directory(str(folder))


def _load(cache, mode):
    env = {**os.environ, "HF_DATASETS_CACHE": str(cache), "HF_HUB_CACHE": str(cache / "hub"),
           "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    done = subprocess.run([sys.executable, "-c", LOAD, str(PATCH), mode], env=env, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _leftover(cache, with_arrow=True):
    """What an interrupted download leaves: the builder's folder without its dataset_info.json."""
    folder = cache / "RMT-team___babilong-1k-samples" / CONFIG / "0.0.0" / f"{HASH}.incomplete"
    folder.mkdir(parents=True)
    if with_arrow:
        (folder / "babilong-1k-samples-qa1.arrow").write_bytes(b"cut short")
    return folder


def test_a_leftover_beside_a_finished_copy_no_longer_breaks_the_offline_load(tmp_path):
    _write_config(tmp_path / "RMT-team___babilong-1k-samples" / CONFIG / "0.0.0" / HASH)
    _leftover(tmp_path)
    plain = _load(tmp_path, "plain")
    assert plain["error"] == "FileNotFoundError" and ".incomplete/dataset_info.json" in plain["message"]
    assert _load(tmp_path, "patched") == dict(rows=["qa1-a", "qa1-b"])


def test_a_config_with_only_a_leftover_says_it_was_never_downloaded(tmp_path):
    _write_config(tmp_path / "RMT-team___babilong-1k-samples" / "64k" / "0.0.0" / HASH)
    folder = _leftover(tmp_path, with_arrow=False)
    result = _load(tmp_path, "patched")
    assert result["error"] == "ValueError"
    assert "never completely downloaded" in result["message"] and str(folder) in result["message"]
    assert "for config '128k'" in result["message"]  # datasets' own message is kept


def test_leftovers_lists_only_unfinished_folders(tmp_path, monkeypatch):
    import datasets.config

    spec = importlib.util.spec_from_file_location("hf_cache_patch", PATCH)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    finished, leftovers = patch.finished, patch.leftovers
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path))
    _write_config(tmp_path / "RMT-team___babilong-1k-samples" / CONFIG / "0.0.0" / HASH)
    folder = _leftover(tmp_path)
    assert leftovers(DATASET) == leftovers(DATASET, CONFIG) == [str(folder)]
    assert leftovers(DATASET, "64k") == []
    assert finished([str(folder), str(folder.with_suffix("")), folder]) == [str(folder.with_suffix(""))]
