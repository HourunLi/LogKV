"""unused/repair_dataset_info.py on a cache entry laid out like cais___mmlu/<subject>/0.0.0/<hash>/."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

datasets = pytest.importorskip("datasets")
pa = pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "unused" / "repair_dataset_info.py"
spec = importlib.util.spec_from_file_location("repair_dataset_info", TOOL)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def make_cache(root, subjects=("virology", "world_religions")):
    """Arrow stream files and an indent-2 dataset_info.json per subject, as in the shared cache."""
    features = datasets.Features(question=datasets.Value("string"), subject=datasets.Value("string"),
                                 choices=datasets.Sequence(datasets.Value("string")),
                                 answer=datasets.ClassLabel(names=["A", "B", "C", "D"]))
    files = {}
    for s, subject in enumerate(subjects):
        directory = Path(root, "cais___mmlu", subject, "0.0.0", f"{s:040x}")
        directory.mkdir(parents=True)
        splits = []
        for split, n in (("test", 171 + s), ("validation", 19 + s), ("dev", 5)):
            table = datasets.Dataset.from_dict(
                dict(question=[f"q{i}" for i in range(n)], subject=[subject] * n,
                     choices=[["a", "b", "c", "d"]] * n, answer=[i % 4 for i in range(n)]),
                features=features).data.table
            with pa.OSFile(str(directory / f"mmlu-{split}.arrow"), "wb") as sink, \
                    pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            splits.append(datasets.SplitInfo(name=split, num_bytes=table.nbytes, num_examples=n,
                                             dataset_name="mmlu"))
        size = sum(split.num_bytes for split in splits)
        info = datasets.DatasetInfo(
            features=features, builder_name="parquet", dataset_name="mmlu", config_name=subject, version="0.0.0",
            splits=datasets.SplitDict.from_split_dict([dataclasses.asdict(split) for split in splits]), download_size=123, dataset_size=size,
            size_in_bytes=123 + size,
            download_checksums={f"hf://datasets/cais/mmlu@x/{subject}/test.parquet": dict(num_bytes=1, checksum=None)})
        info = dataclasses.asdict(info)
        files[subject] = directory / "dataset_info.json"
        files[subject].write_text(json.dumps({key: info[key] for key in sorted(info)}, indent=2), encoding="utf-8")
    return files


def load_offline(root):
    code = ("import datasets, json, sys; d = datasets.load_dataset('cais/mmlu', 'world_religions', cache_dir=sys.argv[1]);"
            "print(json.dumps({k: len(v) for k, v in d.items()}))")
    env = dict(os.environ, HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", HF_HOME=str(Path(root, "hf_home")))
    result = subprocess.run([sys.executable, "-c", code, str(root)], env=env, capture_output=True, text=True)
    return result


@pytest.mark.parametrize("cut", ["inside the last split", "inside features"])
def test_truncated_dataset_info_is_rebuilt_exactly(tmp_path, cut):
    path = make_cache(tmp_path)["world_religions"]
    original = path.read_text(encoding="utf-8")
    # The shared cache's file stopped right after `"num_bytes": ...,` of its dev split.
    end = (original.index('"num_bytes"', original.index('"dev": {')) if cut == "inside the last split"
           else original.index('"choices"'))
    end = original.index("\n", end) + 5
    path.write_text(original[:end], encoding="utf-8")
    broken = load_offline(tmp_path)
    assert broken.returncode and "Expecting property name enclosed in double quotes" in broken.stderr

    assert [p for p, _ in repair.scan(tmp_path)] == [path]
    # By path, or as REPAIR_HF_CACHE=1 runs it: scan the cache, write, then load offline.
    command = ([str(path), "--write"] if cut == "inside the last split"
               else ["--scan", str(tmp_path), "--write", "--check"])
    result = subprocess.run([sys.executable, str(TOOL), *command], capture_output=True, text=True,
                            env=dict(os.environ, HF_HOME=str(tmp_path / "hf_home")))
    assert result.returncode == 0, result.stdout + result.stderr
    if "--check" in command:
        assert "offline load_dataset('cais/mmlu', 'world_religions'): {'test': 172, 'validation': 20, 'dev': 5}" \
            in result.stdout
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(original)
    assert list(json.loads(path.read_text(encoding="utf-8"))) == list(json.loads(original))
    assert path.with_name("dataset_info.json.truncated").read_text(encoding="utf-8") == original[:end]
    loaded = load_offline(tmp_path)
    assert loaded.returncode == 0, loaded.stderr
    assert json.loads(loaded.stdout.splitlines()[-1]) == dict(test=172, validation=20, dev=5)


def test_repair_refuses_counts_that_disagree_with_the_arrow_files(tmp_path):
    path = make_cache(tmp_path)["world_religions"]
    text = path.read_text(encoding="utf-8").replace('"num_examples": 172', '"num_examples": 999')
    path.write_text(text[:text.index('"dev": {') + 40], encoding="utf-8")
    with pytest.raises(ValueError, match="999 != 172 rows on disk"):
        repair.rebuild(path)
    info, report = repair.rebuild(make_cache(tmp_path / "valid")["virology"])
    assert info is None and "nothing to repair" in report[0]
