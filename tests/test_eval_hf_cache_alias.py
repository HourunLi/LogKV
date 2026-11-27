"""Run directly in an environment with datasets 4.8.4; no model/GPU required."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType
from unittest.mock import patch


def _write_cache(directory, dataset_name, config_name, version, features, rows):
    """Write a builder cache snapshot the way an older datasets release left it on disk."""
    from datasets import DatasetInfo, SplitDict, SplitInfo
    from datasets.arrow_writer import ArrowWriter

    directory.mkdir(parents=True)
    splits = SplitDict()
    for split in ("train", "validation"):
        with ArrowWriter(path=str(directory / f"{dataset_name}-{split}.arrow"), features=features) as writer:
            writer.write_batch(rows)
            writer.finalize()
        splits.add(SplitInfo(name=split, num_examples=len(next(iter(rows.values()))), num_bytes=0))
    DatasetInfo(dataset_name=dataset_name, config_name=config_name, version=version,
                features=features, splits=splits).write_to_directory(str(directory))


@contextlib.contextmanager
def _offline_cache(root):
    """Run eval.py's cache tracing against an offline cache at ``root``; yield its log buffer."""
    import datasets.config as config
    import huggingface_hub.constants as hub

    source = Path(__file__).resolve().parents[1] / "eval.py"
    nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)
             and node.name in {"_walk_cache_path", "_trace_hf_cache_resolution"}]
    scope = dict(contextlib=contextlib, os=os, json=json, Path=Path,
                 __file__=str(source), _global_rank=lambda: 0)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    # Runtime reporting also imports lm_eval; its task implementation is not needed to test the loader.
    lm_eval = ModuleType("lm_eval")
    lm_eval.__file__, lm_eval.__version__ = "/test/lm_eval.py", "test"
    logs = io.StringIO()
    with (patch.dict("sys.modules", {"lm_eval": lm_eval}),
          patch.object(config, "HF_DATASETS_CACHE", root),
          patch.object(config, "HF_DATASETS_OFFLINE", True),
          patch.object(config, "HF_HUB_OFFLINE", True),
          patch.object(hub, "HF_HUB_OFFLINE", True),
          contextlib.redirect_stderr(logs),
          scope["_trace_hf_cache_resolution"]()):
        yield logs


def _events(logs, kind):
    return [event for event in (json.loads(line.removeprefix("[hf-cache] "))
                                for line in logs.getvalue().splitlines() if line.startswith("[hf-cache] "))
            if event["event"] == kind]


# (path, name) exactly as lm-eval 0.4.11 calls datasets.load_dataset for
# boolq, piqa, social_iqa, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa.
LM_EVAL_CALLS = [
    ("aps/super_glue", "boolq"),
    ("baber/piqa", None),
    ("allenai/social_i_qa", None),
    ("Rowan/hellaswag", None),
    ("allenai/winogrande", "winogrande_xl"),
    ("allenai/ai2_arc", "ARC-Easy"),
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("allenai/openbookqa", "main"),
]


def test_every_benchmark_loads_when_its_alias_is_a_plain_file():
    import pytest
    datasets = pytest.importorskip("datasets", minversion="4.0")
    from datasets import Features, Value

    features = Features({"text": Value("string")})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "hf_cache"
        for path, name in LM_EVAL_CALLS:
            legacy = path.split("/")[1]
            _write_cache(root / legacy / (name or "default") / "1.0.0" / "0a1b2c", legacy,
                         name or "default", "1.0.0", features, {"text": [f"{legacy}/{name}"]})
        aliases = {root / path.replace("/", "___"): path.split("/")[1] for path, _ in LM_EVAL_CALLS}
        # What the training job lists: namespace___name is -rwxrwxrwx, the legacy directory is drwx.
        for alias, legacy in aliases.items():
            alias.write_text(legacy)
            alias.chmod(0o777)
        with _offline_cache(root) as logs:
            for path, name in LM_EVAL_CALLS:
                legacy = path.split("/")[1]
                loaded = datasets.load_dataset(path=path, name=name)
                assert loaded["validation"]["text"] == [f"{legacy}/{name}"], (path, name)
        followed = _events(logs, "cache_alias_followed")
        assert [(e["dataset"], e["loaded_as"]) for e in followed] == [
            (path, path.split("/")[1]) for path, _ in LM_EVAL_CALLS]
        assert len(_events(logs, "cache_alias_unusable")) == len(LM_EVAL_CALLS)


def test_namespaced_cache_alias_is_followed_when_the_mount_cannot():
    import pytest
    datasets = pytest.importorskip("datasets", minversion="4.0")
    import datasets.load as load
    from datasets import ClassLabel, Features, Value

    siqa_features = Features({key: Value("string") for key in
                              ("context", "question", "answerA", "answerB", "answerC")})
    siqa_features["label"] = ClassLabel(names=["1", "2", "3"])
    siqa_rows = {key: [key + "0", key + "1"] for key in siqa_features if key != "label"}
    siqa_rows["label"] = [0, 2]
    wino_features = Features({key: Value("string") for key in ("sentence", "option1", "option2", "answer")})
    wino_rows = {"sentence": ["a _ b"], "option1": ["x"], "option2": ["y"], "answer": ["1"]}
    # lm-eval 0.4.11 path -> (config, legacy snapshot that namespace___name links to, rows)
    cases = {
        "allenai/social_i_qa": ("default", "social_i_qa/default/0.1.0/674d85e4", siqa_features, siqa_rows),
        "allenai/winogrande": ("winogrande_xl", "winogrande/winogrande_xl/1.1.0/9a1c3e", wino_features, wino_rows),
    }
    original_load, original_factory = datasets.load_dataset, load.CachedDatasetModuleFactory.get_module

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "hf_cache"
        for path, (name, snapshot, features, rows) in cases.items():
            legacy = snapshot.split("/")[0]
            _write_cache(root / snapshot, legacy, name, snapshot.split("/")[2], features, rows)
            (root / path.replace("/", "___")).symlink_to(legacy)

        def events(kind):
            return _events(logs, kind)

        def refused(path, name):
            try:
                datasets.load_dataset(path=path, name=name)
            except ConnectionError:
                return True
            return False

        with _offline_cache(root) as logs:
            # Links resolve: the normal lookup succeeds and the fallback stays out of the way.
            for path, (name, _, _, rows) in cases.items():
                assert datasets.load_dataset(path=path, name=name)["train"].to_dict() == rows
            assert not events("cache_alias_unusable")

            # The training job's mount: each link surfaces as a small file holding its target.
            for path, (_, snapshot, _, _) in cases.items():
                alias = root / path.replace("/", "___")
                alias.unlink()
                alias.write_text(snapshot.split("/")[0])
            for path, (name, snapshot, features, rows) in cases.items():
                loaded = datasets.load_dataset(path=path, name=name)  # keyword form, as lm-eval calls it
                assert loaded["validation"].to_dict() == rows and loaded["train"].features == features
            positional = datasets.load_dataset("allenai/winogrande", "winogrande_xl", split="train")
            assert positional.to_dict() == wino_rows
            for path, (_, snapshot, _, rows) in cases.items():
                legacy = snapshot.split("/")[0]
                unusable = next(e for e in events("cache_alias_unusable") if e["dataset"] == path)
                blocking = [step for step in unusable["walk"] if step.get("blocks_traversal")]
                assert [step["path"] for step in blocking] == [str(root / path.replace("/", "___"))]
                assert blocking[0]["type"] == "reg" and blocking[0]["head"] == legacy
                followed = next(e for e in events("cache_alias_followed") if e["dataset"] == path)
                assert followed["loaded_as"] == legacy and followed["directory"] == str(root / legacy)
                n = len(next(iter(rows.values())))
                assert followed["num_rows"] == {"train": n, "validation": n}

            # A link into the old bucket, which this job does not mount, still has the migrated copy.
            alias = root / "allenai___winogrande"
            alias.unlink()
            alias.symlink_to("/nonexistent/bucket-wulan-green/wubohan/data/hf_cache/hf_cache/winogrande")
            assert datasets.load_dataset(path="allenai/winogrande", name="winogrande_xl")["train"].num_rows == 1

            # Never substitute the legacy directory for anything that is not an alias of it.
            alias.unlink()
            (root / "elsewhere").mkdir()
            alias.symlink_to("elsewhere")                 # resolves, but to a different directory
            assert refused("allenai/winogrande", "winogrande_xl")
            alias.unlink()                                # namespaced entry missing altogether
            assert refused("allenai/winogrande", "winogrande_xl")
            alias.mkdir()                                 # a real (empty) directory is not an alias
            assert refused("allenai/winogrande", "winogrande_xl")
            alias.rmdir()
            alias.write_text("winogrande")
            assert refused("another/dataset", None)
            cwd = os.getcwd()
            try:                                          # the legacy name would load a local folder
                os.chdir(tmp)
                Path(tmp, "winogrande").mkdir()
                assert refused("allenai/winogrande", "winogrande_xl")
            finally:
                os.chdir(cwd)

            # Missing data must fail rather than silently evaluate an incomplete dataset.
            (root / cases["allenai/social_i_qa"][1] / "social_i_qa-train.arrow").unlink()
            try:
                datasets.load_dataset(path="allenai/social_i_qa", name="default")
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("missing Arrow file was ignored")
        assert datasets.load_dataset is original_load
        assert load.CachedDatasetModuleFactory.get_module is original_factory


if __name__ == "__main__":
    test_every_benchmark_loads_when_its_alias_is_a_plain_file()
    test_namespaced_cache_alias_is_followed_when_the_mount_cannot()
    print("Namespaced cache alias fallback passed")
