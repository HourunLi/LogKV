"""Run directly in an environment with datasets 4.8.4; no model/GPU required."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch


def test_social_iqa_recovers_when_directory_glob_is_empty():
    import pytest
    datasets = pytest.importorskip("datasets", minversion="4.0")
    import datasets.config as config
    import datasets.load as load
    import huggingface_hub.constants as hub
    from datasets import ClassLabel, DatasetInfo, Features, SplitDict, SplitInfo, Value
    from datasets.arrow_writer import ArrowWriter

    source = Path(__file__).resolve().parents[1] / "eval.py"
    nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)
             and node.name in {"_read_social_iqa_cache", "_walk_cache_path", "_trace_hf_cache_resolution"}]
    scope = dict(contextlib=contextlib, os=os, json=json, Path=Path,
                 __file__=str(source), _global_rank=lambda: 0)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    # Runtime reporting also imports lm_eval; its task implementation is not needed to test the loader.
    from types import ModuleType
    lm_eval = ModuleType("lm_eval")
    lm_eval.__file__, lm_eval.__version__ = "/test/lm_eval.py", "test"
    features = Features({key: Value("string") for key in
                         ("context", "question", "answerA", "answerB", "answerC")})
    features["label"] = ClassLabel(names=["1", "2", "3"])
    rows = {key: [key + "0", key + "1"] for key in features if key != "label"}
    rows["label"] = [0, 2]
    original_load, original_factory = datasets.load_dataset, load.CachedDatasetModuleFactory.get_module

    with tempfile.TemporaryDirectory() as tmp:
        # Production layout: allenai___social_i_qa is a relative symlink to social_i_qa.
        directory = Path(tmp) / "social_i_qa/default/0.1.0" / (
            "674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8")
        directory.mkdir(parents=True)
        link = Path(tmp) / "allenai___social_i_qa"
        link.symlink_to("social_i_qa")
        splits = SplitDict()
        for split in ("train", "validation"):
            lengths = None if split == "train" else [1, 1]
            splits.add(SplitInfo(name=split, num_examples=2, num_bytes=0, shard_lengths=lengths))
            shards = [rows] if lengths is None else [{key: [value[i]] for key, value in rows.items()}
                                                    for i in range(2)]
            for i, shard in enumerate(shards):
                suffix = "" if lengths is None else f"-{i:05d}-of-{len(shards):05d}"
                with ArrowWriter(path=str(directory / f"social_i_qa-{split}{suffix}.arrow"),
                                 features=features) as writer:
                    writer.write_batch(shard)
                    writer.finalize()
        DatasetInfo(dataset_name="social_i_qa", config_name="default", version="0.1.0",
                    features=features, splits=splits).write_to_directory(str(directory))

        logs = io.StringIO()
        with (patch.dict("sys.modules", {"lm_eval": lm_eval}),
              patch.dict(os.environ, {"SOCIAL_IQA_CACHE_DIR": str(directory)}),
              patch.object(config, "HF_DATASETS_CACHE", Path(tmp)),
              patch.object(config, "HF_DATASETS_OFFLINE", True),
              patch.object(config, "HF_HUB_OFFLINE", True),
              patch.object(hub, "HF_HUB_OFFLINE", True),
              contextlib.redirect_stderr(logs)):
            with scope["_trace_hf_cache_resolution"]():
                normal = datasets.load_dataset("allenai/social_i_qa", name="default")
                assert normal["train"].to_dict() == rows
            assert '"event": "explicit_cache_loaded"' not in logs.getvalue()
            # Force exactly the production failure: no glob matches, despite readable files.
            with (patch.object(load.glob, "glob", return_value=[]),
                  scope["_trace_hf_cache_resolution"]()):
                recovered = datasets.load_dataset("allenai/social_i_qa", name="default")
                for split in splits:
                    assert recovered[split].to_dict() == rows
                    assert recovered[split].features == features
                    assert recovered[split].to_dict() == normal[split].to_dict()
                for kwargs in ({"path": "another/dataset"},
                               {"path": "allenai/social_i_qa", "revision": "specific-commit"}):
                    try:
                        datasets.load_dataset(**kwargs)
                    except ConnectionError:
                        pass
                    else:
                        raise AssertionError("fallback swallowed an unrelated or explicit-revision failure")
                # A mount that does not resolve the link surfaces it as a small regular file;
                # the snapshot path then fails with ENOTDIR exactly as in the training job.
                link.unlink()
                link.write_text("social_i_qa")
                del os.environ["SOCIAL_IQA_CACHE_DIR"]
                mark = len(logs.getvalue())
                via_target = datasets.load_dataset("allenai/social_i_qa", name="default")
                assert via_target["train"].to_dict() == rows
                events = [json.loads(line.removeprefix("[hf-cache] "))
                          for line in logs.getvalue()[mark:].splitlines() if line.startswith("[hf-cache] ")]
                failure = next(e for e in events if e["event"] == "explicit_cache_failure")
                assert failure["error"].startswith("NotADirectoryError")
                blocking = [step for step in failure["walk"] if step.get("blocks_traversal")]
                assert blocking == [step for step in failure["walk"] if step.get("path") == str(link)]
                assert blocking[0]["type"] == "reg" and blocking[0]["head"] == "social_i_qa"
                assert "social_i_qa" in failure["walk"][-1]["entries"]
                loaded = next(e for e in events if e["event"] == "explicit_cache_loaded")
                assert loaded["directory"] == str(directory)
                os.environ["SOCIAL_IQA_CACHE_DIR"] = str(directory)
                # Missing data must fail rather than silently evaluate an incomplete dataset.
                (directory / "social_i_qa-train.arrow").unlink()
                try:
                    datasets.load_dataset("allenai/social_i_qa")
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError("missing Arrow file was ignored")
        assert datasets.load_dataset is original_load
        assert load.CachedDatasetModuleFactory.get_module is original_factory
        assert '"event": "explicit_cache_loaded"' in logs.getvalue()


if __name__ == "__main__":
    test_social_iqa_recovers_when_directory_glob_is_empty()
    print("Social IQA empty-directory-listing recovery passed")
