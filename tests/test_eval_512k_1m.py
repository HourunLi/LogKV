"""Million-token data protocols and launch configuration, without loading a language model."""
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_passes_have_the_right_windows_factors_and_tasks():
    plan = load("ultra_plan", "long_eval_plan.py")
    passes = plan.plan("512k,1m")
    assert [p["length"] for p in passes] == [524288, 1048576]
    assert [p["factor"] for p in passes] == [16., 32.]
    assert [p["config_overrides"]["block_size"] for p in passes] == [525312, 1049600]
    for p in passes:
        assert len(p["niah"]) == 12 and "ruler_qa_squad" not in p["niah"]
        assert p["main"] == [f"{stem}_{p['name']}" for stem in
                              ("babilong", "long_ppl", "long_ppl_books", "niah_turns", "longbench2")]
    assert plan.plan("32k,64k,128k,256k")[-1]["main"][-1] == "infinitebench_256k"
    with pytest.raises(ValueError, match="InfiniteBench"):
        plan.plan("1m", suites="infinitebench")


def test_ten_opt_in_yamls_keep_each_models_checkpoint_and_original_defaults():
    def config(path):
        data = yaml.safe_load(path.read_text())
        base = config(path.parent / data.pop("config")) if data.get("config") else {}
        return {**base, **data}

    for model in ("qwen1.7b-32k", "qwen8b-32k"):
        folder = ROOT / "exp" / model
        source = config(folder / "arc_gamma_k12_b128_p1024_2k.yaml")
        for method in ("baseline", "word_first", "query", "word_first_query", "cpt_dense"):
            data = config(folder / f"eval_512k_1m_{method}.yaml")
            assert data["save_path"] == source["save_path"]
            assert data["eval_lengths"] == ["512k", "1m"]
            assert data["eval_suites"] == ["ruler", "babilong", "books", "longbench2"]
            assert data.get("log_kv_dense_mode", False) == (method == "cpt_dense")
            assert data["eval_main"] is False
        assert config(folder / "eval_full_baseline.yaml")["eval_lengths"] == ["32k", "64k", "128k", "256k"]


def test_eval_shell_honors_the_yamls_suites_without_loading_models(tmp_path):
    from test_long_eval_plan import _launcher

    path = tmp_path / "ultra.yaml"
    path.write_text("checkpoint_dir: /same-cpt\noutput_path: /out\neval_lengths: [512k, 1m]\n"
                    "eval_suites: [ruler, babilong, books, longbench2]\nlimit: 100\n")
    done, passes = _launcher(tmp_path, path)
    assert done.returncode == 0, done.stderr
    assert [p[0] for p in passes] == ["/out/long512k", "/out/long1m"]
    for p in passes:
        assert "long_ppl_books" in p[2] and "infinitebench" not in p[2] and "long_ppl_1m" not in p[2]
    done, passes = _launcher(tmp_path, path, env={"EVAL_SUITES": "babilong"})
    assert done.returncode == 0 and passes[-1][2] == "babilong_1m"


def test_named_million_token_tasks_are_checked_against_the_window():
    import re
    source = ast.parse((ROOT / "eval.py").read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_check_task_lengths"
             or isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_LENGTH_NAMED_TASK" for t in node.targets)]
    scope = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "eval.py", "exec"), scope)
    check = scope["_check_task_lengths"]
    check(["babilong_1m", "long_ppl_books_1m_1m", "longbench2_1m", "niah_turns_1m"], 1049600)
    with pytest.raises(ValueError, match="1048576"):
        check(["long_ppl_books_1m_8k"], 525312)
    with pytest.raises(ValueError, match="longbench2_512k"):
        check(["longbench2_512k"], 1049600)


def test_babilong_ordinary_arrow_is_offline_and_requires_no_links_or_renames(tmp_path, monkeypatch):
    import datasets
    import requests
    from unused import long_eval_download as prep

    monkeypatch.setenv("BABILONG_DIR", str(tmp_path / "babilong"))
    rows = [dict(input=f"Mary went to the kitchen. {i}", question="Where is Mary?", target="kitchen") for i in range(100)]
    raw = json.dumps(rows).encode()
    names = [f"data/qa{i}/{length}.json" for i in range(1, 6) for length in ("512k", "1M")]
    monkeypatch.setattr(prep, "hub_info", lambda repo: SimpleNamespace(sha="pinned", siblings=[
        SimpleNamespace(rfilename=n, size=len(raw), lfs={"sha256": hashlib.sha256(raw).hexdigest()}) for n in names]))
    calls = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, **kwargs): yield raw

    def get(url, **kwargs):
        calls.append(url)
        assert "/resolve/pinned/" in url
        return Response()

    monkeypatch.setattr(requests, "get", get)
    forbidden = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no links, renames, or dataset builders"))
    monkeypatch.setattr(os, "symlink", forbidden)
    monkeypatch.setattr(os, "rename", forbidden)
    monkeypatch.setattr(os, "replace", forbidden)
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    for length in ("512k", "1m"):
        lines = []
        prep.prepare_babilong(length, lines)
        assert sum("offline reload: ok" in line for line in lines) == 5
        data = prep.babi_utils().load_dataset(length=length, qa_split="qa1")["qa1"]
        assert len(data) == 100 and data[0]["target"] == "kitchen"
        assert data.cache_files  # memory-mapped ordinary Arrow, not a per-rank in-memory copy
        assert all(not Path(item["filename"]).is_symlink() for item in data.cache_files)
    assert len(calls) == 10
    prep.prepare_babilong("1m", [])
    assert len(calls) == 10  # complete ordinary files are reusable
    path = prep.babi_utils().plain_paths("1m", "qa1")[0]
    path.write_bytes(b"/a/flattened/link\0")
    with pytest.raises(ValueError, match="unreadable ordinary"):
        prep.babi_utils().load_plain("1m", "qa1")


def test_bad_download_cannot_be_used_as_babilong(tmp_path, monkeypatch):
    import requests
    from unused import long_eval_download as prep
    monkeypatch.setattr(prep, "hub_info", lambda repo: SimpleNamespace(sha="revision", siblings=[
        SimpleNamespace(rfilename="data/qa1/1M.json", size=4, lfs={"sha256": "f" * 64})]))

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, **kwargs): yield b"oops"

    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response())
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prep.download_plain("someone/data", "data/qa1/1M.json", tmp_path, [])


def test_ruler_extension_caps_only_new_lengths_and_marks_the_protocol(monkeypatch):
    from litgpt import ruler_patch as patch

    monkeypatch.setattr(patch, "LONG_RULER_MIN", 64)
    monkeypatch.setenv("RULER_LONG_SAMPLES", "2")
    calls = []

    def generate(haystack, TOKENIZER=None, *, max_seq_length, type_haystack, num_samples=500):
        calls.append((len(haystack), num_samples, max_seq_length))
        return [dict(length=max_seq_length) for _ in range(num_samples)]

    module = SimpleNamespace(__name__="lm_eval.tasks.ruler.prepare_niah", generate_samples=generate)
    patch._patch_long_ruler_module(module)
    tok = lambda text, **kw: {"input_ids": text.split()}
    short = module.generate_samples(["a", "b"], TOKENIZER=tok, max_seq_length=32, type_haystack="essay")
    assert calls[-1] == (2, 500, 32) and "logkv_ruler_protocol" not in short[0]
    long = module.generate_samples(["a", "b"], TOKENIZER=tok, max_seq_length=128, type_haystack="essay")
    assert calls[-1][0] > 128 and calls[-1][1:] == (2, 128)
    assert long[0]["logkv_ruler_protocol"] == "extended-v1" and long[0]["logkv_haystack_repeats"] > 1


def test_ruler_bare_yaml_modules_receive_the_same_patch(monkeypatch):
    import lm_eval.utils as utils
    from litgpt import ruler_patch as patch

    monkeypatch.setattr(utils, "import_function", utils.import_function)
    patch._patch_yaml_function_imports()
    folder = Path(utils.__file__).parent / "tasks/ruler"
    for name, entry in (("vt", "sys_vartrack_w_noise_random"), ("cwe", "sys_word_pair_random"),
                        ("fwe", "sys_kwext"), ("qa_hotpot", "generate_samples")):
        config = utils.load_yaml_config(str(folder / f"{name}.yaml"))
        fn = config["custom_dataset"]
        assert getattr(fn.__globals__[entry], "_logkv_long_ruler", False), name


def test_long_ppl_curves_select_the_same_longer_documents(tmp_path, monkeypatch):
    utils = load("ultra_ppl", "eval_tasks/long_ppl/long_ppl_utils.py")
    monkeypatch.setenv("LONG_PPL_BOOKS", str(tmp_path / "books.jsonl"))
    monkeypatch.setenv("LONG_PPL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LONG_PPL_DOCS", "2")
    (tmp_path / "books.jsonl").write_text("\n".join(json.dumps({"text": " ".join([word] * n)})
                                                   for word, n in (("short", 40), ("long", 100), ("second", 80))))

    class Tokenizer:
        def __call__(self, text, **kw): return {"input_ids": text.split()}
        def decode(self, ids): return " ".join(ids)

    monkeypatch.setattr(utils, "TAIL", 4)
    monkeypatch.setattr(utils, "_tokenizer", lambda kw: ("word-tokenizer", Tokenizer()))
    calls = []
    original = utils._documents

    def docs(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(utils, "_documents", docs)
    a = utils.long_ppl_books(point=32, min_length=64)["test"]
    b = utils.long_ppl_books(point=64, min_length=64)["test"]
    assert calls[0] == calls[1] and len(calls[0]) == 2
    assert a[0]["context"].split()[0] == b[0]["context"].split()[0] == "long"
    assert a[0]["tail_tokens"] == b[0]["tail_tokens"] == 4
    assert [row["length"] for row in b] == [64, 64]
    for point in ("512k", "1m"):
        config = yaml.safe_load((ROOT / f"eval_tasks/long_ppl/long_ppl_books_{point}_{point}.yaml").read_text())
        assert config["dataset_kwargs"]["min_length"] == 1048576


def test_all_new_task_groups_are_registered():
    from lm_eval.tasks import TaskManager
    manager = TaskManager(include_path=str(ROOT / "eval_tasks"), metadata={"pretrained": "unused"})
    for length in ("512k", "1m"):
        assert {f"babilong_qa{i}_{length}" for i in range(1, 6)} <= set(manager.all_tasks)
        assert {f"long_ppl_{length}", f"niah_turns_{length}"} <= set(manager.all_tasks)
        assert {f"babilong_{length}", f"long_ppl_books_{length}", f"longbench2_{length}"} <= set(manager.all_groups)


def test_book_downloader_scans_until_the_shared_million_token_selection_is_full(tmp_path, monkeypatch):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq
    import transformers
    from unused import long_eval_download as prep
    fetch = load("ultra_fetch_books", "unused/fetch_long_eval_data.py")
    utils = load("ultra_download_ppl", "eval_tasks/long_ppl/long_ppl_utils.py")
    monkeypatch.setattr(utils, "LONGEST_1M", 64)  # exercise threshold logic with small real files
    monkeypatch.setattr(fetch, "_long_ppl_utils", lambda: utils)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("LONG_PPL_BOOKS", raising=False)
    monkeypatch.setenv("LONG_PPL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LONG_PPL_DOCS", "2")
    names = ["data/train-0.parquet", "data/train-1.parquet", "data/train-2.parquet"]
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", lambda *a, **kw: names)

    class Tokenizer:
        def __call__(self, text, **kw): return {"input_ids": [1 if w == "first" else 2 for w in text.split()]}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    calls = []

    def download(repo, name, folder, lines):
        path = Path(folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"text": ["short", " ".join(["first" if name == names[0] else "second"] * 80)]}), path)
        calls.append(name)
        return path

    monkeypatch.setattr(prep, "download_plain", download)
    forbidden = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no hub links or renames"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(os, "rename", forbidden)
    monkeypatch.setattr(os, "replace", forbidden)
    lines = []
    root = fetch.fetch_ppl_books("someone/books", "local-tokenizer", lines, longest=64)
    assert calls == names[:2]
    assert root == str(tmp_path / "long_ppl_books_1m/someone--books")
    fetch.cache_ppl_books(root, "local-tokenizer", lines, longest=64)
    docs = utils._documents({"pretrained": "local-tokenizer"}, 64, env="LONG_PPL_BOOKS")
    assert docs == [[1] * 64, [2] * 64]
