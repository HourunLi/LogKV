"""RULER 离线补丁 — 运行时 monkey-patch，不修改 conda 系统包。

在 eval.py 中 `from lm_eval import evaluator` 之后导入本模块即可：

    from lm_eval import evaluator
    from litgpt.ruler_patch import apply_patch  # noqa: I001
    apply_patch()

原理：
1. 全局 monkey-patch requests.Session.request，强制 verify=False（对抗代理 SSL）
2. 劫持 lm_eval.tasks.ruler.qa_utils.download_json，优先从本地缓存加载
3. 安装 import hook，确保 qa_utils 被延迟导入时也能被 patch
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import ssl
import sys
from functools import cache
from pathlib import Path

LONG_RULER_MIN = 524288


def _essay_file() -> Path:
    return Path(os.environ.get("RULER_ESSAYS_FILE") or
                Path(_get_ruler_cache_dir()) / "paul_graham_essays.txt").expanduser()


def _expand_essay(haystack, tokenizer, target: int):
    """Cyclically repeat only the distractor background; the upstream generator inserts each needle once."""
    import math

    if not isinstance(haystack, list) or not haystack:
        raise ValueError("RULER essay background is empty")
    tokens = len(tokenizer(" ".join(haystack), add_special_tokens=False)["input_ids"])
    if not tokens:
        raise ValueError("RULER essay background tokenizes to zero tokens")
    repeats = max(1, math.ceil((target + 2048) / tokens))
    expanded = haystack * repeats
    while len(tokenizer(" ".join(expanded), add_special_tokens=False)["input_ids"]) < target + 1024:
        repeats += 1
        expanded = haystack * repeats
    return expanded, repeats


def _patch_long_ruler_module(module) -> None:
    """Keep <=256K recipes; cap million-token datasets and make their actual lengths observable."""
    import functools
    import inspect

    entry = next((name for name in ("generate_samples", "sys_vartrack_w_noise_random",
                                    "sys_word_pair_random", "sys_kwext") if hasattr(module, name)), None)
    original = getattr(module, entry) if entry else None
    if original is not None and not getattr(original, "_logkv_long_ruler", False):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def generate(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            length = int(values.get("max_seq_length", 0))
            if length < LONG_RULER_MIN:
                return original(*args, **kwargs)
            cap = int(os.environ.get("RULER_LONG_SAMPLES", "100"))
            if cap < 1:
                raise ValueError("RULER_LONG_SAMPLES must be positive")
            if "num_samples" in values:
                values["num_samples"] = min(int(values["num_samples"]), cap)
            repeats = 1
            if values.get("type_haystack") == "essay":
                values["haystack"], repeats = _expand_essay(values["haystack"], values["TOKENIZER"], length)
            rows = original(*bound.args, **bound.kwargs)
            for row in rows:
                actual = int(row.get("length", 0))
                if actual < length - 4096:
                    raise ValueError(f"RULER requested {length} tokens but generated only {actual}; "
                                     "the background/document pool cannot fill this length")
                row["logkv_ruler_protocol"] = "extended-v1"
                row["logkv_haystack_repeats"] = repeats
            return rows

        generate._logkv_long_ruler = True
        setattr(module, entry, generate)
    original_haystack = getattr(module, "get_haystack", None)
    if original_haystack is not None and not getattr(original_haystack, "_logkv_plain_essay", False):
        @functools.wraps(original_haystack)
        def get_haystack(type_haystack, *args, **kwargs):
            path = _essay_file()
            if type_haystack == "essay" and path.is_file():
                if path.is_symlink():
                    raise ValueError(f"RULER_ESSAYS_FILE must be an ordinary file: {path}")
                return path.read_text(encoding="utf-8").split()
            return original_haystack(type_haystack, *args, **kwargs)
        get_haystack._logkv_plain_essay = True
        module.get_haystack = get_haystack


def _patch_ruler_module(module) -> None:
    if module.__name__.rsplit(".", 1)[-1] == "qa_utils":
        _patch_qa_utils_module(module)
    _patch_long_ruler_module(module)


def _patch_yaml_function_imports() -> None:
    """lm-eval also executes task utils under bare names, bypassing normal Python import hooks."""
    import functools
    from types import SimpleNamespace
    import lm_eval.utils as utils

    original = utils.import_function
    if getattr(original, "_logkv_ruler_import", False):
        return

    @functools.wraps(original)
    def import_function(*args, **kwargs):
        function = original(*args, **kwargs)
        namespace = getattr(function, "__globals__", {})
        filename = str(namespace.get("__file__", "")).replace("\\", "/")
        if "/lm_eval/tasks/ruler/" in filename:
            proxy = SimpleNamespace(**namespace)
            _patch_ruler_module(proxy)
            for name in ("generate_samples", "get_haystack", "sys_vartrack_w_noise_random",
                         "sys_word_pair_random", "sys_kwext", "download_json"):
                if name in namespace and hasattr(proxy, name):
                    namespace[name] = getattr(proxy, name)
        return function

    import_function._logkv_ruler_import = True
    utils.import_function = import_function


def _get_ruler_cache_dir() -> str:
    return os.environ.get(
        "RULER_CACHE_DIR",
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "ruler_cache",
        ),
    )


_URL_TO_FILENAME = {
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json": "dev-v2.0.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json": "hotpot_dev_distractor_v1.json",
}


def _load_json_local(filename: str) -> dict | None:
    local_path = os.path.join(_get_ruler_cache_dir(), filename)
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _download_and_cache(url: str, filename: str) -> dict:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests as req

    resp = req.get(url, verify=False, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    cache_dir = _get_ruler_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


@cache
def _patched_download_json(url: str) -> dict:
    offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
    filename = _URL_TO_FILENAME.get(url)

    if filename:
        data = _load_json_local(filename)
        if data is not None:
            return data

    if offline:
        raise RuntimeError(
            f"HF_DATASETS_OFFLINE=1 且 RULER 本地缓存未找到。"
            f"需要将 {filename or url.split('/')[-1]} 放到 {_get_ruler_cache_dir()}/"
        )

    if filename:
        return _download_and_cache(url, filename)

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests as req
    resp = req.get(url, verify=False, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _patch_qa_utils_module(qa_utils) -> None:
    if getattr(qa_utils, "_ruler_patched", False):
        return
    qa_utils.download_json = _patched_download_json
    qa_utils._ruler_patched = True


class _FakeResponse:
    """模拟 requests.Response，用于从本地缓存返回数据。"""

    def __init__(self, data: dict):
        self._data = data
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def _ssl_bypass_enabled() -> bool:
    """是否全局关闭 SSL 验证。默认开启（对抗内网代理的自签证书）；
    设 ``RULER_SSL_NO_VERIFY=0`` 可恢复正常证书校验（本地缓存拦截不受影响）。
    注意：开启时影响评测进程内**所有** requests HTTPS 请求，属于有意的全局行为。"""
    return os.environ.get("RULER_SSL_NO_VERIFY", "1") != "0"


def _patch_requests_ssl() -> None:
    """全局 patch requests：拦截已知 RULER URL 直接返回本地缓存；
    并在 ``RULER_SSL_NO_VERIFY != 0`` 时强制所有 HTTPS 请求跳过 SSL 验证。"""
    import requests

    if getattr(requests.Session, "_ssl_patched", False):
        return

    _original_request = requests.Session.request

    def _patched_request(self, method, url, **kwargs):
        if _ssl_bypass_enabled():
            kwargs.setdefault("verify", False)
        filename = _URL_TO_FILENAME.get(url)
        if filename:
            data = _load_json_local(filename)
            if data is not None:
                return _FakeResponse(data)
        return _original_request(self, method, url, **kwargs)

    requests.Session.request = _patched_request
    requests.Session._ssl_patched = True  # type: ignore[attr-defined]

    _original_get = requests.get

    def _patched_get(url, **kwargs):
        filename = _URL_TO_FILENAME.get(url)
        if filename:
            data = _load_json_local(filename)
            if data is not None:
                return _FakeResponse(data)
        if _ssl_bypass_enabled():
            kwargs.setdefault("verify", False)
        return _original_get(url, **kwargs)

    if not getattr(requests, "_get_patched", False):
        requests.get = _patched_get
        requests._get_patched = True  # type: ignore[attr-defined]


class _QaUtilsImportHook:
    """Meta path finder：当 lm_eval.tasks.ruler.qa_utils 被导入时自动 patch。
    同时兼容旧式 find_module 和新式 find_spec API。"""

    _TARGET = "lm_eval.tasks.ruler.qa_utils"
    _TARGETS = tuple("lm_eval.tasks.ruler." + name for name in (
        "qa_utils", "prepare_niah", "niah_utils", "vt_utils", "cwe_utils", "fwe_utils"))

    def find_module(self, fullname, path=None):
        if fullname in self._TARGETS:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            mod = sys.modules[fullname]
            _patch_ruler_module(mod)
            return mod
        sys.meta_path.remove(self)
        try:
            mod = importlib.import_module(fullname)
        finally:
            sys.meta_path.insert(0, self)
        _patch_ruler_module(mod)
        return mod

    def find_spec(self, fullname, path, target=None):
        """新式 API (PEP 451)，确保 Python 3.10+ 也能触发。"""
        if fullname not in self._TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None:
            return None
        orig_exec = spec.loader.exec_module if hasattr(spec.loader, "exec_module") else None
        if orig_exec:
            def _exec_module_wrapper(module):
                orig_exec(module)
                _patch_ruler_module(module)
            spec.loader.exec_module = _exec_module_wrapper
        return spec


def apply_patch() -> None:
    """安装所有补丁：SSL 绕过（可用 RULER_SSL_NO_VERIFY=0 关闭）+ qa_utils patch + import hook"""

    # 1. SSL 全局绕过（urllib/ssl 层），受 RULER_SSL_NO_VERIFY 开关控制
    if _ssl_bypass_enabled():
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ.setdefault("CURL_CA_BUNDLE", "")
        os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

    # 2. 全局 patch requests.Session.request（缓存拦截始终生效；verify=False 受开关控制）
    try:
        _patch_requests_ssl()
    except Exception:
        pass

    # 3. 禁用 urllib3 SSL 警告（仅在绕过开启时需要）
    if _ssl_bypass_enabled():
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

    # 4. 尝试直接 patch qa_utils（如果已经被导入）
    qa_utils = sys.modules.get("lm_eval.tasks.ruler.qa_utils")
    if qa_utils is not None:
        _patch_qa_utils_module(qa_utils)
    else:
        # 尝试主动导入
        try:
            qa_utils = importlib.import_module("lm_eval.tasks.ruler.qa_utils")
            _patch_qa_utils_module(qa_utils)
        except (ImportError, ModuleNotFoundError, Exception):
            pass

    # 5. 安装 import hook，确保后续延迟导入也能被 patch
    if not any(isinstance(f, _QaUtilsImportHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _QaUtilsImportHook())
    for name in _QaUtilsImportHook._TARGETS:
        if name in sys.modules:
            _patch_ruler_module(sys.modules[name])
    _patch_yaml_function_imports()
