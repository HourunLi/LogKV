"""RULER 离线补丁 — 运行时 monkey-patch，不修改 conda 系统包。

在 eval.py 中 `from lm_eval import evaluator` 之后导入本模块即可：

    from lm_eval import evaluator
    from litgpt.eval.ruler_patch import apply_patch  # noqa: I001
    apply_patch()

原理：劫持 lm_eval.tasks.ruler.qa_utils.download_json，优先从
RULER_CACHE_DIR 本地目录加载 JSON，离线模式下强制使用本地文件，
在线模式下 fallback 到 requests.get(verify=False)。
"""

from __future__ import annotations

import json
import os
import ssl
from functools import cache
from pathlib import Path


def _get_ruler_cache_dir() -> str:
    """返回 RULER 本地缓存目录路径"""
    return os.environ.get(
        "RULER_CACHE_DIR",
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "ruler_cache",
        ),
    )


# URL → 本地文件名的映射
_URL_TO_FILENAME = {
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json": "dev-v2.0.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json": "hotpot_dev_distractor_v1.json",
}

_OFFLINE_MODE = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"


def _load_json_local(filename: str) -> dict | None:
    """从本地缓存加载 JSON 文件"""
    local_path = os.path.join(_get_ruler_cache_dir(), filename)
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _download_and_cache(url: str, filename: str) -> dict:
    """下载 JSON 并缓存到本地，SSL 验证关闭"""
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


def apply_patch() -> None:
    """安装 monkey-patch：劫持 lm_eval.tasks.ruler.qa_utils.download_json"""

    # 设置 SSL 全局绕过（对抗公司代理的 self-signed cert）
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

    try:
        from lm_eval.tasks.ruler import qa_utils
    except ImportError:
        return  # lm_eval 未安装或版本不含 ruler，静默跳过

    if getattr(qa_utils, "_ruler_patched", False):
        return  # 已经打过补丁

    original_download_json = qa_utils.download_json

    @cache
    def patched_download_json(url: str) -> dict:
        filename = _URL_TO_FILENAME.get(url)

        # 1. 先尝试本地缓存
        if filename:
            data = _load_json_local(filename)
            if data is not None:
                return data

        # 2. 离线模式 — 报错
        if _OFFLINE_MODE:
            raise RuntimeError(
                f"HF_DATASETS_OFFLINE=1 且 RULER 本地缓存未找到。"
                f"需要将 {filename} 放到 {_get_ruler_cache_dir()}/"
            )

        # 3. 在线模式 — 下载（关闭 SSL 验证）并缓存
        if filename:
            return _download_and_cache(url, filename)
        else:
            # 未知 URL，用原始函数（但关闭 SSL 验证）
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            import requests as req
            resp = req.get(url, verify=False, timeout=120)
            resp.raise_for_status()
            return resp.json()

    qa_utils.download_json = patched_download_json  # type: ignore[attr-defined]
    qa_utils._ruler_patched = True  # type: ignore[attr-defined]
