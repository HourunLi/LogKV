import os
import re
import functools
from typing import Any
import torch
import random

def _expand_single_string(text: str) -> str:
    """底层的单字符串替换逻辑"""
    def replace_fn(match):
        inner = match.group(1)
        if ':-' in inner:
            var_name, default_val = inner.split(':-', 1)
        elif '-' in inner:
            var_name, default_val = inner.split('-', 1)
        else:
            var_name, default_val = inner, ""
        return os.environ.get(var_name, default_val)
    return re.sub(r'\$\{([^}^{]+)\}', replace_fn, text)

def _deep_expand(obj: Any) -> Any:
    """递归遍历：支持嵌套的字典、列表、元组，精准爆破所有字符串"""
    if isinstance(obj, str):
        return _expand_single_string(obj)
    elif isinstance(obj, dict):
        return {k: _deep_expand(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_expand(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_expand(v) for v in obj)
    # 如果是 int, float, bool 等基本类型，直接原样返回
    return obj

def auto_expand_env_vars(func):
    """
    魔法装饰器：拦截 jsonargparse 传进来的所有参数，清洗后再喂给目标函数。
    functools.wraps 极其关键，它能保留原函数的 Type Hint 签名，
    保证 jsonargparse CLI 依然能正常解析！
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        expanded_args = _deep_expand(args)
        expanded_kwargs = _deep_expand(kwargs)
        return func(*expanded_args, **expanded_kwargs)
    return wrapper


class MicroStepMeanStats:
    """
    同一 global step 内按 micro batch 累加；每个指标各自维护 sum / count。
    accumulate 里没传的指标本步不更新；averages() 只返回本 step 内至少收到过一次样本的指标。
    构造时的名字仅用于预置键（reset 后会清空，之后仍可在 accumulate 里动态出现新键名）。
    """

    def __init__(self, *metric_names: str) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        for k in metric_names:
            self._sums[k] = 0.0
            self._counts[k] = 0

    def accumulate(self, **values: float) -> None:
        for k, v in values.items():
            if k not in self._sums:
                self._sums[k] = 0.0
                self._counts[k] = 0
            self._sums[k] += float(v)
            self._counts[k] += 1

    def averages(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k] > 0}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# set random seeds
def set_random_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)