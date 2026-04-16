import os
import json
import re
import functools
from typing import Any
import torch
import random
import numpy as np
from litgpt.tokenizer import Tokenizer

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


class CPTBinDataset(torch.utils.data.Dataset):
    """
    极速读取预编译的 .bin 文件。CPU 开销几乎为 0。
    """
    def __init__(self, bin_path: str, seq_len: int):
        # 魔法所在：memmap 不会把大文件一次性读进内存，而是按需在硬盘和内存间滑动
        self.data = np.memmap(bin_path, dtype=np.int32, mode='r')
        self.seq_len = seq_len
        self.chunk_size = seq_len + 1
        
        # 精确计算这个文件一共能切出多少个完整的 Batch
        self.total_chunks = len(self.data) // self.chunk_size

    def __len__(self):
        # 现在 Dataloader 终于知道你有多少数据了！
        return self.total_chunks

    def __getitem__(self, idx: int):
        # 直接用索引在极速数组上切片
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        # 截取数据并转为 int64 张量 (PyTorch Embedding 层的硬性要求)
        chunk = self.data[start_idx:end_idx]
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))
        
        # 返回 inputs 和 targets
        return chunk_tensor[:-1], chunk_tensor[1:]


class CPTOnlineJsonlDataset(torch.utils.data.Dataset):
    """
    支持在线 Tokenize 的 JSONL Dataset。
    在内部初始化 Tokenizer，并在 __getitem__ 时实时编码文本。
    """
    def __init__(self, jsonl_path: str, model_dir: str, seq_len: int, text_key: str = "content"):
        self.jsonl_path = jsonl_path
        self.seq_len = seq_len
        self.chunk_size = seq_len + 1  # 额外保留一位用于预测 target
        self.text_key = text_key
        
        print(f"⏳ [Dataset] 正在初始化 Tokenizer ({model_dir})...")
        # 直接在 Dataset 内部实例化 Tokenizer
        self.tokenizer = Tokenizer(model_dir)
        
        # 尝试获取 pad_id，如果你的 Tokenizer 没有定义 pad_id，默认使用 0
        self.pad_id = getattr(self.tokenizer, 'pad_id', 0)

        print(f"⏳ [Dataset] 正在扫描 {jsonl_path} 构建索引...")
        self.line_offsets = []
        with open(jsonl_path, 'rb') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)
                
        self.total_lines = len(self.line_offsets)
        print(f"✅ [Dataset] 索引构建完成，共发现 {self.total_lines} 条记录。")

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx: int):
        # 1. 读取单行
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            line_str = f.readline()
            line_data = json.loads(line_str)
            
        text = line_data.get(self.text_key, "")
        
        # 2. 实时 Tokenize (包含 EOS 结束符)
        tokens = self.tokenizer.encode(text, bos=False, eos=True)
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
            
        # 3. 截断 (Truncation)
        # 如果长度超过了所需的最大容量 (seq_len + 1)
        if len(tokens) > self.chunk_size:
            tokens = tokens[:self.chunk_size]
            
        # 4. 分离 Input 和 Label
        # 这里非常重要：先错位分离，再各自 Padding
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # 5. 独立填充 (Padding)
        if len(input_ids) < self.seq_len:
            pad_len = self.seq_len - len(input_ids)
            # Inputs 用正常的 pad_id 填充 (保证 Embedding 不报错)
            input_ids = input_ids + [self.pad_id] * pad_len
            # Labels 用 -100 填充 (让 CrossEntropyLoss 忽略这些位置的 Loss 计算)
            labels = labels + [-100] * pad_len
            
        # 6. 转换为 Tensor 并返回
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        return input_tensor, label_tensor
    