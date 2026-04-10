import os
import re
import functools
from typing import Any
import torch
import random
import glob
from datasets import load_dataset, concatenate_datasets, Dataset
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm
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

def prepare_data(dataset_dir, dataset_name, model_dir, data_dir=None, rank=0, local_rank=0, world_size=1):
    print(f"[Rank {rank}] ⏳ 正在检查和准备数据...")
    os.makedirs(dataset_dir, exist_ok=True)

    model_name = os.path.basename(os.path.normpath(model_dir))
    bin_cache_dir = os.path.join(data_dir, f"bins_{model_name}") if data_dir else os.path.join(dataset_dir, f"bins_{model_name}")
    os.makedirs(bin_cache_dir, exist_ok=True)

    bin_paths = []
    parquet_files = []

    # 1. 找 parquet 文件
    if data_dir and os.path.isdir(data_dir):
        print(f"[Rank {rank}] 📂 检测到本地数据集目录: {data_dir}")
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"❌ 在 {data_dir} 下没有找到任何 .parquet 文件！")
    else:
        single_parquet = os.path.join(dataset_dir, f"{dataset_name}.parquet")
        if not os.path.exists(single_parquet):
            if dataset_name == "debug":
                print(f"[Rank {rank}] 🌐 正在拉取 FineWeb-Edu...")
                eng_stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)
                eng_list = list(eng_stream.take(10000))
                mixed_dataset = Dataset.from_list(eng_list).select_columns(["text"])
                mixed_dataset.to_parquet(single_parquet)
            else:
                raise NotImplementedError(f"数据集 {dataset_name} 尚未实现")
        parquet_files = [single_parquet]

    # 2. 先把所有目标 bin 路径算出来，所有 rank 都返回同样的 bin_paths
    compile_jobs = []
    for file_idx, pq_file in enumerate(parquet_files):
        rel_path = os.path.relpath(pq_file, data_dir)
        base_name = rel_path.replace(os.sep, "__").replace(".parquet", "")
        bin_file = os.path.join(bin_cache_dir, f"{base_name}.bin")
        bin_paths.append(bin_file)

        if os.path.exists(bin_file):
            continue

        if file_idx % world_size == rank:
            compile_jobs.append((pq_file, bin_file, base_name))

    # 3. 当前 rank 只编译自己那部分
    if compile_jobs:
        tokenizer = Tokenizer(model_dir)


    for pq_file, bin_file, base_name in compile_jobs:
        parquet_file = pq.ParquetFile(pq_file)
        tmp_bin_file = f"{bin_file}.rank{rank}.tmp"
        # if os.path.exists(tmp_bin_file):
        #     os.replace(tmp_bin_file, bin_file)
        #     continue

        total_batches = parquet_file.metadata.num_rows // 8192
        if parquet_file.metadata.num_rows % 8192 != 0:
            total_batches += 1

        with open(tmp_bin_file, "wb") as f:
            pbar = tqdm(
                parquet_file.iter_batches(batch_size=8192, columns=["text"]),
                total=total_batches,
                desc=f"[Rank {rank}] {base_name}",
                position=local_rank,
                leave=True,
                dynamic_ncols=True,
                mininterval=0.5,
            )

            for batch in pbar:
                batch_tokens = []
                for text in batch.to_pandas()["text"]:
                    tokens = tokenizer.encode(text, bos=False, eos=True).tolist()
                    batch_tokens.extend(tokens)

                arr = np.array(batch_tokens, dtype=np.int32)
                f.write(arr.tobytes())

        os.replace(tmp_bin_file, bin_file)

    return bin_paths

