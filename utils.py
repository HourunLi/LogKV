import os
import json
import re
import functools
import shutil
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

import os
import glob
import json
import numpy as np
import pyarrow.parquet as pq
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset, Dataset

# ==============================================================================
# 🚀 必须放在最外层的独立工作函数 (Worker) 
# 因为 Python multiprocessing 不能传递内部函数或带有锁的对象
# ==============================================================================
def _process_data_chunk(kwargs):
    """独立的子进程处理函数"""
    ext = kwargs['ext']
    src_file = kwargs['src_file']
    bin_file = kwargs['bin_file']
    model_dir = kwargs['model_dir']
    
    # ⚠️ 必须在子进程内部初始化 Tokenizer，不能从主进程传进来
    tokenizer = Tokenizer(model_dir)
    tmp_bin_file = f"{bin_file}.tmp"

    with open(tmp_bin_file, "wb") as f:
        # --------------------------------------------------
        # Parquet 切片处理
        # --------------------------------------------------
        if ext == ".parquet":
            parquet_file = pq.ParquetFile(src_file)
            for rg in kwargs['row_groups']:
                # 直接读取指定的 Row Group
                table = parquet_file.read_row_group(rg, columns=["text"])
                batch_tokens = []
                for text in table.to_pandas()["text"]:
                    tokens = tokenizer.encode(text, bos=False, eos=True).tolist()
                    batch_tokens.extend(tokens)
                
                if batch_tokens:
                    arr = np.array(batch_tokens, dtype=np.int32)
                    f.write(arr.tobytes())
                    
        # --------------------------------------------------
        # JSONL 切片处理
        # --------------------------------------------------
        elif ext == ".jsonl":
            start_byte = kwargs['start_byte']
            end_byte = kwargs['end_byte']
            
            with open(src_file, 'r', encoding='utf-8') as jf:
                jf.seek(start_byte)
                # 不是文件头的话，跳过残行
                if start_byte > 0:
                    jf.readline()

                batch_tokens = []
                while True:
                    if jf.tell() >= end_byte:
                        break
                    line = jf.readline()
                    if not line:
                        break
                        
                    try:
                        line_data = json.loads(line)
                        text = line_data.get("text", line_data.get("content", ""))
                        if not text:
                            continue

                        tokens = tokenizer.encode(text, bos=False, eos=True).tolist()
                        batch_tokens.extend(tokens)

                        if len(batch_tokens) > 1000000:
                            arr = np.array(batch_tokens, dtype=np.int32)
                            f.write(arr.tobytes())
                            batch_tokens = []
                    except json.JSONDecodeError:
                        continue
                        
                if batch_tokens:
                    arr = np.array(batch_tokens, dtype=np.int32)
                    f.write(arr.tobytes())

    # 移正文件
    shutil.move(tmp_bin_file, bin_file)
    return bin_file


# ==============================================================================
# 主控函数
# ==============================================================================
def prepare_data(dataset_dir, dataset_name, model_dir, data_dir=None, rank=0, local_rank=0, world_size=1):
    # 动态获取当前机器的 CPU 核心数
    num_cpus = 64
    # 计算当前 Rank 能分到几个 CPU 核心 (至少保证 1 个)
    workers_per_rank = max(1, num_cpus // world_size)
    # 重新定义文件切分总数 = 物理总 GPU 数 × 每个 GPU 挂载的 CPU 进程数
    total_parts = world_size * workers_per_rank

    if rank == 0:
        print(f"⏳ [Rank 0] 系统检测到 {num_cpus} 个 CPU 核心。")
        print(f"🚀 [Rank 0] 并行策略：将每个文件切分为 {total_parts} 份，每个 Rank 分配 {workers_per_rank} 个子进程狂飙！")

    os.makedirs(dataset_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(model_dir))
    bin_cache_dir = os.path.join(data_dir, f"bins_{model_name}") if data_dir else os.path.join(dataset_dir, f"bins_{model_name}")
    os.makedirs(bin_cache_dir, exist_ok=True)

    bin_paths = []
    all_files = []

    # 1. 找文件 (逻辑不变)
    if data_dir and os.path.isdir(data_dir):
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
        all_files = parquet_files + jsonl_files
        if not all_files:
            raise FileNotFoundError(f"❌ 在 {data_dir} 下没有找到任何 .parquet 或 .jsonl 文件！")
    else:
        single_parquet = os.path.join(dataset_dir, f"{dataset_name}.parquet")
        if not os.path.exists(single_parquet):
            if dataset_name == "debug":
                if rank == 0: print(f"🌐 [Rank 0] 正在拉取 FineWeb-Edu...")
                eng_stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)
                mixed_dataset = Dataset.from_list(list(eng_stream.take(10000))).select_columns(["text"])
                mixed_dataset.to_parquet(single_parquet)
            else:
                raise NotImplementedError(f"数据集 {dataset_name} 尚未实现")
        all_files = [single_parquet]

    # 2. 预计算所有的最终输出路径，外层 Dataset 调用时保证各个 Rank 对齐
    for src_file in all_files:
        rel_path = os.path.relpath(src_file, data_dir if data_dir else dataset_dir)
        base_name = rel_path.replace(os.sep, "__").replace(".parquet", "").replace(".jsonl", "")
        for p in range(total_parts):
            bin_paths.append(os.path.join(bin_cache_dir, f"{base_name}_part{p}.bin"))

    # 3. 收集当前 Rank 需要处理的所有 "微任务 (Task)"
    my_tasks = []
    for src_file in all_files:
        ext = os.path.splitext(src_file)[1].lower()
        rel_path = os.path.relpath(src_file, data_dir if data_dir else dataset_dir)
        base_name = rel_path.replace(os.sep, "__").replace(".parquet", "").replace(".jsonl", "")
        
        # 将一个文件切片为 total_parts 份
        for part_id in range(total_parts):
            # 过滤出只属于当前 Rank 的 part (轮询分配)
            if part_id % world_size != rank:
                continue
                
            bin_file = os.path.join(bin_cache_dir, f"{base_name}_part{part_id}.bin")
            if os.path.exists(bin_file):
                continue
                
            task = {
                'ext': ext,
                'src_file': src_file,
                'bin_file': bin_file,
                'model_dir': model_dir
            }

            if ext == ".parquet":
                parquet_file = pq.ParquetFile(src_file)
                num_row_groups = parquet_file.num_row_groups
                # 当前微任务负责的 row_groups (交错切分)
                my_rgs = [i for i in range(num_row_groups) if i % total_parts == part_id]
                if not my_rgs: 
                    continue # 文件极小，没分到
                task['row_groups'] = my_rgs

            elif ext == ".jsonl":
                total_size = os.path.getsize(src_file)
                chunk_size = total_size // total_parts
                task['start_byte'] = part_id * chunk_size
                task['end_byte'] = total_size if part_id == total_parts - 1 else task['start_byte'] + chunk_size
                
            my_tasks.append(task)

    # 4. 进程池多核全开，执行处理任务
    if my_tasks:
        print(f"🔥 [Rank {rank}] 分配到 {len(my_tasks)} 个分块任务，启动 {workers_per_rank} 个进程全力处理...")
        
        with ProcessPoolExecutor(max_workers=workers_per_rank) as executor:
            # 提交所有微任务
            futures = [executor.submit(_process_data_chunk, task) for task in my_tasks]
            
            # 使用 tqdm 监控当前 Rank 的进程池进度
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"[Rank {rank}] 多进程编译中", position=local_rank, leave=True):
                pass
                
        print(f"✅ [Rank {rank}] 所有分块处理完毕！")

    return bin_paths

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
    