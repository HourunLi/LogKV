"""
原始数据（parquet / jsonl）→ token 化写入 .bin 的离线预处理。

缓存目录仅依赖 tokenizer 文件内容（与模型权重无关）；相同 tokenizer 的不同 checkpoint 可共用同一份 bin。
分片规则仅由数据形态与固定参数决定，与 GPU、分布式 rank、进程数无关。

训练脚本请只使用 utils.list_tokenized_bin_paths 等接口读取已有 .bin。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from tqdm import tqdm

from litgpt.tokenizer import Tokenizer

DEFAULT_JSONL_SHARD_BYTES = 256 * 1024 * 1024  # 单文件内按字节切分，与 worker 数量无关


def tokenizer_cache_key(tokenizer_dir: str) -> str:
    """
    根据 tokenizer 相关文件内容生成短缓存 id；内容相同则 id 相同（不同模型目录也可复用 bin）。
    """
    p = Path(os.path.normpath(tokenizer_dir)).resolve()
    h = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        f = p / name
        if f.is_file():
            h.update(name.encode())
            h.update(f.read_bytes())
    if not (p / "tokenizer.json").is_file() and not (p / "tokenizer.model").is_file():
        raise FileNotFoundError(
            f"在 {tokenizer_dir} 下未找到 tokenizer.json 或 tokenizer.model，无法确定 tokenizer 身份。"
        )
    return h.hexdigest()[:16]


def resolve_bin_cache_dir(tokenizer_dir: str, data_dir: str | None, dataset_dir: str) -> str:
    key = tokenizer_cache_key(tokenizer_dir)
    root = data_dir if data_dir else dataset_dir
    return os.path.join(root, f"bins_{key}")


def list_tokenized_bin_paths(
    tokenizer_dir: str,
    dataset_dir: str,
    dataset_name: str,
    data_dir: str | None = None,
) -> list[str]:
    """
    列出已由 prepare_tokenized_data 写入的 .bin 分片（与训练时 tokenizer 目录一致即可定位缓存）。
    """
    cache_dir = resolve_bin_cache_dir(tokenizer_dir, data_dir, dataset_dir)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"未找到 tokenizer 缓存目录: {cache_dir}\n"
            "请先运行: python data.py --tokenizer_dir ... （见 data.py 底部参数说明）"
        )
    bins = sorted(glob.glob(os.path.join(cache_dir, "*.bin")))
    if not bins:
        raise FileNotFoundError(
            f"目录 {cache_dir} 下没有 .bin 文件。请先运行 prepare_tokenized_data 完成预处理。"
        )
    return bins


def _source_base_name(src_file: str, data_dir: str | None, dataset_dir: str) -> str:
    rel_path = os.path.relpath(src_file, data_dir if data_dir else dataset_dir)
    return rel_path.replace(os.sep, "__").replace(".parquet", "").replace(".jsonl", "")


def _build_tasks(
    all_files: list[str],
    bin_cache_dir: str,
    tokenizer_dir: str,
    data_dir: str | None,
    dataset_dir: str,
    jsonl_shard_bytes: int,
) -> list[dict[str, Any]]:
    """构造任务列表：输出文件名仅由数据源 + row group / 字节区间决定。"""
    tasks: list[dict[str, Any]] = []
    for src_file in all_files:
        ext = os.path.splitext(src_file)[1].lower()
        base = _source_base_name(src_file, data_dir, dataset_dir)

        if ext == ".parquet":
            parquet_file = pq.ParquetFile(src_file)
            for rg_idx in range(parquet_file.num_row_groups):
                bin_file = os.path.join(bin_cache_dir, f"{base}__rg{rg_idx:06d}.bin")
                tasks.append(
                    {
                        "ext": ext,
                        "src_file": src_file,
                        "bin_file": bin_file,
                        "tokenizer_dir": tokenizer_dir,
                        "row_groups": [rg_idx],
                    }
                )

        elif ext == ".jsonl":
            total_size = os.path.getsize(src_file)
            chunk_idx = 0
            start = 0
            while start < total_size:
                end = min(start + jsonl_shard_bytes, total_size)
                bin_file = os.path.join(bin_cache_dir, f"{base}__j{chunk_idx:05d}.bin")
                tasks.append(
                    {
                        "ext": ext,
                        "src_file": src_file,
                        "bin_file": bin_file,
                        "tokenizer_dir": tokenizer_dir,
                        "start_byte": start,
                        "end_byte": end,
                    }
                )
                start = end
                chunk_idx += 1
        else:
            raise ValueError(f"不支持的扩展名: {ext}")

    return tasks


# ==============================================================================
# 必须放在最外层的独立工作函数 (Worker)
# 因为 Python multiprocessing 不能传递内部函数或带有锁的对象
# ==============================================================================
def _process_data_chunk(kwargs: dict[str, Any]) -> str | None:
    ext = kwargs["ext"]
    src_file = kwargs["src_file"]
    bin_file = kwargs["bin_file"]
    tokenizer_dir = kwargs["tokenizer_dir"]

    if os.path.exists(bin_file):
        return bin_file

    tokenizer = Tokenizer(tokenizer_dir)
    tmp_bin_file = f"{bin_file}.tmp"

    with open(tmp_bin_file, "wb") as f:
        if ext == ".parquet":
            parquet_file = pq.ParquetFile(src_file)
            for rg in kwargs["row_groups"]:
                table = parquet_file.read_row_group(rg, columns=["text"])
                batch_tokens = []
                for text in table.to_pandas()["text"]:
                    tokens = tokenizer.encode(text, bos=False, eos=True).tolist()
                    batch_tokens.extend(tokens)

                if batch_tokens:
                    arr = np.array(batch_tokens, dtype=np.int32)
                    f.write(arr.tobytes())

        elif ext == ".jsonl":
            start_byte = kwargs["start_byte"]
            end_byte = kwargs["end_byte"]

            with open(src_file, "r", encoding="utf-8") as jf:
                jf.seek(start_byte)
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

    # 无 token 输出时不生成空 bin，避免下游 Dataset 长度为 0
    if os.path.getsize(tmp_bin_file) == 0:
        os.remove(tmp_bin_file)
        return None

    shutil.move(tmp_bin_file, bin_file)
    return bin_file


def prepare_tokenized_data(
    dataset_dir: str,
    dataset_name: str,
    tokenizer_dir: str,
    data_dir: str | None = None,
    jsonl_shard_bytes: int = DEFAULT_JSONL_SHARD_BYTES,
    num_workers: int | None = None,
) -> list[str]:
    """
    将 parquet/jsonl tokenize 为 .bin。

    - Parquet：每个 row group 对应一个 ``{base}__rg{idx}.bin``（与 CPU 并行度无关）。
    - JSONL：按 ``jsonl_shard_bytes`` 字节窗口顺序切分为 ``{base}__j{idx}.bin``（与 CPU 并行度无关）。
    - ``num_workers`` 仅影响预处理速度，不改变输出分片数量与命名。
    """
    if jsonl_shard_bytes < 1024:
        raise ValueError("jsonl_shard_bytes 过小，请至少设为 1024 以上。")

    max_workers = num_workers if num_workers is not None else max(1, os.cpu_count() or 1)

    os.makedirs(dataset_dir, exist_ok=True)
    bin_cache_dir = resolve_bin_cache_dir(tokenizer_dir, data_dir, dataset_dir)
    os.makedirs(bin_cache_dir, exist_ok=True)

    all_files: list[str] = []

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
                print("🌐 正在拉取 FineWeb-Edu (sample)...")
                eng_stream = load_dataset(
                    "HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True
                )
                mixed_dataset = Dataset.from_list(list(eng_stream.take(10000))).select_columns(["text"])
                mixed_dataset.to_parquet(single_parquet)
            else:
                raise NotImplementedError(f"数据集 {dataset_name} 尚未实现")
        all_files = [single_parquet]

    tasks = _build_tasks(all_files, bin_cache_dir, tokenizer_dir, data_dir, dataset_dir, jsonl_shard_bytes)
    pending = [t for t in tasks if not os.path.exists(t["bin_file"])]

    if not pending:
        print("✅ 所有分片已存在，跳过写入。")
    else:
        print(
            f"🚀 待处理 {len(pending)} 个分片（共规划 {len(tasks)} 个输出文件），"
            f"进程池 max_workers={max_workers}。"
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_data_chunk, task) for task in pending]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="tokenize → .bin"):
                pass
        print("✅ 分片处理完毕。")

    return list_tokenized_bin_paths(
        tokenizer_dir=tokenizer_dir,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        data_dir=data_dir,
    )


def main(
    tokenizer_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    dataset_name: str = "debug",
    dataset_dir: str = "data",
    data_dir: str = "data",
    jsonl_shard_bytes: int = DEFAULT_JSONL_SHARD_BYTES,
    num_workers: int | None = None,
):
    """离线预处理入口：python data.py --tokenizer_dir ..."""
    paths = prepare_tokenized_data(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        tokenizer_dir=tokenizer_dir,
        data_dir=data_dir,
        jsonl_shard_bytes=jsonl_shard_bytes,
        num_workers=num_workers,
    )
    print(f"✅ 预处理完成，共 {len(paths)} 个 .bin: {paths[:3]}{'...' if len(paths) > 3 else ''}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
