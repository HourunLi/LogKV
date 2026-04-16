"""
parquet / jsonl → LitData 流式 chunks（litdata.optimize）。

缓存目录名含 tokenizer 内容哈希，与模型权重无关；训练侧用 StreamingDataset + TokensLoader 读取。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset

from litgpt.tokenizer import Tokenizer

DEFAULT_CHUNK_BYTES = "200MB"


def tokenizer_cache_key(tokenizer_dir: str) -> str:
    """tokenizer 文件内容哈希短 id。"""
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


def litdata_chunks_dir(
    tokenizer_dir: str, data_dir: str | None, dataset_dir: str, context_length: int
) -> str:
    """LitData optimize 输出目录（与训练共用；含 context_length 避免块大小不一致）。"""
    key = tokenizer_cache_key(tokenizer_dir)
    root = data_dir if data_dir else dataset_dir
    return os.path.join(root, f"litdata_{key}_ctx{context_length}")


def tokenize_source_file(path: str, tokenizer: Tokenizer):
    """
    供 litdata.optimize 调用：每个样本一条序列（与原先 CPT 一致：bos=False, eos=True）。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["text"])
            for text in table.to_pandas()["text"]:
                if pd.isna(text):
                    continue
                text = str(text).strip()
                if not text:
                    continue
                yield tokenizer.encode(text, bos=False, eos=True)
    elif ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", obj.get("content", ""))
                    if not text:
                        continue
                    yield tokenizer.encode(text, bos=False, eos=True)
                except json.JSONDecodeError:
                    continue
    else:
        raise ValueError(f"不支持的文件类型: {path}")


def _validate_tokenizer(tokenizer: Tokenizer | None) -> None:
    if tokenizer is None:
        raise ValueError("Tokenizer 为 None，请传入有效的 tokenizer_dir。")


def prepare_tokenized_data(
    dataset_dir: str,
    dataset_name: str,
    tokenizer_dir: str,
    data_dir: str | None = None,
    context_length: int = 4096,
    chunk_bytes: str = DEFAULT_CHUNK_BYTES,
    num_workers: int | None = None,
) -> str:
    """
    使用 litdata.optimize 将数据编译为流式 chunks。

    ``context_length`` 为训练序列长度；内部 ``TokensLoader(block_size=context_length + 1)`` 与 LitGPT 一致。
    返回 optimize 输出目录路径。
    """
    from litdata import optimize
    from litdata.streaming import TokensLoader

    os.makedirs(dataset_dir, exist_ok=True)
    out_dir = litdata_chunks_dir(tokenizer_dir, data_dir, dataset_dir, context_length)
    block_size = context_length + 1

    all_files: list[str] = []
    if data_dir and os.path.isdir(data_dir):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet"))) + sorted(
            glob.glob(os.path.join(data_dir, "*.jsonl"))
        )
        if not all_files:
            raise FileNotFoundError(f"在 {data_dir} 下没有找到 .parquet 或 .jsonl。")
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

    tokenizer = Tokenizer(tokenizer_dir)
    _validate_tokenizer(tokenizer)

    max_workers = num_workers if num_workers is not None else max(1, (os.cpu_count() or 1) - 1)
    use_workers = min(max_workers, len(all_files)) if all_files else 1

    if os.path.isdir(out_dir) and any(os.scandir(out_dir)):
        print(
            f"⚠️ 已存在 LitData 缓存: {out_dir}\n"
            "若需更换语料或 tokenizer，请删除该目录后重跑。"
        )
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    print(f"🚀 litdata.optimize → {out_dir}（chunk_bytes={chunk_bytes}, workers={use_workers}）")

    optimize(
        fn=partial(tokenize_source_file, tokenizer=tokenizer),
        inputs=all_files,
        output_dir=out_dir,
        num_workers=use_workers,
        chunk_bytes=chunk_bytes,
        item_loader=TokensLoader(block_size=block_size),
    )
    print("✅ optimize 完成。")
    return out_dir


def main(
    tokenizer_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    dataset_name: str = "debug",
    dataset_dir: str = "data",
    data_dir: str = "data",
    context_length: int = 4096,
    chunk_bytes: str = DEFAULT_CHUNK_BYTES,
    num_workers: int | None = None,
):
    """离线预处理：python data.py --tokenizer_dir ..."""
    out = prepare_tokenized_data(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        tokenizer_dir=tokenizer_dir,
        data_dir=data_dir,
        context_length=context_length,
        chunk_bytes=chunk_bytes,
        num_workers=num_workers,
    )
    print(f"✅ 输出目录: {out}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
