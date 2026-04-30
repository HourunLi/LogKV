#!/usr/bin/env python3
"""
从 eval.py 写出的 lm-eval JSON dump 里，按任务名打印若干条样本的「输入 / 输出」侧信息。

eval.py 保存格式:
  { "benchmark": ..., "checkpoint_dir": ..., "results": <simple_evaluate 返回值> }

simple_evaluate 在 log_samples=True 时会在返回值里带 "samples": { task: [ {...}, ... ] }。
若 dump 里没有 samples，本脚本会提示需在评测时打开 log_samples。

用法:
  python unused/print_lmeval_dump_samples.py /path/to/eval_results_xxx.json longbench_2wikimqa
  python unused/print_lmeval_dump_samples.py /path/to/dump.json hellaswag --limit 2 --max-chars 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"根对象应为 dict，实际为 {type(data)}")
    return data


def _unwrap_harness_payload(root: dict[str, Any]) -> dict[str, Any]:
    """兼容 eval.py 外层包装，得到 simple_evaluate 那一层 dict。"""
    if "samples" in root and isinstance(root.get("samples"), dict):
        return root
    wrapped = root.get("results")
    if isinstance(wrapped, dict):
        if "samples" in wrapped or "results" in wrapped:
            return wrapped
    return root


def _resolve_task_key(samples: dict[str, Any], name: str) -> str:
    if name in samples:
        return name
    name_l = name.lower()
    candidates = [k for k in samples if name_l in k.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise KeyError(
            f"任务名 {name!r} 不唯一，可能匹配: {candidates}。请改用完整 task key。"
        )
    available = sorted(samples.keys())
    raise KeyError(f"未找到任务 {name!r}。已有 samples 的 task: {available}")


def _short(s: Any, max_chars: int) -> str:
    t = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, indent=2)
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    return t[: max_chars // 2] + "\n... [截断] ...\n" + t[-(max_chars // 2) :]


def _print_sample(idx: int, row: dict[str, Any], max_chars: int) -> None:
    print(f"\n{'=' * 72}\n# 样例 index={idx}  doc_id={row.get('doc_id', '?')}\n{'=' * 72}")

    # 常见字段：lm-eval 不同版本 key 略有差异，尽量都打出来
    for key in (
        "doc_id",
        "doc",
        "arguments",
        "input",
        "target",
        "targets",
        "resps",
        "filtered_resps",
        "exact_match",
        "acc",
    ):
        if key not in row:
            continue
        val = row[key]
        body = _short(val, max_chars)
        print(f"\n--- {key} ---\n{body}")

    printed = {
        "doc_id",
        "doc",
        "arguments",
        "input",
        "target",
        "targets",
        "resps",
        "filtered_resps",
        "exact_match",
        "acc",
    }
    extra = {k: v for k, v in row.items() if k not in printed}
    if extra:
        print(f"\n--- 其它字段 keys ---\n{list(extra.keys())}")
        print(_short(extra, max_chars))


def main() -> None:
    p = argparse.ArgumentParser(description="从 lm-eval JSON dump 打印指定任务的输入/输出样例")
    p.add_argument("json_path", type=Path, help="eval.py 导出的 .json 路径")
    p.add_argument("task", type=str, help="任务名（与 samples 里 key 一致，或为其子串唯一匹配）")
    p.add_argument("--limit", type=int, default=3, help="最多打印几条样本（默认 3）")
    p.add_argument("--max-chars", type=int, default=8000, help="每个大字段最多字符数（默认 8000，0 表示不截断）")
    args = p.parse_args()

    root = _load_json(args.json_path)
    payload = _unwrap_harness_payload(root)

    if "benchmark" in root:
        print(f"benchmark: {root.get('benchmark')}")
    if "checkpoint_dir" in root:
        print(f"checkpoint_dir: {root.get('checkpoint_dir')}")

    samples_root = payload.get("samples")
    if not isinstance(samples_root, dict) or not samples_root:
        print(
            "\n⚠️ 该 JSON 中没有可用的 results['samples']。"
            "eval.py 调用 simple_evaluate 时需传入 log_samples=True（或通过 metadata 打开），"
            "否则 dump 里只有聚合指标、没有逐条输入输出。\n"
        )
        print("当前 payload 顶层 keys:", sorted(payload.keys()))
        return

    task_key = _resolve_task_key(samples_root, args.task)
    rows = samples_root[task_key]
    if not isinstance(rows, list):
        print(f"任务 {task_key!r} 的 samples 不是 list: {type(rows)}")
        return

    print(f"\n选用 task key: {task_key!r}，共 {len(rows)} 条，展示前 {min(args.limit, len(rows))} 条\n")

    for i, row in enumerate(rows[: args.limit]):
        if not isinstance(row, dict):
            print(f"\n[样例 {i}] 非 dict，类型={type(row)}: {row!r}")
            continue
        _print_sample(i, row, args.max_chars)


if __name__ == "__main__":
    main()
