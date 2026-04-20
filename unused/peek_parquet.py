"""
Peek Parquet：打印 schema、行数，以及前 N 行各列内容预览（与 peek_jsonl.py 用法类似）。

依赖：pyarrow、pandas（与项目 data.py 一致；需已安装）。

用法：改下面 file_path / NUM_ROWS，然后
  python unused/peek_parquet.py
"""
from __future__ import annotations

import json
import os

# 替换成你的 parquet 路径
file_path = os.path.join(os.path.dirname(__file__), "..", "data", "debug.parquet")

NUM_ROWS = 10
PREVIEW_LEN = 1000


def _preview(val: object, max_len: int) -> str:
    if val is None:
        return "None"
    if isinstance(val, (dict, list)):
        try:
            s = json.dumps(val, ensure_ascii=False)
        except TypeError:
            s = str(val)
    else:
        s = str(val)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def main() -> None:
    import pyarrow.parquet as pq

    path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    pf = pq.ParquetFile(path)
    print("📂", path)
    print("📋 Schema:\n", pf.schema_arrow)
    print("📊 num_rows:", pf.metadata.num_rows, " num_row_groups:", pf.num_row_groups)

    table = pq.read_table(path, max_rows=NUM_ROWS)
    names = table.column_names
    print(f"\n👉 前 {min(NUM_ROWS, table.num_rows)} 行（列: {names}）:\n")
    for i in range(table.num_rows):
        print(f"--- row {i} ---")
        for name in names:
            v = table.column(name)[i].as_py()
            print(f"  [{name}]: {_preview(v, PREVIEW_LEN)}")


if __name__ == "__main__":
    main()
