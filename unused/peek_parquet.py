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


def _read_table_head(path: str, max_rows: int):
    """读前 max_rows 行，兼容旧版 pyarrow（无 read_table(max_rows=)）；不整文件扫描。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    parts: list = []
    total = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        parts.append(t)
        total += t.num_rows
        if total >= max_rows:
            break
    if not parts:
        return pa.table({})
    table = pa.concat_tables(parts) if len(parts) > 1 else parts[0]
    if table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    return table


def main() -> None:
    import pyarrow.parquet as pq

    path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    pf = pq.ParquetFile(path)
    print("📂", path)
    print("📋 Schema:\n", pf.schema_arrow)
    print("📊 num_rows:", pf.metadata.num_rows, " num_row_groups:", pf.num_row_groups)

    table = _read_table_head(path, NUM_ROWS)
    names = table.column_names
    print(f"\n👉 前 {n} 行（列: {names}）:\n")
    for i in range(table.num_rows):
        print(f"--- row {i} ---")
        for name in names:
            v = table.column(name)[i].as_py()
            print(f"  [{name}]: {_preview(v, PREVIEW_LEN)}")


if __name__ == "__main__":
    main()
