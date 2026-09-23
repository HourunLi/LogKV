#!/usr/bin/env python
"""Calibrate SinkWindow's window_size W against a target persistent-cache
byte budget (task spec §5.2): find the largest integer W whose
SinkWindowKVCache structural byte count does not exceed the SemanticLogKV
config's own structural byte count, for a fixed sink_size (default 4, per
the task spec's §4.1 recommendation).

Needs no real weights and no GPU/CUDA -- both the target (LogKV) and
candidate (SinkWindow) caches are built on torch.device("meta") purely for
their shapes/dtypes (same technique as scripts/measure_cache_bytes.py).
cache_bytes() is monotonically non-decreasing in window_size (window_k/v
scale linearly with it, nothing else does), so binary search is valid.

Usage:
    python scripts/calibrate_sinkwindow_w.py \\
        --budget_config exp/qwen1.7b-32k/semantic_stage1_frozen.yaml \\
        --context_length 32768 --sink_size 4 \\
        --output sinkwindow_w_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from litgpt import Config  # noqa: E402
from litgpt.model import GPT  # noqa: E402
from litgpt.cache_accounting import cache_bytes  # noqa: E402
from scripts.measure_cache_bytes import format_bytes, resolve_log_kv_kwargs  # noqa: E402


def sink_window_structural_bytes(
    arch_name: str, context_length: int, batch_size: int, sink_size: int, window_size: int, dtype: torch.dtype,
) -> int:
    config_obj = Config.from_name(arch_name)
    config_obj.block_size = context_length
    with torch.device("meta"):
        model = GPT(config_obj)
    model.set_sink_window_cache(
        batch_size=batch_size, sink_size=sink_size, window_size=window_size, device="meta", dtype=dtype,
    )
    return cache_bytes(model)["structural_bytes"]


def calibrate_window_size(
    *,
    target_bytes: int,
    arch_name: str,
    context_length: int,
    batch_size: int,
    sink_size: int,
    dtype: torch.dtype,
) -> int:
    """Largest window_size (>= 1) with structural bytes <= target_bytes.
    Raises if even window_size=1 (sink alone plus one window slot) is
    already over budget -- that's a real infeasibility, not a search bug.
    """
    lo, hi = 1, max(1, context_length)
    lo_bytes = sink_window_structural_bytes(arch_name, context_length, batch_size, sink_size, lo, dtype)
    if lo_bytes > target_bytes:
        raise ValueError(
            f"Even the minimum window_size={lo} costs {lo_bytes} bytes ({format_bytes(lo_bytes)}), already "
            f"over the target budget {target_bytes} ({format_bytes(target_bytes)}). sink_size={sink_size} "
            "alone may already exceed the budget -- lower sink_size, or the SemanticLogKV config's own "
            "budget is too small to match at all with this sink_size."
        )
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        b = sink_window_structural_bytes(arch_name, context_length, batch_size, sink_size, mid, dtype)
        if b <= target_bytes:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget_config", required=True, help="YAML whose LogKV structural bytes define the target budget")
    parser.add_argument("--context_length", type=int, default=32768)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sink_size", type=int, default=4, help="Task spec §4.1's recommended default")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--align",
        type=int,
        default=1,
        help="Round the final window_size down to a multiple of this, if some algorithm/kernel requires "
        "a particular granularity. Default 1 (no rounding).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.align < 1:
        raise ValueError(f"--align must be >= 1, got {args.align}")

    cfg, log_kv_kwargs = resolve_log_kv_kwargs(args.budget_config)
    arch_name = cfg.get("arch_name", "Qwen/Qwen3-0.6B-Base")
    dtype = getattr(torch, args.dtype)

    target_config = Config.from_name(arch_name)
    target_config.block_size = args.context_length
    with torch.device("meta"):
        target_model = GPT(target_config)
    target_model.set_log_kv_cache(
        batch_size=args.batch_size, max_seq_length=args.context_length, device="meta", dtype=dtype, **log_kv_kwargs,
    )
    target_bytes = cache_bytes(target_model)["structural_bytes"]

    window_size_unaligned = calibrate_window_size(
        target_bytes=target_bytes,
        arch_name=arch_name,
        context_length=args.context_length,
        batch_size=args.batch_size,
        sink_size=args.sink_size,
        dtype=dtype,
    )
    window_size = max(1, (window_size_unaligned // args.align) * args.align)
    window_size_bytes = sink_window_structural_bytes(
        arch_name, args.context_length, args.batch_size, args.sink_size, window_size, dtype,
    )
    remaining_fraction = 1.0 - (window_size_bytes / target_bytes) if target_bytes else 0.0

    report = {
        "budget_config": args.budget_config,
        "arch_name": arch_name,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "target_structural_bytes": target_bytes,
        "target_structural_bytes_human": format_bytes(target_bytes),
        "sink_size": args.sink_size,
        "align": args.align,
        "window_size_unaligned": window_size_unaligned,
        "window_size": window_size,
        "window_size_bytes": window_size_bytes,
        "window_size_bytes_human": format_bytes(window_size_bytes),
        "budget_remaining_fraction": remaining_fraction,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[calibrate] wrote {args.output}")

    if args.align > 1 and window_size_unaligned != window_size:
        print(
            f"[calibrate] window_size rounded down from {window_size_unaligned} to {window_size} for "
            f"--align={args.align}; {remaining_fraction:.4%} of the budget is left unused as a result -- "
            "report this in the final budget table (task spec §5.2)."
        )


if __name__ == "__main__":
    main()
