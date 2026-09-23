#!/usr/bin/env python
"""Measure persistent LogKV cache bytes for a training/eval YAML config.

Builds the model on ``torch.device("meta")`` (the same pattern
``litgpt/scripts/validate.py`` and ``tests/test_model.py`` already use to get
correctly-shaped-and-typed ``GPT.cos``/``GPT.sin`` without materializing real
weights -- see ``tests/test_model.py``'s
``assert model.cos.device.type == "meta"``), so this needs no real
checkpoint, no GPU, and typically no more than a second or two even at
``context_length=32768``. It reports exactly the two numbers
``litgpt/cache_accounting.py`` defines: ``structural_bytes`` (config-only,
what budget matching / SinkWindow W-calibration must use) and
``live_extra_bytes`` (transient decode-timing-dependent workspace,
informational only).

Usage:
    python scripts/measure_cache_bytes.py exp/qwen1.7b-32k/semantic_stage1_frozen.yaml
    python scripts/measure_cache_bytes.py exp/qwen1.7b-32k/base.yaml --per_buffer
    python scripts/measure_cache_bytes.py exp/qwen1.7b-32k/base.yaml --context_length 32768 --batch_size 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from litgpt import Config  # noqa: E402
from litgpt.model import GPT  # noqa: E402
from litgpt.cache_accounting import assert_no_training_only_state, cache_bytes  # noqa: E402
from scripts.yaml_resolve import resolve_yaml  # noqa: E402

# demo.py YAML field name -> GPT.set_log_kv_cache kwarg name (litgpt/model.py:358-391).
# Only the fields set_log_kv_cache actually consumes; the rest of demo.py's
# ~90 main() params (batch size, LR, data, ...) are irrelevant to cache shape.
_YAML_TO_SET_CACHE_KWARG = {
    "log_kv_B": "B",
    "log_kv_recent_size": "recent_size",
    "log_kv_prefill_block": "prefill_block",
    "log_kv_second_order_scale": "second_order_scale",
    "log_kv_importance_pooling": "importance_pooling",
    "log_kv_importance_pooling_lambda": "importance_pooling_lambda",
    "log_kv_importance_pooling_temperature": "importance_pooling_temperature",
    "log_kv_semantic_clusters": "semantic_clusters",
    "log_kv_cluster_k_max": "cluster_k_max",
    "log_kv_cluster_lambda_rel": "cluster_lambda_rel",
    "log_kv_seg_eta": "seg_eta",
    "log_kv_seg_g0": "seg_g0",
    "log_kv_seg_gap_max": "seg_gap_max",
    "log_kv_seg_block_level": "seg_block_level",
    "log_kv_seg_forget": "seg_forget",
    "log_kv_semantic_flush_granularity": "semantic_flush_granularity",
    "log_kv_semantic_cluster_chunk_size": "semantic_cluster_chunk_size",
    "log_kv_semantic_capacity_beta": "semantic_capacity_beta",
    "log_kv_semantic_capacity_hard_cap_mult": "semantic_capacity_hard_cap_mult",
    "log_kv_semantic_legacy_route": "semantic_legacy_route",
    "log_kv_semantic_anchor_mode": "semantic_anchor_mode",
    "log_kv_semantic_pack_backend": "semantic_pack_backend",
    "log_kv_semantic_centroid_backend": "semantic_centroid_backend",
    "log_kv_semantic_summary_size": "semantic_summary_size",
    "log_kv_semantic_replay_updates": "semantic_replay_updates",
}

# GPT.set_log_kv_cache's own defaults (litgpt/model.py:358-391), used for any
# field absent from both the YAML and its `config:` base chain.
_SET_CACHE_DEFAULTS = {
    "B": 512,
    "recent_size": 1024,
    "prefill_block": 256,
    "second_order_scale": 1.0,
    "importance_pooling": False,
    "importance_pooling_lambda": 1.0,
    "importance_pooling_temperature": 1.0,
    "semantic_clusters": False,
    "cluster_k_max": 1,
    "cluster_lambda_rel": 1.0,
    "seg_eta": 1.0,
    "seg_g0": 2048.0,
    "seg_gap_max": None,
    "seg_block_level": 0,
    "seg_forget": 0.5,
    "semantic_flush_granularity": 2,
    "semantic_cluster_chunk_size": 0,
    "semantic_capacity_beta": 0.0,
    "semantic_capacity_hard_cap_mult": 0.0,
    "semantic_legacy_route": False,
    "semantic_anchor_mode": "multi",
    "semantic_pack_backend": "auto",
    "semantic_centroid_backend": "sequential",
    "semantic_summary_size": 1,
    "semantic_replay_updates": False,
}


def resolve_log_kv_kwargs(yaml_path: str) -> tuple[dict, dict]:
    cfg = resolve_yaml(yaml_path)
    kwargs = dict(_SET_CACHE_DEFAULTS)
    for yaml_key, kwarg in _YAML_TO_SET_CACHE_KWARG.items():
        if cfg.get(yaml_key) is not None:
            kwargs[kwarg] = cfg[yaml_key]
    return cfg, kwargs


def format_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024.0:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} PiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Training/eval YAML path (config: chain resolved like demo.py does)")
    parser.add_argument("--context_length", type=int, default=None, help="Overrides the YAML's context_length")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--per_buffer", action="store_true", help="Print the full per-buffer breakdown")
    args = parser.parse_args()

    cfg, log_kv_kwargs = resolve_log_kv_kwargs(args.config)
    arch_name = cfg.get("arch_name", "Qwen/Qwen3-0.6B-Base")
    context_length = args.context_length or int(cfg.get("context_length") or cfg.get("max_seq_length") or 32768)

    config_obj = Config.from_name(arch_name)
    config_obj.block_size = context_length

    with torch.device("meta"):
        model = GPT(config_obj)

    dtype = getattr(torch, args.dtype)
    model.set_log_kv_cache(
        batch_size=args.batch_size,
        max_seq_length=context_length,
        device="meta",
        dtype=dtype,
        **log_kv_kwargs,
    )

    for block in model.transformer.h:
        assert_no_training_only_state(block.attn.kv_cache)

    result = cache_bytes(model, per_buffer=args.per_buffer)

    print(f"config: {args.config}")
    print(
        f"arch_name: {arch_name}  context_length: {context_length}  "
        f"batch_size: {args.batch_size}  dtype: {args.dtype}"
    )
    print(
        f"semantic_clusters: {log_kv_kwargs['semantic_clusters']}  "
        f"cluster_k_max: {log_kv_kwargs['cluster_k_max']}  B: {log_kv_kwargs['B']}  "
        f"recent_size: {log_kv_kwargs['recent_size']}  "
        f"anchor_mode: {log_kv_kwargs['semantic_anchor_mode']}"
    )
    print()
    print(
        "structural_bytes (persistent, config-only -- use this for "
        f"W-calibration/budget matching): {result['structural_bytes']} "
        f"({format_bytes(result['structural_bytes'])})"
    )
    print(
        "live_extra_bytes (transient, decode-timing-dependent -- "
        f"informational only, not part of the frozen budget): {result['live_extra_bytes']} "
        f"({format_bytes(result['live_extra_bytes'])})"
    )
    print(f"total_bytes: {result['total_bytes']} ({format_bytes(result['total_bytes'])})")

    if args.per_buffer:
        print("\nper-buffer breakdown (structural):")
        for entry in result["structural_breakdown"]:
            flag = " [deduped: shares storage with an earlier buffer]" if entry["deduped"] else ""
            print(
                f"  {entry['name']:40s} shape={str(entry['shape']):28s} "
                f"dtype={entry['dtype']:15s} bytes={entry['bytes']}{flag}"
            )


if __name__ == "__main__":
    main()
