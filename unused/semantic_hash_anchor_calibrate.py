"""Cluster a Stage-0 pre-RoPE k_raw dump into fixed hash-routing anchors.

Consumes a Stage-0 dump (see ``unused/semantic_stage0_dump.py``) and, per
``(layer, KV group)``, runs the same DP-means-with-Ward-merge-when-full
routing the production semantic router uses
(``litgpt.semantic_s0.route_dpmeans_segments``, same ``lambda_new``/``k_max``
semantics as ``algorithm-spec.md`` §5.3/§5.6) over the pooled ``k_raw``
sample, then takes the resulting cluster centroids as the K_max fixed
anchors. A token that would legitimately open its own cluster online has a
real anchor waiting for it, instead of landing wherever a content-blind hash
would put it.

Usage::

    # 1. Dump pre-RoPE k_raw for every layer hash routing will run on (the
    #    stage0 tool's own --layers default is a sparse smoke-test subset --
    #    override it here, since every layer needs its own anchors):
    python unused/semantic_stage0_dump.py \\
      --checkpoint_dir <ckpt> --samples <jsonl> --output_dir <dump_dir> \\
      --layers 0,1,2,...,<n_layer-1>

    # 2. Cluster the dump into anchors:
    python unused/semantic_hash_anchor_calibrate.py \\
      --dump <dump_dir>/manifest.json --output anchors.pt \\
      --n_layer 28 --k_max 16 --lambda_rel 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from litgpt.semantic_s0 import load_manifest, manifest_base_dir, route_dpmeans_segments  # noqa: E402


def _iter_records(manifest: dict[str, Any], *, layer_filter: set[int] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample in manifest.get("samples", []):
        for record in sample.get("layers", []):
            layer = int(record["layer"])
            if layer_filter is not None and layer not in layer_filter:
                continue
            out.append(record)
    return out


def _pooled_k_raw(base_dir: Path, records: list[dict[str, Any]], layer: int) -> np.ndarray:
    """Concatenate every sample's k_raw for one layer into a single (G, T, D) array."""
    chunks = [
        np.load(base_dir / record["path"])["k_raw"].astype(np.float32)
        for record in records
        if int(record["layer"]) == layer
    ]
    if not chunks:
        raise ValueError(f"no k_raw records found for layer {layer}")
    groups = {c.shape[0] for c in chunks}
    if len(groups) != 1:
        raise ValueError(f"layer {layer}: ragged KV-group count across samples: {sorted(groups)}")
    return np.concatenate(chunks, axis=1)  # (G, T_total, D)


def calibrate_layer_group(
    k_gtd: np.ndarray, group: int, *, k_max: int, lambda_rel: float, s_h: float, g_max: float
) -> np.ndarray:
    """Return (k_max, D) anchors for one (layer, group)."""
    k = k_gtd[group]  # (T, D)
    result = route_dpmeans_segments(k, lambda_new=lambda_rel * max(s_h, 1e-12), g_max=g_max, k_max=k_max)
    if result.cluster_count == 0:
        raise ValueError(f"group {group}: calibration sample produced zero clusters (empty dump?)")
    dim = k.shape[-1]
    centroids = np.zeros((result.cluster_count, dim), dtype=np.float32)
    for c in range(result.cluster_count):
        centroids[c] = k[result.cluster_ids == c].mean(axis=0)
    if result.cluster_count < k_max:
        # ponytail: calibration sample didn't reach k_max distinct clusters
        # (sample too small, or lambda_rel too loose) -- cycle through the
        # ones found rather than leaving slots undefined. Widen the sample or
        # lower lambda_rel if this fires; it's a calibration-coverage gap,
        # not a routing bug.
        centroids = centroids[np.arange(k_max) % result.cluster_count]
    return centroids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True, help="Stage-0 dump directory or manifest.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n_layer", type=int, required=True, help="Model n_layer (anchors need every layer covered)")
    parser.add_argument("--k_max", type=int, required=True, help="Must match log_kv_cluster_k_max")
    parser.add_argument("--lambda_rel", type=float, default=1.0, help="Must match log_kv_cluster_lambda_rel")
    parser.add_argument("--g_max", type=float, default=float("inf"))
    args = parser.parse_args()

    manifest = load_manifest(args.dump)
    base_dir = manifest_base_dir(args.dump)
    records = _iter_records(manifest, layer_filter=set(range(args.n_layer)))
    found_layers = sorted({int(r["layer"]) for r in records})
    missing = sorted(set(range(args.n_layer)) - set(found_layers))
    if missing:
        raise ValueError(
            f"dump is missing layers {missing} -- re-run unused/semantic_stage0_dump.py with "
            f"--layers covering every layer 0..{args.n_layer - 1}, hash routing needs anchors "
            f"for each one"
        )

    per_layer: dict[int, np.ndarray] = {}
    n_groups: int | None = None
    dim: int | None = None
    for layer in found_layers:
        k_gtd = _pooled_k_raw(base_dir, records, layer)  # (G, T, D)
        n_groups = k_gtd.shape[0] if n_groups is None else n_groups
        dim = k_gtd.shape[-1] if dim is None else dim
        if k_gtd.shape[0] != n_groups or k_gtd.shape[-1] != dim:
            raise ValueError(f"layer {layer}: shape {k_gtd.shape} inconsistent with earlier layers (G={n_groups}, D={dim})")
        per_layer[layer] = np.stack(
            [
                calibrate_layer_group(
                    k_gtd,
                    group,
                    k_max=args.k_max,
                    lambda_rel=args.lambda_rel,
                    s_h=float(manifest["key_scale"][str(layer)]["s_h"][group]),
                    g_max=args.g_max,
                )
                for group in range(n_groups)
            ],
            axis=0,
        )  # (G, k_max, D)

    anchors = np.stack([per_layer[layer] for layer in range(args.n_layer)], axis=0)  # (n_layer, G, k_max, D)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(anchors), args.output)
    print(f"wrote anchors {tuple(anchors.shape)} -> {args.output}")


if __name__ == "__main__":
    main()
