#!/usr/bin/env python
"""Dump pre-RoPE k/v tensors for SemanticLogKV Stage-0 analyses.

Scope: this implements *mechanism A only* (the cheap post-norm_q/norm_k,
pre-apply_rope hook -- see docs/experiments.md's "两套机制" section), never
mechanism B (post-RoPE q, ``attn_mass_by_dist``) or the manifest fields that
depends on (``tail_query_count``, the MinHash triple ``hash_algorithm``/``k``/
``master_seed``) -- those are required for S0.8's 3b sub-item and for
`CLAUDE.md` S13.2's attention-quality-by-distance curve, and are not yet
implemented anywhere in this repo.

"This dump provides data mechanism A can supply" and "there is an analyzer
that turns that data into a given S0.x number" are two different claims --
do not conflate them. This dump's k/v is the only input S0.0/S0.3/S0.4/S0.5/
S0.6/S0.7 and S0.2's calibration-① and S0.8's k/v-dependent sub-items ever
need, but that is a statement about *what data those items depend on*, not
about *what has an analyzer already*. As of this writing, `litgpt/
semantic_s0.py`'s `SweepAccumulator`/`route_dpmeans_segments` only actually
compute S0.0, S0.4, S0.5, and S0.2's calibration-①; S0.3 (needle isolation
rate), S0.6 (anchor-dedup E[M]), and S0.7 (supersession) still need their own
analysis code written on top of this dump's output, and S0.8 cannot run at
all yet regardless of dump data (it needs the batch-approximate routing side
of that comparison, which has no implementation anywhere). S0.1 does not use
this dump at all -- it is a pure-CPU unit test of `log_kv_position.py` (not
yet written). Do not read this script's existence as "Stage 0 dump is done";
it covers the k/v-only slice of Stage 0, and only a subset of that slice has
an analyzer to go with it.

Example:
    python unused/semantic_stage0_dump.py \
      --checkpoint_dir ckpt/qwen1.7b-32k-warmup \
      --samples niah_prompts.jsonl \
      --output_dir stage0_dump \
      --layers 0,7,14,21,27 \
      --max_samples 4
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from litgpt.config import Config  # noqa: E402
from litgpt.log_kv_pin_diag import find_needle_spans  # noqa: E402
from litgpt.model import GPT, CausalSelfAttention  # noqa: E402
from litgpt.semantic_s0 import RunningKeyScale, Stage0DumpRecorder, parse_int_list  # noqa: E402
from litgpt.tokenizer import Tokenizer  # noqa: E402


_CONFIG_FIELDS = {f.name for f in dataclasses.fields(Config)}


def _filter_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k in _CONFIG_FIELDS}


def _config_from_checkpoint(checkpoint_dir: Path, overrides: dict[str, Any] | None) -> Config:
    config_path = checkpoint_dir / "model_config.yaml"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data.update(overrides or {})
        return Config(**_filter_config_dict(data))
    kwargs = {"name": checkpoint_dir.name}
    kwargs.update(overrides or {})
    return Config.from_name(**_filter_config_dict(kwargs))


def _load_checkpoint(checkpoint_dir: Path, device: str) -> dict[str, Any]:
    path = checkpoint_dir / "lit_model.pth"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def _resolve_tokenizer_dir(checkpoint_dir: Path, tokenizer_dir: str | None) -> Path:
    candidates: list[Path] = []
    if tokenizer_dir is not None:
        candidates.append(Path(tokenizer_dir).expanduser())
    candidates.append(checkpoint_dir)
    config_path = checkpoint_dir / "model_config.yaml"
    if config_path.is_file():
        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            hf = data.get("hf_config") or {}
            if hf.get("org") and hf.get("name"):
                candidates.append(Path("checkpoints") / str(hf["org"]) / str(hf["name"]))
        except Exception:
            pass
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file() or (candidate / "tokenizer.model").is_file():
            return candidate
    tried = "\n".join(f"  - {c.resolve()}" for c in candidates)
    raise FileNotFoundError(f"No tokenizer.json/tokenizer.model found. Searched:\n{tried}")


def _load_samples(path: Path | None, prompt: str | None, max_samples: int | None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]]
    if prompt is not None:
        samples = [{"prompt": prompt}]
    elif path is None:
        raise ValueError("Either --samples or --prompt is required")
    elif path.suffix.lower() == ".jsonl":
        samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            samples = payload
        elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            samples = payload["samples"]
        elif isinstance(payload, dict):
            samples = [payload]
        else:
            raise ValueError(f"Unsupported samples payload in {path}")
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    return samples


def _prompt_from_sample(sample: dict[str, Any]) -> str:
    for key in ("prompt", "input", "text", "context"):
        value = sample.get(key)
        if isinstance(value, str):
            return value
    args = sample.get("args")
    if isinstance(args, list) and args and isinstance(args[0], str):
        return args[0]
    raise KeyError("sample must contain one of: prompt/input/text/context, or args[0]")


def _dtype(name: str, device: str) -> torch.dtype:
    name = name.lower()
    if device == "cpu" and name in {"bf16", "bfloat16", "fp16", "float16"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _assert_layers_are_recordable(model: GPT, config: Config, layers: set[int]) -> None:
    """Fail fast, before any forward pass, instead of silently writing a
    manifest with empty/missing layers.

    Two ways ``--layers`` can silently under-record: a typo'd layer index
    that never matches any block (``block.attn.block_idx in layers`` is
    simply always False for it), or a model whose attention module isn't
    ``CausalSelfAttention`` (e.g. ``MultiheadLatentAttention``, which never
    got the ``_semantic_s0_recorder`` hook -- see model.py's
    ``CausalSelfAttention.forward`` -- so setting the attribute on it
    succeeds silently but nothing ever calls ``record_pre_rope``). Both cases
    otherwise only surface much later, as an unexplained gap in
    ``unused/semantic_s0_sweep.py``'s per-layer output.
    """
    n_layer = int(config.n_layer)
    out_of_range = {layer for layer in layers if not (0 <= layer < n_layer)}
    if out_of_range:
        raise ValueError(
            f"--layers contains indices outside [0, {n_layer}): {sorted(out_of_range)}. "
            f"This would silently record nothing for them."
        )
    unsupported = {
        block.attn.block_idx
        for block in model.transformer.h
        if block.attn.block_idx in layers and not isinstance(block.attn, CausalSelfAttention)
    }
    if unsupported:
        raise NotImplementedError(
            f"--layers {sorted(unsupported)} use an attention module that is not "
            f"CausalSelfAttention ({type(model.transformer.h[next(iter(unsupported))].attn).__name__}); "
            f"the Stage-0 recorder hook is only wired into CausalSelfAttention.forward, so these "
            f"layers would silently record nothing."
        )


def _assert_checkpoint_loaded_cleanly(load_result: Any, *, allow_mismatch: bool) -> None:
    """Fail fast on a partial checkpoint/config match instead of silently
    dumping data from a half-random model.

    ``model.load_state_dict(state, strict=False)`` never raises for missing
    or unexpected keys -- it just leaves missing ones at their random init
    value and drops unexpected ones -- so a wrong/stale checkpoint_dir or a
    config that has drifted from what the checkpoint was actually trained
    with produces a manifest that looks exactly like a normal successful
    dump (same schema, same shapes), just built from partially- or entirely-
    random weights. Everything downstream (unused/semantic_s0_sweep.py, S0.2,
    S0.3, ...) would silently analyze that. There's no known source of
    legitimate missing/unexpected keys for a well-matched checkpoint/config
    pair here: this script never calls set_kv_cache/set_log_kv_cache, so the
    only KV-cache-related buffers that could ever appear are registered with
    persistent=False and are therefore excluded from state_dict on both
    sides regardless. Pass ``allow_mismatch=True`` (``--allow_checkpoint_key_
    mismatch``) only after you've inspected the printed keys and are sure the
    mismatch is benign.
    """
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if not missing and not unexpected:
        return

    def _preview(keys: list[str], limit: int = 20) -> str:
        shown = ", ".join(keys[:limit])
        more = f", ... ({len(keys) - limit} more)" if len(keys) > limit else ""
        return f"[{shown}{more}]"

    message = (
        f"checkpoint/config mismatch: {len(missing)} missing key(s) "
        f"{_preview(missing)}, {len(unexpected)} unexpected key(s) {_preview(unexpected)}. "
        f"This checkpoint would load with some weights left at their random init value "
        f"and/or some checkpoint tensors silently ignored -- the resulting dump would look "
        f"like a normal successful S0.0 dump but come from a wrong/partially-random model. "
        f"Pass --allow_checkpoint_key_mismatch if you've verified this specific mismatch is benign."
    )
    if not allow_mismatch:
        raise RuntimeError(message)
    print(f"[stage0-dump] WARNING (--allow_checkpoint_key_mismatch set): {message}", flush=True)


def _span_payload(tokenizer: Tokenizer, prompt: str, sample: dict[str, Any], offset: int, used_len: int) -> list[dict]:
    spans = find_needle_spans(tokenizer, prompt, doc=sample)
    out = []
    for span in spans:
        d = span.to_dict()
        d["used_token_start"] = span.token_start - int(offset)
        d["used_token_end"] = span.token_end - int(offset)
        d["survived_left_truncation"] = d["used_token_end"] > 0 and d["used_token_start"] < used_len
        out.append(d)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tokenizer_dir")
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--layers", default="0,7,14,21,27")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--save_dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Defaults to float32 so downstream sweeps (unused/semantic_s0_sweep.py) route on "
        "the same precision the manifest's calibrated s_h/vh were computed from -- s_h/vh are "
        "always accumulated in fp32 from the pre-quantization tensor (litgpt/semantic_s0.py's "
        "RunningKeyScale.update()), so saving k/v at float16 lets DP-means routing distances "
        "near the lambda_new threshold pick up fp16 rounding noise that the threshold itself "
        "never saw, which can flip a cluster assignment right at the boundary. Pass float16 "
        "only for a quick smoke test where exact routing near the threshold does not matter; "
        "not for a real S0.0/S0.2/S0.3 run.",
    )
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--config_overrides", help="JSON dict merged into model_config.yaml")
    parser.add_argument("--store_prompt", action="store_true")
    parser.add_argument(
        "--allow_checkpoint_key_mismatch",
        action="store_true",
        help="Proceed even if model.load_state_dict finds missing/unexpected keys (leaving some "
        "weights at their random init value and/or silently dropping some checkpoint tensors) "
        "instead of hard-failing. Only use this after inspecting the printed key names and "
        "confirming the mismatch is benign -- a dump from a partially-random model looks just "
        "like a normal successful one downstream.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = set(parse_int_list(args.layers))
    if not layers:
        raise ValueError(
            f"--layers {args.layers!r} parsed to an empty set -- this would silently write a "
            f"manifest with dump.layers=[] and every sample's layers=[] (the hook is armed for no "
            f"block, so record_pre_rope never fires, and recorded_layers == layers == set() passes "
            f"the mismatch check below vacuously), not an obviously-wrong empty output"
        )
    overrides = json.loads(args.config_overrides) if args.config_overrides else None

    tokenizer_dir = _resolve_tokenizer_dir(checkpoint_dir, args.tokenizer_dir)
    tokenizer = Tokenizer(tokenizer_dir)
    config = _config_from_checkpoint(checkpoint_dir, overrides)
    model = GPT(config)
    if args.max_seq_length is not None:
        model.max_seq_length = min(int(args.max_seq_length), model.max_seq_length)
    dtype = _dtype(args.dtype, args.device)
    model = model.to(device=args.device, dtype=dtype)
    state = _load_checkpoint(checkpoint_dir, args.device)
    load_result = model.load_state_dict(state, strict=False)
    _assert_checkpoint_loaded_cleanly(load_result, allow_mismatch=args.allow_checkpoint_key_mismatch)
    model.eval()

    _assert_layers_are_recordable(model, config, layers)

    samples = _load_samples(args.samples, args.prompt, args.max_samples)
    if not samples:
        raise ValueError(
            f"0 samples to dump after loading {args.samples!r} and applying --max_samples="
            f"{args.max_samples!r} -- this would silently write a manifest with samples=[] and exit "
            f"0 instead of an obviously-wrong empty output. Pass --max_samples >= 1, or check "
            f"--samples actually contains rows."
        )
    key_scale = RunningKeyScale()
    # RunningKeyScale is a generic E||x-xbar||^2 accumulator (see its docstring) --
    # this second instance calibrates value-space scale, since normalizing value
    # SSE by the key-space s_h would not make it comparable across layers/groups.
    value_scale = RunningKeyScale()
    manifest_samples: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples):
        prompt = _prompt_from_sample(sample)
        sample_id = str(
            sample.get("sample_id")
            or sample.get("id")
            or sample.get("doc_id")
            or f"sample_{sample_index:04d}"
        )
        tokens = tokenizer.encode(prompt, device=torch.device(args.device)).long()
        original_len = int(tokens.numel())
        offset = 0
        if original_len > model.max_seq_length:
            offset = original_len - model.max_seq_length
            tokens = tokens[-model.max_seq_length :]
        used_len = int(tokens.numel())
        if used_len <= 0:
            raise ValueError(f"empty prompt after tokenization for sample {sample_id}")

        recorder = Stage0DumpRecorder(
            output_dir=output_dir,
            sample_index=sample_index,
            sample_id=sample_id,
            layers=layers,
            key_scale=key_scale,
            value_scale=value_scale,
            save_dtype=args.save_dtype,
            compressed=args.compressed,
        )
        for block in model.transformer.h:
            block.attn._semantic_s0_recorder = recorder if block.attn.block_idx in layers else None
        try:
            with torch.inference_mode():
                _ = model(tokens.view(1, -1), lm_head_start=max(0, used_len - 1))
        finally:
            for block in model.transformer.h:
                block.attn._semantic_s0_recorder = None

        recorded_layers = {r["layer"] for r in recorder.records}
        if recorded_layers != layers:
            missing = sorted(layers - recorded_layers)
            raise RuntimeError(
                f"sample {sample_id}: requested --layers {sorted(layers)} but only recorded "
                f"{sorted(recorded_layers)} (missing {missing}). _assert_layers_are_recordable "
                f"already checked this model/config combination, so a mismatch here means the "
                f"forward pass itself skipped some layers -- do not silently write a manifest "
                f"with missing layers, investigate first."
            )

        sample_payload: dict[str, Any] = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            "prompt_preview": prompt[:240],
            "original_token_count": original_len,
            "used_token_count": used_len,
            "prompt_token_offset": offset,
            "needle_spans": _span_payload(tokenizer, prompt, sample, offset, used_len),
            "layers": sorted(recorder.records, key=lambda r: r["layer"]),
        }
        if args.store_prompt:
            sample_payload["prompt"] = prompt
        manifest_samples.append(sample_payload)
        print(f"[stage0-dump] dumped {sample_id}: T={used_len}, layers={sorted(layers)}", flush=True)

    manifest = {
        "version": 1,
        "kind": "semantic_logkv_stage0_dump",
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "model": {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_query_groups": config.n_query_groups,
            "head_size": config.head_size,
            "block_size": config.block_size,
            "max_seq_length_used": model.max_seq_length,
        },
        "dump": {
            "layers": sorted(layers),
            "save_dtype": args.save_dtype,
            "compressed": bool(args.compressed),
            "hook": "CausalSelfAttention: post norm_q/norm_k, pre apply_rope",
            # Machine-readable marker for downstream tooling: this manifest only
            # ever has mechanism-A fields (k_raw/v/s_h/vh/needle spans). A future
            # mechanism-B (attn_mass_by_dist) or S0.8-3b dump/sweep script must
            # check this before assuming tail_query_count/MinHash-triple fields
            # exist here -- they never will, no matter how "kind" reads. Keep in
            # sync with the module docstring's scope note if this ever changes.
            "scope": "mechanism_a_kv_only",
        },
        "load_state_dict": {
            "missing_keys": len(load_result.missing_keys),
            "unexpected_keys": len(load_result.unexpected_keys),
            "allow_checkpoint_key_mismatch": bool(args.allow_checkpoint_key_mismatch),
        },
        "key_scale": key_scale.to_manifest(),
        "value_scale": value_scale.to_manifest(),
        "samples": manifest_samples,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[stage0-dump] wrote {manifest_path}")


if __name__ == "__main__":
    main()
