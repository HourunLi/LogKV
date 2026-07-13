"""
Continue Pre-Training (CPT) with standard causal LM + log-structured KV cache.

Usage:
    # Single GPU debug
    python demo.py --config exp/qwen0.6b-4k/debug.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 demo.py --config exp/qwen0.6b-32k/cpt-base.yaml

Architecture:
    - Training: standard causal LM (model(idx)), no KV cache
    - Inference (eval): log-structured KV cache with O(T log(N/T)) memory
      via model.set_log_kv_cache() + model(idx, input_pos=input_pos)
"""

import os
import re
import inspect
import shutil
import glob
import tempfile
from pathlib import Path
import yaml
from dataclasses import asdict
import torch
import lightning as L
from datetime import datetime
from litgpt import Config
from litgpt.model import GPT
from litgpt.utils import chunked_cross_entropy, load_checkpoint
from jsonargparse import CLI
import random
import numpy as np
from lightning.fabric.loggers import TensorBoardLogger
from litgpt.model import Block
import math
from lightning.fabric.strategies import DDPStrategy, FSDPStrategy
from datetime import timedelta
from litdata.streaming import (
    CombinedStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
    TokensLoader,
)

from data import litdata_chunks_dir, tokenizer_cache_key
from utils import *

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.bfloat16)


def _unique_save_dir(save_path: str) -> str:
    p = Path(os.path.expandvars(os.path.expanduser(save_path)))
    try:
        p = p.resolve()
    except OSError:
        p = Path(os.path.abspath(p))
    parent = p.parent
    base = p.name
    cand = p
    n = 2
    # Only bump when the candidate already holds a checkpoint, so a pre-existing
    # but checkpoint-less directory (TensorBoard logs, a failed run, a manually
    # created dir) is reused. This keeps the saved path equal to the YAML
    # `save_path` that the pipeline (majob.sh) later evaluates, instead of
    # silently diverging to `_v2` and leaving the eval pointed at an empty dir.
    while (cand / "lit_model.pth").exists():
        cand = parent / f"{base}_v{n}"
        n += 1
    return str(cand)


def _distributed_looks_multi_node() -> bool:
    try:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        lws = int(os.environ.get("LOCAL_WORLD_SIZE", str(ws)))
    except ValueError:
        return False
    return ws > lws


def get_lr(current_step, total_steps, warmup_steps, max_lr, min_lr):
    """Cosine LR schedule with linear warmup."""
    if current_step < warmup_steps:
        return max_lr * (current_step + 1) / max(warmup_steps, 1)
    if current_step > total_steps:
        return min_lr
    decay_ratio = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def build_train_dataset(
    *,
    context_length: int,
    seed: int,
    tok_dir: str,
    data_dir: str,
    dataset_dir: str,
    data_mix_yaml: str | None,
) -> StreamingDataset | CombinedStreamingDataset:
    """Single-source or multi-source (YAML-weighted) streaming dataset."""
    block = context_length + 1

    def _stream(path: str) -> StreamingDataset:
        return StreamingDataset(
            input_dir=path,
            item_loader=TokensLoader(block_size=block),
            shuffle=True,
            seed=seed,
        )

    if not data_mix_yaml:
        chunks = litdata_chunks_dir(tok_dir, data_dir, dataset_dir, context_length)
        if not os.path.isdir(chunks) or not any(os.scandir(chunks)):
            raise FileNotFoundError(
                f"LitData cache not found: {chunks}\nRun: python data.py ... --context_length {context_length}"
            )
        return _stream(chunks)

    yml = os.path.normpath(os.path.expanduser(data_mix_yaml))
    with open(yml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Data mix YAML must be a dict: {yml}")
    try:
        raw_paths, raw_w = cfg["paths"], cfg["weights"]
    except KeyError as e:
        raise ValueError(f"Data mix YAML must contain paths and weights: {yml}") from e
    if len(raw_paths) != len(raw_w):
        raise ValueError("paths and weights must have same length")
    base = cfg.get("base_dir")
    base = os.path.abspath(os.path.expanduser(str(base))) if base else ""

    def _resolve_yaml_path(p: str) -> str:
        p_exp = os.path.expanduser(str(p).strip())
        full = os.path.join(base, p_exp) if base else p_exp
        full = os.path.abspath(os.path.normpath(full))
        if not os.path.isdir(full):
            raise FileNotFoundError(f"paths entry is not a directory: {full!r}")
        key = tokenizer_cache_key(tok_dir)
        exact = os.path.join(full, f"litdata_{key}_ctx{context_length}")
        if os.path.isdir(exact) and any(os.scandir(exact)):
            return os.path.abspath(exact)
        raise FileNotFoundError(
            f"Not found: {exact!r} (run data.py with matching tokenizer and context_length first)"
        )

    dirs: list[str] = []
    for p in raw_paths:
        full = _resolve_yaml_path(str(p))
        if not os.path.isdir(full) or not any(os.scandir(full)):
            raise FileNotFoundError(f"Invalid LitData dir: {full}")
        dirs.append(full)

    s = float(sum(raw_w))
    if s <= 0:
        raise ValueError("weights sum must be positive")
    weights = tuple(float(w) / s for w in raw_w)
    iterate = bool(cfg.get("iterate_over_all", False))

    if len(dirs) == 1:
        return _stream(dirs[0])
    return CombinedStreamingDataset(
        datasets=[_stream(d) for d in dirs],
        seed=seed,
        weights=weights,
        iterate_over_all=iterate,
    )


def _load_yaml_config(yaml_path: str, model_dir: str) -> dict:
    """Load a YAML config file that may refer to another YAML via 'config:' key.

    Supports nesting: if the YAML has a 'config:' key pointing to another YAML,
    the base config is loaded and overridden with the current YAML's values.
    The 'config:' path is relative to the current YAML file.
    """
    yaml_path = os.path.normpath(os.path.expanduser(yaml_path))
    if not os.path.isfile(yaml_path):
        return {}

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    if "config" in cfg:
        base_name = cfg.pop("config")
        # Resolve relative to the current YAML file's directory
        base_dir = os.path.dirname(yaml_path)
        base_path = os.path.join(base_dir, base_name)
        base_cfg = _load_yaml_config(base_path, model_dir)
        merged = {**base_cfg, **cfg}
        return merged

    return cfg


# PyYAML implements YAML 1.1: scientific notation WITHOUT a decimal point
# ("2e-5", "5e-6") does not match its float resolver and loads as *str*.
# Configs in exp/ use exactly that style, so numeric params (learning_rate,
# min_lr, ...) would arrive as strings and crash AdamW / the LR schedule.
_SCI_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?[eE][-+]?\d+")


def _coerce_yaml_sci_floats(cfg: dict) -> dict:
    """Convert top-level str values that are scientific-notation numbers to float."""
    return {
        k: float(v) if isinstance(v, str) and _SCI_FLOAT_RE.fullmatch(v) else v
        for k, v in cfg.items()
    }


@auto_expand_env_vars
def main(
    # ── Model ──
    arch_name: str = "Qwen/Qwen3-0.6B-Base",
    context_length: int = 4096,
    ckpt_dir: str | None = None,
    resume_dir: str | None = None,
    # ── Training ──
    global_batch_size: int = 32,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    max_steps: int = 10,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.1,
    num_devices: int = 1,
    warmup_steps: int = 5,
    min_lr: float = 2e-6,
    entropy_chunk_size: int = 0,
    # ── Data ──
    dataset_name: str = "debug",
    dataset_dir: str = "data",
    data_dir: str = "data",
    tokenizer_dir: str | None = None,
    data_shuffle_seed: int = 42,
    data_mix_yaml: str | None = None,
    num_workers: int = 16,
    # ── IO ──
    save_ckpt: bool = False,
    save_path: str = "./ckpt/cpt",
    save_interval: int = 500,
    merged_state_staging_dir: str | None = None,
    enable_tensorboard: bool = True,
    tensorboard_root: str = "./tb",
    # ── Experiment ──
    expid: str = "debug",
    # ── Log-structured KV cache ──
    log_kv_B: int = 512,
    log_kv_recent_size: int = 1024,
    # Enable logKV simulation during training. Intended for the
    # short adaptation phase after dense pretraining.
    log_kv_training: bool = False,
    # ── RoPE ──
    # Override rotary_percentage to set d_pos = rope_n_elem = rotary_percentage * head_size.
    # 0.25 → d_pos=32 for head_size=128 (DeepSeek V4 style: 32-dim position, 96-dim content).
    # None → use the model's default (typically 1.0 = full RoPE).
    rotary_percentage: float | None = None,
    # ── Eval ──
    run_eval: str = "",  # "before" | "after" | "both"
    eval_benchmark: str = "debug",
    # ── YAML config ──
    config: str | None = None,
):
    # ── Load YAML config if provided ──
    # NOTE: `locals()[k] = v` does NOT write back to a function's real locals in
    # CPython (locals() returns a snapshot), so YAML overrides must be applied by
    # explicit re-binding below. A YAML value overrides the CLI/default value
    # unless it is null (None), matching the previous `if v is not None` intent.
    _yaml: dict = {}
    if config is not None:
        _yaml = _coerce_yaml_sci_floats(_load_yaml_config(os.path.join(os.getcwd(), config), "") or {})
        _valid = set(inspect.signature(main).parameters)
        for _k in _yaml:
            if _k != "config" and _k not in _valid:
                print(f"[demo] WARNING: unknown YAML key ignored: {_k}")

    def _o(name, current):
        v = _yaml.get(name)
        return v if v is not None else current

    arch_name = _o("arch_name", arch_name)
    context_length = _o("context_length", context_length)
    ckpt_dir = _o("ckpt_dir", ckpt_dir)
    resume_dir = _o("resume_dir", resume_dir)
    global_batch_size = _o("global_batch_size", global_batch_size)
    micro_batch_size = _o("micro_batch_size", micro_batch_size)
    num_epochs = _o("num_epochs", num_epochs)
    max_steps = _o("max_steps", max_steps)
    learning_rate = _o("learning_rate", learning_rate)
    weight_decay = _o("weight_decay", weight_decay)
    num_devices = _o("num_devices", num_devices)
    warmup_steps = _o("warmup_steps", warmup_steps)
    min_lr = _o("min_lr", min_lr)
    entropy_chunk_size = _o("entropy_chunk_size", entropy_chunk_size)
    dataset_name = _o("dataset_name", dataset_name)
    dataset_dir = _o("dataset_dir", dataset_dir)
    data_dir = _o("data_dir", data_dir)
    tokenizer_dir = _o("tokenizer_dir", tokenizer_dir)
    data_shuffle_seed = _o("data_shuffle_seed", data_shuffle_seed)
    data_mix_yaml = _o("data_mix_yaml", data_mix_yaml)
    num_workers = _o("num_workers", num_workers)
    save_ckpt = _o("save_ckpt", save_ckpt)
    save_path = _o("save_path", save_path)
    save_interval = _o("save_interval", save_interval)
    merged_state_staging_dir = _o("merged_state_staging_dir", merged_state_staging_dir)
    enable_tensorboard = _o("enable_tensorboard", enable_tensorboard)
    tensorboard_root = _o("tensorboard_root", tensorboard_root)
    expid = _o("expid", expid)
    log_kv_B = _o("log_kv_B", log_kv_B)
    log_kv_recent_size = _o("log_kv_recent_size", log_kv_recent_size)
    log_kv_training = _o("log_kv_training", log_kv_training)
    rotary_percentage = _o("rotary_percentage", rotary_percentage)
    run_eval = _o("run_eval", run_eval)
    eval_benchmark = _o("eval_benchmark", eval_benchmark)

    # 1. Set seeds
    set_random_seeds(42)
    tb_logger = TensorBoardLogger(root_dir=tensorboard_root, name=f"{expid}_{arch_name.replace('/', '-')}")
    loggers = [tb_logger] if enable_tensorboard else []

    # 2. Fabric setup
    use_fsdp = context_length > 4096
    print("Use FSDP:", use_fsdp)
    if not use_fsdp:
        strategy = DDPStrategy(timeout=timedelta(days=3650))
    else:
        strategy = FSDPStrategy(
            sharding_strategy="SHARD_GRAD_OP",
            state_dict_type="full",
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            timeout=timedelta(days=3650),
        )
    fabric = L.Fabric(
        accelerator="cuda",
        devices=num_devices,
        num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)),
        strategy=strategy,
        precision="bf16-true",
        loggers=loggers,
    )
    fabric.launch()
    fabric.print("Tensorboard root:", tensorboard_root)

    config_obj = Config.from_name(arch_name)
    assert config_obj is not None
    config_obj.block_size = context_length
    if rotary_percentage is not None:
        config_obj.rotary_percentage = rotary_percentage
        config_obj.rope_n_elem = int(rotary_percentage * config_obj.head_size)
        fabric.print(f"Overridden rotary_percentage={rotary_percentage}, rope_n_elem={config_obj.rope_n_elem}")
    fabric.print(f"Model config initialized: {config_obj.name}")

    with fabric.init_module(empty_init=True):
        model = GPT(config_obj)

    checkpoint_dir = f"checkpoints/{arch_name}"
    if ckpt_dir is not None:
        checkpoint_dir = ckpt_dir
    load_dir = checkpoint_dir if resume_dir is None else resume_dir
    ckpt_path = Path(load_dir) / "lit_model.pth"
    fabric.print(f"Loading checkpoint from {ckpt_path}...")

    model = fabric.setup_module(model)

    # Single load: load_checkpoint streams weights into the (FSDP-)wrapped model.
    # A prior torch.load() here only to peek at the state dict doubled peak CPU
    # memory for large checkpoints without being used.
    load_checkpoint(fabric, model, ckpt_path, strict=True)
    fabric.print("Weights loaded successfully.")

    # ── Optimizer with weight decay groups ──
    decay_params = []
    no_decay_params = []
    no_decay_keywords = ["bias", "norm", "ln_"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    fabric.print(f"Weight decay params: {len(decay_params)} tensors")
    fabric.print(f"No weight decay params: {len(no_decay_params)} tensors")

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # ── Evaluation helper ──
    def _run_eval(ckpt, benchmark):
        from eval import main as eval_main
        eval_main(checkpoint_dir=ckpt, benchmark=benchmark)

    if run_eval in ("before", "both"):
        _run_eval(load_dir, eval_benchmark)

    # ── Data loading ──
    tok_dir = tokenizer_dir if tokenizer_dir is not None else load_dir
    train_dataset = build_train_dataset(
        context_length=context_length,
        seed=data_shuffle_seed,
        tok_dir=tok_dir,
        data_dir=data_dir,
        dataset_dir=dataset_dir,
        data_mix_yaml=data_mix_yaml,
    )
    if isinstance(train_dataset, CombinedStreamingDataset):
        fabric.print(f"[Rank {fabric.global_rank}] Training data: CombinedStreamingDataset (multi-source)")
    else:
        fabric.print(f"[Rank {fabric.global_rank}] Training data: StreamingDataset (single source)")
    fabric.barrier()
    dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    fabric.print("Starting Continue Pretraining...")
    model.train()

    # Enable logKV training simulation for the adaptation phase.
    if log_kv_training:
        model.enable_log_kv_training(
            batch_size=micro_batch_size,
            max_seq_length=context_length,
            device=fabric.device,
            # Allocate cache buffers in the activation dtype instead of relying on
            # the process default; keeps them aligned with the bf16-true params.
            dtype=next(model.parameters()).dtype,
            B=log_kv_B,
            recent_size=log_kv_recent_size,
        )
        fabric.print(
            f"logKV training ENABLED: B={log_kv_B}, "
            f"recent_size={log_kv_recent_size}, "
            f"chunks/seq={context_length // 2}"
        )

    gradient_accumulation_steps = max(1, global_batch_size // (micro_batch_size * fabric.world_size))
    optimizer.zero_grad(set_to_none=True)
    step_start_time = datetime.now()
    step_stats = MicroStepMeanStats()
    global_step = 0
    total_steps = max_steps
    training_finished = False
    micro_batch_idx = 0
    data_epoch = 1
    loader_iter = iter(dataloader)

    while global_step < max_steps and not training_finished:
        try:
            train_data = next(loader_iter)
        except StopIteration:
            data_epoch += 1
            if data_epoch > num_epochs:
                fabric.print(f"Data exhausted after {num_epochs} epochs, step={global_step}/{max_steps}. Stopping.")
                break
            loader_iter = iter(dataloader)
            continue

        inputs = train_data[:, 0:context_length].contiguous().long()
        targets = train_data[:, 1:context_length + 1].contiguous().long()
        is_accumulating = (micro_batch_idx + 1) % gradient_accumulation_steps != 0
        micro_batch_idx += 1

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            # Standard causal LM forward (no KV cache during training)
            logits = model(inputs)
            loss = chunked_cross_entropy(logits, targets, chunk_size=entropy_chunk_size)
            # Feed the per-micro-batch loss into the step aggregator; without this
            # step_stats.averages() is always empty and neither the console line
            # nor TensorBoard ever shows the training loss.
            step_stats.accumulate(loss=loss.detach().item())

            loss = loss / gradient_accumulation_steps
            fabric.backward(loss)

        if not is_accumulating:
            current_lr = get_lr(global_step, total_steps, warmup_steps, learning_rate, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            now = datetime.now()
            step_time = (now - step_start_time).total_seconds()
            step_start_time = now
            avgs = step_stats.averages()
            metrics_txt = " | ".join(f"{k}: {v:.4f}" for k, v in sorted(avgs.items()))
            if metrics_txt:
                metrics_txt = metrics_txt + " | "
            fabric.print(
                f"[{now.strftime('%H:%M:%S')}] "
                f"Epoch {data_epoch} | Step {global_step + 1} | "
                f"{metrics_txt}"
                f"Time: {step_time:.2f}s"
            )
            for name, val in avgs.items():
                fabric.log(f"train/{name}", val, step=global_step + 1)
            fabric.log("train/learning_rate", current_lr, step=global_step + 1)

            step_stats.reset()
            global_step += 1

            if global_step >= max_steps:
                fabric.print(f"Reached max_steps={max_steps}. Training complete.")
                training_finished = True

            if save_ckpt and save_interval > 0 and global_step % save_interval == 0 and not training_finished:
                step_save_path = f"{save_path}/step_{global_step}"
                step_save_path = _unique_save_dir(step_save_path)
                if fabric.global_rank == 0:
                    os.makedirs(step_save_path, exist_ok=True)
                fabric.barrier()
                state = {"model": model, "optimizer": optimizer, "global_step": global_step}
                fabric.save(f"{step_save_path}/lit_model.pth", state)
                fabric.barrier()
                if fabric.global_rank == 0:
                    for file_path in glob.glob(f"{checkpoint_dir}/*.json") + glob.glob(f"{checkpoint_dir}/*.model"):
                        shutil.copy(file_path, step_save_path)
                    with open(f"{step_save_path}/model_config.yaml", "w", encoding="utf-8") as f:
                        yaml.dump(asdict(config_obj), f)
                fabric.barrier()
                fabric.print(f"Checkpoint saved to {step_save_path}")

    # ── Final save ──
    if save_ckpt:
        save_path = _unique_save_dir(save_path)
        if fabric.global_rank == 0:
            fabric.print(f"Final save dir: {save_path}")
            os.makedirs(save_path, exist_ok=True)
        fabric.barrier()

        state = {"model": model, "optimizer": optimizer, "global_step": global_step}
        fabric.print(f"Saving to {save_path}/lit_model.pth ...")
        fabric.save(f"{save_path}/lit_model.pth", state)
        fabric.barrier()

        if fabric.global_rank == 0:
            for file_path in glob.glob(f"{checkpoint_dir}/*.json") + glob.glob(f"{checkpoint_dir}/*.model"):
                shutil.copy(file_path, save_path)
            with open(f"{save_path}/model_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(asdict(config_obj), f)
            fabric.print(f"Tokenizer and model_config written to {save_path}")
            fabric.print("Done.")
        fabric.barrier()

    if run_eval in ("after", "both"):
        _run_eval(save_path, eval_benchmark)

    fabric.print("Training finished!")


if __name__ == "__main__":
    CLI(main)
