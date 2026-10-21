"""Train niahLogKV's learned retention heads on a frozen LogKV checkpoint.

    torchrun --nproc_per_node=8 train_retention.py \\
        --config exp/qwen1.7b-32k/<checkpoint YAML> --out <retention_heads.pth>

The YAML names the checkpoint (``save_path``, or ``--checkpoint``), its LogKV
cache (built exactly as eval.py builds it) and its training data
(``data_dir`` / ``data_mix_yaml``, read as demo.py reads them). Each step
takes one block of that data, inserts synthetic facts
(litgpt.synthetic_retrieval), prefills it through the frozen model with the
heads attached and trains every layer's head on the spans each flush archives
(litgpt.retention_train). Ranks take different blocks and average the head
gradients. Every ``--val-every`` steps rank 0 reports, on fixed held-out
blocks, the fraction of fact tokens each layer kept exact, and saves the heads.

``--kinds`` limits the value kinds that appear in the facts, e.g. leave out
``uuid`` to test on niah_single_3 values the heads never saw.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def tokenizer_dir(cfg: dict) -> str | None:
    """The tokenizer eval.sh would pass: tokenizer_dir, resume_dir, ckpt_dir, then checkpoints/<arch>."""
    candidates = [cfg.get("tokenizer_dir"), cfg.get("resume_dir"), cfg.get("ckpt_dir"),
                  os.path.join("checkpoints", cfg.get("arch_name", "Qwen/Qwen3-0.6B-Base"))]
    for path in candidates:
        if path and any(os.path.isfile(os.path.join(path, name)) for name in ("tokenizer.json", "tokenizer.model")):
            return path
    return None


def checkpoint_dir(resolved: str, requested: str | None) -> str:
    """The resolved checkpoint, or a clear error when it holds no weights."""
    if not os.path.exists(os.path.join(resolved, "lit_model.pth")):
        raise SystemExit(f"no checkpoint at {resolved} (no lit_model.pth there or in a step_* subdirectory; "
                         f"{'--checkpoint' if requested else 'the YAML save_path'} names it). Train it first "
                         "(majob.sh), or pass --checkpoint <dir holding lit_model.pth or step_* subdirectories>.")
    return resolved


def build_lm(config: str, checkpoint: str | None, device: str):
    """eval.py's LogKVLM for an experiment YAML (the same settings eval.sh passes)."""
    ev = importlib.import_module("eval")
    cfg = ev._coerce_yaml_sci_floats(ev.expand_env_vars(
        ev._load_yaml_config(os.path.join(os.getcwd(), config), checkpoint or "") or {}))
    defaults = {k: p.default for k, p in inspect.signature(ev.main).parameters.items()}
    kwargs = {}
    for name in inspect.signature(ev.LogKVLM.__init__).parameters:
        if name.startswith("log_kv_") and name in defaults:
            value = cfg.get(name)
            kwargs[name] = defaults[name] if value is None else value
    kwargs["log_kv_retention_heads"] = None  # the heads come from this script
    path = checkpoint_dir(ev._resolve_checkpoint_dir(checkpoint or cfg["save_path"]), checkpoint)
    lm = ev.LogKVLM(path, device=device, config_overrides=cfg.get("config_overrides"),
                    tokenizer_dir=tokenizer_dir(cfg), **kwargs)
    return cfg, lm


def blocks(cfg: dict, seed: int):
    """Token blocks of the YAML's training data, as demo.py streams them."""
    demo = importlib.import_module("demo")
    torch.set_default_dtype(torch.float32)  # demo.py sets bfloat16 at import
    tok_dir = cfg.get("tokenizer_dir") or cfg.get("resume_dir") or cfg.get("ckpt_dir")  # demo.py's LitData key
    dataset = demo.build_train_dataset(
        context_length=int(cfg.get("context_length", 32768)), seed=seed, tok_dir=tok_dir,
        data_dir=cfg.get("data_dir", "data"), dataset_dir=cfg.get("dataset_dir", "data"),
        data_mix_yaml=cfg.get("data_mix_yaml"))
    while True:
        for block in dataset:
            yield np.asarray(block, dtype=np.int64)


def _pair(text: str) -> tuple[int, int]:
    low, high = (int(x) for x in text.split(","))
    return low, high


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment YAML of the checkpoint, as for eval.sh")
    p.add_argument("--checkpoint", default=None, help="defaults to the YAML save_path (latest step_*)")
    p.add_argument("--out", required=True, help="where to save the heads (litgpt.retention format)")
    p.add_argument("--init", default=None, help="heads to continue from (e.g. refit after CPT)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--length", type=int, default=32768, help="tokens per training block")
    p.add_argument("--facts", default="2,8", help="facts per block, low,high")
    p.add_argument("--questions", default="1,3", help="questions per block, low,high")
    p.add_argument("--kinds", default=None, help="comma-separated value kinds (default: all)")
    p.add_argument("--val-docs", type=int, default=16)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from litgpt.retention import RetentionHeads
    from litgpt.retention_train import HeadTrainer, pool_recall
    from litgpt.synthetic_retrieval import KINDS, RetrievalKit, insert_facts

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    device = args.device
    if world > 1:
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        torch.distributed.init_process_group("nccl")
        device = f"cuda:{local}"
    cfg, lm = build_lm(args.config, args.checkpoint, device)
    if not lm.log_kv_alpha_exact_tokens:
        raise SystemExit("retention heads score Alpha's exact spans; this config has alpha_exact_tokens = 0")
    data = blocks(cfg, args.seed)
    kinds = tuple(args.kinds.split(",")) if args.kinds else KINDS
    kit = RetrievalKit.from_tokenizer(lm.tokenizer, kinds)
    model = lm.model
    if args.init:
        heads = RetentionHeads.load(args.init)
    else:
        heads = RetentionHeads(model.config.n_layer, model.config.n_query_groups, model.config.head_size,
                               model.config.head_size, args.hidden).float()
    model.set_retention_heads(heads)
    lm._retention_heads = heads
    trainer = HeadTrainer(heads, lr=args.lr)
    facts, questions = _pair(args.facts), _pair(args.questions)

    def document(stream, rng):
        ids, mask = insert_facts(next(stream)[:args.length], rng, kit, facts=facts, questions=questions)
        return torch.as_tensor(ids, device=device)[None], mask

    def forward(ids, mask):
        trainer.set_labels([mask])
        lm._set_eval_cache()
        with torch.no_grad():
            model(ids, input_pos=torch.arange(ids.size(1), device=device), lm_head_start=ids.size(1) - 1)
        trainer.set_labels(None)

    held_out = [document(blocks(cfg, args.seed + 999), np.random.default_rng([args.seed, 999, i]))
                for i in range(args.val_docs)] if rank == 0 else []

    def validate(step):
        trainer.train = False
        trainer.reset_stats()
        recalls = []
        for ids, mask in held_out:
            forward(ids, mask)
            recalls.append(pool_recall(model, [mask]))
        trainer.train = True
        summary = trainer.summary()
        trainer.reset_stats()
        layers = np.nanmean(np.asarray(recalls), axis=0)
        print(json.dumps(dict(step=step, val=summary, kept_per_layer=[round(x, 3) for x in layers],
                              kept_mean=round(float(np.nanmean(layers)), 4))), flush=True)
        heads.save(args.out, config=args.config, checkpoint=args.checkpoint or cfg.get("save_path"), step=step,
                   kinds=list(kinds), facts=list(facts), tag=heads.tag())
        return layers

    with trainer:
        if rank == 0:
            validate(0)  # A fresh head scores below the floor: this is the K/V-only pool.
        start = time.time()
        for step in range(1, args.steps + 1):
            forward(*document(data, np.random.default_rng([args.seed, rank, step])))
            trainer.step()
            if rank == 0 and step % 10 == 0:
                print(json.dumps(dict(step=step, train=trainer.summary(),
                                      seconds=round(time.time() - start, 1))), flush=True)
                trainer.reset_stats()
            if rank == 0 and (step % args.val_every == 0 or step == args.steps):
                validate(step)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
