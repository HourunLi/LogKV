from eval import main as eval_main
import os
import re
import shutil
import glob
import tempfile
from pathlib import Path
# os.environ['https_proxy'] = '127.0.0.1:7897'
# os.environ['http_proxy'] = '127.0.0.1:7897'
import yaml
from dataclasses import asdict
import torch
import lightning as L
from datetime import datetime
from torch.utils.data import DataLoader
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
from contextlib import nullcontext
from litdata.streaming import (
    CombinedStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
    TokensLoader,
)

from data import litdata_chunks_dir, tokenizer_cache_key
from utils import *

torch.set_float32_matmul_precision('high')
torch.set_default_dtype(torch.bfloat16)

def get_lr(current_step, total_steps, warmup_steps, max_lr, min_lr):
    """
    大厂标准 LR 调度器：前段线性 Warmup，后段余弦退火 (Cosine Decay)
    """
    # 1. Warmup 阶段：从 0 线性爬升到 max_lr
    if current_step < warmup_steps:
        # 防止除零错误，最少给极小值
        return max_lr * (current_step + 1) / warmup_steps
        
    # 2. 如果超出了最大训练步数，保持最小学习率
    if current_step > total_steps:
        return min_lr
        
    # 3. 余弦退火阶段：从 max_lr 极其平滑地滑落到 min_lr
    decay_ratio = (current_step - warmup_steps) / (total_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    
    # math.cos 接收弧度 (0 到 pi)，产出 1 到 -1
    # 经过 0.5 * (1 + ...) 变换后，coeff 会从 1 平滑下降到 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    
    return min_lr + coeff * (max_lr - min_lr)

def generate_step_driven_mask(batch_size, seq_len, current_step, total_steps, device, stable=0, schedule=None):
    """
    基于当前训练 Iter 的动态断点生成器
    一条序列只有一个断点。断点之前为 True (Prefill)，断点之后为 False (Decode)
    """
    if stable == -1:
        progress = 1
    else:
        progress = min(current_step / max(1, total_steps - stable), 1)
    
    # 动态计算当前的上下界
    min_prefill_ratio = 0.01 + progress * 0.1
    max_prefill_ratio = 0.05 + progress * 0.9
    
    # 为 Batch 中的【每一条序列】独立地均匀随机生成一个断点
    breakpoints = [int(random.uniform(min_prefill_ratio, max_prefill_ratio) * seq_len) for _ in range(batch_size)]
    
    breakpoints_tensor = torch.tensor(breakpoints, device=device).unsqueeze(1) # [B, 1]
    
    # 创造一个形状为 [B, T] 的递增索引矩阵
    seq_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    
    # 🌟 魔法广播：只要索引 < 断点，就是 Prefill (True)
    prefill_mask = seq_indices < breakpoints_tensor
    
    return prefill_mask


def decode_prefix_mean_ce_multi_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    prefill_mask: torch.Tensor,
    ks: list[int],
    ignore_index: int = -100,
) -> dict[int, torch.Tensor | None]:
    """
    监控 Decode 阶段的上下文休克。
    支持传入多个 k 值，计算每条样本 decode 段前 k 个位置的平均 CE。

    仅在每个样本断点后的 [bp, bp + max(ks)) 窗口上算 CE，避免对整段 T 做
    (B*T, V) 的 cross_entropy（长上下文下即使用 no_grad 也会 OOM）。
    
    Args:
        ks: 一个包含多个 k 值的列表，例如 [1, 5, 10]
        
    Returns:
        dict: 形如 {1: tensor(11.9), 5: tensor(8.2), 10: tensor(5.4)}
    """
    if not ks:
        return {}

    B, T, V = logits.shape
    device = logits.device
    positive_ks = [k for k in ks if k > 0]
    results: dict[int, torch.Tensor | None] = {k: None for k in ks if k <= 0}
    if not positive_ks:
        return results

    k_max = max(positive_ks)
    bp = prefill_mask.sum(dim=1, keepdim=True).to(dtype=torch.long, device=device)  # (B, 1)
    offsets = torch.arange(k_max, device=device, dtype=torch.long).view(1, -1).expand(B, -1)
    t_sel = bp + offsets  # (B, k_max) 绝对位置
    in_bounds = t_sel < T
    t_clamped = t_sel.clamp(max=T - 1)

    b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(t_clamped)
    logit_win = logits[b_idx, t_clamped]  # (B, k_max, V)
    tgt_for_ce = torch.where(
        in_bounds,
        targets[b_idx, t_clamped],
        torch.tensor(ignore_index, device=device, dtype=targets.dtype),
    )
    ce_win = torch.nn.functional.cross_entropy(
        logit_win.reshape(-1, V),
        tgt_for_ce.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, k_max)

    rel = offsets  # 0 .. k_max-1，对应断点后第几个 decode token
    valid_targets_win = in_bounds & (targets[b_idx, t_clamped] != ignore_index)

    for k in ks:
        if k <= 0:
            continue
        m = (rel < k) & valid_targets_win
        if not m.any():
            results[k] = None
        else:
            results[k] = ce_win[m].mean()
    return results


def build_train_dataset(
    *,
    context_length: int,
    seed: int,
    tok_dir: str,
    data_dir: str,
    dataset_dir: str,
    data_mix_yaml: str | None,
) -> StreamingDataset | CombinedStreamingDataset:
    """无 ``data_mix_yaml`` → ``litdata_chunks_dir`` 单源；有 → YAML 的 ``paths``（语料父目录）+ ``weights`` → ``CombinedStreamingDataset``。"""
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
                f"未找到 LitData 缓存: {chunks}\n请先: python data.py ... --context_length {context_length}"
            )
        return _stream(chunks)

    yml = os.path.normpath(os.path.expanduser(data_mix_yaml))
    with open(yml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"数据配比 YAML 须为 dict: {yml}")
    try:
        raw_paths, raw_w = cfg["paths"], cfg["weights"]
    except KeyError as e:
        raise ValueError(f"数据配比 YAML 必须包含 paths 与 weights: {yml}") from e
    if len(raw_paths) != len(raw_w):
        raise ValueError("paths 与 weights 长度须相同")
    base = cfg.get("base_dir")
    base = os.path.abspath(os.path.expanduser(str(base))) if base else ""

    def _resolve_yaml_path(p: str) -> str:
        """每项为语料根目录（与 data.py 的 ``--data_dir`` 对应），解析为 ``<dir>/litdata_<hash>_ctx<len>``。"""
        p_exp = os.path.expanduser(str(p).strip())
        full = os.path.join(base, p_exp) if base else p_exp
        full = os.path.abspath(os.path.normpath(full))
        if not os.path.isdir(full):
            raise FileNotFoundError(f"paths 不是目录: {full!r}")
        key = tokenizer_cache_key(tok_dir)
        exact = os.path.join(full, f"litdata_{key}_ctx{context_length}")
        if os.path.isdir(exact) and any(os.scandir(exact)):
            return os.path.abspath(exact)
        raise FileNotFoundError(
            f"未找到 {exact!r}（需与当前 tokenizer、context_length 一致地先跑 data.py）"
        )

    dirs: list[str] = []
    for p in raw_paths:
        full = _resolve_yaml_path(str(p))
        if not os.path.isdir(full) or not any(os.scandir(full)):
            raise FileNotFoundError(f"LitData 目录无效（需已 data.py optimize）: {full}")
        dirs.append(full)

    s = float(sum(raw_w))
    if s <= 0:
        raise ValueError("weights 之和须为正")
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


@auto_expand_env_vars
def main(
        # MODEL
        arch_name: str = "Qwen/Qwen3-0.6B-Base",
        context_length: int = 4096,
        ckpt_dir: str | None = None,
        resume_dir: str | None = None,
        # TRAINING
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
        # DATA
        dataset_name: str = "debug",
        dataset_dir: str = "data",
        data_dir: str = "data",
        tokenizer_dir: str | None = None,
        data_shuffle_seed: int = 42,
        data_mix_yaml: str | None = None,
        num_workers: int = 16,
        # IO
        save_ckpt: bool = False,
        save_path: str = "./ckpt/cpt",
        enable_tensorboard: bool = True,
        tensorboard_root: str = './tb',
        # RESEARCH
        expid: str = 'debug',
        use_research: bool = False,
        research_swa_size: int = 512,
        research_swa_layers_str: str = "0,2,4,6,8,10,12,14,16,18,20,22,24,26",
        research_identity_layers_str: str = "1,3,5,7,9,11,13,15,17,19,21,23,25,27",
        research_breakpoint_schedule: str | None = None,
        research_breakpoint_schedule_stable: int = 0,
        research_separate_parameter: bool = True,
        research_decode_prefix_tokens: int = 1,
        research_decode_prefix_loss_weight: float = 0,
        research_prefill_supervise: bool = False,
        research_remove_order_str: str = "",
        research_remove_interval: int = 0,
        # EVAL
        run_eval: str = "",  # "before" | "after" | "both"
        eval_benchmark: str = "debug",
        eval_map_branch: bool = False,
):

    # 1. set seeds
    set_random_seeds(42)
    # timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logger = TensorBoardLogger(root_dir=tensorboard_root, name=f"{expid}_{arch_name.replace('/', '-')}")
    loggers = [tb_logger] if enable_tensorboard else []
    
    # 2. 这里的 Fabric 逻辑保持不变...
    find_unused_parameters = True if len(research_remove_order_str) > 0 else False
    use_fsdp = (context_length > 4096)
    print("Use FSDP:", use_fsdp)
    if not use_fsdp:
        strategy = DDPStrategy(
            timeout=timedelta(days=3650),
            find_unused_parameters=find_unused_parameters
        )
    else:
        strategy = FSDPStrategy(
            # FULL_SHARD 等价于 DeepSpeed ZeRO-3，切分权重、梯度和优化器状态
            # 如果显存依然吃紧，可以保持 FULL_SHARD；如果计算通信比瓶颈明显，可改为 SHARD_GRAD_OP (ZeRO-2)
            sharding_strategy="SHARD_GRAD_OP", 
            auto_wrap_policy={Block}, 
            activation_checkpointing_policy={Block}, 
            timeout=timedelta(days=3650),
        )
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=num_devices, 
        num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)), # 兼容单机和多机
        strategy=strategy,
        precision='bf16-true',
        loggers=loggers,
    )
    fabric.launch()
    fabric.print("Tensorboard root:", tensorboard_root)

    swa_layers = [int(x.strip()) for x in research_swa_layers_str.split(",")] if research_swa_layers_str else []
    research_remove_order = [int(x.strip()) for x in research_remove_order_str.split(",")] if research_remove_order_str else []

    config = Config.from_name(arch_name) 
    assert config is not None
    config.block_size = context_length
    config.use_research = use_research
    config.research_swa_size = research_swa_size
    config.research_prefill_swa_layers = swa_layers
    config.research_separate_parameter = research_separate_parameter
    config.research_removed_layers = []
    fabric.print(f"⚙️ 模型 Config 初始化完成: {config.name}")

    with fabric.init_module(empty_init=True):
        model = GPT(config)

    checkpoint_dir = f"checkpoints/{arch_name}"
    if ckpt_dir is not None:
        checkpoint_dir = ckpt_dir
    load_dir = checkpoint_dir if resume_dir is None else resume_dir
    ckpt_path = Path(load_dir) / "lit_model.pth"
    fabric.print(f"🔄 正在从 {ckpt_path} 加载预训练 Checkpoint...")

    # 与 litgpt/pretrain、litgpt/utils.load_checkpoint 一致：先 fabric.setup，再加载。
    # FSDP 下必须在 wrap 之后用 fabric.load_raw，否则易出现分片与全量 state_dict 不匹配。
    merged_state_dict = None
    raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw_ckpt["model"] if "model" in raw_ckpt else raw_ckpt
    del raw_ckpt

    if config.use_research and config.research_separate_parameter:
        fabric.print("🔀 检测到 Block 级参数独立！正在为 h_prefill 组装预训练权重...")
        prefill_weights = {}
        for i, block_idx in enumerate(config.research_prefill_swa_layers):
            orig_prefix = f"transformer.h.{block_idx}."
            new_prefix = f"transformer.h_prefill.{i}."
            for key, value in state_dict.items():
                if key.startswith(orig_prefix):
                    new_key = key.replace(orig_prefix, new_prefix, 1)
                    if new_key not in state_dict.keys():
                        prefill_weights[new_key] = value.clone()
        state_dict.update(prefill_weights)
        fabric.print(f"✅ 成功映射并注入了 {len(prefill_weights)} 个 Block 级别的张量！")
        merged_state_dict = state_dict

    model = fabric.setup_module(model)

    if merged_state_dict is not None:
        fd, tmp_path = tempfile.mkstemp(suffix=".pth")
        os.close(fd)
        try:
            torch.save({"model": merged_state_dict}, tmp_path)
            load_checkpoint(fabric, model, Path(tmp_path), strict=False)
        finally:
            os.unlink(tmp_path)
    else:
        load_checkpoint(fabric, model, ckpt_path, strict=True)

    fabric.print("✅ 真实权重加载成功！所有分支已完成 Pre-trained 初始化。")

    # ==========================================
    # 🌟 工业级优化器初始化：Weight Decay 分组过滤
    # ==========================================
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

    # 打印出来心里有底
    fabric.print(f"⚙️ 施加 Weight Decay 的参数组 (如 Linear): {len(decay_params)} 个张量")
    fabric.print(f"⚙️ 豁免 Weight Decay 的参数组 (如 Norm, Bias): {len(no_decay_params)} 个张量")

    # 组装成包含两组字典的列表喂给优化器
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},    # 🌟 黄金默认值 0.1
        {"params": no_decay_params, "weight_decay": 0.0}  # 绝对不能压缩
    ]

    # 这里假设你配置文件里的 learning_rate 是 2e-5 之类的 CPT 常用学习率
    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=learning_rate, 
        betas=(0.9, 0.95),  # LLM 预训练标配的 betas
        eps=1e-8
    )
    optimizer = fabric.setup_optimizers(optimizer)

    def _run_eval(ckpt):
        eval_main(checkpoint_dir=ckpt, benchmark=eval_benchmark, map_branch=eval_map_branch, config_overrides=asdict(config))

    if run_eval in ("before", "both"):
        _run_eval(load_dir)

    # 3. LitData：单源目录 或 CombinedStreamingDataset 配比多源
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
        fabric.print(f"[Rank {fabric.global_rank}] 训练数据: CombinedStreamingDataset（多源配比）")
    else:
        fabric.print(f"[Rank {fabric.global_rank}] 训练数据: StreamingDataset（单源）")
    fabric.barrier()
    dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    fabric.print("🚀 开始 Continue Pretraining...")
    model.train()

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
                fabric.print(f"🛑 已遍历数据 {num_epochs} 个 epoch，未达到 max_steps={max_steps}，停止。")
                break
            loader_iter = iter(dataloader)
            continue

        inputs = train_data[:, 0:context_length].contiguous().long()
        targets = train_data[:, 1 : context_length + 1].contiguous().long()
        is_accumulating = (micro_batch_idx + 1) % gradient_accumulation_steps != 0
        micro_batch_idx += 1

        prefill_mask = generate_step_driven_mask(
            batch_size=inputs.size(0),
            seq_len=inputs.size(1),
            current_step=global_step + 1,
            total_steps=total_steps,
            device=fabric.device,
            schedule=research_breakpoint_schedule,
            stable=research_breakpoint_schedule_stable,
        )

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(inputs, prefill_mask=prefill_mask)

            masked_targets = targets.masked_fill(prefill_mask == True, -100)
            compariable_decode_loss = chunked_cross_entropy(logits, masked_targets, chunk_size=entropy_chunk_size)
            if use_research and not research_prefill_supervise:
                loss = compariable_decode_loss
            else:
                loss = chunked_cross_entropy(logits, targets, chunk_size=entropy_chunk_size)

            metrics = {
                "compariable_loss": compariable_decode_loss.detach().item(),
            }
            if research_decode_prefix_tokens > 0:
                ks = [research_decode_prefix_tokens, 8, 16]
                if 1 not in ks:
                    ks.insert(0, 1)
                grad_ctx = torch.no_grad() if research_decode_prefix_loss_weight == 0 else nullcontext()
                with grad_ctx:
                    prefix_losses = decode_prefix_mean_ce_multi_k(logits, targets, prefill_mask, ks=ks)
                if len(prefix_losses) > 0:
                    for k, prefix_loss in prefix_losses.items():
                        metrics[f"prefix_loss_{k}"] = prefix_loss.detach().item()
                    if use_research and research_decode_prefix_loss_weight > 0:
                        loss += prefix_losses[research_decode_prefix_tokens] * research_decode_prefix_loss_weight
            metrics["loss"] = loss.detach().item()
            metrics["prefill_ratio"] = (prefill_mask.sum() / prefill_mask.numel()).item()
            step_stats.accumulate(**metrics)

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
                fabric.print(f"🚨 已达到最大训练步数 {max_steps}，提前结束训练！")
                training_finished = True

            if research_remove_interval > 0 and global_step % research_remove_interval == 0:
                remove_index = global_step // research_remove_interval
                if remove_index <= len(research_remove_order):
                    remove_index = research_remove_order[remove_index - 1]
                    config.research_removed_layers.append(remove_index)
                    fabric.print(f"🔥 已移除层 {remove_index}，当前所有已移除层为 {config.research_removed_layers}")

    if save_ckpt:
        os.makedirs(save_path, exist_ok=True)
        fabric.print(f"💾 正在保存模型至 {save_path}")
        
        # 🌟 直接传对象引用！不需要显式调用 model.state_dict()
        # 甚至可以顺手把 optimizer 的状态也存进去，方便中断后继续训练
        state = {
            "model": model, 
            "optimizer": optimizer, 
            "global_step": global_step
        }
        
        # fabric.save 底层会安全地萃取出没有 module. 前缀的纯净权重
        fabric.save(f"{save_path}/lit_model.pth", state)

        for file_path in glob.glob(f"{checkpoint_dir}/*.json") + glob.glob(f"{checkpoint_dir}/*.model"):
            shutil.copy(file_path, save_path)
        with open(f"{save_path}/model_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(asdict(config), f)
            
        fabric.print(f"📦 Tokenizer 和 Config 已自动同步至 {save_path}")
    
    if run_eval in ("after", "both"):
        _run_eval(save_path)

    fabric.print("🎉 训练运行结束！")

if __name__ == "__main__":
    CLI(main)