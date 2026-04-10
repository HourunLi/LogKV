import os
import re
import shutil
import glob
# os.environ['https_proxy'] = '127.0.0.1:7897'
# os.environ['http_proxy'] = '127.0.0.1:7897'
import yaml
from dataclasses import asdict
import torch
import lightning as L
from datetime import datetime
from torch.utils.data import DataLoader, IterableDataset, ConcatDataset
from litgpt import Config
from litgpt.model import GPT
from litgpt.utils import chunked_cross_entropy
from jsonargparse import CLI
import random
import numpy as np
from lightning.fabric.loggers import TensorBoardLogger
import math
from lightning.fabric.strategies import DDPStrategy
from datetime import timedelta
from contextlib import nullcontext
from utils import *

torch.set_float32_matmul_precision('high')

# ==========================================
# 🚀 满血形态：基于 Memmap 的二进制数据集
# ==========================================
class CPTBinDataset(torch.utils.data.Dataset):
    """
    极速读取预编译的 .bin 文件。CPU 开销几乎为 0。
    """
    def __init__(self, bin_path: str, seq_len: int):
        # 魔法所在：memmap 不会把大文件一次性读进内存，而是按需在硬盘和内存间滑动
        self.data = np.memmap(bin_path, dtype=np.int32, mode='r')
        self.seq_len = seq_len
        self.chunk_size = seq_len + 1
        
        # 精确计算这个文件一共能切出多少个完整的 Batch
        self.total_chunks = len(self.data) // self.chunk_size

    def __len__(self):
        # 现在 Dataloader 终于知道你有多少数据了！
        return self.total_chunks

    def __getitem__(self, idx: int):
        # 直接用索引在极速数组上切片
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        # 截取数据并转为 int64 张量 (PyTorch Embedding 层的硬性要求)
        chunk = self.data[start_idx:end_idx]
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))
        
        # 返回 inputs 和 targets
        return chunk_tensor[:-1], chunk_tensor[1:]

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

# ==========================================
# 🌟 你的专属 Schedule 逻辑
# ==========================================
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
    
    Args:
        ks: 一个包含多个 k 值的列表，例如 [1, 5, 10]
        
    Returns:
        dict: 形如 {1: tensor(11.9), 5: tensor(8.2), 10: tensor(5.4)}
    """
    if not ks:
        return {}

    B, T, V = logits.shape
    # 1. 🌟 算力节约：无论传入多少个 k，全量的 CE 只计算这一次！
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, T)
    
    # 2. 定位断点，并缓存时间步矩阵和有效 target 矩阵
    bp = prefill_mask.sum(dim=1, keepdim=True)
    t_idx = torch.arange(T, device=logits.device).view(1, -1)
    valid_targets_mask = (targets != ignore_index)
    
    results = {}
    # 3. 🌟 极速提取：只遍历纯布尔运算
    for k in ks:
        if k <= 0:
            results[k] = None
            continue
        # 组合掩码：≥断点 且 <断点+k 且 不是 Padding
        m = (t_idx >= bp) & (t_idx < bp + k) & valid_targets_mask
        if not m.any():
            results[k] = None
        else:
            results[k] = ce[m].mean()
    return results


# ==========================================
# CPT 主训练循环
# ==========================================
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
        # DATA
        dataset_name: str = "debug",
        dataset_dir: str = "data",
        data_dir: str = "data",
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
):

    # 1. set seeds
    set_random_seeds(42)
    # timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logger = TensorBoardLogger(root_dir=tensorboard_root, name=f"{expid}_{arch_name.replace('/', '-')}")
    loggers = [tb_logger] if enable_tensorboard else []
    
    # 2. 这里的 Fabric 逻辑保持不变...
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=num_devices, 
        num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)), # 兼容单机和多机
        strategy=DDPStrategy(
            timeout=timedelta(days=3650),
            find_unused_parameters=True if len(research_remove_order_str) > 0 else False
        ),  
        precision="bf16-true", 
        loggers=loggers
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
    fabric.print(f"🔄 正在从 {load_dir} 加载预训练 Checkpoint...")
    # 1. 先把原版权重字典加载到内存里
    raw_ckpt = torch.load(f"{load_dir}/lit_model.pth", map_location='cpu')
    state_dict = raw_ckpt['model'] if 'model' in raw_ckpt else raw_ckpt
    del raw_ckpt
    
    # 2. 🌟 核心拦截：Block 级别的映射与克隆
    if config.use_research and config.research_separate_parameter:
        fabric.print("🔀 检测到 Block 级参数独立！正在为 h_prefill 组装预训练权重...")
        prefill_weights = {}

        # 遍历配置中的 SWA 层列表
        # i 是在 h_prefill ModuleList 中的物理索引 (0, 1, 2...)
        # block_idx 是在原版 h 中的逻辑层号 (0, 2, 4...)
        for i, block_idx in enumerate(config.research_prefill_swa_layers):
            orig_prefix = f"transformer.h.{block_idx}."
            new_prefix = f"transformer.h_prefill.{i}."

            # 遍历寻找属于原版 block_idx 的所有权重，并改名挂载到 h_prefill 下
            for key, value in state_dict.items():
                if key.startswith(orig_prefix):
                    # 极其精准的前缀替换
                    new_key = key.replace(orig_prefix, new_prefix, 1)
                    if new_key not in state_dict.keys():
                        prefill_weights[new_key] = value

        # 将克隆出的 prefill 分支权重合并入主字典
        state_dict.update(prefill_weights)
        fabric.print(f"✅ 成功映射并注入了 {len(prefill_weights)} 个 Block 级别的张量！")

    # 3. 严格度降低，因为我们凭空造了全新的网络分支
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys) > 0:
        fabric.print(f"⚠️ 未加载的参数 (通常为 buffer): {missing_keys[:5]}...")
        
    fabric.print("✅ 真实权重加载成功！所有分支已完成 Pre-trained 初始化。")

    # ==========================================
    # 🌟 工业级优化器初始化：Weight Decay 分组过滤
    # ==========================================
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 核心逻辑：矩阵（2维及以上）加 WD，向量（1维）不加 WD
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

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
    model, optimizer = fabric.setup(model, optimizer)

    # 🌟 3. 初始化新的流式 Parquet 数据集
    # tokenizer = Tokenizer(checkpoint_dir)
    # ==========================================
    # 🌟 工业级多卡同步：预编译安全锁
    # ==========================================

    print(f"[Rank {fabric.global_rank}] 开始并行预编译数据...")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    bin_data_paths = prepare_data(
        data_dir=data_dir,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        model_dir=checkpoint_dir,
        rank=fabric.global_rank,
        local_rank=local_rank,
        world_size=fabric.world_size,
    )
    fabric.barrier()

    # 再做一次校验，确保所有 bin 都已经存在
    missing_bins = [p for p in bin_data_paths if not os.path.exists(p)]
    if missing_bins:
        raise RuntimeError(f"[Rank {fabric.global_rank}] 这些 bin 文件不存在: {missing_bins}")

    print(f"[Rank {fabric.global_rank}] 数据预编译完成，共 {len(bin_data_paths)} 个 bin 文件")

    datasets = [CPTBinDataset(bin_path=bp, seq_len=context_length) for bp in bin_data_paths]
    dataset = ConcatDataset(datasets)
    
    # 🌟 修改点：IterableDataset 不支持 shuffle=True 和 drop_last=True
    # 因为数据已经是流式了，我们在 prepare_data 阶段已经做过了全局 Shuffle
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, num_workers=num_workers, shuffle=True)
    dataloader = fabric.setup_dataloaders(dataloader)

    fabric.print("🚀 开始 Continue Pretraining...")
    model.train()

    epochs = num_epochs
    gradient_accumulation_steps = max(1, global_batch_size // (micro_batch_size * fabric.world_size))
    optimizer.zero_grad(set_to_none=True) 
    step_start_time = datetime.now()
    step_stats = MicroStepMeanStats()
    global_step = 0
    total_steps = max_steps
    training_finished = False
    
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            is_accumulating = (batch_idx + 1) % gradient_accumulation_steps != 0

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

                # ==========================================
                # 🌟 核心拦截：利用 Mask 屏蔽 Prefill 部分的 Loss
                # ==========================================
                # 假设你的 prefill_mask 中：True 表示 Prefill，False 表示 Decode
                masked_targets = targets.masked_fill(prefill_mask == True, -100)
                compariable_decode_loss = chunked_cross_entropy(logits, masked_targets, chunk_size=0)
                if use_research and not research_prefill_supervise:
                    loss = compariable_decode_loss
                else:
                    loss = chunked_cross_entropy(logits, targets, chunk_size=0)

                # 记录未缩放 loss，用于统计当前 global step 的平均训练损失
                metrics = {
                    "compariable_loss": compariable_decode_loss.detach().item(),
                }
                if research_decode_prefix_tokens > 0:
                    ks = [research_decode_prefix_tokens, 8, 16]
                    if 1 not in ks:
                        ks.insert(0, 1)
                    # 如果 weight 为 0，只需要观察指标而不需要梯度，使用 no_grad 加速
                    grad_ctx = torch.no_grad() if research_decode_prefix_loss_weight == 0 else nullcontext()
                    with grad_ctx:
                        prefix_losses = decode_prefix_mean_ce_multi_k(logits, targets, prefill_mask, ks=ks)
                    if len(prefix_losses) > 0:
                        for k, prefix_loss in prefix_losses.items():
                            metrics[f"prefix_loss_{k}"] = prefix_loss.detach().item()
                        if use_research and research_decode_prefix_loss_weight > 0:
                            loss += (prefix_losses[research_decode_prefix_tokens] * research_decode_prefix_loss_weight)
                metrics['loss'] = loss.detach().item()
                metrics['prefill_ratio'] = (prefill_mask.sum() / prefill_mask.numel()).item()
                step_stats.accumulate(**metrics)

                loss = loss / gradient_accumulation_steps
                fabric.backward(loss)

            if not is_accumulating:
                current_lr = get_lr(global_step, total_steps, warmup_steps, learning_rate, min_lr)
                # 遍历优化器里的每一个参数组 (包括我们刚才拆分的带 decay 和不带 decay 的组)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

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
                    f"Epoch {epoch+1} | Step {global_step + 1} | "
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
                    break
                
                if research_remove_interval > 0 and global_step % research_remove_interval == 0:
                    remove_index = global_step // research_remove_interval
                    if remove_index <= len(research_remove_order):
                        remove_index = research_remove_order[remove_index - 1]
                        config.research_removed_layers.append(remove_index)
                        fabric.print(f"🔥 已移除层 {remove_index}，当前所有已移除层为 {config.research_removed_layers}")
                
        if training_finished:
            break

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
    
    fabric.print("🎉 训练运行结束！")

if __name__ == "__main__":
    CLI(main)