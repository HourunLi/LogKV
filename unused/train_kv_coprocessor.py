import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from datasets import load_dataset
import os
import glob

# ==========================================
# 0. DDP 初始化与全局配置
# ==========================================
# 初始化分布式环境
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
DEVICE = f"cuda:{local_rank}"

PATH_0_6B = "checkpoints/Qwen/Qwen3-0.6B-Base"
PATH_1_7B = "checkpoints/Qwen/Qwen3-1.7B-Base"
DATA_DIR = "/home/ma-user/work/bucket-wulan-green/zhaoyusheng/fineweb-edu/sample/10BT/"

SEQ_LEN = 1024       
MAX_STEPS = 50000    
LOG_INTERVAL = 10    
ACCUM_STEPS = 4      # 梯度累积步数。等效全局 Batch Size = 8卡 * 4累积 = 32

if global_rank == 0:
    print(f"[*] DDP Initialized. World Size: {world_size}, Global Batch Size: {world_size * ACCUM_STEPS}")

# ==========================================
# 1. 流式数据加载器 (带 DDP 分片支持)
# ==========================================
def get_data_stream(data_dir, tokenizer_path, seq_len, rank, world_size):
    if rank == 0:
        print(f"[*] Initializing Tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(tokenizer_path)
    
    data_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not data_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    # 流式加载
    dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    
    # 【核心修改】：通过 shard 将数据均分给 8 张卡，确保每张卡吃到的数据不同！
    dataset = dataset.shard(num_shards=world_size, index=rank)

    buffer = []
    for example in dataset:
        text = example.get("text", "") 
        if not text:
            continue
        
        tokens = tokenizer.encode(text, device=DEVICE).tolist()
        buffer.extend(tokens)

        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            buffer = buffer[seq_len:] 
            input_ids = torch.tensor(chunk, dtype=torch.long, device=DEVICE).unsqueeze(0)
            yield input_ids

# ==========================================
# 2. 模型加载与 DDP 包装
# ==========================================
def load_model(path, requires_grad=False):
    config = Config.from_file(os.path.join(path, "model_config.yaml"))
    model = GPT(config)
    model.load_state_dict(torch.load(os.path.join(path, "lit_model.pth"), map_location="cpu"))
    model.requires_grad_(requires_grad)
    return model.to(DEVICE)

teacher = load_model(PATH_1_7B, requires_grad=False).eval()
student = load_model(PATH_0_6B, requires_grad=True).train()

# 外科手术：阉割 LM Head
student.lm_head = nn.Identity()

# 【核心修改】：将 Student 包装进 DDP。Teacher 是冻结的，无需包装 DDP，节约通信带宽。
student = DDP(student, device_ids=[local_rank])

optimizer = AdamW(student.parameters(), lr=2e-5)

# ==========================================
# 3. 核心计算步 (解耦 Backward)
# ==========================================
def compute_loss(input_ids, alpha):
    """
    计算前向传播和 Loss，但不包含 backward()，以便外层处理梯度累积
    """
    seq_len = input_ids.size(1)
    input_pos = torch.arange(0, seq_len, device=DEVICE)

    teacher.set_kv_cache(batch_size=1, max_seq_length=seq_len, device=DEVICE)
    with torch.no_grad():
        golden_logits = teacher(input_ids, input_pos=input_pos)
        target_logits = golden_logits[:, -1, :] 
        teacher_vs = [block.attn.kv_cache.v.clone() for block in teacher.transformer.h]

    # 【核心修改】：DDP 包装后，需要通过 .module 访问内部方法和属性
    student.module.set_kv_cache(batch_size=1, max_seq_length=seq_len, device=DEVICE)
    _ = student(input_ids, input_pos=input_pos) 

    teacher.set_kv_cache(batch_size=1, max_seq_length=seq_len, device=DEVICE)
    
    cos_loss = 0.0
    for i in range(len(teacher.transformer.h)):
        # 同样需要通过 .module 提取 Cache
        student_k = student.module.transformer.h[i].attn.kv_cache.k
        student_v = student.module.transformer.h[i].attn.kv_cache.v
        
        teacher.transformer.h[i].attn.kv_cache.k = student_k.clone()
        teacher.transformer.h[i].attn.kv_cache.v = student_v.clone()
        
        cos_loss += 1.0 - F.cosine_similarity(
            student_v.flatten().unsqueeze(0), 
            teacher_vs[i].flatten().unsqueeze(0)
        ).mean()

    last_token = input_ids[:, -1:]
    last_pos = input_pos[-1:]
    
    hybrid_logits = teacher(last_token, input_pos=last_pos) 
    pred_logits = hybrid_logits[:, -1, :]

    kl_loss = F.kl_div(
        F.log_softmax(pred_logits, dim=-1),
        F.softmax(target_logits, dim=-1),
        reduction='batchmean'
    )

    # 之前建议的修复：不除以层数，让余弦损失权重更明显
    total_loss = kl_loss + alpha * cos_loss
    
    return total_loss, kl_loss, cos_loss

# ==========================================
# 4. 主训练循环 (Training Loop)
# ==========================================
if __name__ == "__main__":
    data_stream = get_data_stream(DATA_DIR, PATH_0_6B, SEQ_LEN, global_rank, world_size)
    
    step = 0
    start_alpha = 0.1
    end_alpha = 0.01

    optimizer.zero_grad()

    try:
        for input_ids in data_stream:
            if step >= MAX_STEPS:
                if global_rank == 0:
                    print("[*] Reached maximum steps. Stopping training.")
                break
            
            current_alpha = start_alpha - (start_alpha - end_alpha) * (step / MAX_STEPS)
            
            # 判断是否还在累积阶段
            is_accumulating = (step + 1) % ACCUM_STEPS != 0

            # 【核心修改】：在累积梯度时不进行卡间同步 (no_sync)，只有最后一步才触发 All-Reduce，极大提升效率
            with student.no_sync() if is_accumulating else nullcontext():
                total_loss, kl_loss, cos_loss = compute_loss(input_ids, alpha=current_alpha)
                
                # 缩放 loss 并反向传播
                (total_loss / ACCUM_STEPS).backward()

            if not is_accumulating:
                # 梯度裁剪，防止偶尔的 NaN 或爆炸炸毁权重
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # 仅在主进程 (Rank 0) 上打印日志，保持终端清爽
                if global_rank == 0:
                    real_update_step = (step + 1) // ACCUM_STEPS
                    if real_update_step % LOG_INTERVAL == 0:
                        print(f"Update {real_update_step:05d} | Total Loss: {total_loss.item():.4f} | KL: {kl_loss.item():.4f} | V-Cos: {cos_loss.item():.4f} | Alpha: {current_alpha:.4f}")
                    
                    # 定期保存 (仅 Rank 0 保存，避免冲突)
                    if real_update_step > 0 and real_update_step % 5000 == 0:
                        save_path = f"student_coprocessor_update_{real_update_step}.pth"
                        # 保存去壳后的原始权重 (.module)
                        torch.save(student.module.state_dict(), save_path)
                        print(f"[*] Checkpoint saved to {save_path}")

            step += 1

    except KeyboardInterrupt:
        if global_rank == 0:
            print("\n[*] Training interrupted. Saving final checkpoint...")
            torch.save(student.module.state_dict(), "student_coprocessor_interrupted.pth")
    
    # 销毁进程组
    dist.destroy_process_group()
    if global_rank == 0:
        print("[*] Training Finished!")