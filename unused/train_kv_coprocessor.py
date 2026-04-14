import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from datasets import load_dataset
import os
import glob

# ==========================================
# 0. 全局配置
# ==========================================
PATH_0_6B = "checkpoints/Qwen/Qwen3-0.6B-Base"
PATH_1_7B = "checkpoints/Qwen/Qwen3-1.7B-Base"
# 替换为你的 parquet 文件所在的目录
DATA_DIR = "/home/ma-user/work/bucket-wulan-green/zhaoyusheng/fineweb-edu/sample/10BT/" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 1024       # 每次喂给模型的上下文长度
MAX_STEPS = 50000    # 最大训练步数
LOG_INTERVAL = 10    # 多少步打印一次日志

# ==========================================
# 1. 流式数据加载器 (Streaming DataLoader)
# ==========================================
def get_data_stream(data_dir, tokenizer_path, seq_len):
    """
    流式读取 Parquet 文件，动态进行 Tokenize 并切分成固定长度的块
    """
    print(f"Initializing Tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(tokenizer_path)
    
    # 查找所有的 parquet 文件
    data_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not data_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    print(f"Found {len(data_files)} parquet files. Starting streaming...")

    # 使用 HuggingFace datasets 的 streaming 模式，内存开销极小
    dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

    buffer = []
    for example in dataset:
        # 注意：这里的 'text' 是列名。
        # 如果你的 parquet 文件里文本列叫其他名字（比如 'content'），请修改这里
        text = example.get("text", "") 
        if not text:
            continue
        
        # 将文本编码为 token
        tokens = tokenizer.encode(text, device=DEVICE).tolist()
        buffer.extend(tokens)

        # 当 buffer 里的 token 数量凑够一个 seq_len 时，吐出一个 Batch
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            buffer = buffer[seq_len:] # 截断 buffer
            
            # 组装为 batch_size=1 的 tensor: [1, seq_len]
            input_ids = torch.tensor(chunk, dtype=torch.long, device=DEVICE).unsqueeze(0)
            yield input_ids

# ==========================================
# 2. 模型加载与外科手术
# ==========================================
def load_model(path, requires_grad=False):
    config = Config.from_file(os.path.join(path, "model_config.yaml"))
    model = GPT(config)
    model.load_state_dict(torch.load(os.path.join(path, "lit_model.pth"), map_location="cpu"))
    model.requires_grad_(requires_grad)
    return model.to(DEVICE)

print("Loading Models...")
teacher = load_model(PATH_1_7B, requires_grad=False).eval()
student = load_model(PATH_0_6B, requires_grad=True).train()

print("Replacing the LM Head of 0.6B with Identity...")
student.lm_head = nn.Identity()

optimizer = AdamW(student.parameters(), lr=2e-5)

# ==========================================
# 3. 核心训练步 (我们跑通的逻辑)
# ==========================================
def train_step(input_ids, alpha):
    optimizer.zero_grad()
    seq_len = input_ids.size(1)
    input_pos = torch.arange(0, seq_len, device=DEVICE)

    # Phase 1: Teacher 基准
    teacher.set_kv_cache(batch_size=1, max_seq_length=seq_len, device=DEVICE)
    with torch.no_grad():
        golden_logits = teacher(input_ids, input_pos=input_pos)
        target_logits = golden_logits[:, -1, :] 
        teacher_vs = [block.attn.kv_cache.v.clone() for block in teacher.transformer.h]

    # Phase 2: Student 跑出带有梯度的 KV
    student.set_kv_cache(batch_size=1, max_seq_length=seq_len, device=DEVICE)
    _ = student(input_ids, input_pos=input_pos) 

    # Phase 3: 图拼接与单步验证
    teacher.set_kv_cache(batch_size=1, max_seq_length=seq_len, device=DEVICE)
    
    cos_loss = 0.0
    for i in range(len(teacher.transformer.h)):
        student_k = student.transformer.h[i].attn.kv_cache.k
        student_v = student.transformer.h[i].attn.kv_cache.v
        
        # 写入克隆体，保护梯度！
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

    # Phase 4: Loss 计算
    kl_loss = F.kl_div(
        F.log_softmax(pred_logits, dim=-1),
        F.softmax(target_logits, dim=-1),
        reduction='batchmean'
    )

    total_loss = kl_loss + alpha * (cos_loss / len(teacher.transformer.h))
    
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), kl_loss.item(), cos_loss.item()

# ==========================================
# 4. 主训练循环 (Training Loop)
# ==========================================
if __name__ == "__main__":
    print(f"Starting Training Loop with Real Data from {DATA_DIR}...")
    
    # 获取数据流迭代器
    data_stream = get_data_stream(DATA_DIR, PATH_0_6B, SEQ_LEN)
    
    step = 0
    # 动态衰减的 alpha (余弦方向损失的权重)
    # 初始为 0.1，在 MAX_STEPS 步内线性衰减到 0.01
    start_alpha = 0.1
    end_alpha = 0.01

    try:
        for input_ids in data_stream:
            if step >= MAX_STEPS:
                print("Reached maximum steps. Stopping training.")
                break
            
            # 计算当前的 alpha
            current_alpha = start_alpha - (start_alpha - end_alpha) * (step / MAX_STEPS)
            
            loss, kl, cos = train_step(input_ids, alpha=current_alpha)
            
            if step % LOG_INTERVAL == 0:
                print(f"Step {step:05d} | Total Loss: {loss:.4f} | KL: {kl:.4f} | V-Cos: {cos:.4f} | Alpha: {current_alpha:.4f}")
            
            # 定期保存 Checkpoint (比如每 5000 步)
            if step > 0 and step % 5000 == 0:
                save_path = f"student_coprocessor_step_{step}.pth"
                torch.save(student.state_dict(), save_path)
                print(f"Checkpoint saved to {save_path}")

            step += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        torch.save(student.state_dict(), "student_coprocessor_interrupted.pth")
    
    print("Training Finished!")