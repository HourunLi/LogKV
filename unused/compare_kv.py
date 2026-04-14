import torch
import torch.nn.functional as F
from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
import os

# --- 配置路径 ---
# 请确保当前工作目录在 litgpt 的根目录下
PATH_0_6B = "checkpoints/Qwen/Qwen3-0.6B-Base"
PATH_1_7B = "checkpoints/Qwen/Qwen3-1.7B-Base"
PROMPT = "The theory of relativity, proposed by Albert Einstein, fundamentally changed our understanding of"

def load_model_and_get_kv(checkpoint_dir, input_ids):
    print(f"Loading model from {checkpoint_dir}...")
    
    # 1. 加载配置和实例化模型
    config = Config.from_file(os.path.join(checkpoint_dir, "model_config.yaml"))
    model = GPT(config)
    
    # 2. 加载权重
    state_dict_path = os.path.join(checkpoint_dir, "lit_model.pth")
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    model.eval()
    
    # 3. 初始化 KV Cache
    seq_len = input_ids.size(1)
    model.set_kv_cache(batch_size=1, max_seq_length=seq_len)
    
    # 4. 前向传播 (必须传入 input_pos 触发 Cache 写入)
    # LitGPT 需要明确的位置信息，才会把当前计算的 KV 存入对应的显存块
    input_pos = torch.arange(0, seq_len, device=input_ids.device)
    
    with torch.no_grad():
        _ = model(input_ids, input_pos=input_pos)
        
    # 5. 提取 KV Cache
    # LitGPT 的模块树结构是: model.transformer.h[layer_idx]
    k_caches = []
    v_caches = []
    for block in model.transformer.h:
        # litgpt 的 kv_cache 是一个专门的类，内部存着 k 和 v 张量
        k_caches.append(block.attn.kv_cache.k.clone())
        v_caches.append(block.attn.kv_cache.v.clone())
        
    return k_caches, v_caches

def main():
    # 检查路径
    if not os.path.exists(PATH_0_6B) or not os.path.exists(PATH_1_7B):
        raise FileNotFoundError("找不到 Checkpoint 目录，请检查路径。")

    # 初始化 Tokenizer (两者同源，用 0.6B 的即可)
    tokenizer = Tokenizer(PATH_0_6B)
    input_ids = tokenizer.encode(PROMPT).unsqueeze(0) # 加上 batch 维度: [1, seq_len]
    print(f"Prompt tokens length: {input_ids.size(1)}")

    # 获取 Cache
    k_06, v_06 = load_model_and_get_kv(PATH_0_6B, input_ids)
    k_17, v_17 = load_model_and_get_kv(PATH_1_7B, input_ids)

    n_layers = len(k_06)
    
    print("\n" + "="*50)
    print("逐层 KV Cache 差异对比 (0.6B vs 1.7B)")
    print("="*50)
    print(f"{'Layer':<6} | {'K-MSE':<10} | {'K-CosSim':<10} | {'V-MSE':<10} | {'V-CosSim':<10}")
    print("-" * 50)

    for i in range(n_layers):
        # 取出第 i 层的 KV 张量
        # 把张量展平 (flatten) 以便计算整体的相似度和误差
        k0_flat = k_06[i].flatten().float()
        k1_flat = k_17[i].flatten().float()
        v0_flat = v_06[i].flatten().float()
        v1_flat = v_17[i].flatten().float()

        # 计算 MSE (均方误差)
        k_mse = F.mse_loss(k0_flat, k1_flat).item()
        v_mse = F.mse_loss(v0_flat, v1_flat).item()

        # 计算余弦相似度 (Cosine Similarity)
        # 1.0 表示方向完全一致，0 表示正交，-1 表示方向完全相反
        k_cos = F.cosine_similarity(k0_flat.unsqueeze(0), k1_flat.unsqueeze(0)).item()
        v_cos = F.cosine_similarity(v0_flat.unsqueeze(0), v1_flat.unsqueeze(0)).item()

        print(f"{i:<6} | {k_mse:<10.4f} | {k_cos:<10.4f} | {v_mse:<10.4f} | {v_cos:<10.4f}")

    print("="*50)

if __name__ == "__main__":
    main()