import os
import shutil
import glob
# os.environ['https_proxy'] = '127.0.0.1:7897'
# os.environ['http_proxy'] = '127.0.0.1:7897'

import yaml
from dataclasses import asdict
import torch
import lightning as L
from datetime import datetime
from torch.utils.data import DataLoader, IterableDataset
from litgpt import Config
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
from litgpt.utils import chunked_cross_entropy
from jsonargparse import CLI
from datasets import load_dataset, concatenate_datasets, Dataset
import pyarrow.parquet as pq

# ==========================================
# 🌟 全新升级：基于 Parquet 的流式数据集
# ==========================================
class CPTParquetIterableDataset(IterableDataset):
    """
    流式读取 Parquet 文件，动态 Tokenize 并打包为固定长度 (seq_len)。
    完美兼容 10MB 的 Debug 数据和 10TB 的全量数据，内存占用极低。
    """
    def __init__(self, parquet_path: str, tokenizer: Tokenizer, seq_len: int):
        self.parquet_path = parquet_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.chunk_size = seq_len + 1  # +1 为了错位生成 inputs 和 targets

    def __iter__(self):
        # 建立一个 Token 缓冲区
        buffer = torch.tensor([], dtype=torch.long)
        parquet_file = pq.ParquetFile(self.parquet_path)

        # 流式读取：每次只加载 8192 行数据到内存
        for batch in parquet_file.iter_batches(batch_size=8192, columns=["text"]):
            for text in batch.to_pandas()["text"]:
                # Tokenize: CPT 的黄金法则 -> bos=False, eos=True，隔断两篇文章的上下文
                tokens = self.tokenizer.encode(text, bos=False, eos=True)
                
                # 将新生成的 token 追加进缓冲区
                buffer = torch.cat([buffer, tokens])

                # 只要缓冲区满了，就切出一个完整的 chunk 喂给模型
                while buffer.size(0) >= self.chunk_size:
                    chunk = buffer[:self.chunk_size]
                    # 缓冲区丢弃已被读取的部分
                    buffer = buffer[self.chunk_size:]
                    
                    # 返回 inputs (0 到 N-1) 和 targets (1 到 N)
                    yield chunk[:-1], chunk[1:]
                    
        parquet_file.close()

torch.set_float32_matmul_precision('high')

def prepare_data(data_dir, dataset_name):
    print("⏳ 正在下载和准备数据...")
    os.makedirs(data_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, f"{dataset_name}.parquet")
    
    # 如果 Parquet 文件已经存在，直接跳过下载 (方便反复 Debug)
    if os.path.exists(parquet_path):
        print(f"📦 发现已存在的数据集: {parquet_path}，直接使用！")
        return parquet_path

    if dataset_name == 'debug':
        print("🌐 正在拉取完全开源的 FineWeb-Edu (跳过所有权限验证)...")
        
        # 只用不需要任何认证的 FineWeb，流式读取
        eng_stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        print("📥 正在极速抽取 10000 条长文用于架构 Debug...")
        eng_list = list(eng_stream.take(10000))
        
        mixed_dataset = Dataset.from_list(eng_list)
        mixed_dataset = mixed_dataset.select_columns(["text"])

        print(f"💾 正在导出为高性能 Parquet 格式至 {parquet_path}...")
        mixed_dataset.to_parquet(parquet_path)
        return parquet_path
    else:
        raise NotImplementedError(f"数据集 {dataset_name} 的准备逻辑尚未实现")

# ==========================================
# CPT 主训练循环
# ==========================================
def main(
        # MODEL
        arch_name: str = "Qwen/Qwen3-0.6B-Base",
        context_length: int = 4096,
        # TRAINING
        global_batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 2,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.1,
        # DATA
        dataset_name: str = "debug",
        dataset_dir: str = "data",
        num_workers: int = 0, # 🌟 流式读取暂设为 0，防止多进程读取重复数据
        # IO
        save_ckpt: bool = False,
        save_path: str = "./ckpt/cpt",
        # RESEARCH
        use_research: bool = False,
        research_swa_size: int = 512,
        research_prefill_swa_layers: list[int] = [1,3,5,7,9,11,13,15,17,19,21,23,25,27],
        research_prefill_identity_layers: list[int] = [2,4,6,8,10,12,14,16,18,20,22,24,26,28],
        research_breakpoint_schedule: str | None = None,
):
    # 1. 准备数据并获取 Parquet 路径
    parquet_path = prepare_data(dataset_dir, dataset_name)
    
    # 2. 这里的 Fabric 逻辑保持不变...
    fabric = L.Fabric(accelerator="cuda", devices=2, precision="bf16-true")
    fabric.launch()

    config = Config.from_name(arch_name) 
    assert config is not None
    config.block_size = context_length
    config.use_research = use_research
    config.research_swa_size = research_swa_size
    config.research_prefill_swa_layers = research_prefill_swa_layers
    config.research_prefill_identity_layers = research_prefill_identity_layers
    config.research_breakpoint_schedule = research_breakpoint_schedule
    fabric.print(f"⚙️ 模型 Config 初始化完成: {config.name}")

    with fabric.init_module(empty_init=True):
        model = GPT(config)

    checkpoint_dir = f"checkpoints/{arch_name}"
    fabric.print(f"🔄 正在从 {checkpoint_dir} 加载预训练 Checkpoint...")
    model.load_state_dict(torch.load(f"{checkpoint_dir}/lit_model.pth"), strict=False)
    fabric.print("✅ 真实权重加载成功！")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    # 🌟 3. 初始化新的流式 Parquet 数据集
    tokenizer = Tokenizer(checkpoint_dir)
    dataset = CPTParquetIterableDataset(
        parquet_path=parquet_path,
        tokenizer=tokenizer,
        seq_len=config.block_size,
    )
    
    # 🌟 修改点：IterableDataset 不支持 shuffle=True 和 drop_last=True
    # 因为数据已经是流式了，我们在 prepare_data 阶段已经做过了全局 Shuffle
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, num_workers=num_workers)
    dataloader = fabric.setup_dataloaders(dataloader)

    fabric.print("🚀 开始 Continue Pretraining...")
    model.train()

    epochs = num_epochs
    gradient_accumulation_steps = max(1, global_batch_size // (micro_batch_size * fabric.world_size))
    optimizer.zero_grad(set_to_none=True) 
    step_start_time = datetime.now()
    global_step_loss_sum = 0.0
    global_step_micro_count = 0
    
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            is_accumulating = (batch_idx + 1) % gradient_accumulation_steps != 0

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(inputs)
                loss = chunked_cross_entropy(logits, targets, chunk_size=0)

                # 记录未缩放 loss，用于统计当前 global step 的平均训练损失
                global_step_loss_sum += loss.detach().item()
                global_step_micro_count += 1

                loss = loss / gradient_accumulation_steps
                fabric.backward(loss)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                now = datetime.now()
                step_time = (now - step_start_time).total_seconds()
                step_start_time = now
                global_step_loss = global_step_loss_sum / max(1, global_step_micro_count)
                fabric.print(
                    f"[{now.strftime('%H:%M:%S')}] "
                    f"Epoch {epoch+1} | Global Step {batch_idx//gradient_accumulation_steps} | "
                    f"Loss: {global_step_loss:.4f} | "
                    f"Step Time: {step_time:.2f}s"
                )
                global_step_loss_sum = 0.0
                global_step_micro_count = 0

    if save_ckpt:
        os.makedirs(save_path, exist_ok=True)
        fabric.print(f"💾 正在保存模型至 {save_path}")
        state_dict = {"model": model.state_dict()}
        fabric.save(f"{save_path}/lit_model.pth", state_dict)

        for file_path in glob.glob(f"{checkpoint_dir}/*.json") + glob.glob(f"{checkpoint_dir}/*.model"):
            shutil.copy(file_path, save_path)
        with open(f"{save_path}/model_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(asdict(config), f)
            
        fabric.print(f"📦 Tokenizer 和 Config 已自动同步至 {save_path}")
    
    fabric.print("🎉 训练运行结束！")

if __name__ == "__main__":
    CLI(main)