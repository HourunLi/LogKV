from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from typing import Any

from jsonargparse import CLI
import tqdm

from utils import *

if 'HF_DATASETS_CACHE' not in os.environ and 'PKU' not in os.environ:
    print("设置环境变量...")

    BASE = '/home/ma-user/work/bucket-wulan-green/wubohan/data/hf_cache'
    os.environ['HF_HOME'] = BASE
    os.environ['HF_DATASETS_CACHE'] = f'{BASE}/hf_cache'
    os.environ['HF_EVALUATE_CACHE'] = f'{BASE}/evaluate'
    os.environ['HF_MODULES_CACHE'] = f'{BASE}/modules'
    os.environ['HUGGINGFACE_HUB_CACHE'] = f'{BASE}/hub'
    os.environ['HF_HUB_CACHE'] = f'{BASE}/hub'
    os.environ['NLTK_DATA'] = f'{BASE}/nltk_data'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE'] = '0'
    os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import dataclasses

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
from lm_eval import evaluator
from lm_eval.api.model import LM
from litgpt.generate.base import generate as litgpt_generate

_CONFIG_FIELDS = {f.name for f in dataclasses.fields(Config)}


def _load_lit_model_checkpoint(checkpoint_dir: str, map_location: str | torch.device) -> Any:
    """加载 ``{checkpoint_dir}/lit_model.pth``（单文件 ``torch.save``，与 demo FSDP ``state_dict_type='full'`` 一致）。"""
    lit_path = Path(checkpoint_dir).expanduser() / "lit_model.pth"
    if not lit_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {lit_path}")
    if lit_path.is_dir():
        raise FileNotFoundError(
            f"{lit_path} 为目录（旧版 FSDP/DCP 分片）。请先在仓库根目录对同一路径调用 "
            "`utils.convert_and_replace_fsdp_ckpt` 合并为单文件 lit_model.pth，再跑 eval。"
        )
    return torch.load(str(lit_path), map_location=map_location, weights_only=False)


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _filter_config_dict(d: dict[str, Any]) -> dict[str, Any]:
    dropped = sorted(k for k in d if k not in _CONFIG_FIELDS)
    if dropped:
        print(f"[eval] 过滤掉非 Config 字段: {dropped}")
    return {k: v for k, v in d.items() if k in _CONFIG_FIELDS}


def _normalize_training_config_dict(d: dict[str, Any]) -> tuple[dict[str, Any], list[int] | None]:
    """训练 exp / CLI 里常用 research_*_layers_str；Config 只接受 research_prefill_swa_layers 等。"""
    out = dict(d)
    identity_layers: list[int] | None = None
    if "research_identity_layers_str" in out:
        identity_layers = _parse_csv_ints(out.pop("research_identity_layers_str"))
    if "research_swa_layers_str" in out:
        out["research_prefill_swa_layers"] = _parse_csv_ints(out.pop("research_swa_layers_str"))
    return out, identity_layers


def _config_from_yaml_and_overrides(config_path: str, overrides: dict[str, Any] | None) -> Config:
    with open(config_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if base is None:
        raise ValueError(f"{config_path} is empty or invalid YAML.")
    merged = {**base, **(overrides or {})}
    merged, identity_layers = _normalize_training_config_dict(merged)
    merged = _filter_config_dict(merged)
    cfg = Config(**merged)
    if identity_layers is not None:
        cfg.research_prefill_identity_layers = identity_layers
    return cfg


class CustomResearchLM(LM):
    def __init__(
        self,
        checkpoint_dir: str,
        device="cuda",
        use_research: bool = True,
        map_branch=False,
        config_overrides: dict[str, Any] | None = None,
    ):
        super().__init__()
        self._device = device
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = Tokenizer(checkpoint_dir)
        
        # 控制打印：在多卡下尽量只让主进程打印，防止刷屏
        is_master = not dist.is_initialized() or dist.get_rank() == 0

        # ==========================================
        # 🌟 智能 Config 路由加载 (YAML 版本)
        # ==========================================
        # 试探两种最常见的 yaml 命名方式
        config_path = os.path.join(checkpoint_dir, "model_config.yaml")
        
        if os.path.exists(config_path):
            if is_master: print(f"📄 发现专属架构 YAML 配置文件: {config_path}，正在自动同步架构...")
            # 合并 YAML + overrides，并把 research_*_layers_str 转成 Config 合法字段
            self.config = _config_from_yaml_and_overrides(config_path, config_overrides)
            
            # 确保对象的开关属性被正确覆盖
            self.use_research = getattr(self.config, 'use_research', False)

        else:
            if is_master: print("⚠️ 未发现训练期保存的 YAML 配置文件，正在使用备用参数初始化...")
            self.use_research = use_research
            fallback_kw: dict[str, Any] = dict(
                name=checkpoint_dir.split("/")[-1],
                use_research=use_research,
                research_separate_parameter=True if use_research else False,
                research_swa_layers_str="0,2,4,6,8,10,12,14,16,18,20,22,24,26",
                research_identity_layers_str="1,3,5,7,9,11,13,15,17,19,21,23,25,27",
            )
            if config_overrides:
                fallback_kw.update(config_overrides)
            fallback_kw, identity_layers = _normalize_training_config_dict(fallback_kw)
            fallback_kw = _filter_config_dict(fallback_kw)
            self.config = Config.from_name(**fallback_kw)
            if identity_layers is not None:
                self.config.research_prefill_identity_layers = identity_layers

        # ==========================================
        
        if is_master: print(f"🔧 正在初始化 Transformer (Research模式: {self.use_research})...")
        self.model = GPT(self.config).to(device).bfloat16()
        
        if is_master: print(f"🔄 正在加载权重...")
        checkpoint = _load_lit_model_checkpoint(checkpoint_dir, map_location=device)
        
        # 🌟 核心修复：检查是不是被包裹过的 checkpoint 字典
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            if is_master: print("📦 检测到 Fabric Checkpoint，已自动提取 model 权重。")
        else:
            state_dict = checkpoint
        
        if map_branch and self.config.use_research and self.config.research_separate_parameter:
            print("🔀 检测到 Block 级参数独立！正在为 h_prefill 组装预训练权重...")
            prefill_weights = {}

            # 遍历配置中的 SWA 层列表
            # i 是在 h_prefill ModuleList 中的物理索引 (0, 1, 2...)
            # block_idx 是在原版 h 中的逻辑层号 (0, 2, 4...)
            for i, block_idx in enumerate(self.config.research_prefill_swa_layers):
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
            print(f"✅ 成功映射并注入了 {len(prefill_weights)} 个 Block 级别的张量！")
            
        # 🌟 强烈建议：捕获并打印一下加载结果，看看是不是真的加载成功了！
        load_result = self.model.load_state_dict(state_dict, strict=False) 
        
        if is_master: 
            print(f"✅ 权重加载完毕！缺失的 keys: {len(load_result.missing_keys)} 个")
            # 如果 missing_keys 极其多（比如几百个），说明加载又失败了
            
        self.model.eval()

    # ==========================================
    # 🌟 分布式结果收集
    # ==========================================
    def all_gather_results(self, local_result_list: list):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return local_result_list
            
        dp_size = dist.get_world_size()
        all_results_list = [None for _ in range(dp_size)]
        dist.all_gather_object(all_results_list, local_result_list)
        
        final_results = []
        max_load = max(len(r) for r in all_results_list)
        for i in range(max_load):
            for rank_id in range(dp_size):
                if i < len(all_results_list[rank_id]):
                    final_results.append(all_results_list[rank_id][i])
        return final_results

    # ==========================================
    # 🌟 核心 1：PPL 与 选择题评测 (完美适配你的 forward + 并行)
    # ==========================================
    def loglikelihood(self, requests):
        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        dp_size = dist.get_world_size() if dist.is_initialized() else 1
        
        local_requests = requests[dp_rank::dp_size]
        results = []
        disable_tqdm = (dp_rank != 0)
        
        for req in tqdm.tqdm(local_requests, desc=f'Rank {dp_rank}', position=dp_rank, disable=disable_tqdm):
            context, continuation = req.args[0], req.args[1]
            
            ctx_enc = self.tokenizer.encode(context).tolist()
            # Llama 3 等 use_bos=True：若对 continuation 再 encode 一次会多一个 BOS，拼接后破坏
            # loglikelihood 对齐；Qwen 通常无 BOS，bos=False 与默认行为一致。
            cont_enc = self.tokenizer.encode(continuation, bos=False).tolist()
            
            # 🌟 安全阀：如果 题干 + 选项 > 4096，必须切掉题干最前面的部分
            max_len = self.model.max_seq_length
            if len(ctx_enc) + len(cont_enc) > max_len:
                # 保留完整的选项，切断 context 的头部
                keep_ctx_len = max_len - len(cont_enc)
                ctx_enc = ctx_enc[-keep_ctx_len:]
                print(f"⚠️ [Rank {dp_rank}] 警告: 触发截断，剩余 context 长度: {len(ctx_enc)}")
            
            if len(ctx_enc) == 0:
                ctx_enc = [self.tokenizer.bos_id]
                
            inps = torch.tensor([ctx_enc + cont_enc], dtype=torch.long, device=self._device)
            
            seq_len = inps.size(1)
            ctx_len = len(ctx_enc)
            
            # 🌟 关键修改：生成 [B, T] 的 2D Mask，你的 forward 里有 unsqueeze(-1)！
            mask = torch.zeros((1, seq_len), dtype=torch.bool, device=self._device)
            # 题干设为 True (Prefill/SWA/LayerDrop)
            mask[0, :ctx_len - 1] = True 
            # 选项默认为 False (Decode/Full Attention/Cross-layer KV)

            with torch.no_grad():
                # 为了防止长度爆炸，必须显式调用 set_kv_cache (哪怕这部分是一次性算完的)
                self.model.set_kv_cache(batch_size=1, max_seq_length=seq_len, device=self._device)
                if self.use_research:
                    logits = self.model(inps, prefill_mask=mask)
                else:
                    logits = self.model(inps)
                self.model.clear_kv_cache()
                
            cont_logits = logits[0, ctx_len - 1 : seq_len - 1]
            cont_targets = torch.tensor(cont_enc, dtype=torch.long, device=self._device)
            
            log_probs = F.log_softmax(cont_logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=cont_targets.unsqueeze(-1)).squeeze(-1)
            
            is_greedy = (cont_logits.argmax(dim=-1) == cont_targets).all().item()
            results.append((token_log_probs.sum().item(), is_greedy))
            
        return self.all_gather_results(results)

    # ==========================================
    # 🌟 核心 2：自回归生成任务 (LongBench)
    # ==========================================
    def generate_until(self, requests):
        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        dp_size = dist.get_world_size() if dist.is_initialized() else 1
        
        local_requests = requests[dp_rank::dp_size]
        results = []
        disable_tqdm = (dp_rank != 0)
        
        for req in tqdm.tqdm(local_requests, desc=f'Rank {dp_rank}', position=dp_rank, disable=disable_tqdm):
            prompt = req.args[0]
            gen_args = req.args[1] 
            
            # 解析 lm-eval 传来的生成参数
            max_new_tokens = gen_args.get("until", [self.tokenizer.eos_id])
            if isinstance(max_new_tokens, list):
                max_new_tokens = 256 # 如果传的是 stop words 列表，给个默认最大长度
            else:
                max_new_tokens = gen_args.get("max_length", 256)
                
            prompt_tensor = self.tokenizer.encode(prompt, device=self._device)

            # 🌟 安全阀：为生成的新 Token 预留空间
            max_len = self.model.max_seq_length
            if prompt_tensor.size(0) + max_new_tokens > max_len:
                keep_prompt_len = max_len - max_new_tokens
                prompt_tensor = prompt_tensor[-keep_prompt_len:]
                print(f"⚠️ [Rank {dp_rank}] 警告: 触发生成截断，Prompt 被切至: {keep_prompt_len}")
            
            total_max_len = prompt_tensor.size(0) + max_new_tokens
            with torch.no_grad():
                # 自回归路径带 input_pos，必须初始化 mask_cache / KVCache（与 litgpt/generate/base.py main 一致）
                self.model.set_kv_cache(batch_size=1, max_seq_length=total_max_len, device=self._device)
                try:
                    out = litgpt_generate(
                        self.model,
                        prompt_tensor,
                        max_returned_tokens=total_max_len,
                        temperature=gen_args.get("temperature", 1.0),
                        top_k=gen_args.get("top_k", None),
                        eos_id=self.tokenizer.eos_id,
                    )
                finally:
                    self.model.clear_kv_cache()

            # 截取新生成的部分并解码
            generated_tokens = out[prompt_tensor.size(0):]
            decoded = self.tokenizer.decode(generated_tokens)
            results.append(decoded)
            
        return self.all_gather_results(results)

    def loglikelihood_rolling(self, requests): pass
    @property
    def eot_token_id(self): return self.tokenizer.eos_id
    @property
    def max_length(self): return self.model.max_seq_length
    @property
    def max_gen_toks(self): return 256
    @property
    def batch_size(self): return 1
    @property
    def device(self): return self._device
    def tok_encode(self, string): return self.tokenizer.encode(string).tolist()
    def tok_decode(self, tokens): return self.tokenizer.decode(torch.tensor(tokens))

@auto_expand_env_vars
def main(
    checkpoint_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    benchmark: str = "debug",
    map_branch: bool = False,
    config_overrides: dict[str, Any] | None = None,
    output_path: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
    device = f"cuda:{local_rank}"
    
    if local_rank == 0:
        print(f"🚀 启动魔改版评估管线 | 任务: {benchmark}")

    lm_model = CustomResearchLM(
        checkpoint_dir,
        device=device,
        map_branch=map_branch,
        config_overrides=config_overrides,
    )
    
    results = evaluator.simple_evaluate(
        model=lm_model,
        tasks=["piqa"] if benchmark == "debug" else benchmark.split(","),
        confirm_run_unsafe_code=True,
        batch_size=1,
        metadata=metadata,
    )

    # 与 CustomResearchLM.is_master 一致：多节点时应用全局 rank==0，而非 local_rank==0（每节点各有一个 local 0）
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    if is_main and output_path is not None:
        from lm_eval.utils import make_table
        print(make_table(results))

        # 输出JSON格式结果
        json_output = {
            "benchmark": benchmark,
            "checkpoint_dir": checkpoint_dir,
            "results": results,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(output_path).expanduser()
        if base.suffix.lower() == ".json":
            output_file = base.with_name(f"{base.stem}_{ts}{base.suffix}")
        else:
            output_file = base / f"eval_results_{ts}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存到: {output_file}")

if __name__ == "__main__":
    CLI(main)