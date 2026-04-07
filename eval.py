import os
import ssl
import urllib3

if 'HF_DATASETS_CACHE' not in os.environ:
    print("设置环境变量...")
    os.environ['HF_HOME'] = '/data/zys/data/hf_cache'
    os.environ['HF_DATASETS_CACHE'] = '/data/zys/data/hf_cache/hf_cache'
    os.environ['HF_EVALUATE_CACHE'] = '/data/zys/data/hf_cache/evaluate'
    os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # os.environ['CURL_CA_BUNDLE'] = ''
    # os.environ['REQUESTS_CA_BUNDLE'] = ''
    # urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    # ssl._create_default_https_context = ssl._create_unverified_context

import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from jsonargparse import CLI

# 引入你的模型和工具
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer

# 引入 lm-eval 核心库
from lm_eval import evaluator
from lm_eval.api.model import LM

# 引入我们接下来要修改的 generate 函数
from litgpt.generate.base import generate as litgpt_generate

class CustomResearchLM(LM):
    def __init__(self, checkpoint_dir: str, device="cuda", use_research: bool = True):
        super().__init__()
        self._device = device
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = Tokenizer(checkpoint_dir)
        
        # ==========================================
        # 🌟 智能 Config 路由加载 (YAML 版本)
        # ==========================================
        # 试探两种最常见的 yaml 命名方式
        config_path = os.path.join(checkpoint_dir, "model_config.yaml")
        
        if os.path.exists(config_path):
            print(f"📄 发现专属架构 YAML 配置文件: {config_path}，正在自动同步架构...")
            with open(config_path, "r", encoding="utf-8") as f:
                # 使用 safe_load 是读取 yaml 的最佳安全实践
                config_dict = yaml.safe_load(f)
            
            # 🌟 核心：直接用字典解包重建 Config
            self.config = Config(**config_dict)
            
            # 确保对象的开关属性被正确覆盖
            self.use_research = getattr(self.config, 'use_research', False)
            
        else:
            print("⚠️ 未发现训练期保存的 YAML 配置文件，正在使用备用参数初始化...")
            self.use_research = use_research
            self.config = Config.from_name(
                name=checkpoint_dir.split("/")[-1],
                use_research=use_research,
                research_separate_parameter=True if use_research else False,
                research_swa_layers_str="0,2,4,6,8,10,12,14,16,18,20,22,24,26",
                research_identity_layers_str="1,3,5,7,9,11,13,15,17,19,21,23,25,27"
            )
            self.config.research_prefill_swa_layers = [int(x) for x in self.config.research_swa_layers_str.split(",")]
            self.config.research_prefill_identity_layers = [int(x) for x in self.config.research_identity_layers_str.split(",")]

        # ==========================================
        
        print(f"🔧 正在初始化 Transformer (Research模式: {self.use_research})...")
        self.model = GPT(self.config).to(device).bfloat16()
        
        print(f"🔄 正在加载权重...")
        state_dict = torch.load(f"{checkpoint_dir}/lit_model.pth")
        
        self.model.load_state_dict(state_dict, strict=False) 
        self.model.eval()
        print("✅ 模型就绪！")

    # ==========================================
    # 🌟 核心 1：PPL 与 选择题评测 (完美适配你的 forward)
    # ==========================================
    def loglikelihood(self, requests):
        results = []
        for req in requests:
            context, continuation = req.args[0], req.args[1]
            
            ctx_enc = self.tokenizer.encode(context).tolist()
            cont_enc = self.tokenizer.encode(continuation).tolist()
            # 🌟 安全阀：如果 题干 + 选项 > 4096，必须切掉题干最前面的部分
            max_len = self.model.max_seq_length
            if len(ctx_enc) + len(cont_enc) > max_len:
                # 保留完整的选项，切断 context 的头部
                keep_ctx_len = max_len - len(cont_enc)
                ctx_enc = ctx_enc[-keep_ctx_len:]
                print(f"⚠️ 警告: 触发截断，剩余 context 长度: {len(ctx_enc)}")
            
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
            
        return results

    # ==========================================
    # 🌟 核心 2：自回归生成任务 (LongBench)
    # ==========================================
    def generate_until(self, requests):
        results = []
        for req in requests:
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
                print(f"⚠️ 警告: 触发生成截断，Prompt 被切至: {keep_prompt_len}")
            
            with torch.no_grad():
                # 调用我们马上要在 base.py 里修改的 generate 函数
                out = litgpt_generate(
                    self.model, 
                    prompt_tensor, 
                    max_returned_tokens=prompt_tensor.size(0) + max_new_tokens,
                    temperature=gen_args.get("temperature", 1.0),
                    top_k=gen_args.get("top_k", None),
                    eos_id=self.tokenizer.eos_id
                )
                
            # 截取新生成的部分并解码
            generated_tokens = out[prompt_tensor.size(0):]
            decoded = self.tokenizer.decode(generated_tokens)
            results.append(decoded)
            
        return results

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

def main(
    checkpoint_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    benchmark: str = "debug",  
):
    print(f"🚀 启动魔改版评估管线 | 任务: {benchmark}")
    lm_model = CustomResearchLM(checkpoint_dir)
    results = evaluator.simple_evaluate(
        model=lm_model,
        tasks=["piqa"] if benchmark == "debug" else benchmark.split(","),
        batch_size=1,
        device="cuda"
    )
    from lm_eval.utils import make_table
    print(make_table(results))

if __name__ == "__main__":
    CLI(main)