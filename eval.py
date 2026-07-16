"""
Evaluation pipeline for logKV models.

Usage:
    # Single GPU
    python eval.py --checkpoint_dir ./ckpt/qwen0.6b-32k-cpt-base --benchmark piqa

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 eval.py --checkpoint_dir ./ckpt/... --benchmark "boolq,piqa,..."

    # With YAML config
    python eval.py --config exp/qwen1.7b-32k/eval.yaml

除 logKV 相关内容（LogKVLM、YAML --config 机制、tokenizer 回退）外，
本文件与 kv 分支的 eval.py 逐段对齐（环境变量、输出、指标收集等）。
"""

from __future__ import annotations

import os
import re
import csv
import glob
import json
import time
import inspect
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from typing import Any

import tqdm

from utils import *


class SafeJSONEncoder(json.JSONEncoder):
    """处理无法直接序列化的对象（numpy、torch、函数等）"""
    def default(self, obj):
        # numpy 类型
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        # torch 类型
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        # 函数、方法、可调用对象
        if callable(obj):
            return f"<function: {obj.__name__ if hasattr(obj, '__name__') else str(obj)}>"
        # 其他特殊类型
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        # 类实例
        if hasattr(obj, '__dict__'):
            return f"<{obj.__class__.__name__} object>"
        # 默认处理
        return super().default(obj)


def extract_results_to_csv(results: dict, benchmark: str, output_path: Path) -> None:
    """
    从 results 中提取各数据集的 acc 或 acc_norm 数值，输出为 CSV 文件。
    - HellaSwag 数据集：提取 acc_norm,none
    - 其他数据集：提取 acc,none
    """
    csv_rows = []
    tasks = benchmark.split(",") if benchmark != "debug" else ["piqa"]

    results_data = results.get("results", {})

    for task in tasks:
        task = task.strip()
        task_results = results_data.get(task, {})

        # HellaSwag 使用 acc_norm，其他使用 acc
        if task.lower() in ["hellaswag", "arc-c", "openbookqa"]:
            metric_key = "acc_norm,none"
            metric_name = "acc_norm"
        else:
            metric_key = "acc,none"
            metric_name = "acc"

        value = task_results.get(metric_key, None)

        csv_rows.append({
            "dataset": task,
            # "metric": metric_name,
            "value": value,
        })

    # 写入 CSV
    csv_file = output_path.with_suffix(".csv")
    try:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "value"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"📊 CSV 结果已保存到: {csv_file}")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")

if 'HF_DATASETS_CACHE' not in os.environ and 'PKU' not in os.environ:
    print("设置环境变量...")

    BASE = '/home/ma-user/work/bucket-wulan-green/wubohan/data/hf_cache'
    os.environ['HF_HOME'] = BASE
    os.environ['HF_DATASETS_CACHE'] = f'{BASE}/hf_cache'
    os.environ['HF_EVALUATE_CACHE'] = f'{BASE}/evaluate'
    os.environ['HF_MODULES_CACHE'] = f'{BASE}/modules'
    os.environ['HUGGINGFACE_HUB_CACHE'] = f'{BASE}/hub'
    os.environ['HF_HUB_CACHE'] = f'{BASE}/hub'
    os.environ['RULER_CACHE_DIR'] = f'{BASE}/ruler_cache'
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
from litgpt.ruler_patch import apply_patch
apply_patch()

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


# ==========================================
# 🧩 logKV 专属：YAML --config 机制（与 demo.py 相同约定：
# YAML 非 null 值覆盖 CLI；支持 "config:" 继承）
# ==========================================

def _load_yaml_config(yaml_path: str, model_dir: str) -> dict:
    """Load YAML config with inheritance ('config:' key overrides)."""
    yaml_path = os.path.normpath(os.path.expanduser(yaml_path))
    if not os.path.isfile(yaml_path):
        return {}

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    if "config" in cfg:
        base_name = cfg.pop("config")
        base_dir = os.path.dirname(yaml_path)
        base_path = os.path.join(base_dir, base_name)
        base_cfg = _load_yaml_config(base_path, model_dir)
        return {**base_cfg, **cfg}

    return cfg


# PyYAML implements YAML 1.1: scientific notation WITHOUT a decimal point
# ("2e-5") does not match its float resolver and loads as *str*. Coerce such
# top-level values so numeric params never receive strings.
_SCI_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?[eE][-+]?\d+")


def _coerce_yaml_sci_floats(cfg: dict) -> dict:
    """Convert top-level str values that are scientific-notation numbers to float."""
    return {
        k: float(v) if isinstance(v, str) and _SCI_FLOAT_RE.fullmatch(v) else v
        for k, v in cfg.items()
    }


# ==========================================
# 🧩 logKV 专属：tokenizer 目录回退解析
# ==========================================

def _find_tokenizer_dir(checkpoint_dir: str, tokenizer_dir: str | None) -> Path:
    """Resolve a directory that actually holds tokenizer.json / tokenizer.model.

    The training save dir normally receives a copy of the tokenizer files
    (demo.py copies them next to lit_model.pth), but older checkpoints,
    interrupted runs, or ``save_ckpt: false`` leave it without one. litgpt's
    ``Tokenizer`` then raises a bare ``NotImplementedError``, so resolve the
    directory up front and fail with an actionable message instead.

    Search order:
      1. explicit ``tokenizer_dir`` (CLI/YAML),
      2. ``checkpoint_dir`` itself,
      3. the base model dir recorded in ``model_config.yaml``
         (demo.py convention: checkpoints/<hf org>/<hf name>).
    """
    candidates: list[tuple[str, Path]] = []
    if tokenizer_dir is not None:
        candidates.append(("--tokenizer_dir", Path(tokenizer_dir)))
    candidates.append(("checkpoint_dir", Path(checkpoint_dir)))

    cfg_path = Path(checkpoint_dir) / "model_config.yaml"
    if cfg_path.is_file():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            hf = cfg.get("hf_config") or {}
            if hf.get("org") and hf.get("name"):
                candidates.append(
                    ("base checkpoint (from model_config.yaml)",
                     Path("checkpoints") / hf["org"] / hf["name"])
                )
        except Exception as e:  # noqa: BLE001 — fallback probing only
            print(f"[eval] WARNING: could not read {cfg_path} for tokenizer fallback: {e}")

    for label, d in candidates:
        if (d / "tokenizer.json").is_file() or (d / "tokenizer.model").is_file():
            return d

    tried = "\n".join(f"  - {label}: {d.resolve()}" for label, d in candidates)
    raise FileNotFoundError(
        "No tokenizer.json / tokenizer.model found. Searched:\n"
        f"{tried}\n"
        "Fix: pass --tokenizer_dir <dir containing the tokenizer files>, or copy "
        "the base model's tokenizer files into the checkpoint dir. Note that a "
        "training run with `save_ckpt: false` writes NO checkpoint (no weights, "
        "no tokenizer) — the train->eval pipeline requires `save_ckpt: true`."
    )


class LogKVLM(LM):
    """LM wrapper that scores and generates through the log-structured KV cache.

    Both loglikelihood scoring and generate_until run through the merged-position
    slot cache used by the LogKV training path: full post-RoPE keys are stored as
    exact recent tokens or compressed slots, and slot attention scores one logit
    per slot with the log(w) mass bias. There is no dense fallback — this
    pipeline only evaluates the logKV compression route.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        config_overrides: dict[str, Any] | None = None,
        log_kv_B: int = 512,
        log_kv_recent_size: int = 1024,
        log_kv_prefill_block: int = 256,
        tokenizer_dir: str | None = None,
    ):
        super().__init__()
        self._device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_kv_B = log_kv_B
        self.log_kv_recent_size = log_kv_recent_size
        self.log_kv_prefill_block = log_kv_prefill_block

        # 控制打印：在多卡下尽量只让主进程打印，防止刷屏
        is_master = not dist.is_initialized() or dist.get_rank() == 0

        resolved_tok_dir = _find_tokenizer_dir(checkpoint_dir, tokenizer_dir)
        if is_master:
            print(f"🔤 正在加载 tokenizer: {resolved_tok_dir}")
        self.tokenizer = Tokenizer(resolved_tok_dir)

        # ==========================================
        # 🌟 智能 Config 路由加载 (YAML 版本)
        # ==========================================
        config_path = os.path.join(checkpoint_dir, "model_config.yaml")

        if os.path.exists(config_path):
            if is_master: print(f"📄 发现专属架构 YAML 配置文件: {config_path}，正在自动同步架构...")
            self.config = _config_from_yaml_and_overrides(config_path, config_overrides)
        else:
            if is_master: print("⚠️ 未发现训练期保存的 YAML 配置文件，正在使用备用参数初始化...")
            fallback_kw: dict[str, Any] = {"name": checkpoint_dir.split("/")[-1]}
            if config_overrides:
                fallback_kw.update(config_overrides)
            fallback_kw = _filter_config_dict(fallback_kw)
            self.config = Config.from_name(**fallback_kw)

        # ==========================================

        if is_master: print("🔧 正在初始化 Transformer (logKV 压缩注意力)...")
        self.model = GPT(self.config).to(device).bfloat16()

        if is_master: print(f"🔄 正在加载权重...")
        checkpoint = _load_lit_model_checkpoint(checkpoint_dir, map_location=device)

        # 🌟 核心修复：检查是不是被包裹过的 checkpoint 字典
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            if is_master: print("📦 检测到 Fabric Checkpoint，已自动提取 model 权重。")
        else:
            state_dict = checkpoint

        # 🌟 强烈建议：捕获并打印一下加载结果，看看是不是真的加载成功了！
        load_result = self.model.load_state_dict(state_dict, strict=False)

        if is_master:
            print(f"✅ 权重加载完毕！缺失的 keys: {len(load_result.missing_keys)} 个")
            # 如果 missing_keys 极其多（比如几百个），说明加载又失败了

        self.model.eval()

        # 推理时延指标收集
        self.gen_metrics: list[dict] = []      # generate_until
        self.ppl_metrics: list[dict] = []      # loglikelihood

    def _set_eval_cache(self, max_seq_length: int) -> None:
        """Install the correct per-request inference cache.

        LogKV now stores full post-RoPE keys directly in slots; there is no
        separate per-token position buffer to size or maintain here.
        ``max_seq_length`` only bounds the append-only token counter and level
        hierarchy.
        """
        dtype = next(self.model.parameters()).dtype
        self.model.set_log_kv_cache(
            batch_size=1,
            max_seq_length=max_seq_length,
            device=self._device,
            dtype=dtype,
            B=self.log_kv_B,
            recent_size=max(2, min(self.log_kv_recent_size, max_seq_length)),
            prefill_block=self.log_kv_prefill_block,
        )

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
    # 🌟 核心 1：PPL 与 选择题评测
    # ==========================================
    def _score_tokens(self, ctx_enc: list[int], cont_enc: list[int]) -> tuple[float, bool]:
        """Forward (context + continuation) and return
        ``(sum log p(continuation | context), is_greedy)``.

        Left-truncates so the sequence fits ``max_seq_length``. If the
        continuation alone meets/exceeds the window, the context is dropped and
        the continuation is left-truncated to its last ``max_len - 1`` tokens —
        scoring is then partial but the forward stays in bounds.
        """
        dp_rank = dist.get_rank() if dist.is_initialized() else 0

        # 🌟 安全阀：如果 题干 + 选项 > max_seq_length，必须切掉题干最前面的部分
        max_len = self.model.max_seq_length
        if len(ctx_enc) + len(cont_enc) > max_len:
            keep_ctx_len = max_len - len(cont_enc)
            if keep_ctx_len <= 0:
                cont_enc = cont_enc[-(max_len - 1):]
                ctx_enc = []
            else:
                ctx_enc = ctx_enc[-keep_ctx_len:]
            print(f"⚠️ [Rank {dp_rank}] 警告: 触发截断，剩余 context 长度: {len(ctx_enc)}")

        if len(ctx_enc) == 0:
            ctx_enc = [self.tokenizer.bos_id]

        inps = torch.tensor([ctx_enc + cont_enc], dtype=torch.long, device=self._device)
        seq_len = inps.size(1)
        ctx_len = len(ctx_enc)

        with torch.no_grad():
            # Score with the same merged-position slot attention used by LogKV
            # inference. ``input_pos`` must be append-only contiguous because
            # slot compaction is order-based rather than indexed.
            self._set_eval_cache(seq_len)
            try:
                t0 = time.perf_counter()
                input_pos = torch.arange(seq_len, device=self._device, dtype=torch.int64)
                logits = self.model(inps, input_pos=input_pos)
                t1 = time.perf_counter()
            finally:
                self.model.clear_kv_cache()

        self.ppl_metrics.append({
            "total_seq_len": seq_len, "context_len": ctx_len,
            "cont_len": len(cont_enc),
            "forward_time_ms": round((t1 - t0) * 1000, 2),
        })

        # Continuation logits: positions [ctx_len-1, seq_len-2] predict cont tokens.
        cont_logits = logits[0, ctx_len - 1 : seq_len - 1]
        cont_targets = torch.tensor(cont_enc, dtype=torch.long, device=self._device)
        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=cont_targets.unsqueeze(-1)).squeeze(-1)
        is_greedy = (cont_logits.argmax(dim=-1) == cont_targets).all().item()
        return token_log_probs.sum().item(), is_greedy

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

            results.append(self._score_tokens(ctx_enc, cont_enc))

        # 清理 CUDA 缓存，避免 all_gather 时 OOM
        torch.cuda.empty_cache()
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

            # 解析 lm-eval 的 gen_kwargs：`until` 是停词（字符串列表），长度上限用 `max_gen_toks`（见 lm-eval model_guide）
            max_new_tokens = int(gen_args.get("max_gen_toks", gen_args.get("max_length", self.max_gen_toks)))
            do_sample = bool(gen_args.get("do_sample", False))
            temperature = float(gen_args.get("temperature", 1.0))
            top_p = float(gen_args.get("top_p", 1.0))
            top_k = gen_args.get("top_k", None)
            if not do_sample:
                # litgpt.generate.sample 仅在 temperature<=0 且 top_p<=0 时走 argmax；仅设 temperature=0 而 top_p=1 仍会多项式采样
                temperature = 0.0
                top_p = 0.0

            prompt_tensor = self.tokenizer.encode(prompt, device=self._device)

            # 🌟 安全阀：为生成的新 Token 预留空间
            max_len = self.model.max_seq_length
            if prompt_tensor.size(0) + max_new_tokens > max_len:
                keep_prompt_len = max_len - max_new_tokens
                prompt_tensor = prompt_tensor[-keep_prompt_len:]
                print(f"⚠️ [Rank {dp_rank}] 警告: 触发生成截断，Prompt 被切至: {keep_prompt_len}")

            total_max_len = prompt_tensor.size(0) + max_new_tokens
            prompt_len = prompt_tensor.size(0)

            with torch.no_grad():
                # 🧩 logKV：长上下文生成使用 merged-position slot cache
                self._set_eval_cache(total_max_len)
                try:
                    t0 = time.perf_counter()
                    out = litgpt_generate(
                        self.model,
                        prompt_tensor,
                        max_returned_tokens=total_max_len,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        eos_id=self.tokenizer.eos_id,
                    )
                    t1 = time.perf_counter()
                finally:
                    self.model.clear_kv_cache()

            # 截取新生成的部分并解码
            generated_tokens = out[prompt_tensor.size(0):]
            gen_len = generated_tokens.size(0)
            total_time_s = t1 - t0
            self.gen_metrics.append({
                "prompt_len": prompt_len, "generated_tokens": gen_len,
                "total_time_s": round(total_time_s, 4),
                "decode_ms_per_token": round(total_time_s / max(gen_len, 1) * 1000, 2),
                "gen_tokens_per_sec": round(gen_len / total_time_s, 2) if total_time_s > 0 else 0,
            })

            decoded = self.tokenizer.decode(generated_tokens)
            results.append(decoded)

        # 清理 CUDA 缓存，避免 all_gather 时 OOM
        torch.cuda.empty_cache()
        return self.all_gather_results(results)

    def loglikelihood_rolling(self, requests):
        """Rolling log-likelihood over full documents (lm-eval PPL tasks, e.g.
        wikitext). Returns one summed log-prob (scalar float) per request,
        matching lm-eval's reference implementations — returning tuples here
        breaks perplexity aggregation downstream.

        Long documents are scored in non-overlapping windows of max_seq_length:
        each window's tokens are conditioned on the window prefix (the first
        window starts from BOS). This matches lm-eval's standard rolling-window
        scoring; a ``pass`` stub would return None and break PPL tasks.
        """
        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        dp_size = dist.get_world_size() if dist.is_initialized() else 1

        local_requests = requests[dp_rank::dp_size]
        results = []
        disable_tqdm = (dp_rank != 0)

        max_len = self.model.max_seq_length
        for req in tqdm.tqdm(local_requests, desc=f'Rank {dp_rank}', position=dp_rank, disable=disable_tqdm):
            (text,) = req.args
            tokens = self.tokenizer.encode(text, bos=False).tolist()

            total_logprob = 0.0
            # Non-overlapping windows; window size max_len - 1 leaves room for
            # the 1-token context (BOS or the previous window's last token).
            window = max_len - 1
            for start in range(0, len(tokens), window):
                chunk = tokens[start : start + window]
                if start == 0:
                    ctx = [self.tokenizer.bos_id]
                else:
                    ctx = [tokens[start - 1]]
                logprob, _ = self._score_tokens(ctx, chunk)
                total_logprob += logprob

            results.append(total_logprob)

        torch.cuda.empty_cache()
        return self.all_gather_results(results)

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

def _resolve_checkpoint_dir(checkpoint_dir: str) -> str:
    """自动检测分段 checkpoint。若存在 step_* 子目录（含 lit_model.pth）则选最大 step，否则直接用原路径。"""
    base = Path(os.path.expandvars(os.path.expanduser(checkpoint_dir)))
    step_dirs = [
        d for d in base.glob("step_*")
        if d.is_dir() and (d / "lit_model.pth").exists()
    ]
    if not step_dirs:
        return checkpoint_dir
    max_dir = max(step_dirs, key=lambda d: int(d.name.split("_")[1]))
    print(f"🔍 检测到分段 checkpoint，自动选择最新: {max_dir}")
    return str(max_dir)


@auto_expand_env_vars
def main(
    checkpoint_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    benchmark: str = "debug",
    config_overrides: dict[str, Any] | None = None,
    output_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    # ── 🧩 logKV（本管线只跑压缩路线，无 dense 分支）──
    log_kv_B: int = 512,
    log_kv_recent_size: int = 1024,
    # prefill 分块大小：块内 query 共享块首冻结的 slot 状态。2 = 严格 2-token
    # 流式语义（用于 A/B 验证近似偏差）；越大越快，偏差上界 = 块内 query 比严格
    # 流式多看到 < block 个未压缩 token（缓存状态轨迹两者严格一致）。
    log_kv_prefill_block: int = 256,
    # ── 🧩 logKV：tokenizer 回退（checkpoint 目录缺 tokenizer 文件时用）──
    tokenizer_dir: str | None = None,
    # ── 🧩 logKV：YAML config ──
    config: str | None = None,
):
    # ── 🧩 logKV：加载 YAML config（非 null 值覆盖 CLI，同 demo.py 约定）──
    # NOTE: `locals()[k] = v` does NOT write back to a function's real locals in
    # CPython, so YAML overrides must be applied by explicit re-binding.
    _yaml: dict = {}
    if config is not None:
        _yaml = _coerce_yaml_sci_floats(
            _load_yaml_config(os.path.join(os.getcwd(), config), checkpoint_dir) or {}
        )
        _valid = set(inspect.signature(main).parameters)
        for _k in _yaml:
            if _k != "config" and _k not in _valid:
                print(f"[eval] WARNING: unknown YAML key ignored: {_k}")

    def _o(name, current):
        v = _yaml.get(name)
        return v if v is not None else current

    checkpoint_dir = _o("checkpoint_dir", checkpoint_dir)
    benchmark = _o("benchmark", benchmark)
    config_overrides = _o("config_overrides", config_overrides)
    output_path = _o("output_path", output_path)
    metadata = _o("metadata", metadata)
    log_kv_B = _o("log_kv_B", log_kv_B)
    log_kv_recent_size = _o("log_kv_recent_size", log_kv_recent_size)
    log_kv_prefill_block = _o("log_kv_prefill_block", log_kv_prefill_block)
    tokenizer_dir = _o("tokenizer_dir", tokenizer_dir)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))

    device = f"cuda:{local_rank}"

    if local_rank == 0:
        print(f"🚀 启动魔改版评估管线 | 任务: {benchmark}")
        print(f"🧩 logKV 压缩注意力 | B: {log_kv_B} | recent_size: {log_kv_recent_size} | prefill_block: {log_kv_prefill_block}")

    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)

    lm_model = LogKVLM(
        checkpoint_dir,
        device=device,
        config_overrides=config_overrides,
        log_kv_B=log_kv_B,
        log_kv_recent_size=log_kv_recent_size,
        log_kv_prefill_block=log_kv_prefill_block,
        tokenizer_dir=tokenizer_dir,
    )

    results = evaluator.simple_evaluate(
        model=lm_model,
        tasks=["piqa"] if benchmark == "debug" else benchmark.split(","),
        confirm_run_unsafe_code=True,
        batch_size=1,
        metadata=metadata,
    )

    # 与 LogKVLM.is_master 一致：多节点时应用全局 rank==0，而非 local_rank==0（每节点各有一个 local 0）
    is_main = not dist.is_initialized() or dist.get_rank() == 0

    if is_main:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 🌟 第一步：立即保存原始 results 对象，便于后续恢复
        results_cache_file = Path("eval_results_cache.json")
        try:
            with open(results_cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
            print(f"💾 原始 results 已缓存到: {results_cache_file}")
        except Exception as e:
            print(f"❌ 缓存 results 失败: {e}")
            print(f"⚠️ 尝试使用备用方案...")
            try:
                # 备用方案：先转换为字符串表示
                results_str = str(results)
                with open(results_cache_file, "w", encoding="utf-8") as f:
                    json.dump({"results_str": results_str}, f, indent=2, ensure_ascii=False)
                print(f"💾 results 已以字符串形式缓存到: {results_cache_file}")
            except Exception as e2:
                print(f"❌ 备用方案也失败了: {e2}")
                return

        # 🌟 第二步：打印表格
        from lm_eval.utils import make_table
        print(make_table(results))

        # 🌟 第三步：如果指定了输出路径，保存完整的JSON输出
        if output_path is not None:
            json_output = {
                "timestamp": ts,
                "benchmark": benchmark,
                "checkpoint_dir": checkpoint_dir,
                "results": results,
            }

            base = Path(output_path).expanduser()
            if base.suffix.lower() == ".json":
                output_file = base.with_name(f"{base.stem}_{ts}{base.suffix}")
            else:
                output_file = base / f"eval_results_{ts}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
                print(f"✅ 完整结果已保存到: {output_file}")
            except Exception as e:
                print(f"❌ 保存完整结果失败: {e}")
                print(f"⚠️ 尝试备用方案...")
                try:
                    json_output["results"] = str(results)
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(json_output, f, indent=2, ensure_ascii=False)
                    print(f"✅ 完整结果已以备用方案保存到: {output_file}")
                except Exception as e2:
                    print(f"❌ 备用方案也失败了: {e2}")

            # 🌟 第四步：保存推理时延 xlsx
            inference_xlsx = output_file.with_suffix(".inference_metrics.xlsx")
            try:
                import pandas as pd
                with pd.ExcelWriter(inference_xlsx) as writer:
                    if lm_model.gen_metrics:
                        gen_df = pd.DataFrame(lm_model.gen_metrics)
                        gen_df.to_excel(writer, sheet_name="generate_until", index=False)
                        # 按 prompt 长度分桶统计
                        gen_df["prompt_bucket"] = pd.cut(gen_df["prompt_len"],
                            bins=[0, 1024, 4096, 8192, 16384, 32768, 999999],
                            labels=["0-1K", "1K-4K", "4K-8K", "8K-16K", "16K-32K", "32K+"])
                        summary = gen_df.groupby("prompt_bucket", observed=False).agg(
                            count=("prompt_len", "count"),
                            avg_prompt_len=("prompt_len", "mean"),
                            avg_decode_ms_tok=("decode_ms_per_token", "mean"),
                            avg_gen_tok_sec=("gen_tokens_per_sec", "mean"),
                        ).round(2).reset_index()
                        summary.to_excel(writer, sheet_name="gen_by_bucket", index=False)
                    if lm_model.ppl_metrics:
                        ppl_df = pd.DataFrame(lm_model.ppl_metrics)
                        ppl_df.to_excel(writer, sheet_name="loglikelihood", index=False)
                print(f"📊 推理时延指标已保存至 {inference_xlsx}")
            except Exception as e:
                print(f"⚠️ 推理时延 xlsx 保存失败: {e}")

            # 🌟 第五步：生成 CSV 结果文件
            extract_results_to_csv(results, benchmark, output_file)


@auto_expand_env_vars
def output_from_cache(
    cache_file: str = "eval_results_cache.json",
    benchmark: str = "debug",
    checkpoint_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    output_path: str | None = None,
):
    """从缓存文件读取 results，重新进行输出处理（无需重新运行评测）"""
    cache_path = Path(cache_file)
    if not cache_path.exists():
        print(f"❌ 缓存文件不存在: {cache_path}")
        return

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        # 检查是否是备用方案保存的（字符串形式）
        if "results_str" in cache_data:
            print(f"⚠️ 缓存是字符串形式，无法恢复为结构化数据")
            print(f"缓存内容: {cache_data['results_str'][:500]}...")
            return

        results = cache_data
        print(f"✅ 从缓存读取 results: {cache_path}")
    except Exception as e:
        print(f"❌ 读取缓存失败: {e}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 打印表格
    try:
        from lm_eval.utils import make_table
        print(make_table(results))
    except Exception as e:
        print(f"⚠️ 打印表格失败: {e}")

    # 保存完整的JSON输出
    if output_path is not None:
        json_output = {
            "timestamp": ts,
            "benchmark": benchmark,
            "checkpoint_dir": checkpoint_dir,
            "results": results,
        }

        base = Path(output_path).expanduser()
        if base.suffix.lower() == ".json":
            output_file = base.with_name(f"{base.stem}_{ts}{base.suffix}")
        else:
            output_file = base / f"eval_results_{ts}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
            print(f"✅ 完整结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

        # 生成 CSV 结果文件
        extract_results_to_csv(results, benchmark, output_file)

if __name__ == "__main__":
    run_cli(main)
