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
import contextlib
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
import numpy as np
from typing import Any

import tqdm

from utils import *


# ==========================================
# 🧩 分布式小工具
# ==========================================
# 约定（与 demo.py 的 checkpoint 保存一致）：**任何"看文件系统"的判定只由
# rank 0 做，再 broadcast 给其它 rank**。共享盘（OBS/NFS）的元数据缓存在各节点
# 上的可见时刻不一致，若每个 rank 自己 stat/glob，就可能出现分支不一致 ——
# 一部分 rank 进了集合通信（barrier / all_gather）而另一部分跳过或直接抛异常
# 退出，剩下的 rank 就一直等到 NCCL 超时（这里设的是 12h）。评测跑完之后
# "卡很久" 正是这一类分支不一致 + 进程组没有显式销毁造成的。

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env_int("RANK", 0)


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _env_int("WORLD_SIZE", 1)


def _local_rank() -> int:
    return _env_int("LOCAL_RANK", _global_rank())


def _rank_label() -> str:
    return f"GlobalRank {_global_rank()}/{_world_size()} (local {_local_rank()})"


def _tqdm_position() -> int:
    # Use global rank so aggregated multi-node stdout gets one tqdm row per
    # process instead of collapsing every node onto local ranks 0..7.
    return _global_rank()


_DIST_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "LITGPT_EXPECTED_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
)


def _dist_env_text() -> str:
    return " ".join(f"{key}={os.environ.get(key, '<unset>')}" for key in _DIST_ENV_KEYS)


def _expected_world_size() -> int:
    for key in ("LITGPT_EXPECTED_WORLD_SIZE", "EXPECTED_WORLD_SIZE"):
        value = _env_int(key, 0)
        if value > 0:
            return value
    return 0


def _check_expected_world_size(stage: str) -> None:
    expected = _expected_world_size()
    if expected <= 0:
        return
    actual = _world_size()
    if actual != expected:
        raise RuntimeError(
            f"{stage}: distributed world size mismatch: actual={actual}, expected={expected}. "
            f"This usually means torchrun did not rendezvous across all nodes. env: {_dist_env_text()}"
        )


def _rendezvous_snapshot(stage: str) -> None:
    import socket

    ts = datetime.now().strftime("%H:%M:%S")
    if dist.is_available() and dist.is_initialized():
        dist_text = (
            f"dist_initialized=True backend={dist.get_backend()} "
            f"rank={dist.get_rank()} world={dist.get_world_size()}"
        )
    else:
        dist_text = "dist_initialized=False backend=<unset> rank=<unset> world=<unset>"
    print(f"🧭 [{ts}][{socket.gethostname()}][{stage}] {dist_text} env: {_dist_env_text()}", flush=True)


def _is_main() -> bool:
    """全局 rank == 0（多节点下每个节点都有一个 local_rank 0，不能用 local_rank 判定）。"""
    return _global_rank() == 0


def _dist_ready() -> bool:
    """真正处于多进程集合通信状态时才为 True（单卡下所有集合操作退化为 no-op）。"""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _bcast_device() -> torch.device | None:
    """NCCL 后端下 broadcast_object_list 必须显式给设备，否则会走默认卡导致串扰。"""
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return None


def _broadcast_obj(obj: Any, src: int = 0) -> Any:
    """把 rank ``src`` 上的任意可 pickle 对象广播给所有 rank（单卡时原样返回）。"""
    if not _dist_ready():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src, device=_bcast_device())
    return box[0]


def _hb(stage: str) -> None:
    """无条件心跳打印（所有 rank，不受 is_main 限制），排查多机卡死用。

    标记 dist.init_process_group / 模型加载完成这类一次性里程碑（真正的逐条
    计算进度由 loglikelihood/generate_until 里每个 rank 各自的 tqdm 负责，
    不需要在这里重复打印）。带 hostname，方便从日志里按 rank 分组核对。
    """
    import socket
    host = socket.gethostname()
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"💓 [{ts}][{_rank_label()}][{host}] {stage}", flush=True)


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

# 每个 rank 各自独立加载一遍数据集，HF datasets 的 "Found the latest cached
# dataset configuration ..." 日志和内部 tqdm 进度条会被重复打印 rank 份，
# 跟评测本身的计算进度混在一起。这里关掉，只留下面每个 rank 自己的计算进度。
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
try:
    import datasets as _hf_datasets
    _hf_datasets.disable_progress_bars()
    _hf_datasets.logging.set_verbosity_error()
except Exception:
    pass

import dataclasses

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
from lm_eval import evaluator
from lm_eval.api.model import LM
from litgpt.generate.base import generate as litgpt_generate
from litgpt.log_kv_diag import DIAG as LOG_KV_DIAG, diag_mode
from litgpt.log_kv_pin_diag import PinDiagRecorder
from litgpt.ruler_patch import apply_patch
apply_patch()

_CONFIG_FIELDS = {f.name for f in dataclasses.fields(Config)}


def _load_lit_model_checkpoint(
    checkpoint_dir: str, map_location: str | torch.device, wait_s: float = 120.0
) -> Any:
    """加载 ``{checkpoint_dir}/lit_model.pth``（单文件 ``torch.save``，与 demo FSDP ``state_dict_type='full'`` 一致）。

    多机场景下 rank 0 已经确认过文件存在（见 ``_resolve_checkpoint_dir``），但其它
    节点的共享盘元数据缓存可能还没刷新。此时直接抛 FileNotFoundError 会让这个 rank
    单独退出，其余 rank 卡在后续集合通信里等到 NCCL 超时；所以非 rank0 上先轮询等待
    ``wait_s`` 秒。单进程运行时不等待，行为与之前一致（立刻报错）。
    """
    lit_path = Path(checkpoint_dir).expanduser() / "lit_model.pth"
    if not lit_path.exists() and _dist_ready() and not _is_main():
        deadline = time.time() + wait_s
        while time.time() < deadline and not lit_path.exists():
            time.sleep(2.0)
        if lit_path.exists():
            print(f"[eval][{_rank_label()}] 共享盘元数据延迟，等待后已看到 {lit_path}")
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

def _probe_tokenizer_dir(checkpoint_dir: str, tokenizer_dir: str | None) -> Path:
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

    仅供 ``_find_tokenizer_dir`` 在 rank 0 上调用 —— 探测结果由 rank 0 广播，
    不要在各 rank 上分别调用（分支不一致会挂死，见文件顶部的分布式约定）。
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


def _find_tokenizer_dir(checkpoint_dir: str, tokenizer_dir: str | None) -> Path:
    """rank 0 探测 tokenizer 目录并广播结论；失败信息也一并广播。

    每个 rank 自己探测时，共享盘元数据缓存不一致会让一部分 rank 找到目录、另一部分
    抛 FileNotFoundError 单独退出，剩下的 rank 就卡在后面的集合通信里。这里让 rank 0
    独自判定，成功广播路径、失败广播错误信息，保证所有 rank 要么一起继续、要么一起
    以同一条报错退出。
    """
    payload: tuple[str, str] | None = None
    if _is_main():
        try:
            payload = ("ok", str(_probe_tokenizer_dir(checkpoint_dir, tokenizer_dir)))
        except FileNotFoundError as e:
            payload = ("err", str(e))

    kind, value = _broadcast_obj(payload)
    if kind == "err":
        raise FileNotFoundError(value)
    return Path(value)


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
        log_kv_pin_size: int = 0,
        log_kv_pin_obs_window: int = 64,
        log_kv_second_order_scale: float = 1.0,
        tokenizer_dir: str | None = None,
        pin_diag_recorder: PinDiagRecorder | None = None,
    ):
        super().__init__()
        self._device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_kv_B = log_kv_B
        self.log_kv_recent_size = log_kv_recent_size
        self.log_kv_prefill_block = log_kv_prefill_block
        self.log_kv_pin_size = log_kv_pin_size
        self.log_kv_pin_obs_window = log_kv_pin_obs_window
        self.log_kv_second_order_scale = float(log_kv_second_order_scale)
        self.pin_diag_recorder = pin_diag_recorder

        # 控制打印：在多卡下尽量只让主进程打印，防止刷屏
        is_master = _is_main()

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

        # LogKV eval cache is built lazily once and reset per request (see
        # _set_eval_cache) — never re-allocated per sample.
        self._eval_cache_ready = False

    def _set_eval_cache(self) -> None:
        """Install (once) and reset the LogKV inference cache.

        LogKV buffer sizes depend only on (B, recent_size, max_levels) — not on
        the request length — so the cache is built ONCE at the model's full
        window and reset in place before each request. Rebuilding per request
        re-allocates O(n_layer x (recent + B*logN)) CUDA buffers thousands of
        times over a benchmark run and fragments the allocator (OOM risk).
        ``max_seq_length`` only bounds the append-only token counter and level
        hierarchy, so sizing it at the model's window covers every request.
        """
        if self._eval_cache_ready:
            self.model.reset_log_kv_cache()
            return
        dtype = next(self.model.parameters()).dtype
        max_seq_length = self.model.max_seq_length
        self.model.set_log_kv_cache(
            batch_size=1,
            max_seq_length=max_seq_length,
            device=self._device,
            dtype=dtype,
            B=self.log_kv_B,
            recent_size=max(2, min(self.log_kv_recent_size, max_seq_length)),
            prefill_block=self.log_kv_prefill_block,
            # 显著性钉扎（SnapKV 式观察窗）：0 = 关闭。针对捞针类任务——
            # prompt 末尾的问题在 prefill 时给全前缀打分，top-P token 以
            # 精确槽形态钉在层级之外，免于被 mean-pool 稀释。
            pin_size=self.log_kv_pin_size,
            pin_obs_window=self.log_kv_pin_obs_window,
            second_order_scale=self.log_kv_second_order_scale,
        )
        self._eval_cache_ready = True

    # ==========================================
    # 🌟 分布式结果收集
    # ==========================================
    def all_gather_results(self, local_result_list: list, tag: str = ""):
        if not _dist_ready():
            return local_result_list

        # 各 rank 拿到的样本长度差异很大（长上下文生成尤甚），最慢的 rank 决定整体
        # 结束时间。先显式 barrier 把"等其它 rank"的时间量出来单独打印，否则它会被
        # 算进 all_gather 里，看上去就是"评测跑完之后莫名卡住"。
        t0 = time.perf_counter()
        dist.barrier()
        wait_s = time.perf_counter() - t0
        print(
            f"⏳ [{_rank_label()}] {tag} 本 rank {len(local_result_list)} 条已完成，"
            f"等待其它 rank 用时 {wait_s:.1f}s",
            flush=True,
        )

        dp_size = _world_size()
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
        rank_label = _rank_label()

        # 🌟 安全阀：如果 题干 + 选项 > max_seq_length，必须切掉题干最前面的部分
        max_len = self.model.max_seq_length
        if len(ctx_enc) + len(cont_enc) > max_len:
            keep_ctx_len = max_len - len(cont_enc)
            if keep_ctx_len <= 0:
                cont_enc = cont_enc[-(max_len - 1):]
                ctx_enc = []
            else:
                ctx_enc = ctx_enc[-keep_ctx_len:]
            print(f"⚠️ [{rank_label}] 警告: 触发截断，剩余 context 长度: {len(ctx_enc)}")

        if len(ctx_enc) == 0:
            ctx_enc = [self.tokenizer.bos_id]

        inps = torch.tensor([ctx_enc + cont_enc], dtype=torch.long, device=self._device)
        seq_len = inps.size(1)
        ctx_len = len(ctx_enc)

        with torch.no_grad():
            # Score with the same merged-position slot attention used by LogKV
            # inference. ``input_pos`` must be append-only contiguous because
            # slot compaction is order-based rather than indexed.
            self._set_eval_cache()
            try:
                t0 = time.perf_counter()
                input_pos = torch.arange(seq_len, device=self._device, dtype=torch.int64)
                # lm_head_start: materialize logits only for the scoring span
                # [ctx_len-1, seq_len) — full-sequence logits at 32K are ~10 GB
                # bf16, the sliced tensor is (cont_len + 1) x vocab.
                logits = self.model(inps, input_pos=input_pos, lm_head_start=ctx_len - 1)
                t1 = time.perf_counter()
            finally:
                # In-place state reset (defensive: the next request resets again
                # via _set_eval_cache). Keeping the buffers avoids re-allocation.
                self.model.reset_log_kv_cache()

        self.ppl_metrics.append({
            "total_seq_len": seq_len, "context_len": ctx_len,
            "cont_len": len(cont_enc),
            "forward_time_ms": round((t1 - t0) * 1000, 2),
        })

        # ``logits`` starts at position ctx_len-1 (lm_head_start); its first
        # len(cont_enc) rows are the positions [ctx_len-1, seq_len-2] that
        # predict the continuation tokens.
        cont_logits = logits[0, : len(cont_enc)]
        cont_targets = torch.tensor(cont_enc, dtype=torch.long, device=self._device)

        # fp32 log-softmax in position-chunks: rolling-PPL requests score a
        # full window (cont_len ~ max_seq_length), and a one-shot fp32
        # log_softmax over cont_len x vocab would peak at ~19 GB at 32K. fp32
        # (rather than the model's bf16) keeps per-token log-probs accurate —
        # they are summed over thousands of tokens downstream.
        total_logprob = 0.0
        is_greedy = True
        step = 1024
        for i in range(0, cont_logits.size(0), step):
            blk = cont_logits[i : i + step].to(torch.float32)
            tgt = cont_targets[i : i + step]
            log_probs = torch.log_softmax(blk, dim=-1)
            total_logprob += log_probs.gather(dim=-1, index=tgt.unsqueeze(-1)).sum().item()
            if is_greedy:
                is_greedy = bool((blk.argmax(dim=-1) == tgt).all().item())
        return total_logprob, is_greedy

    def loglikelihood(self, requests):
        dp_rank = _global_rank()
        dp_size = _world_size()

        local_requests = requests[dp_rank::dp_size]
        results = []

        # 每个 rank 都显示自己的进度条（不再只有 rank 0 可见）。
        desc = f"loglikelihood GlobalRank {dp_rank}/{dp_size} (local {_local_rank()})"
        for req in tqdm.tqdm(local_requests, desc=desc, position=_tqdm_position()):
            context, continuation = req.args[0], req.args[1]

            ctx_enc = self.tokenizer.encode(context).tolist()
            # Llama 3 等 use_bos=True：若对 continuation 再 encode 一次会多一个 BOS，拼接后破坏
            # loglikelihood 对齐；Qwen 通常无 BOS，bos=False 与默认行为一致。
            cont_enc = self.tokenizer.encode(continuation, bos=False).tolist()

            results.append(self._score_tokens(ctx_enc, cont_enc))

        # 清理 CUDA 缓存，避免 all_gather 时 OOM
        torch.cuda.empty_cache()
        return self.all_gather_results(results, tag="loglikelihood")

    # ==========================================
    # 🌟 核心 2：自回归生成任务 (LongBench)
    # ==========================================
    def generate_until(self, requests):
        dp_rank = _global_rank()
        dp_size = _world_size()

        local_requests = requests[dp_rank::dp_size]
        results = []

        # 每个 rank 都显示自己的进度条（不再只有 rank 0 可见）。
        desc = f"generate_until GlobalRank {dp_rank}/{dp_size} (local {_local_rank()})"
        for req in tqdm.tqdm(local_requests, desc=desc, position=_tqdm_position()):
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
            original_prompt_tokens = int(prompt_tensor.size(0))
            prompt_token_offset = 0

            # 🌟 安全阀：为生成的新 Token 预留空间
            max_len = self.model.max_seq_length
            if prompt_tensor.size(0) + max_new_tokens > max_len:
                keep_prompt_len = max_len - max_new_tokens
                prompt_token_offset = int(prompt_tensor.size(0) - keep_prompt_len)
                prompt_tensor = prompt_tensor[-keep_prompt_len:]
                print(f"⚠️ [{_rank_label()}] 警告: 触发生成截断，Prompt 被切至: {keep_prompt_len}")

            total_max_len = prompt_tensor.size(0) + max_new_tokens
            prompt_len = prompt_tensor.size(0)

            with torch.no_grad():
                # 🧩 logKV：长上下文生成使用 merged-position slot cache
                # （建一次、按请求原地重置，不逐样本重分配 — 见 _set_eval_cache）
                self._set_eval_cache()
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
                    if self.pin_diag_recorder is not None:
                        self.pin_diag_recorder.record(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            prompt=prompt,
                            doc=getattr(req, "doc", None),
                            request=req,
                            prompt_token_offset=prompt_token_offset,
                            original_prompt_tokens=original_prompt_tokens,
                            used_prompt_tokens=prompt_len,
                        )
                    t1 = time.perf_counter()
                finally:
                    self.model.reset_log_kv_cache()

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
        return self.all_gather_results(results, tag="generate_until")

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
        dp_rank = _global_rank()
        dp_size = _world_size()

        local_requests = requests[dp_rank::dp_size]
        results = []

        max_len = self.model.max_seq_length
        # 每个 rank 都显示自己的进度条（不再只有 rank 0 可见）。
        desc = f"loglikelihood_rolling GlobalRank {dp_rank}/{dp_size} (local {_local_rank()})"
        for req in tqdm.tqdm(local_requests, desc=desc, position=_tqdm_position()):
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
        return self.all_gather_results(results, tag="loglikelihood_rolling")

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
    """自动检测分段 checkpoint。若存在 step_* 子目录（含 lit_model.pth）则选最大 step，否则直接用原路径。

    glob + exists() 必须只由 rank 0 判定再广播：训练刚写完最后一个 step_* 时，
    其它节点的共享盘元数据缓存可能还看不到它，各 rank 各自 glob 就会选到**不同的
    checkpoint**（评测结果静默不可比），或者一部分 rank 找不到权重直接退出、把其余
    rank 留在集合通信里等到 NCCL 超时。
    """
    resolved: str | None = None
    if _is_main():
        base = Path(os.path.expandvars(os.path.expanduser(checkpoint_dir)))
        step_dirs = [
            d for d in base.glob("step_*")
            if d.is_dir() and (d / "lit_model.pth").exists()
        ]
        if step_dirs:
            max_dir = max(step_dirs, key=lambda d: int(d.name.split("_")[1]))
            print(f"🔍 检测到分段 checkpoint，自动选择最新: {max_dir}")
            resolved = str(max_dir)
        else:
            resolved = checkpoint_dir

    return _broadcast_obj(resolved)


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
    # 显著性钉扎（SnapKV 式观察窗，推理专用）：prefill 时 prompt 末尾
    # obs_window 个 query 给全前缀打分，每个 KV 组各钉 pin_size 个 token 为
    # 精确 w=1 槽（层级照常池化，缓存轨迹不变）。0 = 关闭。捞针类任务的关键。
    log_kv_pin_size: int = 0,
    log_kv_pin_obs_window: int = 64,
    # Must match the CPT target scale, not the warm-up intermediate value.
    # 0.0 reproduces the first-order LogKV eval path; 1.0 enables full
    # Sigma/Gamma second-order corrections.
    log_kv_second_order_scale: float = 1.0,
    # ── 🧩 logKV：tokenizer 回退（checkpoint 目录缺 tokenizer 文件时用）──
    tokenizer_dir: str | None = None,
    # ── 只跑一小批样本（Phase 0 诊断用；见 log_kv_diag_mode）。int = 绝对条数，
    # float in (0, 1) = lm-eval 的抽样比例。None = 跑全量。
    limit: int | float | None = None,
    # ── 🧩 logKV 诊断（score/value oracle 归因网格；见 litgpt.log_kv_diag）──
    # None/"off" = 不诊断（默认，零开销）。其余取值：
    # baseline/s_oracle/v_oracle/gamma_only/exact/dense —— 见 log_kv_diag 模块文档。
    # 要求 log_kv_pin_size == 0（诊断假定槽连续覆盖精确前缀，钉扎会打破这一点）。
    log_kv_diag_mode: str | None = None,
    # 诊断汇总 JSON 的落盘目录；缺省时退回 output_path。
    log_kv_diag_output: str | None = None,
    # ── 分层消融（D1 悬而未决的 marginal vs cumulative 问题；见 log_kv_diag）──
    # 层号 >= 此值强制走 exact，其余层走 log_kv_diag_mode（通常是 baseline），
    # 在同一次真实前向里测。None = 不启用（默认）。典型 sweep：28 - N，
    # N ∈ {0,1,2,4,7,14,21,28}（28 为本模型层数，按实际改）。
    log_kv_diag_exact_from_layer: int | None = None,
    # ── D5 重分箱：peakiness 只统计序列最后这么多个 token 的 query ──
    # （question/生成阶段紧邻的 prefill 尾部），None = 不过滤（默认，旧行为）。
    log_kv_diag_peak_window_from_end: int | None = None,
    # ── width 门控消融（D1 显示 rank-1 统计量在 slot_width>=4 起明显失真；
    # 见 log_kv_diag）── slot_width > 此值的槽位强制走 1 阶路径，不管
    # second_order_scale；None = 不启用（默认）。不改变 baseline 实际输出/
    # 传播的隐状态,只在 by_layer_output 里新增一个 err_width_gated 语料,
    # 和 err_baseline/err_baseline_1st_order 在同一次前向、同一组隐状态上
    # 直接可比。典型 sweep：{2, 4, 8, 16, None}。
    log_kv_diag_second_order_max_width: int | None = None,
    # ── layer 门控消融（同上；两者是 AND 关系，可单独用也可联合用）── 层号 >
    # 此值的层强制走 1 阶路径，不管 second_order_scale；None = 不启用（默认）。
    # 典型 sweep：{7, 14, 21, 27, None}（配合已有的 exact_from_layer 分层消融
    # 结果来选阈值）。
    log_kv_diag_second_order_max_layer: int | None = None,
    # ── 🧩 logKV pin 诊断：比较 _log_kv_pin_indices 与 NIAH needle token span ──
    # None = 关闭。开启后只记录 generate_until 请求（NIAH/RULER 属于这个路径），
    # 不改变模型输出；建议配合 --benchmark niah_single_1 --limit 4 单卡先跑。
    log_kv_pin_diag_output: str | None = None,
    log_kv_pin_diag_radius: int = 16,
    log_kv_pin_diag_max_samples: int | None = None,
    log_kv_pin_diag_include_indices: bool = False,
    # ── 🧩 logKV：YAML config ──
    config: str | None = None,
):
    # ── 🧩 logKV：加载 YAML config（非 null 值覆盖 CLI，同 demo.py 约定）──
    # NOTE: `locals()[k] = v` does NOT write back to a function's real locals in
    # CPython, so YAML overrides must be applied by explicit re-binding.
    _yaml: dict = {}
    if config is not None:
        # expand_env_vars：bash 风格 ${VAR} / ${VAR-default} 展开，与 demo.py 的
        # YAML 装载、majob.sh 的 bash 展开共用同一语义（详见 utils.expand_env_vars）。
        _yaml = _coerce_yaml_sci_floats(
            expand_env_vars(_load_yaml_config(os.path.join(os.getcwd(), config), checkpoint_dir) or {})
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
    log_kv_pin_size = _o("log_kv_pin_size", log_kv_pin_size)
    log_kv_pin_obs_window = _o("log_kv_pin_obs_window", log_kv_pin_obs_window)
    log_kv_second_order_scale = float(_o("log_kv_second_order_scale", log_kv_second_order_scale))
    tokenizer_dir = _o("tokenizer_dir", tokenizer_dir)
    limit = _o("limit", limit)
    log_kv_diag_mode = _o("log_kv_diag_mode", log_kv_diag_mode)
    log_kv_diag_output = _o("log_kv_diag_output", log_kv_diag_output)
    log_kv_diag_exact_from_layer = _o("log_kv_diag_exact_from_layer", log_kv_diag_exact_from_layer)
    log_kv_diag_peak_window_from_end = _o(
        "log_kv_diag_peak_window_from_end", log_kv_diag_peak_window_from_end
    )
    log_kv_diag_second_order_max_width = _o(
        "log_kv_diag_second_order_max_width", log_kv_diag_second_order_max_width
    )
    log_kv_diag_second_order_max_layer = _o(
        "log_kv_diag_second_order_max_layer", log_kv_diag_second_order_max_layer
    )
    log_kv_pin_diag_output = _o("log_kv_pin_diag_output", log_kv_pin_diag_output)
    log_kv_pin_diag_radius = _o("log_kv_pin_diag_radius", log_kv_pin_diag_radius)
    log_kv_pin_diag_max_samples = _o("log_kv_pin_diag_max_samples", log_kv_pin_diag_max_samples)
    log_kv_pin_diag_include_indices = _o(
        "log_kv_pin_diag_include_indices", log_kv_pin_diag_include_indices
    )

    local_rank = _local_rank()
    world_size = _world_size()
    if world_size > 1 or _expected_world_size() > 0:
        _rendezvous_snapshot("eval startup before init_process_group")
        _check_expected_world_size("eval startup before init_process_group")

    # 进程组可能由外部创建（demo.py 用 Fabric 起训练后直接 in-process 调 eval.main），
    # 这时**不能**由 eval 销毁它，否则训练侧后续的集合通信全炸。只有 eval 自己建的
    # 才由 eval 负责销毁。
    pg_owned_here = False
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
        pg_owned_here = True
    if world_size > 1:
        _rendezvous_snapshot("eval startup after init_process_group")
        _check_expected_world_size("eval startup after init_process_group")
        _hb("进程组就绪（rendezvous 完成）")

    device = f"cuda:{local_rank}"

    diag_active = log_kv_diag_mode not in (None, "off")
    pin_diag_recorder = (
        PinDiagRecorder(
            radius=log_kv_pin_diag_radius,
            max_samples=log_kv_pin_diag_max_samples,
            include_indices=log_kv_pin_diag_include_indices,
        )
        if log_kv_pin_diag_output is not None
        else None
    )
    if diag_active and log_kv_pin_size != 0:
        raise ValueError(
            f"log_kv_diag_mode={log_kv_diag_mode!r} requires log_kv_pin_size=0: "
            "salience pins scatter duplicate slots and break the diagnostic "
            "slot->token span mapping (see litgpt.log_kv_diag)."
        )

    # 多节点时用全局 rank==0（每个节点都有一个 local_rank 0，用它会重复打印）
    if _is_main():
        print(f"🚀 启动魔改版评估管线 | 任务: {benchmark}")
        print(
            f"🧩 logKV 压缩注意力 | B: {log_kv_B} | recent_size: {log_kv_recent_size} | "
            f"prefill_block: {log_kv_prefill_block} | pin: {log_kv_pin_size} "
            f"(obs {log_kv_pin_obs_window}) | second_order_scale: {log_kv_second_order_scale}"
        )
        if diag_active:
            print(
                f"🔬 诊断模式: {log_kv_diag_mode} | limit: {limit} | "
                f"exact_from_layer: {log_kv_diag_exact_from_layer} | "
                f"peak_window_from_end: {log_kv_diag_peak_window_from_end} | "
                f"second_order_max_width: {log_kv_diag_second_order_max_width} | "
                f"second_order_max_layer: {log_kv_diag_second_order_max_layer} | "
                "每 rank 各自累积统计量，不跨 rank 聚合"
            )
        if pin_diag_recorder is not None:
            print(
                f"📍 pin 诊断开启: output={log_kv_pin_diag_output} | "
                f"radius={log_kv_pin_diag_radius} | max_samples={log_kv_pin_diag_max_samples} | "
                f"include_indices={log_kv_pin_diag_include_indices}"
            )

    # eval_done：所有 rank 都跑完了 simple_evaluate（即全部集合通信都已结束）。
    # 只有这时收尾 barrier 才是安全的；某个 rank 中途抛异常时必须跳过 barrier，
    # 否则其余 rank 会干等到 NCCL 超时（12h）。
    eval_done = False
    try:
        checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)

        lm_model = LogKVLM(
            checkpoint_dir,
            device=device,
            config_overrides=config_overrides,
            log_kv_B=log_kv_B,
            log_kv_recent_size=log_kv_recent_size,
            log_kv_prefill_block=log_kv_prefill_block,
            log_kv_pin_size=log_kv_pin_size,
            log_kv_pin_obs_window=log_kv_pin_obs_window,
            log_kv_second_order_scale=log_kv_second_order_scale,
            tokenizer_dir=tokenizer_dir,
            pin_diag_recorder=pin_diag_recorder,
        )
        if world_size > 1:
            _hb("checkpoint + tokenizer + 模型加载完成，即将进入 simple_evaluate")

        with diag_mode(
            log_kv_diag_mode,
            exact_from_layer=log_kv_diag_exact_from_layer,
            peak_window_from_end=log_kv_diag_peak_window_from_end,
            second_order_max_width=log_kv_diag_second_order_max_width,
            second_order_max_layer=log_kv_diag_second_order_max_layer,
        ) if diag_active else contextlib.nullcontext():
            results = evaluator.simple_evaluate(
                model=lm_model,
                tasks=["piqa"] if benchmark == "debug" else benchmark.split(","),
                confirm_run_unsafe_code=True,
                batch_size=1,
                metadata=metadata,
                limit=limit,
            )
        eval_done = True

        if pin_diag_recorder is not None and _dist_ready():
            gathered_pin_samples = [None for _ in range(_world_size())]
            dist.all_gather_object(gathered_pin_samples, pin_diag_recorder.samples)
            pin_diag_recorder.merge_samples(gathered_pin_samples)

        # 落盘阶段：只有 rank 0 写文件（建目录、写 json/csv/xlsx）。其它 rank 什么都
        # 不做，直接到下面的 barrier 等 rank 0 写完 —— 各 rank 同时往共享盘写同名文件
        # 会互相截断，而 mkdir/exists 各判各的又会引入分支不一致。
        if _is_main():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 🌟 诊断汇总（score/value oracle 归因；见 litgpt.log_kv_diag）：每次调用
            # 对应一个 (task 类型, mode) 组合，独立落一份 JSON —— 不与常规 results 合并，
            # 也不跨 rank 聚合（诊断样本量小，单卡跑就够，见上面的启动打印）。
            if diag_active:
                diag_dir = Path(log_kv_diag_output or output_path or ".").expanduser()
                diag_dir.mkdir(parents=True, exist_ok=True)
                tag = log_kv_diag_mode
                if log_kv_second_order_scale != 1.0:
                    tag += f"_sos{log_kv_second_order_scale:g}"
                if log_kv_diag_exact_from_layer is not None:
                    tag += f"_efl{log_kv_diag_exact_from_layer}"
                if log_kv_diag_peak_window_from_end is not None:
                    tag += f"_pw{log_kv_diag_peak_window_from_end}"
                if log_kv_diag_second_order_max_width is not None:
                    tag += f"_w{log_kv_diag_second_order_max_width}"
                if log_kv_diag_second_order_max_layer is not None:
                    tag += f"_l{log_kv_diag_second_order_max_layer}"
                # checkpoint 目录名嵌进文件名——不同 checkpoint（比如 naive vs
                # warmup）的诊断产物即使不小心落进同一个 output_path，靠文件名
                # 也能分清，不用回头翻时间戳猜是哪次跑的（历史上 D5 数据被搞混过一次）。
                ckpt_name = Path(checkpoint_dir).name
                diag_file = diag_dir / f"diag_{ckpt_name}_{tag}_{benchmark.replace(',', '+')}_{ts}.json"
                with open(diag_file, "w", encoding="utf-8") as f:
                    json.dump(LOG_KV_DIAG.summary(), f, indent=2, ensure_ascii=False)
                print(f"🔬 诊断汇总已保存到: {diag_file}")

            if pin_diag_recorder is not None:
                pin_base = Path(log_kv_pin_diag_output).expanduser()
                if pin_base.suffix.lower() == ".json":
                    pin_file = pin_base
                else:
                    pin_base.mkdir(parents=True, exist_ok=True)
                    ckpt_name = Path(checkpoint_dir).name
                    pin_file = pin_base / f"pin_diag_{ckpt_name}_{benchmark.replace(',', '+')}_{ts}.json"
                pin_file.parent.mkdir(parents=True, exist_ok=True)
                pin_payload = {
                    "timestamp": ts,
                    "benchmark": benchmark,
                    "checkpoint_dir": checkpoint_dir,
                    "config": {
                        "log_kv_B": log_kv_B,
                        "log_kv_recent_size": log_kv_recent_size,
                        "log_kv_prefill_block": log_kv_prefill_block,
                        "log_kv_pin_size": log_kv_pin_size,
                        "log_kv_pin_obs_window": log_kv_pin_obs_window,
                        "log_kv_second_order_scale": log_kv_second_order_scale,
                        "radius": log_kv_pin_diag_radius,
                        "max_samples": log_kv_pin_diag_max_samples,
                        "include_indices": log_kv_pin_diag_include_indices,
                    },
                    "pin_diag": pin_diag_recorder.summary(),
                }
                with open(pin_file, "w", encoding="utf-8") as f:
                    json.dump(pin_payload, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
                print(f"📍 pin 诊断已保存到: {pin_file}")

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

    finally:
        # ── 收尾：所有 rank 在这里汇合，然后显式销毁进程组 ──
        # 这是"评测跑完之后要等很久"的直接修复点。之前的行为是：非 rank0 跑完
        # simple_evaluate 就直接走到函数末尾、跑到解释器退出，而 rank 0 还在写
        # json/csv/xlsx；进程组从头到尾没被销毁，退出时要等 NCCL watchdog 把
        # 通信子 abort 掉才肯收敛，启动器（torchrun / ModelArts）就一直挂在
        # "等最后一个 worker 退出"上。
        # 现在改成：rank 0 写完 → barrier 汇合 → 一起 destroy_process_group() →
        # 一起退出。
        # 这里的异常一律吞掉：finally 里抛出会顶掉 try 中真正的报错，把根因藏起来。
        if _dist_ready():
            if eval_done:
                t0 = time.perf_counter()
                try:
                    dist.barrier()
                    print(f"🤝 [{_rank_label()}] 收尾同步完成，用时 {time.perf_counter() - t0:.1f}s", flush=True)
                except Exception as e:  # noqa: BLE001 — 收尾阶段不掩盖主异常
                    print(f"⚠️ [{_rank_label()}] 收尾 barrier 失败（忽略）: {e}", flush=True)
            else:
                # 有 rank 异常退出：绝不能 barrier，否则其它 rank 等到 NCCL 超时。
                # 直接往下销毁进程组，让对端尽快收到通信中断而不是干等 12h。
                print(f"⚠️ [{_rank_label()}] 评测未正常结束，跳过收尾 barrier", flush=True)

        if pg_owned_here and dist.is_available() and dist.is_initialized():
            try:
                rank_before_destroy = _global_rank()
                dist.destroy_process_group()
                if rank_before_destroy == 0:
                    print("✅ 分布式进程组已销毁，评测进程可以正常退出了", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ [{_rank_label()}] 销毁进程组失败（忽略）: {e}", flush=True)


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
