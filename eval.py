"""
Evaluation pipeline for logKV models.

Supports lm-evaluation-harness benchmarks with distributed inference.
Uses standard causal LM forward (no research architecture).

Usage:
    # Single GPU
    python eval.py --checkpoint_dir ./ckpt/qwen0.6b-32k-cpt-base --benchmark piqa

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 eval.py --checkpoint_dir ./ckpt/... --benchmark "boolq,piqa,..."

    # With YAML config
    python eval.py --config exp/qwen0.6b-32k/eval.yaml
"""

from __future__ import annotations

import os
import re
import csv
import json
import inspect
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from typing import Any

from jsonargparse import CLI
import tqdm

from utils import auto_expand_env_vars

# ── Optional: offload large files to cloud storage ──
if "HF_DATASETS_CACHE" not in os.environ and "PKU" not in os.environ:
    BASE = os.environ.get("HF_CACHE_BASE", os.path.expanduser("~/.cache/huggingface"))
    os.environ["HF_HOME"] = BASE
    os.environ["HF_DATASETS_CACHE"] = f"{BASE}/hf_cache"
    os.environ["HF_EVALUATE_CACHE"] = f"{BASE}/evaluate"
    os.environ["HF_MODULES_CACHE"] = f"{BASE}/modules"
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{BASE}/hub"
    os.environ["HF_HUB_CACHE"] = f"{BASE}/hub"
    os.environ["RULER_CACHE_DIR"] = f"{BASE}/ruler_cache"
    os.environ["NLTK_DATA"] = f"{BASE}/nltk_data"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import dataclasses

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
from lm_eval import evaluator
from lm_eval.api.model import LM
from litgpt.generate.base import generate as litgpt_generate

# RULER benchmark monkey-patch (long-context needle-in-haystack tasks)
try:
    from litgpt.ruler_patch import apply_patch
    apply_patch()
except ImportError:
    pass

_CONFIG_FIELDS = {f.name for f in dataclasses.fields(Config)}


class SafeJSONEncoder(json.JSONEncoder):
    """Handles non-serializable objects (numpy, torch, callables)."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if callable(obj):
            return f"<function: {obj.__name__ if hasattr(obj, '__name__') else str(obj)}>"
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if hasattr(obj, "__dict__"):
            return f"<{obj.__class__.__name__} object>"
        return super().default(obj)


def _load_lit_model_checkpoint(checkpoint_dir: str, map_location: str | torch.device) -> Any:
    """Load lit_model.pth (single-file torch.save, compatible with FSDP state_dict_type='full')."""
    lit_path = Path(checkpoint_dir).expanduser() / "lit_model.pth"
    if not lit_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {lit_path}")
    if lit_path.is_dir():
        raise FileNotFoundError(
            f"{lit_path} is a directory (old FSDP/DCP sharded format). "
            "Merge it first: utils.convert_and_replace_fsdp_ckpt"
        )
    return torch.load(str(lit_path), map_location=map_location, weights_only=False)


def _filter_config_dict(d: dict[str, Any]) -> dict[str, Any]:
    dropped = sorted(k for k in d if k not in _CONFIG_FIELDS)
    if dropped:
        print(f"[eval] Filtering non-Config fields: {dropped}")
    return {k: v for k, v in d.items() if k in _CONFIG_FIELDS}


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


def _config_from_yaml_and_overrides(config_path: str, overrides: dict[str, Any] | None) -> Config:
    with open(config_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if base is None:
        raise ValueError(f"{config_path} is empty or invalid YAML.")
    merged = {**base, **(overrides or {})}
    merged = _filter_config_dict(merged)
    return Config(**merged)


# Preferred metric keys, tried in order. Covers multiple-choice (acc_norm/acc),
# generation & QA (exact_match, f1, rouge...), and LongBench-style custom metrics.
_METRIC_PREFERENCE = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "f1,none",
    "qa_f1_score,none",
    "rouge_l,none",
    "rougeL,none",
    "word_perplexity,none",
    "bleu,none",
)


def _pick_task_metric(task_results: dict) -> tuple[str | None, Any]:
    """Pick the most meaningful (metric_key, value) from a task's results.

    Falls back to the first numeric non-stderr metric, so tasks with custom
    metric names (truthfulqa_gen, LongBench, ...) still land in the CSV instead
    of a hardcoded 'acc,none' miss producing None. (The old code also matched
    'arc-c' which never occurs — the actual task name is 'arc_challenge'.)
    """
    for key in _METRIC_PREFERENCE:
        if key in task_results:
            return key, task_results[key]
    for key, value in task_results.items():
        if key == "alias" or "stderr" in key:
            continue
        if isinstance(value, (int, float)):
            return key, value
    return None, None


def extract_results_to_csv(results: dict, benchmark: str, output_path: Path) -> None:
    """Write one row per reported task with the best-available metric."""
    results_data = results.get("results", {})

    csv_rows = []
    for task in sorted(results_data):
        metric_key, value = _pick_task_metric(results_data[task] or {})
        csv_rows.append({"dataset": task, "metric": metric_key, "value": value})

    csv_file = output_path.with_suffix(".csv")
    try:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "metric", "value"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV results saved to: {csv_file}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")


class LogKVLM(LM):
    """LM wrapper for a causal LM with an optional log-structured KV cache.

    When ``use_log_kv`` is set, both loglikelihood scoring and generate_until run
    through the log-structured (compressed) KV attention, matching how a
    logKV-adapted model was trained. Otherwise both use dense attention.
    Loglikelihood scores a fixed sequence in a single full forward (for lengths
    within ``recent_size`` this is identical to dense causal attention);
    generation streams prefill + decode.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        config_overrides: dict[str, Any] | None = None,
        use_log_kv: bool = False,
        log_kv_B: int = 512,
        log_kv_recent_size: int = 1024,
    ):
        super().__init__()
        self._device = device
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = Tokenizer(checkpoint_dir)
        self.use_log_kv = use_log_kv
        self.log_kv_B = log_kv_B
        self.log_kv_recent_size = log_kv_recent_size

        is_master = not dist.is_initialized() or dist.get_rank() == 0

        # ── Config loading ──
        config_path = os.path.join(checkpoint_dir, "model_config.yaml")

        if os.path.exists(config_path):
            if is_master:
                print(f"Loading model config from: {config_path}")
            self.config = _config_from_yaml_and_overrides(config_path, config_overrides)
        else:
            if is_master:
                print("No model_config.yaml found, using fallback config.")
            fallback_kw: dict[str, Any] = {"name": checkpoint_dir.split("/")[-1]}
            if config_overrides:
                fallback_kw.update(config_overrides)
            fallback_kw = _filter_config_dict(fallback_kw)
            self.config = Config.from_name(**fallback_kw)

        if is_master:
            print(f"Initializing model: {self.config.name}")
        self.model = GPT(self.config).to(device).bfloat16()

        if is_master:
            print("Loading weights...")
        checkpoint = _load_lit_model_checkpoint(checkpoint_dir, map_location=device)

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            if is_master:
                print("Detected Fabric checkpoint, extracted model weights.")
        else:
            state_dict = checkpoint

        load_result = self.model.load_state_dict(state_dict, strict=False)

        if is_master:
            print(f"Weights loaded. Missing keys: {len(load_result.missing_keys)}")
            if len(load_result.missing_keys) > 100:
                print("WARNING: Many missing keys — checkpoint may be incompatible.")

        self.model.eval()

    # ── Distributed result gathering ──

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

    # ── Loglikelihood (PPL + multiple-choice) ──

    def _score_tokens(self, ctx_enc: list[int], cont_enc: list[int]) -> tuple[float, bool]:
        """Forward (context + continuation) and return
        ``(sum log p(continuation | context), is_greedy)``.

        Left-truncates so the sequence fits ``max_seq_length``. If the
        continuation alone meets/exceeds the window, the context is dropped and
        the continuation is left-truncated to its last ``max_len - 1`` tokens —
        scoring is then partial but the forward stays in bounds. (Previously a
        negative ``keep_ctx_len`` sliced the wrong way and left the input longer
        than the model, which raised at forward time.)
        """
        max_len = self.model.max_seq_length
        if len(ctx_enc) + len(cont_enc) > max_len:
            keep_ctx_len = max_len - len(cont_enc)
            if keep_ctx_len <= 0:
                cont_enc = cont_enc[-(max_len - 1):]
                ctx_enc = []
            else:
                ctx_enc = ctx_enc[-keep_ctx_len:]
        if len(ctx_enc) == 0:
            ctx_enc = [self.tokenizer.bos_id]

        inps = torch.tensor([ctx_enc + cont_enc], dtype=torch.long, device=self._device)
        seq_len = inps.size(1)
        ctx_len = len(ctx_enc)

        with torch.no_grad():
            if self.use_log_kv:
                # Score under the same log-structured (compressed) KV attention the
                # model is adapted to. Within recent_size this equals dense causal
                # attention (no compaction); the LogKV fast path keeps it cheap.
                self.model.set_log_kv_cache(
                    batch_size=1, max_seq_length=seq_len, device=self._device,
                    dtype=next(self.model.parameters()).dtype,
                    B=self.log_kv_B, recent_size=self.log_kv_recent_size,
                )
                try:
                    logits = self.model(inps, input_pos=torch.arange(seq_len, device=self._device))
                finally:
                    self.model.clear_kv_cache()
            else:
                # Dense full-attention scoring: one full forward, no KV cache needed.
                logits = self.model(inps)

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
        disable_tqdm = dp_rank != 0

        for req in tqdm.tqdm(local_requests, desc=f"Rank {dp_rank}", position=dp_rank, disable=disable_tqdm):
            context, continuation = req.args[0], req.args[1]
            ctx_enc = self.tokenizer.encode(context).tolist()
            cont_enc = self.tokenizer.encode(continuation, bos=False).tolist()
            results.append(self._score_tokens(ctx_enc, cont_enc))

        return self.all_gather_results(results)

    # ── Generate (long-form generation tasks, e.g. LongBench) ──

    def generate_until(self, requests):
        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        dp_size = dist.get_world_size() if dist.is_initialized() else 1

        local_requests = requests[dp_rank::dp_size]
        results = []
        disable_tqdm = dp_rank != 0

        for req in tqdm.tqdm(local_requests, desc=f"Rank {dp_rank}", position=dp_rank, disable=disable_tqdm):
            prompt = req.args[0]
            gen_args = req.args[1]

            max_new_tokens = int(gen_args.get("max_gen_toks", gen_args.get("max_length", self.max_gen_toks)))
            do_sample = bool(gen_args.get("do_sample", False))
            temperature = float(gen_args.get("temperature", 1.0))
            top_p = float(gen_args.get("top_p", 1.0))
            top_k = gen_args.get("top_k", None)
            if not do_sample:
                temperature = 0.0
                top_p = 0.0

            prompt_tensor = self.tokenizer.encode(prompt, device=self._device)

            # Safety: reserve space for generated tokens
            max_len = self.model.max_seq_length
            if prompt_tensor.size(0) + max_new_tokens > max_len:
                keep_prompt_len = max_len - max_new_tokens
                prompt_tensor = prompt_tensor[-keep_prompt_len:]
                if dp_rank == 0:
                    print(f"Warning: prompt truncated to {keep_prompt_len}")

            total_max_len = prompt_tensor.size(0) + max_new_tokens

            with torch.no_grad():
                if self.use_log_kv:
                    # Use log-structured KV cache for long-context generation
                    self.model.set_log_kv_cache(
                        batch_size=1, max_seq_length=total_max_len, device=self._device,
                        dtype=next(self.model.parameters()).dtype,
                        B=self.log_kv_B, recent_size=self.log_kv_recent_size
                    )
                else:
                    self.model.set_kv_cache(batch_size=1, max_seq_length=total_max_len, device=self._device)

                try:
                    out = litgpt_generate(
                        self.model,
                        prompt_tensor,
                        max_returned_tokens=total_max_len,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        eos_id=self.tokenizer.eos_id,
                    )
                finally:
                    self.model.clear_kv_cache()

            generated_tokens = out[prompt_tensor.size(0):]
            decoded = self.tokenizer.decode(generated_tokens)
            results.append(decoded)

        return self.all_gather_results(results)

    def loglikelihood_rolling(self, requests):
        """Rolling log-likelihood over full documents (lm-eval PPL tasks, e.g.
        wikitext). Returns one summed log-prob (scalar float) per request,
        matching lm-eval's reference implementations — returning tuples here
        breaks perplexity aggregation downstream.

        Long documents are scored in non-overlapping windows of max_seq_length:
        each window's tokens are conditioned on the window prefix (the first
        window starts from BOS). This matches lm-eval's standard rolling-window
        scoring; the previous ``pass`` stub returned None and broke PPL tasks.
        """
        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        dp_size = dist.get_world_size() if dist.is_initialized() else 1

        local_requests = requests[dp_rank::dp_size]
        results = []
        disable_tqdm = dp_rank != 0

        max_len = self.model.max_seq_length
        for req in tqdm.tqdm(local_requests, desc=f"Rank {dp_rank}", position=dp_rank, disable=disable_tqdm):
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

        return self.all_gather_results(results)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self.model.max_seq_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string):
        return self.tokenizer.encode(string).tolist()

    def tok_decode(self, tokens):
        return self.tokenizer.decode(torch.tensor(tokens))


@auto_expand_env_vars
def main(
    checkpoint_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    benchmark: str = "debug",
    config_overrides: dict[str, Any] | None = None,
    output_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    # ── Log-structured KV cache ──
    use_log_kv: bool = False,
    log_kv_B: int = 512,
    log_kv_recent_size: int = 1024,
    # ── YAML config ──
    config: str | None = None,
):
    # ── Load YAML config if provided ──
    # NOTE: `locals()[k] = v` does NOT write back to a function's real locals in
    # CPython, so YAML overrides must be applied by explicit re-binding. A YAML
    # value overrides the CLI/default value unless it is null (None).
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
    use_log_kv = _o("use_log_kv", use_log_kv)
    log_kv_B = _o("log_kv_B", log_kv_B)
    log_kv_recent_size = _o("log_kv_recent_size", log_kv_recent_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))

    device = f"cuda:{local_rank}"

    if local_rank == 0:
        print(f"Starting eval | benchmark: {benchmark} | log_kv: {use_log_kv}")

    lm_model = LogKVLM(
        checkpoint_dir,
        device=device,
        config_overrides=config_overrides,
        use_log_kv=use_log_kv,
        log_kv_B=log_kv_B,
        log_kv_recent_size=log_kv_recent_size,
    )

    results = evaluator.simple_evaluate(
        model=lm_model,
        tasks=["piqa"] if benchmark == "debug" else benchmark.split(","),
        confirm_run_unsafe_code=True,
        batch_size=1,
        metadata=metadata,
    )

    is_main = not dist.is_initialized() or dist.get_rank() == 0

    if is_main:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Cache raw results
        results_cache_file = Path("eval_results_cache.json")
        try:
            with open(results_cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
            print(f"Results cached to: {results_cache_file}")
        except Exception as e:
            print(f"Failed to cache results: {e}")

        # Print table
        try:
            from lm_eval.utils import make_table
            print(make_table(results))
        except Exception:
            pass

        # Save output if requested
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
                print(f"Results saved to: {output_file}")
            except Exception as e:
                print(f"Failed to save results: {e}")

            extract_results_to_csv(results, benchmark, output_file)


@auto_expand_env_vars
def output_from_cache(
    cache_file: str = "eval_results_cache.json",
    benchmark: str = "debug",
    checkpoint_dir: str = "checkpoints/Qwen/Qwen3-0.6B-Base",
    output_path: str | None = None,
):
    """Re-process cached eval results without re-running evaluation."""
    cache_path = Path(cache_file)
    if not cache_path.exists():
        print(f"Cache file not found: {cache_path}")
        return

    try:
        with open(cache_path, encoding="utf-8") as f:
            cache_data = json.load(f)

        if "results_str" in cache_data:
            print("Cache is string form, cannot restore structured data.")
            return

        results = cache_data
        print(f"Loaded results from cache: {cache_path}")
    except Exception as e:
        print(f"Failed to read cache: {e}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        from lm_eval.utils import make_table
        print(make_table(results))
    except Exception:
        pass

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
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save results: {e}")

        extract_results_to_csv(results, benchmark, output_file)


if __name__ == "__main__":
    CLI(main)
