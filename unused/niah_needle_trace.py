"""Where do RULER niah_single_2/3 needles end up in LogKV? (GPU diagnostic)

Runs lm-eval's own NIAH samples through the model and cache that eval.py builds from an experiment
YAML. For each sample it reports:

- whether greedy decoding reproduces the value (RULER's case-insensitive substring match);
- per layer, where the needle tokens sit after prefill: Alpha exact pool, recent window, or a
  ladder entry, with the mass w of the entry that covers each needle token in each KV group;
- for every flush in which the needle span is an Alpha candidate, its rank and the exact-pool
  tokens a pure density order would need to hold it ("tokens_needed": spans that outrank it plus
  the needle itself, with the same 1.1x bonus for spans already in the pool), under the cache's
  own score and under diagnostic scores that do not change the run:
    variance     Alpha's within-span K/V variance ratio, mean over KV groups
    novelty_tok  per-token distance to the nearest live centroid, top quarter of the span's tokens
    instruction  per-token max cosine to the instruction prefix keys, top quarter of the span
    rarity       running in-document unigram surprisal of the token ids, top quarter of the span
- with --oracle, the needle span is forced into the exact pool at the chosen layers, which bounds
  what better selection can buy.

Score prefetching is disabled while tracing so that every selection runs inside the route; the
selection itself is unchanged unless --oracle is given.

    python unused/niah_needle_trace.py --config exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml \
        --hf-tokenizer <Qwen3-1.7B-Base dir> --task niah_single_2 --samples 8 --out trace.jsonl
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import litgpt.alpha_log_kv as al  # noqa: E402
from litgpt.log_kv_cache import LogStructuredKVCache  # noqa: E402

NEEDLE_RE = re.compile(r"One of the special magic (?:numbers|uuids) for (\S+) is: (\S+)\.")
INSTRUCTION_END = "\n"  # The RULER instruction is the first line of the prompt.
SCORES = ("cache", "variance", "novelty_tok", "instruction", "rarity")
_ACTIVE: list = [None]  # The tracer the class-level patches report to.


def _top_quarter(values, lengths):
    """Mean of the top quarter (at least one) of each span's token values; values [G, n_tok]."""
    out = []
    for part in torch.split(values, lengths, dim=1):
        k = max(1, part.size(1) // 4)
        out.append(part.topk(k, dim=1).values.mean(1))
    return torch.stack(out, 1)  # [G, n_spans]


def _top_groups(per_group):
    return per_group.topk(min(2, per_group.size(0)), dim=0).values.mean(0)


class Tracer:
    """Wraps Alpha selection; records needle ranks and optionally forces the needle in."""

    def __init__(self, oracle: str):
        self.oracle_spec = oracle
        self.layer_of: dict[int, int] = {}
        self.current = None
        self.reset_sample(set(), 0, np.zeros(0, dtype=np.int64))

    def reset_sample(self, needle: set[int], instruction_len: int, ids: np.ndarray):
        self.needle, self.instruction_len, self.ids = needle, instruction_len, ids
        self.instruction_keys: dict[int, torch.Tensor] = {}
        self.flushes: dict[int, list[dict]] = {}

    def oracle(self, layer: int) -> bool:
        spec = self.oracle_spec
        if spec == "none":
            return False
        if spec == "all":
            return True
        if spec.startswith("from:"):
            return layer >= int(spec[5:])
        if spec.startswith("layers:"):
            return layer in {int(x) for x in spec[7:].split(",")}
        raise ValueError(f"bad --oracle {spec!r}")

    def install(self, model):
        """Map this model's caches to layers and make this tracer the active one."""
        self.layer_of = {id(block.attn.kv_cache): i for i, block in enumerate(model.transformer.h)}
        _ACTIVE[0] = self
        if getattr(LogStructuredKVCache, "_niah_trace_installed", False):
            return
        route, select = LogStructuredKVCache._alpha_route_flush, al.select_spans

        def traced_route(cache, *args, **kwargs):
            _ACTIVE[0].current = cache
            try:
                return route(cache, *args, **kwargs)
            finally:
                _ACTIVE[0].current = None

        LogStructuredKVCache._alpha_route_flush = traced_route
        LogStructuredKVCache._alpha_prefetch_scores = lambda cache: None
        LogStructuredKVCache._niah_trace_installed = True
        al.select_spans = lambda *args, **kwargs: _ACTIVE[0].select(*args, **kwargs)

    def select(self, k, v, old_spans, old_positions, new_positions, ends, old_width, budget, max_span,
               *, beta_novelty=False, centroids=None, centroid_valid=None):
        cut = al.cut_spans(old_spans, old_positions, new_positions, ends, old_width, max_span)
        if cut is None:
            return al._empty_selection(len(ends))
        scores = al.score_spans(k, v, cut, beta_novelty=beta_novelty, centroids=centroids,
                                centroid_valid=centroid_valid)
        host = scores.float().cpu().tolist()
        layer = self.layer_of.get(id(self.current), -1)
        row, all_pos = cut.candidates[0]
        tokens = cut.token_rows[0]
        lengths = list(cut.length_rows[0])
        span_pos = np.split(all_pos[tokens], np.cumsum(lengths)[:-1])
        needle_spans = [i for i, pos in enumerate(span_pos) if self.needle.intersection(pos.tolist())]
        needle_tokens = set(np.flatnonzero(np.isin(all_pos, list(self.needle))).tolist()) if needle_spans else set()
        chosen = list(host)
        if layer >= 0 and needle_spans and self.oracle(layer):
            for i in needle_spans:
                chosen[i] = 1e30
        selection = al.choose_spans(cut, chosen, budget)
        if layer >= 0:
            self._record(layer, k, v, cut, row, span_pos, lengths, host, needle_spans, needle_tokens,
                         selection, centroids, centroid_valid, beta_novelty)
        return selection

    @torch.no_grad()
    def _record(self, layer, k, v, cut, row, span_pos, lengths, host, needle_spans, needle_tokens,
                selection, centroids, centroid_valid, beta_novelty):
        tokens = torch.as_tensor(cut.token_rows[0], device=k.device)
        positions = np.concatenate(span_pos)
        if layer not in self.instruction_keys and self.instruction_len:
            early = torch.as_tensor(np.flatnonzero(positions < self.instruction_len), device=k.device)
            if early.numel():
                keys = k[0][:, tokens[early]].float()
                self.instruction_keys[layer] = torch.nn.functional.normalize(keys, dim=-1)
        if not needle_spans:
            return
        x = k[0][:, tokens].float()  # [G, n_tok, D]
        scores = {"cache": torch.tensor(host, device=k.device)}
        if beta_novelty:
            variance = al.score_spans(k, v, cut, beta_novelty=False)
            scores["variance"] = variance.float()
        else:
            scores["variance"] = scores["cache"]
        if centroids is not None and centroid_valid is not None and bool(centroid_valid[0].any()):
            c, valid = centroids[0].float(), centroid_valid[0]
            energy = x.square().mean(-1).clamp_min(1e-12)
            dist = energy[..., None] + c.square().mean(-1)[:, None, :] - 2 * (x @ c.transpose(1, 2)) / x.size(-1)
            dist = dist.masked_fill(~valid[:, None, :], float("inf")).amin(-1).clamp_min(0) / energy
            scores["novelty_tok"] = _top_groups(_top_quarter(dist, lengths))
        if layer in self.instruction_keys:
            sim = torch.nn.functional.normalize(x, dim=-1) @ self.instruction_keys[layer].transpose(1, 2)
            scores["instruction"] = _top_groups(_top_quarter(sim.amax(-1), lengths))
        seen = self.ids[: int(positions.max()) + 1]
        counts = np.bincount(seen, minlength=int(self.ids.max()) + 1)
        distinct = max(int((counts > 0).sum()), 1)
        surprisal = -np.log((counts[self.ids[positions]] + 0.5) / (len(seen) + 0.5 * distinct))
        scores["rarity"] = _top_groups(_top_quarter(
            torch.as_tensor(surprisal, dtype=torch.float32, device=k.device)[None], lengths))
        length_t = torch.as_tensor(lengths, dtype=torch.float32, device=k.device)
        # choose_spans favours spans already in the pool by 1.1x; rank every score the same way.
        bonus = torch.as_tensor([1.1 if old else 1.0 for _, old, _ in row], device=k.device)
        needle_len = int(sum(lengths[i] for i in needle_spans))
        kept = set(selection.keep[0])
        entry = {
            "flush_end": int(positions.max()), "candidates": len(lengths),
            "needle_old": bool(any(row[i][1] for i in needle_spans)),
            "needle_kept": bool(needle_tokens) and needle_tokens <= kept, "needle_len": needle_len,
            "ranks": {}, "tokens_needed": {},
        }
        for name, value in scores.items():
            density = value.float() * bonus
            best = max(float(density[i]) for i in needle_spans)
            above = density > best
            entry["ranks"][name] = int(above.sum()) + 1
            # Exact-pool tokens a pure density order needs to hold the needle span.
            entry["tokens_needed"][name] = int(length_t[above].sum()) + needle_len
        self.flushes.setdefault(layer, []).append(entry)


@torch.no_grad()
def needle_fate(cache, needle_positions, value_positions):
    """Where the needle tokens are now: exact pool, recent window, or ladder entries (min w per group)."""
    exact = set(cache._alpha_positions[0]) if getattr(cache, "alpha_exact_tokens", 0) else set()
    recent = set(cache._recent_pos_host[0][:cache.recent_count])
    ladder = [p for p in needle_positions if p not in exact and p not in recent]
    widths = []
    if ladder:
        w, lo, hi = cache.level_w[0], cache.level_p_lo[0], cache.level_p_hi[0]  # [G, K, L, B]
        live = w > 0
        for p in ladder:
            cover = live & (lo <= p) & (hi >= p)
            best = torch.where(cover, w, torch.full_like(w, float("inf"))).flatten(1).amin(1)
            widths.extend(best[torch.isfinite(best)].tolist())
    value_exact = sum(p in exact or p in recent for p in value_positions)
    return {
        "exact": sum(p in exact for p in needle_positions), "recent": sum(p in recent for p in needle_positions),
        "tokens": len(needle_positions), "value_exact": value_exact, "value_tokens": len(value_positions),
        "ladder_w_median": float(np.median(widths)) if widths else None,
        "ladder_w_max": float(max(widths)) if widths else None,
    }


def locate(prompt, offsets):
    """Token indices of the needle sentence and of its value, from character offsets."""
    match = NEEDLE_RE.search(prompt)
    if match is None:
        raise RuntimeError("needle sentence not found in prompt")
    s, e = match.span()
    vs, ve = match.span(2)
    needle = [i for i, (a, b) in enumerate(offsets) if a < e and b > s]
    value = [i for i, (a, b) in enumerate(offsets) if a < ve and b > vs]
    return needle, value, match.group(2), s / max(len(prompt), 1)


def trace_sample(model, tracer, ids, offsets, prompt, answers, decode, generate, reset, max_new):
    needle, value_tokens, value, depth = locate(prompt, offsets)
    instruction_len = next((i for i, (a, b) in enumerate(offsets) if prompt[a:b].count(INSTRUCTION_END)), 0) + 1
    tracer.reset_sample(set(needle), instruction_len, np.asarray(ids, dtype=np.int64))
    snapshot = {}
    forward = model.forward

    def first_forward(*args, **kwargs):
        out = forward(*args, **kwargs)
        if not snapshot:  # The first call is the prefill.
            for i, block in enumerate(model.transformer.h):
                snapshot[i] = needle_fate(block.attn.kv_cache, needle, value_tokens)
        return out

    model.forward = first_forward
    try:
        out = generate(torch.as_tensor(ids, device=next(model.parameters()).device), max_new)
    finally:
        model.forward = forward
        reset()
    text = decode(out[len(ids):])
    correct = sum(a.lower() in text.lower() for a in answers) / max(len(answers), 1)
    return {
        "depth": round(depth, 4), "tokens": len(ids), "needle_tokens": len(needle), "value": value,
        "correct": correct, "generated": text[:200],
        "layers": [{"layer": i, **snapshot.get(i, {}), "flushes": tracer.flushes.get(i, [])}
                   for i in range(len(model.transformer.h))],
    }


def summarize(records, budgets=(256, 1024, 2048)):
    n = len(records)
    if not n:
        return
    print(f"samples={n} accuracy={sum(r['correct'] for r in records) / n:.3f}")
    layers = len(records[0]["layers"])
    exact_all = [np.mean([r["layers"][i].get("exact", 0) + r["layers"][i].get("recent", 0)
                          >= r["layers"][i].get("tokens", 1) for r in records]) for i in range(layers)]
    print("fraction of samples whose whole needle is exact (pool or recent), per layer:")
    print("  " + " ".join(f"{x:.2f}" for x in exact_all))
    print("needle span when it first competes (over samples x layers): pool tokens a density order needs")
    for name in SCORES:
        first = [f["tokens_needed"][name] for r in records for layer in r["layers"]
                 for f in layer["flushes"][:1] if name in f["tokens_needed"]]
        if first:
            fit = " ".join(f"P{b}:{np.mean([t <= b for t in first]):.2f}" for b in budgets)
            print(f"  {name:12s} median={int(np.median(first)):6d}  fraction that fits {fit}")


def build_lm(args):
    ev = importlib.import_module("eval")
    cfg = ev._coerce_yaml_sci_floats(ev.expand_env_vars(
        ev._load_yaml_config(os.path.join(os.getcwd(), args.config), args.checkpoint or "") or {}))
    defaults = {k: p.default for k, p in inspect.signature(ev.main).parameters.items()}
    kwargs = {}
    for name in inspect.signature(ev.LogKVLM.__init__).parameters:
        if name.startswith("log_kv_") and name in defaults:
            value = cfg.get(name)
            kwargs[name] = defaults[name] if value is None else value
    checkpoint = ev._resolve_checkpoint_dir(args.checkpoint or cfg["save_path"])
    lm = ev.LogKVLM(checkpoint, device=args.device, config_overrides=cfg.get("config_overrides"),
                    tokenizer_dir=cfg.get("tokenizer_dir"), **kwargs)
    return ev, lm


def ruler_samples(args, hf_tokenizer):
    from lm_eval.tasks.ruler.niah_utils import TEMPLATE
    from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack
    value_type = {"niah_single_2": "numbers", "niah_single_3": "uuids"}[args.task]
    return generate_samples(get_haystack(type_haystack="essay"), max_seq_length=args.length, template=TEMPLATE,
                            type_haystack="essay", type_needle_k="words", type_needle_v=value_type,
                            num_samples=args.samples, TOKENIZER=hf_tokenizer)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment YAML, as for eval.sh")
    p.add_argument("--checkpoint", default=None, help="defaults to the YAML save_path (latest step_*)")
    p.add_argument("--hf-tokenizer", required=True, help="HF tokenizer dir used by lm-eval to build samples")
    p.add_argument("--task", choices=("niah_single_2", "niah_single_3"), default="niah_single_2")
    p.add_argument("--length", type=int, default=32768)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--oracle", default="none", help="none | all | from:<layer> | layers:<a,b,...>")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="niah_needle_trace.jsonl")
    args = p.parse_args()

    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    ev, lm = build_lm(args)
    if not getattr(lm, "log_kv_alpha_exact_tokens", 0):
        print("note: Alpha is off in this config; only needle fate is traced")
    tracer = Tracer(args.oracle)

    def generate(prompt, max_new):
        lm._set_eval_cache()
        tracer.install(lm.model)
        return ev.litgpt_generate(lm.model, prompt, max_returned_tokens=prompt.size(0) + max_new,
                                  temperature=0.0, top_k=None, top_p=0.0, eos_id=lm.tokenizer.eos_id)

    records = []
    with open(args.out, "w") as out:
        for sample in ruler_samples(args, hf_tokenizer):
            prefix = sample.get("gen_prefix") or ""
            prompt = sample["input"] + (" " + prefix if prefix else "")  # lm-eval's plain-text context
            ids = lm.tokenizer.encode(prompt).tolist()
            encoded = hf_tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            shift = len(ids) - len(encoded["input_ids"])
            if shift < 0 or ids[shift:] != list(encoded["input_ids"]):
                raise RuntimeError("the eval tokenizer and the HF tokenizer disagree on this prompt")
            offsets = [(0, 0)] * shift + [tuple(o) for o in encoded["offset_mapping"]]
            with torch.no_grad():
                record = trace_sample(lm.model, tracer, ids, offsets, prompt, sample["outputs"],
                                      lm.tokenizer.decode, generate, lm._reset_eval_cache, args.max_new)
            record.update(task=args.task, length=args.length, oracle=args.oracle, index=sample["index"])
            out.write(json.dumps(record) + "\n")
            out.flush()
            records.append(record)
            print(f"sample {sample['index']}: depth={record['depth']:.2f} correct={record['correct']} "
                  f"generated={record['generated'][:60]!r}", flush=True)
    summarize(records)


if __name__ == "__main__":
    main()
