"""Would LM surprisal keep the NIAH needle? (GPU diagnostic, no training, does not change any run)

Each token's surprisal, s[t] = -log2 p(ids[t] | ids[:t]) in bits, is the model's own code length for
it: what it cannot predict from context, the definition of an incompressible token. This script
builds the model and cache eval.py builds from an experiment YAML, prefills lm-eval's own NIAH
prompts (RULER's template and the eval_tasks/niah_cue variants), and computes every token's
surprisal from the final hidden states in chunks (logits are never held for the whole prompt).

It then cuts the prompt into Alpha's candidate spans (natural boundaries, span cap) and, for
several span scores built from token surprisal, asks where the needle's span ranks among the spans
that leave the recent window:

    max     the span's largest token surprisal
    top4    the mean of its four largest
    mean    its mean token surprisal
    uni_max the largest running in-document unigram code length (no LM; a rarity reference)

"tokens_needed" is the exact-pool budget a pure score order needs to hold the needle span: the
tokens of every span that outranks it plus its own. It is an offline estimate of the final pool
(the streaming pool also gives kept spans a 1.1x bonus); a needle inside the last recent-window
tokens is kept whatever the scores. The record also lists the spans that outrank the needle the
most, to show what competes with it.

Launch it in the training environment with diag.sh (majob.sh's environment, one process):

    bash diag.sh unused/niah_surprisal.py --config exp/qwen1.7b-32k/arc_gamma_k12_b128_p1024_2k.yaml \\
        --tasks niah_single_2,niah_single_2_words,niah_single_2_words_qend --samples 32 --out surprisal.jsonl

Every run writes its per-task summary to ``<out stem>.summary.txt`` beside ``--out``;
``--report <out>.jsonl`` rebuilds that summary from a finished run without the model.

``--save-surprisal <file>.npz`` also keeps every prompt's ids and token surprisal (with the needle's
positions and the cache settings) for the CPU replay of exact-pool policies, unused/pool_policy_sim.py.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import litgpt.alpha_log_kv as al  # noqa: E402

NEEDLE_RE = re.compile(r"One of the special magic (?:numbers|uuids|words) for (\S+) is: (\S+)\.")
SCORES = ("max", "top4", "mean", "sum", "uni_max")
BUDGETS = (256, 1024, 2048)
# Task name -> RULER generator settings; a suffix _qend / _preamble / _paraphrase picks the niah_cue template.
# The multi-needle tasks are lm-eval's (niah_utils): four keys, four values for one key, or four queried keys.
BASES = {"niah_single_1": dict(type_haystack="repeat", type_needle_v="numbers"),
         "niah_single_2": dict(type_haystack="essay", type_needle_v="numbers"),
         "niah_single_3": dict(type_haystack="essay", type_needle_v="uuids"),
         "niah_single_2_words": dict(type_haystack="essay", type_needle_v="words"),
         "niah_multikey_1": dict(type_haystack="essay", type_needle_v="numbers", num_needle_k=4),
         "niah_multivalue": dict(type_haystack="essay", type_needle_v="numbers", num_needle_v=4),
         "niah_multiquery": dict(type_haystack="essay", type_needle_v="numbers", num_needle_q=4)}
VARIANTS = ("qend", "preamble", "paraphrase")


def token_surprisal(model, ids, prepare=None, chunk: int = 1024) -> np.ndarray:
    """Bits per token, s[t] = -log2 p(ids[t] | ids[:t]) (s[0] = nan), from one prefill of ``ids``.

    ``prepare`` (e.g. eval's cache setup) runs first and the prefill then uses the cache; without it
    the model runs uncached. Only the last position's logits leave the forward; the final hidden
    states are captured and projected ``chunk`` positions at a time.
    """
    device = next(model.parameters()).device
    x = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)[None]
    captured = {}
    hook = model.transformer.h[-1].register_forward_hook(lambda module, inputs, out: captured.update(h=out))
    try:
        with torch.no_grad():
            if prepare is not None:
                prepare()
                model(x, input_pos=torch.arange(x.size(1), device=device), lm_head_start=x.size(1) - 1)
            else:
                model(x, lm_head_start=x.size(1) - 1)
    finally:
        hook.remove()
    h, targets = captured["h"][0], x[0, 1:]
    softcap = model.config.final_logit_softcapping
    out = np.full(x.size(1), np.nan)
    with torch.no_grad():
        for start in range(0, x.size(1) - 1, chunk):
            stop = min(start + chunk, x.size(1) - 1)
            logits = model.lm_head(model.transformer.ln_f(h[start:stop])).float()
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            logp = logits.log_softmax(-1).gather(-1, targets[start:stop, None])[:, 0]
            out[start + 1:stop + 1] = (-logp / math.log(2)).cpu().numpy()
    return out


def unigram_code_length(ids, vocab: int) -> np.ndarray:
    """Running in-document unigram code length in bits (Krichevsky-Trofimov over a ``vocab``-token alphabet)."""
    ids = np.asarray(ids, dtype=np.int64)
    counts: dict[int, int] = {}
    out = np.empty(len(ids))
    for t, token in enumerate(ids.tolist()):
        out[t] = -math.log2((counts.get(token, 0) + 0.5) / (t + 0.5 * vocab))
        counts[token] = counts.get(token, 0) + 1
    return out


def cut(ids, boundary, max_span: int, extra_ends=None) -> tuple[np.ndarray, np.ndarray]:
    """Alpha's candidate spans [start, stop) over the whole prompt: natural boundaries (plus
    ``extra_ends``, e.g. information spans) and the span cap."""
    ids = np.asarray(ids, dtype=np.int64)
    ends = boundary[ids] if extra_ends is None else boundary[ids] | np.asarray(extra_ends, dtype=bool)
    _, starts, stops, _ = al._new_runs(np.arange(len(ids)), ends, 0, 0, max_span)
    return starts, stops


def span_scores(values, starts, stops) -> dict[str, np.ndarray]:
    """Per-span scores from per-token surprisal; nan tokens (the first one) are ignored."""
    out = {name: np.full(len(starts), -np.inf) for name in ("max", "top4", "mean", "sum")}
    for i, (a, b) in enumerate(zip(starts, stops)):
        part = values[a:b]
        part = part[np.isfinite(part)]
        if len(part):
            out["max"][i] = part.max()
            out["top4"][i] = np.sort(part)[-4:].mean()
            out["mean"][i] = part.mean()
            out["sum"][i] = part.sum()  # the span's code length, -log2 p(span)
    return out


def needle_rank(scores, starts, stops, target: int, archived) -> dict:
    """Rank and tokens_needed of span ``target`` among the ``archived`` spans (bool mask)."""
    lengths = stops - starts
    better = archived & (scores > scores[target])
    better[target] = False
    return dict(rank=int(better.sum()) + 1, tokens_needed=int(lengths[better].sum() + lengths[target]),
                percentile=round(float((scores[archived] <= scores[target]).mean()), 4),
                score=round(float(scores[target]), 3))


def locate(prompt, offsets, asked=()):
    """Every needle in prompt order: its sentence's and its value's token indices, the value, and whether
    the question asks for it (``asked``: the sample's answers); plus the first needle's relative depth."""
    needles = []
    for match in NEEDLE_RE.finditer(prompt):
        s, e = match.span()
        vs, ve = match.span(2)
        needles.append(dict(needle=[i for i, (a, b) in enumerate(offsets) if a < e and b > s],
                            value=[i for i, (a, b) in enumerate(offsets) if a < ve and b > vs],
                            text=match.group(2), asked=match.group(2) in asked))
    if not needles:
        raise RuntimeError("needle sentence not found in prompt")
    return needles, NEEDLE_RE.search(prompt).start() / max(len(prompt), 1)


def analyse(ids, surprisal, boundary, max_span, recent, needles, decode, competitors=5, extra_ends=None) -> dict:
    """Each needle's rank under every score and its token surprisal; the top-level fields describe the first
    asked needle (else the first), with the spans that outrank it most. ``extra_ends``: more span ends."""
    starts, stops = cut(ids, boundary, max_span, extra_ends)
    scores = span_scores(surprisal, starts, stops)
    scores["uni_max"] = span_scores(unigram_code_length(ids, len(boundary)), starts, stops)["max"]
    archived = stops <= len(ids) - recent
    per = []
    for order, n in enumerate(needles):
        target = int(np.searchsorted(stops, n["value"][0], side="right"))  # the span holding the first value token
        entry = dict(order=order, value=n["text"], asked=n["asked"], target=target,
                     needle_span=[int(starts[target]), int(stops[target])], in_recent=bool(not archived[target]),
                     value_surprisal=[round(float(surprisal[i]), 2) for i in n["value"]],
                     needle_surprisal_mean=round(float(np.nanmean(surprisal[n["needle"]])), 3))
        if not entry["in_recent"]:
            entry["ranks"] = {name: needle_rank(scores[name], starts, stops, target, archived) for name in SCORES}
        per.append(entry)
    main = next((e for e in per if e["asked"]), per[0])
    record = dict(tokens=len(ids), spans=len(starts), document_surprisal_median=round(float(np.nanmedian(surprisal)), 3),
                  **{k: v for k, v in main.items() if k not in ("order", "target", "value", "asked")})
    if len(per) > 1:
        record["needles"] = [{k: v for k, v in e.items() if k != "target"} for e in per]
    if not main["in_recent"]:
        order = [i for i in np.argsort(-scores["max"]) if archived[i] and i != main["target"]][:competitors]
        record["top_competitors"] = [dict(score=round(float(scores["max"][i]), 2), position=int(starts[i]),
                                          text=decode(ids[starts[i]:stops[i]])[:120]) for i in order]
    return record


def summarize(task, records, budgets=BUDGETS) -> str:
    """One block of lines per task: how many needles each score would keep at each pool budget."""
    ranked = [r for r in records if "ranks" in r]
    lines = [f"{task}: samples={len(records)} needle in the recent window={len(records) - len(ranked)}"]
    if ranked:
        value = np.concatenate([r["value_surprisal"] for r in records])
        lines.append(f"  value-token surprisal: median {np.median(value):.2f} bits; "
                     f"document median {np.median([r['document_surprisal_median'] for r in records]):.2f} bits")
        for name in SCORES:
            needed = [r["ranks"][name]["tokens_needed"] for r in ranked]
            fits = " ".join(f"P{b}:{np.mean([t <= b for t in needed]):.2f}" for b in budgets)
            pct = np.median([r["ranks"][name]["percentile"] for r in ranked])
            lines.append(f"  {name:8s} tokens_needed median={int(np.median(needed)):6d}  percentile median={pct:.4f}  "
                         f"fraction that fits {fits}")
    many = [r["needles"] for r in records if "needles" in r]
    for k in range(max((len(n) for n in many), default=0)):  # multi-needle tasks: by order in the prompt
        at = [n[k] for n in many if len(n) > k]
        ranked = [e for e in at if "ranks" in e]
        fits = " ".join(f"{name} {np.mean([e['ranks'][name]['tokens_needed'] <= 1024 for e in ranked]):.2f}"
                        for name in ("mean", "top4", "max", "sum")) if ranked else "-"
        lines.append(f"  needle {k + 1}: n={len(at)} sentence {np.median([e['needle_surprisal_mean'] for e in at]):.2f} "
                     f"bits, value {np.median(np.concatenate([e['value_surprisal'] for e in at])):.2f} bits; "
                     f"fits P1024 {fits}")
    return "\n".join(lines)


def summary_path(out) -> Path:
    """Where a run's readable summary goes: beside its output, ``<stem>.summary.txt``."""
    out = Path(out).expanduser()
    return out.with_name(out.stem + ".summary.txt")


def write_summary(out, text: str) -> None:
    path = summary_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    print(text, flush=True)
    print(f"summary written to {path}", flush=True)


def report(jsonl) -> str:
    """The per-task summary of a finished run, from its .jsonl (no model)."""
    by_task: dict[str, list] = {}
    for line in open(jsonl):
        if line.strip():
            record = json.loads(line)
            by_task.setdefault(record["task"], []).append(record)
    return "\n".join(summarize(task, records) for task, records in by_task.items())


class SurprisalStore:
    """Token ids and surprisal of every prompt or block, saved with the cache settings a replay needs.

    ``add`` takes one sequence and its fields (scalars or strings); ``save`` writes one .npz with the
    concatenated ``ids`` / ``surprisal``, ``offsets`` (sequence i is ``offsets[i]:offsets[i + 1]``), one
    array per field, and ``meta`` (JSON: recent window, flush granularity, span cap, exact pool,
    span-boundary token ids, config). ``load`` reads it back as (meta, list of per-sequence dicts).
    """

    def __init__(self, path, **meta) -> None:
        self.path, self.meta, self.rows = path, meta, []

    def add(self, ids, surprisal, **fields) -> None:
        self.rows.append((np.asarray(ids, dtype=np.int32), np.asarray(surprisal, dtype=np.float32), fields))

    def save(self) -> None:
        if self.path is None or not self.rows:
            return
        Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        lengths = [len(ids) for ids, _, _ in self.rows]
        fields = {name: np.asarray([f[name] for _, _, f in self.rows]) for name in self.rows[0][2]}
        np.savez_compressed(self.path, ids=np.concatenate([r[0] for r in self.rows]),
                            surprisal=np.concatenate([r[1] for r in self.rows]),
                            offsets=np.concatenate(([0], np.cumsum(lengths))), meta=json.dumps(self.meta), **fields)

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=False)
        meta, offsets = json.loads(str(data["meta"])), data["offsets"]
        names = [k for k in data.files if k not in ("ids", "surprisal", "offsets", "meta")]
        rows = []
        for i in range(len(offsets) - 1):
            a, b = int(offsets[i]), int(offsets[i + 1])
            row = {name: data[name][i].item() for name in names}
            row.update(ids=data["ids"][a:b].astype(np.int64), surprisal=data["surprisal"][a:b].astype(np.float64))
            rows.append(row)
        return meta, rows


def cache_settings(lm, config: str) -> dict:
    """The LogKV settings an offline replay of the exact pool needs (``SurprisalStore`` meta)."""
    return dict(config=config, dense=bool(lm.log_kv_dense_mode), recent=int(lm.log_kv_recent_size),
                flush=int(lm.log_kv_semantic_flush_granularity), max_span=int(lm.log_kv_alpha_span_max_tokens),
                pool=int(lm.log_kv_alpha_exact_tokens or 0),
                boundary_ids=sorted(int(i) for i in lm.tokenizer.alpha_span_boundary_ids))


def niah_samples(task, hf_tokenizer, length, count):
    """lm-eval's samples for ``task`` (RULER generator; the niah_cue templates for the variants)."""
    sys.path.insert(0, str(ROOT / "eval_tasks" / "niah_cue"))
    from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack
    from niah_cue_utils import TEMPLATES
    variant = next((v for v in VARIANTS if task.endswith("_" + v)), "ruler")
    base = task[:-len(variant) - 1] if variant != "ruler" else task
    if base not in BASES:
        raise SystemExit(f"unknown task {task}: use {', '.join(BASES)} with an optional _qend/_preamble/_paraphrase")
    settings = BASES[base]
    return generate_samples(get_haystack(type_haystack=settings["type_haystack"]), max_seq_length=length,
                            template=TEMPLATES[variant], type_needle_k="words", num_samples=count,
                            TOKENIZER=hf_tokenizer, **settings)


def prompt_ids(sample, tokenizer, hf_tokenizer):
    """The eval tokenizer's ids of lm-eval's plain-text context and HF character offsets for them."""
    prefix = sample.get("gen_prefix") or ""
    prompt = sample["input"] + (" " + prefix if prefix else "")
    ids = tokenizer.encode(prompt).tolist()
    encoded = hf_tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    shift = len(ids) - len(encoded["input_ids"])
    if shift < 0 or ids[shift:] != list(encoded["input_ids"]):
        raise RuntimeError("the eval tokenizer and the HF tokenizer disagree on this prompt")
    return prompt, ids, [(0, 0)] * shift + [tuple(o) for o in encoded["offset_mapping"]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="experiment YAML, as for eval.sh")
    p.add_argument("--checkpoint", default=None, help="defaults to the YAML save_path (latest step_*)")
    p.add_argument("--hf-tokenizer", default=None,
                   help="HF tokenizer dir (default: the one eval.sh would use, else the checkpoint's own)")
    p.add_argument("--tasks", default="niah_single_2,niah_single_2_words,niah_single_2_words_qend")
    p.add_argument("--length", type=int, default=32768)
    p.add_argument("--samples", type=int, default=32, help="samples per task")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="niah_surprisal.jsonl")
    p.add_argument("--save-surprisal", default=None,
                   help=".npz of every prompt's ids and token surprisal, for unused/pool_policy_sim.py")
    p.add_argument("--info-spans", action="store_true",
                   help="also cut information spans (log_kv_gamma_surprisal_spans) before ranking")
    p.add_argument("--report", default=None, help="a finished run's .jsonl: write its summary, load no model")
    args = p.parse_args()
    if args.report:
        write_summary(args.report, report(args.report))
        return
    if not args.config:
        p.error("--config is required (or --report <run>.jsonl)")

    import train_retention as tr
    cfg = tr.load_config(args.config)
    lm = tr.build_lm(cfg, args.checkpoint, args.device)
    from transformers import AutoTokenizer
    hf_dir = args.hf_tokenizer or tr.tokenizer_dir(cfg) or tr.tokenizer_dir(
        {"tokenizer_dir": importlib.import_module("eval")._resolve_checkpoint_dir(args.checkpoint or cfg["save_path"])})
    if hf_dir is None:
        raise SystemExit("no HF tokenizer found: pass --hf-tokenizer <dir with tokenizer.json>")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    boundary = np.zeros(max(lm.tokenizer.vocab_size, lm.config.padded_vocab_size), dtype=bool)
    boundary[list(lm.tokenizer.alpha_span_boundary_ids)] = True
    max_span, recent = int(lm.log_kv_alpha_span_max_tokens), int(lm.log_kv_recent_size)
    print(f"spans capped at {max_span} tokens; recent window {recent}; "
          f"{'dense' if lm.log_kv_dense_mode else 'LogKV'} cache from {args.config}", flush=True)
    store = SurprisalStore(args.save_surprisal, **cache_settings(lm, args.config))
    Path(args.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with open(args.out, "w") as out:
        for task in args.tasks.split(","):
            records = []
            for sample in niah_samples(task, hf_tokenizer, args.length, args.samples):
                start = time.time()
                prompt, ids, offsets = prompt_ids(sample, lm.tokenizer, hf_tokenizer)
                needles, depth = locate(prompt, offsets, sample["outputs"])
                main = next((n for n in needles if n["asked"]), needles[0])
                needle, value, text = main["needle"], main["value"], main["text"]
                surprisal = token_surprisal(lm.model, ids, prepare=lm._set_eval_cache)
                lm._reset_eval_cache()
                extra = (al.information_ends(surprisal, min(recent, int(lm.log_kv_semantic_flush_granularity)))
                         if args.info_spans else None)
                record = analyse(np.asarray(ids), surprisal, boundary, max_span, recent, needles,
                                 lambda t: lm.tokenizer.decode(torch.as_tensor(t)), extra_ends=extra)
                record.update(task=task, index=sample["index"], depth=round(depth, 4), value=text)
                asked = [[n["value"][0], n["value"][-1] + 1] for n in needles if n["asked"]] or [[value[0], value[-1] + 1]]
                store.add(ids, surprisal, task=task, index=int(sample["index"]), depth=float(depth),
                          needle_start=int(needle[0]), needle_stop=int(needle[-1]) + 1,
                          value_start=int(value[0]), value_stop=int(value[-1]) + 1, asked_values=json.dumps(asked))
                out.write(json.dumps(record) + "\n")
                out.flush()
                records.append(record)
                ranks = record.get("ranks", {}).get("max", {})
                print(f"{task} sample {sample['index']}: depth={depth:.2f} value={text} "
                      f"rank(max)={ranks.get('rank', 'recent')} tokens_needed={ranks.get('tokens_needed', '-')} "
                      f"seconds={time.time() - start:.1f}",
                      flush=True)
            summaries.append(summarize(task, records))
            write_summary(args.out, "\n".join(summaries))
            store.save()  # after every task, so an interrupted run keeps what it computed


if __name__ == "__main__":
    main()
