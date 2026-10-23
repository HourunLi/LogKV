"""Natural long-range reuse as a retention label: statistics on the training corpus (GPU, no training).

The label asks, for each token of a natural document, what a compressed cache loses for good:
content the model could not predict where it first appeared (high surprisal) and that the text
needs again once it has left the recent window. Operationally, for every n-gram of token ids:

    reuse pair    two consecutive occurrences a < b of the same n-gram with b - a >= gap (default:
                  the YAML's recent window), so when b is read the copy at a is outside the window;
    positive      the n tokens at a, when the n-gram's mean surprisal at its first occurrence is at
                  least the threshold (in bits): common phrases are cheap to predict and are left out.

No sentence form, template or task enters the label. For every (n, threshold) pair the script
reports, over blocks of the YAML's training data (read as demo.py reads them):

    tokens%        positive tokens per block (share of all tokens)
    spans/block    candidate spans (Alpha's cut) holding a positive token, and their tokens
    fits P         blocks whose positive spans fit in the exact pool (log_kv_alpha_exact_tokens)
    s_first/s_next mean n-gram surprisal at its first occurrence and at the reuse b: a drop means
                   the model copied it, no drop means the copy was lost (compare a dense YAML)
    kept by K/V    with a LogKV YAML, the share of positive tokens the cache's own selection kept
                   exact (pool or recent) after the prefill, per layer mean, against the share of
                   all archived tokens it kept
    kinds          what the positive n-grams are: digits, capitalised words, code-like symbols,
                   lowercase words, other

and prints examples for one (n, threshold) pair.

    bash diag.sh unused/reuse_label_stats.py --config exp/qwen1.7b-32k/arc_gamma_k12_b128_p1024_2k.yaml \\
        --blocks 16 --out reuse_labels.json

``--save-surprisal <file>.npz`` also keeps every block's ids and token surprisal (with the cache
settings) for the CPU replay of exact-pool policies, unused/pool_policy_sim.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "unused"))
from niah_surprisal import SurprisalStore, cache_settings, cut, token_surprisal  # noqa: E402

CODE_CHARS = set("_(){}[]=<>/\\;:$#@|*&^%~`")
CAPITALISED = re.compile(r"(?:^|\s)[A-Z][a-z]")


def kind(text: str) -> str:
    """Coarse kind of an n-gram's text."""
    if any(c.isdigit() for c in text):
        return "digits"
    if CAPITALISED.search(text):
        return "capitalised"
    if any(c in CODE_CHARS for c in text):
        return "code"
    if text.strip() and all(c.isalpha() or c in " -'" for c in text.strip()) and text.strip() == text.strip().lower():
        return "lowercase"
    return "other"


def reuse_pairs(ids, surprisal, n: int, gap: int):
    """Consecutive occurrences (a, b) of the same n-gram with b - a >= gap, and the n-gram's mean
    surprisal at its first occurrence, at a and at b (arrays aligned with a)."""
    ids = np.asarray(ids, dtype=np.int64)
    empty = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0), np.empty(0), np.empty(0)
    if len(ids) < n + 1:
        return empty
    windows = np.lib.stride_tricks.sliding_window_view(ids, n)
    _, group = np.unique(windows, axis=0, return_inverse=True)
    group = np.asarray(group).reshape(-1)
    pos = np.arange(len(group))
    order = np.lexsort((pos, group))
    g, p = group[order], pos[order]
    same = g[1:] == g[:-1]
    a, b, ga = p[:-1][same], p[1:][same], g[:-1][same]
    far = b - a >= gap
    if not far.any():
        return empty
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    first = np.empty(int(group.max()) + 1, dtype=np.int64)
    first[g[starts]] = p[starts]
    values = np.lib.stride_tricks.sliding_window_view(np.nan_to_num(np.asarray(surprisal, float), nan=0.0), n)
    mean = values.mean(1)
    a, b, ga = a[far], b[far], ga[far]
    return a, b, mean[first[ga]], mean[a], mean[b]


def block_stats(ids, surprisal, n, threshold, gap, boundary, max_span, pool, decode, kept=None, recent=None):
    """Statistics of one block for one (n, threshold); ``kept`` is the per-layer exact mask, if any, and
    ``recent`` the window whose tokens count as not yet archived (default: ``gap``)."""
    a, b, s_first, s_prev, s_next = reuse_pairs(ids, surprisal, n, gap)
    keep = s_first >= threshold
    a, b, s_first, s_next = a[keep], b[keep], s_first[keep], s_next[keep]
    positive = np.zeros(len(ids), dtype=bool)
    for start in a:
        positive[start:start + n] = True
    starts, stops = cut(ids, boundary, max_span)
    span_of = np.searchsorted(stops, np.flatnonzero(positive), side="right")
    spans = np.unique(span_of)
    span_tokens = int((stops[spans] - starts[spans]).sum()) if len(spans) else 0
    kinds = [kind(decode(ids[x:x + n])) for x in a]
    out = dict(pairs=len(a), tokens=int(positive.sum()), share=float(positive.mean()), spans=len(spans),
               span_tokens=span_tokens, fits=span_tokens <= pool,
               s_first=float(s_first.mean()) if len(a) else float("nan"),
               s_next=float(s_next.mean()) if len(a) else float("nan"),
               kinds={k: kinds.count(k) for k in set(kinds)})
    if kept is not None and positive.any():
        archived = np.zeros(len(ids), dtype=bool)
        archived[:max(len(ids) - (gap if recent is None else recent), 0)] = True
        out["kept_positive"] = float(np.mean([layer[positive].mean() for layer in kept]))
        out["kept_archived"] = float(np.mean([layer[archived].mean() for layer in kept]))
    return out, list(zip(a.tolist(), b.tolist(), s_first.tolist(), s_next.tolist()))


def exact_masks(model, length):
    """Per layer, the positions the LogKV cache keeps exact (pool or recent) after the prefill; None if dense."""
    masks = []
    for block in model.transformer.h:
        cache = block.attn.kv_cache
        if not hasattr(cache, "_alpha_positions"):
            return None
        mask = np.zeros(length, dtype=bool)
        for kept in (cache._alpha_positions[0], cache._recent_pos_host[0][:cache.recent_count]):
            kept = np.asarray(kept, dtype=np.int64)
            mask[kept[kept < length]] = True
        masks.append(mask)
    return masks


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def aggregate(rows):
    """Means over blocks; kinds are summed and given as shares."""
    if not rows:
        return {}
    kinds: dict[str, int] = {}
    for r in rows:
        for k, c in r["kinds"].items():
            kinds[k] = kinds.get(k, 0) + c
    total = max(sum(kinds.values()), 1)
    out = dict(blocks=len(rows), pairs_per_block=float(np.mean([r["pairs"] for r in rows])),
               token_share=float(np.mean([r["share"] for r in rows])),
               spans_per_block=float(np.mean([r["spans"] for r in rows])),
               span_tokens_median=float(np.median([r["span_tokens"] for r in rows])),
               fits_pool=float(np.mean([r["fits"] for r in rows])),
               s_first=_finite_mean([r["s_first"] for r in rows]), s_next=_finite_mean([r["s_next"] for r in rows]),
               kinds={k: round(c / total, 3) for k, c in sorted(kinds.items(), key=lambda kv: -kv[1])})
    if any("kept_positive" in r for r in rows):
        out["kept_positive"] = float(np.mean([r["kept_positive"] for r in rows if "kept_positive" in r]))
        out["kept_archived"] = float(np.mean([r["kept_archived"] for r in rows if "kept_archived" in r]))
    return out


def report(grid, pool):
    print(f"{'n':>2} {'bits':>5} {'tokens%':>8} {'spans/blk':>9} {'span tok':>8} {'fits P' + str(pool):>9} "
          f"{'s_first':>7} {'s_next':>6} {'kept K/V':>8} {'kept all':>8}  kinds")
    for (n, threshold), row in sorted(grid.items()):
        if not row:
            continue
        kept = (f"{row['kept_positive']:8.3f} {row['kept_archived']:8.3f}" if "kept_positive" in row
                else f"{'-':>8} {'-':>8}")
        kinds = ", ".join(f"{k} {v:.2f}" for k, v in row["kinds"].items())
        print(f"{n:>2} {threshold:>5g} {100 * row['token_share']:8.3f} {row['spans_per_block']:9.1f} "
              f"{row['span_tokens_median']:8.0f} {row['fits_pool']:9.2f} {row['s_first']:7.2f} {row['s_next']:6.2f} "
              f"{kept}  {kinds}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment YAML (checkpoint, cache, training data)")
    p.add_argument("--checkpoint", default=None, help="defaults to the YAML save_path (latest step_*)")
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--length", type=int, default=32768)
    p.add_argument("--seed", type=int, default=4321, help="data order (the heads trained with 1234)")
    p.add_argument("--ngrams", default="2,3,4")
    p.add_argument("--thresholds", default="4,6,8,10", help="first-occurrence mean surprisal, bits")
    p.add_argument("--gap", type=int, default=None, help="default: the YAML's recent window")
    p.add_argument("--example", default="3,6", help="n,threshold whose examples are printed")
    p.add_argument("--examples", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="reuse_labels.json")
    p.add_argument("--save-surprisal", default=None,
                   help=".npz of every block's ids and token surprisal, for unused/pool_policy_sim.py")
    args = p.parse_args()

    import torch
    import train_retention as tr
    cfg = tr.load_config(args.config)
    lm = tr.build_lm(cfg, args.checkpoint, args.device)
    boundary = np.zeros(max(lm.tokenizer.vocab_size, lm.config.padded_vocab_size), dtype=bool)
    boundary[list(lm.tokenizer.alpha_span_boundary_ids)] = True
    max_span, recent = int(lm.log_kv_alpha_span_max_tokens), int(lm.log_kv_recent_size)
    gap = args.gap or recent
    pool = int(lm.log_kv_alpha_exact_tokens or 0)
    ngrams = [int(x) for x in args.ngrams.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    example_n, example_t = int(args.example.split(",")[0]), float(args.example.split(",")[1])

    def decode(t):
        return lm.tokenizer.decode(torch.as_tensor(np.asarray(t, dtype=np.int64)))

    print(f"gap {gap} tokens; spans capped at {max_span}; exact pool {pool}; "
          f"{'dense' if lm.log_kv_dense_mode else 'LogKV'} cache from {args.config}", flush=True)
    store = SurprisalStore(args.save_surprisal, **cache_settings(lm, args.config), seed=args.seed)
    rows = {(n, t): [] for n in ngrams for t in thresholds}
    examples = []
    data = tr.blocks(cfg, args.seed)
    for index in range(args.blocks):
        start = time.time()
        ids = np.asarray(next(data)[:args.length], dtype=np.int64)
        surprisal = token_surprisal(lm.model, ids, prepare=lm._set_eval_cache)
        kept = exact_masks(lm.model, len(ids)) if not lm.log_kv_dense_mode else None
        store.add(ids, surprisal, block=index)
        lm._reset_eval_cache()
        for n in ngrams:
            for threshold in thresholds:
                row, pairs = block_stats(ids, surprisal, n, threshold, gap, boundary, max_span, pool, decode,
                                         kept, recent)
                rows[(n, threshold)].append(row)
                if (n, threshold) == (example_n, example_t):
                    for a, b, s_first, s_next in pairs[:max(0, args.examples - len(examples))]:
                        examples.append(dict(block=index, position=a, gap=b - a, s_first=round(s_first, 2),
                                             s_next=round(s_next, 2), text=decode(ids[a:a + n]),
                                             context=decode(ids[max(0, a - 12):a + n + 4])))
        print(f"block {index}: median token surprisal {np.nanmedian(surprisal):.2f} bits, "
              f"seconds={time.time() - start:.1f}", flush=True)
    store.save()
    grid = {key: aggregate(value) for key, value in rows.items()}
    report(grid, pool)
    print(f"\nexamples (n={example_n}, >= {example_t:g} bits):")
    for e in examples:
        print(f"  gap={e['gap']:6d} s_first={e['s_first']:5.2f} s_next={e['s_next']:5.2f} "
              f"{e['text']!r:30s} ... {e['context']!r}")
    Path(args.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=args.config, gap=gap, pool=pool, max_span=max_span,
                       grid=[dict(n=n, threshold=t, **row) for (n, t), row in sorted(grid.items())],
                       examples=examples), f, indent=1, ensure_ascii=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
