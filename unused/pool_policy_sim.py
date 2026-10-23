"""Offline replay of exact-pool policies on saved token surprisal (CPU only, no model).

Input: the .npz files that ``--save-surprisal`` writes (unused/niah_surprisal.py for NIAH prompts,
unused/reuse_label_stats.py for natural training blocks). They hold token ids, each token's
surprisal in bits and the cache settings: recent window R, flush granularity F, span cap, exact pool
P and the span-boundary tokens.

The replay follows the cache's rhythm: tokens enter one at a time; once R + F are waiting, the oldest
F leave the recent window, every candidate span (Alpha's cut) that has fully left it competes with
the spans already in the pool, and a whole-span greedy by value fills P tokens (pool spans get the
hysteresis factor, 1.1 in the cache). A span that is not kept is gone (it is averaged into the ladder).

A policy gives each span the value

    v(T) = (I + beta * M(T)) * g(T - last(T))

    I        admission information: the span's surprisal score (mean | top4 | topq | max), or 1 for
             the recency baseline
    g        forgetting: none, power (1 + age / tau) ** -alpha, or exp(-age / tau)
    last(T)  the span's end, or with reuse on, the latest re-occurrence up to T of one of its
             informative n-grams (n-grams whose surprisal inside the span is at least reuse_bits)
    M(T)     with reuse boost, the summed surprisal of those re-occurrences up to T (0 otherwise)

Everything is causal: a value at T uses only tokens before T. Reported per policy:

    NIAH     the share of prompts whose needle value is still exact (pool or recent) when the
             prompt ends, per task, and for needles in the first third of the prompt (early)
    natural  hit rate of far reuse (unused/reuse_label_stats.py's label, n-gram n_label, first
             occurrence >= label_bits, b - a >= R): is the copy at a exact when b is read; streams
             of 1 and --concat blocks (positions continue across the joined blocks)
    reused   the share of pool tokens (by span) whose span saw a re-occurrence while kept

    python unused/pool_policy_sim.py --niah niah.npz --natural natural.npz --concat 4 --out sim.json

The table (all policies, then the ten best by mean NIAH and by the longest streams' hit rate) is also
written to ``<out stem>.summary.txt``; ``--report <out>.json`` rebuilds it from a finished run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "unused"))
from niah_surprisal import SurprisalStore, cut, write_summary  # noqa: E402
from reuse_label_stats import reuse_pairs  # noqa: E402

SCORES = ("mean", "top4", "topq", "max")


@dataclass(frozen=True)
class Policy:
    score: str = "mean"          # mean | top4 | topq | max | recency
    decay: str = "none"          # none | power | exp
    tau: float = 16384.
    alpha: float = 1.
    reuse: str = "none"          # none | refresh | boost
    beta: float = 0.5
    hysteresis: float = 1.1

    @property
    def name(self) -> str:
        decay = {"none": "none", "power": f"pow(tau={self.tau:g},a={self.alpha:g})",
                 "exp": f"exp(tau={self.tau:g})"}[self.decay]
        reuse = f"boost(b={self.beta:g})" if self.reuse == "boost" else self.reuse
        return f"{self.score}|{decay}|{reuse}" + (f"|h={self.hysteresis:g}" if self.hysteresis != 1.1 else "")


def default_policies() -> list[Policy]:
    """The grid the first replay compares: admission score x forgetting x reuse, plus recency."""
    decays = [dict(decay="none"), dict(decay="power", tau=4096., alpha=.5), dict(decay="power", tau=4096., alpha=1.),
              dict(decay="power", tau=16384., alpha=.5), dict(decay="power", tau=16384., alpha=1.),
              dict(decay="exp", tau=16384.)]
    reuses = [dict(reuse="none"), dict(reuse="refresh"), dict(reuse="boost", beta=.5)]
    out = [Policy(score=s, **d, **r) for s, d, r in itertools.product(("mean", "top4", "topq"), decays, reuses)]
    return out + [Policy(score="recency", decay="exp", tau=4096.)]


def span_base(s, starts, stops) -> dict[str, np.ndarray]:
    """Admission scores per span from token surprisal (nan tokens ignored, empty spans score 0)."""
    out = {name: np.zeros(len(starts)) for name in SCORES}
    for i, (a, b) in enumerate(zip(starts, stops)):
        part = s[a:b]
        part = np.sort(part[np.isfinite(part)])
        if len(part):
            out["mean"][i], out["max"][i] = part.mean(), part[-1]
            out["top4"][i] = part[-4:].mean()
            out["topq"][i] = part[-max(1, len(part) // 4):].mean()
    out["recency"] = np.ones(len(starts))
    return out


class Stream:
    """One token sequence prepared for replay: spans, admission scores, n-gram re-occurrence index."""

    def __init__(self, ids, s, boundary, max_span, reuse_n=2, reuse_bits=6., label_n=3, label_bits=6.,
                 recent=2048, **fields):
        self.ids, self.s = np.asarray(ids, dtype=np.int64), np.asarray(s, dtype=np.float64)
        self.fields = fields
        self.starts, self.stops = cut(self.ids, boundary, max_span)
        self.base = span_base(self.s, self.starts, self.stops)
        n = reuse_n
        self.span_keys: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * len(self.starts)
        self.occurrences: dict[int, np.ndarray] = {}
        self.info = np.zeros(len(self.ids))
        if len(self.ids) > n:
            windows = np.lib.stride_tricks.sliding_window_view(self.ids, n)
            _, group = np.unique(windows, axis=0, return_inverse=True)
            group = np.asarray(group).reshape(-1)
            ends = np.arange(n - 1, len(self.ids))  # an n-gram is known once its last token is read
            values = np.lib.stride_tricks.sliding_window_view(np.nan_to_num(self.s, nan=0.), n).mean(1)
            self.key = np.full(len(self.ids), -1, dtype=np.int64)
            self.key[ends] = group
            self.info[ends] = values
            order = np.argsort(group, kind="stable")
            split = np.flatnonzero(np.diff(group[order])) + 1
            for chunk in np.split(order, split):
                self.occurrences[int(group[chunk[0]])] = ends[chunk]
            for i, (a, b) in enumerate(zip(self.starts, self.stops)):
                inside = np.arange(max(a + n - 1, n - 1), b)
                inside = inside[self.info[inside] >= reuse_bits]
                self.span_keys[i] = np.unique(self.key[inside])
        a, b, first, _, _ = reuse_pairs(self.ids, self.s, label_n, recent)
        keep = first >= label_bits
        order = np.argsort(b[keep], kind="stable")
        self.events_a, self.events_b = a[keep][order], b[keep][order]
        self._memo: dict[tuple[int, int], tuple] = {}

    def span_of(self, positions):
        return np.searchsorted(self.stops, positions, side="right")

    def reuse(self, span: int, time: int):
        """Latest re-occurrence (< time) after the span of one of its informative n-grams, and the summed
        surprisal of all such re-occurrences; (None, 0.) when there is none."""
        if (span, time) in self._memo:  # every policy replays the same flush times
            return self._memo[(span, time)]
        latest, mass, stop = None, 0., int(self.stops[span])
        for key in self.span_keys[span]:
            occ = self.occurrences[int(key)]
            lo, hi = np.searchsorted(occ, stop), np.searchsorted(occ, time)
            if hi > lo:
                latest = int(occ[hi - 1]) if latest is None else max(latest, int(occ[hi - 1]))
                mass += float(self.info[occ[lo:hi]].sum())
        self._memo[(span, time)] = latest, mass
        return latest, mass


def concat(rows):
    """One stream from several rows of a SurprisalStore (positions continue)."""
    return dict(ids=np.concatenate([r["ids"] for r in rows]), surprisal=np.concatenate([r["surprisal"] for r in rows]))


def value(policy: Policy, stream: Stream, spans: np.ndarray, time: int) -> np.ndarray:
    base = stream.base[policy.score][spans].copy()
    last = stream.stops[spans].astype(np.float64)
    if policy.reuse != "none":
        for j, span in enumerate(spans):
            latest, mass = stream.reuse(int(span), time)
            if latest is not None:
                last[j] = max(last[j], latest + 1)
            if policy.reuse == "boost":
                base[j] += policy.beta * mass
    age = np.maximum(time - last, 0.)
    if policy.decay == "power":
        return base * (1. + age / policy.tau) ** -policy.alpha
    if policy.decay == "exp":
        return base * np.exp(-age / policy.tau)
    return base


def replay(stream: Stream, policy: Policy, pool: int, recent: int, flush: int) -> dict:
    """Run one policy over one stream; returns the final pool, the archived prefix and hit counts."""
    n = len(stream.ids)
    lengths = stream.stops - stream.starts
    kept = np.zeros(0, dtype=np.int64)
    snapshots = []  # (time, kept spans) after every flush
    archived, next_span, reused_tokens, kept_tokens = 0, 0, 0, 0
    time = recent + flush
    while time <= n:
        archived += flush
        first = next_span
        while next_span < len(stream.stops) and stream.stops[next_span] <= archived:
            next_span += 1
        candidates = np.arange(first, next_span)
        spans = np.concatenate((kept, candidates))
        v = value(policy, stream, spans, time)
        v[:len(kept)] *= policy.hysteresis
        chosen, used = [], 0
        for j in np.argsort(-v, kind="stable"):
            if used + lengths[spans[j]] <= pool:
                chosen.append(spans[j])
                used += lengths[spans[j]]
        dropped = np.setdiff1d(kept, chosen)
        for span in dropped:  # a span leaving the pool: was it re-occurred while kept?
            reused_tokens += lengths[span] * (stream.reuse(int(span), time)[0] is not None)
            kept_tokens += lengths[span]
        kept = np.sort(np.asarray(chosen, dtype=np.int64))
        snapshots.append((time, set(kept.tolist())))
        time += flush
    end = n
    for span in kept:
        reused_tokens += lengths[span] * (stream.reuse(int(span), end)[0] is not None)
        kept_tokens += lengths[span]
    # Far reuse: is the copy at a exact when b is read? Exact: in the pool, or its span has not fully
    # left the recent window yet (the cache holds such a span open).
    hits, times = 0, [t for t, _ in snapshots]
    for a, b in zip(stream.events_a, stream.events_b):
        k = int(np.searchsorted(times, b, side="right")) - 1
        span = int(stream.span_of(a))
        archived_then = 0 if k < 0 else (k + 1) * flush
        if stream.stops[span] > archived_then or span in snapshots[k][1]:
            hits += 1
    return dict(kept=set(kept.tolist()), archived=archived, hits=hits, events=len(stream.events_a),
                reused_tokens=reused_tokens, kept_tokens=kept_tokens)


def needle_kept(stream: Stream, result: dict) -> bool:
    """Every token of the needle value is exact when the prompt ends."""
    lo, hi = stream.fields["value_start"], stream.fields["value_stop"]
    spans = np.unique(stream.span_of(np.arange(lo, hi)))
    return all(stream.stops[s] > result["archived"] or s in result["kept"] for s in spans)


def evaluate(policies, niah, natural, pool, recent, flush):
    rows = []
    for policy in policies:
        row = dict(policy=policy.name, **asdict(policy))
        by_task: dict[str, list[bool]] = {}
        early = []
        reused = kept = 0
        for stream in niah:
            result = replay(stream, policy, pool, recent, flush)
            ok = needle_kept(stream, result)
            by_task.setdefault(stream.fields["task"], []).append(ok)
            if stream.fields["depth"] < 1 / 3:
                early.append(ok)
            reused, kept = reused + result["reused_tokens"], kept + result["kept_tokens"]
        row["niah"] = {task: float(np.mean(v)) for task, v in by_task.items()}
        row["niah_early"] = float(np.mean(early)) if early else float("nan")
        for label, streams in natural.items():
            hits = events = 0
            for stream in streams:
                result = replay(stream, policy, pool, recent, flush)
                hits, events = hits + result["hits"], events + result["events"]
                reused, kept = reused + result["reused_tokens"], kept + result["kept_tokens"]
            row[f"natural_{label}"] = hits / events if events else float("nan")
        row["reused"] = reused / kept if kept else float("nan")
        rows.append(row)
        print(format_row(row), flush=True)
    return rows


def summary(saved: dict) -> str:
    """All policies in grid order, then the ten best by mean NIAH and by the longest natural hit rate."""
    rows = saved["rows"]
    lines = [f"pool {saved['pool']}, recent {saved['recent']}, flush {saved['flush']}; "
             f"reuse {saved['reuse_n']}-grams >= {saved['reuse_bits']:g} bits; "
             f"far-reuse label {saved['label_n']}-grams >= {saved['label_bits']:g} bits"
             + (f"; {saved['streams']}" if saved.get("streams") else ""), "", "all policies:"]
    lines += [format_row(row) for row in rows]

    def niah_mean(row):
        return float(np.mean(list(row["niah"].values()))) if row["niah"] else float("nan")

    lines += ["", "best 10 by mean NIAH:"] + [format_row(r) for r in sorted(rows, key=niah_mean, reverse=True)[:10]]
    natural = [k for k in (rows[0] if rows else {}) if k.startswith("natural_")]
    if natural:
        key = natural[-1]
        best = sorted(rows, key=lambda r: r[key], reverse=True)[:10]
        lines += ["", f"best 10 by {key}:"] + [format_row(r) for r in best]
    return "\n".join(lines)


def format_row(row) -> str:
    niah = " ".join(f"{task.replace('niah_single_', 's')}={v:.2f}" for task, v in row["niah"].items())
    natural = " ".join(f"{k[len('natural_'):]}={row[k]:.3f}" for k in row if k.startswith("natural_"))
    return f"{row['policy']:42s} {niah} early={row['niah_early']:.2f} | hit {natural} | reused={row['reused']:.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--niah", default=None, help="unused/niah_surprisal.py --save-surprisal output")
    p.add_argument("--natural", default=None, help="unused/reuse_label_stats.py --save-surprisal output")
    p.add_argument("--concat", default="4", help="also join this many natural blocks per stream (comma list)")
    p.add_argument("--pool", type=int, default=None, help="exact pool tokens (default: the saved P)")
    p.add_argument("--recent", type=int, default=None)
    p.add_argument("--flush", type=int, default=None)
    p.add_argument("--reuse-n", type=int, default=2, help="n-gram length the policies watch for re-occurrence")
    p.add_argument("--reuse-bits", type=float, default=6., help="informative n-grams: surprisal in the span")
    p.add_argument("--label-n", type=int, default=3, help="far-reuse label n-gram (natural hit rate)")
    p.add_argument("--label-bits", type=float, default=6., help="far-reuse label first-occurrence threshold")
    p.add_argument("--policies", default=None, help="JSON list of Policy fields (default: the built-in grid)")
    p.add_argument("--out", default="pool_policy_sim.json")
    p.add_argument("--report", default=None, help="a finished run's .json: write its summary, replay nothing")
    args = p.parse_args()
    if args.report:
        with open(args.report) as f:
            write_summary(args.report, summary(json.load(f)))
        return
    if not args.niah and not args.natural:
        raise SystemExit("give --niah and/or --natural (.npz from --save-surprisal)")

    loaded = {name: SurprisalStore.load(path) for name, path in (("niah", args.niah), ("natural", args.natural)) if path}
    meta = next(iter(loaded.values()))[0]
    pool, recent, flush = args.pool or meta["pool"], args.recent or meta["recent"], args.flush or meta["flush"]
    boundary = np.zeros(max(max(meta["boundary_ids"]) + 1, 1), dtype=bool)
    boundary[meta["boundary_ids"]] = True

    def prepare(row, **fields):
        ids = row["ids"]
        table = boundary if boundary.size > ids.max() else np.pad(boundary, (0, int(ids.max()) + 1 - boundary.size))
        return Stream(ids, row["surprisal"], table, meta["max_span"], args.reuse_n, args.reuse_bits, args.label_n,
                      args.label_bits, recent, **fields)

    niah = []
    if "niah" in loaded:
        for row in loaded["niah"][1]:
            niah.append(prepare(row, **{k: v for k, v in row.items() if k not in ("ids", "surprisal")}))
    natural = {}
    if "natural" in loaded:
        rows = loaded["natural"][1]
        natural["32k"] = [prepare(row) for row in rows]
        for k in (int(x) for x in args.concat.split(",") if x):
            if k > 1 and len(rows) >= k:
                natural[f"{k}x"] = [prepare(concat(rows[i:i + k])) for i in range(0, len(rows) - k + 1, k)]
    policies = default_policies() if args.policies is None else [Policy(**d) for d in json.loads(args.policies)]
    print(f"pool {pool}, recent {recent}, flush {flush}; {len(niah)} NIAH prompts, "
          + ", ".join(f"{len(v)} natural streams ({k})" for k, v in natural.items()) + f"; {len(policies)} policies",
          flush=True)
    rows = evaluate(policies, niah, natural, pool, recent, flush)
    saved = dict(pool=pool, recent=recent, flush=flush, reuse_n=args.reuse_n, reuse_bits=args.reuse_bits,
                 label_n=args.label_n, label_bits=args.label_bits, rows=rows,
                 streams=f"{len(niah)} NIAH prompts, " + ", ".join(f"{len(v)} natural streams ({k})"
                                                                    for k, v in natural.items()))
    Path(args.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(saved, f, indent=1)
    print(f"saved {args.out}")
    write_summary(args.out, summary(saved))


if __name__ == "__main__":
    main()
