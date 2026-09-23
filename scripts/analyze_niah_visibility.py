#!/usr/bin/env python
"""Position-grouping analysis for the 32K Dense/SinkWindow/SemanticLogKV
comparison's NIAH results (task spec §7.2/§8.2).

Two independent groupings, both computed from the frozen export produced by
``scripts/export_niah_samples.py`` (``needle_answer_spans``, tagged by
``source`` -- see ``litgpt/needle_spans.py``):

1. **Needle-depth quintile**, by the needle's real token position in the
   final prompt (0-20% / 20-40% / 40-60% / 60-80% / 80-100%) -- the actual
   token coordinate, not the generator's nominal "depth" parameter, which
   the task spec explicitly says to record the difference against.
2. **SinkWindow-visibility**, by where the ANSWER content (the
   ``answer_exact`` span specifically, not the wider ``answer_sentence``
   span) sits relative to a plain ``(sink_size, window_size)`` pair at the
   position where prefill ends and generation begins: ``in_window`` /
   ``in_sink`` / ``outside`` / ``boundary``. Takes ``sink_size``/
   ``window_size`` as plain ints, not a ``SinkWindowKVCache`` import -- see
   the implementation plan's branch-coupling note (this script lives on the
   base branch and must not depend on the SinkWindow branch existing).

Also supports a best-effort ``cross-check`` between the frozen export and a
branch's real per-sample eval output (see ``eval.py``'s ``log_samples`` /
the ``*_niah_samples.jsonl`` it now writes).

KNOWN GAP (read before trusting ``analyze``'s output): ``eval.py``'s real
evaluation run still drives RULER's *own* internal document generator via
lm_eval's normal task mechanism -- it does not yet replay prompts from the
frozen export file directly. Wiring that up means overriding how the RULER
task sources ``eval_docs``, which needs a real ``lm_eval`` installation to
develop and verify against (not available in this environment) and was
deliberately left as a flagged gap rather than shipped unverified. Run
``cross-check`` after every real branch eval, before trusting its
``analyze`` output: it fails loudly if the frozen export and the real run's
targets disagree at the same ``(task_name, doc_id)``, which is exactly the
failure mode an unchecked assumption of determinism would otherwise hide
silently.

Usage:
    # Run this first, on every branch's real eval output:
    python scripts/analyze_niah_visibility.py cross-check \\
        --export niah_samples_32k.jsonl \\
        --run eval_results_..._niah_samples.jsonl

    # Depth-quintile + visibility grouping for two branches, with a paired
    # bootstrap CI at each grouping cell where both have scores:
    python scripts/analyze_niah_visibility.py analyze \\
        --export niah_samples_32k.jsonl \\
        --run_a sinkwindow_eval_..._niah_samples.jsonl --label_a SinkWindow \\
        --run_b semantic_eval_..._niah_samples.jsonl --label_b SemanticLogKV \\
        --sink_size 4 --window_size 2048 \\
        --score_key exact_match \\
        --output visibility_report.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

DEPTH_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
VISIBILITY_LABELS = ["in_window", "in_sink", "outside", "boundary"]


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _export_index(export_rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Key by (task_name, within-task index), matching both
    export_niah_samples.py's sample_id="{task_name}_{i:04d}" and lm_eval's
    log_samples doc_id (both are the same eval_docs enumeration index for
    the same task -- see the KNOWN GAP note on why this equivalence still
    needs cross-check, not blind trust).
    """
    out = {}
    for row in export_rows:
        idx = int(row["sample_id"].rsplit("_", 1)[-1])
        out[(row["task_name"], idx)] = row
    return out


def _answer_exact_span(export_row: dict) -> tuple[int, int] | None:
    for span in export_row.get("needle_answer_spans", []):
        if span.get("source") == "answer_exact":
            return int(span["token_start"]), int(span["token_end"])
    return None


def _needle_token_start(export_row: dict) -> int | None:
    """Earliest token_start across all spans found for this sample -- the
    needle's real position. Any span source qualifies here (this grouping is
    about where the needle *sentence* sits, diagnostic-oriented; visibility
    bucketing below is the one that must use answer_exact specifically).
    """
    spans = export_row.get("needle_answer_spans", [])
    if not spans:
        return None
    return min(int(s["token_start"]) for s in spans)


def depth_quintile(needle_token_start: int, prompt_token_count: int) -> str:
    if prompt_token_count <= 0:
        raise ValueError(f"prompt_token_count must be > 0, got {prompt_token_count}")
    frac = max(0.0, min(1.0, needle_token_start / prompt_token_count))
    bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for lo, hi, label in zip(bounds, bounds[1:], DEPTH_LABELS):
        if frac < hi:
            return label
    return DEPTH_LABELS[-1]


def visibility_category(
    answer_start: int,
    answer_end: int,
    prompt_token_count: int,
    sink_size: int,
    window_size: int,
) -> str:
    """One of VISIBILITY_LABELS, from the answer span's token bounds at the
    position where prefill ends (task spec: "prefill 结束、开始回答时的
    SinkWindow 可见位置"), i.e. prompt_token_count. W is defined to include
    the current position, so the visible window is
    [max(0, prompt_token_count - window_size), prompt_token_count).
    """
    if answer_end <= answer_start:
        raise ValueError(f"answer span must be non-empty, got [{answer_start}, {answer_end})")
    window_start = max(0, prompt_token_count - window_size)
    in_sink = answer_start >= 0 and answer_end <= sink_size
    in_window = answer_start >= window_start and answer_end <= prompt_token_count
    if in_sink or in_window:
        # Overlap between the two visible regions (small prompt / large W)
        # counts once, per the task spec's "两部分重叠时只保留一次" -- treat
        # as in_window when both hold, it's the region closer to the query.
        return "in_window" if in_window else "in_sink"
    fully_outside = answer_end <= window_start and answer_start >= sink_size
    return "outside" if fully_outside else "boundary"


def cross_check(export_rows: list[dict], run_rows: list[dict], *, id_key: str = "doc_id") -> dict[str, Any]:
    """See module docstring's KNOWN GAP. Loud, not silent: reports every
    (task_name, doc_id) where the real run's target disagrees with the
    frozen export's recorded answer(s) for what should be the same document.
    """
    export_by_id = _export_index(export_rows)
    mismatches = []
    matched = 0
    unmatched_ids = 0
    for rec in run_rows:
        task_name = rec.get("task_name")
        doc_id = rec.get(id_key)
        if task_name is None or doc_id is None:
            unmatched_ids += 1
            continue
        export_row = export_by_id.get((task_name, int(doc_id)))
        if export_row is None:
            unmatched_ids += 1
            continue
        run_target = str(rec.get("target"))
        export_answers = [str(a) for a in (export_row.get("answer") or [])]
        ok = bool(export_answers) and any(a == run_target or a in run_target for a in export_answers)
        if ok:
            matched += 1
        else:
            mismatches.append(
                {"task_name": task_name, "doc_id": doc_id, "export_answer": export_answers, "run_target": run_target}
            )
    return {
        "matched": matched,
        "mismatched": len(mismatches),
        "unmatched_ids": unmatched_ids,
        "mismatches_sample": mismatches[:20],
        "verdict": "CONSISTENT" if not mismatches else "DIVERGED -- do not trust `analyze` output until resolved",
    }


def bootstrap_ci(diffs: list[float], *, n_resamples: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Sample-resampling bootstrap CI for a paired-difference mean (task spec
    §8.2: "对配对差值进行按样本重采样的 bootstrap,给出 95% 置信区间"). Reflects
    test-sample uncertainty only -- not training-seed variance, per the same
    section's own caveat.
    """
    if not diffs:
        raise ValueError("diffs must be non-empty")
    n = len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        resample_sum = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(resample_sum / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return {
        "mean_diff": sum(diffs) / n,
        "ci_low": means[lo_idx],
        "ci_high": means[hi_idx],
        "n_samples": n,
        "n_resamples": n_resamples,
    }


def _score(rec: dict, score_key: str) -> float | None:
    if score_key in rec:
        v = rec[score_key]
        return float(v) if isinstance(v, (int, float, bool)) else None
    for maybe in ("metrics", "exact_match", "acc"):
        v = rec.get(maybe)
        if isinstance(v, dict) and score_key in v:
            inner = v[score_key]
            return float(inner) if isinstance(inner, (int, float, bool)) else None
    return None


def _join_run(export_rows: list[dict], run_rows: list[dict], score_key: str, sink_size: int, window_size: int) -> list[dict]:
    export_by_id = _export_index(export_rows)
    joined = []
    missing_score = 0
    missing_export = 0
    for rec in run_rows:
        task_name = rec.get("task_name")
        doc_id = rec.get("doc_id")
        if task_name is None or doc_id is None:
            continue
        export_row = export_by_id.get((task_name, int(doc_id)))
        if export_row is None:
            missing_export += 1
            continue
        score = _score(rec, score_key)
        if score is None:
            missing_score += 1
            continue
        needle_pos = _needle_token_start(export_row)
        answer_span = _answer_exact_span(export_row)
        prompt_len = int(export_row["prompt_token_count"])
        row = {
            "sample_id": export_row["sample_id"],
            "task_name": task_name,
            "score": score,
            "prompt_token_count": prompt_len,
        }
        if needle_pos is not None:
            row["depth_quintile"] = depth_quintile(needle_pos, prompt_len)
        if answer_span is not None:
            row["visibility"] = visibility_category(answer_span[0], answer_span[1], prompt_len, sink_size, window_size)
        joined.append(row)
    if missing_score or missing_export:
        print(
            f"[analyze] WARNING: {missing_export} records had no matching export row, "
            f"{missing_score} had no readable score for --score_key={score_key!r} "
            "(pass the real metric key name from a sample record if this looks wrong)"
        )
    return joined


def _group_report(rows: list[dict], group_key: str, labels: list[str]) -> dict[str, dict]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if group_key in r:
            by_group[r[group_key]].append(r["score"])
    report = {}
    for label in labels:
        scores = by_group.get(label, [])
        report[label] = {
            "n": len(scores),
            "mean": (sum(scores) / len(scores)) if scores else None,
        }
    return report


def _paired_report(rows_a: list[dict], rows_b: list[dict], group_key: str, labels: list[str]) -> dict[str, Any]:
    by_id_a = {r["sample_id"]: r for r in rows_a}
    by_id_b = {r["sample_id"]: r for r in rows_b}
    common_ids = set(by_id_a) & set(by_id_b)
    report = {}
    for label in labels:
        diffs = [
            by_id_a[sid]["score"] - by_id_b[sid]["score"]
            for sid in common_ids
            if by_id_a[sid].get(group_key) == label and by_id_b[sid].get(group_key) == label
        ]
        report[label] = bootstrap_ci(diffs) if diffs else {"n_samples": 0}
    return report


def cmd_cross_check(args: argparse.Namespace) -> None:
    export_rows = _read_jsonl(args.export)
    run_rows = _read_jsonl(args.run)
    result = cross_check(export_rows, run_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["mismatched"]:
        raise SystemExit(
            f"cross-check found {result['mismatched']} divergent samples -- "
            "the real eval run did not evaluate the frozen export's prompts. "
            "See this script's module docstring, KNOWN GAP."
        )


def cmd_analyze(args: argparse.Namespace) -> None:
    export_rows = _read_jsonl(args.export)
    rows_a = _join_run(export_rows, _read_jsonl(args.run_a), args.score_key, args.sink_size, args.window_size)
    report: dict[str, Any] = {
        "label_a": args.label_a,
        "depth_quintile": {args.label_a: _group_report(rows_a, "depth_quintile", DEPTH_LABELS)},
        "visibility": {args.label_a: _group_report(rows_a, "visibility", VISIBILITY_LABELS)},
    }
    if args.run_b is not None:
        rows_b = _join_run(export_rows, _read_jsonl(args.run_b), args.score_key, args.sink_size, args.window_size)
        report["label_b"] = args.label_b
        report["depth_quintile"][args.label_b] = _group_report(rows_b, "depth_quintile", DEPTH_LABELS)
        report["visibility"][args.label_b] = _group_report(rows_b, "visibility", VISIBILITY_LABELS)
        report["paired_diff_depth_quintile_bootstrap_ci"] = _paired_report(rows_a, rows_b, "depth_quintile", DEPTH_LABELS)
        report["paired_diff_visibility_bootstrap_ci"] = _paired_report(rows_a, rows_b, "visibility", VISIBILITY_LABELS)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[analyze] wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("cross-check", help="Verify a real run's targets match the frozen export (KNOWN GAP mitigation)")
    p_check.add_argument("--export", required=True, type=Path)
    p_check.add_argument("--run", required=True, type=Path, help="eval.py's *_niah_samples.jsonl for one branch")
    p_check.set_defaults(func=cmd_cross_check)

    p_analyze = sub.add_parser("analyze", help="Depth-quintile + visibility grouping, optionally paired between two branches")
    p_analyze.add_argument("--export", required=True, type=Path)
    p_analyze.add_argument("--run_a", required=True, type=Path)
    p_analyze.add_argument("--label_a", default="A")
    p_analyze.add_argument("--run_b", type=Path, default=None)
    p_analyze.add_argument("--label_b", default="B")
    p_analyze.add_argument("--sink_size", type=int, required=True)
    p_analyze.add_argument("--window_size", type=int, required=True)
    p_analyze.add_argument(
        "--score_key",
        default="exact_match",
        help="Metric key inside each lm_eval sample record (best-guess default -- "
        "inspect one real record and pass the actual key if this doesn't match)",
    )
    p_analyze.add_argument("--output", type=Path, default=None)
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
