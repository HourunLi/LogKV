#!/usr/bin/env python
"""Assemble one branch's experiment manifest (task spec §11 item 7):
"配置、代码版本、数据指纹、checkpoint 路径和逐样本结果" -- config, code
version, data fingerprint, checkpoint path, and per-sample results.

Pulls together artifacts other tools in this experiment already produce
rather than recomputing anything:
  - code version: the git commit this checkout is at (and dirty-tree flag).
  - config: demo.py's resolved_config.yaml (see demo.py's
    _dump_resolved_config -- the actual effective config, not the launch YAML).
  - data fingerprint: sha256 of the frozen NIAH export JSONL
    (scripts/export_niah_samples.py's output) -- proves which exact sample
    set a run's numbers came from.
  - checkpoint path: as given.
  - per-sample results: eval.py's *_niah_samples.jsonl (from --log_samples),
    referenced by path + row count + sha256, not duplicated inline.

Usage:
    python scripts/write_experiment_manifest.py \\
        --label SinkWindow \\
        --checkpoint_dir /path/to/ckpt-logKV/sinkwindow_stage1 \\
        --niah_export niah_samples_32k.jsonl \\
        --niah_samples eval_results_..._niah_samples.jsonl \\
        --output manifests/sinkwindow_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _git_info(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    commit = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="e.g. Dense / SinkWindow / SemanticLogKV")
    parser.add_argument("--checkpoint_dir", required=True, type=Path)
    parser.add_argument(
        "--resolved_config",
        type=Path,
        default=None,
        help="Defaults to <checkpoint_dir>/resolved_config.yaml (demo.py's dump)",
    )
    parser.add_argument("--niah_export", required=True, type=Path, help="scripts/export_niah_samples.py's frozen JSONL")
    parser.add_argument("--niah_samples", required=True, type=Path, help="eval.py's *_niah_samples.jsonl for this branch")
    parser.add_argument("--eval_results_json", type=Path, default=None, help="eval.py's main aggregate results JSON, if available")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    resolved_config_path = args.resolved_config or (args.checkpoint_dir / "resolved_config.yaml")
    resolved_config = None
    if resolved_config_path.is_file():
        with open(resolved_config_path, encoding="utf-8") as f:
            resolved_config = yaml.safe_load(f)
    else:
        print(f"[manifest] WARNING: {resolved_config_path} not found -- config section will be null")

    checkpoint_meta_path = args.checkpoint_dir / "checkpoint_meta.yaml"
    checkpoint_meta = None
    if checkpoint_meta_path.is_file():
        with open(checkpoint_meta_path, encoding="utf-8") as f:
            checkpoint_meta = yaml.safe_load(f)

    manifest = {
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(args.repo_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_meta": checkpoint_meta,
        "resolved_config_path": str(resolved_config_path),
        "resolved_config": resolved_config,
        "niah_export": {
            "path": str(args.niah_export),
            "sha256": _sha256_file(args.niah_export),
            "n_samples": _count_lines(args.niah_export),
        },
        "niah_per_sample_results": {
            "path": str(args.niah_samples),
            "sha256": _sha256_file(args.niah_samples),
            "n_records": _count_lines(args.niah_samples),
        },
    }
    if args.eval_results_json is not None and args.eval_results_json.is_file():
        manifest["eval_results_json"] = {
            "path": str(args.eval_results_json),
            "sha256": _sha256_file(args.eval_results_json),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[manifest] wrote {args.output}")
    if manifest["git"]["dirty"]:
        print(
            f"[manifest] WARNING: repo has {len(manifest['git']['dirty_files'])} uncommitted change(s) at manifest "
            "time -- commit before trusting this manifest's git.commit as the code version that produced these results."
        )


if __name__ == "__main__":
    main()
