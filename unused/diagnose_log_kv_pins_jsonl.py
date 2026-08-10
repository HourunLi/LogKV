"""Run LogKV pin-vs-needle diagnostics on prompt JSON/JSONL files.

This is a small offline companion to ``eval.py --log_kv_pin_diag_output``. Use
it when you already have concrete NIAH prompts dumped somewhere and want to
avoid going through lm-eval task construction again.

Expected input records:

    {"prompt": "...", "outputs": ["12345"]}
    {"prompt": "...", "needle": "The best thing ..."}

Any extra fields are kept as the sample ``doc`` for needle extraction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import LogKVLM, SafeJSONEncoder  # noqa: E402
from litgpt.log_kv_pin_diag import PinDiagRecorder  # noqa: E402


class _GenerateUntilRequest:
    def __init__(self, prompt: str, doc: dict[str, Any], max_new_tokens: int) -> None:
        self.args = (
            prompt,
            {
                "max_gen_toks": max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 0.0,
            },
        )
        self.doc = doc
        self.task_name = doc.get("task_name") or doc.get("task") or "jsonl_pin_diag"
        self.doc_id = doc.get("doc_id") or doc.get("idx") or doc.get("id")


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return data["samples"]
    raise ValueError(f"{path} must be a JSON list, a JSON object with samples, or JSONL.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="JSON/JSONL file with prompt records.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--log_kv_B", type=int, default=512)
    parser.add_argument("--log_kv_recent_size", type=int, default=1024)
    parser.add_argument("--log_kv_prefill_block", type=int, default=1024)
    parser.add_argument("--log_kv_pin_size", type=int, default=256)
    parser.add_argument("--log_kv_pin_obs_window", type=int, default=64)
    parser.add_argument("--log_kv_pin_min_distance", type=int, default=0)
    parser.add_argument("--log_kv_second_order_scale", type=float, default=0.2)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--include_indices", action="store_true")
    args = parser.parse_args()

    records = _load_records(Path(args.samples))
    if args.limit is not None:
        records = records[: args.limit]

    recorder = PinDiagRecorder(
        radius=args.radius,
        max_samples=args.limit,
        include_indices=args.include_indices,
    )
    lm = LogKVLM(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        log_kv_B=args.log_kv_B,
        log_kv_recent_size=args.log_kv_recent_size,
        log_kv_prefill_block=args.log_kv_prefill_block,
        log_kv_pin_size=args.log_kv_pin_size,
        log_kv_pin_obs_window=args.log_kv_pin_obs_window,
        log_kv_pin_min_distance=args.log_kv_pin_min_distance,
        log_kv_second_order_scale=args.log_kv_second_order_scale,
        tokenizer_dir=args.tokenizer_dir,
        pin_diag_recorder=recorder,
    )

    requests = []
    for i, record in enumerate(records):
        prompt = record.get("prompt") or record.get("input") or record.get("context")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"record {i} has no non-empty prompt/input/context string")
        doc = dict(record)
        doc.setdefault("idx", i)
        requests.append(_GenerateUntilRequest(prompt, doc, args.max_new_tokens))

    lm.generate_until(requests)
    payload = {
        "samples_file": str(Path(args.samples).expanduser()),
        "checkpoint_dir": args.checkpoint_dir,
        "config": {
            "log_kv_B": args.log_kv_B,
            "log_kv_recent_size": args.log_kv_recent_size,
            "log_kv_prefill_block": args.log_kv_prefill_block,
            "log_kv_pin_size": args.log_kv_pin_size,
            "log_kv_pin_obs_window": args.log_kv_pin_obs_window,
            "log_kv_pin_min_distance": args.log_kv_pin_min_distance,
            "log_kv_second_order_scale": args.log_kv_second_order_scale,
            "radius": args.radius,
            "include_indices": args.include_indices,
        },
        "pin_diag": recorder.summary(),
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, cls=SafeJSONEncoder), encoding="utf-8")
    print(f"pin diag saved to {output}")
    print(f"verdict: {payload['pin_diag']['verdict']}")


if __name__ == "__main__":
    main()
