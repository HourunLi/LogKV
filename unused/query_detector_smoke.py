"""Run the optional detector on local text, without loading the LM or accessing the Hub."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litgpt.query_detector import DEFAULT_MODEL_DIR, QueryDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--no-rules", action="store_true")
    args = parser.parse_args()
    detector = QueryDetector(args.model_dir, device=args.device, threshold=args.threshold,
                             use_rules=not args.no_rules)
    texts = [args.text_file.read_text()] if args.text_file else [
        "Please list the project owners.",
        "The river flows through the valley.",
        'The reporter asked, "What is the budget?" The manager did not answer.',
    ]
    for text in texts:
        spans = detector.detect(text)
        print(json.dumps({"spans": [dict(start=s.start, end=s.end, score=s.score, text=text[s.start:s.end])
                                    for s in spans]}, ensure_ascii=False))
    print(json.dumps({"detector_stats": detector.stats()}))


if __name__ == "__main__":
    main()
