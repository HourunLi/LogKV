"""CPU-only regressions: python -m unittest discover -s tests -p test_eval_results_table.py"""
import ast
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_results import markdown_rows, result_rows

spec = importlib.util.spec_from_file_location("tables", ROOT / "unused/parse_lmeval_table.py")
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


class ResultTableTests(unittest.TestCase):
    def test_raw_wrapped_groups_filters_and_stderr(self):
        raw = {"results": {
            "niah_single_1": {"alias": "NIAH", "65536,none": .8, "65536_stderr,none": .02,
                              "131072,none": .6, "65536,strict": .7},
            "long_ppl_128k": {"ppl,none": 12.4, "bits_per_byte,none": 1.7},
            "longbench_qasper": {"qa_f1_score,none": .3, "bad,none": "N/A"},
        }, "groups": {"longbench2": {"acc,none": .25, "acc_stderr,none": "N/A"}}}
        rows = result_rows(raw)
        self.assertEqual(rows, result_rows({"checkpoint_dir": "model", "results": raw}))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0]["context_length"], 65536)
        self.assertEqual(rows[0]["stderr"], .02)
        self.assertEqual(rows[-1]["result_type"], "group")
        self.assertIsNone(rows[-1]["context_length"])
        self.assertEqual(rows[3]["context_length"], 131072)
        self.assertEqual(rows[2]["filter"], "strict")

    def test_console_logs_multiple_headers_and_continuation_rows(self):
        header = "|Tasks|Version|Filter|n-shot|Metric| |Value| |Stderr|\n"
        text = "logging before\n" + header + "|---|---|---|---|---|---|---|---|---|\n"
        text += "|niah_single_2|1|none|0|65536|↑|0.9|±|N/A|\n"
        text += "| | |none|0|131072|↑|0.7|±|N/A|\nprogress log\n" + header
        text += "| - babilong_qa1_128k|1|none|0|acc|↑|0.42|±|0.1|\n"
        row = tables.parse_single_model("m", text)
        self.assertEqual(row["niah_single_2@64k"], .9)
        self.assertEqual(row["niah_single_2@128k"], .7)
        self.assertEqual(row["babilong_qa1_128k"], .42)
        self.assertEqual(len(list(markdown_rows(text))), 3)

    def test_new_lengths_variants_and_longbench_metric_fallback(self):
        raw = {"results": {
            "niah_single_2_words": {"32768,none": .9, "65536,none": .7, "524288,none": .1},
            "longbench_qasper": {"qa_f1_score,none": .3},
            "longbench_repobench-p_e": {"code_sim_score,none": .2, "score,none": .25},
        }}
        row = tables.wide_row("m", result_rows(raw))
        self.assertEqual(row["niah_single_2_words"], .9)
        self.assertEqual(row["niah_single_2_words@64k"], .7)
        self.assertEqual(row["niah_single_2_words@512k"], .1)
        self.assertEqual(row["Qasper"], .3)
        self.assertEqual(row["repobench-p_e"], .25)

    def test_complete_means_and_256k_twelve_task_protocol(self):
        row = {"Model": "m", "BoolQ": .6, "long_ppl_128k": 25., "niah_single_1@64k": .9}
        row.update({f"{task}@256k": .5 for task in tables.RULER_AT["256k"]})
        frame = tables.build_table([row], compact=True)
        self.assertNotIn("AVG_RULER@64k", frame)
        self.assertEqual(frame.iloc[0]["AVG_RULER@64k_n"], 1)
        self.assertEqual(frame.iloc[0]["AVG_RULER@256k"], .5)
        self.assertEqual(frame.iloc[0]["AVG_RULER@256k_n"], 12)
        self.assertEqual(frame.iloc[0]["AVG_all"], .6)  # no long-context scores or PPL mixed in

    def test_cli_join_csv_xlsx_and_detailed_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for name, scores in (("main", {"long_ppl_128k": {"ppl,none": 12.4}}),
                                 ("niah", {"niah_single_1": {"131072,none": .8}})):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"checkpoint_dir": "/models/m", "model_max_seq_length": 132096,
                                            "results": {"results": scores}}))
                inputs += ["--input", f"word_first128={path}"]
            output, detail = root / "summary.csv", root / "metrics.csv"
            tables.main([*inputs, "--compact", "-o", str(output), "--long-output", str(detail)])
            with output.open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ContextLimit"], "132096")
            self.assertEqual(rows[0]["niah_single_1@128k"], "0.8")
            with detail.open(encoding="utf-8-sig") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)
            xlsx = root / "summary.xlsx"
            tables.main([*inputs, "--compact", "-o", str(xlsx)])
            import pandas as pd
            with pd.ExcelFile(xlsx) as workbook:
                self.assertEqual(workbook.sheet_names, ["summary", "metrics"])

    def test_merge_rejects_duplicate_results_and_different_context_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.json" for i in range(2)]
            for path in paths:
                path.write_text(json.dumps({"results": {"boolq": {"acc,none": .6}}}))
            with self.assertRaisesRegex(ValueError, "duplicate result"):
                tables.collect_inputs([("m", path) for path in paths])
            for i, path in enumerate(paths):
                path.write_text(json.dumps({"model_max_seq_length": 65536 * (i + 1),
                                            "results": {f"task{i}": {"acc,none": .6}}}))
            with self.assertRaisesRegex(ValueError, "ContextLimit"):
                tables.collect_inputs([("m", path) for path in paths])

    def test_eval_csv_contains_subtasks_ppl_and_length_metrics(self):
        tree = ast.parse((ROOT / "eval.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_results_to_csv")
        scope = {"csv": csv, "Path": Path}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "eval.py", "exec"), scope)
        raw = {"results": {"babilong_qa1_64k": {"acc,none": .8}, "long_ppl_64k": {"ppl,none": 12.},
                           "niah_single_1": {"65536,none": .7, "131072,none": .6}}}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "results.json"
            scope["extract_results_to_csv"](raw, "babilong_64k,long_ppl_64k,niah_single_1", path)
            with path.with_suffix(".csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1]["context_length"], "131072")

    def test_import_has_no_output_and_bad_input_is_not_silently_empty(self):
        result = subprocess.run([sys.executable, "-c", "import unused.parse_lmeval_table"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, "")
        with self.assertRaisesRegex(ValueError, "results mapping"):
            result_rows({"results_str": "old fallback"})


if __name__ == "__main__":
    unittest.main()
