"""unused/parse_lmeval_table.py reads what is pasted into its model_outputs: result tables, or whole eval logs
(the tables eval.py prints, with whatever surrounds them), one row per model across the logs pasted under its
name, with the long-context columns at 64K / 128K / 256K."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RULER_256K = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
              "niah_multikey_3", "niah_multivalue", "niah_multiquery", "ruler_vt", "ruler_cwe", "ruler_fwe",
              "ruler_qa_hotpot"]


@pytest.fixture(scope="module")
def parse():
    pytest.importorskip("pandas")
    spec = importlib.util.spec_from_file_location("parse_lmeval_table", ROOT / "unused" / "parse_lmeval_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # builds the column tables only; the CSV is printed under __main__
    return module


def _table(results, groups=()):
    """What eval.py prints: lm-eval's make_table over results (groups first, their subtasks as " - name")."""
    pytest.importorskip("pytablewriter")
    from lm_eval.utils import make_table

    entries = {}
    for name, metrics in results.items():
        entry = {f"{m},none": v for m, v in metrics.items()}
        entry.update({f"{m}_stderr,none": "N/A" for m in metrics})
        entry["alias"] = f" - {name}" if any(name in members for members in dict(groups).values()) else name
        entries[name] = entry
    return make_table(dict(results=entries, versions={k: 0 for k in entries}, **{"n-shot": {k: 0 for k in entries}},
                           higher_is_better={k: {} for k in entries}))


def _log(*tables, prefix="[rank0]: 2026-11-09 10:00:00 | INFO | "):
    """A log: progress lines around the tables, and a prefix (with its own bars) before every line."""
    lines = ["Running loglikelihood requests: 100%|##########| 400/400 [01:00<00:00]"]
    for table in tables:
        lines += ["[eval] 结果:"] + table.splitlines() + ["KV compression: ..."]
    return "\n".join(prefix + line for line in lines)


def test_one_row_from_a_models_logs_at_three_lengths(parse):
    niah = {t: {"4096": -1.0, "262144": 0.5 + i / 100} for i, t in enumerate(RULER_256K)}  # NIAH stage: lengths only
    babilong = {"babilong_256k": {"acc": 0.3}, **{f"babilong_qa{i}_256k": {"acc": i / 10} for i in range(1, 6)}}
    main = {**babilong, "long_ppl_256k": {"ppl": 9.5, "bits_per_byte": 0.8},
            "longbench2_256k": {"acc": 0.27}, "longbench2_academic_single": {"acc": 0.2}}
    groups = [("babilong_256k", [f"babilong_qa{i}_256k" for i in range(1, 6)]),
              ("longbench2_256k", ["longbench2_academic_single"])]
    log_256k = _log(_table(niah), _table(main, groups))
    log_128k = _log(_table({"niah_single_1": {"131072": 0.9}, "ruler_qa_squad": {"131072": 0.4},
                            "longbench2_128k": {"acc": 0.3}}), prefix="")
    # As pasted into model_outputs: one model's two logs one after the other, and a second model.
    table = parse.build_table({"wf": log_256k + "\n" + log_128k, "wf_256k": log_256k}).set_index("Model")
    row = table.loc["wf"]
    assert list(table.columns[:1]) == ["Common Sense Reasoning (ACC)"] and "Model" not in table.columns
    assert row["niah_single_1@256k"] == 0.5 and row["ruler_qa_hotpot@256k"] == 0.61
    assert "ruler_qa_squad@256k" not in table.columns  # SQuAD cannot fill 256K
    assert row["AVG_RULER@256k"] == pytest.approx(sum(0.5 + i / 100 for i in range(12)) / 12, abs=1e-4)
    assert row["babilong_qa3_256k"] == 0.3 and row["AVG_babilong_256k"] == pytest.approx(0.3)
    assert row["long_ppl_256k"] == 9.5  # ppl, not bits_per_byte
    assert row["LongBench2@256k"] == 0.27 and row["LongBench2@128k"] == 0.3
    assert row["niah_single_1@128k"] == 0.9 and row["ruler_qa_squad@128k"] == 0.4
    assert row["niah_single_1"] != row["niah_single_1"]  # the 32K column stays empty (NaN)
    assert table.loc["wf_256k", "LongBench2@256k"] == 0.27
    columns = list(table.columns)
    assert columns.index("LongBench2") < columns.index("LongBench2@64k") < columns.index("LongBench2@256k")


def test_pasted_tables_still_parse_as_before(parse):
    table = parse.build_table({"m": _table({"piqa": {"acc": 0.7}, "niah_single_1": {"32768": 0.9}})}).set_index("Model")
    assert table.loc["m", "PIQA"] == 0.7 and table.loc["m", "niah_single_1"] == 0.9
    assert parse.table_lines("no table here\n|a|b|\n") == []


def test_the_header_only_grows_at_the_end(parse):
    """Tables already filled line up by position: tests/parse_lmeval_table_columns.txt holds the header as it
    was, and the output must still start with it. Add new columns at the end of ORDERED_COLUMNS and append
    them to that file."""
    frozen = (ROOT / "tests" / "parse_lmeval_table_columns.txt").read_text().splitlines()
    header = list(parse.build_table({"m": ""}).columns)
    assert header[:len(frozen)] == frozen
    # The 64K / 128K layout (144 columns, up to LongBench2) comes before everything 256K added.
    assert frozen[143] == "LongBench2" and frozen[144] == "niah_single_1@256k"
