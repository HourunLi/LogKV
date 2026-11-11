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
    assert row["niah_single_1_256k"] == 0.5 and row["ruler_qa_hotpot_256k"] == 0.61
    assert "ruler_qa_squad_256k" not in table.columns  # SQuAD cannot fill 256K: no column
    assert row["AVG_RULER_256k"] == pytest.approx(sum(0.5 + i / 100 for i in range(12)) / 12, abs=1e-4)
    assert row["babilong_qa3_256k"] == 0.3 and row["AVG_babilong_256k"] == pytest.approx(0.3)
    assert row["long_ppl_256k"] == 9.5  # ppl, not bits_per_byte
    assert row["LongBench2_256k"] == 0.27 and row["LongBench2_128k"] == 0.3
    assert row["niah_single_1_128k"] == 0.9 and row["ruler_qa_squad_128k"] == 0.4
    assert row["niah_single_1"] != row["niah_single_1"]  # the 32K column stays empty (NaN)
    assert table.loc["wf_256k", "LongBench2_256k"] == 0.27
    columns = list(table.columns)
    assert columns.index("LongBench2_32k") < columns.index("LongBench2_64k") < columns.index("LongBench2_256k")
    assert "LongBench2" not in columns  # a result without its context budget has no column


def test_pasted_tables_still_parse_as_before(parse):
    table = parse.build_table({"m": _table({"piqa": {"acc": 0.7}, "niah_single_1": {"32768": 0.9},
                                            "ruler_vt": {"32768": 0.4}})}).set_index("Model")
    row = table.loc["m"]
    assert row["PIQA"] == 0.7 and row["niah_single_1"] == 0.9
    # The 32K block repeats the main line's needles under the names every length uses.
    assert row["niah_single_1_32k"] == 0.9 and row["AVG_NIAH_32k"] == 0.9 and row["ruler_vt_32k"] == 0.4
    assert row["AVG_RULER_32k"] == pytest.approx(0.65)
    assert row["AVG_all"] == pytest.approx(0.8)  # the length blocks stay out of AVG_all
    assert row["niah_single_1_64k"] != row["niah_single_1_64k"]
    assert parse.table_lines("no table here\n|a|b|\n") == []


def test_the_header_keeps_each_length_together(parse):
    """tests/parse_lmeval_table_columns.txt is the header: the 32K main line (89 columns), then one block per length
    (32K, 64K, 128K, 256K), each length's columns contiguous, in the same order in every block, and only the columns
    that length's evaluations fill. A new column goes into the blocks of the lengths it runs at and into that file."""
    import re

    frozen = (ROOT / "tests" / "parse_lmeval_table_columns.txt").read_text().splitlines()
    header = list(parse.build_table({"m": ""}).columns)
    assert header == frozen
    assert header[89] == "AVG_NIAH_words" and header[90] == "niah_single_1_32k"  # the main line, then 32K's block
    # The task's length: the first _<length> in the name (long_ppl_books_64k_8k is 64K's curve).
    lengths = [m.group(1) if (m := re.search(r"_(32k|64k|128k|256k)(?=_|$)", c)) else None for c in header[90:]]
    assert None not in lengths
    order = [n for i, n in enumerate(lengths) if i == 0 or n != lengths[i - 1]]
    assert order == ["32k", "64k", "128k", "256k"]  # no length comes back after another
    # Every block in the same order; a block only has the columns its evaluations fill.
    shapes = {n: [re.sub(rf"_{n}(?=_|$)", "_<n>", c, count=1) for c, m in zip(header[90:], lengths) if m == n]
              for n in ("32k", "64k", "128k", "256k")}
    full = list(shapes["128k"])  # every column a block can have: 128K's and the 256K point of the book curve
    full.insert(full.index("long_ppl_<n>"), "long_ppl_books_<n>_256k")
    for n, shape in shapes.items():
        assert shape == [c for c in full if c in shape], n
    assert len(shapes["64k"]) == 44 and len(shapes["256k"]) == 45
    assert not any(c.startswith("infinitebench") or c == "AVG_InfiniteBench_<n>" for c in shapes["32k"])  # 64K on
    assert [c for c in shapes["32k"] if c.startswith("long_ppl")] == [
        "long_ppl_books_<n>_8k", "long_ppl_books_<n>_16k", "long_ppl_books_<n>_32k", "long_ppl_<n>"]
    assert "ruler_qa_squad_<n>" not in shapes["256k"] and "long_ppl_books_<n>_256k" in shapes["256k"]


def test_turn_and_book_curve_columns(parse):
    turns = {"niah_turns_256k": {"acc": 0.625, "turn1": 0.9, "turn2": 0.6, "turn3": 0.5, "turn4": 0.5}}
    books = {"long_ppl_books_256k": {}, **{f"long_ppl_books_256k_{q}": {"ppl": 10. - i, "bits_per_byte": 0.9}
                                           for i, q in enumerate(("8k", "16k", "32k", "64k", "128k", "256k"))}}
    log = _log(_table(turns), _table({k: v for k, v in books.items() if v},
                                     [("long_ppl_books_256k", [k for k in books if k != "long_ppl_books_256k"])]))
    row = parse.build_table({"wf": log}).set_index("Model").loc["wf"]
    assert row["niah_turns_256k"] == 0.625 and row["niah_turns_256k_turn1"] == 0.9 and row["niah_turns_256k_turn4"] == 0.5
    assert [row[f"long_ppl_books_256k_{q}"] for q in ("8k", "256k")] == [10., 5.]  # ppl, not bits_per_byte
    assert row["long_ppl_books_128k_8k"] != row["long_ppl_books_128k_8k"]  # another configuration's curve: empty


def test_infinitebench_columns(parse):
    names = [f"infinitebench_{t}_128k" for t in parse.INFINITEBENCH]
    scores = {name: {"score": i / 10} for i, name in enumerate(names)}
    log = _log(_table({"infinitebench_128k": {"score": 0.45}, **scores}, [("infinitebench_128k", names)]))
    row = parse.build_table({"wf": log}).set_index("Model").loc["wf"]
    assert row["infinitebench_passkey_128k"] == 0.0 and row["infinitebench_math_find_128k"] == 0.9
    assert row["AVG_InfiniteBench_128k"] == pytest.approx(0.45)
    assert row["AVG_InfiniteBench_64k"] != row["AVG_InfiniteBench_64k"]  # another length: empty


def test_each_average_follows_its_tasks(parse):
    """Inside a length's block the columns are grouped by task, each average right after what it averages."""
    header = list(parse.build_table({"m": ""}).columns)
    for n in ("32k", "64k", "128k", "256k"):
        for avg in (f"AVG_NIAH_{n}", f"AVG_NIAH_multi_{n}", f"AVG_RULER_{n}", f"AVG_babilong_{n}",
                    *([f"AVG_InfiniteBench_{n}"] if n != "32k" else [])):
            places = sorted(header.index(t) for t in parse.AVG_GROUPS[avg])
            assert header.index(avg) == places[-1] + 1, avg
            if avg != f"AVG_RULER_{n}":  # AVG_RULER averages all of RULER, after its last group
                assert places == list(range(places[0], places[-1] + 1)), avg  # its tasks side by side
        turns = [header.index(f"niah_turns_{n}_turn{i}") for i in range(1, 5)]
        assert header.index(f"niah_turns_{n}") == turns[-1] + 1 and turns == list(range(turns[0], turns[0] + 4))
