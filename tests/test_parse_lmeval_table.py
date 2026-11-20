"""unused/parse_lmeval_table.py reads what is pasted into its model_outputs: result tables, or whole eval logs
(the tables eval.py prints, with whatever surrounds them), one row per model across the logs pasted under its
name, with the long-context columns from 32K through 1M."""
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
    from 32K through 1M, each length's columns contiguous, in the same order in every block, and only the columns
    that length's evaluations fill. A new column goes into the blocks of the lengths it runs at and into that file."""
    import re
    import yaml

    expected_lengths = ("32k", "64k", "128k", "256k", "512k", "1m")
    frozen = (ROOT / "tests" / "parse_lmeval_table_columns.txt").read_text().splitlines()
    header = list(parse.build_table({"m": ""}).columns)
    assert header == frozen
    assert header[89] == "AVG_NIAH_words" and header[90] == "niah_single_1_32k"  # the main line, then 32K's block
    # The task's length: the first _<length> in the name (long_ppl_books_64k_8k is 64K's curve).
    lengths = [m.group(1) if (m := re.search(rf"_({'|'.join(expected_lengths)})(?=_|$)", c)) else None
               for c in header[90:]]
    assert None not in lengths
    order = [n for i, n in enumerate(lengths) if i == 0 or n != lengths[i - 1]]
    assert order == list(expected_lengths)
    # Every block in the same order; a block only has the columns its evaluations fill.
    shapes = {n: [re.sub(rf"_{n}(?=_|$)", "_<n>", c, count=1) for c, m in zip(header[90:], lengths) if m == n]
              for n in expected_lengths}
    full = list(shapes["128k"])
    ppl_index = full.index("long_ppl_<n>")
    full[ppl_index:ppl_index] = [f"long_ppl_books_<n>_{point}" for point in ("256k", "512k", "1m")]
    for n, shape in shapes.items():
        assert shape == [c for c in full if c in shape], n
        books = yaml.safe_load((ROOT / f"eval_tasks/long_ppl/long_ppl_books_{n}.yaml").read_text())["task"]
        assert [c for c in header if c.startswith(f"long_ppl_books_{n}_")] == books
    assert [len(shapes[n]) for n in expected_lengths] == [63, 64, 65, 65, 55, 56]
    for n in ("256k", "512k", "1m"):
        assert "ruler_qa_squad_<n>" not in shapes[n]
    for n in ("512k", "1m"):
        assert not any("infinitebench" in c.lower() for c in shapes[n])


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
    for n in parse.BLOCK_LENGTHS:
        averages = [f"AVG_NIAH_{n}", f"AVG_NIAH_multi_{n}", f"AVG_RULER_{n}", f"AVG_babilong_{n}"]
        if n in parse.INFINITEBENCH_LENGTHS:
            averages.append(f"AVG_InfiniteBench_{n}")
        for avg in averages:
            places = sorted(header.index(t) for t in parse.AVG_GROUPS[avg])
            assert header.index(avg) == places[-1] + 1, avg
            if avg != f"AVG_RULER_{n}":  # AVG_RULER averages all of RULER, after its last group
                assert places == list(range(places[0], places[-1] + 1)), avg  # its tasks side by side
        turns = [header.index(f"niah_turns_{n}_turn{i}") for i in range(1, 5)]
        assert header.index(f"niah_turns_{n}") == turns[-1] + 1 and turns == list(range(turns[0], turns[0] + 4))


def test_512k_and_1m_logs_fill_their_own_task_columns(parse):
    logs = [_table({"piqa": {"acc": 0.7}})]
    for n, metric, score in (("512k", "524288", 0.2), ("1m", "1048576", 0.4)):
        ruler = {task: {metric: score} for task in RULER_256K}
        ruler["ruler_qa_squad"] = {metric: 0.9}  # invalid for both lengths
        books = {f"long_ppl_books_{n}_{point}": {"bits_per_byte": 0.5, "ppl": 11.0}
                 for point in parse.BOOK_POINTS[n]}
        babi = {f"babilong_qa{i}_{n}": {"acc": score} for i in range(1, 6)}
        lb2 = {"longbench2_code": {"acc": score + 0.05}, "longbench2_translate": {"acc": score}}
        main = {f"longbench2_{n}": {"acc": score + 0.1}, **lb2,
                f"niah_turns_{n}": {"turn1": score, "turn4": score, "acc": score},
                f"long_ppl_{n}": {"bits_per_byte": 0.6, "ppl": 10.0}, **books, **babi,
                f"infinitebench_passkey_{n}": {"score": 0.9}}
        logs.extend([_table(ruler), _table(main, [(f"longbench2_{n}", lb2)])])
    table = parse.build_table({"wf": _log(*logs)}).set_index("Model")
    row = table.loc["wf"]
    for n, score in (("512k", 0.2), ("1m", 0.4)):
        assert row[f"ruler_qa_hotpot_{n}"] == row[f"AVG_RULER_{n}"] == score
        assert row[f"babilong_qa3_{n}"] == row[f"AVG_babilong_{n}"] == score
        assert row[f"longbench2_code_{n}"] == pytest.approx(score + 0.05)
        assert row[f"LongBench2_{n}"] == pytest.approx(score + 0.1)
        assert row[f"niah_turns_{n}_turn1"] == row[f"niah_turns_{n}"] == score
        assert row[f"long_ppl_books_{n}_{n}"] == 11.0 and row[f"long_ppl_{n}"] == 10.0
        assert f"ruler_qa_squad_{n}" not in table and f"infinitebench_passkey_{n}" not in table
    assert "long_ppl_books_512k_1m" not in table and row["AVG_all"] == 0.7


LONGBENCH2_64K = """|longbench2_64k                |      0|none  |      |acc   |↑  |0.2565|±  |0.0195|
| - longbench2_academic_multi  |      0|none  |     0|acc   |↑  |0.4000|±  |0.0700|
| - longbench2_academic_single |      0|none  |     0|acc   |↑  |0.2727|±  |0.0679|
| - longbench2_agent_history   |      0|none  |     0|acc   |↑  |0.2500|±  |0.0993|
| - longbench2_code            |      0|none  |     0|acc   |↑  |0.2200|±  |0.0592|
| - longbench2_detective       |      0|none  |     0|acc   |↑  |0.1364|±  |0.0749|
| - longbench2_dialogue_history|      0|none  |     0|acc   |↑  |0.2105|±  |0.0961|
| - longbench2_event_order     |      0|none  |     0|acc   |↑  |0.1000|±  |0.0688|
| - longbench2_fin_multi       |      0|none  |     0|acc   |↑  |0.1333|±  |0.0909|
| - longbench2_fin_single      |      0|none  |     0|acc   |↑  |0.2727|±  |0.0972|
| - longbench2_govt_multi      |      0|none  |     0|acc   |↑  |0.3043|±  |0.0981|
| - longbench2_govt_single     |      0|none  |     0|acc   |↑  |0.3333|±  |0.1143|
| - longbench2_graph           |      0|none  |     0|acc   |↑  |0.2000|±  |0.1069|
| - longbench2_legal_multi     |      0|none  |     0|acc   |↑  |0.2143|±  |0.1138|
| - longbench2_legal_single    |      0|none  |     0|acc   |↑  |0.2632|±  |0.1038|
| - longbench2_lit_single      |      0|none  |     0|acc   |↑  |0.2333|±  |0.0785|
| - longbench2_many_shot       |      0|none  |     0|acc   |↑  |0.2857|±  |0.1010|
| - longbench2_news_multi      |      0|none  |     0|acc   |↑  |0.1304|±  |0.0718|
| - longbench2_table           |      0|none  |     0|acc   |↑  |0.2222|±  |0.1008|
| - longbench2_translate       |      0|none  |     0|acc   |↑  |0.4500|±  |0.1141|
| - longbench2_user_guide      |      0|none  |     0|acc   |↑  |0.2750|±  |0.0715|"""


def test_longbench2_subtasks_go_to_their_groups_length(parse):
    """LongBench v2's subtasks are listed without a length, under their longbench2_<length> group: a model's logs of
    several lengths, one after the other, fill each length's columns; the group's total comes after its subtasks."""
    log_128k = LONGBENCH2_64K.replace("longbench2_64k  ", "longbench2_128k ").replace("0.4000", "0.5000")
    lines = ["|Tasks|Version|Filter|n-shot|Metric| |Value| |Stderr|", "|---|--:|---|--:|---|---|--:|---|--:|"]
    log = "\n".join(["[eval] 结果:", *lines, LONGBENCH2_64K, "|piqa|1|none|0|acc|↑|0.7|±|0.01|",
                     "|longbench2_academic_multi|0|none|0|acc|↑|0.9|±|0.1|",  # alone, not under a group: dropped
                     "[eval] 结果:", *lines, log_128k])
    table = parse.build_table({"wf": log}).set_index("Model")
    row = table.loc["wf"]
    assert row["longbench2_academic_multi_64k"] == 0.4 and row["longbench2_user_guide_64k"] == 0.275
    assert row["LongBench2_64k"] == 0.2565  # lm-eval's total over all questions, not a mean of the 20
    assert row["longbench2_academic_multi_128k"] == 0.5 and row["LongBench2_128k"] == 0.2565
    assert row["longbench2_academic_multi_32k"] != row["longbench2_academic_multi_32k"]  # no 32K log: empty
    assert row["PIQA"] == 0.7
    filled = [c for c in table.columns if c.startswith("longbench2_") and c.endswith("_64k")]
    assert len(filled) == 20 and all(row[c] == row[c] for c in filled)
    header = list(table.columns)
    assert header.index("LongBench2_64k") == header.index("longbench2_user_guide_64k") + 1
    assert header.index("longbench2_academic_multi_64k") == header.index("AVG_babilong_64k") + 1


def test_truthfulqa_scale_in_averages(parse):
    rows = ("|truthfulqa_gen|3|none|0|rouge1_max|↑|25|±|0|\n"
            "|mmlu|2|none|0|acc|↑|0.75|±|0|")
    row = parse.build_table({"model": rows}).iloc[0]
    assert row["truthfulqa_gen"] == 25
    assert row["AVG_General"] == row["AVG_all"] == 0.5
