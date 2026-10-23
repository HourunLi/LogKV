import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "unused"))
import niah_surprisal as ns  # noqa: E402
import reuse_label_stats as rl  # noqa: E402
from litgpt.config import Config  # noqa: E402
from litgpt.model import GPT  # noqa: E402

KWARGS = dict(batch_size=1, B=8, prefill_block=8, second_order_scale=0., semantic_clusters=True, cluster_k_max=4,
              semantic_flush_granularity=8, semantic_anchor_mode="mid", semantic_replay_updates=True,
              alpha_exact_tokens=8, alpha_span_max_tokens=4)


def _model():
    torch.manual_seed(3)
    config = Config(block_size=256, n_layer=2, n_embd=32, n_head=4, n_query_groups=2, vocab_size=41,
                    padding_multiple=1, rotary_percentage=1.)
    model = GPT(config).eval()
    model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7])
    return model


def test_surprisal_is_the_next_token_code_length_computed_in_chunks():
    model = _model()
    ids = torch.randint(0, 41, (60,)).tolist()
    s = ns.token_surprisal(model, ids, chunk=7)
    with torch.no_grad():
        logp = model(torch.tensor([ids])).float().log_softmax(-1)[0]
    expect = [-logp[t - 1, ids[t]].item() / math.log(2) for t in range(1, len(ids))]
    assert math.isnan(s[0]) and s[1:].tolist() == pytest.approx(expect, abs=1e-4)


def test_cached_surprisal_matches_the_uncached_one_inside_the_recent_window():
    model = _model()
    ids = torch.randint(0, 41, (40,)).tolist()
    plain = ns.token_surprisal(model, ids)
    cached = ns.token_surprisal(model, ids, prepare=lambda: model.set_log_kv_cache(recent_size=64, **KWARGS))
    assert cached[1:].tolist() == pytest.approx(plain[1:].tolist(), abs=1e-3)
    # Beyond the window the cache compresses: values stay finite, and the exact masks cover pool + recent.
    long = torch.randint(0, 41, (200,)).tolist()
    s = ns.token_surprisal(model, long, prepare=lambda: model.set_log_kv_cache(recent_size=16, **KWARGS))
    assert np.isfinite(s[1:]).all()
    masks = rl.exact_masks(model, len(long))
    assert len(masks) == 2 and all(16 <= m.sum() <= 16 + 8 and m[-16:].all() for m in masks)


def test_spans_tile_the_prompt_at_boundaries_and_the_cap():
    boundary = np.zeros(10, dtype=bool)
    boundary[3] = True
    ids = np.array([1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1])
    starts, stops = ns.cut(ids, boundary, max_span=4)
    assert starts.tolist() == [0, 3, 7, 10] and stops.tolist() == [3, 7, 10, 11]


def test_the_needle_span_ranks_by_its_surprisal_and_the_recent_window_keeps_it():
    boundary = np.zeros(10, dtype=bool)
    boundary[3] = True
    ids = np.array(([1, 1, 3] * 20))
    s = np.full(len(ids), 1.)
    s[0] = np.nan
    s[[10, 31]] = 9.  # the needle value at 10; a competitor at 31
    s[31] = 12.
    record = ns.analyse(ids, s, boundary, max_span=8, recent=9, needle=[9, 10, 11], value=[10],
                        decode=lambda t: " ".join(map(str, t)))
    assert record["needle_span"] == [9, 12] and not record["in_recent"]
    assert record["ranks"]["max"] == dict(rank=2, tokens_needed=6, percentile=pytest.approx(0.94, abs=0.01),
                                          score=9.)
    assert record["top_competitors"][0]["position"] == 30
    assert ns.analyse(ids, s, boundary, 8, 30, [9, 10, 11], [40], str)["in_recent"]


def test_reuse_pairs_need_the_gap_and_a_surprising_first_occurrence():
    ids = np.zeros(300, dtype=np.int64)
    ids[::2] = 1
    ids[10:13] = [7, 8, 9]       # first occurrence
    ids[60:63] = [7, 8, 9]       # within the gap of the first: not a far pair
    ids[200:203] = [7, 8, 9]     # far from the previous copy (60)
    s = np.full(300, 0.5)
    s[10:13] = 9.
    a, b, s_first, s_prev, s_next = rl.reuse_pairs(ids, s, 3, gap=100)
    triples = [(int(x), int(y)) for x, y in zip(a, b) if ids[x] == 7]
    assert triples == [(60, 200)] and s_first[list(a).index(60)] == pytest.approx(9.)
    boundary = np.zeros(10, dtype=bool)
    # Overlapping 3-grams that share two of the surprising tokens average 6.17 bits; 8 keeps only 7 8 9.
    row, pairs = rl.block_stats(ids, s, 3, 6., 100, boundary, 8, 1024, lambda t: "x7 8 9", recent=100)
    assert row["pairs"] == 3 and row["tokens"] == 5
    row, pairs = rl.block_stats(ids, s, 3, 8., 100, boundary, 8, 1024, lambda t: "x7 8 9", recent=100)
    assert row["pairs"] == 1 and row["tokens"] == 3 and row["fits"] and row["kinds"] == {"digits": 1}
    assert pairs == [(60, 200, pytest.approx(9.), pytest.approx(0.5))]
    row, _ = rl.block_stats(ids, s, 3, 10., 100, boundary, 8, 1024, str)
    assert row["pairs"] == 0


def test_kinds_of_reused_ngrams():
    assert [rl.kind(t) for t in (" 4829", " Garrick Soren", " foo_bar(", " candle harbor", " ...")] == \
        ["digits", "capitalised", "code", "lowercase", "other"]


def test_diag_launcher_is_majobs_environment_then_the_script(tmp_path):
    root = ROOT
    majob, diag = (root / "majob.sh").read_text().splitlines(), (root / "diag.sh").read_text().splitlines()
    assert diag[:56] == majob[:56] and "Distributed" not in "\n".join(diag)
    tail = "\n".join(diag[56:])
    script = tmp_path / "hello.py"
    script.write_text("import sys; print('args', sys.argv[1:])\n")
    env = {**os.environ, "PYTHON_BIN": sys.executable}
    ok = subprocess.run(["bash", "-c", tail, "bash", str(script), "--x", "1"], capture_output=True, text=True, env=env)
    assert ok.returncode == 0 and "args ['--x', '1']" in ok.stdout
    missing = subprocess.run(["bash", "-c", tail, "bash"], capture_output=True, text=True, env=env)
    assert missing.returncode == 1 and "未提供要运行的脚本" in missing.stdout
