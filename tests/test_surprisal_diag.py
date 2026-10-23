import json
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


import pool_policy_sim as ps  # noqa: E402


def test_surprisal_store_round_trips_sequences_fields_and_settings(tmp_path):
    path = tmp_path / "s.npz"
    store = ns.SurprisalStore(str(path), recent=16, flush=8, max_span=4, pool=8, boundary_ids=[3])
    store.add([1, 2, 3], [np.nan, 1., 2.], task="niah_single_2", index=0, depth=.5)
    store.add([4, 5], [np.nan, 3.], task="niah_single_3", index=1, depth=.25)
    store.save()
    meta, rows = ns.SurprisalStore.load(str(path))
    assert meta["flush"] == 8 and meta["boundary_ids"] == [3]
    assert rows[1]["ids"].tolist() == [4, 5] and rows[1]["surprisal"][1] == 3. and math.isnan(rows[0]["surprisal"][0])
    assert rows[0]["task"] == "niah_single_2" and rows[1]["depth"] == .25 and rows[1]["index"] == 1


def _stream(n=400, needle=40, reuse_at=None, **kwargs):
    """Spans of 4 tokens (boundary id 3); a surprising needle span at ``needle``; a repeat of its
    rare bigram at ``reuse_at``; everything else 1 bit."""
    ids = np.full(n, 1, dtype=np.int64)
    ids[3::4] = 3
    s = np.full(n, 1.)
    s[0] = np.nan
    ids[needle:needle + 2] = [20, 21]
    s[needle:needle + 2] = 9.
    if reuse_at is not None:
        ids[reuse_at:reuse_at + 2] = [20, 21]
    boundary = np.zeros(32, dtype=bool)
    boundary[3] = True
    return ps.Stream(ids, s, boundary, 4, reuse_n=2, reuse_bits=6., label_n=2, label_bits=6., recent=16,
                     value_start=needle, value_stop=needle + 2, task="t", depth=needle / n, **kwargs)


def test_surprisal_admission_keeps_the_needle_and_recency_forgets_it():
    stream = _stream()
    kept = ps.replay(stream, ps.Policy(score="mean"), pool=8, recent=16, flush=8)
    assert ps.needle_kept(stream, kept) and sum(stream.stops[s] - stream.starts[s] for s in kept["kept"]) <= 8
    recency = ps.replay(stream, ps.Policy(score="recency", decay="exp", tau=16.), pool=8, recent=16, flush=8)
    assert not ps.needle_kept(stream, recency)


def test_forgetting_drops_an_old_needle_and_reuse_keeps_it():
    def run(reuse_at=None, **policy):
        stream = _stream(reuse_at=reuse_at)
        return ps.needle_kept(stream, ps.replay(stream, ps.Policy(**policy), pool=4, recent=16, flush=8))
    # A one-span pool. The needle span averages 5 bits, every other span 1 bit; with tau = 150 the
    # needle falls below a fresh span about 270 tokens after it was written.
    assert run(score="mean")                                                       # no forgetting
    assert not run(score="mean", decay="exp", tau=150.)                            # forgotten by the end
    assert run(reuse_at=290, score="mean", decay="exp", tau=150., reuse="refresh")  # its bigram recurs in time
    assert not run(reuse_at=290, score="mean", decay="exp", tau=150.)              # the same, reuse ignored


def test_far_reuse_hits_count_the_copy_exact_when_it_is_read():
    stream = _stream(reuse_at=380)
    assert stream.events_a.tolist() == [40] and stream.events_b.tolist() == [380]  # n-gram starts
    assert ps.replay(stream, ps.Policy(score="mean"), pool=8, recent=16, flush=8)["hits"] == 1
    assert ps.replay(stream, ps.Policy(score="recency", decay="exp", tau=16.), 8, 16, 8)["hits"] == 0


def test_default_grid_names_are_unique():
    names = [p.name for p in ps.default_policies()]
    assert len(names) == len(set(names)) == 55


def test_finished_runs_rebuild_their_summaries_from_the_saved_files(tmp_path):
    record = dict(task="niah_single_2", value_surprisal=[9.], document_surprisal_median=2.,
                  ranks={name: dict(rank=1, tokens_needed=20, percentile=1., score=9.) for name in ns.SCORES})
    (tmp_path / "s.jsonl").write_text((json.dumps(record) + "\n") * 2)
    text = ns.report(tmp_path / "s.jsonl")
    assert text.startswith("niah_single_2: samples=2") and "P1024:1.00" in text
    ns.write_summary(tmp_path / "s.jsonl", text)
    assert (tmp_path / "s.summary.txt").read_text().strip() == text
    row = dict(n=3, threshold=6., blocks=1, pairs_per_block=1., token_share=.01, spans_per_block=2., span_tokens_median=40.,
               fits_pool=1., s_first=7., s_next=3., kinds={"capitalised": 1.})
    (tmp_path / "r.json").write_text(json.dumps(dict(config="c.yaml", gap=2048, pool=1024, grid=[row], examples=[])))
    assert "    6    1.000       2.0       40      1.00    7.00   3.00" in rl.report_file(tmp_path / "r.json")
    saved = dict(pool=8, recent=16, flush=8, reuse_n=2, reuse_bits=6., label_n=3, label_bits=6.,
                 rows=[dict(policy="mean|none|none", niah={"niah_single_2": 1.}, niah_early=1., natural_32k=.5, reused=.2)])
    text = ps.summary(saved)
    assert "best 10 by mean NIAH:" in text and "best 10 by natural_32k:" in text and text.count("mean|none|none") == 3
