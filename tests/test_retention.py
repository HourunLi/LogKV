"""niahLogKV: learned retention heads score Alpha's exact spans; synthetic facts train them."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import litgpt.retention as rt
from litgpt.alpha_log_kv import cut_spans, score_spans
from litgpt.config import Config
from litgpt.log_kv_cache import LogStructuredKVCache
from litgpt.model import GPT
from litgpt.retention import RETENTION_CAP, RETENTION_FLOOR, RetentionHead, RetentionHeads
from litgpt.retention_train import HeadTrainer, pool_recall
from litgpt.synthetic_retrieval import FACT_TEMPLATES, KINDS, QUESTION_TEMPLATES, RetrievalKit, insert_facts

KWARGS = dict(batch_size=2, B=8, recent_size=8, prefill_block=8, second_order_scale=0., semantic_clusters=True,
              cluster_k_max=4, semantic_flush_granularity=8, semantic_anchor_mode="mid",
              semantic_replay_updates=True, alpha_exact_tokens=8, alpha_span_max_tokens=4)


def _model(n_layer=2):
    torch.manual_seed(5)
    config = Config(block_size=128, n_layer=n_layer, n_embd=32, n_head=4, n_query_groups=2, vocab_size=41,
                    padding_multiple=1, rotary_percentage=1.)
    model = GPT(config).eval()
    model.set_alpha_span_boundary_ids([0, 3, 7], line_end_ids=[7])
    return model


def _firing(heads, scale=3., bias=1.):
    for head in heads.heads:
        torch.nn.init.normal_(head.out.weight, std=scale)
        torch.nn.init.constant_(head.out.bias, bias)
    return heads


def _pools(model, x, heads, stream=False):
    model.set_retention_heads(heads)
    model.set_log_kv_cache(retention_head=heads is not None, **KWARGS)
    with torch.no_grad():
        if stream:
            for t in range(x.size(1)):
                model(x[:, t:t + 1], input_pos=torch.tensor([t]))
        else:
            model(x, input_pos=torch.arange(x.size(1)))
    return [block.attn.kv_cache._alpha_positions for block in model.transformer.h]


def test_span_values_take_the_largest_token_logit_in_the_cue_form():
    torch.manual_seed(0)
    head = RetentionHead(2, 4, 3)
    torch.nn.init.normal_(head.out.weight, std=4.)
    torch.nn.init.constant_(head.out.bias, .5)
    k, v = torch.randn(2, 2, 10, 4), torch.randn(2, 2, 10, 3)
    token_batch = torch.tensor([0, 0, 0, 1, 1, 0, 1])
    tokens = torch.tensor([1, 2, 3, 0, 4, 9, 7])
    sizes = torch.tensor([3, 2, 1, 1])
    values = rt.span_values(head, k, v, None, token_batch, tokens, sizes)
    z = head(k[token_batch, :, tokens], v[token_batch, :, tokens]).detach().numpy()
    best = [z[0:3].max(), z[3:5].max(), z[5], z[6]]
    expect = [max(math.exp(min(b, RETENTION_CAP)) - math.exp(RETENTION_FLOOR), 0) / n for b, n in zip(best, [3, 2, 1, 1])]
    assert values.tolist() == pytest.approx(expect, rel=1e-5)
    assert not values.requires_grad
    # RMS-normalised inputs: the head ignores the scale of K and V.
    assert torch.allclose(head(k[0, :, :3].transpose(0, 1) * 7, v[0, :, :3].transpose(0, 1) * .1),
                          head(k[0, :, :3].transpose(0, 1), v[0, :, :3].transpose(0, 1)), atol=1e-5)


def test_score_spans_adds_the_head_values_on_the_device():
    torch.manual_seed(1)
    k, v = torch.randn(2, 2, 24, 8), torch.randn(2, 2, 24, 8)
    ends = [[(i + 1) % 4 == 0 for i in range(16)]] * 2
    cut = cut_spans([[(4, True), (4, True)], [(4, True)]], [list(range(8)), list(range(4))],
                    [list(range(8, 24)), list(range(4, 20))], ends, 8, 8)
    head = _firing(RetentionHeads(1, 2, 8, 8))[0]
    plain = score_spans(k, v, cut, beta_novelty=False)
    learned = score_spans(k, v, cut, beta_novelty=False, retention=head)
    token_batch = np.repeat(np.arange(2), [len(r) for r in cut.token_rows])
    extra = rt.span_values(head, k, v, cut, torch.as_tensor(token_batch), torch.as_tensor(np.concatenate(cut.token_rows)),
                           torch.as_tensor(np.concatenate(cut.length_rows)))
    assert torch.allclose(learned, plain + extra) and extra.max() > 0


def test_a_fresh_head_leaves_selection_unchanged_and_a_trained_one_changes_it():
    x = torch.randint(8, 20, (2, 64), generator=torch.Generator().manual_seed(105))
    model = _model()
    base = _pools(model, x, None)
    assert _pools(model, x, RetentionHeads(2, 2, 8, 8)) == base  # every fresh logit is below the floor
    firing = _firing(RetentionHeads(2, 2, 8, 8))
    assert _pools(model, x, firing) == _pools(model, x, firing, stream=True)  # prefill == decode
    assert _pools(model, x, firing)[0] != base[0]


def test_a_head_that_detects_the_needle_keeps_it_in_the_exact_pool():
    torch.manual_seed(7)
    n = 512
    k, v = torch.randn(2, 2, n, 8), torch.randn(2, 2, n, 8)
    needle = list(range(300, 306))
    v[:, :, needle, 0] += 6.  # A direction only the needle's values carry.
    rng = np.random.default_rng(7)
    ends = np.zeros(n, dtype=bool)
    ends[rng.choice(np.arange(8, n), 60, replace=False)] = True
    ends[needle[0] - 1] = ends[needle[-1]] = True
    ends[needle[0]:needle[-1]] = False
    head = RetentionHead(2, 8, 8, hidden=1)
    with torch.no_grad():
        head.inp.weight.zero_()
        head.inp.bias.fill_(-1.)
        head.inp.weight[0, 16] = head.inp.weight[0, 24] = 1.  # V[group 0, 0] and V[group 1, 0]
        head.out.weight.fill_(4.)
        head.out.bias.fill_(-3.)
    kept = {}
    for name, attached in (("plain", None), ("head", head)):
        c = LogStructuredKVCache((2, 2, 1024, 8), (2, 2, 1024, 8), B=8, recent_size=16, semantic_clusters=True,
                                 cluster_k_max=4, semantic_anchor_mode="mid", allocate_second_order=False,
                                 semantic_replay_updates=True, semantic_flush_granularity=16,
                                 cos_cache=torch.ones(1024, 8), sin_cache=torch.zeros(1024, 8), rope_n_elem=8,
                                 alpha_exact_tokens=16, alpha_span_max_tokens=8, beta_novelty=True,
                                 beta_adaptive_merge=True, gamma_level0_reinsert=True, gamma_top_merge="lightest",
                                 retention_head=attached is not None)
        c.attach_retention_head(attached)
        with torch.no_grad():
            for s in range(0, n, 16):
                c.add_recent(k[:, :, s:s + 16], v[:, :, s:s + 16], k_raw=k[:, :, s:s + 16],
                             span_ends=[ends[s:s + 16].tolist()] * 2)
        kept[name] = [set(row) for row in c._alpha_positions]
    assert all(set(needle) <= row for row in kept["head"])
    assert not any(set(needle) <= row for row in kept["plain"])


def test_heads_are_validated_saved_and_bound_to_replay(tmp_path):
    with pytest.raises(ValueError, match="requires alpha_exact_tokens"):
        LogStructuredKVCache((1, 2, 64, 8), (1, 2, 64, 8), B=8, recent_size=8, retention_head=True)
    model = _model()
    with pytest.raises(RuntimeError, match="set_retention_heads"):
        model.set_log_kv_cache(retention_head=True, **KWARGS)
    with pytest.raises(ValueError, match="3 retention heads for 2 layers"):
        model.set_retention_heads(RetentionHeads(3, 2, 8, 8))
    model.set_retention_heads(RetentionHeads(2, 1, 8, 8))
    with pytest.raises(ValueError, match="expects 1 groups"):
        model.set_log_kv_cache(retention_head=True, **KWARGS)
    heads = _firing(RetentionHeads(2, 2, 8, 8))
    heads.save(tmp_path / "heads.pth", step=3)
    loaded = RetentionHeads.load(tmp_path / "heads.pth")
    assert loaded.tag() == heads.tag() and loaded.meta == {"step": 3}
    assert all(not p.requires_grad and p.dtype == torch.float32 for p in loaded.parameters())
    model.set_retention_heads(loaded)
    model.set_log_kv_cache(retention_head=True, **KWARGS)
    cache = model.transformer.h[0].attn.kv_cache
    assert cache._retention is loaded[0] and cache._retention_tag == loaded.tag()
    assert "retention" not in " ".join(model.state_dict())  # not part of the LM checkpoint
    model.set_log_kv_cache(**KWARGS)
    assert model.transformer.h[0].attn.kv_cache._retention is None
    # The tag is part of the training replay signature.
    c = LogStructuredKVCache((2, 2, 4096, 8), (2, 2, 4096, 8), B=8, recent_size=16, semantic_clusters=True,
                             cluster_k_max=4, semantic_anchor_mode="mid", allocate_second_order=False,
                             semantic_replay_updates=True, semantic_flush_granularity=16,
                             cos_cache=torch.ones(4096, 8), sin_cache=torch.zeros(4096, 8), rope_n_elem=8,
                             alpha_exact_tokens=16, alpha_span_max_tokens=8, retention_head=True)
    c.attach_retention_head(loaded[0])
    from litgpt.log_kv_cache import LogKVStreamTrainingAttention
    q, kk, vv = [torch.randn(2, 2, 48, 8, requires_grad=True) for _ in range(3)]
    span_ends = [[i % 5 == 4 for i in range(48)]] * 2
    y = LogKVStreamTrainingAttention.apply(q, kk, vv, c, .3, 16, 0., kk, span_ends)
    other = RetentionHeads(1, 2, 8, 8)[0]
    other.tag = "different"
    c.attach_retention_head(other)
    with pytest.raises(RuntimeError, match="configuration changed"):
        y.sum().backward()


# A character-level stand-in tokenizer: one id per character.
def _kit(kinds=KINDS):
    return RetrievalKit([], lambda s: [ord(c) for c in s], [ord("."), ord("\n")], kinds)


def _decode(ids):
    return "".join(map(chr, ids))


def _haystack(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    words = ["lorem", "ipsum", "dolor", "sit", "amet"]
    text = " ".join(rng.choice(words) + ("." if rng.random() < .1 else "") for _ in range(n))
    return np.array([ord(c) for c in text[:n]], dtype=np.int64)


def test_insert_facts_keeps_the_length_and_marks_exactly_the_fact_sentences():
    hay = _haystack()
    kit = _kit()
    ids, mask, facts, asked = insert_facts(hay, np.random.default_rng(3), kit, facts=(4, 4), questions=(2, 2),
                                           return_facts=True)
    assert len(ids) == len(hay) and ids.dtype == np.int64 and mask.dtype == bool and len(facts) == 4
    # The marked tokens are the facts' sentences, in block order; nothing else is marked.
    assert _decode(ids[mask]) == "".join(" " + f.text for f in facts)
    tail = "\n\n" + "\n\n".join(f.question for f in asked)
    assert _decode(ids).endswith(tail) and len(asked) == 2 and not mask[-len(tail):].any()
    # Each fact starts right after a sentence end (one is always within reach in this haystack).
    starts = np.flatnonzero(mask & ~np.concatenate(([False], mask[:-1])))
    assert all(chr(ids[s - 1]) in "." for s in starts if not mask[s - 1])
    # The haystack keeps its order; its own tail made the room.
    assert _decode(ids[~mask][:-len(tail)]) == _decode(hay[:len(hay) - mask.sum() - len(tail)])
    again = insert_facts(hay, np.random.default_rng(3), kit, facts=(4, 4), questions=(2, 2))
    assert (again[0] == ids).all() and (again[1] == mask).all()


def test_facts_use_the_chosen_value_kinds_and_never_rulers_wording():
    for template in FACT_TEMPLATES + QUESTION_TEMPLATES:
        assert not any(w in template.lower() for w in ("special magic", "one of the", "mentioned in the provided"))
    rng = np.random.default_rng(0)
    kit = _kit(kinds=("number",))
    for _ in range(20):
        fact = kit.fact(rng)
        assert fact.value.isdigit() and 3 <= len(fact.value) <= 10 and fact.value in fact.text
        assert fact.key in fact.question and fact.value in fact.question
    kit = _kit(kinds=("word", "name"))
    assert not any(any(ch.isdigit() for ch in kit.fact(rng).value) for _ in range(20))
    with pytest.raises(ValueError, match="subset"):
        _kit(kinds=("emoji",))
    with pytest.raises(ValueError, match="too short"):
        insert_facts(_haystack(300), rng, _kit(), facts=(8, 8))
    values = {kind: _kit().value(rng, kind) for kind in KINDS}
    assert len(values["uuid"]) == 36 and values["date"][:1].isalnum() and "." in values["decimal"]


def test_head_trainer_learns_to_separate_the_marked_spans():
    torch.manual_seed(0)
    model = _model()
    heads = RetentionHeads(2, 2, 8, 8, hidden=16).float()
    model.set_retention_heads(heads)
    model.set_log_kv_cache(retention_head=True, **{**KWARGS, "batch_size": 1})
    trainer = HeadTrainer(heads, lr=3e-2)
    gen = torch.Generator().manual_seed(0)

    def block():
        x = torch.randint(8, 30, (1, 96), generator=gen)
        mask = np.zeros(96, dtype=bool)
        start = int(torch.randint(10, 80, (1,), generator=gen))
        x[0, start:start + 4] = torch.tensor([35, 36, 37, 38])  # The "facts" use their own token ids.
        x[0, start - 1] = x[0, start + 3] = 3
        mask[start:start + 4] = True
        return x, mask

    gaps = []
    with trainer:
        for step in range(60):
            x, mask = block()
            trainer.set_labels([mask])
            model.reset_log_kv_cache()
            with torch.no_grad():
                model(x, input_pos=torch.arange(96))
            trainer.step()
            if step % 20 == 19:
                s = trainer.summary()
                gaps.append(s["pos_logit"] - s["neg_logit"])
                trainer.reset_stats()
        trainer.set_labels(None)
    assert gaps[-1] > gaps[0] and gaps[-1] > 1
    recall = pool_recall(model, [mask])
    assert len(recall) == 2 and all(0 <= r <= 1 for r in recall)
    assert rt.RETENTION_TAP is None  # the context restores the previous tap


def test_demo_inserts_facts_into_the_picked_rows_only():
    source = Path(__file__).resolve().parents[1] / "demo.py"
    fn = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "_with_synthetic_facts")
    scope = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    rows = torch.as_tensor(np.stack([_haystack(seed=s) for s in range(4)]))
    kit = _kit()
    same = scope["_with_synthetic_facts"](rows, kit, 0.0, [1], "2,3", "1,1")
    assert same is rows
    mixed = scope["_with_synthetic_facts"](rows, kit, 1.0, [1], "2,3", "1,1")
    assert mixed.shape == rows.shape and mixed.dtype == rows.dtype
    assert all(not torch.equal(mixed[b], rows[b]) for b in range(4))


def test_trainer_finds_the_tokenizer_like_eval_sh_and_names_a_missing_checkpoint(tmp_path):
    import train_retention as tr
    base, ckpt = tmp_path / "base", tmp_path / "ckpt"
    base.mkdir()
    ckpt.mkdir()
    (base / "tokenizer.json").write_text("{}")
    assert tr.tokenizer_dir({"ckpt_dir": str(base)}) == str(base)
    assert tr.tokenizer_dir({"tokenizer_dir": str(ckpt), "ckpt_dir": str(base)}) == str(base)  # first that has one
    assert tr.tokenizer_dir({"ckpt_dir": str(ckpt)}) is None
    with pytest.raises(SystemExit, match="no checkpoint at .*YAML save_path"):
        tr.checkpoint_dir(str(ckpt), None)
    (ckpt / "lit_model.pth").write_text("")
    assert tr.checkpoint_dir(str(ckpt), None) == str(ckpt)


def test_trainer_options_come_from_the_command_line_then_the_yaml(tmp_path):
    import train_retention as tr
    cfg = {"log_kv_retention_heads": "/h/heads.pth", "retention_steps": 500, "retention_lr": "3e-4",
           "retention_kinds": "number,word", "retention_resume": "false"}
    opts = tr.resolve_options(cfg, {"steps": None, "lr": 1e-2})
    assert opts["out"] == "/h/heads.pth" and opts["steps"] == 500 and opts["lr"] == 1e-2
    assert opts["kinds"] == "number,word" and opts["resume"] is False and opts["hidden"] == 64
    assert tr.resolve_options({**cfg, "retention_out": "/o.pth"}, {})["out"] == "/o.pth"
    with pytest.raises(SystemExit, match="no output path"):
        tr.resolve_options({}, {})
    heads = RetentionHeads(1, 1, 4, 4)
    path = str(tmp_path / "h.pth")
    assert tr.resume_step(str(tmp_path / "missing.pth"), 100) == (None, 0)
    heads.save(path, step=40)
    assert tr.resume_step(path, 100) == (path, 40) and tr.resume_step(path, 30) == (path, 30)


def test_heads_save_without_rename_and_resume_from_a_cut_off_save(tmp_path, monkeypatch):
    import os
    import train_retention as tr
    heads = RetentionHeads(1, 1, 4, 4)
    out = str(tmp_path / "sub" / "h.pth")
    tr.atomic_save(heads.blob(step=20), out)
    assert tr.resume_step(out, 100) == (out, 20) and not os.path.exists(out + ".tmp")

    def no_rename(src, dst):  # an OBS bucket mount
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(tr.os, "replace", no_rename)
    tr.atomic_save(heads.blob(step=40), out)
    assert tr.resume_step(out, 100) == (out, 40) and not os.path.exists(out + ".tmp")
    assert RetentionHeads.load(out).tag() == heads.tag()
    # killed while copying over the old file: the complete .tmp wins
    heads.save(out + ".tmp", step=60)
    Path(out).write_bytes(Path(out + ".tmp").read_bytes()[:100])
    assert tr.resume_step(out, 100) == (out + ".tmp", 60)


def _launcher(tmp_path, yaml_text):
    import os
    import subprocess
    import sys
    source = (Path(__file__).resolve().parents[1] / "retention.sh").read_text()
    start = source.index("CONFIG_FILE=$1")
    stop = source.index("RETENTION_ARGS=${RETENTION_ARGS:-}", start)
    config = tmp_path / "run.yaml"
    config.write_text(yaml_text)
    env = {**os.environ, "PYTHON_BIN": sys.executable}
    return subprocess.run(["bash", "-c", source[start:stop] + 'echo "OUT=${RETENTION_OUT} SAVE=${SAVE_DIR}"',
                           "bash", str(config)], capture_output=True, text=True, env=env)


def test_retention_launcher_reads_the_yaml_and_checks_the_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt"
    base = f"save_path: {ckpt}\nlog_kv_semantic_clusters: true\nlog_kv_retention_heads: {tmp_path}/heads/h.pth\n"
    missing = _launcher(tmp_path, base)
    assert missing.returncode == 1 and "没有 lit_model.pth" in missing.stdout
    (ckpt / "step_400").mkdir(parents=True)
    (ckpt / "step_400" / "lit_model.pth").write_text("x")
    found = _launcher(tmp_path, base)
    assert found.returncode == 0, found.stdout + found.stderr
    assert f"OUT={tmp_path}/heads/h.pth SAVE={ckpt}" in found.stdout and (tmp_path / "heads").is_dir()
    other = _launcher(tmp_path, base + f"retention_out: {tmp_path}/o.pth\n")
    assert f"OUT={tmp_path}/o.pth" in other.stdout
    no_alpha = _launcher(tmp_path, base + "log_kv_alpha_exact_tokens: 0\n")
    assert no_alpha.returncode == 1 and "Alpha" in no_alpha.stdout
    no_out = _launcher(tmp_path, f"save_path: {ckpt}\nlog_kv_semantic_clusters: true\n")
    assert no_out.returncode == 1 and "retention_out" in no_out.stdout


def test_training_state_resumes_heads_optimizer_and_positive_weight(tmp_path):
    import train_retention as tr
    torch.manual_seed(0)
    heads = RetentionHeads(2, 1, 4, 4, hidden=8)
    trainer = HeadTrainer(heads, lr=1e-2)
    for p in heads.parameters():
        p.grad = torch.randn_like(p)
    trainer.step()
    trainer.counts += (90., 3.)
    out = str(tmp_path / "h.pth")
    tr.save_training(heads, trainer, out, step=10)
    assert tr.resume_step(out, 100) == (out + ".state", 10)  # a tie goes to the training state
    assert RetentionHeads.load(out).tag() == RetentionHeads.load(out + ".state").tag() == heads.tag()

    resumed = RetentionHeads.load(out + ".state")
    again = HeadTrainer(resumed, lr=3e-3)
    again.load_state_dict(torch.load(out + ".state", map_location="cpu", weights_only=True)["trainer"])
    assert np.array_equal(again.counts, trainer.counts)
    assert again.optimizer.param_groups[0]["lr"] == 3e-3  # the configured rate, not the saved one
    before, after = trainer.optimizer.state_dict()["state"], again.optimizer.state_dict()["state"]
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key]["exp_avg"], after[key]["exp_avg"])
        assert torch.equal(before[key]["exp_avg_sq"], after[key]["exp_avg_sq"])
        assert float(before[key]["step"]) == float(after[key]["step"]) == 1.
    # A newer heads file without a training state (an older run) still wins on its step.
    heads.save(out, step=20)
    assert tr.resume_step(out, 100) == (out, 20)
