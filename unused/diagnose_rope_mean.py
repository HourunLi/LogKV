"""Compare a single fixed real slot with its dense members on a baseline prefill."""
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from niah_needle_trace import build_lm
import litgpt.model as model_module


def measure(cache, query, keys, values, scale, samples):
    # Group 0, first GQA query head, last query of the block (fully causal).
    w = cache.level_w[0, 0]
    lo, hi = cache.level_p_lo[0, 0], cache.level_p_hi[0, 0]
    # Both endpoints identify a pair exactly. For larger masses only use
    # contiguous entries: uniqueness implies every position is a member.
    eligible = ((w == 2) | ((w > 2) & (hi - lo + 1 == w))).nonzero()
    if not len(eligible):
        return
    scores = (keys @ query) * scale
    probs = scores.softmax(0)
    full_output = probs @ values
    indices = torch.linspace(0, len(eligible)-1, min(samples, len(eligible))).long()
    for index in eligible[indices].tolist():
        slot = tuple(index)
        a, b, mass = int(lo[slot]), int(hi[slot]), int(w[slot])
        members = [a, b] if mass == 2 else list(range(a, b+1))
        subset = probs[members]
        exact_mass = subset.sum()
        midpoint = (2 * int(cache.level_sum_wp[0, 0][slot]) + mass) // (2 * mass)
        raw = cache.level_k[0, 0][slot]
        mid = cache._ladder_read_keys(raw, lo.new_tensor(midpoint)).float()
        rest = scores.clone()
        rest[members] = -torch.inf
        rest_log_mass = rest.logsumexp(0)
        rest_value = rest.softmax(0) @ values
        record = dict(mass=mass, span=b-a+1, peak_member_share=float(scores[members].softmax(0).max()))
        for name, key in [('mid', mid), ('post', keys[members].mean(0))]:
            logit = query.dot(key) * scale + math.log(mass)
            slot_prob = (logit - rest_log_mass).sigmoid()
            output = slot_prob * values[members].mean(0) + (1-slot_prob) * rest_value
            record[name + '_mass_abs_error'] = float((slot_prob - exact_mass).abs())
            record[name + '_output_l2'] = float((output-full_output).norm())
        yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--length', type=int, default=32768)
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    _, lm = build_lm(args)
    lm._set_eval_cache()
    target = lm.model.transformer.h[args.layer].attn.kv_cache
    if target.semantic_anchor_mode != 'mid' or args.slots < 1:
        parser.error('use the mid baseline and at least one slot')
    ids = lm.tokenizer.encode(args.text.read_text())[:args.length].to(args.device)
    original = model_module.log_kv_chunk_attention
    key_blocks, value_blocks, records = [], [], []

    def observe(cache, q, k, v, scale, *other, **kwargs):
        if cache is target:
            key_blocks.append(k[0, 0].float())
            value_blocks.append(v[0, 0].float())
            for row in measure(cache, q[0, 0, -1].float(), torch.cat(key_blocks),
                               torch.cat(value_blocks), scale, args.slots):
                records.append(dict(layer=args.layer, query_pos=cache.token_count+k.size(2)-1, **row))
        return original(cache, q, k, v, scale, *other, **kwargs)

    with torch.inference_mode(), patch.object(model_module, 'log_kv_chunk_attention', observe):
        lm.model(ids[None], input_pos=torch.arange(len(ids), device=ids.device), lm_head_start=len(ids)-1)
    if not records:
        raise SystemExit('No merged slots observed. Supply a longer text.')
    args.out.write_text(''.join(json.dumps(row)+'\n' for row in records))


if __name__ == '__main__':
    main()
