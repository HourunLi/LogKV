"""Bounded whole-span selection. Scores are compression-risk proxies, not labels."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpanSelection:
    keep: list[list[int]]
    archive: list[list[int]]
    spans: list[list[tuple[int, bool]]]
    positions: list[list[int]]


def _cut_new_spans(new_pos, ends, carry, last_pos, max_span):
    """Cut a flush into spans exactly as a token-by-token scan would.

    A span restarts after a boundary token or at a position gap; a run is split
    every max_span tokens, the first piece also counting a continued carry of
    `carry` unfinished tokens. Only runs are visited in Python, never tokens.
    Returns piece starts, stops, closed flags and whether piece 0 continues.
    """
    n = len(new_pos)
    if not n:
        return [], [], [], False
    pos = np.asarray(new_pos, dtype=np.int64)
    ends = np.asarray(ends, dtype=bool)
    continues = bool(carry) and int(pos[0]) == last_pos + 1
    breaks = (np.flatnonzero(ends[:-1] | (pos[1:] != pos[:-1] + 1)) + 1).tolist()
    starts, stops = [], []
    for k, (a, b) in enumerate(zip([0] + breaks, breaks + [n])):
        step = max_span - carry if k == 0 and continues else max_span
        while a < b:
            starts.append(a)
            stops.append(min(b, a + step))
            a, step = stops[-1], max_span
    closed = [True] * len(starts)
    # Only the final piece can stay open: no boundary on the last token and
    # below the cap (counting the carry if it is also the first piece).
    final = stops[-1] - starts[-1] + (carry if continues and len(starts) == 1 else 0)
    closed[-1] = bool(ends[-1]) or final == max_span
    return starts, stops, closed, continues


def select_spans(k, v, old_spans, old_positions, new_positions, ends, old_width, budget, max_span):
    """Indices refer to [padded old pool | current flush], shared across KV groups.

    A final unfinished span is mandatory until a boundary or the length cap.
    Scoring makes one batched GPU pass and one small score transfer per flush.
    """
    candidates, bi, ti, lengths = [], [], [], []
    for b, (spans, old_pos, new_pos, boundaries) in enumerate(zip(old_spans, old_positions, new_positions, ends)):
        row, offset, pending = [], 0, None
        for length, closed in spans:
            indices = np.arange(offset, offset + length, dtype=np.int64)
            offset += length
            if closed:
                row.append((indices, True, True))
            else:
                pending = indices
        all_pos = old_pos + [0] * (old_width - len(old_pos)) + list(new_pos)
        carry = 0 if pending is None else len(pending)
        starts, stops, closed, continues = _cut_new_spans(
            new_pos, boundaries, carry, all_pos[pending[-1]] if carry else 0, max_span)
        if carry and not continues and starts:
            row.append((pending, False, True))  # A position gap closes the carried span.
        for piece, (start, stop, done) in enumerate(zip(starts, stops, closed)):
            indices = np.arange(old_width + start, old_width + stop, dtype=np.int64)
            if piece == 0 and continues:
                indices = np.concatenate((pending, indices))
            row.append((indices, False, done))
        if carry and not starts:
            row.append((pending, False, False))
        candidates.append((row, all_pos))
        if row:
            sizes = [len(indices) for indices, _, _ in row]
            bi.append(np.full(sum(sizes), b, dtype=np.int64))
            ti.append(np.concatenate([indices for indices, _, _ in row]))
            lengths.extend(sizes)
    if not lengths:
        return SpanSelection([[] for _ in ends], [[] for _ in ends], [[] for _ in ends], [[] for _ in ends])
    # One upload for token coordinates and span sizes. Pinned + non_blocking:
    # a pageable copy would first wait for every queued kernel.
    host = torch.from_numpy(np.concatenate((*bi, *ti, np.asarray(lengths, dtype=np.int64))))
    packed = host.pin_memory().to(k.device, non_blocking=True) if k.is_cuda else host.to(k.device)
    tokens = (host.numel() - len(lengths)) // 2
    index, sizes = packed[:2 * tokens].view(2, tokens), packed[2 * tokens:]
    scores = None
    for values in (k, v):
        x = values[index[0], :, index[1]].detach().float()
        means = torch.segment_reduce(x, "mean", lengths=sizes, unsafe=True)
        energy = torch.segment_reduce(x.square().mean(-1), "mean", lengths=sizes, unsafe=True)
        residual = (energy - means.square().mean(-1)).clamp_min(0)
        part = (residual / energy.clamp_min(1e-12)).mean(-1)
        scores = part if scores is None else scores + part
    host_scores = scores.cpu().tolist()
    keep, archive, output_spans, output_positions = [], [], [], []
    offset = 0
    for row, positions in candidates:
        mandatory = {i for i, (_, _, closed) in enumerate(row) if not closed}
        room = budget - sum(len(row[i][0]) for i in mandatory)
        density = [host_scores[offset + i] * (1.1 if old else 1.) for i, (_, old, _) in enumerate(row)]
        chosen = mandatory | {i for i, (_, old, _) in enumerate(row) if old}
        room -= sum(len(row[i][0]) for i in chosen - mandatory)
        victims = lambda: sorted(chosen - mandatory, key=lambda i: (density[i], -i))
        # Make room for the bounded unfinished tail first.
        for i in victims():
            if room >= 0:
                break
            chosen.remove(i)
            room += len(row[i][0])
        # ponytail: bounded whole-span swaps, not an exact knapsack solver.
        # A short new span must pay for EVERY token in the whole spans evicted.
        ranked = sorted((i for i in range(len(row)) if i not in chosen), key=lambda i: (-density[i], i))
        for i in ranked:
            length = len(row[i][0])
            available, loss, remove = room, 0., []
            if available < length:
                for j in victims():
                    remove.append(j)
                    available += len(row[j][0])
                    loss += density[j] * len(row[j][0])
                    if available >= length:
                        break
            if available >= length and (not remove or density[i] * length > loss):
                chosen.difference_update(remove)
                chosen.add(i)
                room = available - length
        selected = [i for i in range(len(row)) if i in chosen]
        rejected = [i for i in range(len(row)) if i not in chosen]
        keep.append(np.concatenate([row[i][0] for i in selected]).tolist() if selected else [])
        archive.append(np.concatenate([row[i][0] for i in rejected]).tolist() if rejected else [])
        output_spans.append([(len(row[i][0]), row[i][2]) for i in selected])
        output_positions.append([positions[i] for i in keep[-1]])
        offset += len(row)
    return SpanSelection(keep, archive, output_spans, output_positions)
