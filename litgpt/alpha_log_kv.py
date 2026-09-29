"""Bounded whole-span selection. Scores are compression-risk proxies, not labels."""
from dataclasses import dataclass

import torch


@dataclass
class SpanSelection:
    keep: list[list[int]]
    archive: list[list[int]]
    spans: list[list[tuple[int, bool]]]
    positions: list[list[int]]


def select_spans(k, v, old_spans, old_positions, new_positions, ends, old_width, budget, max_span):
    """Indices refer to [padded old pool | current flush], shared across KV groups.

    A final unfinished span is mandatory until a boundary or the length cap.
    Scoring makes one batched GPU pass and one small score transfer per flush.
    """
    candidates, bi, ti, lengths = [], [], [], []
    for b, (spans, old_pos, new_pos, boundaries) in enumerate(zip(old_spans, old_positions, new_positions, ends)):
        row, offset, pending = [], 0, []
        for length, closed in spans:
            indices = list(range(offset, offset + length))
            offset += length
            if closed:
                row.append((indices, True, True))
            else:
                pending = indices
        all_pos = old_pos + [0] * (old_width - len(old_pos)) + list(new_pos)
        for j, boundary in enumerate(boundaries):
            i = old_width + j
            if pending and all_pos[i] != all_pos[pending[-1]] + 1:
                row.append((pending, False, True))
                pending = []
            pending.append(i)
            if boundary or len(pending) == max_span:
                row.append((pending, False, True))
                pending = []
        if pending:
            row.append((pending, False, False))
        candidates.append((row, all_pos))
        for indices, _, _ in row:
            bi.extend([b] * len(indices))
            ti.extend(indices)
            lengths.append(len(indices))
    if not lengths:
        return SpanSelection([[] for _ in ends], [[] for _ in ends], [[] for _ in ends], [[] for _ in ends])
    index = torch.tensor([bi, ti], device=k.device, dtype=torch.long)
    sizes = torch.tensor(lengths, device=k.device, dtype=torch.long)
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
        keep.append([token for i in selected for token in row[i][0]])
        archive.append([token for i in range(len(row)) if i not in chosen for token in row[i][0]])
        output_spans.append([(len(row[i][0]), row[i][2]) for i in selected])
        output_positions.append([positions[i] for i in keep[-1]])
        offset += len(row)
    return SpanSelection(keep, archive, output_spans, output_positions)
