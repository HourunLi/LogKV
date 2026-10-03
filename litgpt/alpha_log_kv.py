"""Bounded whole-span selection. Scores are compression-risk proxies, not labels."""
from dataclasses import dataclass

import torch


@dataclass
class SpanSelection:
    keep: list[list[int]]
    archive: list[list[int]]
    spans: list[list[tuple[int, bool]]]
    positions: list[list[int]]


def select_spans(k, v, old_spans, old_positions, new_positions, ends, old_width, budget, max_span,
                 *, beta_novelty=False, centroids=None, centroid_valid=None):
    """Indices refer to [padded old pool | current flush], shared across KV groups.

    A final unfinished span is mandatory until a boundary or the length cap.
    Scoring makes one batched GPU pass and one small score transfer per flush.
    Beta scores novelty against the old centroids captured before this flush.
    """
    if beta_novelty and (centroids is None or centroid_valid is None):
        raise ValueError("Beta span scoring requires centroids and centroid_valid from before the flush")
    candidates, bi, ti, lengths = [], [], [], []
    span_bi, span_i = [], []
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
        for j, (indices, _, _) in enumerate(row):
            bi.extend([b] * len(indices))
            ti.extend(indices)
            lengths.append(len(indices))
            if beta_novelty:
                span_bi.append(b)
                span_i.append(j)
    if not lengths:
        return SpanSelection([[] for _ in ends], [[] for _ in ends], [[] for _ in ends], [[] for _ in ends])
    # One metadata upload instead of three blocking pageable H2D copies.
    metadata = torch.tensor(bi + ti + lengths + span_bi + span_i, device=k.device, dtype=torch.long)
    index = metadata[:2 * len(bi)].view(2, -1)
    sizes = metadata[2 * len(bi):2 * len(bi) + len(lengths)]
    scores = None
    for component, values in enumerate((k, v)):
        x = values[index[0], :, index[1]].detach().float()
        means = torch.segment_reduce(x, "mean", lengths=sizes, unsafe=True)
        energy = torch.segment_reduce(x.square().mean(-1), "mean", lengths=sizes, unsafe=True)
        mean_square = means.square().mean(-1)
        residual = (energy - mean_square).clamp_min(0)
        part = residual / energy.clamp_min(1e-12)
        if beta_novelty and component == 0:
            key_means, key_energy, key_mean_square = means, energy, mean_square
        if not beta_novelty:
            part = part.mean(-1)
        scores = part if scores is None else scores + part
    if beta_novelty:
        span_index = metadata[2 * len(bi) + len(lengths):].view(2, -1)
        # Batch the ragged spans instead of repeating all 12 D-dimensional
        # centers per span; the distance workspace is only span x cluster.
        padded = key_means.new_zeros(k.size(0), max(len(row) for row, _ in candidates), k.size(1), k.size(-1))
        padded[span_index[0], span_index[1]] = key_means
        centers = centroids.detach().float()
        dots = (padded.transpose(1, 2) @ centers.transpose(-1, -2)).transpose(1, 2)
        dots = dots[span_index[0], span_index[1]] / k.size(-1)
        distances = (key_mean_square[..., None]
                     + centers.square().mean(-1)[span_index[0]] - 2 * dots).clamp_min_(0)
        valid = centroid_valid[span_index[0]]
        nearest = distances.masked_fill(~valid, torch.inf).amin(-1)
        nearest = torch.where(centroid_valid.any(-1)[span_index[0]], nearest, 0.)
        novelty = nearest / key_energy.clamp_min(1e-12)
        scores = scores * .5 + novelty / (1 + novelty)
        scores = scores.topk(min(2, k.size(1)), dim=-1).values.mean(-1)
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
        victim_order = None
        for i in ranked:
            length = len(row[i][0])
            available, loss, remove = room, 0., []
            if available < length:
                # Failed replacements do not change the ordering. Sorting it
                # again for every rejected candidate dominated short spans.
                if victim_order is None:
                    victim_order = victims()
                for j in victim_order:
                    remove.append(j)
                    available += len(row[j][0])
                    loss += density[j] * len(row[j][0])
                    if available >= length:
                        break
            if available >= length and (not remove or density[i] * length > loss):
                chosen.difference_update(remove)
                chosen.add(i)
                room = available - length
                victim_order = None
        selected = [i for i in range(len(row)) if i in chosen]
        keep.append([token for i in selected for token in row[i][0]])
        archive.append([token for i in range(len(row)) if i not in chosen for token in row[i][0]])
        output_spans.append([(len(row[i][0]), row[i][2]) for i in selected])
        output_positions.append([positions[i] for i in keep[-1]])
        offset += len(row)
    return SpanSelection(keep, archive, output_spans, output_positions)
