"""Bounded whole-span selection. Scores are compression-risk proxies, not labels;
Gamma's optional cue evidence adds a lexical relevance proxy."""
import math
from dataclasses import dataclass

import numpy as np
import torch

# A cue seen in more than 1% of the stream so far carries no evidence, and a
# span's evidence counts only above CUE_FLOOR nats; the cap keeps exp finite.
# Frequencies count CUE_PRIOR extra tokens without the cue, so the first
# blocks of a stream are not judged on a few hundred tokens alone.
CUE_TAU = math.log(100.)
CUE_FLOOR = 4.
CUE_CAP = 30.
CUE_PRIOR = 2048


@dataclass
class SpanSelection:
    keep: list[list[int]]
    archive: list[list[int]]
    spans: list[list[tuple[int, bool]]]
    positions: list[list[int]]


def _new_runs(new_pos, ends, pending, pending_pos, max_span):
    """Whole-array form of the per-token span loop over one flush row.

    Returns (pending_closed, starts, stops, closed). A run [start, stop) of new
    tokens closes after a boundary or the length cap, or before a position
    jump. The first run continues the old unfinished span unless a jump
    closes that span first; the cap counts its old tokens too.
    """
    count = len(new_pos)
    empty = np.empty(0, dtype=np.int64)
    if not count:
        return False, empty, empty, np.empty(0, dtype=bool)
    pos = np.asarray(new_pos, dtype=np.int64)
    end = np.asarray(ends, dtype=bool)
    pending_closed = bool(pending) and int(pos[0]) != pending_pos + 1
    prefix = 0 if pending_closed else pending
    jump = np.zeros(count, dtype=bool)
    jump[1:] = pos[1:] != pos[:-1] + 1
    # Hard segments restart at jumps and after boundaries; the cap splits
    # each segment into max_span chunks, counted from its first token.
    first = np.ones(count, dtype=bool)
    first[1:] = jump[1:] | end[:-1]
    segment = np.cumsum(first) - 1
    index = np.arange(count, dtype=np.int64)
    filled = index - np.flatnonzero(first)[segment] + np.where(segment == 0, prefix, 0) + 1
    cap = filled % max_span == 0
    if prefix >= max_span:
        # An over-long old span never equals the cap again in the loop.
        cap &= segment != 0
    close = end | cap
    stop = close.copy()
    stop[:-1] |= jump[1:]
    stop[-1] = True
    stops = np.flatnonzero(stop) + 1
    starts = np.concatenate(([0], stops[:-1]))
    closed = close[stops - 1]
    closed[:-1] |= jump[stops[:-1]]
    return pending_closed, starts, stops, closed


@dataclass
class SpanCut:
    """Candidate whole spans of one flush; host data only."""
    candidates: list  # Per row: (spans as (token ranges, old, closed), positions of [pool | flush]).
    token_rows: list  # Per row: flat token indices of the spans, in span order.
    length_rows: list  # Per row: span lengths.


def _empty_selection(rows):
    return SpanSelection([[] for _ in range(rows)], [[] for _ in range(rows)],
                         [[] for _ in range(rows)], [[] for _ in range(rows)])


class _CueRow:
    __slots__ = ("seen", "head_ids", "head_ends", "cues", "region", "pos", "slot", "chunks")

    def __init__(self):
        self.seen = 0
        self.head_ids, self.head_ends = [], []
        self.cues, self.region = None, 0
        self.pos = self.slot = np.empty(0, dtype=np.int64)
        self.chunks = []

    def add(self, ids, ends, tokens):
        start, self.seen = self.seen, self.seen + len(ids)
        if self.cues is not None:
            self.record(ids, start)
            return
        # Until the head is full, every token so far is in it (start == len(head_ids)).
        take = tokens - start
        self.head_ids.extend(ids[:take].tolist())
        self.head_ends.extend(ends[:take].tolist())
        if len(self.head_ids) < tokens:
            return
        head = np.asarray(self.head_ids, dtype=np.int64)
        kinds = np.asarray(self.head_ends, dtype=np.int64)
        # The region ends at the last line end in the head, else the last span end.
        closed = np.flatnonzero(kinds == 2)
        if not len(closed):
            closed = np.flatnonzero(kinds)
        self.region = int(closed[-1]) + 1 if len(closed) else tokens
        self.cues = np.unique(head[:self.region])
        self.record(head, 0)
        self.record(ids[take:], tokens)

    def record(self, ids, start):
        if not len(ids):
            return
        slot = np.minimum(np.searchsorted(self.cues, ids), len(self.cues) - 1)
        hit = self.cues[slot] == ids
        if hit.any():
            self.chunks.append((np.flatnonzero(hit) + start, slot[hit]))

    def occurrences(self):
        if self.chunks:
            self.pos = np.concatenate([self.pos, *(pos for pos, _ in self.chunks)])
            self.slot = np.concatenate([self.slot, *(slot for _, slot in self.chunks)])
            self.chunks = []
        return self.pos, self.slot


class CueStream:
    """Where each row's opening tokens (cues) recur; host only, shared by all layers.

    A row's cues are the token ids of its opening region: the first
    ``tokens`` positions cut after their last line end, else after their last
    span end (whole when neither occurs). Evidence is read as of a position
    (the cache passes the end of the block it archives), so layer-major
    prefill, any prefill block and token-by-token decode select alike. State
    grows with the cue occurrences only.
    """

    def __init__(self, tokens: int) -> None:
        if tokens < 1:
            raise ValueError("cue tokens must be >= 1")
        self.tokens = int(tokens)
        self._rows: list[_CueRow] = []

    def ingest(self, ids, ends, starts) -> None:
        """Append one forward's tokens per row; a row restarts at position 0.

        ``ends`` per token: 0 inside a span, 1 at a span end, 2 at a line end.
        """
        ids = np.asarray(ids, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        while len(self._rows) < len(ids):
            self._rows.append(_CueRow())
        for b, (row_ids, row_ends, start) in enumerate(zip(ids, ends, starts)):
            if start == 0:
                self._rows[b] = _CueRow()
            elif start != self._rows[b].seen:
                raise ValueError(f"cue stream row {b} expects position {self._rows[b].seen}, got {start}")
            self._rows[b].add(row_ids, row_ends, self.tokens)

    def span_values(self, row, positions, lengths, stop):
        """``max(exp(evidence) - exp(CUE_FLOOR), 0) / length`` per span, as of ``stop``.

        Evidence sums, over the distinct cues a span holds outside the opening
        region, how many nats rarer than 1% of the first ``stop`` positions (plus
        CUE_PRIOR) each one is. The floor leaves spans with weak evidence at
        their K/V score. Before ``stop`` covers the opening region, every value
        is 0: the region must not depend on tokens after ``stop``.
        """
        lengths = np.asarray(lengths, dtype=np.int64)
        values = np.zeros(len(lengths))
        state = self._rows[row] if row < len(self._rows) else None
        if state is None or state.cues is None or stop < self.tokens or not len(lengths):
            return values
        pos, slot = state.occurrences()
        if not len(pos):
            return values
        n = len(state.cues)
        counts = np.bincount(slot[:np.searchsorted(pos, stop)], minlength=n)
        weight = np.maximum(np.log((stop + 1 + CUE_PRIOR) / (counts + 1)) - CUE_TAU, 0.)
        positions = np.asarray(positions, dtype=np.int64)
        at = np.minimum(np.searchsorted(pos, positions), len(pos) - 1)
        hit = (pos[at] == positions) & (positions >= state.region)
        pairs = np.unique(np.repeat(np.arange(len(lengths)), lengths)[hit] * n + slot[at[hit]])
        evidence = np.bincount(pairs // n, weights=weight[pairs % n], minlength=len(lengths))
        value = np.exp(np.minimum(evidence, CUE_CAP)) - math.exp(CUE_FLOOR)
        return np.maximum(value, 0.) / np.maximum(lengths, 1)


def add_cue_values(cut, host_scores, stream, stops):
    """Host scores plus each span's cue value; ``stops[b]`` ends row b's archived block."""
    scores, offset = list(host_scores), 0
    for b, ((_, positions), tokens, lengths) in enumerate(zip(cut.candidates, cut.token_rows, cut.length_rows)):
        if lengths:
            values = stream.span_values(b, positions[tokens], lengths, stops[b])
            for i, value in enumerate(values.tolist()):
                scores[offset + i] += value
        offset += len(lengths)
    return scores


def select_spans(k, v, old_spans, old_positions, new_positions, ends, old_width, budget, max_span,
                 *, beta_novelty=False, centroids=None, centroid_valid=None, cues=None):
    """Indices refer to [padded old pool | current flush], shared across KV groups.

    A final unfinished span is mandatory until a boundary or the length cap.
    Scoring makes one batched GPU pass and one small score transfer per flush.
    Beta scores novelty against the old centroids captured before this flush.
    ``cues`` is an optional ``(CueStream, stops)`` pair (Gamma).
    """
    cut = cut_spans(old_spans, old_positions, new_positions, ends, old_width, max_span)
    if cut is None:
        return _empty_selection(len(ends))
    scores = score_spans(k, v, cut, beta_novelty=beta_novelty, centroids=centroids, centroid_valid=centroid_valid)
    scores = scores.cpu().tolist()
    if cues is not None:
        scores = add_cue_values(cut, scores, *cues)
    return choose_spans(cut, scores, budget)


def cut_spans(old_spans, old_positions, new_positions, ends, old_width, max_span):
    """Whole candidate spans per row (host only); None when there are none."""
    from litgpt.log_kv_cache import _spans_to_index_array

    candidates, token_rows, length_rows = [], [], []
    for spans, old_pos, new_pos, boundaries in zip(old_spans, old_positions, new_positions, ends):
        # A span is (token ranges, old, closed); ranges are [start, stop) pairs.
        row, offset, pending = [], 0, None
        for length, closed in spans:
            if closed:
                row.append(([(offset, offset + length)], True, True))
            else:
                pending = (offset, offset + length)
            offset += length
        all_pos = np.zeros(old_width + len(new_pos), dtype=np.int64)
        all_pos[:len(old_pos)] = old_pos
        all_pos[old_width:] = new_pos
        pending_len = 0 if pending is None else pending[1] - pending[0]
        pending_closed, starts, stops, closed = _new_runs(
            new_pos, boundaries, pending_len, int(all_pos[pending[1] - 1]) if pending_len else 0, max_span)
        if pending_closed:
            row.append(([pending], False, True))
        bounds = (starts + old_width).tolist(), (stops + old_width).tolist(), closed.tolist()
        for i, (start, stop, done) in enumerate(zip(*bounds)):
            ranges = [(start, stop)]
            if not i and pending_len and not pending_closed:
                ranges.insert(0, pending)
            row.append((ranges, False, done))
        if pending_len and not pending_closed and not len(starts):
            row.append(([pending], False, False))
        ranges = [r for span, _, _ in row for r in span]
        tokens = _spans_to_index_array(ranges)
        lengths = [sum(stop - start for start, stop in span) for span, _, _ in row]
        candidates.append((row, all_pos))
        token_rows.append(tokens)
        length_rows.append(lengths)
    if not any(length_rows):
        return None
    return SpanCut(candidates, token_rows, length_rows)


def score_spans(k, v, cut, *, beta_novelty=False, centroids=None, centroid_valid=None):
    """Device tensor of span scores in candidate order; nothing is read back."""
    from litgpt.log_kv_cache import _upload

    if beta_novelty and (centroids is None or centroid_valid is None):
        raise ValueError("Beta span scoring requires centroids and centroid_valid from before the flush")
    token_rows, length_rows = cut.token_rows, cut.length_rows
    sizes = [len(lengths) for lengths in length_rows]
    tokens = np.concatenate(token_rows)
    lengths = np.concatenate([np.asarray(row, dtype=np.int64) for row in length_rows])
    token_batch = np.repeat(np.arange(len(token_rows)), [len(row) for row in token_rows])
    parts = [token_batch, tokens, lengths]
    if beta_novelty:
        parts += [np.repeat(np.arange(len(sizes)), sizes),
                  np.concatenate([np.arange(n, dtype=np.int64) for n in sizes])]
    # One asynchronous metadata upload for gathers, segments and span slots.
    metadata = _upload(np.concatenate(parts), k.device)
    index = metadata[:2 * len(tokens)].view(2, -1)
    span_sizes = metadata[2 * len(tokens):2 * len(tokens) + len(lengths)]
    scores = None
    for component, values in enumerate((k, v)):
        x = values[index[0], :, index[1]].detach().float()
        means = torch.segment_reduce(x, "mean", lengths=span_sizes, unsafe=True)
        energy = torch.segment_reduce(x.square().mean(-1), "mean", lengths=span_sizes, unsafe=True)
        mean_square = means.square().mean(-1)
        residual = (energy - mean_square).clamp_min(0)
        part = residual / energy.clamp_min(1e-12)
        if beta_novelty and component == 0:
            key_means, key_energy, key_mean_square = means, energy, mean_square
        if not beta_novelty:
            part = part.mean(-1)
        scores = part if scores is None else scores + part
    if beta_novelty:
        span_index = metadata[2 * len(tokens) + len(lengths):].view(2, -1)
        # Batch the ragged spans instead of repeating all 12 D-dimensional
        # centers per span; the distance workspace is only span x cluster.
        padded = key_means.new_zeros(k.size(0), max(sizes), k.size(1), k.size(-1))
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
    return scores


def choose_spans(cut, host_scores, budget):
    """Bounded whole-span greedy from host scores in candidate order."""
    keep, archive, output_spans, output_positions = [], [], [], []
    offset = 0
    for (row, positions), tokens, size in zip(cut.candidates, cut.token_rows, cut.length_rows):
        mandatory = {i for i, (_, _, closed) in enumerate(row) if not closed}
        room = budget - sum(size[i] for i in mandatory)
        density = [host_scores[offset + i] * (1.1 if old else 1.) for i, (_, old, _) in enumerate(row)]
        chosen = mandatory | {i for i, (_, old, _) in enumerate(row) if old}
        room -= sum(size[i] for i in chosen - mandatory)
        victims = lambda: sorted(chosen - mandatory, key=lambda i: (density[i], -i))
        # Make room for the bounded unfinished tail first.
        for i in victims():
            if room >= 0:
                break
            chosen.remove(i)
            room += size[i]
        # ponytail: bounded whole-span swaps, not an exact knapsack solver.
        # A short new span must pay for EVERY token in the whole spans evicted.
        ranked = sorted((i for i in range(len(row)) if i not in chosen), key=lambda i: (-density[i], i))
        victim_order = None
        for i in ranked:
            length = size[i]
            available, loss, remove = room, 0., []
            if available < length:
                # Failed replacements do not change the ordering. Sorting it
                # again for every rejected candidate dominated short spans.
                if victim_order is None:
                    victim_order = victims()
                for j in victim_order:
                    remove.append(j)
                    available += size[j]
                    loss += density[j] * size[j]
                    if available >= length:
                        break
            if available >= length and (not remove or density[i] * length > loss):
                chosen.difference_update(remove)
                chosen.add(i)
                room = available - length
                victim_order = None
        selected = sorted(chosen)
        kept = np.zeros(len(row), dtype=bool)
        kept[selected] = True
        kept = np.repeat(kept, size)
        keep.append(tokens[kept].tolist())
        archive.append(tokens[~kept].tolist())
        output_spans.append([(size[i], row[i][2]) for i in selected])
        output_positions.append(positions[tokens[kept]].tolist())
        offset += len(row)
    return SpanSelection(keep, archive, output_spans, output_positions)
