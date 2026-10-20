"""Bounded whole-span selection. Scores are compression-risk proxies, not labels;
Gamma's optional cue evidence adds a lexical relevance proxy, and its optional
literal evidence a lexical incompressibility proxy."""
import math
import re
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
# Gamma cue guard (optional): the opening region must end at a line end, and a
# cue counted more than CUE_MAX_COUNT times before the decision gives no
# evidence. A word that points at needles recurs about as often as there are
# needles; the topic words of an ordinary opening recur dozens of times.
CUE_MAX_COUNT = 16
# Query-aware cue (optional): when the whole prompt is known before prefill, the
# tokens of its closing question are cues too. The query region is the last
# CUE_QUERY_TOKENS prompt tokens, starting after the first line end among them.
CUE_QUERY_TOKENS = 64
# Gamma literals: a literal's symbols are its ASCII digits and the runs of at
# most 3 letters between two digits (hex and code fragments). Their adaptive
# order-0 code length measures how random they are: a random 7-digit string
# codes to about 18 nats, a year or 1,000,000 to about 9-10. The first
# occurrence of a literal coding above LITERAL_FLOOR nats gains value as a cue
# does, from the same exp form and cap.
LITERAL_FLOOR = 12.
# With the query cue on, literal values sit one tier below cue values: the code
# is shifted so the literal floor meets CUE_FLOOR and capped at LITERAL_TIER_CAP,
# so a span with more cue evidence than that outranks every literal, while a
# random 7-digit number (at the cap) still ranks above K/V scores.
LITERAL_TIER_CAP = 6.
_LITERAL_RUNS = re.compile(r"[0-9]|[A-Za-z]+")


@dataclass
class SpanSelection:
    keep: list[list[int]]
    archive: list[list[int]]
    spans: list[list[tuple[int, bool]]]
    positions: list[list[int]]
    archive_positions: list[np.ndarray]  # Positions of `archive`, row by row.


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
    """Candidate whole spans of one flush; host NumPy data only.

    Per row, candidates are the pool's closed spans, then the pending pool
    span and the new runs; the pending span is a candidate of its own when a
    position jump closed it, otherwise it heads the first new run.
    """
    positions: list  # Per row: positions of [padded old pool | flush].
    token_rows: list  # Per row: flat token indices of the spans, in span order.
    length_rows: list  # Per row: span lengths.
    old_rows: list  # Per row: closed spans already in the pool (they get the 1.1x bonus).
    closed_rows: list  # Per row: False only for the final unfinished span.


def _empty_selection(rows):
    return SpanSelection([[] for _ in range(rows)], [[] for _ in range(rows)],
                         [[] for _ in range(rows)], [[] for _ in range(rows)],
                         [np.empty(0, dtype=np.int64) for _ in range(rows)])


_NO_IDS = np.empty(0, dtype=np.int64)


class _CueRow:
    __slots__ = ("seen", "head_ids", "head_ends", "cues", "query", "region", "pos", "slot", "chunks", "weights")

    def __init__(self, query=_NO_IDS):
        self.seen = 0
        self.head_ids, self.head_ends = [], []
        self.cues, self.region = None, 0
        self.query = query  # sorted, unique question ids of this request (query-aware cue)
        self.pos = self.slot = _NO_IDS
        self.chunks = []
        self.weights = {}  # stop -> per-cue weight; every layer archives the same blocks

    def add(self, ids, ends, tokens, guard=False):
        start, self.seen = self.seen, self.seen + len(ids)
        if self.cues is not None:
            self.record(ids, start)
            return
        if not tokens:  # No opening region: the question alone, from position 0.
            self.cues = self.query
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
        # The region ends at the last line end in the head, else the last span end;
        # with the guard, a head without a line end adds no cues of its own.
        closed = np.flatnonzero(kinds == 2)
        if not len(closed) and guard:
            self.region, region_cues = tokens, _NO_IDS
        else:
            if not len(closed):
                closed = np.flatnonzero(kinds)
            self.region = int(closed[-1]) + 1 if len(closed) else tokens
            region_cues = head[:self.region]
        self.cues = np.union1d(region_cues, self.query)  # sorted and unique: each cue counts once
        self.record(head, 0)
        self.record(ids[take:], tokens)

    def record(self, ids, start):
        if not len(ids) or not len(self.cues):
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

    With ``guard``, a head without a line end has no cues (an instruction
    ends its line; an ordinary opening of collapsed text does not), and a cue
    counted more than CUE_MAX_COUNT times gives no evidence.

    Query-aware cue: ``set_queries`` gives each row the token ids of its
    request's question (``query_region``), consumed when the row restarts at
    position 0; they join the row's cues. ``tokens=0`` keeps the question only.
    """

    def __init__(self, tokens: int, guard: bool = False) -> None:
        if tokens < 0:
            raise ValueError("cue tokens must be >= 0")
        self.tokens = int(tokens)
        self.guard = bool(guard)
        self._rows: list[_CueRow] = []
        self._queries: dict[int, np.ndarray] = {}  # row -> question ids for its next request

    def set_queries(self, queries) -> None:
        """Question token ids per row (None for none) for each row's next request."""
        for b, ids in enumerate(queries):
            if ids is None:
                self._queries.pop(b, None)
            else:
                self._queries[b] = np.unique(np.asarray(ids, dtype=np.int64))

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
                self._rows[b] = _CueRow(self._queries.pop(b, _NO_IDS))
            elif start != self._rows[b].seen:
                raise ValueError(f"cue stream row {b} expects position {self._rows[b].seen}, got {start}")
            self._rows[b].add(row_ids, row_ends, self.tokens, self.guard)

    def span_values(self, row, positions, lengths, stop):
        """``max(exp(evidence) - exp(CUE_FLOOR), 0) / length`` per span, as of ``stop``.

        Evidence sums, over the distinct cues a span holds outside the opening
        region, how many nats rarer than 1% of the first ``stop`` positions (plus
        CUE_PRIOR) each one is. The floor leaves spans with weak evidence at
        their K/V score. Before ``stop`` covers the opening region, every value
        is 0: the region must not depend on tokens after ``stop``. With the
        guard, a cue counted more than CUE_MAX_COUNT times weighs 0.
        """
        lengths = np.asarray(lengths, dtype=np.int64)
        values = np.zeros(len(lengths))
        state = self._rows[row] if row < len(self._rows) else None
        if state is None or state.cues is None or not len(state.cues) or stop < self.tokens or not len(lengths):
            return values
        pos, slot = state.occurrences()
        if not len(pos):
            return values
        n = len(state.cues)
        weight = state.weights.get(stop)
        if weight is None:
            counts = np.bincount(slot[:np.searchsorted(pos, stop)], minlength=n)
            weight = np.maximum(np.log((stop + 1 + CUE_PRIOR) / (counts + 1)) - CUE_TAU, 0.)
            if self.guard:
                weight[counts > CUE_MAX_COUNT] = 0.
            if stop <= state.seen:  # Later tokens can no longer change these counts.
                state.weights[stop] = weight
        positions = np.asarray(positions, dtype=np.int64)
        at = np.minimum(np.searchsorted(pos, positions), len(pos) - 1)
        hit = (pos[at] == positions) & (positions >= state.region)
        pairs = np.unique(np.repeat(np.arange(len(lengths)), lengths)[hit] * n + slot[at[hit]])
        evidence = np.bincount(pairs // n, weights=weight[pairs % n], minlength=len(lengths))
        value = np.exp(np.minimum(evidence, CUE_CAP)) - math.exp(CUE_FLOOR)
        return np.maximum(value, 0.) / np.maximum(lengths, 1)


def query_region(ids, line_end, tokens=CUE_QUERY_TOKENS):
    """Token ids of a prompt's closing question, for the query-aware cue.

    The last ``tokens`` ids, starting after the first line end among them (all
    of them when there is none, or when only the last token ends a line).
    ``line_end`` is a boolean table over the vocabulary. Host only, O(tokens).
    """
    tail = np.asarray(ids, dtype=np.int64)[-tokens:]
    inside = tail < len(line_end)
    ends = np.flatnonzero(inside & line_end[np.where(inside, tail, 0)])
    if len(ends) and ends[0] + 1 < len(tail):
        tail = tail[ends[0] + 1:]
    return tail


def literal_code(text):
    """(adaptive order-0 code length in nats, symbols) of a literal's text.

    Symbols are the ASCII digits and the runs of at most 3 letters that sit
    between two digits; other letters belong to ordinary words. The code uses a
    0.5 prior over 10 symbols, or 36 once a letter is a symbol, so repeated
    digits code short and random ones long. Text without a digit codes to 0.
    """
    runs = _LITERAL_RUNS.findall(text)
    symbols = []
    for i, run in enumerate(runs):
        if run.isdigit():
            symbols.append(run)
        elif len(run) <= 3 and 0 < i < len(runs) - 1 and runs[i - 1].isdigit() and runs[i + 1].isdigit():
            symbols.extend(run.lower())
    alphabet = 10 if all(c.isdigit() for c in symbols) else 36
    counts, code = {}, 0.
    for i, c in enumerate(symbols):
        code -= math.log((counts.get(c, 0) + .5) / (i + .5 * alphabet))
        counts[c] = counts.get(c, 0) + 1
    return code, "".join(symbols)


class _LiteralRow:
    __slots__ = ("seen", "prev_lit", "lo", "hi", "code", "ids", "first", "open_lo", "open_ids", "arrays", "at_stop")

    def __init__(self):
        self.seen, self.prev_lit = 0, False
        # Closed runs holding a digit, in stream order; a repeat codes to 0.
        self.lo, self.hi, self.code, self.ids = [], [], [], []
        self.first = {}  # symbols -> start of their first closed occurrence
        self.open_lo, self.open_ids = 0, []  # the last run, which the next token may extend
        self.arrays = None
        # stop -> (n, last, open) from _cut: every layer reads the same stops, and in
        # whole-prompt prefill all of them come after a single ingest. O(1) per stop.
        self.at_stop = {}


_LITERAL_TABLES: dict = {}  # id(texts) -> (texts, lit, digit, lead): one decode per vocabulary


def _literal_tables(texts, chars):
    hit = _LITERAL_TABLES.get(id(texts))
    if hit is not None and hit[0] is texts:
        return hit[1:]
    n = len(texts)
    bodies = [t.lstrip() for t in texts]
    lit = np.fromiter((bool(t) and chars.issuperset(t) for t in bodies), dtype=bool, count=n)
    lead = np.fromiter((t[:1].isspace() for t in texts), dtype=bool, count=n)
    digit = lit & np.fromiter((any("0" <= c <= "9" for c in t) for t in bodies), dtype=bool, count=n)
    _LITERAL_TABLES.clear()  # One tokenizer per process; drop tables of a replaced vocabulary.
    _LITERAL_TABLES[id(texts)] = (texts, lit, digit, lead)
    return lit, digit, lead


class LiteralStream:
    """First occurrences of random-looking literals; host only, shared by all layers.

    A literal is a run of tokens made of ASCII letters, digits and ``-_.,:/+#``
    with no whitespace inside: it ends at any other token and restarts at a
    token led by whitespace, so numbers inside CJK text or prose stand alone.
    Only runs holding a digit are kept, with their token ids, so state grows
    with the literals in the stream. Values are read as of a position (the
    cache passes the end of the block it archives): a run that continues past
    it counts with its tokens before it, so layer-major prefill, any prefill
    block and token-by-token decode select alike.

    ``below_cue`` (query-aware cue on) puts literal values one tier below cue
    values (LITERAL_TIER_CAP).
    """

    CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.,:/+#")

    def __init__(self, texts, below_cue: bool = False) -> None:
        self.texts = texts if isinstance(texts, tuple) else tuple(texts)
        if not self.texts:
            raise ValueError("LiteralStream needs the tokenizer's token texts")
        self.lit, self.digit, self.lead = _literal_tables(self.texts, self.CHARS)
        self.below_cue = bool(below_cue)
        self._rows: list[_LiteralRow] = []

    def _tables(self, ids):
        inside = ids < len(self.texts)
        at = np.where(inside, ids, 0)
        return inside & self.lit[at], inside & self.digit[at], ~inside | self.lead[at]

    def ingest(self, ids, starts) -> None:
        """Append one forward's tokens per row; a row restarts at position 0."""
        ids = np.asarray(ids, dtype=np.int64)
        while len(self._rows) < len(ids):
            self._rows.append(_LiteralRow())
        for b, (row_ids, start) in enumerate(zip(ids, starts)):
            if start == 0:
                self._rows[b] = _LiteralRow()
            elif start != self._rows[b].seen:
                raise ValueError(f"literal stream row {b} expects position {self._rows[b].seen}, got {start}")
            self._add(self._rows[b], row_ids)

    def _add(self, state, ids):
        if not len(ids):
            return
        start, state.seen = state.seen, state.seen + len(ids)
        lit, digit, lead = self._tables(ids)
        begin = lit & (lead | ~np.concatenate(([state.prev_lit], lit[:-1])))
        state.prev_lit = bool(lit[-1])
        # The open run continues up to the first token that is not a literal or starts one.
        stops = np.flatnonzero(~lit | begin)
        head = int(stops[0]) if len(stops) else len(ids)
        state.open_ids.extend(ids[:head].tolist())
        if not len(stops):
            return
        self._close(state, state.open_lo, state.open_ids)
        state.open_ids = []
        firsts = np.flatnonzero(begin)
        if not len(firsts):
            return
        # Non-literal tokens between runs hold no digit, so these are digits per run.
        held = np.add.reduceat(digit.astype(np.int64), firsts)
        ends = stops[np.minimum(np.searchsorted(stops, firsts, side="right"), len(stops) - 1)]
        closed = ends > firsts
        for j in np.flatnonzero((held > 0) & closed):
            self._close(state, start + int(firsts[j]), ids[firsts[j]:ends[j]].tolist())
        if not closed[-1]:  # The last run reaches the end of this forward.
            state.open_lo, state.open_ids = start + int(firsts[-1]), ids[firsts[-1]:].tolist()

    def _code(self, state, lo, run):
        """Code of a run, 0 unless its symbols occur here for the first time."""
        if not run or not self._tables(np.asarray(run, dtype=np.int64))[1].any():
            return None
        code, symbols = literal_code("".join(self.texts[i] for i in run))
        return code if state.first.get(symbols, lo) >= lo else 0., symbols

    def _close(self, state, lo, run):
        coded = self._code(state, lo, run)
        if coded is None:
            return
        code, symbols = coded
        state.first.setdefault(symbols, lo)
        state.lo.append(lo)
        state.hi.append(lo + len(run))
        state.code.append(code)
        state.ids.append(run)
        state.arrays = None
        state.at_stop.clear()

    def span_values(self, row, positions, lengths, stop):
        """``max(exp(code) - exp(LITERAL_FLOOR), 0) / length`` per span, as of ``stop``.

        A span's code is the largest among the literals it touches that occur
        there for the first time; a repeat of an earlier literal codes to 0.
        Below the cue: ``max(exp(min(code - LITERAL_FLOOR + CUE_FLOOR,
        LITERAL_TIER_CAP)) - exp(CUE_FLOOR), 0) / length``, the same floor.
        """
        lengths = np.asarray(lengths, dtype=np.int64)
        values = np.zeros(len(lengths))
        state = self._rows[row] if row < len(self._rows) else None
        if state is None or not len(lengths) or not len(positions):
            return values
        cut = state.at_stop.get(stop)
        if cut is None:
            cut = self._cut(state, stop)
            if stop <= state.seen:  # Later tokens can no longer change this cut.
                state.at_stop[stop] = cut
        n, last, opened = cut
        if not n and opened is None:
            return values
        positions = np.asarray(positions, dtype=np.int64)
        if n:
            lo, hi, code = state.arrays
            at = np.searchsorted(lo[:n], positions, side="right") - 1
            inside = at >= 0
            at = np.maximum(at, 0)
            code_at = code[at]  # a copy
            if last is not None:
                code_at[at == n - 1] = last
            token_code = np.where(inside & (positions < np.minimum(hi[at], stop)), code_at, 0.)
        else:
            token_code = np.zeros(len(positions))
        if opened is not None:
            token_code[(positions >= state.open_lo) & (positions < stop)] = opened
        best = np.maximum.reduceat(token_code, np.cumsum(lengths) - lengths)
        if self.below_cue:
            value = np.exp(np.minimum(best - (LITERAL_FLOOR - CUE_FLOOR), LITERAL_TIER_CAP)) - math.exp(CUE_FLOOR)
        else:
            value = np.exp(np.minimum(best, CUE_CAP)) - math.exp(LITERAL_FLOOR)
        return np.maximum(value, 0.) / np.maximum(lengths, 1)

    def _cut(self, state, stop):
        """The literal runs as of ``stop``: a run past it counts its tokens before it.

        Returns ``(n, last, open)``: closed runs ``[0, n)`` start before ``stop``;
        ``last`` recodes run ``n - 1`` when it continues past ``stop`` (else
        None); ``open`` codes the open run's tokens before ``stop`` (None when
        it has none or no digit). Span values read the closed runs' arrays in
        place, so a cut never copies them.
        """
        if state.arrays is None:
            state.arrays = (np.asarray(state.lo, dtype=np.int64), np.asarray(state.hi, dtype=np.int64),
                            np.asarray(state.code, dtype=np.float64))
        lo, hi, _ = state.arrays
        n = int(np.searchsorted(lo, stop))  # Runs that start before stop.
        last = opened = None
        if n and hi[n - 1] > stop:  # A closed run continues past stop.
            coded = self._code(state, int(lo[n - 1]), state.ids[n - 1][:stop - int(lo[n - 1])])
            last = 0. if coded is None else coded[0]
        if state.open_ids and state.open_lo < stop:  # So does the open run.
            coded = self._code(state, state.open_lo, state.open_ids[:stop - state.open_lo])
            if coded is not None:
                opened = coded[0]
        return n, last, opened


class SummedValues:
    """Span values of several Gamma streams added together (cue and literal)."""

    def __init__(self, streams) -> None:
        self.streams = tuple(streams)

    def span_values(self, row, positions, lengths, stop):
        return sum(stream.span_values(row, positions, lengths, stop) for stream in self.streams)


def add_cue_values(cut, host_scores, stream, stops):
    """Host scores plus each span's Gamma value; ``stops[b]`` ends row b's archived block.

    ``stream`` is a CueStream, a LiteralStream or their SummedValues.
    """
    values = [stream.span_values(b, positions[tokens], lengths, stops[b])
              for b, (positions, tokens, lengths) in enumerate(zip(cut.positions, cut.token_rows, cut.length_rows))]
    return (np.asarray(host_scores, dtype=np.float64) + np.concatenate(values)).tolist() if values else list(host_scores)


def select_spans(k, v, old_spans, old_positions, new_positions, ends, old_width, budget, max_span,
                 *, beta_novelty=False, centroids=None, centroid_valid=None, cues=None):
    """Indices refer to [padded old pool | current flush], shared across KV groups.

    A final unfinished span is mandatory until a boundary or the length cap.
    Scoring makes one batched GPU pass and one small score transfer per flush.
    Beta scores novelty against the old centroids captured before this flush.
    ``cues`` is an optional ``(stream, stops)`` pair (Gamma cues and/or literals).
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
    positions, token_rows, length_rows, old_rows, closed_rows = [], [], [], [], []
    for spans, old_pos, new_pos, boundaries in zip(old_spans, old_positions, new_positions, ends):
        # Pool spans are (length, closed) and tile [0, pool length); the last
        # unfinished one is pending.
        pool = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
        sizes, done = pool[:, 0], pool[:, 1] != 0
        new_pos = np.asarray(new_pos, dtype=np.int64)  # Host rows are lists: convert once.
        all_pos = np.zeros(old_width + len(new_pos), dtype=np.int64)
        all_pos[:len(old_pos)] = old_pos
        all_pos[old_width:] = new_pos
        open_spans = np.flatnonzero(~done)
        pending_start = pending_len = 0
        if len(open_spans):
            last = int(open_spans[-1])
            pending_start, pending_len = int(sizes[:last].sum()), int(sizes[last])
        pending_closed, starts, stops, closed = _new_runs(
            new_pos, boundaries, pending_len, int(all_pos[pending_start + pending_len - 1]) if pending_len else 0,
            max_span)
        runs = stops - starts
        if pending_closed:
            runs, closed = np.concatenate(([pending_len], runs)), np.concatenate(([True], closed))
        elif pending_len and len(runs):
            runs[0] += pending_len
        elif pending_len:
            runs, closed = np.array([pending_len]), np.array([False])
        # Every token appears once: closed pool spans, the pending span, the flush.
        tokens = np.concatenate((np.flatnonzero(np.repeat(done, sizes)),
                                 np.arange(pending_start, pending_start + pending_len),
                                 np.arange(old_width, old_width + len(new_pos))))
        n_old = int(done.sum())
        positions.append(all_pos)
        token_rows.append(tokens)
        length_rows.append(np.concatenate((sizes[done], runs)))
        old_rows.append(np.arange(n_old + len(runs)) < n_old)
        closed_rows.append(np.concatenate((np.ones(n_old, dtype=bool), closed)))
    if not any(len(lengths) for lengths in length_rows):
        return None
    return SpanCut(positions, token_rows, length_rows, old_rows, closed_rows)


def score_spans(k, v, cut, *, beta_novelty=False, centroids=None, centroid_valid=None):
    """Device tensor of span scores in candidate order; nothing is read back.

    CUDA (and the Triton interpreter) scores every span in one fused kernel;
    the PyTorch form below is the reference it follows up to rounding.
    """
    from litgpt.log_kv_cache import _fused_on, _triton_alpha, _upload

    if beta_novelty and (centroids is None or centroid_valid is None):
        raise ValueError("Beta span scoring requires centroids and centroid_valid from before the flush")
    token_rows, length_rows = cut.token_rows, cut.length_rows
    sizes = [len(lengths) for lengths in length_rows]
    tokens = np.concatenate(token_rows)
    lengths = np.concatenate([np.asarray(row, dtype=np.int64) for row in length_rows])
    fused = _triton_alpha() if _fused_on(k) else None
    if fused is not None:
        # One upload: token indices, then each span's start, length and row.
        metadata = _upload(np.concatenate((tokens, np.cumsum(lengths) - lengths, lengths,
                                           np.repeat(np.arange(len(sizes)), sizes))), k.device)
        return fused.span_scores(k, v, metadata[:len(tokens)], metadata[len(tokens):].view(3, -1),
                                 centroids if beta_novelty else None, centroid_valid if beta_novelty else None)
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
    keep, archive, output_spans, output_positions, archive_positions = [], [], [], [], []
    offset = 0
    for positions, tokens, lengths, old_spans, closed_spans in zip(
            cut.positions, cut.token_rows, cut.length_rows, cut.old_rows, cut.closed_rows):
        n = len(lengths)
        size, old, closed = lengths.tolist(), old_spans.tolist(), closed_spans.tolist()
        mandatory = {i for i in range(n) if not closed[i]}
        room = budget - sum(size[i] for i in mandatory)
        density = [score * 1.1 if pooled else score for score, pooled in zip(host_scores[offset:offset + n], old)]
        chosen = mandatory | {i for i in range(n) if old[i]}
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
        ranked = sorted((i for i in range(n) if i not in chosen), key=lambda i: (-density[i], i))
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
        kept = np.zeros(n, dtype=bool)
        kept[selected] = True
        kept = np.repeat(kept, lengths)
        keep.append(tokens[kept].tolist())
        archive.append(tokens[~kept].tolist())
        output_spans.append([(size[i], closed[i]) for i in selected])
        output_positions.append(positions[tokens[kept]].tolist())
        archive_positions.append(positions[tokens[~kept]])
        offset += n
    return SpanSelection(keep, archive, output_spans, output_positions, archive_positions)
