"""Training the learned retention heads on a frozen LogKV model (niahLogKV).

``HeadTrainer`` installs itself as ``litgpt.retention.RETENTION_TAP``: every
selection that scores with a head hands it the candidate spans' K/V, and the
trainer takes the spans archived for the first time (new in this flush, so
each token counts once per layer). A span is positive when it holds a token
of the block's fact mask (``set_labels``). The loss is binary cross-entropy
with positives weighted by the running negative/positive ratio; each call
backpropagates into the head alone (the K/V are detached), and ``step``
averages gradients across ranks and updates the heads. Selection keeps using
the heads being trained, so deeper layers see the K/V they will see at
inference.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import litgpt.retention as rt

MAX_POS_WEIGHT = 200.


class HeadTrainer:
    def __init__(self, heads: rt.RetentionHeads, lr: float = 1e-3, weight_decay: float = 0.01) -> None:
        self.heads = heads
        heads.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=weight_decay)
        self.labels: list[np.ndarray] | None = None
        self.train = True
        self.counts = np.array([1., 1.])  # running [negatives, positives], for the positive weight
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = dict(loss=0., spans=0, positives=0, pos_logit=0., neg_logit=0.)

    def set_labels(self, masks) -> None:
        """Per-row bool fact masks over positions for the next forward (None: ignore taps)."""
        self.labels = None if masks is None else [np.asarray(m, dtype=bool) for m in masks]

    def __enter__(self) -> "HeadTrainer":
        self._previous, rt.RETENTION_TAP = rt.RETENTION_TAP, self
        return self

    def __exit__(self, *exc) -> None:
        rt.RETENTION_TAP = self._previous

    def _span_labels(self, cut):
        """(span labels, new-span flags) over all rows, in candidate order."""
        labels, new = [], []
        for b, (positions, tokens, lengths, old) in enumerate(
                zip(cut.positions, cut.token_rows, cut.length_rows, cut.old_rows)):
            if not len(lengths):
                continue
            mask = self.labels[b]
            token_positions = np.asarray(positions)[np.asarray(tokens)]
            hit = mask[np.minimum(token_positions, len(mask) - 1)] & (token_positions < len(mask))
            starts = np.cumsum(lengths) - lengths
            labels.append(np.maximum.reduceat(hit, starts))
            new.append(~np.asarray(old, dtype=bool))
        return np.concatenate(labels), np.concatenate(new)

    def __call__(self, head, k, v, cut, token_batch, tokens, span_sizes) -> None:
        if self.labels is None:
            return
        labels, new = self._span_labels(cut)
        if not new.any():
            return
        # Only the new spans' tokens go through the head.
        span_new = torch.as_tensor(new, device=k.device)
        token_new = span_new.repeat_interleave(span_sizes)
        y = torch.as_tensor(labels[new], dtype=torch.float32, device=k.device)
        with torch.enable_grad():
            z = rt.span_logits(head, k.detach(), v.detach(), token_batch[token_new], tokens[token_new],
                               span_sizes[span_new])
            positives = float(y.sum())
            self.counts += (len(y) - positives, positives)
            weight = torch.where(y > 0, min(self.counts[0] / self.counts[1], MAX_POS_WEIGHT), 1.)
            loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * weight).mean()
            if self.train:
                loss.backward()
        z = z.detach()
        self.stats["loss"] += float(loss) * len(y)
        self.stats["spans"] += len(y)
        self.stats["positives"] += int(positives)
        self.stats["pos_logit"] += float(z[y > 0].sum())
        self.stats["neg_logit"] += float(z[y == 0].sum())

    def step(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            for p in self.heads.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                torch.distributed.all_reduce(p.grad)
                p.grad /= world
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def summary(self) -> dict:
        s = self.stats
        negatives = max(s["spans"] - s["positives"], 1)
        return dict(loss=s["loss"] / max(s["spans"], 1), spans=s["spans"], positives=s["positives"],
                    pos_logit=s["pos_logit"] / max(s["positives"], 1), neg_logit=s["neg_logit"] / negatives)


def pool_recall(model, masks) -> list[float]:
    """Per layer: the fraction of fact tokens kept exact (pool or recent) at the end of the forward."""
    out = []
    for block in model.transformer.h:
        cache = block.attn.kv_cache
        kept = total = 0
        for b, mask in enumerate(masks):
            facts = np.flatnonzero(mask)
            if not len(facts):
                continue
            exact = np.zeros(len(mask), dtype=bool)
            for kept_positions in (cache._alpha_positions[b], cache._recent_pos_host[b][:cache.recent_count]):
                kept_positions = np.asarray(kept_positions, dtype=np.int64)
                exact[kept_positions[kept_positions < len(mask)]] = True
            kept += int(exact[facts].sum())
            total += len(facts)
        out.append(kept / total if total else float("nan"))
    return out
