"""Fixed-budget eviction KV caches: the training-free baselines LogKV is compared with at equal memory.

Each layer keeps at most ``capacity`` K/V entries per KV head for the whole request, prefill included,
so the cache never grows with the sequence. The input is attended in chunks of ``chunk`` tokens over
[kept entries + the chunk (causal)]; the cache then keeps ``capacity`` of those entries by its policy.
Keys are stored after RoPE at their original positions, so evicting an entry moves no other entry.

    streaming  StreamingLLM: the first ``sinks`` positions and the most recent ones.
    h2o        H2O: the ``recent`` newest entries, the rest by the attention they have received so far
               (summed over every query since they arrived, averaged over a KV group's query heads).
    snapkv     SnapKV, per chunk: the chunk's last ``window`` queries score every entry (softmax
               attention averaged over those queries and the group's heads, then average-pooled along
               the sequence with ``kernel``); the ``recent`` newest entries stay (default ``window``, at
               most half the budget), the rest by score.
               The scores are recomputed for every chunk, decode steps included (window 1 there).

Persistent memory of one entry in one layer, per KV head: K and V (``head_size`` each), an int64 position
and, for h2o, a float32 score (``slot_bytes``). The chunk's keys and scores are transient workspace of
O(capacity + chunk) entries.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

POLICIES = ("streaming", "h2o", "snapkv")


def slot_bytes(policy: str, n_groups: int, head_size: int, element_size: int) -> int:
    """Persistent bytes of one entry across the KV heads of one layer."""
    return n_groups * (2 * head_size * element_size + 8 + (4 if policy == "h2o" else 0))


def capacity_for(budget_bytes: int, policy: str, n_layer: int, n_groups: int, head_size: int,
                 element_size: int) -> int:
    """The most entries per KV head whose persistent buffers fit ``budget_bytes`` over all layers."""
    return int(budget_bytes) // (n_layer * slot_bytes(policy, n_groups, head_size, element_size))


class BudgetKVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        n_groups: int,
        head_size: int,
        capacity: int,
        *,
        policy: str,
        sinks: int = 4,
        recent: int | None = None,
        window: int = 64,
        kernel: int = 5,
        chunk: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if policy not in POLICIES:
            raise ValueError(f"unknown baseline policy {policy!r}; choose from {POLICIES}")
        if capacity < 2:
            raise ValueError(f"baseline capacity must hold at least 2 entries, got {capacity}")
        if window < 1 or chunk < 1 or kernel < 1 or kernel % 2 == 0:
            raise ValueError("window and chunk must be positive and kernel a positive odd number")
        self.policy = policy
        self.capacity = int(capacity)
        self.sinks = int(sinks) if policy == "streaming" else 0
        if not 0 <= self.sinks < self.capacity:
            raise ValueError(f"sinks must be in [0, capacity), got {sinks} for capacity {capacity}")
        default = {"streaming": self.capacity - self.sinks, "h2o": self.capacity // 2,
                   "snapkv": min(window, self.capacity // 2)}[policy]
        self.recent = int(default if recent is None or policy == "streaming" else recent)
        if not 0 <= self.recent <= self.capacity:
            raise ValueError(f"recent must be in [0, capacity], got {self.recent} for capacity {capacity}")
        self.window, self.kernel, self.chunk = int(window), int(kernel), int(chunk)
        shape = (batch_size, n_groups, self.capacity, head_size)
        self.register_buffer("k", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("positions", torch.full(shape[:3], -1, device=device, dtype=torch.long),
                             persistent=False)
        if policy == "h2o":
            self.register_buffer("score", torch.zeros(shape[:3], device=device, dtype=torch.float32),
                                 persistent=False)
        self.count = 0  # entries held in every KV head
        self.seen = 0  # tokens forwarded since the last reset

    def reset_parameters(self) -> None:
        self.positions.fill_(-1)
        if self.policy == "h2o":
            self.score.zero_()
        self.count = self.seen = 0

    def attend(self, attn: nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               input_pos: torch.Tensor) -> torch.Tensor:
        """Attention of q (B, n_head, T, hs) over the kept entries and the new k / v (B, n_groups, T, hs),
        which then enter the cache. Returns (B, T, n_head * hs)."""
        if attn.config.attention_logit_softcapping is not None:
            raise ValueError("eviction baselines do not support attention logit softcapping")
        if input_pos.dim() == 2:
            if not torch.equal(input_pos, input_pos[:1].expand_as(input_pos)):
                raise ValueError("eviction baselines need the same positions in every batch row")
            input_pos = input_pos[0]
        batch, _, length, _ = q.shape
        if int(input_pos[0]) != self.seen:
            raise ValueError(f"eviction baselines append in order: expected position {self.seen}, "
                             f"got {int(input_pos[0])}")
        if self.k.dtype != k.dtype:
            self.k, self.v = self.k.to(k.dtype), self.v.to(v.dtype)
        scale = attn.mscale * attn.mscale / math.sqrt(attn.config.attention_scores_scalar or attn.config.head_size)
        outputs = []
        for start in range(0, length, self.chunk):
            end = min(start + self.chunk, length)
            outputs.append(self._chunk(attn, q[:, :, start:end], k[:, :, start:end], v[:, :, start:end],
                                       input_pos[start:end], scale))
        return torch.cat(outputs, dim=1).reshape(batch, length, -1)

    def _chunk(self, attn, q, k, v, pos, scale) -> torch.Tensor:
        batch, heads, t, head_size = q.shape
        groups, n = k.size(1), self.count
        rep = heads // groups
        keys = torch.cat((self.k[:batch, :, :n], k), dim=2)
        values = torch.cat((self.v[:batch, :, :n], v), dim=2)
        mask = torch.ones(t, n + t, dtype=torch.bool, device=q.device)
        mask[:, n:] = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        y = attn.scaled_dot_product_attention(
            q, keys.repeat_interleave(rep, dim=1), values.repeat_interleave(rep, dim=1), mask[None, None])

        all_pos = torch.cat((self.positions[:batch, :, :n], pos.view(1, 1, t).expand(batch, groups, t)), dim=2)
        score = None
        if self.policy == "h2o":
            previous = torch.zeros(batch, groups, n + t, dtype=torch.float32, device=q.device)
            previous[:, :, :n] = self.score[:batch, :, :n]
            score = previous + self._mass(q, keys, mask, scale)
        elif self.policy == "snapkv" and n + t > self.capacity:
            w = min(self.window, t)
            mass = self._mass(q[:, :, t - w:], keys, mask[t - w:], scale) / w
            score = F.avg_pool1d(mass.view(batch * groups, 1, n + t), self.kernel, stride=1,
                                 padding=self.kernel // 2, count_include_pad=False).view(batch, groups, n + t)

        if n + t > self.capacity:
            keep = self._select(all_pos, score)
            index = keep.unsqueeze(-1).expand(-1, -1, -1, head_size)
            self.k[:batch].copy_(keys.gather(2, index))
            self.v[:batch].copy_(values.gather(2, index))
            self.positions[:batch].copy_(all_pos.gather(2, keep))
            if self.policy == "h2o":
                self.score[:batch].copy_(score.gather(2, keep))
            self.count = self.capacity
        else:
            self.k[:batch, :, n:n + t] = k
            self.v[:batch, :, n:n + t] = v
            self.positions[:batch, :, n:n + t] = all_pos[:, :, n:]
            if self.policy == "h2o":
                self.score[:batch, :, :n + t] = score
            self.count = n + t
        self.seen += t
        return y

    def _mass(self, q, keys, mask, scale, block: int = 128) -> torch.Tensor:
        """Softmax attention each entry receives from the rows of q, summed over the rows and averaged
        over the query heads of each KV group: (B, n_groups, entries), float32."""
        batch, heads, rows, head_size = q.shape
        groups, entries = keys.size(1), keys.size(2)
        rep = heads // groups
        k32 = keys.float()
        total = torch.zeros(batch, groups, entries, dtype=torch.float32, device=q.device)
        for s in range(0, rows, block):
            qs = q[:, :, s:s + block].float()
            qs = qs.view(batch, groups, rep, qs.size(2), head_size)  # head h reads group h // rep
            logits = torch.einsum("bgrqd,bgld->bgrql", qs, k32) * scale
            logits.masked_fill_(~mask[s:s + block].view(1, 1, 1, -1, entries), float("-inf"))
            total += logits.softmax(dim=-1).sum(dim=(2, 3)) / rep
        return total

    def _select(self, all_pos: torch.Tensor, score: torch.Tensor | None) -> torch.Tensor:
        """Indices of the ``capacity`` entries to keep, per batch row and KV head, in position order (the
        entries stay sorted by position, so SnapKV's pooling runs over sequence neighbours)."""
        newest = all_pos.max(dim=-1, keepdim=True).values
        protected = all_pos > newest - self.recent
        if self.policy == "streaming":
            protected |= all_pos < self.sinks
            priority = all_pos.to(torch.float32)
        else:
            priority = score.clone()
        keep = priority.masked_fill(protected, float("inf")).topk(self.capacity, dim=-1).indices
        return keep.sort(dim=-1).values
