"""Lightweight diagnostics for LogKV salience-pin score/mass calibration.

This is intentionally separate from ``log_kv_diag.py``. The oracle diagnostic
there assumes every slot maps to one contiguous token span and therefore rejects
salience pins, which are scattered exact duplicates. This module only compares
the score and softmax mass assigned to pooled slots vs pinned slots.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import math
from typing import Any

import torch


class _RunningStat:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf

    def add_tensor(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        with torch.no_grad():
            xf = x.detach().to(torch.float32)
            finite = xf[torch.isfinite(xf)]
            if finite.numel() == 0:
                return
            self.count += int(finite.numel())
            self.total += float(finite.sum().item())
            self.sumsq += float(finite.square().sum().item())
            self.min = min(self.min, float(finite.min().item()))
            self.max = max(self.max, float(finite.max().item()))

    def add_state(self, state: dict[str, Any] | None) -> None:
        if not state or int(state.get("count", 0)) == 0:
            return
        self.count += int(state["count"])
        self.total += float(state["sum"])
        self.sumsq += float(state["sumsq"])
        self.min = min(self.min, float(state["min"]))
        self.max = max(self.max, float(state["max"]))

    def state_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}
        return {
            "count": self.count,
            "sum": self.total,
            "sumsq": self.sumsq,
            "min": self.min,
            "max": self.max,
        }

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.min,
            "max": self.max,
        }


def _tensor_state(x: torch.Tensor) -> dict[str, Any]:
    stat = _RunningStat()
    stat.add_tensor(x)
    return stat.state_dict()


class _Bucket:
    def __init__(self) -> None:
        self.calls = 0
        self.query_vectors = 0
        self.q_pos_min: int | None = None
        self.q_pos_max: int | None = None
        self.stats: defaultdict[str, _RunningStat] = defaultdict(_RunningStat)

    def add_meta(self, *, q_vectors: int, q_abs_start: int, q_abs_end: int) -> None:
        self.calls += 1
        self.query_vectors += int(q_vectors)
        if q_abs_start < q_abs_end:
            self.q_pos_min = q_abs_start if self.q_pos_min is None else min(self.q_pos_min, q_abs_start)
            self.q_pos_max = q_abs_end - 1 if self.q_pos_max is None else max(self.q_pos_max, q_abs_end - 1)

    def add_state(self, name: str, state: dict[str, Any] | None) -> None:
        self.stats[name].add_state(state)

    def add_tensor(self, name: str, x: torch.Tensor) -> None:
        self.stats[name].add_tensor(x)

    def state_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "query_vectors": self.query_vectors,
            "q_pos_min": self.q_pos_min,
            "q_pos_max": self.q_pos_max,
            "stats": {name: stat.state_dict() for name, stat in self.stats.items()},
        }

    def merge_state(self, state: dict[str, Any]) -> None:
        self.calls += int(state.get("calls", 0))
        self.query_vectors += int(state.get("query_vectors", 0))
        q_min = state.get("q_pos_min")
        q_max = state.get("q_pos_max")
        if q_min is not None:
            self.q_pos_min = int(q_min) if self.q_pos_min is None else min(self.q_pos_min, int(q_min))
        if q_max is not None:
            self.q_pos_max = int(q_max) if self.q_pos_max is None else max(self.q_pos_max, int(q_max))
        for name, stat_state in state.get("stats", {}).items():
            self.stats[name].add_state(stat_state)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "query_vectors": self.query_vectors,
            "q_pos_min": self.q_pos_min,
            "q_pos_max": self.q_pos_max,
            "stats": {name: stat.summary() for name, stat in sorted(self.stats.items())},
        }


class PinScoreDiag:
    def __init__(self) -> None:
        self.enabled = False
        self.window_from_end = 512
        self.reset(window_from_end=self.window_from_end)

    def reset(self, *, window_from_end: int = 512) -> None:
        self.window_from_end = max(1, int(window_from_end))
        self.buckets: defaultdict[str, _Bucket] = defaultdict(_Bucket)

    def capture_score_stats(
        self,
        scores: torch.Tensor,
        *,
        q_slice: tuple[int, int],
        pin_slot_range: tuple[int, int],
        pooled_slot_range: tuple[int, int],
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        q0, q1 = q_slice
        ps0, ps1 = pin_slot_range
        po0, po1 = pooled_slot_range
        if q0 >= q1 or ps0 >= ps1 or po0 >= po1:
            return None
        return {
            "dot_pin_score": _tensor_state(scores[..., q0:q1, ps0:ps1]),
            "dot_pooled_score": _tensor_state(scores[..., q0:q1, po0:po1]),
        }

    def record(
        self,
        *,
        layer: int,
        branch: str,
        q_offset: int,
        q_slice: tuple[int, int],
        pin_slot_range: tuple[int, int],
        pooled_slot_range: tuple[int, int],
        dot_stats: dict[str, Any] | None,
        final_scores: torch.Tensor,
        attn: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        q0, q1 = q_slice
        ps0, ps1 = pin_slot_range
        po0, po1 = pooled_slot_range
        if q0 >= q1 or ps0 >= ps1 or po0 >= po1:
            return

        pin_count = ps1 - ps0
        pooled_count = po1 - po0
        final_pin = final_scores[..., q0:q1, ps0:ps1]
        final_pooled = final_scores[..., q0:q1, po0:po1]
        attn_pin = attn[..., q0:q1, ps0:ps1].to(torch.float32)
        attn_pooled = attn[..., q0:q1, po0:po1].to(torch.float32)

        pin_mass_total = attn_pin.sum(dim=-1)
        pooled_mass_total = attn_pooled.sum(dim=-1)
        pin_mass_per_slot = pin_mass_total / max(pin_count, 1)
        pooled_mass_per_slot = pooled_mass_total / max(pooled_count, 1)
        ratio = pin_mass_per_slot / pooled_mass_per_slot.clamp_min(1e-30)
        log10_ratio = torch.log10(ratio.clamp_min(1e-30))

        q_vectors = int(pin_mass_per_slot.numel())
        q_abs_start = int(q_offset + q0)
        q_abs_end = int(q_offset + q1)

        for key in ("total", f"layer_{int(layer)}", f"layer_{int(layer)}:{branch}"):
            bucket = self.buckets[key]
            bucket.add_meta(q_vectors=q_vectors, q_abs_start=q_abs_start, q_abs_end=q_abs_end)
            if dot_stats:
                bucket.add_state("dot_pin_score", dot_stats.get("dot_pin_score"))
                bucket.add_state("dot_pooled_score", dot_stats.get("dot_pooled_score"))
            bucket.add_tensor("final_pin_score", final_pin)
            bucket.add_tensor("final_pooled_score", final_pooled)
            bucket.add_tensor("pin_mass_total", pin_mass_total)
            bucket.add_tensor("pooled_mass_total", pooled_mass_total)
            bucket.add_tensor("pin_mass_per_slot", pin_mass_per_slot)
            bucket.add_tensor("pooled_mass_per_slot", pooled_mass_per_slot)
            bucket.add_tensor("pin_to_pooled_mass_per_slot_ratio", ratio)
            bucket.add_tensor("pin_to_pooled_mass_per_slot_log10_ratio", log10_ratio)

    def state_dict(self) -> dict[str, Any]:
        return {
            "window_from_end": self.window_from_end,
            "buckets": {key: bucket.state_dict() for key, bucket in self.buckets.items()},
        }

    def merge_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        self.window_from_end = max(self.window_from_end, int(state.get("window_from_end", self.window_from_end)))
        for key, bucket_state in state.get("buckets", {}).items():
            self.buckets[key].merge_state(bucket_state)

    def summary(self) -> dict[str, Any]:
        return {
            "window_from_end": self.window_from_end,
            "total": self.buckets["total"].summary() if "total" in self.buckets else _Bucket().summary(),
            "by_layer": {
                key.removeprefix("layer_"): bucket.summary()
                for key, bucket in sorted(self.buckets.items())
                if key.startswith("layer_") and ":" not in key
            },
            "by_layer_branch": {
                key: bucket.summary()
                for key, bucket in sorted(self.buckets.items())
                if key.startswith("layer_") and ":" in key
            },
        }


DIAG = PinScoreDiag()


@contextmanager
def pin_score_diag_mode(enabled: bool, *, window_from_end: int = 512):
    old_enabled = DIAG.enabled
    old_window = DIAG.window_from_end
    if enabled:
        DIAG.reset(window_from_end=window_from_end)
        DIAG.enabled = True
    try:
        yield DIAG
    finally:
        DIAG.enabled = old_enabled
        if not enabled:
            DIAG.window_from_end = old_window


def merge_pin_score_diag_states(states: list[dict[str, Any] | None]) -> dict[str, Any]:
    merged = PinScoreDiag()
    valid_states = [state for state in states if state]
    if valid_states:
        merged.reset(window_from_end=int(valid_states[0].get("window_from_end", merged.window_from_end)))
    for state in valid_states:
        merged.merge_state(state)
    return merged.summary()
