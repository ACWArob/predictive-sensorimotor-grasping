"""Per-object residual ('fast system'): one-grasp correction that never touches shared weights.

Each object instance carries a small offset r (log space, one per predicted quantity) with its own
variance P. A control-point mismatch is a Kalman update: when the predictor was unsure (P large vs
sensor noise R) the gain is ~1, so ONE grasp fixes the next prediction for THIS object, which is
the Johansson one-lift update. When the predictor was confident but badly wrong (a deceptive
object), the surprise itself inflates P first, so the gain is ~1 there too. Different objects never share r, so nothing leaks and nothing is
forgotten (Complementary Learning Systems: fast, separate, cheap to overwrite).

Also stores a learned fragility ceiling per object after a crush.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import N_OUT


@dataclass
class _Entry:
    r: np.ndarray
    P: np.ndarray
    f_ceiling: float | None = None
    n_updates: int = 0


@dataclass
class ObjectResidualStore:
    sensor_sd: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.10]))   # R per output (log)
    process_sd: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01]))  # Q drift per grasp
    surprise_sigmas: float = 3.0
    entries: dict = field(default_factory=dict)

    def get(self, obj_id: str, prior_std: np.ndarray):
        """Residual and its variance. A new object starts at r=0 with P = predictor's own variance."""
        if obj_id not in self.entries:
            self.entries[obj_id] = _Entry(np.zeros(N_OUT), np.asarray(prior_std, float) ** 2)
        e = self.entries[obj_id]
        e.P = e.P + self.process_sd ** 2
        return e

    def update(self, obj_id: str, dim: int, innovation: float) -> float:
        """Kalman step on one output. innovation = observed - (model mean + r). Returns gain used."""
        e = self.entries[obj_id]
        R = self.sensor_sd[dim] ** 2
        # Unexpected uncertainty (Yu & Dayan 2005): a surprise far outside what the model expected
        # means the prior itself is wrong for this object, so trust the sensation: inflate P.
        if innovation ** 2 > self.surprise_sigmas ** 2 * (e.P[dim] + R):
            e.P[dim] = innovation ** 2
        K = e.P[dim] / (e.P[dim] + R)
        e.r[dim] += K * innovation
        e.P[dim] = (1 - K) * e.P[dim]
        e.n_updates += 1
        return float(K)

    def lower_bound(self, obj_id: str, dim: int, value: float):
        """One-sided evidence (e.g. it slipped at this ratio, so the true ratio is at least this)."""
        e = self.entries[obj_id]
        if value > e.r[dim]:
            e.r[dim] = value
            e.P[dim] = min(e.P[dim], self.sensor_sd[dim] ** 2)

    def set_ceiling(self, obj_id: str, f: float):
        e = self.entries[obj_id]
        e.f_ceiling = f if e.f_ceiling is None else min(e.f_ceiling, f)
