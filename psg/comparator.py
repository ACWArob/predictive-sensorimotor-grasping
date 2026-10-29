"""Comparator ('tactile control points'): actual vs predicted at each contact event.

Two responses, two thresholds:
  reflex  -> change grip NOW (grip-load coupling / slip response). Not learning.
  update  -> one-shot Kalman update of this object's residual. Learning, but only for this object.

Used as the per-step callback of the grasp controller (sim/env.py on sim, DeliGrasp loop on robot).
Prediction error is never a reward; the value head never sees it except through the gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .residual import ObjectResidualStore
from .types import LOAD, RATIO, ControlPoint, Prediction


@dataclass
class ComparatorConfig:
    z_reflex: float = 2.0        # load above predicted by this many sd before lift-off -> couple grip to load
    z_update: float = 1.0        # |z| at lift-off above this -> residual update
    reflex_margin: float = 1.3   # grip = load * ratio * this, when coupling
    slip_boost: float = 1.4      # grip *= this on a slip event
    reflex_on: bool = True
    update_on: bool = True


class Comparator:
    def __init__(self, obj_id: str, pred: Prediction, model_mean: np.ndarray,
                 store: ObjectResidualStore | None, cfg: ComparatorConfig):
        self.obj_id, self.pred, self.model_mean = obj_id, pred, model_mean
        self.store, self.cfg = store, cfg
        self.log: list[ControlPoint] = []
        self._coupled = False
        self._load_actual = None

    def _cap(self, f):
        return f if self.pred.f_ceiling is None else min(f, self.pred.f_ceiling)

    def __call__(self, event: str, info: dict):
        p, c = self.pred, self.cfg
        if event == "contact":
            self.log.append(ControlPoint("contact", info["t"], None, None, None, "none"))
            return None

        if event == "load":
            # Object should have left the table at ~p.load. Still pulling harder than expected -> heavier
            # than it looks. Human response: scale grip with load (Johansson & Westling).
            hi = p.load * np.exp(c.z_reflex * p.std[LOAD])
            if c.reflex_on and info["load_N"] > hi:
                need = self._cap(info["load_N"] * p.ratio * c.reflex_margin)
                if need > info["grip_N"]:
                    if not self._coupled:
                        self._coupled = True
                        self.log.append(ControlPoint("load", -1, p.load, info["load_N"], None, "reflex"))
                    return need
            return None

        if event == "liftoff":
            actual = max(info["load_N"], 1e-3)
            self._load_actual = actual
            z = (np.log(actual) - p.mean[LOAD]) / p.std[LOAD]
            action = "none"
            if c.update_on and self.store is not None and abs(z) > c.z_update:
                innov = np.log(actual) - p.mean[LOAD]
                self.store.update(self.obj_id, LOAD, innov)
                action = "update"
            self.log.append(ControlPoint("liftoff", info["t"], p.load, actual, float(z), action))
            return None

        if event == "slip":
            load = self._load_actual or p.load
            implied = info["grip_N"] / max(load, 1e-3)            # it slipped at this ratio
            action = "none"
            if c.update_on and self.store is not None:
                need_log = np.log(implied * 1.1) - self.model_mean[RATIO]
                self.store.lower_bound(self.obj_id, RATIO, need_log)
                action = "update"
            out = None
            if c.reflex_on:
                out = self._cap(info["grip_N"] * c.slip_boost)
                action = "reflex+update" if action == "update" else "reflex"
            self.log.append(ControlPoint("slip", -1, p.ratio, implied, None, action))
            return out
        return None
