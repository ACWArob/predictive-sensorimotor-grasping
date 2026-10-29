"""Baselines, given the same reflexes as ours so only the prediction / learning differs.

DeliGraspLike : force from a semantic prior (looks-like material -> assumed density, friction),
                like an LLM estimating mass and friction from a description. No carry-over.
ExpForceLike  : retrieval. kNN over past grasps by APPEARANCE embedding, reuse the force that
                worked (Exp-Force, arXiv 2603.08668, without the VLM). Grows online.
"""
from __future__ import annotations

import numpy as np

from .env import G
from .objects import MATERIALS

# What a language model 'believes' about each material (deliberately not the sim truth)
PRIOR = {"wood": (650, 0.5), "plastic": (500, 0.4), "metal": (3000, 0.35),
         "foam": (80, 0.7), "ceramic": (1500, 0.55), "rubber": (1000, 0.85)}


class SlipReflex:
    """DeliGrasp-style: on slip, grip *= 1.4. Same boost ours uses."""
    def __init__(self, ceiling=None):
        self.ceiling = ceiling

    def __call__(self, event, info):
        if event == "slip":
            f = info["grip_N"] * 1.4
            return f if self.ceiling is None else min(f, self.ceiling)
        return None


class DeliGraspLike:
    name = "deligrasp"

    def __init__(self, margin=1.5, lift_accel_factor=1.33):
        self.margin, self.acc = margin, lift_accel_factor

    def plan(self, obj, percept):
        dens, mu = PRIOR[obj.material]
        m = dens * 8 * np.prod(obj.half_size)
        return m * G * self.acc / (2 * mu) * self.margin, SlipReflex()

    def observe(self, *a, **k):
        pass


class ExpForceLike:
    name = "expforce"

    def __init__(self, k=3, margin=1.2):
        self.k, self.margin = k, margin
        self.X, self.F = [], []
        self.fallback = DeliGraspLike()

    def plan(self, obj, percept):
        if len(self.X) < self.k:
            return self.fallback.plan(obj, percept)
        X = np.array(self.X)
        d = np.linalg.norm(X - percept.embedding, axis=1)
        idx = np.argsort(d)[: self.k]
        w = 1.0 / (d[idx] + 1e-3)
        return float(np.sum(w * np.array(self.F)[idx]) / w.sum()) * self.margin, SlipReflex()

    def observe(self, obj, percept, grip_cmd, res):
        if res.crushed:
            f = 0.7 * res.grip_max
        elif res.success:
            f = res.grip_max / self.margin if res.n_reflex == 0 else res.grip_max
        else:
            f = res.grip_max * 1.5
        self.X.append(percept.embedding)
        self.F.append(f)

    def seed(self, rows):
        """Offline experience (Exp-Force retrieves from a prior dataset)."""
        for emb, f in rows:
            self.X.append(emb)
            self.F.append(f)
