"""Value head ('basal ganglia'): learns the safety margin from grasp success alone.

Sim / first version: a LinUCB contextual bandit over a few margin multipliers, context =
[1, predicted log-load, log-ratio, total std]. It answers "given what I predict and how sure I am,
how much extra grip pays off?" (too little -> drop, too much -> crush fragile things).

Robot version: replace with an RL-Token-style actor-critic on the frozen VLA's readout token
(arXiv 2604.23073), same interface: margin(ctx) and update(ctx, action, reward).

Reward = success (1) / fail (0) minus a small force cost. Prediction error is NOT reward.
"""
from __future__ import annotations

import numpy as np

MARGINS = np.array([1.05, 1.2, 1.4, 1.7, 2.1])


class ValueHead:
    def __init__(self, alpha=0.5, lam=1.0, force_cost=0.05):
        self.alpha, self.lam, self.force_cost = alpha, lam, force_cost
        d = 4
        self.A = np.stack([lam * np.eye(d) for _ in MARGINS])
        self.b = np.zeros((len(MARGINS), d))

    @staticmethod
    def context(mean, std):
        return np.array([1.0, mean[0], mean[1], float(np.linalg.norm(std))])

    def choose(self, ctx, explore=True):
        scores = []
        for a in range(len(MARGINS)):
            Ainv = np.linalg.inv(self.A[a])
            th = Ainv @ self.b[a]
            bonus = self.alpha * np.sqrt(ctx @ Ainv @ ctx) if explore else 0.0
            scores.append(ctx @ th + bonus)
        a = int(np.argmax(scores))
        return a, float(MARGINS[a])

    def reward(self, success: bool, margin: float):
        return float(success) - self.force_cost * (margin - 1.0)

    def update(self, ctx, a: int, r: float):
        self.A[a] += np.outer(ctx, ctx)
        self.b[a] += r * ctx
