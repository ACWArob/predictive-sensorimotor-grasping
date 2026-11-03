"""PredictiveGraspAgent: wires predictor, residual store, comparator and value head.

Modes (the ablation ladder, same code path, flags only):
  predictor         forward model sets grip; reflexes on; no learning online
  predictor+update  + one-shot per-object residual updates (the Johansson part)
  full              + value head chooses the safety margin when the gate says 'familiar'
  full-nogate       value head always acts (tests the two-signal split)

Usage per grasp:
    pred, cmp = agent.plan(percept)
    result = controller.execute(pred.grip_cmd, callback=cmp)   # sim: run_grasp, robot: DeliGrasp
    agent.observe(percept, pred, cmp, success, crushed, grip_max)
"""
from __future__ import annotations

import numpy as np

from .comparator import Comparator, ComparatorConfig
from .predictor import ForwardModel
from .residual import ObjectResidualStore
from .types import GraspLog, Percept, Prediction
from .value_head import ValueHead

MODES = ("predictor", "predictor+update", "full", "full-nogate")


class PredictiveGraspAgent:
    def __init__(self, model: ForwardModel, mode="full", gate_std=0.35, base_margin=1.25,
                 std_margin_gain=1.0, f_max=80.0, cmp_cfg: ComparatorConfig | None = None,
                 value_head: ValueHead | None = None):
        assert mode in MODES, mode
        self.model, self.mode = model, mode
        self.gate_std, self.base_margin, self.k_std, self.f_max = gate_std, base_margin, std_margin_gain, f_max
        self.store = ObjectResidualStore()
        self.cmp_cfg = cmp_cfg or ComparatorConfig(update_on=mode != "predictor")
        self.vh = value_head or ValueHead()
        self.logs: list[GraspLog] = []
        self._pending = None

    # ---------------------------------------------------------------------
    def plan(self, p: Percept):
        mu, sd = self.model.predict(p.x())
        mu, sd = mu[0], sd[0]
        e = self.store.get(p.obj_id, sd)
        use_res = self.mode != "predictor"
        mean = mu + (e.r if use_res else 0.0)
        std = np.sqrt(e.P) if use_res else sd
        total_sd = float(np.linalg.norm(std))
        familiar = total_sd < self.gate_std

        # default margin: humans raise the safety margin when unsure
        margin = self.base_margin * np.exp(self.k_std * total_sd)
        arm = None
        ctx = ValueHead.context(mean, std)
        if self.mode == "full-nogate" or (self.mode == "full" and familiar):
            arm, margin = self.vh.choose(ctx)

        need = float(np.exp(mean[0] + mean[1]))
        grip = min(need * margin, self.f_max)
        if use_res and e.f_ceiling is not None:
            grip = min(grip, e.f_ceiling)
        pred = Prediction(mean, std, float(margin), grip, e.f_ceiling if use_res else None, familiar)
        cmp = Comparator(p.obj_id, pred, mu, self.store if use_res else None, self.cmp_cfg)
        self._pending = (ctx, arm)
        return pred, cmp

    # ---------------------------------------------------------------------
    def observe(self, p: Percept, pred: Prediction, cmp: Comparator, success: bool,
                crushed: bool, grip_max: float):
        if self.mode != "predictor" and crushed:
            self.store.set_ceiling(p.obj_id, 0.7 * grip_max)        # learned fragility
        elif self.mode != "predictor" and not success:
            # failure-aware: a drop means we under-gripped, so never let the residual stay where it was
            e = self.store.entries[p.obj_id]
            e.r[0] += np.log(1.5)
            e.P[0] = max(e.P[0], np.log(1.5) ** 2)
        ctx, arm = self._pending
        if arm is not None:
            self.vh.update(ctx, arm, self.vh.reward(success and not crushed, pred.margin))
        self.logs.append(GraspLog(p.obj_id, self.mode, pred, cmp.log, success, crushed, grip_max))

    def pretrain_value_head(self, rows):
        """Offline bandit warm-start from logged (percept, margin, success) rows (Phase 0 data)."""
        for x, margin, success in rows:
            mu, sd = self.model.predict(x)
            ctx = ValueHead.context(mu[0], sd[0])
            a = int(np.argmin(np.abs(np.log(margin) - np.log([1.05, 1.2, 1.4, 1.7, 2.1]))))
            self.vh.update(ctx, a, self.vh.reward(success, margin))
