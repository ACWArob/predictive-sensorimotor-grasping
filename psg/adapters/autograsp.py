"""Glue between PSG and MAGPIE AutoGrasp on the real robot.

Everything robot-specific is behind the four small protocols below; implement them against the
existing MAGPIE modules (names in the TODOs are the stack's components, not verified function
names). PSG itself never imports ROS, the gripper driver, or Gemini.

Where it sits in the loop:

    frame -> Gemini detect -> SAM3 mask -> DINOv2 embedding ----------+
                                       -> GraspGenX/PCA -> Gemini arbitration -> grasp pose
                                                                       |
            Percept(obj_id, embedding, grasp_descriptor) <-------------+
                     |
            agent.plan()  -> grip_cmd + Comparator (callback)
                     |
            DeliGrasp force-closure loop  ---- events: contact / load / liftoff / slip ---> Comparator
                     |                                                   (returns new grip = reflex)
            outcome (lift + 3 s hold) -> agent.observe() -> residual / ceiling / value head
                     |
            LeRobot v3 episode + control-point log
"""
from __future__ import annotations

import json
import os
from typing import Callable, Protocol

import numpy as np

from ..agent import PredictiveGraspAgent
from ..types import Percept

Callback = Callable[[str, dict], "float | None"]


class Perception(Protocol):
    def embedding(self, mask) -> np.ndarray: ...              # TODO: DINOv2 CLS on the SAM3 crop (768-d)
    def instance_id(self, mask, embedding) -> str: ...         # TODO: tracker / scene slot (see README: look-alikes)


class GraspPlanner(Protocol):
    def best_grasp(self, mask) -> tuple[object, np.ndarray]: ...   # TODO: GraspGenX + Gemini arbitration -> (pose, descriptor)


class ForceController(Protocol):
    def execute(self, pose, grip_N: float, on_event: Callback) -> dict: ...
    # TODO: wrap DeliGrasp. Must:
    #   - start closing at grip_N
    #   - call on_event('contact', {'t'}) when both pads load
    #   - call on_event('load', {'load_N','grip_N'}) every control tick between contact and lift-off
    #   - call on_event('liftoff', {'load_N','grip_N','t'}) when the object leaves the support
    #   - call on_event('slip', {'slip_m','grip_N'}) on the slip-guard trigger
    #   - apply any float returned as the new grip setpoint (clip to the gripper limit)
    #   - return {'success': bool, 'crushed': bool, 'grip_max': float}
    # load_N: vertical load on the fingers = wrist F/T z minus gripper weight, or motor-current proxy.


class EpisodeLogger(Protocol):
    def write(self, record: dict) -> None: ...                # TODO: append to the LeRobot v3 episode metadata


def grasp_once(agent: PredictiveGraspAgent, perception: Perception, planner: GraspPlanner,
               controller: ForceController, logger: EpisodeLogger, mask) -> dict:
    emb = perception.embedding(mask)
    obj_id = perception.instance_id(mask, emb)
    pose, desc = planner.best_grasp(mask)
    p = Percept(obj_id, emb, np.asarray(desc, float))
    pred, cmp = agent.plan(p)
    out = controller.execute(pose, pred.grip_cmd, cmp)
    agent.observe(p, pred, cmp, out["success"], out.get("crushed", False), out["grip_max"])
    rec = {"obj_id": obj_id, "mode": agent.mode, "grip_cmd": pred.grip_cmd, "margin": pred.margin,
           "pred_load": pred.load, "pred_ratio": pred.ratio, "pred_std": pred.std.tolist(),
           "familiar": pred.familiar, **out,
           "control_points": [cp.__dict__ for cp in cmp.log]}
    logger.write(rec)
    return rec


def save_residuals(agent: PredictiveGraspAgent, path: str):
    """Residuals are the robot's per-object knowledge; persist between sessions."""
    data = {k: {"r": e.r.tolist(), "P": e.P.tolist(), "f_ceiling": e.f_ceiling}
            for k, e in agent.store.entries.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def load_residuals(agent: PredictiveGraspAgent, path: str):
    if not os.path.exists(path):
        return
    from ..residual import _Entry
    for k, v in json.load(open(path)).items():
        agent.store.entries[k] = _Entry(np.array(v["r"]), np.array(v["P"]), v["f_ceiling"])
