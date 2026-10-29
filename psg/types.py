"""Data types shared by every module. Units: newtons, metres, seconds. Log quantities are natural log."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Output dims of the forward model (both in log space so errors are multiplicative)
LOAD = 0    # log vertical load the fingers carry at lift-off (N)  ~ log m(g+a)
RATIO = 1   # log grip/load ratio at which the object slips        ~ log 1/(2 mu)
N_OUT = 2


@dataclass
class Percept:
    """What the robot knows before touching. On MAGPIE: DINOv2 embedding of the SAM3 mask + GraspGenX pose."""
    obj_id: str                 # instance key (tracker / scene slot), NOT the visual embedding
    embedding: np.ndarray       # appearance vector (DINOv2 768-d on the robot, 64-d in sim)
    grasp: np.ndarray           # grasp descriptor (width, approach dir, ...); may be empty

    def x(self) -> np.ndarray:
        return np.concatenate([self.embedding, self.grasp])


@dataclass
class Prediction:
    """Predicted sensory signature of the grasp + the command derived from it."""
    mean: np.ndarray            # (N_OUT,) log space, already includes the object residual
    std: np.ndarray             # (N_OUT,) total std (model + residual uncertainty)
    margin: float               # safety multiplier applied on top of predicted need
    grip_cmd: float             # N, commanded pre-grasp grip
    f_ceiling: float | None     # N, learned fragility cap for this object (None = unknown)
    familiar: bool              # gate: predictor confident enough to let the value head act

    @property
    def load(self) -> float:
        return float(np.exp(self.mean[LOAD]))

    @property
    def ratio(self) -> float:
        return float(np.exp(self.mean[RATIO]))


@dataclass
class ControlPoint:
    """One checkpoint where prediction meets sensation."""
    name: str                   # contact | liftoff | slip | crush
    t: float
    predicted: float | None
    actual: float | None
    z: float | None             # (log actual - log predicted) / std
    action: str                 # none | reflex | update | reflex+update


@dataclass
class GraspLog:
    obj_id: str
    mode: str
    prediction: Prediction
    control_points: list = field(default_factory=list)
    success: bool | None = None
    crushed: bool | None = None
    grip_max: float | None = None
