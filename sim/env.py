"""Minimal MuJoCo grasp-and-lift env for comparing grasp-force strategies.

A floating parallel-jaw gripper (z slide + two force-controlled finger slides)
grasps a box on a table, lifts it 10 cm, and holds it. Only the commanded grip
force is a free choice, so every method is compared on exactly the thing the
predictor / value head control. Physics truth per object: mass, friction, crush
limit. The controller sees control-point events and may change grip force live
(the reflex), exactly like DeliGrasp's slip response on the real robot.

Events passed to the callback:
    'contact'  both fingers touching             info: t
    'load'     every step between contact and lift-off   info: load_N, grip_N
    'liftoff'  object leaves the table           info: load_N, grip_N, t
    'slip'     object slides >3 mm in the hand   info: slip_m, grip_N
The callback returns a new grip force (N) or None.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

G = 9.81
F_MAX_ACTUATOR = 80.0
LIFT_HEIGHT = 0.10
SLIP_EVENT_M = 0.003
DROP_M = 0.015

_XML = """
<mujoco model="grip_lift">
  <option timestep="0.001" gravity="0 0 -9.81" cone="elliptic" impratio="10" noslip_iterations="3"/>
  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </default>
  <worldbody>
    <geom name="table" type="plane" size="1 1 .05" friction="0.6 0.005 0.0001"/>
    <body name="hand" pos="0 0 {hz}">
      <joint name="lift" type="slide" axis="0 0 1" damping="50"/>
      <geom type="box" size=".02 .02 .01" pos="0 0 {top}" mass="0.5" contype="0" conaffinity="0"/>
      <body name="fl" pos="{nfx} 0 0">
        <joint name="jl" type="slide" axis="1 0 0" range="0 {fx}" damping="5"/>
        <geom name="fl" type="box" size=".006 .02 {fz}" mass="0.05" friction="{mu} 0.005 0.0001"/>
      </body>
      <body name="fr" pos="{fx} 0 0">
        <joint name="jr" type="slide" axis="-1 0 0" range="0 {fx}" damping="5"/>
        <geom name="fr" type="box" size=".006 .02 {fz}" mass="0.05" friction="{mu} 0.005 0.0001"/>
      </body>
    </body>
    <body name="obj" pos="0 0 {oz}">
      <freejoint/>
      <geom name="obj" type="box" size="{sx} {sy} {sz}" mass="{m}" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <position name="lift" joint="lift" kp="4000" ctrlrange="-1 1"/>
    <motor name="ml" joint="jl" ctrlrange="0 {fmax}"/>
    <motor name="mr" joint="jr" ctrlrange="0 {fmax}"/>
  </actuator>
</mujoco>
"""


@dataclass
class PhysObject:
    """Ground truth the robot never sees directly."""
    name: str
    half_size: tuple            # (sx, sy, sz) metres
    mass: float                 # kg
    mu: float                   # friction coefficient
    f_crush: float              # N; grip above this breaks/dents it
    material: str = "plastic"   # what it LOOKS like (may lie)

    @property
    def f_required(self) -> float:
        """Minimum grip to hold statically: two frictional contacts carry m*g."""
        return self.mass * G / (2.0 * self.mu)


@dataclass
class GraspResult:
    success: bool
    crushed: bool
    dropped: bool
    max_slip_m: float
    grip_initial: float
    grip_max: float
    n_reflex: int
    t_contact: float | None
    load_liftoff: float | None
    grip_liftoff: float | None
    slipped: bool = False               # a slip event fired during the hold
    hold_load: float | None = None      # mean vertical load while held = m*g (mass label)
    slip_ratio: float | None = None     # grip/load at which it slipped (friction label, probe only)
    events: list = field(default_factory=list)


def _build(obj: PhysObject):
    sx, sy, sz = obj.half_size
    fx = sx + 0.012
    xml = _XML.format(hz=sz, top=sz + 0.03, fx=fx, nfx=-fx, fz=min(sz, 0.03) * 0.9,
                      sx=sx, sy=sy, sz=sz, oz=sz, m=obj.mass, mu=obj.mu,
                      fmax=F_MAX_ACTUATOR)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def _finger_forces(model, data, obj_id, finger_ids, table_id):
    """Vertical load the fingers carry, mean normal grip, both-finger contact, table contact."""
    load, normals, touching, on_table = 0.0, [], set(), False
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if obj_id not in pair:
            continue
        other = (pair - {obj_id}).pop() if len(pair) == 2 else obj_id
        mujoco.mj_contactForce(model, data, i, f6)
        world = c.frame.reshape(3, 3).T @ f6[:3]          # force on geom2, world frame
        if c.geom1 == obj_id:
            world = -world                                   # make it the force ON the object
        if other == table_id:
            on_table = on_table or abs(f6[0]) > 1e-3
        elif other in finger_ids:
            touching.add(other)
            load += world[2]
            normals.append(abs(f6[0]))
    grip = float(np.mean(normals)) if normals else 0.0
    return load, grip, len(touching) == 2, on_table


def run_grasp(obj: PhysObject, grip_cmd: float, callback=None,
              close_s=0.3, lift_s=1.0, hold_s=1.0, probe_slip=False) -> GraspResult:
    """probe_slip: after a successful hold, ramp grip down until it slips to measure the
    grip/load ratio (the friction label). Used only by the Phase-0 data collector."""
    model, data = _build(obj)
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("obj", "fl", "fr", "table")}
    finger_ids = {ids["fl"], ids["fr"]}
    oz0 = obj.half_size[2]

    grip = float(np.clip(grip_cmd, 0, F_MAX_ACTUATOR))
    res = GraspResult(False, False, False, 0.0, grip, grip, 0, None, None, None)
    contact_seen = lifted = slip_fired = False
    hz0 = None
    hold_loads = []
    reflexed = set()

    def apply(newf, why):
        nonlocal grip
        if newf is None:
            return
        newf = float(np.clip(newf, 0, F_MAX_ACTUATOR))
        if newf > grip + 1e-6 and why not in reflexed:
            reflexed.add(why)
            res.n_reflex += 1           # count reflex episodes, not control steps
        grip = newf
        res.events.append((why, round(data.time, 3), round(grip, 2)))

    n_close, n_lift, n_hold = (int(s / model.opt.timestep) for s in (close_s, lift_s, hold_s))
    for k in range(n_close + n_lift + n_hold):
        if k < n_close:
            ramp = min(1.0, k / (0.1 / model.opt.timestep))
            data.ctrl[0] = 0.0
            data.ctrl[1] = data.ctrl[2] = grip * ramp
        else:
            frac = min(1.0, (k - n_close) / n_lift)
            data.ctrl[0] = LIFT_HEIGHT * frac
            data.ctrl[1] = data.ctrl[2] = grip
        mujoco.mj_step(model, data)

        load, grip_meas, both, on_table = _finger_forces(model, data, ids["obj"], finger_ids, ids["table"])
        res.grip_max = max(res.grip_max, grip)
        if grip > obj.f_crush:
            res.crushed = True

        if both and not contact_seen:
            contact_seen = True
            res.t_contact = data.time
            if callback:
                apply(callback("contact", {"t": data.time}), "contact")

        oz = data.body("obj").xpos[2]
        hz = data.body("hand").xpos[2]
        if k >= n_close and not lifted:
            if callback and contact_seen:
                apply(callback("load", {"load_N": load, "grip_N": grip}), "load")
            if oz > oz0 + 0.001:
                lifted = True
                hz0 = hz - oz
                res.load_liftoff, res.grip_liftoff = load, grip
                if callback:
                    apply(callback("liftoff", {"load_N": load, "grip_N": grip, "t": data.time}), "liftoff")
        if lifted and k >= n_close + n_lift + n_hold // 2:
            hold_loads.append(load)
        if lifted:
            slip = max(0.0, (hz - oz) - hz0)
            res.max_slip_m = max(res.max_slip_m, slip)
            if slip > SLIP_EVENT_M and not slip_fired:
                slip_fired = True
                res.slipped = True
                if callback:
                    apply(callback("slip", {"slip_m": slip, "grip_N": grip}), "slip")

    final_oz = data.body("obj").xpos[2]
    res.dropped = (not lifted) or res.max_slip_m > DROP_M or final_oz < oz0 + 0.5 * LIFT_HEIGHT
    res.success = not res.crushed and not res.dropped
    if hold_loads:
        res.hold_load = float(np.mean(hold_loads))

    if probe_slip and res.success:
        rel0 = data.body("hand").xpos[2] - data.body("obj").xpos[2]
        g = grip
        while g > 0.05:
            g *= 0.995
            data.ctrl[1] = data.ctrl[2] = g
            for _ in range(5):
                mujoco.mj_step(model, data)
            rel = data.body("hand").xpos[2] - data.body("obj").xpos[2]
            if rel - rel0 > 0.002:
                res.slip_ratio = g / max(res.hold_load, 1e-3)
                break
    return res


if __name__ == "__main__":
    o = PhysObject("test", (0.03, 0.03, 0.04), mass=0.4, mu=0.5, f_crush=40)
    print(f"required {o.f_required:.2f} N")
    for f in (2, 4, 6, 10, 45):
        r = run_grasp(o, f)
        print(f, r.success, r.crushed, r.dropped, round(r.max_slip_m, 4),
              None if r.load_liftoff is None else round(r.load_liftoff, 2))
