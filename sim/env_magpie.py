"""Same grasp-and-lift test as sim/env.py, but with the real MAGPIE gripper.

Uses the MAGPIE MuJoCo model from HAMS (CL_Assets/mujoco_assets/magpie_gripper.xml + STL
meshes): two 4-bar linkages driven at the crank (left/right_hinge_1), closed by `connect`
equalities, with 4 pad boxes per finger. Grip force is applied the way the Dynamixel does it,
as crank torque: tau = grip * J(theta), where J = d(pad lateral position)/d(crank angle) is
measured once from the model (virtual work, so `grip` is the normal force per finger, the same
unit env.py uses). Before the timed grasp the fingers pre-shape to just outside the object
under position control, as DeliGrasp does on hardware.

run_grasp() has the same signature, events and GraspResult as sim/env.py.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from sim.env import G, DROP_M, F_MAX_ACTUATOR, LIFT_HEIGHT, SLIP_EVENT_M, GraspResult, PhysObject  # noqa: F401

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "magpie")
PAD_Z = (0.0113, 0.0313, 0.0513, 0.0713)        # pad centres along the finger (HAMS defaults)
PAD_Y, PAD_T = 0.02227, 0.0015
_MESHES = [("mount", "mount.stl"), ("base_bot", "base_bot.stl"), ("base_top", "base_top.stl"),
           ("left_crank", "left_crank.stl"), ("right_crank", "right_crank.stl"),
           ("left_finger", "left_finger_combined.stl"), ("right_finger", "right_finger_combined.stl"),
           ("left_rocker", "left_rocker.stl"), ("right_rocker", "right_rocker.stl")]


def _gripper_body():
    s = open(os.path.join(ASSETS, "magpie_gripper.xml")).read()
    return s[s.index("<body"):]


def _xml(hand_z, obj: PhysObject | None, mu_pad=0.6, transport=False):
    pads = "".join(
        f'<default class="pad_box{i + 1}"><geom mass="0" type="box" pos="0.003 {sgn * PAD_Y} {z}" '
        f'size="0.008 {PAD_T} 0.01" friction="{mu_pad} 0.005 0.0001" priority="2" contype="1" conaffinity="1" '
        f'solimp="0.95 0.99 0.001" solref="0.004 1" rgba="0.55 0.55 0.58 1"/></default>'
        for i, (sgn, z) in enumerate([(-1, z) for z in PAD_Z] + [(1, z) for z in PAD_Z]))
    meshes = "".join(f'<mesh name="magpie_{n}" file="{f}" scale="0.001 0.001 0.001"/>' for n, f in _MESHES)
    objx = ""
    if obj is not None:
        sx, sy, sz = obj.half_size
        objx = (f'<body name="obj" pos="0 0 {sz}"><freejoint/><geom name="obj" type="box" size="{sx} {sy} {sz}" '
                f'mass="{obj.mass}" friction="{obj.mu} 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001"/></body>')
    return f"""
<mujoco model="magpie_lift">
  <compiler angle="radian" meshdir="{ASSETS}" autolimits="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" cone="elliptic" impratio="10" noslip_iterations="3"/>
  <visual><global offwidth="1280" offheight="1280"/></visual>
  <default>
    <default class="magpie">
      <joint damping="0.01" armature="0.002" frictionloss="0.0"/>
      <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.12 0.12 0.13 1"/></default>
      <default class="collision"><geom type="mesh" group="3" contype="0" conaffinity="0"/>{pads}</default>
      <site size="0.0035" type="sphere" rgba="1 0 0 1" group="4"/>
    </default>
  </default>
  <asset>{meshes}</asset>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.6 0.6 0.6"/>
    <geom name="table" type="plane" size="1 1 .05" friction="0.6 0.005 0.0001" rgba="0.82 0.82 0.80 1"/>
    <body name="hand" pos="0 0 {hand_z}">
      <joint name="lift" type="slide" axis="0 0 1" damping="50"/>
      {'<joint name="tx" type="slide" axis="1 0 0" damping="50"/>' if transport else ''}
      <geom type="box" size=".02 .02 .01" pos="0 0 0.01" mass="0.5" contype="0" conaffinity="0" rgba="0.2 0.2 0.22 1"/>
      <body name="flip" euler="3.14159265 0 0" childclass="magpie">{_gripper_body()}</body>
    </body>
    {objx}
  </worldbody>
  <equality>
    <connect body1="base_top" body2="left_rocker" anchor="-0.00075 0.060 0.065" solref="0.005 1" solimp="0.95 0.99 0.001"/>
    <connect body1="base_top" body2="right_rocker" anchor="-0.00075 -0.060 0.065" solref="0.005 1" solimp="0.95 0.99 0.001"/>
  </equality>
  <actuator>
    <position name="lift" joint="lift" kp="4000" ctrlrange="-1 1"/>
    <motor name="ml" joint="left_hinge_1" ctrlrange="-5 5"/>
    <motor name="mr" joint="right_hinge_1" ctrlrange="-5 5"/>
    {'<position name="tx" joint="tx" kp="4000" ctrlrange="-2 2"/>' if transport else ''}
  </actuator>
</mujoco>"""


def _ids(model):
    g = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)       # noqa: E731
    left = {g(f"left_pad{i}") for i in range(1, 5)}
    right = {g(f"right_pad{i}") for i in range(1, 5)}
    j = lambda n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
    jd = lambda n: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
    return left, right, (j("left_hinge_1"), j("right_hinge_1")), (jd("left_hinge_1"), jd("right_hinge_1"))


_TABLE = None


def kinematics():
    """Crank angle -> (inner half-gap between pads, fingertip-pad z below hand). Cached."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    cache = os.path.join(ASSETS, "kinematics.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        _TABLE = (z["th"], z["half_gap"], z["tip_dz"])
        return _TABLE
    model = mujoco.MjModel.from_xml_string(_xml(0.5, None))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    left, right, (ql, qr), (dl, dr) = _ids(model)
    l2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_pad2")
    r2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad2")
    l4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_pad4")
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    th, hg, tz = [], [], []
    for a in np.arange(0.0, 1.16, 0.02):
        for _ in range(1500):
            data.ctrl[1] = 2.0 * (a - data.qpos[ql]) - 0.2 * data.qvel[dl]
            data.ctrl[2] = 2.0 * (-a - data.qpos[qr]) - 0.2 * data.qvel[dr]
            mujoco.mj_step(model, data)
        th.append(data.qpos[ql])
        hg.append(abs(data.geom_xpos[l2][1] - data.geom_xpos[r2][1]) / 2 - PAD_T)
        tz.append(data.geom_xpos[l4][2] - data.xpos[hand][2])
    _TABLE = (np.array(th), np.array(hg), np.array(tz))
    np.savez(cache, th=_TABLE[0], half_gap=_TABLE[1], tip_dz=_TABLE[2])
    return _TABLE


def _angle_for_half_gap(h):
    th, hg, _ = kinematics()
    o = np.argsort(hg)
    return float(np.interp(h, hg[o], th[o]))


def _jac(theta):
    th, hg, _ = kinematics()
    J = np.abs(np.gradient(hg, th))
    return float(np.interp(theta, th, J))


def build(obj: PhysObject, transport=False):
    th, hg, tz = kinematics()
    sy = obj.half_size[1]
    a_pre = _angle_for_half_gap(sy + 0.001)
    a_obj = _angle_for_half_gap(sy)
    tip = float(np.interp(a_obj, th, tz))                  # negative: fingertip pad below hand
    hand_z = max(0.004 + 0.010 - tip, obj.half_size[2] * 2 + 0.02)
    model = mujoco.MjModel.from_xml_string(_xml(hand_z, obj, mu_pad=obj.mu, transport=transport))
    return model, mujoco.MjData(model), a_pre


def _forces(model, data, obj_id, left, right, table_id):
    load, normals, sides, on_table = 0.0, [], set(), False
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if obj_id not in pair:
            continue
        other = (pair - {obj_id}).pop() if len(pair) == 2 else obj_id
        mujoco.mj_contactForce(model, data, i, f6)
        world = c.frame.reshape(3, 3).T @ f6[:3]
        if c.geom1 == obj_id:
            world = -world
        if other == table_id:
            on_table = on_table or abs(f6[0]) > 1e-3
        elif other in left or other in right:
            sides.add("L" if other in left else "R")
            load += world[2]
            normals.append(abs(f6[0]))
    return load, float(np.sum(normals) / 2) if normals else 0.0, len(sides) == 2, on_table


def run_grasp(obj: PhysObject, grip_cmd: float, callback=None,
              close_s=0.3, lift_s=1.0, hold_s=1.0, probe_slip=False, on_step=None, preshape_s=0.4,
              transport=None) -> GraspResult:
    """transport: optional dict(traj, T, ctrl, tau=0.05, delay=0.02) run after the hold, see run_transport."""
    model, data, a_pre = build(obj, transport=transport is not None)
    left, right, (ql, qr), (dl, dr) = _ids(model)
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj")
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    oz0 = obj.half_size[2]

    # pre-shape: position control to just outside the object (not timed, not scored)
    for k in range(int(preshape_s / model.opt.timestep)):
        data.ctrl[1] = 2.0 * (a_pre - data.qpos[ql]) - 0.2 * data.qvel[dl]
        data.ctrl[2] = 2.0 * (-a_pre - data.qpos[qr]) - 0.2 * data.qvel[dr]
        mujoco.mj_step(model, data)
        if on_step is not None and k % 2 == 0:
            on_step(model, data, {"k": -1, "grip": 0.0, "crushed": False, "lifted": False, "slipped": False, "load": 0.0})

    grip = float(np.clip(grip_cmd, 0, F_MAX_ACTUATOR))
    res = GraspResult(False, False, False, 0.0, grip, grip, 0, None, None, None)
    contact_seen = lifted = slip_fired = False
    hz0 = None
    hold_loads, reflexed = [], set()

    def apply(newf, why):
        nonlocal grip
        if newf is None:
            return
        newf = float(np.clip(newf, 0, F_MAX_ACTUATOR))
        if newf > grip + 1e-6 and why not in reflexed:
            reflexed.add(why)
            res.n_reflex += 1
        grip = newf
        res.events.append((why, round(data.time, 3), round(grip, 2)))

    def drive(f):
        data.ctrl[1] = f * _jac(data.qpos[ql])
        data.ctrl[2] = -f * _jac(-data.qpos[qr])

    n_close, n_lift, n_hold = (int(s / model.opt.timestep) for s in (close_s, lift_s, hold_s))
    for k in range(n_close + n_lift + n_hold):
        if k < n_close:
            data.ctrl[0] = 0.0
            drive(grip * min(1.0, k / (0.1 / model.opt.timestep)))
        else:
            data.ctrl[0] = LIFT_HEIGHT * min(1.0, (k - n_close) / n_lift)
            drive(grip)
        mujoco.mj_step(model, data)

        load, _, both, _ = _forces(model, data, obj_id, left, right, table_id)
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
                slip_fired = res.slipped = True
                if callback:
                    apply(callback("slip", {"slip_m": slip, "grip_N": grip}), "slip")
        if on_step is not None:
            on_step(model, data, {"k": k, "grip": grip, "crushed": res.crushed, "lifted": lifted,
                                  "slipped": res.slipped, "load": load})

    if transport is not None and lifted:
        res.transport = run_transport(model, data, obj, obj_id, left, right, table_id, ql, qr, grip,
                                      on_step=on_step, **transport)
    elif transport is not None:
        res.transport = None
    final_oz = data.body("obj").xpos[2] if transport is None else oz0 + LIFT_HEIGHT
    res.dropped = (not lifted) or res.max_slip_m > DROP_M or final_oz < oz0 + 0.5 * LIFT_HEIGHT
    res.success = not res.crushed and not res.dropped
    if hold_loads:
        res.hold_load = float(np.mean(hold_loads))
    if probe_slip and res.success:
        rel0 = data.body("hand").xpos[2] - data.body("obj").xpos[2]
        g = grip
        while g > 0.05:
            g *= 0.995
            drive(g)
            for _ in range(5):
                mujoco.mj_step(model, data)
            rel = data.body("hand").xpos[2] - data.body("obj").xpos[2]
            if rel - rel0 > 0.002:
                res.slip_ratio = g / max(res.hold_load, 1e-3)
                break
    return res


if __name__ == "__main__":
    th, hg, tz = kinematics()
    print("half-gap range", hg.min().round(4), hg.max().round(4), "J", np.abs(np.gradient(hg, th))[[0, 20, 40]].round(4))
    o = PhysObject("test", (0.03, 0.03, 0.04), mass=0.4, mu=0.5, f_crush=40)
    print(f"required {o.f_required:.2f} N")
    for f in (2, 3.5, 4, 6, 10, 45):
        r = run_grasp(o, f)
        print(f, r.success, r.crushed, r.dropped, round(r.max_slip_m, 4),
              None if r.load_liftoff is None else round(r.load_liftoff, 2), None if r.hold_load is None else round(r.hold_load, 2))


def min_jerk(d, T):
    """Position, velocity, acceleration of a minimum-jerk move of length d in time T."""
    def f(t):
        u = np.clip(t / T, 0.0, 1.0)
        p = d * (10 * u**3 - 15 * u**4 + 6 * u**5)
        v = d / T * (30 * u**2 - 60 * u**3 + 30 * u**4) if 0 < t < T else 0.0
        a = d / T**2 * (60 * u - 180 * u**2 + 120 * u**3) if 0 < t < T else 0.0
        return p, v, a
    return f


def transport_traj(T, dx=0.30, dz_up=0.08):
    """Carry-and-place, the kind of motion a policy does after the grasp: up 8 cm and over 30 cm
    together in T, then down 8 cm to place in 0.6 T, then a 0.3 s settle. Returns (x, z, ax, az)(t)
    and total duration. Peak accelerations scale with 1/T^2."""
    mx, mz, md = min_jerk(dx, T), min_jerk(dz_up, T), min_jerk(-dz_up, 0.6 * T)

    def traj(t):
        x, _, ax = mx(t)
        z, _, az = mz(t)
        if t > T:
            z2, _, az2 = md(t - T)
            z, az = z + z2, az2
        return x, z, ax, az
    return traj, 1.6 * T + 0.3


def run_transport(model, data, obj, obj_id, left, right, table_id, ql, qr, grip0, traj, T, ctrl,
                  tau=0.05, delay=0.02, lead=None, on_step=None):
    """Move the grasped object along traj while ctrl(t, info) sets the grip command.
    The gripper follows its command with a first-order lag (tau) and the controller sees the felt
    load `delay` s late, as on the real robot. info holds the delayed felt load vector, the current
    grip, and the PLANNED hand acceleration `lead` s ahead (the efference copy of the policy's plan).
    Returns metrics: held, crushed, in-hand shift (m), peak grip and peak needed grip."""
    dt = model.opt.timestep
    lead = tau + delay if lead is None else lead
    z0, x0 = data.ctrl[0], 0.0
    hand, objb = data.body("hand"), data.body("obj")
    rel0 = objb.xpos - hand.xpos
    rel0 = rel0.copy()
    grip_act, buf, abuf = grip0, [], []
    peak_grip, peak_need, crushed, shift_max = grip0, 0.0, False, 0.0
    sum_grip = sum_need = 0.0
    n = int(T / dt)
    f6 = np.zeros(6)
    jx = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tx")]
    jz = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift")]
    for k in range(n):
        t = k * dt
        x, z, _, _ = traj(t)
        data.ctrl[0] = z0 + z
        data.ctrl[3] = x0 + x
        # felt load vector on the object from the fingers (world frame)
        lv = np.zeros(3)
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {c.geom1, c.geom2}
            if obj_id not in pair or not (pair & (left | right)):
                continue
            mujoco.mj_contactForce(model, data, i, f6)
            w = c.frame.reshape(3, 3).T @ f6[:3]
            lv += -w if c.geom1 == obj_id else w
        buf.append(lv)
        abuf.append((data.qacc[jx], data.qacc[jz]))
        felt = buf[max(0, len(buf) - 1 - int(delay / dt))]
        a_meas = np.array(abuf[max(0, len(abuf) - 1 - int(delay / dt))])
        a_prev = np.array(abuf[max(0, len(abuf) - 1 - int(delay / dt) - 20)])
        _, _, axp, azp = traj(t + lead)
        info = {"t": t, "load_vec": felt, "grip": grip_act, "a_plan": np.array([axp, 0.0, azp]),
                "a_meas": a_meas, "a_meas_prev": a_prev, "lead": lead, "delay": delay, "traj": traj}
        cmd = float(np.clip(ctrl(t, info), 0.0, F_MAX_ACTUATOR))
        grip_act += (cmd - grip_act) * dt / tau
        data.ctrl[1] = grip_act * _jac(data.qpos[ql])
        data.ctrl[2] = -grip_act * _jac(-data.qpos[qr])
        mujoco.mj_step(model, data)
        tang = np.hypot(lv[0], lv[2])                    # felt load the fingers hold by friction
        # grip the object needs right now, from the hand's actual acceleration: m |g + a| / 2mu
        need = obj.mass * np.hypot(data.qacc[jx], G + data.qacc[jz]) / (2 * obj.mu)
        peak_need = max(peak_need, need)
        sum_grip += grip_act
        sum_need += need
        peak_grip = max(peak_grip, grip_act)
        crushed = crushed or grip_act > obj.f_crush
        shift_max = max(shift_max, float(np.linalg.norm((objb.xpos - hand.xpos) - rel0)))
        if on_step is not None:
            on_step(model, data, {"k": 10_000 + k, "grip": grip_act, "crushed": crushed, "lifted": True,
                                  "slipped": shift_max > SLIP_EVENT_M, "load": tang, "x": x, "need": need})
        if shift_max > 0.03:                               # fell out of the hand
            break
    shift_end = float(np.linalg.norm((objb.xpos - hand.xpos) - rel0))
    held = shift_max <= 0.03
    return dict(held=held, crushed=crushed, shift_mm=1000 * shift_end, shift_max_mm=1000 * shift_max,
                peak_grip=peak_grip, peak_need=float(peak_need), mean_grip_ratio=float(sum_grip / max(sum_need, 1e-9)))
