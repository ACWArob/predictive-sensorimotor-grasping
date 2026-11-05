"""After the grasp: does the grip survive what the policy does next? (MAGPIE model)

Each trial = the normal grasp-and-lift, then a carry-and-place move like a policy makes after the
grasp (up 8 cm and over 30 cm, then down to place), at three speeds. The gripper follows its command
with a 50 ms lag and the controller feels the load 20 ms late, as on the robot.

Grip during the move, per system:
  v1_hold         DeliGrasp grip, held fixed; slip reflex x1.4                  (today)
  v4_hold         v4 grasp (prediction + memory + coupling at lift), then held fixed + slip reflex
  v4_reactive     v4, then grip tracks the FELT load (feedback; sees it late, gripper lags)
  v5_ff_no_memory v4, then grip set from the PLANNED acceleration of the move: m_hat |g + a_plan| x ratio
                  x k, ahead by the lag (efference copy, Flanagan & Wing 1997), max'd with feedback; k = 1.4 fixed
  v5_chunk        v5_warm, but it only sees the policy's current 0.5 s action chunk (VLA-style), not the whole move
  v5_learned_hold v5_warm's learned margin, but one constant grip during the move (no prediction of the move)
  v5_extrap       v5_warm with NO plan: predicts the policy's next accelerations from its own recent motion
  v5_warm         v5 whose skill-level k was first learned on 20 Phase-0 training objects (carried, not scored)
  v5_feedforward  the same, but k is LEARNED at two levels: per object (x1.3 after a slip in the move,
                  slowly relaxed when grip is far above need) and per skill (all objects; x1.1 after any
                  slip), so the policy phase feeds the memory, not just the lift

Task success = lifted, carried without dropping, not crushed, and the object ends < 5 mm from where
it sat in the hand, so the next policy step (place, insert, pour) starts from the pose it expects.

    python -m harness.transport_test --seeds 3 --jobs 2  ->  harness/out_transport/
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness.compare_autograsp as H                                    # noqa: E402
from psg import ForwardModel, PredictiveGraspAgent                       # noqa: E402
from sim import env_magpie as E                                          # noqa: E402
from sim.baselines import PRIOR, DeliGraspLike                           # noqa: E402
from sim.env import G                                                    # noqa: E402
from sim.objects import percept                                          # noqa: E402
from sim.run_experiments import collect, f_need                          # noqa: E402

SPEEDS = {"medium": 0.8, "fast": 0.6}
if os.environ.get("TT_MIXED"):
    SPEEDS = {"mixed": None}     # every trial a different move: T in [0.45, 1.3] s, dx in [0.1, 0.35] m
SYSTEMS = ["v1_hold", "v4_hold", "v4_reactive", "v5_ff_no_memory", "v5_feedforward", "v5_warm"]
if os.environ.get("TT_SYSTEMS"):
    SYSTEMS = os.environ["TT_SYSTEMS"].split(",")
SUITES = ["S1_honest_unseen", "S2_deceptive", "S4_fragile"]
OUT = os.environ.get("TT_OUT", os.path.join(ROOT, "harness", "out_transport"))
MIN_GRIP, MARGIN, SLIP_MM = 0.3, 1.4, 3.0
CHUNK = 0.5          # s, action-chunk length of a VLA-style policy (pi0 / ACT style)


def make_ctrl(kind, grip0, ratio, k=MARGIN):
    """Grip controller for the move. ratio = estimated grip/load ratio (1/2mu)."""
    st = {"grip": grip0, "reflexed": False, "m": None, "slipped": False}

    def ctrl(t, info):
        lv, a = info["load_vec"], info["a_plan"]
        felt = float(np.hypot(lv[0], lv[2]))
        if st["m"] is None:                                   # mass from the felt load at the start of the move
            st["m"] = max(float(np.linalg.norm(lv)) / G, 1e-3)
        g = st["grip"]
        if kind == "hold_learned":
            # no prediction of the move at all: one constant grip from the learned margin
            g = max(MIN_GRIP, st["m"] * G * ratio * k)
        elif kind == "reactive":
            g = max(MIN_GRIP, felt * ratio * MARGIN)
        elif kind.startswith("feedforward"):
            if kind == "feedforward_chunk":
                # a VLA commits to action chunks; the plan is known inside the current chunk only
                t_end = (np.floor(t / CHUNK) + 1) * CHUNK
                if t + info["lead"] > t_end:
                    _, _, ax_, az_ = info["traj"](t_end - 1e-3)       # last planned accel of this chunk
                    a = np.array([ax_, 0.0, az_])
            elif kind == "feedforward_extrap":
                # no plan at all: predict the policy's motion from its own recent motion (felt late)
                am, ap = info["a_meas"], info["a_meas_prev"]
                jerk = (am - ap) / 0.02
                ahead = info["lead"] + info["delay"]
                a2 = np.clip(am + jerk * ahead, -30, 30)
                a = np.array([a2[0], 0.0, a2[1]])
            ff = st["m"] * float(np.hypot(a[0], G + a[2])) * ratio * k
            g = max(MIN_GRIP, ff, felt * ratio * k)
        # slip reflex, same for everyone: the in-hand shift is visible as the felt load collapsing
        return g * (1.4 if st["reflexed"] else 1.0)

    def on_shift():
        st["reflexed"] = True
    ctrl.on_shift, ctrl.state = on_shift, st
    return ctrl


def run_trial(name, system, obj, rng, T, dx=0.30):
    pc = percept(obj, rng)
    traj, Ttot = E.transport_traj(T, dx=dx)
    if name.startswith("v1"):
        f0, cb = system.plan(obj, pc)
        dens, mu_prior = PRIOR[obj.material]
        ratio = 1 / (2 * mu_prior)
        kind = "feedforward" if name == "v1_feedforward" else "hold"
        pred = cmp = None
    else:
        pred, cmp = system.plan(pc)
        f0, cb = pred.grip_cmd, cmp
        ratio = float(np.exp(pred.mean[1]))
        kind = {"v4_hold": "hold", "v4_reactive": "reactive", "v5_ff_no_memory": "feedforward",
                "v5_feedforward": "feedforward", "v5_warm": "feedforward",
                "v5_chunk": "feedforward_chunk", "v5_extrap": "feedforward_extrap",
                "v5_learned_hold": "hold_learned"}[name]
    k = MARGIN
    if name in ("v5_feedforward", "v5_warm", "v5_chunk", "v5_extrap", "v5_learned_hold"):
        mem = system.__dict__.setdefault("tmem", {"skill": MARGIN, "obj": {}})
        k = mem["obj"].setdefault(pc.obj_id, mem["skill"])
    box = {}

    def ctrl(t, info, box=box):
        if "c" not in box:
            box["c"] = make_ctrl(kind, info["grip"], ratio, k)
        c = box["c"]
        # slip reflex trigger from the in-hand shift seen through the felt load (delayed); here simply
        # from the shift the sim reports via info when available
        return c(t, info)

    def on_step(m, d, info, box=box):
        if info["k"] >= 10_000 and info["slipped"] and "c" in box and not box["c"].state["reflexed"]:
            box["c"].on_shift()
            box["slip"] = True
    res = E.run_grasp(obj, f0, cb, transport=dict(traj=traj, T=Ttot, ctrl=ctrl), on_step=on_step)
    if isinstance(system, PredictiveGraspAgent):
        system.observe(pc, pred, cmp, res.success, res.crushed, res.grip_max)
        if name in ("v5_feedforward", "v5_warm", "v5_chunk", "v5_extrap", "v5_learned_hold") and res.success:            # memory learns from the policy phase
            mem, tr_ = system.tmem, (res.transport or {})
            slipped = box.get("slip") or not tr_.get("held", False)
            if slipped:
                mem["obj"][pc.obj_id] = min(k * 1.3, 6.0)
                mem["skill"] = min(mem["skill"] * 1.1, 4.0)
            elif tr_.get("peak_grip", 0) > 2.5 * max(tr_.get("peak_need", 1e-6), 1e-6):
                mem["obj"][pc.obj_id] = max(k * 0.95, 1.2)
    else:
        system.observe(obj, pc, f0, res)
    tr = res.transport or {}
    need0 = f_need(obj)
    held = bool(res.success and tr.get("held", False))
    crushed = bool(res.crushed or tr.get("crushed", False))
    shift = tr.get("shift_mm", np.nan)
    return dict(obj=obj.name, lifted=int(res.success), held=int(held), crushed=int(crushed),
                shift_mm=round(float(shift), 2) if held else np.nan,
                rot_deg=round(float(tr.get("rot_deg", np.nan)), 1) if held else np.nan,
                task_ok=int(held and not crushed and shift < 5.0 and tr.get("rot_deg", 0.0) < 10.0),
                grip_lift=round(res.grip_max, 3), peak_grip=round(tr.get("peak_grip", np.nan), 3),
                peak_need=round(tr.get("peak_need", np.nan), 3),
                move_excess=round(tr.get("peak_grip", np.nan) / max(tr.get("peak_need", np.nan), 1e-6), 3) if tr else np.nan,
                static_need=round(need0, 3), reflex=int(bool(box.get("slip"))),
                mean_grip_ratio=round(tr.get("mean_grip_ratio", np.nan), 3) if tr else np.nan, move_T=T, move_dx=dx)


def run_seed(seed):
    H.use_gripper("magpie")
    rows, t0 = [], time.time()
    X, Y, vrows, _ = collect(np.random.default_rng(seed))
    model = ForwardModel(seed=seed).fit(X, Y)
    suites = H.suites(seed)
    for speed, T in SPEEDS.items():
        for suite in SUITES:
            trials = suites[suite]
            for name in SYSTEMS:
                if name.startswith("v1"):
                    system = DeliGraspLike()
                else:
                    system = PredictiveGraspAgent(model, mode="predictor+update", couple=True)
                if name in ("v5_warm", "v5_chunk", "v5_extrap", "v5_learned_hold"):                         # learn the skill-level k on Phase-0 objects first
                    from sim.objects import train_set
                    wrng = np.random.default_rng(seed * 1000 + 99)
                    for wobj in train_set(np.random.default_rng(seed * 1000 + 98), n=20):
                        run_trial(name, system, wobj, wrng, T if T is not None else float(wrng.uniform(0.45, 1.3)),
                                  0.30 if T is not None else float(wrng.uniform(0.10, 0.35)))
                    system.tmem["obj"] = {}
                    system.store = type(system.store)()
                rng = np.random.default_rng(seed * 7 + zlib.crc32(suite.encode()) % 1000)
                mrng = np.random.default_rng(seed * 31 + zlib.crc32(suite.encode()) % 1000)   # same moves for all systems
                for i, (obj, k) in enumerate(trials):
                    Ti, dxi = (T, 0.30) if T is not None else (float(mrng.uniform(0.45, 1.3)), float(mrng.uniform(0.10, 0.35)))
                    rows.append(dict(seed=seed, speed=speed, suite=suite, system=name, trial=i, rep=k,
                                     **run_trial(name, system, obj, rng, Ti, dxi)))
    print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    return rows


def report(rows):
    import pandas as pd
    d = pd.DataFrame(rows)
    out = ["# After the grasp: carry-and-place on the MAGPIE model", "",
           f"{d.seed.nunique()} seeds; every system gets the same objects, order, noise, physics and move. "
           "Task success = held through the move, not crushed, object < 5 mm from its start pose in the hand.", ""]
    for suite in SUITES:
        out += [f"## {suite}", "", "| System | " + " | ".join(f"{s} task success" for s in SPEEDS) +
                " | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |",
                "|---" * (len(SPEEDS) + 5) + "|"]
        for name in SYSTEMS:
            x = d[(d.suite == suite) & (d.system == name)]
            f = x[x.speed == "fast"]
            cells = [f"{x[x.speed == s].task_ok.mean():.0%}" for s in SPEEDS]
            out.append(f"| {name} | " + " | ".join(cells) +
                       f" | {int((f.lifted & ~f.held.astype(bool)).sum())} | {int(f.crushed.sum())} | "
                       f"{f.shift_mm.median():.1f} | {f.move_excess.median():.2f}× |")
        out.append("")
    s2 = d[(d.suite == "S2_deceptive") & (d.speed == "fast")]
    out += ["## Deceptive objects, fast move: task success by grasp number", "",
            "| grasp | " + " | ".join(SYSTEMS) + " |", "|---" * (len(SYSTEMS) + 1) + "|"]
    for k in range(5):
        out.append(f"| {k + 1} | " + " | ".join(f"{s2[(s2.system == s) & (s2.rep == k)].task_ok.mean():.0%}" for s in SYSTEMS) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs) as p:
            parts = p.map(run_seed, range(a.seeds))
    else:
        parts = [run_seed(s) for s in range(a.seeds)]
    rows = [r for p in parts for r in p]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    try:
        rep = report(rows)
    except Exception as e:                                  # mixed-move runs have no 'fast' column
        rep = f"(summary skipped: {e})\n"
    open(os.path.join(OUT, "report.md"), "w").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
