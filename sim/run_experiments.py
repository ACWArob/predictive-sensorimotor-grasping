"""Run every comparison in MuJoCo and write results/ (CSV + plots).

    python -m sim.run_experiments --seeds 3

Phase 0  collect logged grasps on honest training objects (random force around a prior) + slip probes
Phase 1  fit the forward model on control-point labels; warm-start value head and Exp-Force memory
Exp C    30 unseen honest objects, one grasp each       (Exp-Force's metric: force error, % appropriate)
Exp A    6 deceptive objects x 5 interleaved rounds     (size-weight protocol: does it fix itself by grasp 2?)
Exp B    bottle_empty x3 -> bottle_full x3 -> bottle_empty x3   (does a correction leak to a look-alike?)
Ablation all of the above per mode
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psg import ForwardModel, PredictiveGraspAgent                       # noqa: E402
from sim.baselines import PRIOR, DeliGraspLike, ExpForceLike             # noqa: E402
from sim.env import G, run_grasp                                         # noqa: E402
from sim.objects import deceptive_set, heldout_set, percept, train_set   # noqa: E402

LIFT_ACC = 1.33        # measured: lift-off load / (m g) with this controller
PROBE_BIAS = 0.92      # measured: probed slip ratio / (1 / 2mu) with these contacts
METHODS = ["deligrasp", "expforce", "predictor", "predictor+update", "full", "full-nogate"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def f_need(obj):
    return obj.mass * G * LIFT_ACC * PROBE_BIAS / (2 * obj.mu)


# ---------------------------------------------------------------- Phase 0 / 1
def collect(rng, n_obj=48, grasps_per_obj=4):
    X, Y, vrows, erows = [], [], [], []
    for obj in train_set(rng, n_obj):
        dens, mu = PRIOR[obj.material]
        prior_f = dens * 8 * np.prod(obj.half_size) * G * LIFT_ACC / (2 * mu)
        ratio = np.nan
        pc = percept(obj, rng)
        probe = run_grasp(obj, min(3 * prior_f + 1, 60), probe_slip=True)
        if probe.slip_ratio:
            ratio = np.log(probe.slip_ratio)
        if probe.load_liftoff:
            X.append(pc.x()); Y.append([np.log(probe.load_liftoff), ratio])
        for _ in range(grasps_per_obj):
            pc = percept(obj, rng)
            f = prior_f * rng.uniform(0.5, 3.0)
            r = run_grasp(obj, f)
            if r.load_liftoff:
                X.append(pc.x()); Y.append([np.log(r.load_liftoff), ratio])
            vrows.append((pc.x(), f, r.success and not r.crushed))
            if r.success and not r.crushed:
                erows.append((pc.embedding, f))
    return np.array(X), np.array(Y), vrows, erows


def build(method, model, vrows, erows):
    if method == "deligrasp":
        return DeliGraspLike()
    if method == "expforce":
        m = ExpForceLike(); m.seed(erows); return m
    agent = PredictiveGraspAgent(model, mode=method)
    if method.startswith("full"):
        rows = []
        for x, f, ok in vrows:
            mu, _ = model.predict(x)
            rows.append((x, f / np.exp(mu[0].sum()), ok))
        agent.pretrain_value_head(rows)
    return agent


def grasp(method, obj, rng):
    pc = percept(obj, rng)
    pred_load = None
    if isinstance(method, PredictiveGraspAgent):
        pred, cmp = method.plan(pc)
        f0 = pred.grip_cmd
        res = run_grasp(obj, f0, cmp)
        method.observe(pc, pred, cmp, res.success, res.crushed, res.grip_max)
        pred_load = pred.load
    else:
        f0, cb = method.plan(obj, pc)
        res = run_grasp(obj, f0, cb)
        method.observe(obj, pc, f0, res)
    need = f_need(obj)
    return dict(obj=obj.name, f_init=f0, f_need=need, force_ratio=f0 / need,
                appropriate=int(need <= f0 <= min(2.5 * need, obj.f_crush)),
                success=int(res.success), clean=int(res.success and not res.slipped), slipped=int(res.slipped),
                crushed=int(res.crushed), dropped=int(res.dropped), n_reflex=res.n_reflex,
                pred_load=pred_load, true_load=res.load_liftoff)


# ---------------------------------------------------------------- experiments
def run_seed(seed):
    rng = np.random.default_rng(seed)
    X, Y, vrows, erows = collect(rng)
    model = ForwardModel(seed=seed).fit(X, Y)
    held = heldout_set(np.random.default_rng(seed + 100))
    dec = deceptive_set(np.random.default_rng(seed + 200))
    dobj = {o.name: o for o, _ in dec}
    rows = []
    for name in METHODS:
        m = build(name, model, vrows, erows)
        for o in held:
            rows.append(dict(seed=seed, method=name, exp="C_heldout", trial=0, **grasp(m, o, rng)))
        m = build(name, model, vrows, erows)
        for rnd in range(5):
            for o, _ in dec:
                rows.append(dict(seed=seed, method=name, exp="A_deceptive", trial=rnd, **grasp(m, o, rng)))
        m = build(name, model, vrows, erows)
        seq = ["bottle_empty"] * 3 + ["bottle_full"] * 3 + ["bottle_empty"] * 3
        for i, n in enumerate(seq):
            rows.append(dict(seed=seed, method=name, exp="B_noleak", trial=i, **grasp(m, dobj[n], rng)))
    return rows, (X, Y)


def summarize(rows):
    import collections
    out = []
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["method"], r["exp"])].append(r)
    for (m, e), rs in sorted(by.items()):
        out.append(dict(method=m, exp=e, n=len(rs),
                        success=np.mean([r["success"] for r in rs]),
                        clean=np.mean([r["clean"] for r in rs]),
                        crushed=np.mean([r["crushed"] for r in rs]),
                        appropriate=np.mean([r["appropriate"] for r in rs]),
                        force_mae_N=np.mean([abs(r["f_init"] - r["f_need"]) for r in rs]),
                        reflexes=np.mean([r["n_reflex"] for r in rs])))
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    all_rows, t0 = [], time.time()
    for s in range(a.seeds):
        rows, _ = run_seed(s)
        all_rows += rows
        print(f"seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    write_csv(os.path.join(OUT, "grasps.csv"), all_rows)
    summ = summarize(all_rows)
    write_csv(os.path.join(OUT, "summary.csv"), summ)
    for r in summ:
        print(f"{r['exp']:12s} {r['method']:17s} success {r['success']:.2f} clean {r['clean']:.2f} "
              f"crush {r['crushed']:.2f} approp {r['appropriate']:.2f} MAE {r['force_mae_N']:.2f}N "
              f"reflex {r['reflexes']:.2f}")
    from sim.plots import make_plots
    make_plots(all_rows, OUT)


if __name__ == "__main__":
    main()
