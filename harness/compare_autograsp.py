"""AutoGrasp v1 (DeliGrasp-style) vs v2 (PSG) paired test harness. See harness/TEST_PLAN.md.

    python -m harness.compare_autograsp --seeds 5
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

from psg import ForwardModel, PredictiveGraspAgent                       # noqa: E402
from sim.baselines import DeliGraspLike                                  # noqa: E402
from sim.env import run_grasp                                            # noqa: E402
from sim.objects import deceptive_set, heldout_set, make, percept, random_size  # noqa: E402
from sim.run_experiments import collect, f_need                          # noqa: E402

OUT = os.path.join(ROOT, "harness", "out")
SYSTEMS = {"v1_deligrasp": None, "v2_psg": "full", "v3_psg_couple": "full",
           "v2a_no_value_head": "predictor+update", "v2b_excess_reward": "full",
           # ablation ladder: which part of v3 does the work?
           "a1_v1_plus_coupling": "predictor",      # DeliGrasp prior + coupling; no learned model, no memory
           "a2_predict_only": "predictor",          # learned forward model; no coupling, no memory
           "a3_predict_couple": "predictor",        # forward model + coupling; no memory
           "a4_predict_memory": "predictor+update", # forward model + one-shot memory; no coupling (= v2a)
           "v4_no_value_head": "predictor+update"}  # v3 without the value head (forward model + memory + coupling)
COUPLED = {"v3_psg_couple", "a1_v1_plus_coupling", "a3_predict_couple", "v4_no_value_head"}
import os as _os
if _os.environ.get("HARNESS_SYSTEMS"):
    SYSTEMS = {k: SYSTEMS[k] for k in _os.environ["HARNESS_SYSTEMS"].split(",")}


def fragile_set(rng):
    """Look sturdy, crush easily."""
    return [make("fragile_thin_cup", "ceramic", (0.03, 0.03, 0.045), rng, density=400, crush=4.0),
            make("fragile_foam_block", "wood", (0.03, 0.03, 0.03), rng, density=120, crush=3.0),
            make("fragile_shell", "metal", (0.025, 0.025, 0.04), rng, density=300, crush=5.0)]


class PriorModel:
    """DeliGrasp's semantic prior dressed as a forward model, so v1 can get grip-load coupling
    without any learned prediction: load = m_prior * g * 1.33, ratio = 1 / (2 mu_prior), and an
    std chosen so the default margin equals DeliGrasp's 1.5x."""
    SD = np.log(1.5 / 1.25) / np.sqrt(2)

    def __init__(self):
        self.obj = None

    def predict(self, x):
        from sim.baselines import PRIOR
        from sim.env import G
        dens, mu = PRIOR[self.obj.material]
        m = dens * 8 * np.prod(self.obj.half_size)
        mean = np.array([[np.log(m * G * 1.33), np.log(1 / (2 * mu))]])
        return mean, np.full((1, 2), self.SD)


def new_system(name, model, vrows):
    if name == "v1_deligrasp":
        return DeliGraspLike()
    from psg.value_head import ValueHead
    vh = ValueHead(excess_cost=0.3) if name == "v2b_excess_reward" else None
    if name == "a1_v1_plus_coupling":
        return PredictiveGraspAgent(PriorModel(), mode="predictor", couple=True)
    agent = PredictiveGraspAgent(model, mode=SYSTEMS[name], couple=name in COUPLED, value_head=vh)
    rows = [(x, f / np.exp(model.predict(x)[0][0].sum()), ok) for x, f, ok in vrows]
    agent.pretrain_value_head(rows)
    return agent


def do_grasp(system, obj, rng):
    pc = percept(obj, rng)
    if isinstance(system, PredictiveGraspAgent):
        if isinstance(system.model, PriorModel):
            system.model.obj = obj
        pred, cmp = system.plan(pc)
        f0 = pred.grip_cmd
        res = run_grasp(obj, f0, cmp)
        system.observe(pc, pred, cmp, res.success, res.crushed, res.grip_max)
    else:
        f0, cb = system.plan(obj, pc)
        res = run_grasp(obj, f0, cb)
        system.observe(obj, pc, f0, res)
    need = f_need(obj)
    return dict(obj=obj.name, f_init=round(f0, 3), f_need=round(need, 3), excess=round(f0 / need, 3),
                grip_max=round(res.grip_max, 3), peak_excess=round(res.grip_max / need, 3), success=int(res.success),
                clean=int(res.success and not res.slipped), crushed=int(res.crushed),
                dropped=int(res.dropped), n_reflex=res.n_reflex)


def suites(seed):
    r = lambda k: np.random.default_rng(seed * 1000 + k)                 # noqa: E731
    honest = heldout_set(r(1))
    dec = [o for o, _ in deceptive_set(r(2))]
    d = {o.name: o for o in dec}
    frag = fragile_set(r(3))
    rep = heldout_set(r(4), n=10)
    return {
        "S1_honest_unseen": [(o, 0) for o in honest],
        "S2_deceptive": [(o, k) for k in range(5) for o in dec],
        "S3_lookalike_leak": [(d[n], i) for i, n in enumerate(["bottle_empty"] * 3 + ["bottle_full"] * 3 + ["bottle_empty"] * 3)],
        "S4_fragile": [(o, k) for k in range(3) for o in frag],
        "S5_repeat_honest": [(o, k) for k in range(5) for o in rep],
    }


def use_gripper(name):
    """'box' = sim/env.py parallel jaw; 'magpie' = real MAGPIE 4-bar model (sim/env_magpie.py).
    Swaps the simulator for both Phase-0 data collection and the test grasps."""
    global run_grasp
    if name == "magpie":
        import sim.run_experiments as RE
        from sim import env_magpie
        env_magpie.kinematics()
        RE.run_grasp = run_grasp = env_magpie.run_grasp


def run_seed(seed, gripper="box"):
    use_gripper(gripper)
    rows, t0 = [], time.time()
    X, Y, vrows, _ = collect(np.random.default_rng(seed))
    model = ForwardModel(seed=seed).fit(X, Y)
    for suite, trials in suites(seed).items():
        for name in SYSTEMS:
            system = new_system(name, model, vrows)            # fresh memory per suite
            rng = np.random.default_rng(seed * 7 + zlib.crc32(suite.encode()) % 1000)   # identical percept noise for both
            for i, (obj, k) in enumerate(trials):
                rows.append(dict(seed=seed, suite=suite, system=name, trial=i, rep=k, **do_grasp(system, obj, rng)))
    print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    return rows


def run(seeds, gripper="box", jobs=1):
    if jobs > 1:
        from functools import partial
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            parts = pool.map(partial(run_seed, gripper=gripper), range(seeds))
    else:
        parts = [run_seed(s, gripper) for s in range(seeds)]
    return [r for p in parts for r in p]


def boot_ci(diff, n=2000, rng=np.random.default_rng(0)):
    diff = np.asarray(diff, float)
    if len(diff) == 0:
        return (np.nan, np.nan)
    b = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


SUITES = ["S1_honest_unseen", "S2_deceptive", "S3_lookalike_leak", "S4_fragile", "S5_repeat_honest"]


def report_pair(rows, base, chal):
    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]
    med = lambda rs, k: float(np.median([r[k] for r in rs]))             # noqa: E731
    lines = [f"## {chal} vs {base}", "",
             "| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    v = {}
    for s in SUITES:
        a, b = sel(suite=s, system=base), sel(suite=s, system=chal)
        key = lambda r: (r["seed"], r["trial"])                          # noqa: E731
        A, Bm = {key(r): r for r in a}, {key(r): r for r in b}
        diff = [Bm[k]["success"] - A[k]["success"] for k in A]
        lo, hi = boot_ci(diff)
        sa, sb = np.mean([r["success"] for r in a]), np.mean([r["success"] for r in b])
        ca, cb = sum(r["crushed"] for r in a), sum(r["crushed"] for r in b)
        lines.append(f"| {s} | {sa:.0%} | {sb:.0%} | {sb - sa:+.0%} ({lo:+.0%}, {hi:+.0%}) | "
                     f"{med(a, 'excess'):.2f}× / {med(a, 'peak_excess'):.2f}× | {med(b, 'excess'):.2f}× / {med(b, 'peak_excess'):.2f}× | "
                     f"{ca} | {cb} | {sum(d == 1 for d in diff)} | {sum(d == -1 for d in diff)} |")
        v[s] = dict(sa=sa, sb=sb, lo=lo, ca=ca, cb=cb, a=a, b=b)
    late = lambda rs: float(np.median([r["peak_excess"] for r in rs if r["trial"] >= 6]))   # noqa: E731
    s5b = v["S5_repeat_honest"]["b"]
    g1 = np.mean([r["success"] for r in s5b if r["rep"] == 0])
    gl = np.mean([r["success"] for r in s5b if r["rep"] > 0])
    s1 = v["S1_honest_unseen"]
    checks = [
        ("1. S1 no regression (≥ base − 2 pp), peak force closer to 1.0", s1["sb"] >= s1["sa"] - 0.02 and abs(np.log(med(s1["b"], "peak_excess"))) < abs(np.log(med(s1["a"], "peak_excess")))),
        ("2. S2 ≥ base + 20 pp, CI excludes 0", v["S2_deceptive"]["sb"] - v["S2_deceptive"]["sa"] >= 0.20 and v["S2_deceptive"]["lo"] > 0),
        ("3. S3 no leak (median peak excess on 2nd empty block ≤ 2.5×)", late(v["S3_lookalike_leak"]["b"]) <= 2.5),
        ("4. S4 crushes ≤ base", v["S4_fragile"]["cb"] <= v["S4_fragile"]["ca"]),
        ("5. S5 repeats never worse than first try", gl >= g1),
    ]
    lines += ["", "Pre-registered checks:", ""] + [f"- {'PASS' if ok else 'FAIL'}: {n}" for n, ok in checks]
    lines += ["", f"S5: grasp 1 {g1:.1%}, grasps 2–5 {gl:.1%}.", "", "S2 success by grasp number:", "", "| grasp | base | new |", "|---|---|---|"]
    for k in range(5):
        lines.append(f"| {k + 1} | {np.mean([r['success'] for r in v['S2_deceptive']['a'] if r['rep'] == k]):.0%} | "
                     f"{np.mean([r['success'] for r in v['S2_deceptive']['b'] if r['rep'] == k]):.0%} |")
    return lines


def report(rows):
    systems = list(dict.fromkeys(r["system"] for r in rows))
    lines = ["# AutoGrasp harness report", "",
             f"{len({r['seed'] for r in rows})} seeds; same objects, order, percept noise and physics for every system. "
             "Excess = grip ÷ force the object needed (init = commanded at contact, peak = max during the grasp).", ""]
    for chal in systems[1:]:
        lines += report_pair(rows, systems[0], chal) + [""]
    if len(systems) > 2:
        lines += report_pair(rows, systems[1], systems[2]) + [""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gripper", choices=["box", "magpie"], default="box")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    rows = run(a.seeds, a.gripper, a.jobs)
    for r in rows:
        r["gripper"] = a.gripper
    with open(os.path.join(out, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    rep = report(rows)
    open(os.path.join(out, "report.md"), "w").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
