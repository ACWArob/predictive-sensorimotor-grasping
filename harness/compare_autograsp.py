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
SYSTEMS = {"v1_deligrasp": None, "v2_psg": "full"}


def fragile_set(rng):
    """Look sturdy, crush easily."""
    return [make("fragile_thin_cup", "ceramic", (0.03, 0.03, 0.045), rng, density=400, crush=4.0),
            make("fragile_foam_block", "wood", (0.03, 0.03, 0.03), rng, density=120, crush=3.0),
            make("fragile_shell", "metal", (0.025, 0.025, 0.04), rng, density=300, crush=5.0)]


def new_system(name, model, vrows):
    if name == "v1_deligrasp":
        return DeliGraspLike()
    agent = PredictiveGraspAgent(model, mode=SYSTEMS[name])
    rows = [(x, f / np.exp(model.predict(x)[0][0].sum()), ok) for x, f, ok in vrows]
    agent.pretrain_value_head(rows)
    return agent


def do_grasp(system, obj, rng):
    pc = percept(obj, rng)
    if isinstance(system, PredictiveGraspAgent):
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
                grip_max=round(res.grip_max, 3), success=int(res.success),
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


def run(seeds):
    rows = []
    for seed in range(seeds):
        t0 = time.time()
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


def boot_ci(diff, n=2000, rng=np.random.default_rng(0)):
    diff = np.asarray(diff, float)
    if len(diff) == 0:
        return (np.nan, np.nan)
    b = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def report(rows):
    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]
    lines = ["# AutoGrasp v1 vs v2: paired harness report", "",
             f"{len({r['seed'] for r in rows})} seeds; same objects, order, percept noise and physics for both systems.", "",
             "| Suite | v1 success | v2 success | Δ (95% CI) | v1 median excess | v2 median excess | v1 crushes | v2 crushes | v2-only wins | v1-only wins |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for s in ["S1_honest_unseen", "S2_deceptive", "S3_lookalike_leak", "S4_fragile", "S5_repeat_honest"]:
        a, b = sel(suite=s, system="v1_deligrasp"), sel(suite=s, system="v2_psg")
        key = lambda r: (r["seed"], r["trial"])                          # noqa: E731
        A, Bm = {key(r): r for r in a}, {key(r): r for r in b}
        diff = [Bm[k]["success"] - A[k]["success"] for k in A]
        lo, hi = boot_ci(diff)
        sa, sb = np.mean([r["success"] for r in a]), np.mean([r["success"] for r in b])
        ea, eb = np.median([r["excess"] for r in a]), np.median([r["excess"] for r in b])
        ca, cb = sum(r["crushed"] for r in a), sum(r["crushed"] for r in b)
        w2, w1 = sum(d == 1 for d in diff), sum(d == -1 for d in diff)
        lines.append(f"| {s} | {sa:.0%} | {sb:.0%} | {sb - sa:+.0%} ({lo:+.0%}, {hi:+.0%}) | {ea:.2f}× | {eb:.2f}× | {ca} | {cb} | {w2} | {w1} |")
        verdict[s] = dict(sa=sa, sb=sb, lo=lo, ea=ea, eb=eb, ca=ca, cb=cb, a=a, b=b)

    v = verdict
    s3 = v["S3_lookalike_leak"]
    late = lambda rs: np.median([r["excess"] for r in rs if r["trial"] >= 6])   # noqa: E731
    s5b = v["S5_repeat_honest"]["b"]
    g1 = np.mean([r["success"] for r in s5b if r["rep"] == 0])
    gl = np.mean([r["success"] for r in s5b if r["rep"] > 0])
    checks = [
        ("1. S1 no regression, force closer to 1.0", v["S1_honest_unseen"]["sb"] >= v["S1_honest_unseen"]["sa"] - 0.02 and abs(np.log(v["S1_honest_unseen"]["eb"])) < abs(np.log(v["S1_honest_unseen"]["ea"]))),
        ("2. S2 v2 ≥ v1 + 20 pp, CI excludes 0", v["S2_deceptive"]["sb"] - v["S2_deceptive"]["sa"] >= 0.20 and v["S2_deceptive"]["lo"] > 0),
        ("3. S3 no leak (late excess v2 ≤ 2.5×)", late(s3["b"]) <= 2.5),
        ("4. S4 v2 crushes ≤ v1", v["S4_fragile"]["cb"] <= v["S4_fragile"]["ca"]),
        ("5. S5 repeats never worse than first try", gl >= g1),
    ]
    lines += ["", "## Pre-registered checks", ""] + [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks]
    lines += ["", f"S3 detail: median excess on the second 'empty' block: v1 {late(s3['a']):.2f}×, v2 {late(s3['b']):.2f}×.",
              f"S5 detail: v2 success grasp 1 {g1:.0%}, grasps 2–5 {gl:.0%}."]
    s2a, s2b = v["S2_deceptive"]["a"], v["S2_deceptive"]["b"]
    lines += ["", "## S2 by grasp number (success)", "", "| grasp | v1 | v2 |", "|---|---|---|"]
    for k in range(5):
        lines.append(f"| {k + 1} | {np.mean([r['success'] for r in s2a if r['rep'] == k]):.0%} | {np.mean([r['success'] for r in s2b if r['rep'] == k]):.0%} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = run(a.seeds)
    with open(os.path.join(OUT, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    rep = report(rows)
    open(os.path.join(OUT, "report.md"), "w").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
