# AutoGrasp harness report

5 seeds; same objects, order, percept noise and physics for every system. Excess = grip ÷ force the object needed (init = commanded at contact, peak = max during the grasp).

## v2_psg vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 99% | 100% | +1% (+0%, +3%) | 1.84× / 1.84× | 1.35× / 1.47× | 0 | 0 | 2 | 0 |
| S2_deceptive | 50% | 97% | +47% (+39%, +55%) | 5.34× / 5.40× | 1.19× / 1.26× | 25 | 5 | 70 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 10.01× / 10.01× | 1.23× / 1.25× | 0 | 0 | 0 | 0 |
| S4_fragile | 33% | 84% | +51% (+38%, +67%) | 9.13× / 9.13× | 1.56× / 1.56× | 30 | 7 | 23 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 1.27× / 1.31× | 0 | 0 | 0 | 0 |

Pre-registered checks:

- PASS: 1. S1 no regression (≥ base − 2 pp), peak force closer to 1.0
- PASS: 2. S2 ≥ base + 20 pp, CI excludes 0
- PASS: 3. S3 no leak (median peak excess on 2nd empty block ≤ 2.5×)
- PASS: 4. S4 crushes ≤ base
- PASS: 5. S5 repeats never worse than first try

S5: grasp 1 100.0%, grasps 2–5 100.0%.

S2 success by grasp number:

| grasp | base | new |
|---|---|---|
| 1 | 50% | 83% |
| 2 | 50% | 100% |
| 3 | 50% | 100% |
| 4 | 50% | 100% |
| 5 | 50% | 100% |

## v3_psg_couple vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 99% | 100% | +1% (+0%, +3%) | 1.84× / 1.84× | 0.49× / 1.29× | 0 | 0 | 2 | 0 |
| S2_deceptive | 50% | 99% | +49% (+41%, +57%) | 5.34× / 5.40× | 0.48× / 1.30× | 25 | 1 | 74 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 10.01× / 10.01× | 1.09× / 1.24× | 0 | 0 | 0 | 0 |
| S4_fragile | 33% | 100% | +67% (+51%, +80%) | 9.13× / 9.13× | 0.90× / 1.28× | 30 | 0 | 30 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 0.45× / 1.30× | 0 | 0 | 0 | 0 |

Pre-registered checks:

- PASS: 1. S1 no regression (≥ base − 2 pp), peak force closer to 1.0
- PASS: 2. S2 ≥ base + 20 pp, CI excludes 0
- PASS: 3. S3 no leak (median peak excess on 2nd empty block ≤ 2.5×)
- PASS: 4. S4 crushes ≤ base
- PASS: 5. S5 repeats never worse than first try

S5: grasp 1 100.0%, grasps 2–5 100.0%.

S2 success by grasp number:

| grasp | base | new |
|---|---|---|
| 1 | 50% | 97% |
| 2 | 50% | 100% |
| 3 | 50% | 100% |
| 4 | 50% | 100% |
| 5 | 50% | 100% |

## v3_psg_couple vs v2_psg

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 100% | 100% | +0% (+0%, +0%) | 1.35× / 1.47× | 0.49× / 1.29× | 0 | 0 | 0 | 0 |
| S2_deceptive | 97% | 99% | +3% (+1%, +5%) | 1.19× / 1.26× | 0.48× / 1.30× | 5 | 1 | 4 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 1.23× / 1.25× | 1.09× / 1.24× | 0 | 0 | 0 | 0 |
| S4_fragile | 84% | 100% | +16% (+4%, +27%) | 1.56× / 1.56× | 0.90× / 1.28× | 7 | 0 | 7 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.27× / 1.31× | 0.45× / 1.30× | 0 | 0 | 0 | 0 |

Pre-registered checks:

- PASS: 1. S1 no regression (≥ base − 2 pp), peak force closer to 1.0
- FAIL: 2. S2 ≥ base + 20 pp, CI excludes 0
- PASS: 3. S3 no leak (median peak excess on 2nd empty block ≤ 2.5×)
- PASS: 4. S4 crushes ≤ base
- PASS: 5. S5 repeats never worse than first try

S5: grasp 1 100.0%, grasps 2–5 100.0%.

S2 success by grasp number:

| grasp | base | new |
|---|---|---|
| 1 | 83% | 97% |
| 2 | 100% | 100% |
| 3 | 100% | 100% |
| 4 | 100% | 100% |
| 5 | 100% | 100% |

