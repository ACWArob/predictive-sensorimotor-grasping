# AutoGrasp harness report

5 seeds; same objects, order, percept noise and physics for every system. Excess = grip ÷ force the object needed (init = commanded at contact, peak = max during the grasp).

## v2_psg vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 1.39× / 1.57× | 0 | 0 | 0 | 0 |
| S2_deceptive | 40% | 97% | +57% (+49%, +65%) | 5.34× / 5.34× | 1.25× / 1.42× | 25 | 5 | 85 | 0 |
| S3_lookalike_leak | 80% | 100% | +20% (+9%, +31%) | 10.01× / 10.01× | 1.21× / 1.49× | 0 | 0 | 9 | 0 |
| S4_fragile | 33% | 82% | +49% (+36%, +64%) | 9.13× / 9.13× | 1.59× / 1.61× | 30 | 8 | 22 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 1.34× / 1.42× | 0 | 0 | 0 | 0 |

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
| 1 | 40% | 83% |
| 2 | 40% | 100% |
| 3 | 40% | 100% |
| 4 | 40% | 100% |
| 5 | 40% | 100% |

## v2a_no_value_head vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 1.50× / 1.56× | 0 | 0 | 0 | 0 |
| S2_deceptive | 40% | 97% | +57% (+49%, +65%) | 5.34× / 5.34× | 1.30× / 1.41× | 25 | 5 | 85 | 0 |
| S3_lookalike_leak | 80% | 100% | +20% (+9%, +31%) | 10.01× / 10.01× | 1.24× / 1.35× | 0 | 0 | 9 | 0 |
| S4_fragile | 33% | 78% | +44% (+31%, +60%) | 9.13× / 9.13× | 1.42× / 1.46× | 30 | 10 | 20 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 1.44× / 1.45× | 0 | 0 | 0 | 0 |

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
| 1 | 40% | 83% |
| 2 | 40% | 100% |
| 3 | 40% | 100% |
| 4 | 40% | 100% |
| 5 | 40% | 100% |

## v2b_excess_reward vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 1.51× / 1.59× | 0 | 0 | 0 | 0 |
| S2_deceptive | 40% | 97% | +57% (+49%, +65%) | 5.34× / 5.34× | 1.46× / 1.60× | 25 | 5 | 85 | 0 |
| S3_lookalike_leak | 80% | 100% | +20% (+9%, +33%) | 10.01× / 10.01× | 1.29× / 1.45× | 0 | 0 | 9 | 0 |
| S4_fragile | 33% | 82% | +49% (+36%, +62%) | 9.13× / 9.13× | 1.73× / 1.84× | 30 | 8 | 22 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 1.48× / 1.58× | 0 | 0 | 0 | 0 |

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
| 1 | 40% | 83% |
| 2 | 40% | 100% |
| 3 | 40% | 100% |
| 4 | 40% | 100% |
| 5 | 40% | 100% |

## v2a_no_value_head vs v2_psg

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 100% | 100% | +0% (+0%, +0%) | 1.39× / 1.57× | 1.50× / 1.56× | 0 | 0 | 0 | 0 |
| S2_deceptive | 97% | 97% | +0% (+0%, +0%) | 1.25× / 1.42× | 1.30× / 1.41× | 5 | 5 | 0 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 1.21× / 1.49× | 1.24× / 1.35× | 0 | 0 | 0 | 0 |
| S4_fragile | 82% | 78% | -4% (-11%, +0%) | 1.59× / 1.61× | 1.42× / 1.46× | 8 | 10 | 0 | 2 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.34× / 1.42× | 1.44× / 1.45× | 0 | 0 | 0 | 0 |

Pre-registered checks:

- PASS: 1. S1 no regression (≥ base − 2 pp), peak force closer to 1.0
- FAIL: 2. S2 ≥ base + 20 pp, CI excludes 0
- PASS: 3. S3 no leak (median peak excess on 2nd empty block ≤ 2.5×)
- FAIL: 4. S4 crushes ≤ base
- PASS: 5. S5 repeats never worse than first try

S5: grasp 1 100.0%, grasps 2–5 100.0%.

S2 success by grasp number:

| grasp | base | new |
|---|---|---|
| 1 | 83% | 83% |
| 2 | 100% | 100% |
| 3 | 100% | 100% |
| 4 | 100% | 100% |
| 5 | 100% | 100% |

