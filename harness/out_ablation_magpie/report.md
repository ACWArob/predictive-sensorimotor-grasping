# AutoGrasp harness report

5 seeds; same objects, order, percept noise and physics for every system. Excess = grip ÷ force the object needed (init = commanded at contact, peak = max during the grasp).

## a1_v1_plus_coupling vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 99% | 100% | +1% (+0%, +3%) | 1.84× / 1.84× | 0.65× / 1.64× | 0 | 0 | 2 | 0 |
| S2_deceptive | 50% | 83% | +33% (+26%, +41%) | 5.34× / 5.40× | 1.87× / 2.54× | 25 | 25 | 50 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 10.01× / 10.01× | 3.50× / 3.50× | 0 | 0 | 0 | 0 |
| S4_fragile | 33% | 67% | +33% (+20%, +47%) | 9.13× / 9.13× | 3.20× / 3.20× | 30 | 15 | 15 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 0.64× / 1.66× | 0 | 0 | 0 | 0 |

Pre-registered checks:

- PASS: 1. S1 no regression (≥ base − 2 pp), peak force closer to 1.0
- PASS: 2. S2 ≥ base + 20 pp, CI excludes 0
- FAIL: 3. S3 no leak (median peak excess on 2nd empty block ≤ 2.5×)
- PASS: 4. S4 crushes ≤ base
- PASS: 5. S5 repeats never worse than first try

S5: grasp 1 100.0%, grasps 2–5 100.0%.

S2 success by grasp number:

| grasp | base | new |
|---|---|---|
| 1 | 50% | 83% |
| 2 | 50% | 83% |
| 3 | 50% | 83% |
| 4 | 50% | 83% |
| 5 | 50% | 83% |

## a2_predict_only vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 99% | 100% | +1% (+0%, +3%) | 1.84× / 1.84× | 1.39× / 1.39× | 0 | 0 | 2 | 0 |
| S2_deceptive | 50% | 83% | +33% (+26%, +41%) | 5.34× / 5.40× | 2.61× / 3.16× | 25 | 25 | 50 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 10.01× / 10.01× | 5.98× / 5.98× | 0 | 0 | 0 | 0 |
| S4_fragile | 33% | 49% | +16% (+7%, +27%) | 9.13× / 9.13× | 7.23× / 7.23× | 30 | 23 | 7 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 1.34× / 1.35× | 0 | 0 | 0 | 0 |

Pre-registered checks:

- PASS: 1. S1 no regression (≥ base − 2 pp), peak force closer to 1.0
- PASS: 2. S2 ≥ base + 20 pp, CI excludes 0
- FAIL: 3. S3 no leak (median peak excess on 2nd empty block ≤ 2.5×)
- PASS: 4. S4 crushes ≤ base
- PASS: 5. S5 repeats never worse than first try

S5: grasp 1 100.0%, grasps 2–5 100.0%.

S2 success by grasp number:

| grasp | base | new |
|---|---|---|
| 1 | 50% | 83% |
| 2 | 50% | 83% |
| 3 | 50% | 83% |
| 4 | 50% | 83% |
| 5 | 50% | 83% |

## a3_predict_couple vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 99% | 100% | +1% (+0%, +3%) | 1.84× / 1.84× | 0.50× / 1.30× | 0 | 0 | 2 | 0 |
| S2_deceptive | 50% | 98% | +48% (+40%, +56%) | 5.34× / 5.40× | 0.91× / 1.58× | 25 | 3 | 72 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 10.01× / 10.01× | 2.09× / 2.09× | 0 | 0 | 0 | 0 |
| S4_fragile | 33% | 98% | +64% (+49%, +78%) | 9.13× / 9.13× | 2.53× / 2.53× | 30 | 1 | 29 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 0.48× / 1.29× | 0 | 0 | 0 | 0 |

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
| 2 | 50% | 97% |
| 3 | 50% | 100% |
| 4 | 50% | 97% |
| 5 | 50% | 100% |

## a4_predict_memory vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 99% | 100% | +1% (+0%, +3%) | 1.84× / 1.84× | 1.39× / 1.39× | 0 | 0 | 2 | 0 |
| S2_deceptive | 50% | 97% | +47% (+39%, +55%) | 5.34× / 5.40× | 1.19× / 1.25× | 25 | 5 | 70 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 10.01× / 10.01× | 1.12× / 1.16× | 0 | 0 | 0 | 0 |
| S4_fragile | 33% | 82% | +49% (+36%, +64%) | 9.13× / 9.13× | 1.29× / 1.30× | 30 | 8 | 22 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 1.31× / 1.32× | 0 | 0 | 0 | 0 |

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

## v4_no_value_head vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 99% | 100% | +1% (+0%, +3%) | 1.84× / 1.84× | 0.50× / 1.30× | 0 | 0 | 2 | 0 |
| S2_deceptive | 50% | 99% | +49% (+41%, +57%) | 5.34× / 5.40× | 0.47× / 1.30× | 25 | 1 | 74 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 10.01× / 10.01× | 1.09× / 1.24× | 0 | 0 | 0 | 0 |
| S4_fragile | 33% | 100% | +67% (+53%, +80%) | 9.13× / 9.13× | 0.98× / 1.27× | 30 | 0 | 30 | 0 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 1.84× / 1.84× | 0.47× / 1.29× | 0 | 0 | 0 | 0 |

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

## v3_psg_couple vs v1_deligrasp

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 99% | 100% | +1% (+0%, +3%) | 1.84× / 1.84× | 0.49× / 1.29× | 0 | 0 | 2 | 0 |
| S2_deceptive | 50% | 99% | +49% (+41%, +57%) | 5.34× / 5.40× | 0.48× / 1.30× | 25 | 1 | 74 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 10.01× / 10.01× | 1.09× / 1.24× | 0 | 0 | 0 | 0 |
| S4_fragile | 33% | 100% | +67% (+53%, +80%) | 9.13× / 9.13× | 0.90× / 1.28× | 30 | 0 | 30 | 0 |
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

## a2_predict_only vs a1_v1_plus_coupling

| Suite | base success | new success | Δ (95% CI) | base excess (init / peak) | new excess (init / peak) | base crushes | new crushes | new-only wins | base-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 100% | 100% | +0% (+0%, +0%) | 0.65× / 1.64× | 1.39× / 1.39× | 0 | 0 | 0 | 0 |
| S2_deceptive | 83% | 83% | +0% (+0%, +0%) | 1.87× / 2.54× | 2.61× / 3.16× | 25 | 25 | 0 | 0 |
| S3_lookalike_leak | 100% | 100% | +0% (+0%, +0%) | 3.50× / 3.50× | 5.98× / 5.98× | 0 | 0 | 0 | 0 |
| S4_fragile | 67% | 49% | -18% (-29%, -7%) | 3.20× / 3.20× | 7.23× / 7.23× | 15 | 23 | 0 | 8 |
| S5_repeat_honest | 100% | 100% | +0% (+0%, +0%) | 0.64× / 1.66× | 1.34× / 1.35× | 0 | 0 | 0 | 0 |

Pre-registered checks:

- PASS: 1. S1 no regression (≥ base − 2 pp), peak force closer to 1.0
- FAIL: 2. S2 ≥ base + 20 pp, CI excludes 0
- FAIL: 3. S3 no leak (median peak excess on 2nd empty block ≤ 2.5×)
- FAIL: 4. S4 crushes ≤ base
- PASS: 5. S5 repeats never worse than first try

S5: grasp 1 100.0%, grasps 2–5 100.0%.

S2 success by grasp number:

| grasp | base | new |
|---|---|---|
| 1 | 83% | 83% |
| 2 | 83% | 83% |
| 3 | 83% | 83% |
| 4 | 83% | 83% |
| 5 | 83% | 83% |

