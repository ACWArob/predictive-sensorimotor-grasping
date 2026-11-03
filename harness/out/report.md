# AutoGrasp v1 vs v2: paired harness report

5 seeds; same objects, order, percept noise and physics for both systems.

| Suite | v1 success | v2 success | Δ (95% CI) | v1 median excess | v2 median excess | v1 crushes | v2 crushes | v2-only wins | v1-only wins |
|---|---|---|---|---|---|---|---|---|---|
| S1_honest_unseen | 100% | 99% | -1% (-2%, +0%) | 1.84× | 1.29× | 0 | 0 | 0 | 1 |
| S2_deceptive | 40% | 97% | +57% (+49%, +65%) | 5.34× | 1.24× | 25 | 5 | 85 | 0 |
| S3_lookalike_leak | 80% | 100% | +20% (+9%, +31%) | 10.01× | 1.19× | 0 | 0 | 9 | 0 |
| S4_fragile | 33% | 82% | +49% (+36%, +64%) | 9.13× | 1.59× | 30 | 8 | 22 | 0 |
| S5_repeat_honest | 100% | 100% | -0% (-1%, +0%) | 1.84× | 1.26× | 0 | 0 | 0 | 1 |

## Pre-registered checks

- PASS: 1. S1 no regression, force closer to 1.0
- PASS: 2. S2 v2 ≥ v1 + 20 pp, CI excludes 0
- PASS: 3. S3 no leak (late excess v2 ≤ 2.5×)
- PASS: 4. S4 v2 crushes ≤ v1
- FAIL: 5. S5 repeats never worse than first try

S3 detail: median excess on the second 'empty' block: v1 10.20×, v2 1.07×.
S5 detail: v2 success grasp 1 100%, grasps 2–5 100%.

## S2 by grasp number (success)

| grasp | v1 | v2 |
|---|---|---|
| 1 | 40% | 83% |
| 2 | 40% | 100% |
| 3 | 40% | 100% |
| 4 | 40% | 100% |
| 5 | 40% | 100% |
