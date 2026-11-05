# After the grasp: carry-and-place on the MAGPIE model

3 seeds; every system gets the same objects, order, noise, physics and move. Task success = held through the move, not crushed, object < 5 mm from its start pose in the hand.

## S1_honest_unseen

| System | medium task success | fast task success | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |
|---|---|---|---|---|---|---|
| v1_hold | 99% | 86% | 0 | 0 | 0.1 | 1.77× |
| v4_hold | 88% | 43% | 4 | 0 | 5.4 | 1.45× |
| v4_reactive | 62% | 31% | 2 | 0 | 6.6 | 1.45× |
| v5_ff_no_memory | 81% | 57% | 4 | 0 | 4.4 | 1.53× |
| v5_feedforward | 88% | 92% | 1 | 0 | 0.1 | 2.05× |
| v5_warm | 98% | 98% | 0 | 0 | 0.0 | 2.46× |

## S2_deceptive

| System | medium task success | fast task success | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |
|---|---|---|---|---|---|---|
| v1_hold | 39% | 33% | 0 | 15 | 0.0 | 10.35× |
| v4_hold | 89% | 57% | 14 | 1 | 3.7 | 1.30× |
| v4_reactive | 72% | 32% | 13 | 1 | 6.6 | 1.27× |
| v5_ff_no_memory | 89% | 57% | 13 | 1 | 4.2 | 1.41× |
| v5_feedforward | 99% | 90% | 4 | 1 | 0.1 | 1.63× |
| v5_warm | 99% | 98% | 0 | 1 | 0.0 | 2.31× |

## S4_fragile

| System | medium task success | fast task success | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |
|---|---|---|---|---|---|---|
| v1_hold | 33% | 33% | 0 | 18 | 0.0 | 7.99× |
| v4_hold | 100% | 78% | 0 | 0 | 3.0 | 1.42× |
| v4_reactive | 81% | 41% | 0 | 0 | 5.2 | 1.36× |
| v5_ff_no_memory | 96% | 67% | 0 | 0 | 3.9 | 1.48× |
| v5_feedforward | 100% | 96% | 0 | 0 | 0.1 | 1.41× |
| v5_warm | 100% | 100% | 0 | 0 | 0.0 | 2.31× |

## Deceptive objects, fast move: task success by grasp number

| grasp | v1_hold | v4_hold | v4_reactive | v5_ff_no_memory | v5_feedforward | v5_warm |
|---|---|---|---|---|---|---|
| 1 | 33% | 61% | 33% | 56% | 67% | 94% |
| 2 | 33% | 56% | 33% | 44% | 94% | 100% |
| 3 | 33% | 61% | 33% | 67% | 89% | 100% |
| 4 | 33% | 56% | 39% | 50% | 100% | 94% |
| 5 | 33% | 50% | 22% | 67% | 100% | 100% |
