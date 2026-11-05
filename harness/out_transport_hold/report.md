# After the grasp: carry-and-place on the MAGPIE model

3 seeds; every system gets the same objects, order, noise, physics and move. Task success = held through the move, not crushed, object < 5 mm from its start pose in the hand.

## S1_honest_unseen

| System | medium task success | fast task success | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |
|---|---|---|---|---|---|---|
| v5_learned_hold | 97% | 96% | 0 | 0 | 0.0 | 2.20× |

## S2_deceptive

| System | medium task success | fast task success | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |
|---|---|---|---|---|---|---|
| v5_learned_hold | 99% | 98% | 0 | 1 | 0.0 | 2.08× |

## S4_fragile

| System | medium task success | fast task success | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |
|---|---|---|---|---|---|---|
| v5_learned_hold | 100% | 100% | 0 | 0 | 0.0 | 2.08× |

## Deceptive objects, fast move: task success by grasp number

| grasp | v5_learned_hold |
|---|---|
| 1 | 94% |
| 2 | 100% |
| 3 | 94% |
| 4 | 100% |
| 5 | 100% |
