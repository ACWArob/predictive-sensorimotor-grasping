# After the grasp: carry-and-place on the MAGPIE model

3 seeds; every system gets the same objects, order, noise, physics and move. Task success = held through the move, not crushed, object < 5 mm from its start pose in the hand.

## S1_honest_unseen

| System | medium task success | fast task success | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |
|---|---|---|---|---|---|---|
| v5_chunk | 98% | 98% | 0 | 0 | 0.0 | 2.46× |
| v5_extrap | 99% | 96% | 0 | 0 | 0.0 | 2.53× |

## S2_deceptive

| System | medium task success | fast task success | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |
|---|---|---|---|---|---|---|
| v5_chunk | 99% | 98% | 0 | 1 | 0.0 | 2.31× |
| v5_extrap | 99% | 99% | 0 | 1 | 0.0 | 2.27× |

## S4_fragile

| System | medium task success | fast task success | fast: dropped | fast: crushed | fast: shift (mm, median) | fast: peak grip ÷ need |
|---|---|---|---|---|---|---|
| v5_chunk | 100% | 100% | 0 | 0 | 0.0 | 2.31× |
| v5_extrap | 100% | 100% | 0 | 0 | 0.0 | 2.25× |

## Deceptive objects, fast move: task success by grasp number

| grasp | v5_chunk | v5_extrap |
|---|---|---|
| 1 | 94% | 94% |
| 2 | 100% | 100% |
| 3 | 100% | 100% |
| 4 | 94% | 100% |
| 5 | 100% | 100% |
