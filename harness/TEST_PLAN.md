# AutoGrasp v1 vs v2 test harness: what we test, and how

**Question:** does the new AutoGrasp (v2 = PSG: predict, compare, correct) beat the current AutoGrasp (v1 = DeliGrasp: semantic force prior + slip reflex) on the grasps that matter, without paying for it elsewhere?

Both versions run the **same objects, in the same order, with the same seeds and the same physics**, so every difference is the method. Results are paired: a grasp counts as a v2 win when v2 succeeds and v1 fails on the identical trial.

## The two systems

| | v1: current AutoGrasp | v2: PSG AutoGrasp |
|---|---|---|
| Initial grip | material prior (mass and μ the way an LLM would guess them) × 1.5 margin | forward model: predicted lift-off load × slip ratio × margin(σ) |
| During the grasp | slip → grip ×1.4 | same slip reflex, **plus** grip-load coupling when load runs past prediction + 2σ |
| After the grasp | nothing carried over | per-object Kalman residual (surprise > 3σ resets variance); crush → force ceiling |
| Learned parts | none | predictor (offline, logged grasps); value head (margin, gated) |
| Data it gets | same Phase-0 log (48 honest objects, ~240 grasps) | same |

**v3 = PSG + grip-load coupling from contact**: close at 0.35× the planned grip, then raise grip with the felt load (grip = load × predicted ratio × 1.4), stepping up whenever the fingers are near their friction limit. This is the Johansson & Westling loading phase. It was added after v2 still crushed objects that look heavy but are fragile.

All PSG versions have a 0.3 N grip floor (the MAGPIE gripper's ~0.15 N/finger sensing floor). It was added after the harness showed that below it the simulated fingers never reach very light foam objects. v1 never commands that little, so it is unaffected.

## Test suites

| Suite | Objects | Grasps | What it probes | v2 should |
|---|---|---|---|---|
| S1 honest-unseen | 30 new objects whose looks match their weight | 1 each | no regression on easy objects | match v1 success, lower force error |
| S2 deceptive | size-weight pair, hollow metal, lead-filled plastic, empty/full bottles | 5 each, interleaved | fixing a wrong prior | ≥ v1 on grasp 1 (reflex), ~100% from grasp 2 |
| S3 look-alike leak | bottle empty ×3 → full ×3 → empty ×3 | 9 | correction must not spill onto an identical-looking object | grip on the 2nd "empty" block back near 1–1.5× need |
| S4 fragile | low crush-limit objects (foam, thin shells) that look sturdy | 3 each | over-gripping | fewer crushes than v1 after grasp 1 |
| S5 repeat-honest | 10 honest objects × 5 grasps | 50 | does learning ever hurt a correct prior? | no drop vs grasp 1 |

## Metrics (per grasp, logged to CSV)

- **success**: lifted 10 cm and held 1 s, not crushed
- **clean**: success with no slip event
- **first-try success**: success on the first grasp of each object
- **excess force**: initial grip / force actually needed (1.0 = perfect; >2.5 = wasteful; <1 = will slip)
- **crush**: grip ever exceeded the object's crush limit
- **grasps-to-correct**: first grasp index after which excess force stays in [1, 2.5]
- **reflexes**: number of mid-grasp grip changes

## Pass criteria (decided before running)

1. S1: v2 success ≥ v1 success − 2 pp, and v2 median excess force closer to 1.0.
2. S2: v2 success − v1 success ≥ 20 pp, with the 95% bootstrap CI excluding 0.
3. S3: v2 excess force on grasps 7–9 within 2× of grasps 1–3's correct value; v1 or retrieval may leak.
4. S4: v2 crushes ≤ v1 crushes over all grasps.
5. S5: v2 success on grasps 2–5 ≥ grasp 1.

Run: `python -m harness.compare_autograsp --seeds 5` → `harness/out/report.md`, `harness/out/trials.csv`.

The same plan carries to hardware (UR5 + DeliGrasp, `use_psg:=false` vs `true`) and to the H1-2 HAMS sim once its MAGPIE hands expose a load signal. Only the grasp executor changes.

## Gripper models

`--gripper box` (default) uses the simple parallel jaw in `sim/env.py`. `--gripper magpie` uses the MAGPIE MuJoCo model from HAMS (`sim/env_magpie.py`, meshes in `sim/assets/magpie/`): 4-bar fingers, grip applied as crank torque τ = grip × J(θ), fingers pre-shaped to 1 mm outside the object before the timed grasp. Both Phase-0 training data and test grasps then use MAGPIE. Results: `harness/out_magpie/`.
