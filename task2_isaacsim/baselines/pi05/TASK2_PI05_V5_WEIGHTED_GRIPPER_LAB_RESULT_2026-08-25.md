# Task 2 PI0.5 V5 weighted-gripper lab result (2026-08-25)

## Outcome

**NO-GO for simulator and live execution.** Uniformly increasing the right
gripper flow-loss weight improves open predictions but does not resolve the
open-to-close transition. It eventually damages grasp acquisition and late
grasp retention.

The experiment started from the V2 30k checkpoint. It preserved the original
single task prompt, V2 relative-arm/absolute-spine action mapping, hard5
runtime contract, six phase-balanced sampling groups, immutable 179/20
episode split, expert-only training, and frozen vision encoder. Only action 18
used weight `8.0`; all other dimensions used `1.0`. Weights were normalized by
their mean before reducing the loss.

## Held-out hard5 gate

Each value is the fraction of 20 held-out episodes in which all five predicted
actions have the required right-gripper state. `hold average` is the mean of
early, mid, and late hold success.

| checkpoint | safe | approach open | grasp close | hold early | hold mid | hold late | release open | hold average | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V2 30k baseline | 1.00 | 0.70 | 0.95 | 1.00 | 1.00 | 0.80 | 0.95 | 0.933 | NO-GO |
| V5 step 100 | 1.00 | 0.75 | 0.90 | 1.00 | 1.00 | 0.80 | 0.95 | 0.933 | NO-GO |
| V5 step 3000 | 1.00 | 0.95 | 0.70 | 1.00 | 1.00 | 0.45 | 0.95 | 0.817 | NO-GO |

Gate thresholds are safety `1.00`, approach open `>=0.90`, grasp close
`>=0.80`, hold average `>=0.90`, and release open `>=0.90`. The gate uses the
original task prompt and policy outputs, with no phase prompt, GT action after
handoff, gripper clamp, or ROS publication.

Lab evidence is stored outside Git under:

```text
/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_v5_weighted_gripper_3k/
  v2_030000_gripper_hold_baseline.json
  gripper_hold_gate_step100.json
  gripper_hold_gate.json
```

The complete 3k run is under:

```text
/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v5_weighted_gripper_3k/
```

## Interpretation and next experiment

The V2 checkpoint already predicts closed grippers under most teacher-forced
closed-phase observations, while its corrected live rollout reopens after it
drifts from the expert trajectory. The combination indicates closed-loop
covariate shift and ambiguous phase timing, not simply an underweighted output
dimension. The monotonic open/close trade-off across V2, step 100, and step
3000 supports this diagnosis.

The next training run should collect simulator recovery trajectories from
policy-induced states (DAgger-style), especially just before acquisition and
after small pad/arm slips. Randomize pad/target pose, camera/light conditions,
and small dual-arm/spine offsets, then keep a scenario-level held-out split.
Label corrective expert actions with the same single task prompt and action
contract. Add an explicit temporal/history signal only if recovery data alone
does not make the transition identifiable.

Do not start online RL yet. First define and validate a simulator reward from
contact, stable lift, target overlap, retained grasp, safe motion, and terminal
release. Offline advantage-weighted or conservative RL can then use the
recovery buffer without exposing the live robot to exploratory failures.
