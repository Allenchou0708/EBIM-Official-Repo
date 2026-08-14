# Task 2 PI0.5 V3 Hard5 and Phase-Progression Lab Result — 2026-08-14

## Outcome

The V3 calibration resolved the original spine target symptom but did not
grasp the thermal pad. A real 600-action GUI run completed safely with the
checkpoint's intended five-action receding horizon. The runner and camera
input contracts are now verified; the single leading cause is **PI0.5 policy
phase progression / closed-loop robustness**, especially wrist-height
compensation and the transition into right-gripper close.

Do not spend another GUI attempt on this checkpoint. The next model experiment
should add phase-specific language labels and retain one final checkpoint only.

## Scope and artifacts

- Repository/branch: `submit`, starting from
  `85b759a652b58ccd7b5d4f08ebbb6b034bd55fa1`; no new branch.
- Raw organizer dataset:
  `/scratch1/2026_ebim/allen_task2_pi05/datasets/task2_fixpos_200_46ab41f`.
- V3 run root:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v3_from_v1_3k`.
- V3 checkpoint:
  `training/checkpoints/003000/pretrained_model`.
- Final runtime image: `ebim-task2-pi05-submit:v3-hard5-20260814`, image ID
  `sha256:260bacb68192...`; health PASS.
- V3 produced one new checkpoint only.
- Final legacy-horizon comparison manifest:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260814_140631/live_runner_manifest.json`.
- Hard5 shadow manifest:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260814_142603/live_runner_manifest.json`.
- Final hard5 GUI manifest:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260814_143403/live_runner_manifest.json`.

No dataset replay was run. The final GUI result made an evaluator invocation
unnecessary: the right gripper never closed, so there was no grasp or pad
transfer to score.

## V3 model and offline gate

V3 initialized from the existing V1 30k checkpoint, retained relative arm
targets, changed spine action 19 to the simulator's absolute target space, and
ran 3000 phase-balanced expert-only steps at learning rate `5e-6`. Final
training loss was `0.021`; the final checkpoint was the only saved V3
checkpoint.

Dataset evidence confirms the required kinematic behavior: while median spine
height rises by `0.4413 m`, median left/right end-effector world Z changes only
`+0.0221/+0.0196 m`. The arms must therefore compensate almost all spine
motion. V3 improved held-out arm and close landmarks over V2, but its frame-0
spine prediction jumped immediately to about `0.48 m`, and its early arm
compensation remained too weak over the full closed-loop rise.

## Confirmed local runner fixes

Two local contract mismatches were isolated and fixed:

1. Checkpoint metadata specifies `n_action_steps: 5`, while the asynchronous
   runner had executed about 26 actions per decision (23 decisions for 600
   actions). Default `hard5` now executes exactly chunk indices `0..4`, then
   holds the last legal absolute target during the next inference. The final
   run used 120 decisions for 600 actions, with no queue replacement.
2. Cross-camera skew had been computed from host callback arrival times. An
   eight-second ROS probe found an exactly synchronized three-camera tuple;
   the newest tuple differed by only `0.05 s`. The apparent `0.3–0.52 s` skew
   was transport/callback delay. The runner now uses ROS image-header simulator
   timestamps for capture skew and host monotonic time only for frame age.

The zero-publication shadow passed with `policy_indices=[0,1,2,3,4]`, zero
commands, zero discarded actions, and capture skew `0.0333 s`.

## Final 600-action GUI evidence

- Completed: `true`; `120/120` valid decisions, `600/600` policy actions,
  zero invalid actions, no publish block, no queue replacement.
- Every event executed exactly policy indices `0..4`.
- Inference latency p50/p95/max:
  `0.5217/0.5547/0.5839 s`.
- Accepted capture skew median/max: `0.0167/0.0833 s`; accepted state age max
  `0.1241 s`. Camera keys, RGB shapes, encoding, and simulator timestamps were
  valid.
- Spine target min/median/max/final:
  `0.4698/0.5072/0.5360/0.4968 m`.
- Right gripper minimum: `0.9691`; actions below `0.5`: `0/600`.
- Left gripper also remained open for all 600 actions.
- Right joint 4 reached its lower bound in `97/600` effective actions; the
  other right joints did not hit a bound.
- Measured right end-effector Z was `0.9139 m` initially, rose to `1.3250 m`
  by action 100, and ended at `0.9527 m`. The left end effector rose to
  `1.3641 m` by action 100 and ended at `1.1844 m`. This reproduces the
  operator's observation that the hands rise with the spine rather than
  retaining the demonstrated table-relative height.

Hard5 prevents the rapid multi-joint saturation seen in the legacy-horizon
run, so the horizon fix is real. It does not create the missing compensation
or close transition; those are checkpoint behavior.

## Single leading cause

**The VLA does not robustly progress from spine-rise/approach into aligned
pre-grasp and close under its own observations.** Low aggregate training loss
and good teacher-forced landmark predictions do not cover the live states
created by insufficient early arm compensation. The wrist views then leave
the demonstration manifold, the model continues an open-gripper approach,
and right joint 4 eventually saturates.

Simulator actuation is not the leading cause: raw replay already established
the command interface, the runner published finite bounded targets, simulator
clock pacing was correct, and the final run completed without evaluator,
freshness, reset, NaN, collision, or command-contention failure. Camera input
is not the leading cause either: the three streams have correct keys/shapes and
accepted capture skew below `0.084 s`.

## Why phase language is the next experiment

The current dataset supplies the same task sentence at every frame. Phase
balancing changes sampling frequency but does not tell the model whether the
same scene should mean "maintain wrist height and approach", "close", or
"transfer". The final run's `0/600` close actions is direct evidence that this
ambiguity remains.

Use five frame-level language conditions derived from the existing event
manifest:

1. Raise the spine while keeping both wrists at table height.
2. Align the right gripper with the thermal pad.
3. Close the right gripper and lift the thermal pad.
4. Carry the thermal pad to the target RAM board.
5. Place the pad and open the right gripper.

Run one V4 continuation from V3, save only its final checkpoint, and gate it
offline before any GUI run. The required gates are: frame-0 first-50 wrist
compensation, no premature close in stages 1–2, close recall in stage 3, and
bounded right-joint targets under perturbed/recovery states. A live phase
manager must switch prompts from observable state/contact/visual conditions;
changing prompts at inference without training these labels is not a valid
test. Migrating to a separate imitation-learning architecture is lower
priority because the existing PI0.5 stack now has a verified runtime contract
and the failure is specifically phase-conditioned behavior.

## Verification

- Full PI0.5 contract/live unit suite in the final image with the repository
  mounted read-only: `60` tests passed; the packaged live suite passed `13`
  tests and image health passed.
- Python compile, `bash -n`, and `git diff --check`: passed before handoff.
- Shadow: passed, zero ROS command publications.
- Final GUI: one completed hard5 600-action run; no additional GUI attempt.
