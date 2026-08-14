# Task 2 PI0.5 V2 Absolute-Spine Phase-Balanced Lab Result — 2026-08-14

## Outcome

**PAUSED by operator request.** V2 fixes the previous spine-target learning
failure and the live runner's wall-clock pacing bug, but the 6k checkpoint did
not grasp the thermal pad. Do not spend another training run until arm/spine
frame parity is isolated.

The current leading cause is a mismatch between the demonstrated arm target
frame/state and live spine motion. During the corrected GUI run, the operator
observed that both arms rose with the spine instead of maintaining the same
world height as in the dataset. This changes the wrist-camera observations and
is consistent with the right arm drifting away from the demonstrated grasp
trajectory. It is more specific than a generic claim of insufficient VLA
training.

## Repository and runtime

- Repository: `/home/robot/2026_ebim_ssd/benchmark_task2_591def2`
- Branch/base: `submit@85b759a652b58ccd7b5d4f08ebbb6b034bd55fa1`
- Training image: `ebim-task2-pi05:200-submit-20260812`
- Live image: `ebim-task2-pi05-submit:v2-20260814`
- Raw dataset: `/scratch1/2026_ebim/allen_task2_pi05/datasets/task2_fixpos_200_46ab41f`
- V2 run root: `/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v2_12k`
- Selected checkpoint: `training_12k/checkpoints/006000/pretrained_model`

## V2 model/data change

- Existing V1 checkpoints retain their relative-spine decoding contract.
- V2 keeps both arm targets relative but learns action 19 (spine) as an
  absolute simulator target.
- V2 spine action q01/q50/q99 is `0.45 / 0.50 / 0.55 m`.
- Six event phases are sampled at approximately
  `20/20/20/15/15/10%`: startup rise, approach, grasp acquisition, lift and
  transfer, lower and place, release and retreat.
- Training starts from `lerobot/pi05_base`, with batch 1, frozen vision, and
  action-expert finetuning. Only the 6k checkpoint was retained and evaluated;
  training was intentionally stopped before 12k after the offline spine gate
  passed.

## Source and offline gates

- Focused unit tests: **46 passed**.
- Python compile, shell syntax, and `git diff --check`: passed.
- LeRobot/Draccus parser gate: passed with action 19 mapping `null`.
- Portable reproduction dry-run and live image health check: passed.
- Training step 6000: loss `0.022`, gradient norm `0.631`, GPU memory
  `12.87 GB`.
- Injected low-spine recovery gate: all 160 downstream chunks predicted their
  first-25 spine targets above `0.40 m`; mean was about `0.4986 m`.
- Event-landmark offline spine MAE was `0.0038–0.0061 m`. Right-gripper close
  accuracy was weaker (`0.65` at the close landmark), so the 6k checkpoint was
  a spine GO but not proof of grasp success.

## Evidence-first runner finding and fix

The first V2 GUI run published 600 actions in 22.79 wall seconds using
`host_monotonic`. Its policy targets were approximately `0.49–0.51 m`, but the
measured spine ended at only `0.3071 m`. Episode 9 reaches `0.48 m` at frame
202, or 6.73 simulator seconds. The existing replay evidence reports GUI
RTF `0.2116`, proving that host-paced 30 Hz commands consumed the policy
trajectory about five times faster than simulator physics.

The live runner now uses `/isaac/clock` for 30 Hz action pacing, chunk
alignment, and the action-queue watchdog. Camera/state freshness, process
timeout, and stalled-stream safety remain on host monotonic time. No safety
threshold was relaxed.

A 12-action preflight exposed and fixed the same clock-domain bug in the
queue watchdog: it had treated a valid 1.67-simulator-second chunk as stale
after two wall seconds at low RTF. This preflight covered only 0.367 simulator
seconds and did not reach manipulation.

## Corrected 600-action GUI result

Manifest:
`/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260814_034540/live_runner_manifest.json`

- Completed: **true**; 600/600 actions, 22/22 valid decisions, zero invalid
  actions, no queue underflow, no publication block.
- Timing: `19.8833` simulator seconds over `114.264` wall seconds.
- Spine: `0.4884 m` at action 200, max `0.5187 m`, final `0.4824 m`.
  This resolves the original `~0.38 m`/slow-rise spine symptom and matches the
  episode-9 timing closely.
- Right-gripper predictions: minimum `0.9291`; zero actions below `0.5`.
  The policy never issued a grasp close during the 600-action run.
- Final thermal-pad pose remained exactly at its spawn pose
  `[1.75, 1.95, 0.85]`; the pad was not grasped.
- Operator visual observation: the arms moved upward with the spine rather
  than holding the dataset's world height; the resulting wrist-camera views
  diverged and the right arm drifted.

## Paused next step

Before any 12k continuation or new imitation-learning model:

1. Compare dataset and live right/left end-effector world-Z versus
   `spine.height` for frames/actions 0–250.
2. Trace whether dataset arm joint targets, runtime relative-arm decoding, and
   simulator articulation control all assume the same spine/root frame.
3. Run a short, bounded GUI contract probe that raises the spine while holding
   the demonstrated frame-0 end-effector world pose. Confirm that wrist-camera
   views stay aligned with dataset frame 0.
4. Only after parity passes, re-evaluate the existing 6k checkpoint. If the
   gripper still never closes, continue one final checkpoint with heavier
   grasp-acquisition/close sampling rather than a broad retrain.

No replay work is requested, no 12k checkpoint was created, and the simulator
was stopped after evidence capture.
