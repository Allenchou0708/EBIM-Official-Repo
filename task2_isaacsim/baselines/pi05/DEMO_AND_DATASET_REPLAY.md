# Task 2 PI0.5 V2-full 30k operator runbook

This runbook covers one experiment only: a fresh 30k run from
`lerobot/pi05_base` with the V2 data/action contract, phase-balanced sampling,
`train_expert_only=false`, and the vision encoder still frozen. Arms are
relative, action 19 (spine) is absolute, `n_action_steps=5`, and live execution
uses `hard5` indices 0--4.

Do not run V3 or V4. V4 already failed its held-out offline gate. Do not add an
IK arm override or loosen the `0.10 s` camera/state capture-skew gate.

## One-time environment

Run this block in every terminal:

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export PI05_CONFIG=configs/task2_fixpos_200_v2_full_30k.yaml
export PI05_RUN=task2_pi05_v2_full_30k
export PI05_SMOKE_RUN=task2_pi05_v2_full_30k_smoke_1step
export PI05_RUN_ROOT="$TASK2_PI05_ROOT/outputs/$PI05_RUN"
export PI05_DATASET="$TASK2_PI05_ROOT/datasets/task2_fixpos_200_46ab41f"
export PI05_STAGING_AUDIT="$TASK2_PI05_ROOT/evidence/task2_pi05_v2_full_30k_preflight/startup_staging_audit.json"
export PI05_CKPT_015="$PI05_RUN_ROOT/training/checkpoints/015000/pretrained_model"
export PI05_CKPT_030="$PI05_RUN_ROOT/training/checkpoints/030000/pretrained_model"
export PI05_RELATIVE_DATASET="$PI05_RUN_ROOT/relative_dataset"
export PI05_TRAIN_IMAGE=ebim-task2-pi05:200-submit-20260812
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:v3-hard5-20260814
export ROS_DOMAIN_ID=62
```

If `.env.pi05` exists, ensure it does not override these values. Bulk data,
logs, checkpoints, and images stay below `$TASK2_PI05_ROOT`, never in Git.

Before an experiment boundary, require the pushed `submit` ref and a clean
tree:

```bash
test "$(git branch --show-current)" = submit
test "$(git rev-parse HEAD)" = "$(git rev-parse collab/submit)"
git diff --quiet
git diff --cached --quiet
git status --short --branch
```

## Dataset and preflight

Download only if needed, then rerun the immutable dataset audit:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh dataset \
  --config "$PI05_CONFIG"
```

Build the phase manifest, 199-episode pre-grasp audit, and dataset-derived
staging route:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh audit-staging
```

The selected route must report `guessed_ik_used=false`, train episode 176,
frame 408, open grippers, a `0.5 m` spine command (measured reference about
`0.485743 m`), legal arm targets, and the right local-Y tool axis as vertical.
The exact vector and distributions are in the audit/report. This is a
robust-median representative, not a hand-written pose. Staging follows its raw
frame-0-to-408 targets with simulator-time velocity limiting. It first ramps
from the measured reset state to dataset frame 0 under those same limits, and
requires all arm/spine/gripper/EE checks stable for `1.0 simulator-second`;
timeout stops.

Run the parser/config and focused source gates:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh parser-gate \
  --profile v2_full_30k

docker run --rm --entrypoint bash \
  -e PYTHONPATH=/opt/ebim \
  -v "$PWD:/opt/ebim:ro" \
  "$PI05_TRAIN_IMAGE" -lc \
  'python -m unittest -q task2_isaacsim.tests.test_pi05_live_runner task2_isaacsim.tests.test_pi05_contract'

bash -n task2_isaacsim/baselines/pi05/run_pi05.sh
git diff --check
```

The parser output must show `train_expert_only=false`,
`freeze_vision_encoder=true`, and mapping index 19 as `null`. The GPU doctor is
also intentional:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh doctor \
  --profile v2_full_30k
```

On the current RTX 5090 it reports about `31.35 GiB` and exits NO-GO until a
same-contract one-step smoke succeeds. Do not bypass this with expert-only,
LoRA, smaller data, or an unfrozen vision encoder.

## One-step VRAM gate and 30k training

The smoke uses the full train split and identical model/data contract, but
runs one optimizer step, logs that step, and saves no checkpoint:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh train \
  --config "$PI05_CONFIG" \
  --run "$PI05_SMOKE_RUN" \
  --one-step-smoke --dry-run

test ! -e "$TASK2_PI05_ROOT/outputs/$PI05_SMOKE_RUN"
bash task2_isaacsim/baselines/pi05/run_pi05.sh train \
  --config "$PI05_CONFIG" \
  --run "$PI05_SMOKE_RUN" \
  --one-step-smoke
```

Verify its manifest before proceeding:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-smoke \
  --run "$PI05_SMOKE_RUN"
```

Any OOM, nonzero return code, trainable count below 3.5B, missing finite loss,
or missing memory metric is NO-GO. If it passes, the operator starts the fresh
30k run in the terminal. It initializes from `lerobot/pi05_base`; it is not a
V2/V4 continuation:

```bash
test ! -e "$PI05_RUN_ROOT"
bash task2_isaacsim/baselines/pi05/run_pi05.sh train \
  --config "$PI05_CONFIG" \
  --run "$PI05_RUN" \
  --vram-smoke-run "$PI05_SMOKE_RUN"
```

This foreground command writes `train.log` and `run_manifest.json`. It uses
30,000 steps and `save_freq=15000`; the only numbered checkpoints must be
`015000` and `030000` (`last` may be a pointer). Do not start a second training
process or a GUI simulator while it is running.

## Completed-training checks and offline gate

After training exits zero, run the strict manifest/config/log/checkpoint check:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-training \
  --run "$PI05_RUN"
```

It verifies the exact two numbered checkpoints, model and optimizer state,
training step, finite final loss, full-parameter mode, base-model revision,
frozen vision, `n_action_steps=5`, 30k/15k cadence, and the V2 action mapping.
Useful read-only size checks are:

```bash
find "$PI05_RUN_ROOT/training/checkpoints" -maxdepth 1 \
  -type d -name '[0-9][0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | sort
du -sh "$PI05_RUN_ROOT" \
  "$PI05_RUN_ROOT/training/checkpoints/015000" \
  "$PI05_RUN_ROOT/training/checkpoints/030000"
tail -n 80 "$PI05_RUN_ROOT/train.log"
```

Then evaluate both numbered checkpoints on the immutable 20-episode held-out
split. This performs no ROS publication:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh offline-gate \
  --run "$PI05_RUN" --max-frames 128
```

Both entries in `offline_gate.json` must have finite loss, finite 20-D output,
valid joint/gripper bounds, and reproducible offline replay. Failure is NO-GO
for GUI, even if the other checkpoint passes.

## GUI, shadow, hard5, monitor, and evidence

### Terminal 1: simulator

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh sim-up --gui
```

Wait for the room, dual-FR3 robot, pad, boards, `/isaac/clock`, and cameras.
The launcher log is persisted below
`$TASK2_PI05_ROOT/evidence/task2_200_submit_20260812/launcher/`.

### Checkpoint 015000

First run a zero-policy-publication shadow. It resets the scene, loads the
policy, follows the verified fixed-base route, runs the audited pre-grasp joint
staging, discards all staging observations, captures a fresh settled three-
camera/joint/odom/two-EE tuple, and forms one decision:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_CKPT_015" \
  --dataset-root "$PI05_RELATIVE_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label 015000-shadow \
  --shadow --max-actions 5 --max-duration-s 60
```

Locate and verify only that labeled output:

```bash
export PI05_SHADOW_015="$(ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_015000-shadow_* | head -1)"
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-shadow \
  --run-dir "$PI05_SHADOW_015"
```

Open `$PI05_SHADOW_015/settled_fresh_wrist_right.ppm` and verify that the pad
front is actually visible. If it is not, stop: do not pass the confirmation
flag and do not relax freshness/skew. After visual confirmation, run formal
hard5:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_CKPT_015" \
  --dataset-root "$PI05_RELATIVE_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label 015000-hard5 \
  --confirm-right-wrist-pad-visible \
  --max-actions 600 --max-duration-s 300
```

### Checkpoint 030000

Do not reuse the 015000 output variables. Run a new shadow and inspect its own
right-wrist frame:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_CKPT_030" \
  --dataset-root "$PI05_RELATIVE_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label 030000-shadow \
  --shadow --max-actions 5 --max-duration-s 60

export PI05_SHADOW_030="$(ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_030000-shadow_* | head -1)"
test -f "$PI05_SHADOW_030/live_runner_manifest.json"
test -f "$PI05_SHADOW_030/settled_fresh_wrist_right.ppm"
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-shadow \
  --run-dir "$PI05_SHADOW_030"
```

Visually inspect its PPM. Only after both gates pass:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_CKPT_030" \
  --dataset-root "$PI05_RELATIVE_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label 030000-hard5 \
  --confirm-right-wrist-pad-visible \
  --max-actions 600 --max-duration-s 300
```

Each formal run must execute only chunk indices 0--4 per decision. During
inference it republishes the last legal absolute target using `/isaac/clock`;
arm/gripper `JointState` headers use the current simulator stamp. Host monotonic
time is limited to process/transport watchdogs. Stop immediately for unintended
contact, base input, publisher contention, stale streams, bound projection
failure, or staging timeout.

### Monitor, evaluator, and stop

Optional read-only ROS monitors from a sourced ROS Jazzy terminal:

```bash
ros2 topic echo /isaac/clock --once
ros2 topic echo /isaac/joint_states_full --once
ros2 topic echo /isaac/odom --once
ros2 topic info /isaac/spine_target --verbose
```

Run the evaluator only after a meaningful grasp/lift/placement attempt:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh evaluate
```

Press Ctrl-C in the runner terminal first, then Ctrl-C in the simulator
terminal, then clean up:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh down
```

List evidence without mixing checkpoints:

```bash
ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_015000-{shadow,hard5}_* 2>/dev/null
ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_030000-{shadow,hard5}_* 2>/dev/null
find "$TASK2_PI05_ROOT/evidence/task2_pi05_v2_full_30k_preflight" \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

## GO/NO-GO summary

GUI policy publication is GO only when all of these pass: pushed clean source;
parser mapping with absolute spine and `train_expert_only=false`; successful
same-contract VRAM smoke; training verification; both-checkpoint offline gate;
focused tests; dataset audit; staging execution with every feedback group in
tolerance; one valid shadow decision with zero policy command publications;
fresh full observation skew at or below `0.10 s`; and visual confirmation that
the right-wrist frame sees the pad front. Any failure is NO-GO.

Historical details are intentionally outside this operator path:

- V2 action/phase result: `TASK2_PI05_V2_ABSOLUTE_SPINE_PHASE_BALANCED_LAB_RESULT_2026-08-14.md`.
- Clock/load/GUI crash fixes: `TASK2_PI05_STAGE_CLOCK_AND_GUI_CRASH_FIX_2026-08-14.md`.
- Pre-grasp evidence: `TASK2_PI05_PRELOAD_AND_PREGRASP_AUDIT_2026-08-14.md`.
- V4 NO-GO: `TASK2_PI05_V4_SEVEN_PROMPT_OFFLINE_GATE_LAB_RESULT_2026-08-14.md`.

Raw replay and V1/V2 comparisons are historical contract evidence only and are
not rerun in this experiment.
