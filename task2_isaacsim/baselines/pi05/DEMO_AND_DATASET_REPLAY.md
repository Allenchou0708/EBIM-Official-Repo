# Task 2 PI0.5 V1/V2 030000 operator runbook

This runbook tests exactly two models: V1 step 030000 and V2 expert-only step
030000. Do not test V3, V4, or either 015000 checkpoint. V1 decodes spine
action 19 relative to state index 28; V2 keeps action 19 as an absolute target.
Both use relative arm actions, frozen vision, `train_expert_only=true`,
`n_action_steps=5`, and GUI `hard5` indices 0--4.

## One-time environment

Run in every terminal:

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export PI05_TRAIN_IMAGE=ebim-task2-pi05:200-submit-20260812
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:v3-hard5-20260814
export PI05_DATASET="$TASK2_PI05_ROOT/datasets/task2_fixpos_200_46ab41f"
export PI05_STAGING_AUDIT="$TASK2_PI05_ROOT/evidence/task2_pi05_v2_full_30k_preflight/startup_staging_audit.json"
export PI05_V1_ROOT="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1"
export PI05_V2_ROOT="$TASK2_PI05_ROOT/outputs/task2_pi05_v2_expert_30k"
export PI05_V1_CKPT="$PI05_V1_ROOT/training/checkpoints/030000/pretrained_model"
export PI05_V2_CKPT="$PI05_V2_ROOT/training/checkpoints/030000/pretrained_model"
export PI05_V1_DATASET="$PI05_V1_ROOT/relative_dataset"
export PI05_V2_DATASET="$PI05_V2_ROOT/relative_dataset"
export ROS_DOMAIN_ID=62
```

Bulk checkpoints, logs, images, and evidence stay below `$TASK2_PI05_ROOT`, not
in Git. Before a GUI boundary, require branch/ref consistency; also inspect any
listed worktree changes rather than silently discarding them:

```bash
test "$(git branch --show-current)" = submit
test "$(git rev-parse HEAD)" = "$(git rev-parse collab/submit)"
git status --short --branch
```

## Dataset and source preflight

The dataset is immutable revision
`46ab41f16fe836ee8ca791c7afaade44783eefe6`. Download only if absent, then
refresh its audit and deterministic staging evidence:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh dataset \
  --config configs/task2_fixpos_200_v2_expert_30k.yaml
bash task2_isaacsim/baselines/pi05/run_pi05.sh audit-staging
```

The staging audit must say `guessed_ik_used=false` and select train episode
176, frame 408. Its target is a dataset-derived joint-space route with open
grippers, a 0.5 m spine command (measured reference about 0.485743 m), left
hand near the table, and audited vertical right pre-grasp. It ramps from the
measured reset state at dataset-derived velocity limits and requires every
arm/spine/gripper/EE group stable for 1.0 simulator-second. Timeout or any
tolerance failure stops the run; there is no IK override.

Run parser and source gates:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh parser-gate \
  --profile v2_expert_30k

docker run --rm --entrypoint bash \
  -e PYTHONPATH=/opt/ebim \
  -v "$PWD:/opt/ebim:ro" \
  "$PI05_TRAIN_IMAGE" -lc \
  'python -m unittest -q task2_isaacsim.tests.test_pi05_live_runner task2_isaacsim.tests.test_pi05_contract'

bash -n task2_isaacsim/baselines/pi05/run_pi05.sh
git diff --check
```

The V2 parser gate must show index 19 as `null`, expert-only true, frozen
vision, and five action steps. The selected-checkpoint gate checks V1 and V2
separately:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-models-30k
```

## Completed V2 training provenance

V2 training is complete. Do not rerun it for this comparison. The exact
operator command was:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh train \
  --config configs/task2_fixpos_200_v2_expert_30k.yaml \
  --run task2_pi05_v2_expert_30k
```

The checkpoint is authoritative: actual batch size was 1, not 2. The outer
YAML edit to 2 was not passed into the LeRobot profile. The run completed
30,000 steps in about three hours, final logged loss 0.009, with about 12.87
GiB reported GPU memory. Verify its log, manifest, exact 015000/030000 cadence,
and optimizer state with:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-training \
  --run task2_pi05_v2_expert_30k --mode v2_expert_30k
tail -n 80 "$PI05_V2_ROOT/train.log"
find "$PI05_V2_ROOT/training/checkpoints" -maxdepth 1 \
  -type d -name '[0-9][0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | sort
```

## Offline V1/V2 030000 gates

These commands mount only each model's 030000 checkpoint and publish no ROS
commands. The first verifies deterministic finite 20-D outputs on 60 samples
per model; the second computes native loss over 128 frames and repeats a
20-frame bounds/reproducibility shadow:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh offline-models-30k \
  --samples-per-episode 3
bash task2_isaacsim/baselines/pi05/run_pi05.sh heldout-models-30k \
  --max-frames 128
```

Current results are PASS for both. Mean held-out loss is 11.507817 for V1 and
11.411030 for V2. Evidence is under:

```bash
find "$TASK2_PI05_ROOT/evidence/task2_pi05_v1_v2_30k" \
  -maxdepth 2 -type f -printf '%P\n' | sort
```

Do not rerun an existing held-out gate in place; it intentionally refuses to
overwrite evidence.

## GUI simulator and model-isolated runs

### Terminal 1: simulator

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh sim-up --gui
```

Wait until the room, dual-FR3 robot, pad, boards, cameras, and `/isaac/clock`
are active. Keep this terminal visible for simulator errors.

### V1 030000: shadow, inspect, hard5

Run shadow first. It resets the scene, executes base and manipulation staging,
discards all pre-staging observations, captures a settled fresh tuple, and
forms one decision with zero policy command publications:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V1_CKPT" \
  --dataset-root "$PI05_V1_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label v1-030000-shadow \
  --shadow --max-actions 5 --max-duration-s 60

export PI05_V1_SHADOW="$(ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_v1-030000-shadow_* | head -1)"
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-shadow \
  --run-dir "$PI05_V1_SHADOW" --contract v1
```

Open `$PI05_V1_SHADOW/settled_fresh_wrist_right.ppm` and confirm the right
wrist camera sees the pad front. If it does not, V1 is NO-GO. After visual
confirmation only:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V1_CKPT" \
  --dataset-root "$PI05_V1_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label v1-030000-hard5 \
  --confirm-right-wrist-pad-visible \
  --max-actions 600 --max-duration-s 300
```

Stop and reset before switching contracts. Never reuse V1 output variables
for V2.

### V2 030000: shadow, inspect, hard5

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V2_CKPT" \
  --dataset-root "$PI05_V2_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label v2-030000-shadow \
  --shadow --max-actions 5 --max-duration-s 60

export PI05_V2_SHADOW="$(ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_v2-030000-shadow_* | head -1)"
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-shadow \
  --run-dir "$PI05_V2_SHADOW" --contract v2
```

Open `$PI05_V2_SHADOW/settled_fresh_wrist_right.ppm` and confirm the pad front
is visible. After visual confirmation only:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V2_CKPT" \
  --dataset-root "$PI05_V2_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label v2-030000-hard5 \
  --confirm-right-wrist-pad-visible \
  --max-actions 600 --max-duration-s 300
```

## Clock, monitor, stop, and evidence

The control contract is already implemented and source-tested:

- base pulses, actuator publication, staging dwell, and hard5 pacing use
  `/isaac/clock`;
- arm/gripper `JointState` commands carry the current simulator stamp;
- host monotonic time is used only for process/transport watchdogs;
- policy inference starts only after staging observations are discarded and a
  new three-camera + joint + odom + two-EE tuple passes freshness and capture
  skew at or below 0.10 s;
- capture-to-ready inference latency is recorded in simulator time;
- hard5 executes only indices 0--4 and holds the last legal absolute target
  during the next inference;
- bounds, fixed-base projection, and publisher contention remain fail-closed.

Read-only monitors from a sourced ROS Jazzy terminal:

```bash
ros2 topic echo /isaac/clock --once
ros2 topic echo /isaac/joint_states_full --once
ros2 topic echo /isaac/odom --once
ros2 topic info /isaac/spine_target --verbose
```

Stop immediately for contact risk, clock reset/stall, stale/skewed inputs,
publisher contention, staging timeout/tolerance failure, or projection failure.
Press Ctrl-C in the runner, then simulator, then:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh down
```

Keep outputs separated:

```bash
ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_v1-030000-{shadow,hard5}_* 2>/dev/null
ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_v2-030000-{shadow,hard5}_* 2>/dev/null
```

GUI publication is GO per model only after checkpoint/config, source tests,
dataset staging audit, offline loss/replay, executed staging manifest, zero-
publication shadow, fresh tuple, and right-wrist visual confirmation all pass.
Any failure is NO-GO for that model. Never loosen freshness/skew thresholds and
never introduce guessed IK.

Historical detail: [V2 action/phase result](TASK2_PI05_V2_ABSOLUTE_SPINE_PHASE_BALANCED_LAB_RESULT_2026-08-14.md),
[clock and GUI crash fixes](TASK2_PI05_STAGE_CLOCK_AND_GUI_CRASH_FIX_2026-08-14.md),
[pre-grasp audit](TASK2_PI05_PRELOAD_AND_PREGRASP_AUDIT_2026-08-14.md),
[full-mode OOM fallback](TASK2_PI05_V2_FULL_OOM_AND_EXPERT_30K_FALLBACK_2026-08-14.md),
and [V4 NO-GO](TASK2_PI05_V4_SEVEN_PROMPT_OFFLINE_GATE_LAB_RESULT_2026-08-14.md).
