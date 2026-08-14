# PI0.5 V1/V2 GUI comparison and organizer dataset replay

This runbook provides two independent paths:

- an apples-to-apples GUI comparison of the V1 30k and V2 6k
  checkpoints, using the same final `submit` runner, runtime image, hard5
  horizon, scene staging, and action budget;
- a model-free organizer trajectory replay, which publishes one raw 20-D
  training episode according to its recorded timestamps and `/isaac/clock` to
  arms, grippers, and spine while publishing no base command.

The V1/V2 comparison is an operator experiment, not a new training or tuning
run. V2 already completed one verified 600-action GUI attempt without grasping
the pad. Dataset replay is only a motion-contract test unless the episode's
object poses are also reconstructed in the live scene. Neither path by itself
proves Task 2 success.

The commands below use the current pushed `submit` revision documented in
`TASK2_PI05_STAGE_CLOCK_AND_GUI_CRASH_FIX_2026-08-14.md`. Its default `hard5`
runner executes checkpoint action indices `0..4`, holds the last legal
absolute target during inference, paces commands with
`/isaac/clock`, and measures cross-camera capture skew from ROS image
timestamps. It additionally requires joint, odom, and both EE header stamps to
be within `0.10 s` of that camera tuple and stamps arm and gripper `JointState`
commands with the current `/isaac/clock` sample.

## One-time environment

Run this block in every terminal. Change the paths when relocating bulk
artifacts. ROS domain 62 was used for the verified V1 recovery and V2 GUI runs.
If `task2_isaacsim/baselines/pi05/.env.pi05` exists, make sure it does not
override these root, image, or ROS-domain values.

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export PI05_V1_CHECKPOINT="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1/training/checkpoints/030000/pretrained_model"
export PI05_V1_RELATIVE_DATASET="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1/relative_dataset"
export PI05_V2_CHECKPOINT="$TASK2_PI05_ROOT/outputs/task2_pi05_v2_12k/training_12k/checkpoints/006000/pretrained_model"
export PI05_V2_RELATIVE_DATASET="$TASK2_PI05_ROOT/outputs/task2_pi05_v2_12k/relative_dataset"
export PI05_RAW_DATASET="$TASK2_PI05_ROOT/datasets/task2_fixpos_200_46ab41f"
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:v3-hard5-20260814
export ROS_DOMAIN_ID=62
```

Before starting the simulator, verify that the comparison inputs exist and the
runner is at the intended commit:

```bash
test "$(git branch --show-current)" = submit
test "$(git rev-parse HEAD)" = "$(git rev-parse collab/submit)"
git diff --quiet
git diff --cached --quiet
test -d "$PI05_V1_CHECKPOINT"
test -d "$PI05_V1_RELATIVE_DATASET"
test -d "$PI05_V2_CHECKPOINT"
test -d "$PI05_V2_RELATIVE_DATASET"
docker image inspect "$PI05_LIVE_IMAGE" --format '{{.Id}} {{.RepoTags}}'
git status --short --branch
```

Do not proceed if the branch, remote-tracking, or clean-tree checks fail.

Generated run summaries and traces are written below `$TASK2_PI05_ROOT/outputs/`;
final lab reports and the 200-episode audit are below
`$TASK2_PI05_ROOT/evidence/`.

## Terminal 1: GUI simulator

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh sim-up --gui
```

Wait until the room, mobile dual-FR3 robot, thermal pad, RAM boards, and camera
streams have initialized before starting another command path. The launcher
prints the persistent Isaac/Kit log path under
`$TASK2_PI05_ROOT/evidence/task2_200_submit_20260812/launcher/`; its final line
records `isaac_gui_exit_code`.

## Terminal 2A: V1 shadow and GUI

Run the V1 shadow after the simulator is ready. It loads the real checkpoint
and forms one decision, but creates no command publishers:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V1_CHECKPOINT" \
  --dataset-root "$PI05_V1_RELATIVE_DATASET" \
  --shadow --max-actions 5 --max-duration-s 60
```

Only after the shadow reports one valid decision, zero command publications,
and no freshness, bounds, or contention failure, run the V1 GUI attempt:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V1_CHECKPOINT" \
  --dataset-root "$PI05_V1_RELATIVE_DATASET" \
  --max-actions 600 \
  --max-duration-s 300
```

Let the runner exit before starting V2. Press Ctrl-C immediately for unintended
contact, unstable motion, or an operator safety concern.

## Terminal 2B: V2 shadow and GUI

The V2 shadow performs another clean scene reset and stages the same base and
spine start state:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V2_CHECKPOINT" \
  --dataset-root "$PI05_V2_RELATIVE_DATASET" \
  --shadow --max-actions 5 --max-duration-s 60
```

Only after that shadow passes, run the V2 GUI attempt with the same bounds as
V1:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V2_CHECKPOINT" \
  --dataset-root "$PI05_V2_RELATIVE_DATASET" \
  --max-actions 600 \
  --max-duration-s 300
```

Every `run-task` invocation requests a scene reset, aligns the spine to 0 m,
and runs the evaluator-camera preflight. The runner then loads the checkpoint
before following the fixed base route to approximately
`(2.10, 3.05, -1.571)`. It discards observations captured during staging and
requires a fresh, settled tuple before inference. During manipulation the base
is fixed and action 19 controls the spine. Use each checkpoint's matching
relative-dataset view because V1 and V2 serialize different spine decoding
contracts. Base command pulses, braking pauses, and settle duration use
`/isaac/clock`; correction pulses are `0.05 s` (`0.10 s` forward/back), and
the 90-second process watchdog remains host-monotonic.

Do not use `--runtime-mode legacy`, the old
`ebim-task2-pi05-submit:final-20260813` image, or dataset replay during this
comparison. Those changes would confound checkpoint behavior with runner,
image, or controller behavior.

## What to compare and where to find the manifests

For both GUI attempts, observe the same milestones:

1. Whether the spine rises and its approximate working height.
2. Whether both wrists compensate for spine motion or ride upward with it.
3. Whether the right gripper approaches and aligns with the thermal pad.
4. Whether the right gripper visibly closes on the pad.
5. Whether the pad lifts or moves toward the target RAM board.

Each shadow and GUI invocation creates a new timestamped directory below
`$TASK2_PI05_ROOT/outputs/live_submit_*`. After each run, note its path before
starting the next command. To list the newest outputs:

```bash
ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_* | head
```

The primary numerical evidence is `live_runner_manifest.json` inside that
directory. Keep the V1 and V2 GUI directory names with your visual notes so the
two runs are not confused.

The known V2 simulator-clock reference completed `600/600` actions and reached
spine `0.4884 m` at action 200, maximum `0.5187 m`, and final `0.4824 m`, but
both wrists rose too far, the right gripper produced `0/600` close actions,
and the pad was not grasped. That older result predates hard5, so the current
V2 run is a new execution-horizon comparison. Prior V1 GUI evidence covered
only 25 actions, so the 600-action V1 run is also a new full-horizon
observation.

Do not loosen the camera freshness/skew thresholds. Startup now reports an
explicit inventory and retries its DDS participant once before model load.
For the current runner, a valid event must also contain all four state
capture stamps and `observation_capture_skew_s <= 0.10`. Check the newest
shadow manifest before arming either checkpoint:

```bash
PI05_LAST_RUN="$(ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_* | head -1)"
python3 - "$PI05_LAST_RUN/live_runner_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
event = manifest["events"][0]
fresh = event["freshness"]
print("valid:", event["valid"])
print("camera stamps:", fresh["camera_capture_times_s"])
print("state stamps:", fresh["state_capture_times_s"])
print("camera/state skew s:", fresh["observation_capture_skew_s"])
print("capture-to-ready sim s:", event["capture_to_ready_sim_s"])
print("command publications:", manifest["command_publications"])
PY
```

The capture-to-ready value is causal policy latency, not a synchronization
failure. It must be reported rather than described as a same-time action: in
`hard5`, index 0 starts when inference is ready and is then paced on simulator
time.

## Organizer dataset replay (not part of the V1/V2 comparison)

Offline summary, with no simulator control:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --episode 9 --summary-only
```

Alignment-only gate:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --episode 9 --align-only
```

First-30-frame direction and tracking gate:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --episode 9 --max-frames 30
```

Full replay, only after both gates pass:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --episode 9
```

The default `--episode auto` reads the committed train/held-out audit and picks
the train episode whose frame-0 base, spine, and arm state is closest to the
room ready pose. Use `--episode N` only for an intentional comparison. The
loader rejects non-contiguous frames, non-finite or non-20D actions, non-37D
states, arm targets outside FR3 limits, grippers outside `[0,1]`, and spine
outside the demonstrated `[0,0.6]` m range. It never loads a checkpoint or a
mapped-relative dataset.

Before replay, the launcher resets the scene and stages the base to the chosen
episode's frame-0 odometry. The ROS node then aligns arms, grippers, and spine;
it refuses to start the trajectory unless all tolerances remain satisfied for
one second. Only action indices `3:20` are published. The node creates no base
publisher, refuses existing command publishers, records live 37-D state, and
leaves the simulator at the last legal target after Ctrl-C. Each raw row is
released only when `/isaac/clock` reaches the row's recorded timestamp; no
wall-clock 30 FPS sleep is used and action values/order remain unchanged.

The 200-episode initial-state audit is offline and may be run independently:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh audit-initial-states \
  --dataset-root "$PI05_RAW_DATASET"
```

## Terminal 3: monitor and conditional evaluator

Optional read-only monitors can run from a ROS Jazzy host shell:

```bash
ros2 topic echo /isaac/joint_states_full --once
ros2 topic echo /isaac/odom --once
ros2 topic info /isaac/spine_target --verbose
```

For V1/V2, run the official evaluator only if the right gripper actually
closes on the pad and creates a meaningful lift or transfer attempt. For raw
replay, run it only after trajectory Gate B passes and the live object reset is
known to match the selected episode:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh evaluate
```

Episode 9 has `task2_extras/episode_000009.npz` with frame-wise poses for the
thermal pad, target, base, and three boards. The strict tracking gate did not
pass, so no runtime object-pose injection or evaluator call was made in this
handoff. A missed grasp in an unmatched reset is not evidence that the dataset
is wrong.

## Stop and cleanup

Press Ctrl-C in Terminal 2 first. A VLA runner stops publication safely; replay
stops publishing and leaves the last legal target active. Then press Ctrl-C in
Terminal 1 and run:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh down
```

## Expected and abnormal behavior

Expected staging moves the base from the room reset near `(4.4, 2.6)` to the
dataset pose near `(2.10, 3.05, -1.571)`. Episode 9 begins with both arms in the
ready pose, open grippers, and spine at 0 m. Its raw command milestones are
spine 0.10 m at frame 38, spine 0.30 m at frame 42, and right-gripper close at
frame 464.

Stop and investigate if base staging does not pass, alignment times out, a
second command publisher is reported, action validation fails, or either VLA
reports `live_stream_stale`. Camera stale/skew is a safety stop, not an
instruction to raise thresholds.

## Verified simulator-clock replay result

- Dataset: `task2_fixpos_200_46ab41f`; explicit episode 9.
- Raw trajectory: 956 frames; timestamps 0.0--31.833334 s with median step
  0.0333333 s.
- Frame-0 alignment PASS: arm max 0.00487 rad, spine 0.000011 m,
  gripper 0.000025 fraction, base 0.00958 m / 0.0220 rad.
- First 30 frames PASS: 30 live states, 150 actuator publications, zero base
  publications; left/right arm mean L2 0.0339/0.0323 rad and spine mean
  absolute error 0.00000535 m.
- Full publication contract PASS: 956/956 frames in 150.47 wall seconds, 4,780
  actuator publications, 956 live 37-D records, zero base publications, and no
  interruption. `/isaac/clock` advanced 31.833335 s, for RTF 0.2116.
- Tracking improved materially: left/right arm mean L2 errors are
  0.0496/0.2122 rad and spine mean absolute error is 0.00597 m. This is an
  operationally correct replay, but the strict gate is NO-GO because right arm
  is 0.0122 rad over its 0.20 target and live right-gripper close at frame 526
  is 52 frames after recorded frame 474.
- The initial base mismatch is small and within alignment tolerance. The first
  documented actuator-limit violation is raw right joint 3 at frame 235:
  2.899 rad/s required versus the FR3 2.62 rad/s limit. Further right joint
  1/2/3 violations occur at frames 265--272 (up to 3.768 rad/s).

Evidence is in:

```text
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_174703
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_174723
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_174854
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_175044
```

The result proves raw action ordering and the ROS actuator path are wired
correctly. It does not prove grasp success. The next useful experiment is to
reproduce the organizer recording controller/drive gains and gripper dynamics,
then repeat this same raw sim-time gate. Do not smooth or resample the raw
episode merely to pass the metric.

## Controller and scene parity stop decision

The organizer keyboard path and the ROS replay path do not apply an equivalent
arm controller input. At every 60 Hz render tick, the organizer's RMPflow path
applies both `joint_positions` and `joint_velocities`. The recorder publishes
only `get_applied_action().joint_positions`, and the 30 Hz dataset action stores
only arm position targets. The RMPflow velocity targets and the intervening
60 Hz target sequence therefore cannot be recovered from episode 9, the
dataset metadata, or repository history. The current ROS command subscriber
applies position targets only. Gripper commands are position-only in both
paths, but the dataset also contains no drive/gain provenance that explains the
52-frame right-gripper response offset.

No controller patch was made: deriving velocities from adjacent 30 Hz samples
would be a new guessed controller, not organizer parity, and an open-ended
drive/gain sweep is outside this deadline path. The unchanged simulator-clock
baseline remains controller Gate NO-GO because right-arm mean L2 is 0.2122 rad
and right-gripper close offset is +52 frames. Consequently no additional raw
replay or evaluator was run for the controller-parity handoff.

After a clean room reset, all six published rigid-object poses match episode 9
frame 0 to floating-point precision. The 14 arm joints, 0 m spine, and open
grippers also match their recorded ready state within normal simulation
settling. The room reset base is intentionally near
`(4.400, 2.621, -1.571)`, while episode 9 starts near
`(2.100, 3.051, -1.571)`; the documented `fixed_stage_base.py` odometry route
closes that known staging difference before either replay or VLA inference.
No object-pose injection was added.

## Why nominally equal-rate cameras become skewed

All three sensor YAML entries document 24 Hz, not 30 Hz, but the current camera
graph builder does not consume that `publish_hz` field as a shared clock. Each
camera instead has its own playback-tick graph, render product, ROS camera
helper, serialization path, and DDS delivery queue. GUI rendering cannot
sustain a common wall-clock cadence, and the 1280x720 head frame costs more
than either 848x480 wrist frame. Equal documented rates therefore do not imply
synchronized arrival. A measured GUI run delivered head/left/right at only
6.25/8.19/10.07 Hz, with different maximum gaps.

Keep the freshness/skew stop. The current runner rejects the latest
camera/state tuple unless all ROS header stamps fit within `0.10 s`; host
monotonic time is used only for arrival age. For a short-term demo, reduce
unrelated GUI, render, and recording load and let the runner wait for that
coherent tuple. A shared capture barrier or timestamp buffer remains the
durable way to recover an older coherent tuple instead of rejecting the latest
independently arriving samples.
