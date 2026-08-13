# PI0.5 deadline demo and organizer dataset replay

This runbook provides two bounded GUI paths:

- a partial 30k PI0.5 demo, which retains every freshness, finite-action,
  bounds, emergency-stop, and publisher-contention gate;
- a model-free organizer trajectory replay, which publishes one raw 20-D
  training episode at 30 FPS to arms, grippers, and spine while publishing no
  base command.

Neither path by itself proves Task 2 success. The partial VLA remains a known
Pick 0 / IoU 0 candidate. Dataset replay is a motion-contract test unless the
episode's object poses are also reconstructed in the live scene.

## One-time environment

Run this block in every terminal. Change the four paths when relocating bulk
artifacts. ROS domain 61 was used for the 2026-08-13 lab run because domain 0
contained a lingering DDS endpoint named `keyboard_to_base`.

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export PI05_CHECKPOINT="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1/training/checkpoints/030000/pretrained_model"
export PI05_RELATIVE_DATASET="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1/relative_dataset"
export PI05_RAW_DATASET="$TASK2_PI05_ROOT/datasets/task2_fixpos_200_46ab41f"
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:first-chunk-fix-20260812
export ROS_DOMAIN_ID=61
```

All generated summaries and traces are written below
`$TASK2_PI05_ROOT/outputs/`; the directory can therefore be relocated with
`TASK2_PI05_ROOT`.

## Terminal 1: GUI simulator

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh sim-up --gui
```

Wait until the room, mobile dual-FR3 robot, thermal pad, RAM boards, and camera
streams have initialized before starting another command path.

## Terminal 2A: bounded partial PI0.5 demo

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --checkpoint "$PI05_CHECKPOINT" \
  --dataset-root "$PI05_RELATIVE_DATASET" \
  --max-actions 25 \
  --max-duration-s 300
```

The launcher requests a scene reset, follows the fixed base route to
approximately `(2.10, 3.05, -1.571)`, aligns spine to 0 m, runs the evaluator
camera preflight, loads the 30k checkpoint, and permits at most 25 actions or
300 seconds. During manipulation the base is fixed and action 19 controls the
spine.

Lab result on 2026-08-13: base staging and preflight passed, but the runner
formed no inference-ready observation in 300 seconds and published zero
actions. The manifest is:

```text
/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260813_152438/live_runner_manifest.json
```

Therefore this exact partial demo entrance is reproducible and safely bounded,
but it is not currently a usable visible VLA manipulation demo. Do not loosen
the camera freshness/skew thresholds to force it to run.

## Terminal 2B: organizer dataset replay

Offline summary, with no simulator control:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --summary-only
```

Alignment-only gate:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --align-only
```

First-30-frame direction and tracking gate:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --max-frames 30
```

Full replay, only after both gates pass:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET"
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
leaves the simulator at the last legal target after Ctrl-C.

## Terminal 3: monitor and conditional evaluator

Optional read-only monitors can run from a ROS Jazzy host shell:

```bash
ros2 topic echo /isaac/joint_states_full --once
ros2 topic echo /isaac/odom --once
ros2 topic info /isaac/spine_target --verbose
```

Run the official evaluator only after the live object reset is known to match
the selected episode:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh evaluate
```

Episode 9 has `task2_extras/episode_000009.npz` with frame-wise poses for the
thermal pad, target, base, and three boards. The current room launcher has no
runtime object-pose injection path, so the 2026-08-13 replay is classified as
**motion-contract replay only** and evaluator invocation was deliberately
withheld. A missed grasp in this mismatched reset is not evidence that the
dataset is wrong.

## Stop and cleanup

Press Ctrl-C in Terminal 2 first. The replay stops publishing and leaves the
last legal target active. Then press Ctrl-C in Terminal 1 and run:

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
second command publisher is reported, action validation fails, or the partial
VLA reports `live_stream_stale`. Camera stale/skew is a safety stop, not an
instruction to raise thresholds.

## Verified 2026-08-13 replay result

- Dataset: `task2_fixpos_200_46ab41f`; selected train episode 9.
- Raw trajectory: 956 frames, 31.87 seconds at 30 FPS.
- Frame-0 alignment PASS: arm max 0.00489 rad, spine 0.000009 m,
  gripper 0.000023 fraction, base 0.00997 m / 0.0293 rad.
- First 30 frames PASS: 30 live states, 150 actuator publications, zero base
  publications; left/right arm maximum L2 0.0741/0.0776 rad and spine maximum
  absolute error 0.0000126 m.
- Full publication contract PASS: 956/956 frames in 31.92 seconds, 4,780
  actuator publications, 956 live 37-D records, zero base publications, and no
  interruption.
- Full trajectory following is rate-limited: left/right arm mean L2 errors
  were 0.701/0.730 rad (max 1.028/2.464), spine mean/max absolute error was
  0.142/0.375 m, and right-gripper close occurred at live frame 617: 143 frames
  (4.77 seconds) after recorded state frame 474, or 153 frames (5.1 seconds)
  after raw close command frame 464. Recorded/live spine first reached 0.10 m
  at frames 69/191 and 0.30 m at frames 135/468. The live spine ultimately
  reached 0.489 m and ended at 0.487 m, close to the recorded final 0.485 m,
  but lagged during the fast command ramp.

Evidence is in:

```text
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_154551
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_154728
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_154935
```

The result proves raw action ordering and the ROS actuator path are wired
correctly. It does not prove 30 FPS target tracking or grasp success. The next
useful experiment is an offline/resampled playback analysis against actuator
velocity limits, followed by a rate-limited replay only if the demonstration's
simulation-time cadence can be reproduced without changing action semantics.

## Why nominally equal-rate cameras become skewed

All three sensor YAML entries document 24 Hz, not 30 Hz, but the current camera
graph builder does not consume that `publish_hz` field as a shared clock. Each
camera instead has its own playback-tick graph, render product, ROS camera
helper, serialization path, and DDS delivery queue. GUI rendering cannot
sustain a common wall-clock cadence, and the 1280x720 head frame costs more
than either 848x480 wrist frame. Equal documented rates therefore do not imply
synchronized arrival. A measured GUI run delivered head/left/right at only
6.25/8.19/10.07 Hz, with different maximum gaps.

Keep the freshness/skew stop. For a short-term demo, reduce unrelated GUI,
render, and recording load and wait for a coherent triplet before inference.
The durable fix is to gate on sensor/simulation timestamps and use a shared
capture barrier or timestamp buffer (approximate-time synchronization), rather
than combining the latest independently arriving image from each camera.
