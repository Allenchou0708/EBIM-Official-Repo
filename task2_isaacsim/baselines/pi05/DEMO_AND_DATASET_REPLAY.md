# PI0.5 deadline demo and organizer dataset replay

This runbook provides two bounded GUI paths:

- a partial 30k PI0.5 demo, which retains every freshness, finite-action,
  bounds, emergency-stop, and publisher-contention gate;
- a model-free organizer trajectory replay, which publishes one raw 20-D
  training episode according to its recorded timestamps and `/isaac/clock` to
  arms, grippers, and spine while publishing no base command.

Neither path by itself proves Task 2 success. The partial VLA remains a known
Pick 0 / IoU 0 candidate. Dataset replay is a motion-contract test unless the
episode's object poses are also reconstructed in the live scene.

## One-time environment

Run this block in every terminal. Change the four paths when relocating bulk
artifacts. ROS domain 62 was used for the verified 2026-08-13 recovery run.

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export PI05_CHECKPOINT="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1/training/checkpoints/030000/pretrained_model"
export PI05_RELATIVE_DATASET="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1/relative_dataset"
export PI05_RAW_DATASET="$TASK2_PI05_ROOT/datasets/task2_fixpos_200_46ab41f"
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:hard5-20260813
export ROS_DOMAIN_ID=62
```

Generated run summaries and traces are written below `$TASK2_PI05_ROOT/outputs/`;
final lab reports and the 200-episode audit are below
`$TASK2_PI05_ROOT/evidence/`.

## Terminal 1: GUI simulator

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh sim-up --gui
```

Wait until the room, mobile dual-FR3 robot, thermal pad, RAM boards, and camera
streams have initialized before starting another command path.

## Terminal 2A: zero-publication shadow gate

Run this first after the simulator is ready. It loads the real checkpoint and
forms one decision, but does not create command publishers:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --checkpoint "$PI05_CHECKPOINT" \
  --dataset-root "$PI05_RELATIVE_DATASET" \
  --shadow --max-actions 25 --max-duration-s 30
```

## Terminal 2B: bounded partial PI0.5 demo

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --checkpoint "$PI05_CHECKPOINT" \
  --dataset-root "$PI05_RELATIVE_DATASET" \
  --max-actions 25 \
  --max-duration-s 60
```

The launcher requests a scene reset, follows the fixed base route to
approximately `(2.10, 3.05, -1.571)`, aligns spine to 0 m, runs the evaluator
camera preflight, loads the 30k checkpoint, and permits at most 25 actions or
60 seconds. During manipulation the base is fixed and action 19 controls the
spine.

The earlier zero-decision run was caused by the final policy container uniquely
overriding `FASTDDS_BUILTIN_TRANSPORTS=DEFAULT`; helper containers and the image
used `UDPv4`. With the final runner restored to `UDPv4`, head, both wrists, both
EE poses, odometry, and every joint required by the 37-D state arrived in
1.48 seconds. The verified shadow produced one valid decision with zero command
publications. The one bounded GUI run then produced one valid decision and 25
action steps (125 topic publications), with no invalid action.

Recovered manifests are:

```text
/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260813_173553/live_runner_manifest.json
/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260813_174206/live_runner_manifest.json
```

Do not loosen the camera freshness/skew thresholds. Startup now reports an
explicit inventory and retries its DDS participant once before model load.

## Terminal 2C: organizer dataset replay

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

Run the official evaluator only after trajectory Gate B passes and the live
object reset is known to match the selected episode:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh evaluate
```

Episode 9 has `task2_extras/episode_000009.npz` with frame-wise poses for the
thermal pad, target, base, and three boards. The strict tracking gate did not
pass, so no runtime object-pose injection or evaluator call was made in this
handoff. A missed grasp in an unmatched reset is not evidence that the dataset
is wrong.

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
