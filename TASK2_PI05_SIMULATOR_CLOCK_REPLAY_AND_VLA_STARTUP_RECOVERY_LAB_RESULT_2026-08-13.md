# Task 2 PI0.5 simulator-clock replay and VLA startup recovery lab result

Date: 2026-08-13

Branch/base: `submit@2c62fea1c90d0e0096ea859669275079c60e1626`

Scope: startup recovery, one shadow, one bounded GUI model run, simulator-clock
episode-9 replay, and 200-episode initial-state audit. No training was run.

## Outcome

The VLA zero-decision startup failure was reproduced and traced to a runner
contract bug. The final policy container uniquely set
`FASTDDS_BUILTIN_TRANSPORTS=DEFAULT`, while the working image/helper path used
`UDPv4`. In the failing process none of the required head, wrist, EE, odometry,
or joint-state samples was visible. Restoring `UDPv4` and adding an explicit
startup inventory recovered all required inputs in 1.48 seconds.

The real 30k checkpoint passed a zero-publication shadow gate with one valid
decision, then completed the only bounded GUI model attempt with one valid
decision, 25 action steps, 125 command-topic publications, and no invalid
action. No freshness or bounds threshold was weakened.

Episode 9 replay is now released by the recorded dataset timestamp against
`/isaac/clock`; raw actions and ordering are unchanged. The replay was visibly
and operationally correct and completed all 956 frames. It materially improved
tracking, but the handoff's strict Gate B is NO-GO because right-arm mean L2 is
0.2122 rad (target below 0.20) and right-gripper close is 52 frames late
(target within 15). Therefore object reset and evaluator were not attempted.

## VLA startup evidence

| Check | Result |
|---|---|
| Reproduced zero-decision run | 0 decisions, 0 publications, final reason `startup` |
| Failing process inventory | head/left wrist/right wrist absent; both EE absent; odom absent; arm/spine joints absent; 37-D state invalid |
| Root cause | final runner transport override `DEFAULT`; working image/helper transport `UDPv4` |
| Recovered inventory | every required sample present on attempt 1 in 1.477 s |
| Shadow | 1/1 valid decision; 0 command publications; inference 0.568 s; capture-to-ready 0.606 s |
| Bounded GUI | 1/1 valid decision; 25 action steps; 125 command publications; 0 invalid actions |

The runner now prints the exact startup sample inventory before model load,
retries its DDS participant once, writes `startup_inputs_missing` when still
incomplete, and exits without loading the model or publishing commands.

## Exact tested terminals

Environment in every terminal:

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export PI05_CHECKPOINT="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1/training/checkpoints/030000/pretrained_model"
export PI05_RELATIVE_DATASET="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1/relative_dataset"
export PI05_RAW_DATASET="$TASK2_PI05_ROOT/datasets/task2_fixpos_200_46ab41f"
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:hard5-20260813
export ROS_DOMAIN_ID=62
```

Terminal 1:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh sim-up --gui
```

Terminal 2 shadow, then the single bounded GUI attempt:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --checkpoint "$PI05_CHECKPOINT" --dataset-root "$PI05_RELATIVE_DATASET" \
  --shadow --max-actions 25 --max-duration-s 30

bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --checkpoint "$PI05_CHECKPOINT" --dataset-root "$PI05_RELATIVE_DATASET" \
  --max-actions 25 --max-duration-s 60
```

Terminal 2 replay gates:

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --episode 9 --summary-only
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --episode 9 --align-only
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --episode 9 --max-frames 30
bash task2_isaacsim/baselines/pi05/run_pi05.sh replay-dataset \
  --dataset-root "$PI05_RAW_DATASET" --episode 9
```

Terminal 3 read-only monitor and final cleanup:

```bash
ros2 topic echo /isaac/clock --once
ros2 topic echo /isaac/joint_states_full --once
ros2 topic info /isaac/spine_target --verbose

bash task2_isaacsim/baselines/pi05/run_pi05.sh down
```

## Dataset timestamp and simulator-time finding

Episode 9 contains 956 frames from timestamp 0.0 through 31.833334 seconds;
the median recorded step is 0.0333333 seconds. The previous replay advanced at
30 wall-clock FPS, but GUI physics ran near RTF 0.21. It therefore released
targets about 4.7 times faster than simulation advanced. The fixed run waited
for `/isaac/clock` to reach each raw row's recorded timestamp. The full run
advanced 31.833335 simulated seconds in 150.474 wall seconds (RTF 0.2116).

## Episode 9 tracking

| Measurement | Previous wall-clock replay | Simulator-clock replay | Gate |
|---|---:|---:|---|
| Published raw frames | 956/956 | 956/956 | PASS |
| Base publications | 0 | 0 | PASS |
| Interrupted / second publisher | no / no | no / no | PASS |
| Left arm mean L2 | 0.7006 rad | 0.0496 rad | PASS (<0.20) |
| Right arm mean L2 | 0.7300 rad | 0.2122 rad | NO-GO (<0.20) |
| Spine mean absolute error | 0.1418 m | 0.00597 m | PASS (<0.05) |
| Recorded/live right close | 474 / 617 | 474 / 526 | NO-GO (offset 52) |

The frame-0 base mismatch was only 0.00958 m and 0.02197 rad, within the
alignment gate, so it is not the leading explanation for the remaining error.
The first documented actuator-rate violation occurs earlier: raw right joint 3
requires 2.899 rad/s at frame 235 versus the FR3 limit of 2.62 rad/s. Six raw
right-arm steps exceed documented limits; the maximum is 3.768 rad/s. The
right gripper additionally receives a one-frame open-to-closed target step at
frame 464; recorded state crosses 0.5 at frame 474, but the live driver crosses
at frame 526. Root-cause ranking is therefore:

1. organizer-recording versus current right-arm drive/rate behavior;
2. current right-gripper drive/contact dynamics;
3. small initial base-pose difference.

No action smoothing, resampling, model tuning, or controller tuning was used.

## 200-episode initial-state audit

No trustworthy per-episode success label exists in the parquet schema, dataset
features, or extras. The data may be grouped by initial state, not by success
rate.

All 200 episodes are in one dominant live-reset cluster under 0.03 rad arm,
0.01 m spine/base position, 0.02 gripper-fraction, and 0.01 rad base-yaw
tolerances. All six frame-0 object poses are identical within float precision.
Representative episode IDs closest to live reset are 9, 18, and 2.

| Field | Median | Range |
|---|---:|---:|
| Base x | 2.10002685 m | 2.09995794--2.10009766 |
| Base y | 3.05290461 m | 3.05054355--3.05413198 |
| Base yaw | -1.57069314 rad | -1.57096064 to -1.57056165 |
| Spine | 0.00001384 m | 0.0--0.00354851 |
| Left gripper | 0.99998313 | 0.99997252--1.0 |
| Right gripper | 0.99998212 | 0.99978620--1.0 |

The full 14-arm-joint median/range table is in `initial_state_audit.md`.

Episode 9 exact frame-0 robot state:

```text
arms = [0.0, -0.7853999734, 0.0, -2.3561999798, 0.0, 1.5707999468,
        0.7853999734, 0.0, -0.7853999734, 0.0, -2.3561999798, 0.0,
        1.5707999468, 0.7853999734]
spine = 0.0 m
grippers = [1.0, 1.0]
base x/y/yaw = [2.1000483036, 3.0505435467, -1.5709606409]
```

Episode 9 exact object frame-0 poses use source order `xyz,wxyz`:

| Object | Pose `x y z qw qx qy qz` |
|---|---|
| board_0 | 1.95000005, 1.95000005, 0.75, 0.70710999, 0, 0, 0.70710355 |
| board_1 | 2.04999995, 1.95000005, 0.75, 0.70710999, 0, 0, 0.70710355 |
| board_2 | 2.25, 1.95000005, 0.75, 0.70710999, 0, 0, 0.70710355 |
| board_target | 2.15000010, 1.95000005, 0.75, 0.70710999, 0, 0, 0.70710355 |
| thermalpad | 1.75, 1.95000005, 0.85000002, 0.70710999, 0, 0, 0.70710355 |
| thermalpad_base | 1.74000001, 1.90999997, 0.76700002, 1, 0, 0, 0 |

## Conditional object reset and evaluator

Object pose injection: not attempted because strict Gate B did not pass.

Evaluator: not called for the same reason.
This avoids presenting a mismatched-scene evaluator result as a dataset failure.

## Verification

- `python3 -m unittest task2_isaacsim.tests.test_pi05_dataset_replay`: PASS.
- Docker live-runner focused tests: 12/12 PASS.
- `py_compile` for changed Python paths: PASS.
- `bash -n task2_isaacsim/baselines/pi05/run_pi05.sh`: PASS.
- `git diff --check`: PASS.

## Evidence paths

```text
/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260813_172028
/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260813_173553
/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260813_174206
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_174703
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_174723
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_174854
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_175044
/scratch1/2026_ebim/allen_task2_pi05/outputs/dataset_replay_20260813_175945
/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_sim_clock_startup_20260813
```

## Recommended final submission action

Keep the verified DDS startup fix, explicit startup diagnostics, and raw
simulator-time replay. Do not enable the conditional evaluator path from this
handoff. The next model-free experiment should reproduce the organizer
recording controller/drive gains and gripper dynamics, then rerun the same raw
episode-9 sim-time gate. Do not smooth the demonstration to force a pass.
