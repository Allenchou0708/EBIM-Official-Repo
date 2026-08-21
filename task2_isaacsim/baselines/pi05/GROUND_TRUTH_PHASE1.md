# Task 2 Phase I ground-truth controller

## Why this path is being used

The EBiM Competition organizing committee's announcement forwarded on
2026-08-19 explicitly relaxed the Phase I simulation rule.  For Phase I
only, a submission may use object positions and states published directly
by the simulator.  The announcement explains that Phase I is intended to
familiarize teams with the tasks and prepare them for real-world evaluation,
not to require every team to solve the complete perception problem first.

This repository is therefore testing a deterministic ground-truth controller
as a legal Phase I recovery path.  The earlier PI0.5 policy pipeline remains
useful technical work, but its live safety gate stopped on the first shadow
decision because the raw gripper action was outside its contract and
inference was slower than the five-step action horizon.  Ground truth removes
the perception/model-output variables while the simulator manipulation,
controller, safety gates, packaging, and evidence path are debugged.

This is not a claim that ground truth is valid for Phase II.  The organizer
announcement says that real-world evaluation must use sensing available on
the physical platform.  A submission must also declare whether it uses ground
truth or its own perception; a declaration inconsistent with the submitted
code may lead to disqualification.

The same announcement gives the following Phase I weights:

- policy with ground truth submitted during the extension: `0.90`;
- policy with own perception submitted during the extension: `0.95`;
- Technical Report: `0.65`.

The reopened deadline is Aug 22, 2026 AoE, which the announcement states is
Aug 23, 2026 11:59 UTC / 19:59 Taipei.  Policy evaluation uses three runs and
takes their mean.  These facts make a disclosed, tested ground-truth policy a
reasonable Phase I target, while retaining the perception work for Phase II.

Source material is the organizer email titled “The EBiM Competition — Phase I
reopened until Aug 22 (AoE), new submission options, Phase II dates.”  The
forwarded `.eml` is reference material and is not copied into Git.

## Controller contract

The current implementation uses:

- simulator ground-truth `thermalpad` and `board_target` poses plus pad points;
- dataset episode 19 link8 landmarks with live translation/yaw anchoring;
- stamped world-frame end-effector and gripper commands;
- simulator-clock freshness and skew checks;
- publisher-contention, base, spine, and gripper gates;
- measured link8 feedback with bounded position/orientation correction;
- physical pad-height and target-distance gates before phase advancement;
- JSON manifests and camera captures as run evidence.

The controller must not report success merely because a target command was
published or an arm reached its pose.  Lift, transport, place, release, and
retract advance only when the measured pad state satisfies their physical
conditions.

## Live status on 2026-08-21

Current result: **GO for a disclosed Phase I ground-truth submission**.

The physical simulator gate now completes grasp, lift, transport, Cartesian
alignment, release, and post-release stability.  In addition to a nominal
success, three independent runs were started from fresh simulator processes,
received one ROS scene-reset request each, and reported `randomized 6 objects`
before execution.  All three finished with `stable_place_and_release`:

| evidence | max lift above target | final XY error | final Z error |
| --- | ---: | ---: | ---: |
| `run55_random_rmp530_damped/task.json` | 17.21 cm | 5.33 cm | 0.52 mm |
| `run56_random_success2/retry_yaw_clamp_task.json` | 17.80 cm | 4.66 cm | 3.02 mm |
| `run58_random_success3_tight/task.json` | 16.83 cm | 4.34 cm | 1.18 mm |

The corresponding SHA-256 values, in table order, are:

- `2b0e0b48f1617b5c3cc03f9c1849418932acf7be24a47efeb92a9e61642228ec`;
- `43f9e2a0b3d047358c54c6a6f695ab0fd98c4e6cfdf6ce5ac9478c9c60944c95`;
- `c33910fe6beff862be00f0f4170a78a2355f56cc54f984e451bec17f4eaa5cd3`.

The controller uses these conservative physical gates:

- at least 13 cm maximum lift;
- a frame-520 checkpoint requiring the pad to remain at least 12 cm above the
  target and to have moved at least 15 cm in the table plane;
- final pad-to-target XY and Z errors no greater than 6 cm;
- an open gripper and a 0.5 s post-release stability dwell.

The randomized recovery path clamps grasp-base yaw compensation to ±5 degrees
to prevent a long base-offset rotation from colliding with the room, stages
q399 directly at the live GT-aligned base, uses the canonical Robotiq drive
gains, replays the dense episode-19 trajectory at 40 Hz, and hands off at
frame 530 to bounded RMPflow alignment.  Cartesian motion stops integrating as
soon as release begins, preventing a released pad from driving the target out
of the workspace.

Primary evidence root:

```text
/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_ground_truth_controller_20260820
```

Nominal reference evidence is
`run49_release_gate_rmp550/task.json` (`stable_place_and_release`, 3.91 cm XY,
3.04 mm Z).  Failed exploratory runs remain under the same evidence root and
must not be mixed with the three successful randomized manifests above.

## Reproduction

Launch Isaac Sim with ground truth, pose control, reset support, and recording:

```bash
task2_isaacsim/scripts/run_isaacsim_teleop.sh \
  --scene room --controller-mode none --no-browser --no-republisher -- \
  --arm-pose-command-control --publish-recording-topics \
  --publish-ground-truth --scene-reset-hotkey --randomize-objects
```

For an evaluation sample, publish exactly one reset request, wait for
`Scene reset #1 done (randomized 6 objects)`, stage the spine with
`fixed_stage_spine.py`, then run:

```bash
python3 task2_isaacsim/baselines/pi05/live/ground_truth_joint_lift.py \
  --output /data/evidence/task.json
```

The successful settings are now defaults.  The ROS process must use domain 0,
Fast DDS, UDPv4 transport, the repository on `PYTHONPATH`, and the same host
network as Isaac Sim.  Evidence paths are intentionally outside Git.
