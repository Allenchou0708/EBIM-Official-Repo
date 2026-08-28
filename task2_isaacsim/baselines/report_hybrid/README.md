# Task 2 report_hybrid simulator policy

This baseline implements the Phase II design documented in
[`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md): an odometry-relative S0, robot
RGB-D perception, camera-FK projection, demonstration-landmark retargeting
through the benchmark's Lula/RMPFlow target interface, and a forward-only
gripper-latched state machine. It does not use PI0.5 for phase, gripper, base,
spine, or left-arm ownership.

Runtime policy inputs are limited to robot cameras, camera calibration/FK,
odometry, joint state, and end-effector state. `external_audit.py` is a
separate simulator diagnostic and must never run in, import into, or feed the
policy process.

## Focused test

```bash
python3 -m unittest task2_isaacsim.tests.test_report_hybrid
```

## Simulator sequence

Start the official Task 2 room with robot RGB-D, recording state, and the
existing pose-command RMPFlow bridge enabled. Do not run browser, keyboard-arm,
GELLO, or another base publisher at the same time.

```bash
task2_isaacsim/scripts/run_isaacsim_teleop.sh \
  --scene room --headless --no-browser --no-republisher -- \
  --record --robot-camera-depth --arm-pose-command-control
```

Then run the three gates in a ROS 2 environment joined to the host graph:

```bash
python3 -m task2_isaacsim.baselines.report_hybrid.policy_node \
  --mode stage --output /evidence/gate_c_stage.json

python3 -m task2_isaacsim.baselines.report_hybrid.policy_node \
  --mode shadow --output /evidence/gate_c_shadow.json

python3 -m task2_isaacsim.baselines.report_hybrid.external_audit \
  --shadow /evidence/gate_c_shadow.json \
  --output /evidence/gate_c_external_audit.json
```

Only if shadow and external audit pass may a single live attempt run:

```bash
python3 -m task2_isaacsim.baselines.report_hybrid.policy_node \
  --mode execute --output /evidence/live_attempt.json
```

The external official evaluation service may be invoked after the policy has
released and retreated. Its output is scoring evidence, never policy input.
