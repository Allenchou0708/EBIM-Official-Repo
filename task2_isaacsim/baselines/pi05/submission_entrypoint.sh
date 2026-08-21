#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u

case "${1:-}" in
  ground-truth-controller)
    shift
    exec /opt/lerobot/.venv/bin/python \
      /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/ground_truth_joint_lift.py "$@"
    ;;
  stage-base)
    shift
    exec /opt/lerobot/.venv/bin/python \
      /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/fixed_stage_base.py "$@"
    ;;
  stage-spine)
    shift
    exec /opt/lerobot/.venv/bin/python \
      /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/fixed_stage_spine.py "$@"
    ;;
  run-task)
    shift
    exec /opt/lerobot/.venv/bin/python \
      -m task2_isaacsim.baselines.pi05.live.runner "$@"
    ;;
  health)
    exec /opt/lerobot/.venv/bin/python -c '
import math
from task2_isaacsim.baselines.pi05.live.ground_truth_joint_lift import (
    REFERENCE_BASE_XYYAW,
    REFERENCE_PAD_XYYAW,
    anchored_base_pose,
)
pose = (
    REFERENCE_PAD_XYYAW[0], REFERENCE_PAD_XYYAW[1], 0.85,
    math.cos(REFERENCE_PAD_XYYAW[2] / 2.0), 0.0, 0.0,
    math.sin(REFERENCE_PAD_XYYAW[2] / 2.0),
)
actual = anchored_base_pose(pose, REFERENCE_PAD_XYYAW)
assert max(abs(a - b) for a, b in zip(actual, REFERENCE_BASE_XYYAW)) < 1e-9
print("task2-phase1-gt health: PASS")
'
    ;;
  unit-tests)
    exec /opt/lerobot/.venv/bin/python -m unittest \
      task2_isaacsim.tests.test_ground_truth_joint_lift \
      task2_isaacsim.tests.test_ground_truth_pregrasp
    ;;
  -h|--help|help|"")
    cat <<'EOF'
Usage:
  submission_entrypoint.sh ground-truth-controller CONTROLLER_ARGUMENTS...
  submission_entrypoint.sh stage-base STAGER_ARGUMENTS...
  submission_entrypoint.sh stage-spine STAGER_ARGUMENTS...
  submission_entrypoint.sh unit-tests
  submission_entrypoint.sh run-task RUNNER_ARGUMENTS...
  submission_entrypoint.sh health

The Phase I submission is the disclosed ground-truth policy. Use the host-side
live/run_ground_truth_random_gui.sh launcher to reset the official Isaac Sim
scene and orchestrate stage-base, stage-spine, and ground-truth-controller.
The legacy run-task command is retained only for the non-submitted PI0.5
learned-policy experiments.
EOF
    ;;
  *)
    echo "Unknown submission command: $1" >&2
    exit 2
    ;;
esac
