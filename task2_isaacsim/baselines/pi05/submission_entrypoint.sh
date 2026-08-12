#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u

case "${1:-}" in
  run-task)
    shift
    exec /opt/lerobot/.venv/bin/python \
      -m task2_isaacsim.baselines.pi05.live.runner "$@"
    ;;
  health)
    exec /opt/lerobot/.venv/bin/python -c '
from task2_isaacsim.baselines.pi05.contract import ACTION_SIZE, PI05_CONTRACT
from task2_isaacsim.baselines.pi05.live.core import safe_action
assert PI05_CONTRACT.max_action_dim == 32
raw, effective = safe_action([0.0] * ACTION_SIZE)
assert len(raw) == len(effective) == ACTION_SIZE
assert effective[:3] == (0.0, 0.0, 0.0)
print("task2-pi05-submit health: PASS")
'
    ;;
  -h|--help|help|"")
    cat <<'EOF'
Usage:
  submission_entrypoint.sh run-task RUNNER_ARGUMENTS...
  submission_entrypoint.sh health

The formal evaluation entry point is run-task. The host run_pi05.sh launcher
performs scene reset, fixed-base staging, initial-spine staging, and camera
preflight before invoking it.
EOF
    ;;
  *)
    echo "Unknown submission command: $1" >&2
    exit 2
    ;;
esac
