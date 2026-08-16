#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash

case "${1:-help}" in
  run-task)
    shift
    exec /opt/lerobot/.venv/bin/python \
      -m task2_isaacsim.baselines.pi05.live.runner "$@"
    ;;
  health)
    exec /opt/lerobot/.venv/bin/python -c '
from pathlib import Path
from task2_isaacsim.baselines.pi05.contract import ACTION_SIZE, PI05_CONTRACT
from task2_isaacsim.baselines.pi05.live.core import safe_action
assert PI05_CONTRACT.max_action_dim == 32
raw, effective = safe_action([0.0] * ACTION_SIZE)
assert len(raw) == len(effective) == ACTION_SIZE
assert effective[:3] == (0.0, 0.0, 0.0)
checkpoint = Path("/data/checkpoint")
if checkpoint.exists():
    assert (checkpoint / "model.safetensors").is_file()
print("EBiM Task 2 PI0.5 health: PASS")
'
    ;;
  help|-h|--help)
    echo "Usage:"
    echo "  docker run ... IMAGE health"
    echo "  docker run ... IMAGE run-task [runner arguments]"
    ;;
  *)
    echo "Unknown command: $1" >&2
    exit 2
    ;;
esac
