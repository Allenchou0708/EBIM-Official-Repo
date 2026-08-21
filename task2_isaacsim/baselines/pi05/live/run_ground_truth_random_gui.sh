#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMAGE="${PI05_LIVE_IMAGE:-ebim-task2-pi05-live:queue-replace-20260812}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
EVIDENCE_DIR="${TASK2_EVIDENCE_DIR:-}"

usage() {
  cat <<'EOF'
Usage:
  run_ground_truth_random_gui.sh launch-random
  run_ground_truth_random_gui.sh launch-nominal
  run_ground_truth_random_gui.sh trials [COUNT]
  run_ground_truth_random_gui.sh nominal [COUNT]

Run one launch command in the GUI terminal, then its matching trial command in
a second terminal.  Each attempt verifies whether its reset was randomized.
CONTACT_SWEEP_X_M/Y_M can override the calibrated pre-contact compensation.
EOF
}

launch_gui() {
  local randomize="${1}"
  cd "${REPO_ROOT}"
  export ROS_DOMAIN_ID
  local args=(
    task2_isaacsim/scripts/run_isaacsim_teleop.sh
    --scene room --controller-mode none --no-browser --no-republisher -- \
    --arm-pose-command-control \
    --configure-gripper-drives \
    --arm-teleop-gripper-closed 0.804 \
    --publish-recording-topics \
    --publish-ground-truth \
    --scene-reset-hotkey
  )
  [[ "${randomize}" == true ]] && args+=(--randomize-objects)
  exec "${args[@]}"
}

run_ros() {
  local docker_args=(docker run --rm --network host --ipc=host \
    --user "$(id -u):$(id -g)" \
    --entrypoint /bin/bash \
    -e HOME=/tmp/ebim-live-home \
    -e USER="${USER:-robot}" \
    -e LOGNAME="${USER:-robot}" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -e PYTHONPATH=/workspace/EBiM_Challenge:/opt/ros/jazzy/lib/python3.12/site-packages \
    -v "${REPO_ROOT}:/workspace/EBiM_Challenge:ro")
  if [[ -n "${EVIDENCE_DIR}" ]]; then
    mkdir -p "${EVIDENCE_DIR}"
    docker_args+=(-v "${EVIDENCE_DIR}:/evidence")
  fi
  "${docker_args[@]}" "${IMAGE}" -lc "$1"
}

request_scene_reset() {
  local expected_randomized="${1}"
  run_ros '
    source /opt/ros/jazzy/setup.bash
    ready=false
    for _ in $(seq 1 10); do
      if ros2 topic info /isaac/task2/scene_reset_request >/dev/null 2>&1 \
        && ros2 topic info /isaac/task2/scene_reset >/dev/null 2>&1; then
        ready=true
        break
      fi
      sleep 0.5
    done
    if [[ "${ready}" != true ]]; then
      echo "Task 2 reset topics are unavailable; restart the GUI simulator" >&2
      exit 3
    fi
    reset_event=/tmp/task2_reset_event.txt
    ros2 topic echo --no-daemon --spin-time 2 --once --timeout 25 \
      --qos-reliability reliable /isaac/task2/scene_reset \
      std_msgs/msg/String > "${reset_event}" &
    echo_pid=$!
    sleep 2
    ros2 topic pub --once /isaac/task2/scene_reset_request \
      std_msgs/msg/String "{data: reset}" >/dev/null
    wait "${echo_pid}"
    cat "${reset_event}"
    grep -q "randomized.*'"${expected_randomized}"'" "${reset_event}"
  '
}

stage_spine() {
  run_ros '
    source /opt/ros/jazzy/setup.bash
    exec /opt/lerobot/.venv/bin/python \
      /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/fixed_stage_spine.py
  '
}

run_controller() {
  local placement_contract="${1}"
  local default_sweep_x_m=0.050 default_sweep_y_m=0.010
  local alignment_release_xy_m=0.025 success_xy_m=0.030
  if [[ "${placement_contract}" == nominal ]]; then
    default_sweep_x_m=0.003
    default_sweep_y_m=-0.012
    alignment_release_xy_m=0.055
    success_xy_m=0.055
  fi
  local contact_sweep_x_m="${CONTACT_SWEEP_X_M:-${default_sweep_x_m}}"
  local contact_sweep_y_m="${CONTACT_SWEEP_Y_M:-${default_sweep_y_m}}"
  local controller_output=/tmp/task2_ground_truth_gui_result.json
  [[ -n "${EVIDENCE_DIR}" ]] && controller_output=/evidence/controller_result.json
  run_ros '
    source /opt/ros/jazzy/setup.bash
    exec /opt/lerobot/.venv/bin/python \
      /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/ground_truth_joint_lift.py \
      --absolute-dataset-joints \
      --preposition-at-live-grasp-base \
      --grasp-yaw-limit-deg 5 \
      --grasp-depth-bias-m 0.005 \
      --grasp-dwell-s 0.5 \
      --trajectory-rate-hz 30 \
      --cartesian-handoff-frame 560 \
      --cartesian-handoff-settle-s 0.05 \
      --cartesian-speed-mps 0.30 \
      --cartesian-max-xy-displacement-m 0.16 \
      --cartesian-descent-start-xy-m 0.040 \
      --cartesian-descent-speed-mps 0.080 \
      --precontact-lift-m 0.105 \
      --alignment-release-xy-m '"${alignment_release_xy_m}"' \
      --alignment-release-height-m 0.006 \
      --alignment-release-height-tolerance-m 0.008 \
      --placement-contract '"${placement_contract}"' \
      --contact-sweep-x-m '"${contact_sweep_x_m}"' \
      --contact-sweep-y-m '"${contact_sweep_y_m}"' \
      --flat-pad-z-span-m 0.020 \
      --alignment-stable-s 0.50 \
      --post-release-settle-s 0.80 \
      --post-release-retract-m 0.080 \
      --post-release-retract-speed-mps 0.04 \
      --post-release-retract-tolerance-m 0.010 \
      --success-xy-m '"${success_xy_m}"' \
      --success-z-m 0.012 \
      --max-duration-s 90 \
      --output '"${controller_output}"'
  '
}

print_summary() {
  python3 -c '
import json
import sys

result = json.loads(sys.stdin.read().splitlines()[-1])
fields = (
    "success",
    "reason",
    "maximum_pad_height_above_target_m",
    "release_pad_height_above_target_m",
    "release_pad_target_xy_error_m",
    "final_pad_target_xy_error_m",
    "final_pad_target_z_error_m",
    "final_pad_z_span_m",
    "placement_contract",
    "retract_completed_sim",
    "elapsed_wall_s",
)
print(json.dumps({key: result.get(key) for key in fields}, sort_keys=True))
' <<<"$1"
}

run_trials() {
  local count="${1:-3}"
  local placement_contract="${2}"
  local expected_randomized="${3}"
  [[ "${count}" =~ ^[1-9][0-9]*$ ]] || {
    echo "COUNT must be a positive integer" >&2
    exit 2
  }
  cd "${REPO_ROOT}"
  local successes=0 attempts=0
  while (( attempts < count )); do
    attempts=$((attempts + 1))
    echo "=== ${placement_contract} attempt ${attempts} (${successes}/${count} passed): reset ==="
    request_scene_reset "${expected_randomized}"
    echo "=== ${placement_contract} attempt ${attempts}: spine ==="
    stage_spine
    echo "=== ${placement_contract} attempt ${attempts}: place ==="
    local result controller_rc
    set +e
    result="$(run_controller "${placement_contract}" 2>&1)"
    controller_rc=$?
    set -e
    if [[ -z "${result}" ]]; then
      echo "Controller exited ${controller_rc} without output" >&2
      exit "${controller_rc}"
    fi
    printf '%s\n' "${result}" | tail -n 20 >&2
    print_summary "${result}"
    if (( controller_rc == 0 )); then
      successes=$((successes + 1))
      echo "=== ${placement_contract} attempt ${attempts}: PASS (${successes}/${count}) ==="
    elif (( controller_rc == 2 )); then
      echo "=== ${placement_contract} attempt ${attempts}: FAIL ===" >&2
    else
      echo "Controller exited unexpectedly with ${controller_rc}" >&2
      exit "${controller_rc}"
    fi
  done
  if (( successes != count )); then
    echo "${successes}/${count} ${placement_contract} contact-placement trials passed." >&2
    return 2
  fi
  echo "All ${count} ${placement_contract} contact-placement trials passed."
}

case "${1:-}" in
  launch-random)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    launch_gui true
    ;;
  launch-nominal)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    launch_gui false
    ;;
  trials)
    [[ $# -le 2 ]] || { usage >&2; exit 2; }
    run_trials "${2:-3}" randomized-flat true
    ;;
  nominal)
    [[ $# -le 2 ]] || { usage >&2; exit 2; }
    run_trials "${2:-1}" nominal false
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
