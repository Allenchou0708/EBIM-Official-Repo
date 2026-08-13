#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/.env.pi05" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env.pi05"
  set +a
fi

TASK2_PI05_ROOT="${TASK2_PI05_ROOT:-/scratch1/2026_ebim/allen_task2_pi05}"
PI05_TRAIN_IMAGE="${PI05_TRAIN_IMAGE:-ebim-task2-pi05:200-submit-20260812}"
PI05_LIVE_IMAGE="${PI05_LIVE_IMAGE:-ebim-task2-pi05-submit:local}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-}"
PI05_RELATIVE_DATASET="${PI05_RELATIVE_DATASET:-}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_DOMAIN_ID
ISAACSIM_CONTAINER="${ISAACSIM_CONTAINER:-isaac-sim-5-1-0-workshop}"
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/task2_fixpos_200_expert.yaml"

usage() {
  cat <<'EOF'
Usage:
  ./run_pi05.sh doctor
  ./run_pi05.sh dataset [--config PATH]
  ./run_pi05.sh train [--config PATH] [--run NAME]
  ./run_pi05.sh sim-up [--gui]
  ./run_pi05.sh run-task [--checkpoint PATH] [--dataset-root PATH] [--max-actions N] [--max-duration-s S]
  ./run_pi05.sh replay-dataset [--dataset-root PATH] [--episode auto|N] [--summary-only|--align-only|--max-frames N]
  ./run_pi05.sh evaluate
  ./run_pi05.sh down
EOF
}

resolve_config() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    realpath "${path}"
  else
    realpath "${SCRIPT_DIR}/${path}"
  fi
}

config_value() {
  local config="$1"
  local key="$2"
  docker run --rm --entrypoint python \
    -v "${config}:/config.yaml:ro" \
    "${PI05_TRAIN_IMAGE}" -c '
import sys, yaml
value = yaml.safe_load(open("/config.yaml"))
for part in sys.argv[1].split("."):
    value = value[part]
print(value)
' "${key}"
}

parse_config_run() {
  CONFIG="${DEFAULT_CONFIG}"
  RUN_NAME=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config) CONFIG="$(resolve_config "$2")"; shift 2 ;;
      --run) RUN_NAME="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  CONFIG="$(resolve_config "${CONFIG}")"
}

training_base_args() {
  TRAINING_ARGS=(
    --rm --gpus all --ipc=host
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp/ebim-home
    -e HF_HOME=/cache/huggingface
    -e EBIM_PI05_IMAGE="${PI05_TRAIN_IMAGE}"
    -v "${TASK2_PI05_ROOT}/cache:/cache"
  )
}

command_doctor() {
  training_base_args
  docker run "${TRAINING_ARGS[@]}" "${PI05_TRAIN_IMAGE}" \
    doctor --profile expert
}

command_dataset() {
  parse_config_run "$@"
  local local_dir dataset_root audit
  local_dir="$(config_value "${CONFIG}" dataset.local_dir)"
  dataset_root="${TASK2_PI05_ROOT}/datasets/${local_dir}"
  audit="${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/task2_fixpos_200_audit.json"
  mkdir -p "${TASK2_PI05_ROOT}/datasets" "$(dirname "${audit}")"
  training_base_args
  if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
    docker run "${TRAINING_ARGS[@]}" \
      -v "${CONFIG}:/data/config.yaml:ro" \
      -v "${TASK2_PI05_ROOT}/datasets:/data/datasets" \
      "${PI05_TRAIN_IMAGE}" download-organizer \
      --config /data/config.yaml --destination "/data/datasets/${local_dir}"
  fi
  docker run "${TRAINING_ARGS[@]}" \
    -v "${CONFIG}:/data/config.yaml:ro" \
    -v "${dataset_root}:/data/dataset:ro" \
    -v "$(dirname "${audit}"):/data/evidence" \
    "${PI05_TRAIN_IMAGE}" audit-dataset \
    --config /data/config.yaml --dataset-root /data/dataset \
    --output /data/evidence/$(basename "${audit}")
}

command_train() {
  parse_config_run "$@"
  local local_dir dataset_root audit output episodes
  local_dir="$(config_value "${CONFIG}" dataset.local_dir)"
  RUN_NAME="${RUN_NAME:-$(config_value "${CONFIG}" training.run)}"
  dataset_root="${TASK2_PI05_ROOT}/datasets/${local_dir}"
  audit="${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/task2_fixpos_200_audit.json"
  output="${TASK2_PI05_ROOT}/outputs/${RUN_NAME}"
  episodes="$(python3 -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["split"]["train"])))' "${audit}")"
  training_base_args
  docker run "${TRAINING_ARGS[@]}" \
    -v "${dataset_root}:/data/dataset:ro" \
    -v "${TASK2_PI05_ROOT}/outputs:/data/outputs" \
    -v "$(dirname "${audit}"):/data/evidence:ro" \
    "${PI05_TRAIN_IMAGE}" train --profile expert \
    --dataset-root /data/dataset \
    --audit-report /data/evidence/$(basename "${audit}") \
    --output-dir "/data/outputs/${RUN_NAME}" \
    --episodes "${episodes}" --execute
}

live_shell() {
  docker run --rm --network host --entrypoint bash \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -v "${TASK2_PI05_ROOT}/evidence:/data/evidence" \
    "${PI05_LIVE_IMAGE}" -lc "source /opt/ros/jazzy/setup.bash && $1"
}

command_sim_up() {
  [[ "${1:-}" = "--gui" ]] || { echo "sim-up requires --gui" >&2; exit 2; }
  exec "${REPO_ROOT}/task2_isaacsim/scripts/run_isaacsim_teleop.sh" \
    --scene room --controller-mode none --no-browser --no-republisher -- \
    --disable-browser-command-topics --record
}

command_run_task() {
  local checkpoint="${PI05_CHECKPOINT}" dataset="${PI05_RELATIVE_DATASET}"
  local max_actions=600 max_duration_s=300 max_decisions
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint) checkpoint="$(realpath "$2")"; shift 2 ;;
      --dataset-root) dataset="$(realpath "$2")"; shift 2 ;;
      --max-actions) max_actions="$2"; shift 2 ;;
      --max-duration-s) max_duration_s="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ -n "${checkpoint}" ]] || { echo "--checkpoint is required" >&2; exit 2; }
  checkpoint="$(realpath "${checkpoint}")"
  [[ -d "${checkpoint}" ]] || { echo "Checkpoint directory not found" >&2; exit 2; }
  [[ "${max_actions}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--max-actions must be a positive integer" >&2
    exit 2
  }
  [[ "${max_duration_s}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "--max-duration-s must be a non-negative number" >&2
    exit 2
  }
  max_decisions=$(( (max_actions + 23) / 24 + 2 ))
  if (( max_decisions < 40 )); then
    max_decisions=40
  fi
  local run_root output
  if [[ -z "${dataset}" ]]; then
    run_root="$(realpath "${checkpoint}/../../../..")"
    dataset="${run_root}/relative_dataset"
  fi
  dataset="$(realpath "${dataset}")"
  [[ -d "${dataset}" ]] || { echo "Relative dataset directory not found" >&2; exit 2; }
  output="${TASK2_PI05_ROOT}/outputs/live_submit_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${output}" "${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/launcher"
  live_shell "ros2 topic pub --once /isaac/task2/scene_reset_request std_msgs/msg/String '{data: reset}'"
  live_shell "python3 /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/fixed_stage_base.py --target 2.10 3.05 -1.571 --position-tolerance-m 0.015 --yaw-tolerance-rad 0.04 --output /data/evidence/task2_200_submit_20260812/launcher/fixed_base.json"
  live_shell "python3 /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/fixed_stage_spine.py --target-m 0.0 --measured-target-m 0.0 --output /data/evidence/task2_200_submit_20260812/launcher/initial_spine.json"
  live_shell "python3 /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/eval_camera_preflight.py --output /data/evidence/task2_200_submit_20260812/launcher/eval_preflight.json"
  exec docker run --rm --gpus all --network host --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/ebim-live-home -e USER=ebim -e LOGNAME=ebim \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -e FASTDDS_BUILTIN_TRANSPORTS=DEFAULT \
    -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "${TASK2_PI05_ROOT}/cache:/cache" \
    -v "${checkpoint}:/data/checkpoint:ro" \
    -v "${dataset}:/data/dataset:ro" -v "${output}:/data/output" \
    "${PI05_LIVE_IMAGE}" run-task --checkpoint /data/checkpoint \
    --dataset-root /data/dataset --dataset-repo-id hermanprawiro/task2_fixpos_200 \
    --output-dir /data/output --base-target 2.10 3.05 -1.571 \
    --base-coordinate-frame dataset_odom_world_verified_against_room_scene \
    --confirm-fixed-base-staging --arm-simulator \
    --max-decisions "${max_decisions}" \
    --max-publish-actions "${max_actions}" \
    --max-duration-s "${max_duration_s}"
}

command_replay_dataset() {
  local dataset="${TASK2_PI05_ROOT}/datasets/task2_fixpos_200_46ab41f"
  local audit="${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/task2_fixpos_200_audit.json"
  local episode="auto" mode="" max_frames="" output
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset-root) dataset="$(realpath "$2")"; shift 2 ;;
      --episode) episode="$2"; shift 2 ;;
      --summary-only) mode="--summary-only"; shift ;;
      --align-only) mode="--align-only"; shift ;;
      --max-frames) max_frames="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ -d "${dataset}" ]] || { echo "Raw dataset directory not found" >&2; exit 2; }
  [[ -f "${audit}" ]] || { echo "Train/held-out audit report not found" >&2; exit 2; }
  [[ "${episode}" = "auto" || "${episode}" =~ ^[0-9]+$ ]] || {
    echo "--episode must be auto or a non-negative integer" >&2
    exit 2
  }
  [[ -z "${max_frames}" || "${max_frames}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--max-frames must be a positive integer" >&2
    exit 2
  }
  output="${TASK2_PI05_ROOT}/outputs/dataset_replay_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${output}"
  local module_command
  module_command="source /opt/ros/jazzy/setup.bash && exec /opt/lerobot/.venv/bin/python -m task2_isaacsim.baselines.pi05.dataset_replay --dataset-root /data/dataset --episode ${episode} --audit-report /data/audit.json --output-dir /data/output"
  [[ -n "${mode}" ]] && module_command+=" ${mode}"
  [[ -n "${max_frames}" ]] && module_command+=" --max-frames ${max_frames}"
  if [[ "${mode}" != "--summary-only" ]]; then
    local selected target
    docker run --rm --entrypoint bash \
      -e PYTHONPATH=/workspace/EBiM_Challenge \
      -v "${REPO_ROOT}:/workspace/EBiM_Challenge:ro" \
      -v "${dataset}:/data/dataset:ro" -v "${audit}:/data/audit.json:ro" \
      -v "${output}:/data/output" "${PI05_LIVE_IMAGE}" -lc \
      "source /opt/ros/jazzy/setup.bash && /opt/lerobot/.venv/bin/python -m task2_isaacsim.baselines.pi05.dataset_replay --dataset-root /data/dataset --episode ${episode} --audit-report /data/audit.json --output-dir /data/output --summary-only"
    selected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["episode"])' "${output}/trajectory_summary.json")"
    target="$(python3 -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["frame0_base_pose"])))' "${output}/trajectory_summary.json")"
    episode="${selected}"
    module_command="source /opt/ros/jazzy/setup.bash && exec /opt/lerobot/.venv/bin/python -m task2_isaacsim.baselines.pi05.dataset_replay --dataset-root /data/dataset --episode ${episode} --audit-report /data/audit.json --output-dir /data/output"
    [[ -n "${mode}" ]] && module_command+=" ${mode}"
    [[ -n "${max_frames}" ]] && module_command+=" --max-frames ${max_frames}"
    live_shell "ros2 topic pub --once /isaac/task2/scene_reset_request std_msgs/msg/String '{data: reset}'"
    # shellcheck disable=SC2086
    live_shell "python3 /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/fixed_stage_base.py --target ${target} --position-tolerance-m 0.015 --yaw-tolerance-rad 0.04 --output /data/evidence/task2_200_submit_20260812/launcher/replay_base.json"
  fi
  exec docker run --rm --network host --ipc=host --entrypoint bash \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" -e PYTHONPATH=/workspace/EBiM_Challenge \
    -v "${REPO_ROOT}:/workspace/EBiM_Challenge:ro" \
    -v "${dataset}:/data/dataset:ro" -v "${audit}:/data/audit.json:ro" \
    -v "${output}:/data/output" "${PI05_LIVE_IMAGE}" -lc "${module_command}"
}

command_evaluate() {
  "${REPO_ROOT}/scripts/evaluation/task2/run.sh" up
  "${REPO_ROOT}/scripts/evaluation/task2/run.sh" evaluate
}

command_down() {
  "${REPO_ROOT}/scripts/evaluation/task2/run.sh" down || true
  docker compose -f "${REPO_ROOT}/task2_isaacsim/docker-compose.yml" down
  docker exec "${ISAACSIM_CONTAINER}" pkill -f task2_isaacsim/scripts/scene_room.py || true
}

case "${1:-}" in
  doctor) shift; command_doctor "$@" ;;
  dataset) shift; command_dataset "$@" ;;
  train) shift; command_train "$@" ;;
  sim-up) shift; command_sim_up "$@" ;;
  run-task) shift; command_run_task "$@" ;;
  replay-dataset) shift; command_replay_dataset "$@" ;;
  evaluate) shift; command_evaluate "$@" ;;
  down) shift; command_down "$@" ;;
  *) usage; exit 2 ;;
esac
