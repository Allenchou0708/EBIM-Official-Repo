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
PI05_LIVE_IMAGE="${PI05_LIVE_IMAGE:-ebim-task2-pi05-submit:local}"
ISAACSIM_CONTAINER="${ISAACSIM_CONTAINER:-isaac-sim-5-1-0-workshop}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_DOMAIN_ID

DATASET_ROOT="${TASK2_PI05_DATASET:-${TASK2_PI05_ROOT}/datasets/task2_fixpos_200_46ab41f}"
STAGING_AUDIT="${TASK2_PI05_STAGING_AUDIT:-${TASK2_PI05_ROOT}/evidence/task2_pi05_camera_ready_pad_relative_20260815/startup_staging_audit.json}"
LEROBOT_SRC="${TASK2_PI05_LEROBOT_SRC:-${TASK2_PI05_ROOT}/runtime/lerobot_dd256899946c/src}"

OURS_20K="${TASK2_PI05_ROOT}/outputs/task2_sim_manipulation_only_expert_b16_20k_20260902/checkpoints/020000/pretrained_model"
SUBMITTED_30K="${TASK2_PI05_ROOT}/external_checkpoints/official_task2_pi05_20260903/sinica_pi05_submitted_30k"
ROBOT_DREAMS_20K="${TASK2_PI05_ROOT}/external_checkpoints/official_task2_pi05_20260903/robot_dreams_pi05_fullft_20k"

usage() {
  cat <<'EOF'
Task 2 PI0.5 simulator controller

Usage:
  ./run_pi05.sh models
  ./run_pi05.sh sim-up --gui
  ./run_pi05.sh run [OPTIONS]
  ./run_pi05.sh down

Run options:
  --model ours-20k|submitted-30k|robot-dreams-20k
  --seed N                         default: 1001
  --execution-horizon N            default: 15, range: 1..50
  --robot-dreams-native             three cameras, 37-D state, full 20-D
                                    fixed-base output, async horizon 50
  --hybrid-transport                VLA approach/grasp, then RMPflow
                                    lift/transfer/place/release
  --shadow                          infer once; do not publish VLA actions
  --run-label LABEL
  --max-actions N                   default: 600
  --max-duration-s S                default: 300
  --base-stage-max-duration-s S     default: 180
  --spine-stage-max-duration-s S    default: 180
  --manipulation-stage-max-duration-s S  default: 240

All profiles use deterministic base -> spine -> RMPflow observation staging.
Normal execution owns only the right arm and gripper. Robot Dreams native mode
uses its original whole-body contract while fixing base output to zero. Hybrid
transport returns to right-only ownership and hands a latched grasp to RMPflow.
EOF
}

resolve_model() {
  case "$1" in
    ours-20k|20k)
      MODEL_NAME="ours-20k"
      MODEL_PATH="${OURS_20K}"
      MODEL_CONTRACT="8-D right arm + right gripper; expert-only manipulation 20k"
      ;;
    submitted-30k|submitted)
      MODEL_NAME="submitted-30k"
      MODEL_PATH="${SUBMITTED_30K}"
      MODEL_CONTRACT="20-D whole body; submitted 30k; right 8-D published only"
      ;;
    robot-dreams-20k|robot-dreams)
      MODEL_NAME="robot-dreams-20k"
      MODEL_PATH="${ROBOT_DREAMS_20K}"
      MODEL_CONTRACT="20-D whole body; full-FT 20k"
      ;;
    *)
      echo "Unknown model profile: $1" >&2
      echo "Choose ours-20k, submitted-30k, or robot-dreams-20k." >&2
      exit 2
      ;;
  esac
}

command_models() {
  local name
  for name in ours-20k submitted-30k robot-dreams-20k; do
    resolve_model "${name}"
    if [[ -f "${MODEL_PATH}/model.safetensors" ]]; then
      printf '%-20s ready  %s\n  %s\n' "${MODEL_NAME}" "${MODEL_CONTRACT}" "${MODEL_PATH}"
    else
      printf '%-20s MISSING\n  %s\n' "${MODEL_NAME}" "${MODEL_PATH}"
    fi
  done
}

live_shell() {
  docker run --rm --network host --entrypoint bash \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -e PYTHONPATH=/workspace/EBiM_Challenge \
    -v "${REPO_ROOT}:/workspace/EBiM_Challenge:ro" \
    -v "${TASK2_PI05_ROOT}/evidence:/data/evidence" \
    "${PI05_LIVE_IMAGE}" -lc "source /opt/ros/jazzy/setup.bash && $1"
}

command_sim_up() {
  [[ "${1:-}" = "--gui" && $# -eq 1 ]] || {
    echo "sim-up requires exactly --gui" >&2
    exit 2
  }
  local log_dir log_path status
  log_dir="${TASK2_PI05_ROOT}/evidence/pi05_controller/launcher"
  log_path="${log_dir}/isaac_gui_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "${log_dir}"
  echo "Isaac GUI log: ${log_path}"
  set +e
  "${REPO_ROOT}/task2_isaacsim/scripts/run_isaacsim_teleop.sh" \
    --scene room --controller-mode none --no-browser --no-republisher -- \
    --disable-browser-command-topics --arm-pose-command-control \
    --record 2>&1 | tee "${log_path}"
  status="${PIPESTATUS[0]}"
  set -e
  echo "isaac_gui_exit_code=${status}" | tee -a "${log_path}"
  return "${status}"
}

require_positive_integer() {
  local option="$1" value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${option} must be a positive integer" >&2
    exit 2
  }
}

require_nonnegative_number() {
  local option="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "${option} must be a non-negative number" >&2
    exit 2
  }
}

command_run() {
  local requested_model="ours-20k"
  local seed=1001 execution_horizon=15 max_actions=600 max_duration_s=300
  local base_stage_max_duration_s=180 spine_stage_max_duration_s=180
  local manipulation_stage_max_duration_s=240
  local run_label="" shadow=false robot_dreams_native=false
  local hybrid_transport=false
  local execution_horizon_explicit=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) requested_model="$2"; shift 2 ;;
      --seed) seed="$2"; shift 2 ;;
      --execution-horizon)
        execution_horizon="$2"; execution_horizon_explicit=true; shift 2 ;;
      --robot-dreams-native) robot_dreams_native=true; shift ;;
      --hybrid-transport) hybrid_transport=true; shift ;;
      --shadow) shadow=true; shift ;;
      --run-label) run_label="$2"; shift 2 ;;
      --max-actions) max_actions="$2"; shift 2 ;;
      --max-duration-s) max_duration_s="$2"; shift 2 ;;
      --base-stage-max-duration-s) base_stage_max_duration_s="$2"; shift 2 ;;
      --spine-stage-max-duration-s) spine_stage_max_duration_s="$2"; shift 2 ;;
      --manipulation-stage-max-duration-s)
        manipulation_stage_max_duration_s="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown run option: $1" >&2; exit 2 ;;
    esac
  done

  resolve_model "${requested_model}"
  if ${robot_dreams_native}; then
    [[ "${MODEL_NAME}" = "robot-dreams-20k" ]] || {
      echo "--robot-dreams-native requires --model robot-dreams-20k" >&2
      exit 2
    }
    if ! ${execution_horizon_explicit}; then
      execution_horizon=50
    fi
  fi
  if ${robot_dreams_native} && ${hybrid_transport}; then
    echo "--robot-dreams-native and --hybrid-transport are mutually exclusive" >&2
    exit 2
  fi
  require_positive_integer "--execution-horizon" "${execution_horizon}"
  (( execution_horizon <= 50 )) || {
    echo "--execution-horizon must not exceed 50" >&2
    exit 2
  }
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
  require_positive_integer "--max-actions" "${max_actions}"
  require_nonnegative_number "--max-duration-s" "${max_duration_s}"
  require_nonnegative_number "--base-stage-max-duration-s" "${base_stage_max_duration_s}"
  require_nonnegative_number "--spine-stage-max-duration-s" "${spine_stage_max_duration_s}"
  require_nonnegative_number "--manipulation-stage-max-duration-s" "${manipulation_stage_max_duration_s}"

  [[ -f "${MODEL_PATH}/model.safetensors" ]] || {
    echo "Model checkpoint is missing: ${MODEL_PATH}" >&2
    exit 2
  }
  [[ -f "${DATASET_ROOT}/meta/info.json" ]] || {
    echo "Dataset is missing: ${DATASET_ROOT}" >&2
    exit 2
  }
  [[ -f "${STAGING_AUDIT}" ]] || {
    echo "Staging audit is missing: ${STAGING_AUDIT}" >&2
    exit 2
  }
  [[ -d "${LEROBOT_SRC}/lerobot" ]] || {
    echo "LeRobot runtime source is missing: ${LEROBOT_SRC}" >&2
    exit 2
  }

  if [[ -z "${run_label}" ]]; then
    run_label="${MODEL_NAME}-seed${seed}-h${execution_horizon}"
  fi
  [[ "${run_label}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "--run-label contains unsupported characters" >&2
    exit 2
  }

  local output="${TASK2_PI05_ROOT}/outputs/live_${run_label}_$(date +%Y%m%d_%H%M%S)"
  local queue_refill_actions=24
  local max_decisions=$(( (max_actions + execution_horizon - 1) / execution_horizon ))
  local -a runner_mode=(--arm-simulator)
  local -a ownership_mode=(--right-only-policy-after-staging)
  local runtime_mode=hard5
  if ${robot_dreams_native}; then
    ownership_mode=(--whole-body-policy-after-staging)
    runtime_mode=legacy
    # In asynchronous mode a newly inferred 50-action chunk replaces the
    # residual queue once only queue_refill_actions remain.  Execution horizon
    # therefore does not determine the number of inference calls.  Leave two
    # calls of margin; max-publish-actions remains the hard rollout bound.
    max_decisions=$(( (max_actions + queue_refill_actions - 1) / queue_refill_actions + 2 ))
  fi
  if ${shadow}; then
    max_decisions=1
    runner_mode=()
  fi
  mkdir -p "${output}" "${TASK2_PI05_ROOT}/evidence/pi05_controller"

  echo "model=${MODEL_NAME}"
  echo "contract=${MODEL_CONTRACT}"
  echo "checkpoint=${MODEL_PATH}"
  echo "output=${output}"
  echo "policy_execution_horizon=${execution_horizon}"
  if ${robot_dreams_native}; then
    echo "policy_input_contract=three_cameras_plus_37d_state"
    echo "policy_command_ownership=left_and_right_arms_grippers_spine_base_fixed"
  else
    echo "policy_command_ownership=right_arm_and_right_gripper_only"
  fi
  if ${hybrid_transport}; then
    echo "post_grasp_control=rmpflow_lift_transfer_place_release"
  fi

  live_shell "ros2 topic pub --once /isaac/task2/scene_reset_request std_msgs/msg/String '{data: seed=${seed}}'"
  live_shell "python3 /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/eval_camera_preflight.py --output /data/evidence/pi05_controller/eval_preflight.json"

  local -a hybrid_args=()
  if ${hybrid_transport}; then
    hybrid_args=(--hybrid-rmpflow-transport)
  fi

  exec docker run --rm --gpus all --network host --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/ebim-live-home -e USER=ebim -e LOGNAME=ebim \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -e PYTHONPATH=/workspace/EBiM_Challenge \
    -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "${TASK2_PI05_ROOT}/cache:/cache" \
    -v "${REPO_ROOT}:/workspace/EBiM_Challenge:ro" \
    -v "${LEROBOT_SRC}:/opt/lerobot/src:ro" \
    -v "${MODEL_PATH}:/data/checkpoint:ro" \
    -v "${DATASET_ROOT}:/data/dataset:ro" \
    -v "${STAGING_AUDIT}:/data/staging_audit.json:ro" \
    -v "${output}:/data/output" \
    "${PI05_LIVE_IMAGE}" run-task \
    --checkpoint /data/checkpoint \
    --dataset-root /data/dataset \
    --dataset-repo-id hermanprawiro/task2_fixpos_200 \
    --output-dir /data/output \
    --base-target 2.1000518798828125 3.0632177591323853 -1.570482850074768 \
    --base-coordinate-frame dataset_odom_world_verified_against_room_scene \
    --confirm-fixed-base-staging \
    --stage-base-after-policy-load \
    --base-stage-max-duration-s "${base_stage_max_duration_s}" \
    --stage-spine-after-base \
    --spine-stage-max-duration-s "${spine_stage_max_duration_s}" \
    --stage-manipulation-after-base \
    --staging-mode rmpflow_observation \
    --observation-reference /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/observation_reference_180dev_v1.json \
    --staging-audit /data/staging_audit.json \
    --manipulation-stage-max-duration-s "${manipulation_stage_max_duration_s}" \
    --runtime-mode "${runtime_mode}" \
    --execution-horizon "${execution_horizon}" \
    --queue-refill-actions "${queue_refill_actions}" \
    --seed "${seed}" \
    "${ownership_mode[@]}" \
    "${hybrid_args[@]}" \
    --confirm-right-wrist-pad-visible \
    --position-tolerance-m 0.015 \
    --yaw-tolerance-rad 0.04 \
    "${runner_mode[@]}" \
    --max-decisions "${max_decisions}" \
    --max-publish-actions "${max_actions}" \
    --max-duration-s "${max_duration_s}"
}

command_down() {
  "${REPO_ROOT}/scripts/evaluation/task2/run.sh" down || true
  docker compose -f "${REPO_ROOT}/task2_isaacsim/docker-compose.yml" down
  docker exec "${ISAACSIM_CONTAINER}" \
    pkill -f task2_isaacsim/scripts/scene_room.py || true
}

case "${1:-}" in
  models) shift; [[ $# -eq 0 ]] || { echo "models takes no arguments" >&2; exit 2; }; command_models ;;
  sim-up) shift; command_sim_up "$@" ;;
  run|run-task) shift; command_run "$@" ;;
  down) shift; [[ $# -eq 0 ]] || { echo "down takes no arguments" >&2; exit 2; }; command_down ;;
  -h|--help|help|"") usage ;;
  *) echo "Unknown command: $1" >&2; usage; exit 2 ;;
esac
