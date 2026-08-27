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
PI05_V3_INIT_CHECKPOINT="${PI05_V3_INIT_CHECKPOINT:-${TASK2_PI05_ROOT}/outputs/task2_200_30k_v1/training/checkpoints/030000/pretrained_model}"
PI05_V4_INIT_CHECKPOINT="${PI05_V4_INIT_CHECKPOINT:-${TASK2_PI05_ROOT}/outputs/task2_pi05_v2_12k/training_12k/checkpoints/006000/pretrained_model}"
PI05_V4_DATASET="${PI05_V4_DATASET:-${TASK2_PI05_ROOT}/outputs/task2_pi05_v4_pregrasp/phase_conditioned_dataset}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-}"
PI05_RELATIVE_DATASET="${PI05_RELATIVE_DATASET:-}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_DOMAIN_ID
ISAACSIM_CONTAINER="${ISAACSIM_CONTAINER:-isaac-sim-5-1-0-workshop}"
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/task2_fixpos_200_expert.yaml"

usage() {
  cat <<'EOF'
Usage:
  ./run_pi05.sh doctor [--profile NAME]
  ./run_pi05.sh parser-gate [--profile NAME]
  ./run_pi05.sh dataset [--config PATH]
  ./run_pi05.sh audit-staging [--dataset-root PATH] [--output-dir PATH]
  ./run_pi05.sh train [--config PATH] [--run NAME] [--dry-run] [--one-step-smoke] [--vram-smoke-run NAME]
  ./run_pi05.sh verify-training [--run NAME] [--mode v2_expert_30k|v2_full_30k]
  ./run_pi05.sh verify-smoke [--run NAME]
  ./run_pi05.sh verify-models-30k
  ./run_pi05.sh verify-shadow --run-dir PATH [--contract v1|v2]
  ./run_pi05.sh offline-models-30k [--samples-per-episode N]
  ./run_pi05.sh heldout-models-30k [--max-frames N]
  ./run_pi05.sh offline-gate [--run NAME] [--max-frames N]
  ./run_pi05.sh train-v4 [--run NAME]
  ./run_pi05.sh gate-v4 [--checkpoint PATH] [--maximum-episodes N]
  ./run_pi05.sh train-v5 [--run NAME] [--steps N]
  ./run_pi05.sh gate-v5 [--checkpoint PATH] [--output PATH] [--maximum-episodes N]
  ./run_pi05.sh sim-up --gui [--hybrid]
  ./run_pi05.sh stage-init [--staging-audit PATH] [--output-dir PATH] [--max-duration-s S]
  ./run_pi05.sh run-task [--policy-type pi05|act] [--hybrid-gt-pregrasp] [--runtime-mode hard5|legacy] [--checkpoint PATH] [--dataset-root PATH] [--staging-audit PATH] [--run-label LABEL] [--shadow] [--confirm-right-wrist-pad-visible] [--max-actions N] [--max-duration-s S]
  ./run_pi05.sh replay-dataset [--dataset-root PATH] [--episode auto|N] [--summary-only|--align-only|--max-frames N]
  ./run_pi05.sh audit-initial-states [--dataset-root PATH] [--output-dir PATH]
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
  ONE_STEP_SMOKE=false
  VRAM_SMOKE_RUN=""
  TRAIN_EXECUTE=true
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config) CONFIG="$(resolve_config "$2")"; shift 2 ;;
      --run) RUN_NAME="$2"; shift 2 ;;
      --one-step-smoke) ONE_STEP_SMOKE=true; shift ;;
      --vram-smoke-run) VRAM_SMOKE_RUN="$2"; shift 2 ;;
      --dry-run) TRAIN_EXECUTE=false; shift ;;
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
    -v "${REPO_ROOT}:/opt/ebim:ro"
  )
  if [[ -d "${PI05_V3_INIT_CHECKPOINT}" ]]; then
    TRAINING_ARGS+=(
      -v "${PI05_V3_INIT_CHECKPOINT}:/data/init-checkpoint:ro"
    )
  fi
}

command_doctor() {
  local profile=expert
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile) profile="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  training_base_args
  docker run "${TRAINING_ARGS[@]}" "${PI05_TRAIN_IMAGE}" \
    doctor --profile "${profile}"
}

command_parser_gate() {
  local profile=v2_full_30k
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile) profile="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  training_base_args
  docker run "${TRAINING_ARGS[@]}" "${PI05_TRAIN_IMAGE}" \
    parser-gate --profile "${profile}"
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
  local local_dir dataset_root audit output episodes profile container_root
  local dataset_container audit_container output_container
  local_dir="$(config_value "${CONFIG}" dataset.local_dir)"
  RUN_NAME="${RUN_NAME:-$(config_value "${CONFIG}" training.run)}"
  profile="$(config_value "${CONFIG}" training.profile)"
  dataset_root="${TASK2_PI05_ROOT}/datasets/${local_dir}"
  audit="${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/task2_fixpos_200_audit.json"
  output="${TASK2_PI05_ROOT}/outputs/${RUN_NAME}"
  container_root="/data/task2_pi05"
  case "${dataset_root}" in
    "${TASK2_PI05_ROOT}"/*)
      dataset_container="${container_root}/${dataset_root#"${TASK2_PI05_ROOT}"/}"
      ;;
    *) echo "Dataset must be below TASK2_PI05_ROOT" >&2; exit 2 ;;
  esac
  audit_container="${container_root}/${audit#"${TASK2_PI05_ROOT}"/}"
  output_container="${container_root}/${output#"${TASK2_PI05_ROOT}"/}"
  episodes="$(python3 -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["split"]["train"])))' "${audit}")"
  training_base_args
  local -a smoke_args=()
  local -a execute_args=(--execute)
  ${TRAIN_EXECUTE} || execute_args=()
  if ${ONE_STEP_SMOKE}; then
    smoke_args=(
      --allow-vram-smoke
      --override=--steps=1
      --override=--save_checkpoint=false
      --override=--log_freq=1
    )
  fi
  if [[ -n "${VRAM_SMOKE_RUN}" ]]; then
    smoke_args+=(
      --vram-smoke-report
      "${container_root}/outputs/${VRAM_SMOKE_RUN}/run_manifest.json"
    )
  fi
  docker run "${TRAINING_ARGS[@]}" \
    -v "${TASK2_PI05_ROOT}:${container_root}" \
    "${PI05_TRAIN_IMAGE}" train --profile "${profile}" \
    --dataset-root "${dataset_container}" \
    --audit-report "${audit_container}" \
    --audit-dataset-root /data/dataset \
    --output-dir "${output_container}" \
    --episodes "${episodes}" "${smoke_args[@]}" "${execute_args[@]}"
}

command_audit_staging() {
  local dataset="${TASK2_PI05_ROOT}/datasets/task2_fixpos_200_46ab41f"
  local audit="${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/task2_fixpos_200_audit.json"
  local output="${TASK2_PI05_ROOT}/evidence/task2_pi05_camera_ready_pad_relative_20260815"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset-root) dataset="$(realpath "$2")"; shift 2 ;;
      --output-dir) output="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  dataset="$(realpath "${dataset}")"
  mkdir -p "${output}"
  output="$(realpath "${output}")"
  [[ -f "${audit}" ]] || { echo "Dataset audit report not found" >&2; exit 2; }
  training_base_args
  docker run "${TRAINING_ARGS[@]}" \
    -v "${dataset}:/data/dataset:ro" \
    -v "${audit}:/data/dataset_audit.json:ro" \
    -v "${output}:/data/output" \
    "${PI05_TRAIN_IMAGE}" audit-staging \
    --dataset-root /data/dataset \
    --audit-report /data/dataset_audit.json \
    --output-dir /data/output
}

command_verify_training() {
  local run_name="task2_pi05_v2_expert_30k" mode="v2_expert_30k"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run_name="$2"; shift 2 ;;
      --mode) mode="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  local run_root output
  run_root="${TASK2_PI05_ROOT}/outputs/${run_name}"
  output="${TASK2_PI05_ROOT}/evidence/${run_name}/training_verification.json"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -m task2_isaacsim.baselines.pi05.verify_v2_full_run \
    --run-root "${run_root}" --output "${output}" --mode "${mode}"
}

command_verify_smoke() {
  local run_name="task2_pi05_v2_full_30k_smoke_1step"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run_name="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -m task2_isaacsim.baselines.pi05.verify_vram_smoke \
    --manifest "${TASK2_PI05_ROOT}/outputs/${run_name}/run_manifest.json"
}

command_verify_shadow() {
  local run_dir="" contract="v2"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-dir) run_dir="$(realpath "$2")"; shift 2 ;;
      --contract) contract="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ -n "${run_dir}" ]] || { echo "--run-dir is required" >&2; exit 2; }
  [[ "${contract}" = "v1" || "${contract}" = "v2" ]] || {
    echo "--contract must be v1 or v2" >&2
    exit 2
  }
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -m task2_isaacsim.baselines.pi05.verify_shadow_run \
    --run-dir "${run_dir}" --contract "${contract}"
}

command_verify_models_30k() {
  local v1 v2 output
  v1="${TASK2_PI05_ROOT}/outputs/task2_200_30k_v1/training/checkpoints/030000"
  v2="${TASK2_PI05_ROOT}/outputs/task2_pi05_v2_expert_30k/training/checkpoints/030000"
  output="${TASK2_PI05_ROOT}/evidence/task2_pi05_v1_v2_30k/model_verification.json"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -m task2_isaacsim.baselines.pi05.verify_v1_v2_30k \
    --v1-checkpoint "${v1}" --v2-checkpoint "${v2}" --output "${output}"
}

command_offline_models_30k() {
  local samples=3
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --samples-per-episode) samples="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ "${samples}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--samples-per-episode must be a positive integer" >&2
    exit 2
  }
  local v1 v2 dataset audit output episodes
  v1="${TASK2_PI05_ROOT}/outputs/task2_200_30k_v1/training/checkpoints/030000/pretrained_model"
  v2="${TASK2_PI05_ROOT}/outputs/task2_pi05_v2_expert_30k/training/checkpoints/030000/pretrained_model"
  dataset="${TASK2_PI05_ROOT}/datasets/task2_fixpos_200_46ab41f"
  audit="${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/task2_fixpos_200_audit.json"
  output="${TASK2_PI05_ROOT}/evidence/task2_pi05_v1_v2_30k"
  [[ -d "${v1}" ]] || { echo "V1 030000 checkpoint not found" >&2; exit 2; }
  [[ -d "${v2}" ]] || { echo "V2 030000 checkpoint not found" >&2; exit 2; }
  [[ -d "${dataset}" ]] || { echo "Raw dataset not found" >&2; exit 2; }
  [[ -f "${audit}" ]] || { echo "Dataset audit not found" >&2; exit 2; }
  mkdir -p "${output}"
  episodes="$(python3 -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["split"]["held_out"])))' "${audit}")"
  training_base_args
  local contract checkpoint report
  for contract in v1 v2; do
    if [[ "${contract}" = "v1" ]]; then checkpoint="${v1}"; else checkpoint="${v2}"; fi
    report="${output}/${contract}_030000_offline.json"
    docker run "${TRAINING_ARGS[@]}" \
      -v "${checkpoint}:/data/checkpoint:ro" \
      -v "${dataset}:/data/dataset:ro" \
      -v "${audit}:/data/dataset_audit.json:ro" \
      -v "${output}:/data/output" \
      "${PI05_TRAIN_IMAGE}" offline-inference \
      --checkpoint /data/checkpoint \
      --dataset-root /data/dataset \
      --dataset-repo-id hermanprawiro/task2_fixpos_200 \
      --episodes "${episodes}" \
      --audit-report /data/dataset_audit.json \
      --samples-per-episode "${samples}" --seed 1000 \
      --output "/data/output/$(basename "${report}")"
  done
}

command_heldout_models_30k() {
  local max_frames=128
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --max-frames) max_frames="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ "${max_frames}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--max-frames must be a positive integer" >&2
    exit 2
  }
  local v1 v2 dataset audit output contract checkpoint report report_dir
  v1="${TASK2_PI05_ROOT}/outputs/task2_200_30k_v1/training/checkpoints/030000"
  v2="${TASK2_PI05_ROOT}/outputs/task2_pi05_v2_expert_30k/training/checkpoints/030000"
  dataset="${TASK2_PI05_ROOT}/datasets/task2_fixpos_200_46ab41f"
  audit="${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/task2_fixpos_200_audit.json"
  output="${TASK2_PI05_ROOT}/evidence/task2_pi05_v1_v2_30k"
  [[ -d "${v1}/pretrained_model" ]] || { echo "V1 030000 checkpoint not found" >&2; exit 2; }
  [[ -d "${v2}/pretrained_model" ]] || { echo "V2 030000 checkpoint not found" >&2; exit 2; }
  [[ -d "${dataset}" ]] || { echo "Raw dataset not found" >&2; exit 2; }
  [[ -f "${audit}" ]] || { echo "Dataset audit not found" >&2; exit 2; }
  mkdir -p "${output}"
  training_base_args
  for contract in v1 v2; do
    if [[ "${contract}" = "v1" ]]; then checkpoint="${v1}"; else checkpoint="${v2}"; fi
    report="${output}/${contract}_030000_heldout_gate.json"
    report_dir="${output}/${contract}_030000_heldout_gate_checkpoint_reports"
    [[ ! -e "${report}" && ! -e "${report_dir}" ]] || {
      echo "Held-out output already exists for ${contract}: ${report}" >&2
      exit 2
    }
    docker run "${TRAINING_ARGS[@]}" \
      -v "${checkpoint}:/data/checkpoints/030000:ro" \
      -v "${dataset}:/data/dataset:ro" \
      -v "${audit}:/data/dataset_audit.json:ro" \
      -v "${output}:/data/output" \
      "${PI05_TRAIN_IMAGE}" checkpoint-sweep \
      --checkpoints-root /data/checkpoints \
      --dataset-root /data/dataset \
      --dataset-repo-id hermanprawiro/task2_fixpos_200 \
      --audit-report /data/dataset_audit.json \
      --output "/data/output/$(basename "${report}")" \
      --max-frames "${max_frames}" --seed 1000
  done
}

command_offline_gate() {
  local run_name="task2_pi05_v2_full_30k" max_frames=128
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run_name="$2"; shift 2 ;;
      --max-frames) max_frames="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ "${max_frames}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--max-frames must be a positive integer" >&2
    exit 2
  }
  local checkpoints dataset audit output_dir
  checkpoints="${TASK2_PI05_ROOT}/outputs/${run_name}/training/checkpoints"
  dataset="${TASK2_PI05_ROOT}/datasets/task2_fixpos_200_46ab41f"
  audit="${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/task2_fixpos_200_audit.json"
  output_dir="${TASK2_PI05_ROOT}/evidence/task2_pi05_v2_full_30k_offline_gate"
  [[ -d "${checkpoints}" ]] || { echo "Checkpoint root not found" >&2; exit 2; }
  [[ -d "${dataset}" ]] || { echo "Raw dataset not found" >&2; exit 2; }
  [[ -f "${audit}" ]] || { echo "Dataset audit not found" >&2; exit 2; }
  mkdir -p "${output_dir}"
  training_base_args
  docker run "${TRAINING_ARGS[@]}" \
    -v "${checkpoints}:/data/checkpoints:ro" \
    -v "${dataset}:/data/dataset:ro" \
    -v "${audit}:/data/dataset_audit.json:ro" \
    -v "${output_dir}:/data/output" \
    "${PI05_TRAIN_IMAGE}" checkpoint-sweep \
    --checkpoints-root /data/checkpoints \
    --dataset-root /data/dataset \
    --dataset-repo-id hermanprawiro/task2_fixpos_200 \
    --audit-report /data/dataset_audit.json \
    --output /data/output/offline_gate.json \
    --max-frames "${max_frames}" --seed 1000
}

command_train_v4() {
  local run_name="task2_pi05_v4_from_v2_3k"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run_name="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  local output pose_audit
  output="${TASK2_PI05_ROOT}/outputs/${run_name}"
  pose_audit="${TASK2_PI05_ROOT}/evidence/task2_pi05_pregrasp_20260814/pregrasp_pose_audit.json"
  [[ -d "${PI05_V4_INIT_CHECKPOINT}" ]] || { echo "V2 init checkpoint not found" >&2; exit 2; }
  [[ -d "${PI05_V4_DATASET}" ]] || { echo "V4 phase dataset not found" >&2; exit 2; }
  [[ -f "${pose_audit}" ]] || { echo "V4 pose audit not found" >&2; exit 2; }
  mkdir -p "${output}"
  training_base_args
  docker run "${TRAINING_ARGS[@]}" \
    -e EBIM_PI05_IMAGE="${PI05_TRAIN_IMAGE}" \
    -e HF_DATASETS_CACHE=/data/output/hf_datasets_cache \
    -v "${PI05_V4_INIT_CHECKPOINT}:/data/v2-checkpoint:ro" \
    -v "${PI05_V4_DATASET}:/data/dataset:ro" \
    -v "${pose_audit}:/data/pregrasp_pose_audit.json:ro" \
    -v "${output}:/data/output" \
    "${PI05_TRAIN_IMAGE}" v4-train \
    --checkpoint /data/v2-checkpoint \
    --dataset-root /data/dataset \
    --pose-audit /data/pregrasp_pose_audit.json \
    --output-dir /data/output --execute
}

command_gate_v4() {
  local checkpoint="${TASK2_PI05_ROOT}/outputs/task2_pi05_v4_from_v2_3k/training/checkpoints/003000/pretrained_model"
  local maximum_episodes="" pose_audit output
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint) checkpoint="$(realpath "$2")"; shift 2 ;;
      --maximum-episodes) maximum_episodes="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ -d "${checkpoint}" ]] || { echo "V4 checkpoint not found" >&2; exit 2; }
  [[ -z "${maximum_episodes}" || "${maximum_episodes}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--maximum-episodes must be a positive integer" >&2
    exit 2
  }
  pose_audit="${TASK2_PI05_ROOT}/evidence/task2_pi05_pregrasp_20260814/pregrasp_pose_audit.json"
  output="$(realpath "${checkpoint}/../../../..")/offline_gate"
  mkdir -p "${output}"
  training_base_args
  local -a episode_arg=()
  [[ -n "${maximum_episodes}" ]] && episode_arg=(--maximum-episodes "${maximum_episodes}")
  docker run "${TRAINING_ARGS[@]}" \
    -e HF_DATASETS_CACHE=/data/output/hf_datasets_cache \
    -v "${checkpoint}:/data/checkpoint:ro" \
    -v "${PI05_V4_DATASET}:/data/dataset:ro" \
    -v "${pose_audit}:/data/pregrasp_pose_audit.json:ro" \
    -v "${output}:/data/output" \
    "${PI05_TRAIN_IMAGE}" v4-offline-gate \
    --checkpoint /data/checkpoint --dataset-root /data/dataset \
    --pose-audit /data/pregrasp_pose_audit.json \
    --output /data/output/v4_offline_gate.json "${episode_arg[@]}"
}

command_train_v5() {
  local run_name="task2_pi05_v5_weighted_gripper_3k" steps=3000
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run_name="$2"; shift 2 ;;
      --steps) steps="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ "${steps}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--steps must be a positive integer" >&2
    exit 2
  }
  [[ "${run_name}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "--run must be a simple run name" >&2
    exit 2
  }
  local checkpoint dataset phase_manifest output
  checkpoint="${TASK2_PI05_ROOT}/outputs/task2_pi05_v2_expert_30k/training/checkpoints/030000/pretrained_model"
  dataset="${TASK2_PI05_ROOT}/outputs/task2_pi05_v2_expert_30k/relative_dataset"
  phase_manifest="${TASK2_PI05_ROOT}/outputs/task2_pi05_v2_expert_30k/phase_manifest.json"
  output="${TASK2_PI05_ROOT}/outputs/${run_name}"
  [[ -d "${checkpoint}" ]] || { echo "V2 30k checkpoint not found" >&2; exit 2; }
  [[ -d "${dataset}" ]] || { echo "V2 relative dataset not found" >&2; exit 2; }
  [[ -f "${phase_manifest}" ]] || { echo "V2 phase manifest not found" >&2; exit 2; }
  mkdir -p "${output}"
  training_base_args
  docker run "${TRAINING_ARGS[@]}" \
    -v "${TASK2_PI05_ROOT}:/data/task2_pi05" \
    -v "${checkpoint}:/data/v2-checkpoint:ro" \
    "${PI05_TRAIN_IMAGE}" v5-train \
    --checkpoint /data/v2-checkpoint \
    --dataset-root /data/task2_pi05/outputs/task2_pi05_v2_expert_30k/relative_dataset \
    --phase-manifest /data/task2_pi05/outputs/task2_pi05_v2_expert_30k/phase_manifest.json \
    --output-dir "/data/task2_pi05/outputs/${run_name}" \
    --steps "${steps}" --execute
}

command_gate_v5() {
  local checkpoint maximum_episodes="" output
  checkpoint="${TASK2_PI05_ROOT}/outputs/task2_pi05_v5_weighted_gripper_3k/training/checkpoints/003000/pretrained_model"
  output="${TASK2_PI05_ROOT}/evidence/task2_pi05_v5_weighted_gripper_3k/gripper_hold_gate.json"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint) checkpoint="$(realpath "$2")"; shift 2 ;;
      --output) output="$2"; shift 2 ;;
      --maximum-episodes) maximum_episodes="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ -d "${checkpoint}" ]] || { echo "V5 checkpoint not found" >&2; exit 2; }
  [[ -z "${maximum_episodes}" || "${maximum_episodes}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--maximum-episodes must be a positive integer" >&2
    exit 2
  }
  local dataset phase_manifest
  dataset="${TASK2_PI05_ROOT}/outputs/task2_pi05_v2_expert_30k/relative_dataset"
  phase_manifest="${TASK2_PI05_ROOT}/outputs/task2_pi05_v2_expert_30k/phase_manifest.json"
  mkdir -p "$(dirname "${output}")"
  output="$(realpath "${output}")"
  [[ "${output}" = "${TASK2_PI05_ROOT}"/* ]] || {
    echo "--output must be below TASK2_PI05_ROOT" >&2
    exit 2
  }
  training_base_args
  local -a episode_arg=()
  [[ -n "${maximum_episodes}" ]] && episode_arg=(--maximum-episodes "${maximum_episodes}")
  docker run "${TRAINING_ARGS[@]}" \
    -v "${TASK2_PI05_ROOT}:/data/task2_pi05" \
    -v "${checkpoint}:/data/checkpoint:ro" \
    "${PI05_TRAIN_IMAGE}" v5-gripper-gate \
    --checkpoint /data/checkpoint \
    --dataset-root /data/task2_pi05/outputs/task2_pi05_v2_expert_30k/relative_dataset \
    --phase-manifest /data/task2_pi05/outputs/task2_pi05_v2_expert_30k/phase_manifest.json \
    --output "/data/task2_pi05/${output#"${TASK2_PI05_ROOT}/"}" \
    "${episode_arg[@]}"
}

live_shell() {
  docker run --rm --network host --entrypoint bash \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -v "${TASK2_PI05_ROOT}/evidence:/data/evidence" \
    "${PI05_LIVE_IMAGE}" -lc "source /opt/ros/jazzy/setup.bash && $1"
}

hybrid_shell() {
  docker run --rm --network host --ipc=host --entrypoint bash \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/ebim-live-home -e USER=ebim -e LOGNAME=ebim \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -e PYTHONPATH=/workspace/EBiM_Challenge \
    -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -v "${REPO_ROOT}:/workspace/EBiM_Challenge:ro" \
    -v "${HYBRID_STAGING_AUDIT}:/data/staging_audit.json:ro" \
    -v "${HYBRID_OUTPUT}:/data/output" \
    "${PI05_LIVE_IMAGE}" -lc \
    "source /opt/ros/jazzy/setup.bash && exec $1"
}

command_sim_up() {
  local gui=false hybrid=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gui) gui=true; shift ;;
      --hybrid) hybrid=true; shift ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  ${gui} || { echo "sim-up requires --gui" >&2; exit 2; }
  local log_dir log_path status
  local -a simulator_args=(
    --disable-browser-command-topics --record
  )
  if ${hybrid}; then
    simulator_args+=(
      --arm-pose-command-control
      --configure-gripper-drives
      --arm-teleop-gripper-closed 0.804
      --publish-ground-truth
      --scene-reset-hotkey
    )
  fi
  log_dir="${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/launcher"
  log_path="${log_dir}/isaac_gui_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "${log_dir}"
  echo "Isaac GUI log: ${log_path}"
  set +e
  "${REPO_ROOT}/task2_isaacsim/scripts/run_isaacsim_teleop.sh" \
    --scene room --controller-mode none --no-browser --no-republisher -- \
    "${simulator_args[@]}" 2>&1 | tee "${log_path}"
  status="${PIPESTATUS[0]}"
  set -e
  echo "isaac_gui_exit_code=${status}" | tee -a "${log_path}"
  return "${status}"
}

command_stage_init() {
  local staging_audit="${TASK2_PI05_ROOT}/evidence/task2_pi05_camera_ready_pad_relative_20260815/startup_staging_audit.json"
  local output_dir="${TASK2_PI05_ROOT}/outputs/stage_init_$(date +%Y%m%d_%H%M%S)"
  local max_duration_s=600
  local base_target="2.100026845932007 3.0529046058654785 -1.5706931352615356"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --staging-audit) staging_audit="$(realpath "$2")"; shift 2 ;;
      --output-dir) output_dir="$2"; shift 2 ;;
      --max-duration-s) max_duration_s="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ "${max_duration_s}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "--max-duration-s must be a non-negative number" >&2
    exit 2
  }
  staging_audit="$(realpath "${staging_audit}")"
  [[ -f "${staging_audit}" ]] || { echo "Staging audit not found" >&2; exit 2; }
  mkdir -p "${output_dir}"
  output_dir="$(realpath "${output_dir}")"

  live_shell "ros2 topic pub --once /isaac/task2/scene_reset_request std_msgs/msg/String '{data: reset}'"
  live_shell "python3 /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/fixed_stage_base.py --target ${base_target} --position-tolerance-m 0.03 --yaw-tolerance-rad 0.04 --output /data/evidence/stage_init_base.json"
  docker run --rm --network host --ipc=host --entrypoint bash \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/ebim-live-home -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
    -e PYTHONPATH=/workspace/EBiM_Challenge \
    -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -v "${REPO_ROOT}:/workspace/EBiM_Challenge:ro" \
    -v "${staging_audit}:/data/staging_audit.json:ro" \
    -v "${output_dir}:/data/output" \
    "${PI05_LIVE_IMAGE}" -lc \
    "source /opt/ros/jazzy/setup.bash && exec python3 -m task2_isaacsim.baselines.pi05.live.fixed_stage_manipulation --audit /data/staging_audit.json --output /data/output/stage_init_manifest.json --max-duration-s ${max_duration_s}"
}

command_run_task() {
  local checkpoint="${PI05_CHECKPOINT}" dataset="${PI05_RELATIVE_DATASET}"
  local staging_audit="${TASK2_PI05_ROOT}/evidence/task2_pi05_camera_ready_pad_relative_20260815/startup_staging_audit.json"
  local base_target="2.100026845932007 3.0529046058654785 -1.5706931352615356"
  local runtime_mode=hard5 max_actions=600 max_duration_s=300 max_decisions shadow=false
  local policy_type=pi05
  local run_label="unlabeled" confirm_right_wrist_pad_visible=false
  local hybrid_gt_pregrasp=false
  local -a runner_mode=(--arm-simulator)
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint) checkpoint="$(realpath "$2")"; shift 2 ;;
      --dataset-root) dataset="$(realpath "$2")"; shift 2 ;;
      --staging-audit) staging_audit="$(realpath "$2")"; shift 2 ;;
      --run-label) run_label="$2"; shift 2 ;;
      --runtime-mode) runtime_mode="$2"; shift 2 ;;
      --policy-type) policy_type="$2"; shift 2 ;;
      --hybrid-gt-pregrasp) hybrid_gt_pregrasp=true; shift ;;
      --shadow) shadow=true; shift ;;
      --confirm-right-wrist-pad-visible) confirm_right_wrist_pad_visible=true; shift ;;
      --max-actions) max_actions="$2"; shift 2 ;;
      --max-duration-s) max_duration_s="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ -n "${checkpoint}" ]] || { echo "--checkpoint is required" >&2; exit 2; }
  [[ "${run_label}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "--run-label must use only letters, digits, dot, underscore, or dash" >&2
    exit 2
  }
  [[ "${runtime_mode}" = "hard5" || "${runtime_mode}" = "legacy" ]] || {
    echo "--runtime-mode must be hard5 or legacy" >&2
    exit 2
  }
  [[ "${policy_type}" = "pi05" || "${policy_type}" = "act" ]] || {
    echo "--policy-type must be pi05 or act" >&2
    exit 2
  }
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
  if [[ "${runtime_mode}" = "hard5" ]]; then
    max_decisions=$(( (max_actions + 4) / 5 ))
  else
    max_decisions=$(( (max_actions + 23) / 24 + 2 ))
    if (( max_decisions < 40 )); then
      max_decisions=40
    fi
  fi
  if ${shadow}; then
    max_decisions=1
    runner_mode=()
  fi
  local run_root output
  if [[ -z "${dataset}" ]]; then
    run_root="$(realpath "${checkpoint}/../../../..")"
    dataset="${run_root}/relative_dataset"
  fi
  dataset="$(realpath "${dataset}")"
  [[ -d "${dataset}" ]] || { echo "Relative dataset directory not found" >&2; exit 2; }
  staging_audit="$(realpath "${staging_audit}")"
  [[ -f "${staging_audit}" ]] || { echo "Staging audit not found" >&2; exit 2; }
  output="${TASK2_PI05_ROOT}/outputs/live_submit_${run_label}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${output}" "${TASK2_PI05_ROOT}/evidence/task2_200_submit_20260812/launcher"
  live_shell "ros2 topic pub --once /isaac/task2/scene_reset_request std_msgs/msg/String '{data: reset}'"
  live_shell "python3 /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/live/eval_camera_preflight.py --output /data/evidence/task2_200_submit_20260812/launcher/eval_preflight.json"
  local -a staging_runner_args=(
    --stage-base-after-policy-load
    --stage-manipulation-after-base
    --staging-audit /data/staging_audit.json
    --manipulation-stage-max-duration-s 600
  )
  if ${hybrid_gt_pregrasp}; then
    HYBRID_OUTPUT="${output}"
    HYBRID_STAGING_AUDIT="${staging_audit}"
    export HYBRID_OUTPUT HYBRID_STAGING_AUDIT
    hybrid_shell "python3 -m task2_isaacsim.baselines.pi05.live.fixed_stage_base --target-from-live-pad --grasp-yaw-limit-deg 5 --grasp-depth-bias-m 0.005 --position-tolerance-m 0.006 --yaw-tolerance-rad 0.01 --velocity-threshold 0.025 --settle-duration-s 0.5 --max-duration-s 240 --output /data/output/gt_base_stage.json"
    hybrid_shell "python3 -m task2_isaacsim.baselines.pi05.live.fixed_stage_spine --max-duration-s 90 --output /data/output/gt_spine_stage.json"
    hybrid_shell "python3 -m task2_isaacsim.baselines.pi05.live.ground_truth_joint_lift --absolute-dataset-joints --no-preposition-at-live-grasp-base --grasp-yaw-limit-deg 5 --grasp-depth-bias-m 0.005 --pregrasp-only --output /data/output/gt_pregrasp_manifest.json --max-duration-s 120"
    staging_runner_args=(
      --hybrid-pregrasp-manifest /data/output/gt_pregrasp_manifest.json
    )
  fi
  local -a visibility_arg=()
  ${confirm_right_wrist_pad_visible} && visibility_arg=(--confirm-right-wrist-pad-visible)
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
    -v "${checkpoint}:/data/checkpoint:ro" \
    -v "${dataset}:/data/dataset:ro" \
    -v "${staging_audit}:/data/staging_audit.json:ro" \
    -v "${output}:/data/output" \
    "${PI05_LIVE_IMAGE}" run-task --checkpoint /data/checkpoint \
    --policy-type "${policy_type}" \
    --dataset-root /data/dataset --dataset-repo-id hermanprawiro/task2_fixpos_200 \
    --output-dir /data/output --base-target ${base_target} \
    --base-coordinate-frame dataset_odom_world_verified_against_room_scene \
    --confirm-fixed-base-staging \
    "${staging_runner_args[@]}" \
    --runtime-mode "${runtime_mode}" \
    --position-tolerance-m 0.03 --yaw-tolerance-rad 0.04 \
    "${runner_mode[@]}" \
    "${visibility_arg[@]}" \
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

command_audit_initial_states() {
  local dataset="${TASK2_PI05_ROOT}/datasets/task2_fixpos_200_46ab41f"
  local output="${TASK2_PI05_ROOT}/evidence/task2_pi05_sim_clock_startup_20260813"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset-root) dataset="$(realpath "$2")"; shift 2 ;;
      --output-dir) output="$2"; shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  dataset="$(realpath "${dataset}")"
  mkdir -p "${output}"
  output="$(realpath "${output}")"
  docker run --rm --entrypoint bash \
    -e PYTHONPATH=/workspace/EBiM_Challenge \
    -v "${REPO_ROOT}:/workspace/EBiM_Challenge:ro" \
    -v "${dataset}:/data/dataset:ro" -v "${output}:/data/output" \
    "${PI05_LIVE_IMAGE}" -lc \
    "/opt/lerobot/.venv/bin/python -m task2_isaacsim.baselines.pi05.initial_state_audit --dataset-root /data/dataset --output-dir /data/output"
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
  parser-gate) shift; command_parser_gate "$@" ;;
  dataset) shift; command_dataset "$@" ;;
  audit-staging) shift; command_audit_staging "$@" ;;
  verify-training) shift; command_verify_training "$@" ;;
  verify-smoke) shift; command_verify_smoke "$@" ;;
  verify-models-30k) shift; command_verify_models_30k "$@" ;;
  verify-shadow) shift; command_verify_shadow "$@" ;;
  offline-models-30k) shift; command_offline_models_30k "$@" ;;
  heldout-models-30k) shift; command_heldout_models_30k "$@" ;;
  offline-gate) shift; command_offline_gate "$@" ;;
  train) shift; command_train "$@" ;;
  train-v4) shift; command_train_v4 "$@" ;;
  gate-v4) shift; command_gate_v4 "$@" ;;
  train-v5) shift; command_train_v5 "$@" ;;
  gate-v5) shift; command_gate_v5 "$@" ;;
  sim-up) shift; command_sim_up "$@" ;;
  stage-init) shift; command_stage_init "$@" ;;
  run-task) shift; command_run_task "$@" ;;
  replay-dataset) shift; command_replay_dataset "$@" ;;
  audit-initial-states) shift; command_audit_initial_states "$@" ;;
  evaluate) shift; command_evaluate "$@" ;;
  down) shift; command_down "$@" ;;
  *) usage; exit 2 ;;
esac
