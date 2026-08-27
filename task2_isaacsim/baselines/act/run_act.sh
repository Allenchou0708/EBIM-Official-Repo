#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PI05_DIR="${REPO_ROOT}/task2_isaacsim/baselines/pi05"

TASK2_PI05_ROOT="${TASK2_PI05_ROOT:-/scratch1/2026_ebim/allen_task2_pi05}"
TASK2_ACT_ROOT="${TASK2_ACT_ROOT:-/scratch1/2026_ebim/allen_task2_act}"
ACT_DATASET_SOURCE="${ACT_DATASET_SOURCE:-${TASK2_PI05_ROOT}/datasets/task2_fixpos_200_46ab41f}"
ACT_DATASET_ROOT="${ACT_DATASET_ROOT:-${TASK2_ACT_ROOT}/datasets/task2_fixpos_200_act}"
ACT_OUTPUT_PATH="${ACT_OUTPUT_PATH:-${TASK2_ACT_ROOT}/outputs/task2_act_paper_defaults}"
ACT_CHECKPOINT="${ACT_CHECKPOINT:-${ACT_OUTPUT_PATH}/checkpoints/last/pretrained_model}"
PI05_LIVE_IMAGE="${PI05_LIVE_IMAGE:-ebim-task2-pi05-submit:v3-hard5-20260814}"
export TASK2_PI05_ROOT PI05_LIVE_IMAGE

usage() {
  cat <<'EOF'
Usage:
  ./run_act.sh prepare
  ./run_act.sh train [train.py arguments]
  ./run_act.sh test [--batch-size N] [--checkpoint PATH] [--output PATH]
  ./run_act.sh sim-up --gui
  ./run_act.sh run-task [--checkpoint PATH] [--runtime-mode hard5|legacy] [--max-actions N] [...]
  ./run_act.sh evaluate
  ./run_act.sh down
EOF
}

command_prepare() {
  mkdir -p "${TASK2_ACT_ROOT}/datasets"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m task2_isaacsim.baselines.act.prepare_dataset \
    --source "${ACT_DATASET_SOURCE}" --output "${ACT_DATASET_ROOT}" "$@"
}

command_train() {
  "${SCRIPT_DIR}/train_act.sh" \
    --dataset-root "${ACT_DATASET_ROOT}" \
    --output-path "${ACT_OUTPUT_PATH}" "$@"
}

command_test() {
  local checkpoint="${ACT_CHECKPOINT}"
  local output="${TASK2_ACT_ROOT}/evidence/act_test_metrics.json"
  local batch_size=8
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint) checkpoint="$(realpath "$2")"; shift 2 ;;
      --output) output="$2"; shift 2 ;;
      --batch-size) batch_size="$2"; shift 2 ;;
      *) echo "Unknown test argument: $1" >&2; exit 2 ;;
    esac
  done
  mkdir -p "$(dirname "${output}")"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m task2_isaacsim.baselines.act.test \
    --checkpoint "${checkpoint}" --dataset-root "${ACT_DATASET_ROOT}" \
    --batch-size "${batch_size}" --output "${output}"
}

command_run_task() {
  local -a arguments=(--policy-type act --checkpoint "${ACT_CHECKPOINT}" --dataset-root "${ACT_DATASET_ROOT}")
  "${PI05_DIR}/run_pi05.sh" run-task "${arguments[@]}" "$@"
}

command_evaluate() {
  "${PI05_DIR}/run_pi05.sh" evaluate "$@"
  local evaluator_env="${EVAL_TASK2_ENV_FILE:-${REPO_ROOT}/scripts/evaluation/task2/.env}"
  if [[ -f "${evaluator_env}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${evaluator_env}"
    set +a
  fi
  local evaluation_root="${ISAAC_DOCKER_ROOT:-${HOME}/docker/ebim-challenge}/eval-task2/evaluate"
  local metric_output="${TASK2_ACT_ROOT}/evidence/act_official_metric.json"
  mkdir -p "$(dirname "${metric_output}")"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m task2_isaacsim.baselines.act.official_metric \
    --input-dir "${evaluation_root}" --output "${metric_output}"
}

case "${1:-}" in
  prepare) shift; command_prepare "$@" ;;
  train) shift; command_train "$@" ;;
  test) shift; command_test "$@" ;;
  sim-up) shift; "${PI05_DIR}/run_pi05.sh" sim-up "$@" ;;
  run-task) shift; command_run_task "$@" ;;
  evaluate) shift; command_evaluate "$@" ;;
  down) shift; "${PI05_DIR}/run_pi05.sh" down "$@" ;;
  *) usage; exit 2 ;;
esac
