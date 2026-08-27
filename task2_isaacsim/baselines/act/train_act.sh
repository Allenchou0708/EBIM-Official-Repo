#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DATASET_ROOT="${ACT_DATASET_ROOT:-/scratch1/2026_ebim/allen_task2_act/datasets/task2_fixpos_200_act}"
OUTPUT_PATH="${ACT_OUTPUT_PATH:-/scratch1/2026_ebim/allen_task2_act/outputs/task2_act_paper_defaults}"
STEPS=46000
BATCH_SIZE=8
LEARNING_RATE=1e-5
# Original ACT saves every 100 epochs. At 180 episodes and batch 8 this is
# 100 * ceil(180 / 8) = 2300 optimizer steps.
CHECKPOINT_EVERY=2300

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$(realpath "$2")"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  exec python -m task2_isaacsim.baselines.act.train \
  --dataset-root "${DATASET_ROOT}" \
  --output-path "${OUTPUT_PATH}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --checkpoint-every "${CHECKPOINT_EVERY}"
