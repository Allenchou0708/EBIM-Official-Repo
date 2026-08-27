#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${ACT_CHECKPOINT:-${ACT_OUTPUT_PATH:-/scratch1/2026_ebim/allen_task2_act/outputs/task2_act_paper_defaults}/checkpoints/last/pretrained_model}"
BATCH_SIZE=8
OUTPUT=""
RUN_SIM=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --sim) RUN_SIM=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

offline_args=(--checkpoint "${CHECKPOINT}" --batch-size "${BATCH_SIZE}")
[[ -n "${OUTPUT}" ]] && offline_args+=(--output "${OUTPUT}")
"${SCRIPT_DIR}/run_act.sh" test "${offline_args[@]}"

if ${RUN_SIM}; then
  "${SCRIPT_DIR}/inference_act.sh" --checkpoint "${CHECKPOINT}"
  "${SCRIPT_DIR}/run_act.sh" evaluate
fi
