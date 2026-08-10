#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

IMAGE="${PI05_LIVE_IMAGE:-ebim-task2-pi05-live:local}"
ROOT="${TASK2_PI05_ROOT:-/scratch1/2026_ebim/allen_task2_pi05}"
TRAIN_RUN="${PI05_TRAIN_RUN:-organizer_expert_6k_7dc8a54_20260809_v2}"
OUTPUT_NAME="${PI05_LIVE_OUTPUT_NAME:-live_runner_6000_20260810_v1}"
CHECKPOINT_HOST="${ROOT}/outputs/${TRAIN_RUN}/training/checkpoints/006000/pretrained_model"
DATASET_HOST="${ROOT}/outputs/${TRAIN_RUN}/relative_dataset"
OUTPUT_HOST="${ROOT}/outputs/${OUTPUT_NAME}"

mkdir -p "${OUTPUT_HOST}/hf_datasets_cache"

exec docker run --rm --gpus all --ipc=host --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/ebim-live-home \
  -e USER="${USER:-ebim}" \
  -e LOGNAME="${USER:-ebim}" \
  -e HF_HOME=/cache/huggingface \
  -e HF_DATASETS_CACHE=/data/output/hf_datasets_cache \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "${ROOT}/cache:/cache:ro" \
  -v "${CHECKPOINT_HOST}:/data/checkpoint:ro" \
  -v "${DATASET_HOST}:/data/dataset:ro" \
  -v "${OUTPUT_HOST}:/data/output" \
  "${IMAGE}" \
  --checkpoint /data/checkpoint \
  --dataset-root /data/dataset \
  --output-dir /data/output \
  "$@"
