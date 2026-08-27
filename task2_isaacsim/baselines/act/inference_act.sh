#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reuses the proven Task 2 staging, freshness, action-bound and ROS publisher
# path. ACT replaces only the checkpoint inference implementation.
exec "${SCRIPT_DIR}/run_act.sh" run-task \
  --runtime-mode hard5 --max-actions 600 "$@"
