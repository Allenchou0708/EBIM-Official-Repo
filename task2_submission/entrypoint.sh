#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Keep the isolated submission package and the root image on one audited
# command implementation.
exec /workspace/EBiM_Challenge/task2_isaacsim/baselines/pi05/submission_entrypoint.sh "$@"
