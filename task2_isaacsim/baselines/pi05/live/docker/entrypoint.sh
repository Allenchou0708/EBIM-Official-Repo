#!/usr/bin/env bash
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u
exec /opt/lerobot/.venv/bin/python \
  -m task2_isaacsim.baselines.pi05.live.runner "$@"
