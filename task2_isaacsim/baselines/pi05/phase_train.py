#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run LeRobot PI0.5 training with Task 2's phase-balanced sampler."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .phase_balance import PhaseBalancedSampler


def main() -> None:
    from lerobot.scripts import lerobot_train

    manifest_path = os.environ.get("EBIM_PHASE_MANIFEST", "").strip()
    if not manifest_path:
        raise ValueError("EBIM_PHASE_MANIFEST is required")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not any(
        argument.startswith("--dataset.episodes") for argument in sys.argv[1:]
    ):
        sys.argv.append(
            "--dataset.episodes="
            + json.dumps(manifest["train_episodes"], separators=(",", ":"))
        )
    lerobot_train.EpisodeAwareSampler = PhaseBalancedSampler
    lerobot_train.main()


if __name__ == "__main__":
    main()
