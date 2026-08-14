#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify the same-contract one-step V2-full VRAM smoke manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from task2_isaacsim.baselines.pi05.contract import (
    V2_RELATIVE_ACTION_STATE_INDICES,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    command = set(manifest["command"])
    profile = manifest.get("profile", {})
    policy = profile.get("policy", {})
    checks = {
        "returncode_zero": manifest.get("returncode") == 0,
        "mode_v2_full_30k": (
            profile.get("mode") == "v2_full_30k"
        ),
        "full_expert_training": policy.get("train_expert_only") is False,
        "vision_frozen": policy.get("freeze_vision_encoder") is True,
        "v2_absolute_spine_mapping": (
            profile.get("task2_relative_action_state_indices")
            == list(V2_RELATIVE_ACTION_STATE_INDICES)
        ),
        "one_step": "--steps=1" in command,
        "no_checkpoint": "--save_checkpoint=false" in command,
        "step_logged": "--log_freq=1" in command,
        "full_parameter_mode": (
            manifest.get("parameter_counts", {}).get(
                "trainable_parameters", 0
            )
            >= 3_500_000_000
        ),
        "finite_loss_reported": (
            manifest.get("training_metrics", {}).get("loss") is not None
            and math.isfinite(
                float(manifest["training_metrics"]["loss"])
            )
        ),
        "finite_vram_reported": (
            manifest.get("training_metrics", {}).get("memory_gib") is not None
            and math.isfinite(
                float(manifest["training_metrics"]["memory_gib"])
            )
        ),
        "parameter_mode_clean": not manifest.get("parameter_mode_errors"),
    }
    report = {
        "valid": all(checks.values()),
        "manifest": str(args.manifest.resolve()),
        "checks": checks,
        "parameter_counts": manifest.get("parameter_counts"),
        "training_metrics": manifest.get("training_metrics"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
