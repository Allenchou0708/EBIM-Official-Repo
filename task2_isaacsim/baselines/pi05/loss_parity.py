#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic PI0.5 flow-matching loss contract evidence.

The test is intentionally framework-independent. It documents the numerical
difference between reducing over Task 2's 20 real action dimensions and over
the full 32-D PI0.5 boundary without changing LeRobot's native loss.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

from .contract import ACTION_SIZE, PI05_ACTION_SIZE, pad_action


def flow_target(
    actions: Sequence[float], noise: Sequence[float]
) -> tuple[float, ...]:
    """Return the common OpenPI/LeRobot flow target, ``noise - action``."""

    if len(actions) != len(noise):
        raise ValueError("actions and noise must have the same size")
    target = tuple(
        float(n) - float(a) for a, n in zip(actions, noise, strict=True)
    )
    if not all(math.isfinite(value) for value in target):
        raise ValueError("flow target contains non-finite values")
    return target


def loss_parity_report(
    actions_20: Sequence[float] | None = None,
    noise_32: Sequence[float] | None = None,
    prediction_32: Sequence[float] | None = None,
) -> dict[str, object]:
    """Build reproducible per-dimension and reduction evidence."""

    if actions_20 is None:
        actions_20 = tuple(
            (index - 9.5) / 10.0 for index in range(ACTION_SIZE)
        )
    actions_32 = pad_action(actions_20)

    if noise_32 is None:
        noise_32 = tuple(
            (((index * 7) % 19) - 9) / 10.0
            for index in range(PI05_ACTION_SIZE)
        )
    if len(noise_32) != PI05_ACTION_SIZE:
        raise ValueError(f"noise must have {PI05_ACTION_SIZE} values")

    target = flow_target(actions_32, noise_32)
    if prediction_32 is None:
        prediction_32 = tuple(
            value + ((index % 5) - 2) * 0.01
            for index, value in enumerate(target)
        )
    if len(prediction_32) != PI05_ACTION_SIZE:
        raise ValueError(f"prediction must have {PI05_ACTION_SIZE} values")

    per_dimension = tuple(
        (float(prediction) - expected) ** 2
        for prediction, expected in zip(prediction_32, target, strict=True)
    )
    actual_20 = sum(per_dimension[:ACTION_SIZE]) / ACTION_SIZE
    padded_12 = sum(per_dimension[ACTION_SIZE:]) / (
        PI05_ACTION_SIZE - ACTION_SIZE
    )
    full_32 = sum(per_dimension) / PI05_ACTION_SIZE

    return {
        "schema_version": 1,
        "target_formula": "noise - action",
        "action_dimensions": ACTION_SIZE,
        "padded_dimensions": PI05_ACTION_SIZE - ACTION_SIZE,
        "flow_target": list(target),
        "per_dimension_mse": list(per_dimension),
        "lerobot_native_20d_loss": actual_20,
        "openpi_style_32d_loss": full_32,
        "padded_12d_mean_mse": padded_12,
        "padded_12d_sum_contribution": sum(per_dimension[ACTION_SIZE:]),
        "loss_scale_ratio_32d_over_20d": full_32 / actual_20
        if actual_20
        else None,
        "decision": "keep_lerobot_native_20d_loss",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = loss_parity_report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"PASS: loss-parity evidence written to {args.output}")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
