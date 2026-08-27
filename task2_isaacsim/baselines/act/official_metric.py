#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Report the orientation-gated valid-placement IoU from a Task 2 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def score_result(payload: dict[str, Any]) -> dict[str, Any]:
    target_exists = payload.get("target_bbox") is not None
    pad_exists = payload.get("pad_bbox") is not None
    pick_success = bool(target_exists and pad_exists)
    orientation_success = bool(payload.get("is_orientation_correct", False))
    raw_iou = float(payload.get("iou_thermalpad_vs_target_current", 0.0))
    valid_placement_iou = float(pick_success) * float(orientation_success) * raw_iou
    return {
        "pick_success": pick_success,
        "placement_orientation_success": orientation_success,
        "placement_iou": raw_iou,
        "valid_placement_iou": valid_placement_iou,
        "orientation_case": payload.get("orientation_case"),
        "completion_time_s": None,
        "completion_time_note": (
            "The local frame evaluator does not measure the official "
            "tie-breaker time."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result", type=Path)
    group.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.result:
        result_path = args.result
    else:
        candidates = sorted(
            args.input_dir.glob("eval_camera_iou_*.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            print(f"FAIL: no eval_camera_iou_*.json under {args.input_dir}")
            return 2
        result_path = candidates[-1]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = {"evaluator_result": str(result_path), **score_result(payload)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
