#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate Task 2 LeRobot metadata before starting a PI05 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from task2_isaacsim.baselines.pi05.contract import (  # noqa: E402
    EVAL_CAMERA_KEY,
    PI05_CONTRACT,
    POLICY_CAMERA_RENAME_MAP,
    validate_dataset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check an EBiM Task 2 LeRobot v3 dataset for PI05 use."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable result.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info, errors = validate_dataset_root(args.dataset_root)
    result = {
        "ok": not errors,
        "dataset_root": str(args.dataset_root.resolve()),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "pi05": PI05_CONTRACT.to_dict(),
        "rename_map": POLICY_CAMERA_RENAME_MAP,
        "excluded_policy_inputs": [EVAL_CAMERA_KEY, "task2_extras/**"],
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif errors:
        print("FAIL: Task 2 dataset is not PI05-ready")
        for error in errors:
            print(f"  - {error}")
    else:
        print("PASS: Task 2 dataset metadata matches the PI05 pilot contract")
        print(f"  episodes={result['episodes']} frames={result['frames']}")
        print("  policy cameras=head,wrist_left,wrist_right")
        print("  excluded=eval_camera,task2_extras/**")
        print("  state=37 action=20 PI05_action_boundary=32")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
