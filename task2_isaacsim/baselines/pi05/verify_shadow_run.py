#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify a staged hard5 shadow manifest before GUI policy publication."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from task2_isaacsim.baselines.pi05.contract import (
    RELATIVE_ACTION_STATE_INDICES,
    V2_RELATIVE_ACTION_STATE_INDICES,
)


def verify_shadow_run(
    run_dir: Path,
    *,
    contract: str,
    require_right_only: bool = False,
) -> dict[str, object]:
    expected_mapping = list(
        RELATIVE_ACTION_STATE_INDICES
        if contract == "v1"
        else V2_RELATIVE_ACTION_STATE_INDICES
    )
    args_run_dir = run_dir.resolve()
    manifest_path = args_run_dir / "live_runner_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = manifest.get("events", [])
    event = events[0] if events else {}
    staging = manifest.get("staging", {})
    stage_result = staging.get("result", {})
    rmpflow_waypoints = staging.get("rmpflow_waypoints", [])
    wrist_path = args_run_dir / "settled_fresh_wrist_right.ppm"
    mapping = manifest.get("checkpoint_relative_action_state_indices", [])
    right_only_mapping = [0, 1, 2, 3, 4, 5, 6, None]
    stage_groups = stage_result.get("feedback", {}).get("groups", {})
    right_only = manifest.get("right_only_policy_after_staging") is True
    spine_control = manifest.get("spine_control", {})
    deterministic_hold = manifest.get(
        "deterministic_hold_command_action_3_20"
    )
    effective_actions = event.get("effective_actions", [])
    right_only_projection = bool(
        right_only
        and isinstance(deterministic_hold, list)
        and len(deterministic_hold) == 17
        and effective_actions
        and all(
            action[:3] == [0.0, 0.0, 0.0]
            and action[3:10] == deterministic_hold[:7]
            and action[17] == deterministic_hold[14]
            and action[19] == deterministic_hold[16]
            for action in effective_actions
        )
    )
    capture_skew = event.get("freshness", {}).get(
        "observation_capture_skew_s", float("inf")
    )
    capture_to_ready = event.get("capture_to_ready_sim_s", float("inf"))
    capture_skew_valid = isinstance(
        capture_skew, (int, float)
    ) and math.isfinite(capture_skew)
    capture_to_ready_valid = isinstance(
        capture_to_ready, (int, float)
    ) and math.isfinite(capture_to_ready)
    required_stage_groups = {
        "left_arm",
        "right_arm",
        "spine",
        "left_gripper",
        "right_gripper",
        "left_ee_height",
        "right_camera_ready_pad_relative_position",
        "right_camera_ready_orientation",
    }
    if staging.get("mode") == "rmpflow_observation":
        expected_kinds = ["continuous_observation"]
        staging_valid = (
            len(rmpflow_waypoints) == len(expected_kinds)
            and [item.get("target_kind") for item in rmpflow_waypoints]
            == expected_kinds
            and all(
                item.get("returncode") == 0
                and item.get("result", {}).get("success") is True
                and item.get("result", {}).get("reason")
                == "stable_rmpflow_observation_pose"
                and item.get("result", {}).get("control")
                == "simulator_side_rmpflow_pose_staging"
                and item.get("result", {}).get("ground_truth_subscriptions")
                == []
                for item in rmpflow_waypoints
            )
            and stage_result.get("target_kind") == "continuous_observation"
            and stage_result.get("success") is True
        )
    else:
        staging_valid = (
            stage_result.get("feedback", {}).get("within_tolerance") is True
            and set(stage_groups) == required_stage_groups
            and all(stage_groups.values())
        )
    checks = {
        "completed": manifest.get("completed") is True,
        "one_valid_decision": (
            manifest.get("valid_decisions") == 1 and len(events) == 1
            and event.get("valid") is True
        ),
        "zero_policy_command_publications": (
            manifest.get("command_publications") == 0
            and manifest.get("ros_publication") is False
        ),
        "checkpoint_mapping_matches_contract": (
            mapping == expected_mapping
            or (right_only and mapping == right_only_mapping)
        ),
        "spine_mapping_matches_contract": (
            (len(mapping) == 20
             and mapping[19] == (28 if contract == "v1" else None))
            or (
                right_only
                and mapping == right_only_mapping
                and spine_control.get("policy_controlled") is False
            )
        ),
        "staging_all_within_tolerance": staging_valid,
        "fresh_capture_skew": (
            capture_skew_valid and 0.0 <= capture_skew <= 0.10
        ),
        "sim_capture_to_ready_reported": (
            capture_to_ready_valid and capture_to_ready >= 0.0
        ),
        "hard5_indices": event.get("policy_indices") == [0, 1, 2, 3, 4],
        "right_wrist_evidence_present": (
            wrist_path.is_file() and wrist_path.stat().st_size > 16
        ),
        "right_only_policy_ownership": (
            not require_right_only
            or (
                right_only_projection
                and manifest.get("policy_owned_groups")
                == ["right_arm", "right_gripper"]
                and event.get("deterministic_hold_groups")
                == ["left_arm", "left_gripper", "spine"]
                and spine_control.get("policy_controlled") is False
            )
        ),
    }
    return {
        "valid": all(checks.values()),
        "contract": contract,
        "manifest": str(manifest_path.resolve()),
        "checkpoint": manifest.get("checkpoint"),
        "checks": checks,
        "staging_feedback": stage_result.get("feedback"),
        "rmpflow_waypoints": rmpflow_waypoints,
        "freshness": event.get("freshness"),
        "capture_to_ready_sim_s": event.get("capture_to_ready_sim_s"),
        "right_wrist_image": str(wrist_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", choices=("v1", "v2"), default="v2")
    parser.add_argument("--require-right-only", action="store_true")
    args = parser.parse_args()
    report = verify_shadow_run(
        args.run_dir,
        contract=args.contract,
        require_right_only=args.require_right_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
