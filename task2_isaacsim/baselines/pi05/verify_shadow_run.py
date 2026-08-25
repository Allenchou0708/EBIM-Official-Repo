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


def verify_shadow_run(run_dir: Path, *, contract: str) -> dict[str, object]:
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
    wrist_path = args_run_dir / "settled_fresh_wrist_right.ppm"
    mapping = manifest.get("checkpoint_relative_action_state_indices", [])
    stage_groups = stage_result.get("feedback", {}).get("groups", {})
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
    ownership = manifest.get("ownership_handoff", {})
    hybrid = ownership.get("mode") == "gt_pregrasp_to_pi05"
    hybrid_stage_valid = (
        stage_result.get("success") is True
        and stage_result.get("reason")
        == "stable_dataset_ground_truth_pregrasp"
        and stage_result.get("provenance", {}).get("controller")
        == "formal_phase1_ground_truth_joint_lift"
        and stage_result.get("provenance", {}).get("dataset_frame") == 399
        and stage_result.get("provenance", {}).get("absolute_dataset_joints")
        is True
        and stage_result.get("provenance", {}).get("guessed_ik_used") is False
        and stage_result.get("provenance", {}).get("staged_groups")
        == [
            "base",
            "spine",
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
        ]
        and stage_result.get("final_left_preposition_max_joint_error_rad", 1.0)
        <= 0.02
        and stage_result.get("final_right_preposition_max_joint_error_rad", 1.0)
        <= 0.02
        and ownership.get("cartesian_publishers_before_policy")
        == {"left": 0, "right": 0}
        and ownership.get("policy_publishers_before_activation")
        == {
            "left_arm": 0,
            "right_arm": 0,
            "left_gripper": 0,
            "right_gripper": 0,
            "spine": 0,
        }
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
        "checkpoint_mapping_matches_contract": mapping == expected_mapping,
        "spine_mapping_matches_contract": (
            len(mapping) == 20
            and mapping[19] == (28 if contract == "v1" else None)
        ),
        "staging_all_within_tolerance": (
            hybrid_stage_valid
            if hybrid
            else (
                stage_result.get("feedback", {}).get("within_tolerance") is True
                and set(stage_groups) == required_stage_groups
                and all(stage_groups.values())
            )
        ),
        "ownership_handoff_timestamps": (
            not hybrid
            or (
                ownership.get("gt_handoff_sim_time") is not None
                and ownership.get("first_post_handoff_observation_sim_time")
                is not None
                and ownership.get("first_policy_decision_sim_time") is not None
            )
        ),
        "full_dual_arm_handoff_state": (
            not hybrid
            or (
                ownership.get("post_handoff_arm_max_error_rad", 1.0) <= 0.03
                and ownership.get("post_handoff_grippers_open") is True
            )
        ),
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
    }
    return {
        "valid": all(checks.values()),
        "contract": contract,
        "manifest": str(manifest_path.resolve()),
        "checkpoint": manifest.get("checkpoint"),
        "checks": checks,
        "staging_feedback": stage_result.get("feedback"),
        "freshness": event.get("freshness"),
        "capture_to_ready_sim_s": event.get("capture_to_ready_sim_s"),
        "right_wrist_image": str(wrist_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    report = verify_shadow_run(args.run_dir, contract=args.contract)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
