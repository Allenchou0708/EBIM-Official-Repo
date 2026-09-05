from __future__ import annotations

import json
import unittest
from pathlib import Path

from task2_real.runtime_core import (
    ActionSafetyError,
    POLICY_COMMAND_TOPICS,
    action_limits,
    evaluate_handoff,
    safe_right_action_window,
    validate_site_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def verified_profile() -> dict:
    profile = json.loads((ROOT / "site_profile_munich.template.json").read_text())
    profile["status"] = "verified_test_fixture"
    profile["calibration"]["base"].update(
        {
            "verified": True,
            "relative_waypoints": [[-0.5, 0.0, 0.0], [-0.5, 0.4, 0.0]],
            "target_relative": [-0.5, 0.4, 0.0],
        }
    )
    profile["calibration"]["spine"].update(
        {"verified": True, "unit": "controller_verified_unit", "tolerance": 2.0}
    )
    profile["calibration"]["staging"]["collision_checked"] = True
    profile["safety"].update(
        {
            "right_joint_lower_rad": [-3.0] * 7,
            "right_joint_upper_rad": [3.0] * 7,
            "external_joint_torque_max_abs_nm": 5.0,
            "force_norm_max_n": 15.0,
            "torque_norm_max_nm": 3.0,
        }
    )
    return profile


def ready_snapshot(profile: dict, now: float = 10.0) -> dict:
    staging = profile["calibration"]["staging"]
    return {
        "base_position_error_m": 0.01,
        "base_yaw_error_rad": 0.01,
        "base_linear_speed_mps": 0.0,
        "base_angular_speed_rps": 0.0,
        "base_settled_for_s": 1.2,
        "spine_position": 434.0,
        "spine_velocity": 0.0,
        "left_joint_positions_rad": staging["left_safe_hold_joint_rad"],
        "right_joint_positions_rad": staging["right_observation_joint_rad"],
        "left_gripper_open_fraction": 1.0,
        "right_gripper_open_fraction": 1.0,
        "post_settle_observation": True,
        "capture_times_s": {
            "head": now - 0.04,
            "wrist_right": now - 0.02,
            "right_joints": now - 0.01,
            "right_gripper": now - 0.01,
        },
        "publisher_counts": {
            "base": 1,
            "spine": 1,
            "left_arm": 1,
            "left_gripper": 1,
            "right_arm": 0,
            "right_gripper": 0,
        },
        "right_external_joint_torques_nm": [0.0] * 7,
        "right_external_force_n": [0.0] * 3,
        "right_external_torque_nm": [0.0] * 3,
    }


class RuntimeCoreTest(unittest.TestCase):
    def test_template_is_intentionally_not_armable(self) -> None:
        profile = json.loads(
            (ROOT / "site_profile_munich.template.json").read_text()
        )
        validated = validate_site_profile(profile, require_verified=False)
        self.assertIn("base", validated["calibration_blockers"])
        self.assertIn("spine", validated["calibration_blockers"])
        with self.assertRaisesRegex(ValueError, "unverified site calibration"):
            validate_site_profile(profile, require_verified=True)

    def test_handoff_passes_only_for_fresh_settled_tuple(self) -> None:
        profile = verified_profile()
        result = evaluate_handoff(profile, ready_snapshot(profile), now=10.0)
        self.assertTrue(result["ready"])
        self.assertEqual(
            set(result["policy_command_topics"]),
            {"right_arm", "right_gripper"},
        )

    def test_handoff_rejects_stale_force_and_contention(self) -> None:
        profile = verified_profile()
        snapshot = ready_snapshot(profile)
        snapshot["capture_times_s"]["head"] = 8.0
        snapshot["right_external_force_n"] = [20.0, 0.0, 0.0]
        snapshot["publisher_counts"]["right_arm"] = 1
        result = evaluate_handoff(profile, snapshot, now=10.0)
        self.assertFalse(result["ready"])
        self.assertTrue(
            {
                "camera_freshness",
                "external_force_limit",
                "publisher_contention",
            }
            <= set(result["reasons"])
        )

    def test_handoff_rejects_future_timestamp_clock_mismatch(self) -> None:
        profile = verified_profile()
        snapshot = ready_snapshot(profile)
        snapshot["capture_times_s"]["wrist_right"] = 10.01
        result = evaluate_handoff(profile, snapshot, now=10.0)
        self.assertFalse(result["ready"])
        self.assertIn("camera_freshness", result["reasons"])

    def test_safe_action_window_is_right_only_and_hard5(self) -> None:
        profile = verified_profile()
        limits = action_limits(profile)
        measured = [0.0] * 7
        actions = [[0.01 * index] * 7 + [1.0] for index in range(1, 8)]
        safe = safe_right_action_window(
            actions,
            measured_joints_rad=measured,
            created_at=9.9,
            now=10.0,
            limits=limits,
        )
        self.assertEqual(len(safe), 5)
        contract = json.loads((ROOT / "contract.json").read_text())
        self.assertEqual(
            POLICY_COMMAND_TOPICS,
            {
                "right_arm": contract["ros2_topics"]["right_arm_command"][0],
                "right_gripper": contract["ros2_topics"]["right_gripper_command"][0],
            },
        )

    def test_action_rejects_large_delta_or_stale_window(self) -> None:
        profile = verified_profile()
        limits = action_limits(profile)
        with self.assertRaisesRegex(ActionSafetyError, "step exceeds"):
            safe_right_action_window(
                [[0.2] * 7 + [1.0]],
                measured_joints_rad=[0.0] * 7,
                created_at=9.9,
                now=10.0,
                limits=limits,
            )
        with self.assertRaisesRegex(ActionSafetyError, "stale"):
            safe_right_action_window(
                [[0.01] * 7 + [1.0]],
                measured_joints_rad=[0.0] * 7,
                created_at=9.0,
                now=10.0,
                limits=limits,
            )


if __name__ == "__main__":
    unittest.main()
