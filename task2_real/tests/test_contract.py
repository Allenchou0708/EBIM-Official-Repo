from __future__ import annotations

import json
import unittest
from pathlib import Path

from task2_real.contract import audit_official_metadata, validate_contract


ROOT = Path(__file__).resolve().parents[1]


class RealRobotContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads((ROOT / "contract.json").read_text())

    def test_right_only_policy_has_disjoint_ownership(self) -> None:
        validated = validate_contract(self.contract)
        self.assertEqual(validated["ownership"]["pi05"], ["right_arm", "right_gripper"])
        self.assertEqual(validated["policy_view"]["state"]["size"], 8)
        self.assertEqual(validated["policy_view"]["action"]["size"], 8)

    def test_released_metadata_conflicts_are_explicit(self) -> None:
        info = {
            "features": {
                "observation.state": {"shape": [42], "names": ["right_joint"]},
                "action": {"shape": [17], "names": ["right_joint_target", "spine_target_height"]},
            }
        }
        modality = {
            "state": {"right_arm_external_wrench": {"start": 50, "end": 56}},
            "action": {"right_gripper_position": {"start": 15, "end": 16}},
        }
        self.assertEqual(
            audit_official_metadata(info, modality),
            self.contract["dataset"]["documented_metadata_conflicts"],
        )

    def test_exact_ros_topics_are_locked(self) -> None:
        topics = validate_contract(self.contract)["ros2_topics"]
        self.assertEqual(topics["base_command"][0], "/swerve_drive_controller/cmd_vel")
        self.assertEqual(topics["spine_command"][0], "/spine/target_height")
        self.assertEqual(topics["right_arm_command"][0], "/right/gello/joint_states")
        self.assertEqual(
            topics["right_wrist_rgb"][0],
            "/wrist_camera_right/camera/color/image_raw",
        )
        self.assertEqual(
            topics["right_external_joint_torques"],
            [
                "/right/franka_robot_state_broadcaster/external_joint_torques",
                "sensor_msgs/msg/JointState",
            ],
        )
        self.assertEqual(
            topics["right_external_wrench"],
            [
                "/right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
                "geometry_msgs/msg/WrenchStamped",
            ],
        )


if __name__ == "__main__":
    unittest.main()
