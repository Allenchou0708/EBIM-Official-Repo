#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stage, shadow, or execute the camera/reference Task 2 hybrid policy.

The policy node never subscribes to evaluator or task-object ground truth.
Use ``external_audit.py`` in a different process for simulator-only audit.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, String

from task2_isaacsim.baselines.report_hybrid.core import (
    ForwardOnlyFSM,
    assert_policy_topics,
    compose_initial_relative_xyyaw,
    dark_pad_mask,
    deproject_masked_depth,
    load_reference,
    pose_distance,
    red_target_mask,
    retarget_landmarks,
    robust_center_yaw,
    wrap_angle,
)
from task2_isaacsim.common.state_contract import (
    RIGHT_GRIPPER_DRIVER,
    SPINE_JOINT,
    gripper_open_fraction,
    resolve_joint,
)
from task2_isaacsim.scripts.topics import camera_topic, load_topics

TOPICS = load_topics()


def stamp_seconds(message: Any) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def image_array(message: Image) -> np.ndarray:
    if message.encoding.lower() not in {"rgb8", "rgba8"}:
        raise ValueError(f"unsupported RGB encoding {message.encoding}")
    channels = 4 if message.encoding.lower() == "rgba8" else 3
    row = np.frombuffer(message.data, dtype=np.uint8).reshape(
        int(message.height), int(message.step)
    )
    return (
        row[:, : int(message.width) * channels]
        .reshape(int(message.height), int(message.width), channels)[..., :3]
        .copy()
    )


def depth_array(message: Image) -> np.ndarray:
    if message.encoding.upper() == "32FC1":
        dtype = np.dtype("<f4" if not message.is_bigendian else ">f4")
        row_values = int(message.step) // dtype.itemsize
        row = np.frombuffer(message.data, dtype=dtype).reshape(
            int(message.height), row_values
        )
        return row[:, : int(message.width)].astype(np.float32, copy=True)
    if message.encoding.upper() == "16UC1":
        dtype = np.dtype("<u2" if not message.is_bigendian else ">u2")
        row_values = int(message.step) // dtype.itemsize
        row = np.frombuffer(message.data, dtype=dtype).reshape(
            int(message.height), row_values
        )
        return row[:, : int(message.width)].astype(np.float32) * 0.001
    raise ValueError(f"unsupported depth encoding {message.encoding}")


class HybridNode(Node):
    def __init__(self, *, publishers: str | None):
        super().__init__("task2_report_hybrid")
        self.reference = load_reference()
        self.sim_time: float | None = None
        self.stop_requested = False
        self.odom: tuple[float, ...] | None = None
        self.odom_velocity = (math.inf, math.inf, math.inf)
        self.joints: dict[str, float] = {}
        self.ee: dict[str, tuple[float, ...]] = {}
        self.rgb: dict[str, tuple[float, np.ndarray]] = {}
        self.depth: dict[str, tuple[float, np.ndarray]] = {}
        self.intrinsics: dict[str, tuple[float, tuple[float, ...]]] = {}
        self.camera_pose: dict[str, tuple[float, tuple[float, ...]]] = {}
        self.publish_count = 0

        robot = TOPICS["cameras"]["robot"]
        policy_topics = [
            TOPICS["clock"],
            TOPICS["recording"]["odom"],
            TOPICS["recording"]["joint_states_full"],
            *TOPICS["recording"]["ee_pose"].values(),
            TOPICS["recording"]["camera_pose"]["head"],
            TOPICS["recording"]["camera_pose"]["wrist_right"],
        ]
        for key in ("head", "wrist_right"):
            namespace = robot[key]["namespace"]
            policy_topics.extend(
                camera_topic(TOPICS, namespace, kind)
                for kind in ("image", "depth", "camera_info")
            )
        self.policy_topics = assert_policy_topics(policy_topics)

        self.create_subscription(
            Clock, TOPICS["clock"], self._on_clock, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, TOPICS["recording"]["odom"], self._on_odom, 10
        )
        self.create_subscription(
            JointState,
            TOPICS["recording"]["joint_states_full"],
            self._on_joints,
            10,
        )
        for side, topic in TOPICS["recording"]["ee_pose"].items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda message, side=side: self._on_ee(side, message),
                10,
            )
        for key in ("head", "wrist_right"):
            namespace = robot[key]["namespace"]
            self.create_subscription(
                Image,
                camera_topic(TOPICS, namespace, "image"),
                lambda message, key=key: self._on_rgb(key, message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Image,
                camera_topic(TOPICS, namespace, "depth"),
                lambda message, key=key: self._on_depth(key, message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                camera_topic(TOPICS, namespace, "camera_info"),
                lambda message, key=key: self._on_info(key, message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                PoseStamped,
                TOPICS["recording"]["camera_pose"][key],
                lambda message, key=key: self._on_camera_pose(key, message),
                10,
            )

        self.command_publishers: dict[str, Any] = {}
        if publishers == "stage":
            self.command_publishers = {
                "base": self.create_publisher(
                    String, TOPICS["teleop"]["pedal_state"], 10
                ),
                "spine": self.create_publisher(
                    Float64, TOPICS["teleop"]["spine_target"], 10
                ),
                "left_ee": self.create_publisher(
                    PoseStamped,
                    TOPICS["cartesian_control"]["ee_target"]["left"],
                    10,
                ),
                "right_ee": self.create_publisher(
                    PoseStamped,
                    TOPICS["cartesian_control"]["ee_target"]["right"],
                    10,
                ),
                "left_gripper": self.create_publisher(
                    JointState,
                    TOPICS["cartesian_control"][
                        "gripper_open_fraction_target"
                    ]["left"],
                    10,
                ),
                "right_gripper": self.create_publisher(
                    JointState,
                    TOPICS["cartesian_control"][
                        "gripper_open_fraction_target"
                    ]["right"],
                    10,
                ),
            }
        elif publishers == "execute":
            self.command_publishers = {
                "left_ee": self.create_publisher(
                    PoseStamped,
                    TOPICS["cartesian_control"]["ee_target"]["left"],
                    10,
                ),
                "right_ee": self.create_publisher(
                    PoseStamped,
                    TOPICS["cartesian_control"]["ee_target"]["right"],
                    10,
                ),
                "left_gripper": self.create_publisher(
                    JointState,
                    TOPICS["cartesian_control"][
                        "gripper_open_fraction_target"
                    ]["left"],
                    10,
                ),
                "right_gripper": self.create_publisher(
                    JointState,
                    TOPICS["cartesian_control"][
                        "gripper_open_fraction_target"
                    ]["right"],
                    10,
                ),
            }

    def _on_clock(self, message: Clock) -> None:
        self.sim_time = (
            float(message.clock.sec) + float(message.clock.nanosec) * 1e-9
        )

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        yaw = math.atan2(
            2
            * (
                pose.orientation.w * pose.orientation.z
                + pose.orientation.x * pose.orientation.y
            ),
            1 - 2 * (pose.orientation.y**2 + pose.orientation.z**2),
        )
        self.odom = (pose.position.x, pose.position.y, yaw)
        twist = message.twist.twist
        self.odom_velocity = (twist.linear.x, twist.linear.y, twist.angular.z)

    def _on_joints(self, message: JointState) -> None:
        self.joints = {
            str(name): float(position)
            for name, position in zip(message.name, message.position)
        }

    def _on_ee(self, side: str, message: PoseStamped) -> None:
        p, q = message.pose.position, message.pose.orientation
        self.ee[side] = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def _on_rgb(self, key: str, message: Image) -> None:
        try:
            self.rgb[key] = (stamp_seconds(message), image_array(message))
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _on_depth(self, key: str, message: Image) -> None:
        try:
            self.depth[key] = (stamp_seconds(message), depth_array(message))
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _on_info(self, key: str, message: CameraInfo) -> None:
        self.intrinsics[key] = (
            stamp_seconds(message),
            (
                float(message.k[0]),
                float(message.k[4]),
                float(message.k[2]),
                float(message.k[5]),
            ),
        )

    def _on_camera_pose(self, key: str, message: PoseStamped) -> None:
        p, q = message.pose.position, message.pose.orientation
        self.camera_pose[key] = (
            stamp_seconds(message),
            (p.x, p.y, p.z, q.x, q.y, q.z, q.w),
        )

    def spin_until(self, predicate, timeout_wall_s: float) -> bool:
        started = time.monotonic()
        while (
            not self.stop_requested
            and time.monotonic() - started < timeout_wall_s
        ):
            rclpy.spin_once(self, timeout_sec=0.02)
            if predicate():
                return True
        return False

    def command_topics(self) -> dict[str, str]:
        return {
            "base": TOPICS["teleop"]["pedal_state"],
            "spine": TOPICS["teleop"]["spine_target"],
            "left_ee": TOPICS["cartesian_control"]["ee_target"]["left"],
            "right_ee": TOPICS["cartesian_control"]["ee_target"]["right"],
            "left_gripper": TOPICS["cartesian_control"][
                "gripper_open_fraction_target"
            ]["left"],
            "right_gripper": TOPICS["cartesian_control"][
                "gripper_open_fraction_target"
            ]["right"],
        }

    def conflicts(self) -> dict[str, int]:
        topics = self.command_topics()
        return {
            key: max(
                0,
                len(self.get_publishers_info_by_topic(topic))
                - (1 if key in self.command_publishers else 0),
            )
            for key, topic in topics.items()
        }

    def publish_pose(self, key: str, pose: tuple[float, ...]) -> None:
        message = PoseStamped()
        if self.sim_time is not None:
            message.header.stamp = rclpy.time.Time(
                nanoseconds=max(0, int(self.sim_time * 1e9))
            ).to_msg()
        message.header.frame_id = "world"
        (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ) = pose[:3]
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = pose[3:]
        self.command_publishers[key].publish(message)
        self.publish_count += 1

    def publish_gripper(self, side: str, open_fraction: float) -> None:
        message = JointState()
        if self.sim_time is not None:
            message.header.stamp = rclpy.time.Time(
                nanoseconds=max(0, int(self.sim_time * 1e9))
            ).to_msg()
        message.position = [float(open_fraction)]
        self.command_publishers[f"{side}_gripper"].publish(message)
        self.publish_count += 1

    def publish_base(self, token: str) -> None:
        self.command_publishers["base"].publish(String(data=token))
        self.publish_count += 1

    def stage(self) -> dict:
        result: dict[str, Any] = {
            "mode": "stage",
            "stages": [],
            "success": False,
        }
        if not self.spin_until(
            lambda: self.odom is not None and self.sim_time is not None, 20.0
        ):
            result["reason"] = "missing_odometry_or_clock"
            return result
        conflicts = self.conflicts()
        if any(conflicts.values()):
            result.update(reason="publisher_contention", conflicts=conflicts)
            return result
        initial = tuple(self.odom)
        target = compose_initial_relative_xyyaw(
            initial, self.reference["s0"]["initial_relative_base_xyyaw"]
        )
        if not self.spin_until(
            lambda: self.command_publishers["base"].get_subscription_count()
            == 1,
            30.0,
        ):
            result["reason"] = "base_command_subscriber_unavailable"
            return result
        started = time.monotonic()
        sim_started = self.sim_time
        base_success = False
        while (
            not self.stop_requested
            and time.monotonic() - started < 300.0
            and (
                sim_started is None
                or self.sim_time is None
                or self.sim_time - sim_started < 45.0
            )
        ):
            rclpy.spin_once(self, timeout_sec=0.03)
            if self.odom is None:
                continue
            x, y, yaw = self.odom
            dx, dy = target[0] - x, target[1] - y
            position_error = math.hypot(dx, dy)
            yaw_error = wrap_angle(target[2] - yaw)
            if position_error <= 0.035 and abs(yaw_error) <= 0.06:
                self.publish_base("NONE")
                if max(abs(v) for v in self.odom_velocity) <= 0.025:
                    base_success = True
                    break
                continue
            if position_error > 0.035:
                forward = math.cos(yaw) * dx + math.sin(yaw) * dy
                left = -math.sin(yaw) * dx + math.cos(yaw) * dy
                if abs(forward) >= abs(left):
                    token = "FWD" if forward > 0 else "BACK"
                else:
                    token = "A" if left > 0 else "B"
            else:
                token = "A+C" if yaw_error > 0 else "B+C"
            self.publish_base(token)
        self.publish_base("NONE")
        result["stages"].append(
            {
                "name": "base",
                "success": base_success,
                "initial": initial,
                "target": target,
                "final": self.odom,
                "elapsed_wall_s": time.monotonic() - started,
                "elapsed_sim_s": None
                if sim_started is None or self.sim_time is None
                else self.sim_time - sim_started,
            }
        )
        if not base_success:
            result["reason"] = "base_stage_timeout"
            return result

        spine_target = float(self.reference["s0"]["spine_command_m"])
        spine_measured = float(self.reference["s0"]["spine_measured_m"])
        started = time.monotonic()
        spine_success = False
        while not self.stop_requested and time.monotonic() - started < 45.0:
            rclpy.spin_once(self, timeout_sec=0.03)
            self.command_publishers["spine"].publish(
                Float64(data=spine_target)
            )
            self.publish_count += 1
            if (
                self.joints
                and abs(
                    resolve_joint(self.joints, SPINE_JOINT) - spine_measured
                )
                <= 0.015
            ):
                spine_success = True
                break
        result["stages"].append(
            {
                "name": "spine",
                "success": spine_success,
                "target": spine_target,
                "measured": resolve_joint(self.joints, SPINE_JOINT)
                if self.joints
                else None,
            }
        )
        if not spine_success:
            result["reason"] = "spine_stage_timeout"
            return result

        self.publish_gripper("left", 1.0)
        self.publish_gripper("right", 1.0)
        for side, reference_key in (
            ("left", "left_safe_ee_xyzw"),
            ("right", "right_observation_ee_xyzw"),
        ):
            target_pose = tuple(self.reference["s0"][reference_key])
            started = time.monotonic()
            success = False
            while (
                not self.stop_requested and time.monotonic() - started < 45.0
            ):
                rclpy.spin_once(self, timeout_sec=0.03)
                self.publish_pose(f"{side}_ee", target_pose)
                self.publish_gripper(side, 1.0)
                if side in self.ee:
                    translation, rotation = pose_distance(
                        self.ee[side], target_pose
                    )
                    if translation <= 0.025 and rotation <= 10.0:
                        success = True
                        break
            result["stages"].append(
                {
                    "name": f"{side}_safe_observe",
                    "success": success,
                    "target": target_pose,
                    "final": self.ee.get(side),
                }
            )
            if not success:
                result["reason"] = f"{side}_stage_timeout"
                return result
        result.update(
            success=True, reason="s0_settled", publications=self.publish_count
        )
        return result

    def observations_fresh(
        self, maximum_age_s: float = 0.20, maximum_skew_s: float = 0.12
    ) -> tuple[bool, dict]:
        if self.sim_time is None:
            return False, {"reason": "no_clock"}
        stamps = []
        missing = []
        for key in ("head", "wrist_right"):
            for store_name, store in (
                ("rgb", self.rgb),
                ("depth", self.depth),
                ("intrinsics", self.intrinsics),
                ("camera_pose", self.camera_pose),
            ):
                if key not in store:
                    missing.append(f"{key}.{store_name}")
                else:
                    stamps.append(float(store[key][0]))
        if missing:
            return False, {"missing": missing}
        ages = [self.sim_time - value for value in stamps]
        return (
            min(ages) >= -0.02
            and max(ages) <= maximum_age_s
            and max(stamps) - min(stamps) <= maximum_skew_s,
            {"ages_s": ages, "skew_s": max(stamps) - min(stamps)},
        )

    def perceive(self) -> dict:
        fresh, timing = self.observations_fresh()
        if not fresh:
            raise RuntimeError(f"stale_or_incomplete_observation: {timing}")
        head_mask = red_target_mask(self.rgb["head"][1])
        wrist_mask = dark_pad_mask(self.rgb["wrist_right"][1])
        if int(head_mask.sum()) < 80:
            raise RuntimeError(
                f"head_target_not_visible: pixels={int(head_mask.sum())}"
            )
        if int(wrist_mask.sum()) < 120:
            raise RuntimeError(
                f"wrist_pad_not_visible: pixels={int(wrist_mask.sum())}"
            )
        target_points = deproject_masked_depth(
            self.depth["head"][1],
            head_mask,
            self.intrinsics["head"][1],
            self.camera_pose["head"][1],
            minimum_depth_m=0.15,
            maximum_depth_m=5.0,
            stride=2,
        )
        pad_points = deproject_masked_depth(
            self.depth["wrist_right"][1],
            wrist_mask,
            self.intrinsics["wrist_right"][1],
            self.camera_pose["wrist_right"][1],
            minimum_depth_m=0.02,
            maximum_depth_m=1.0,
            stride=2,
        )
        target_bounds = np.all(
            (target_points >= np.array([2.02, 1.78, 0.70]))
            & (target_points <= np.array([2.28, 2.12, 0.82])),
            axis=1,
        )
        pad_bounds = np.all(
            (pad_points >= np.array([1.58, 1.75, 0.80]))
            & (pad_points <= np.array([1.92, 2.18, 0.93])),
            axis=1,
        )
        target_points = target_points[target_bounds]
        pad_points = pad_points[pad_bounds]
        target_center, target_yaw = robust_center_yaw(target_points)
        pad_center, pad_yaw = robust_center_yaw(pad_points)
        return {
            "timing": timing,
            "target_pixels": int(head_mask.sum()),
            "pad_pixels": int(wrist_mask.sum()),
            "target_points": int(len(target_points)),
            "pad_points": int(len(pad_points)),
            "target_center_xyz": target_center.tolist(),
            "target_yaw_rad": target_yaw,
            "pad_center_xyz": pad_center.tolist(),
            "pad_yaw_rad": pad_yaw,
        }

    def shadow(self, timeout_wall_s: float) -> dict:
        result: dict[str, Any] = {
            "schema_version": 1,
            "mode": "zero_publication_shadow",
            "policy_topics": list(self.policy_topics),
            "created_publishers": [],
            "publication_count": self.publish_count,
            "success": False,
        }
        self.spin_until(lambda: self.observations_fresh()[0], timeout_wall_s)
        result["command_publisher_counts"] = {
            key: len(self.get_publishers_info_by_topic(topic))
            for key, topic in self.command_topics().items()
        }
        try:
            perception = self.perceive()
            targets = retarget_landmarks(
                self.reference,
                perception["pad_center_xyz"],
                perception["target_center_xyz"],
                perception["pad_yaw_rad"],
                perception["target_yaw_rad"],
            )
            result.update(
                success=True,
                perception=perception,
                reachable_targets={
                    key: list(value) for key, value in targets.items()
                },
                reason="fresh_camera_fk_and_bounded_targets",
            )
        except (RuntimeError, ValueError) as error:
            result["reason"] = str(error)
        result["publication_count"] = self.publish_count
        return result

    def move_right(
        self,
        pose: tuple[float, ...],
        gripper_open: float,
        timeout_s: float = 20.0,
    ) -> bool:
        left_hold = tuple(self.reference["s0"]["left_safe_ee_xyzw"])
        started = time.monotonic()
        while (
            not self.stop_requested and time.monotonic() - started < timeout_s
        ):
            rclpy.spin_once(self, timeout_sec=0.03)
            self.publish_pose("left_ee", left_hold)
            self.publish_gripper("left", 1.0)
            self.publish_pose("right_ee", pose)
            self.publish_gripper("right", gripper_open)
            if "right" in self.ee:
                translation, rotation = pose_distance(self.ee["right"], pose)
                if translation <= 0.018 and rotation <= 8.0:
                    return True
        return False

    def execute(self) -> dict:
        result: dict[str, Any] = {
            "schema_version": 1,
            "mode": "live_execute",
            "success": False,
            "task_success": False,
            "phase_history": ["observe"],
        }
        conflicts = self.conflicts()
        if any(conflicts.values()):
            result.update(reason="publisher_contention", conflicts=conflicts)
            return result
        self.spin_until(lambda: self.observations_fresh()[0], 15.0)
        try:
            perception = self.perceive()
            targets = retarget_landmarks(
                self.reference,
                perception["pad_center_xyz"],
                perception["target_center_xyz"],
                perception["pad_yaw_rad"],
                perception["target_yaw_rad"],
            )
        except (RuntimeError, ValueError) as error:
            result["reason"] = str(error)
            return result
        result["perception"] = perception
        result["targets"] = {
            key: list(value) for key, value in targets.items()
        }
        fsm = ForwardOnlyFSM()
        schedule = (
            ("approach", ("approach",)),
            ("insert", ("insert",)),
            ("acquire", ("acquire",)),
            ("peel_lift", ("peel_lift",)),
            ("transfer_place", ("transfer", "place")),
            ("release", ("release",)),
            ("retreat", ("retreat",)),
        )
        for phase, landmark_names in schedule:
            fsm.advance(phase)
            result["phase_history"].append(phase)
            for landmark_name in landmark_names:
                if not self.move_right(
                    targets[landmark_name],
                    fsm.right_gripper_open_fraction,
                    timeout_s=25.0,
                ):
                    result.update(
                        reason=f"{landmark_name}_tracking_timeout",
                        publications=self.publish_count,
                    )
                    return result
            if phase == "acquire":
                started = time.monotonic()
                acquired = False
                while time.monotonic() - started < 4.0:
                    rclpy.spin_once(self, timeout_sec=0.03)
                    self.publish_gripper("right", 0.0)
                    if (
                        self.joints
                        and gripper_open_fraction(
                            resolve_joint(self.joints, RIGHT_GRIPPER_DRIVER)
                        )
                        <= 0.20
                    ):
                        acquired = True
                        break
                if not acquired:
                    result.update(
                        reason="gripper_did_not_close",
                        publications=self.publish_count,
                    )
                    return result
            if phase == "peel_lift":
                self.spin_until(lambda: "wrist_right" in self.rgb, 2.0)
                if (
                    "wrist_right" not in self.rgb
                    or int(dark_pad_mask(self.rgb["wrist_right"][1]).sum())
                    < 120
                ):
                    result.update(
                        reason="retained_pad_visual_gate_failed",
                        publications=self.publish_count,
                    )
                    return result
        fsm.advance("done")
        result["phase_history"].append("done")
        fsm.validate_terminal()
        result.update(
            success=True,
            reason="release_and_retreat_complete_external_evaluation_required",
            publications=self.publish_count,
            close_transitions=fsm.close_count,
            open_transitions=fsm.open_count,
            left_gripper_open=True,
            base_spine_locked_after_acquire=True,
        )
        return result


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("stage", "shadow", "execute"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shadow-timeout-s", type=float, default=15.0)
    args = parser.parse_args()
    rclpy.init()
    node = HybridNode(publishers=None if args.mode == "shadow" else args.mode)
    signal.signal(
        signal.SIGINT, lambda *_: setattr(node, "stop_requested", True)
    )
    signal.signal(
        signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True)
    )
    try:
        if args.mode == "stage":
            result = node.stage()
        elif args.mode == "shadow":
            result = node.shadow(args.shadow_timeout_s)
        else:
            result = node.execute()
    finally:
        if args.mode == "stage" and "base" in node.command_publishers:
            node.publish_base("NONE")
        node.destroy_node()
        rclpy.shutdown()
    write_result(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
