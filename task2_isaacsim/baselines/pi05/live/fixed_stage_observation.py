#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stage collision-free observation poses through simulator-side RMPflow.

The targets are fixed, development-split robust references.  This process
never subscribes to task objects, evaluator output, or other ground truth.
It owns both end-effector pose targets only for staging and exits after a
measured settle/jitter gate so PI0.5 can take over the right arm cleanly.
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
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from task2_isaacsim.baselines.pi05.live.ground_truth_pregrasp import (
    _orientation_error_deg,
    _position_error,
    _slerp,
    _stamp_seconds,
)
from task2_isaacsim.common.state_contract import (
    LEFT_GRIPPER_DRIVER,
    LEFT_JOINTS,
    RIGHT_GRIPPER_DRIVER,
    RIGHT_JOINTS,
    SPINE_JOINT,
    gripper_open_fraction,
    resolve_joint,
)
from task2_isaacsim.scripts.topics import load_topics

TOPICS = load_topics()
EE_TARGET_TOPICS = TOPICS["cartesian_control"]["ee_target"]
GRIPPER_TARGET_TOPICS = TOPICS["cartesian_control"][
    "gripper_open_fraction_target"
]
SPINE_TARGET_TOPIC = TOPICS["teleop"]["spine_target"]
ARM_JOINTS = (*LEFT_JOINTS, *RIGHT_JOINTS)
RMPFLOW_STAGE_PLAN = (
    ("continuous_observation", 30.0),
)
MINIMUM_JERK_PEAK_DERIVATIVE = 1.875


def minimum_jerk_fraction(fraction: float) -> float:
    """C2-continuous 0→1 interpolation with zero endpoint velocity."""
    value = max(0.0, min(1.0, float(fraction)))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def transition_duration_s(
    initial: dict[str, tuple[float, ...]],
    targets: dict[str, tuple[float, ...]],
    *,
    max_linear_speed_m_s: float,
    max_angular_speed_deg_s: float,
    minimum_s: float,
    maximum_s: float,
) -> float:
    if max_linear_speed_m_s <= 0.0 or max_angular_speed_deg_s <= 0.0:
        raise ValueError("transition speeds must be positive")
    required = max(
        max(
            MINIMUM_JERK_PEAK_DERIVATIVE
            * _position_error(initial[side], targets[side])
            / max_linear_speed_m_s,
            MINIMUM_JERK_PEAK_DERIVATIVE
            * _orientation_error_deg(initial[side], targets[side])
            / max_angular_speed_deg_s,
        )
        for side in ("left", "right")
    )
    return max(minimum_s, min(maximum_s, required))


def continuous_landmark_pose(
    poses: list[tuple[float, ...]], fraction: float
) -> tuple[float, ...]:
    """Interpolate a chord-parameterized C1 path without waypoint stops."""

    if len(poses) < 2:
        raise ValueError("continuous pose path requires at least two poses")
    values = [np.asarray(pose, dtype=np.float64) for pose in poses]
    points = np.asarray([value[:3] for value in values], dtype=np.float64)
    chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(chords <= 1.0e-9):
        raise ValueError("continuous pose path has duplicate xyz landmarks")
    distances = np.concatenate(([0.0], np.cumsum(chords)))
    distance = max(0.0, min(1.0, float(fraction))) * float(distances[-1])
    segment = min(
        max(int(np.searchsorted(distances, distance, side="right") - 1), 0),
        len(values) - 2,
    )
    local = (distance - distances[segment]) / chords[segment]
    tangents = np.empty_like(points)
    tangents[0] = (points[1] - points[0]) / chords[0]
    tangents[-1] = (points[-1] - points[-2]) / chords[-1]
    for index in range(1, len(points) - 1):
        tangents[index] = (
            (points[index + 1] - points[index - 1])
            / (distances[index + 1] - distances[index - 1])
        )
    local2 = local * local
    local3 = local2 * local
    position = (
        (2.0 * local3 - 3.0 * local2 + 1.0) * points[segment]
        + (local3 - 2.0 * local2 + local)
        * chords[segment]
        * tangents[segment]
        + (-2.0 * local3 + 3.0 * local2) * points[segment + 1]
        + (local3 - local2) * chords[segment] * tangents[segment + 1]
    )
    orientation = _slerp(
        values[segment][3:7], values[segment + 1][3:7], float(local)
    )
    return (*position.tolist(), *orientation)


def continuous_path_duration_s(
    poses: list[tuple[float, ...]],
    *,
    max_linear_speed_m_s: float,
    max_angular_speed_deg_s: float,
    minimum_s: float,
    maximum_s: float,
) -> float:
    points = np.asarray([pose[:3] for pose in poses], dtype=np.float64)
    path_m = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    path_deg = sum(
        _orientation_error_deg(first, second)
        for first, second in zip(poses, poses[1:])
    )
    required = MINIMUM_JERK_PEAK_DERIVATIVE * max(
        path_m / max_linear_speed_m_s,
        path_deg / max_angular_speed_deg_s,
    )
    return max(minimum_s, min(maximum_s, required))


def load_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported observation reference schema")
    if payload.get("source", {}).get("split") != "development_only":
        raise ValueError("observation reference must be fit on development only")
    if int(payload["source"]["support_unique_episodes"]) < 20:
        raise ValueError("observation reference has insufficient episode support")
    for key, length in (
        ("base_xyyaw", 3),
        ("left_safe_ee_world_xyzw", 7),
        ("right_safe_orientation_waypoint_ee_world_xyzw", 7),
        ("right_orientation_midpoint_waypoint_ee_world_xyzw", 7),
        ("right_clearance_waypoint_ee_world_xyzw", 7),
        ("right_observation_ee_world_xyzw", 7),
    ):
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"invalid {key}")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite {key}")
    if not math.isfinite(float(payload["spine_command_m"])):
        raise ValueError("invalid spine command")
    return payload


class ObservationStager(Node):
    def __init__(self) -> None:
        super().__init__("pi05_rmpflow_observation_stager")
        self.sim_time: float | None = None
        self.base: tuple[float, float, float] | None = None
        self.base_time: float | None = None
        self.ee: dict[str, tuple[float, ...]] = {}
        self.ee_time: dict[str, float] = {}
        self.joints: dict[str, float] = {}
        self.joints_time: float | None = None
        self.stop_requested = False
        self.publish_count = 0
        self.pose_publishers = {
            side: self.create_publisher(PoseStamped, topic, 10)
            for side, topic in EE_TARGET_TOPICS.items()
        }
        self.gripper_publishers = {
            side: self.create_publisher(JointState, topic, 10)
            for side, topic in GRIPPER_TARGET_TOPICS.items()
        }
        self.spine_publisher = self.create_publisher(
            Float64, SPINE_TARGET_TOPIC, 10
        )
        self._owned_subscriptions = [
            self.create_subscription(
                Clock, TOPICS["clock"], self._on_clock, qos_profile_sensor_data
            ),
            self.create_subscription(
                Odometry,
                TOPICS["recording"]["odom"],
                self._on_odom,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                JointState,
                TOPICS["recording"]["joint_states_full"],
                self._on_joints,
                qos_profile_sensor_data,
            ),
        ]
        for side, topic in TOPICS["recording"]["ee_pose"].items():
            self._owned_subscriptions.append(
                self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda message, side=side: self._on_ee(side, message),
                    qos_profile_sensor_data,
                )
            )

    def _on_clock(self, message: Clock) -> None:
        self.sim_time = _stamp_seconds(message.clock)

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        yaw = math.atan2(
            2.0
            * (
                pose.orientation.w * pose.orientation.z
                + pose.orientation.x * pose.orientation.y
            ),
            1.0
            - 2.0
            * (
                pose.orientation.y * pose.orientation.y
                + pose.orientation.z * pose.orientation.z
            ),
        )
        self.base = (pose.position.x, pose.position.y, yaw)
        self.base_time = _stamp_seconds(message.header.stamp)

    def _on_ee(self, side: str, message: PoseStamped) -> None:
        pose = message.pose
        self.ee[side] = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.ee_time[side] = _stamp_seconds(message.header.stamp)

    def _on_joints(self, message: JointState) -> None:
        self.joints = {
            str(name): float(value)
            for name, value in zip(message.name, message.position)
        }
        self.joints_time = _stamp_seconds(message.header.stamp)

    def fresh(self, maximum_skew_s: float) -> bool:
        if self.sim_time is None:
            return False
        times = [
            self.base_time,
            self.joints_time,
            *(self.ee_time.get(side) for side in ("left", "right")),
        ]
        names = (*ARM_JOINTS, LEFT_GRIPPER_DRIVER, RIGHT_GRIPPER_DRIVER, SPINE_JOINT)
        return (
            all(
                stamp is not None
                and 0.0 <= self.sim_time - stamp <= maximum_skew_s
                for stamp in times
            )
            and all(name in self.joints for name in names)
        )

    def subscribers_ready(self) -> bool:
        publishers = (
            *self.pose_publishers.values(),
            *self.gripper_publishers.values(),
            self.spine_publisher,
        )
        return all(publisher.get_subscription_count() >= 1 for publisher in publishers)

    def conflicts(self) -> dict[str, int]:
        topics = {
            **{f"pose_{side}": topic for side, topic in EE_TARGET_TOPICS.items()},
            **{
                f"gripper_{side}": topic
                for side, topic in GRIPPER_TARGET_TOPICS.items()
            },
            "spine": SPINE_TARGET_TOPIC,
        }
        return {
            key: max(0, len(self.get_publishers_info_by_topic(topic)) - 1)
            for key, topic in topics.items()
        }

    def arm_positions(self) -> tuple[float, ...]:
        return tuple(resolve_joint(self.joints, name) for name in ARM_JOINTS)

    def publish(
        self,
        targets: dict[str, tuple[float, ...]],
        spine_command_m: float,
        gripper_open_fractions: dict[str, float] | None = None,
    ) -> None:
        assert self.sim_time is not None
        stamp = Time(nanoseconds=max(0, int(self.sim_time * 1.0e9))).to_msg()
        for side, target in targets.items():
            message = PoseStamped()
            message.header.stamp = stamp
            message.header.frame_id = "world"
            message.pose.position.x, message.pose.position.y, message.pose.position.z = target[:3]
            (
                message.pose.orientation.x,
                message.pose.orientation.y,
                message.pose.orientation.z,
                message.pose.orientation.w,
            ) = target[3:7]
            self.pose_publishers[side].publish(message)
        requested_grippers = gripper_open_fractions or {
            "left": 1.0,
            "right": 1.0,
        }
        for side, publisher in self.gripper_publishers.items():
            message = JointState()
            message.header.stamp = stamp
            message.name = [f"{side}_open_fraction"]
            message.position = [float(requested_grippers[side])]
            publisher.publish(message)
        self.spine_publisher.publish(Float64(data=spine_command_m))
        self.publish_count += 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-kind",
        choices=(
            "safe_orientation",
            "orientation_midpoint",
            "clearance",
            "observation",
            "continuous_observation",
        ),
        default="observation",
    )
    parser.add_argument("--maximum-skew-s", type=float, default=0.10)
    parser.add_argument("--max-duration-s", type=float, default=90.0)
    parser.add_argument("--max-linear-speed-m-s", type=float, default=0.05)
    parser.add_argument("--max-angular-speed-deg-s", type=float, default=20.0)
    parser.add_argument("--minimum-transition-s", type=float, default=6.0)
    parser.add_argument("--maximum-transition-s", type=float, default=18.0)
    parser.add_argument("--stable-dwell-s", type=float, default=1.5)
    parser.add_argument("--position-tolerance-m", type=float, default=0.025)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=8.0)
    parser.add_argument("--base-position-tolerance-m", type=float, default=0.03)
    parser.add_argument("--base-yaw-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--spine-tolerance-m", type=float, default=0.02)
    parser.add_argument("--gripper-min-open-fraction", type=float, default=0.95)
    parser.add_argument("--settle-max-joint-speed-rad-s", type=float, default=0.08)
    parser.add_argument("--settle-max-ee-drift-m", type=float, default=0.005)
    return parser


def main() -> int:  # noqa: C901 - bounded live controller and evidence loop
    args = build_parser().parse_args()
    reference = load_reference(args.reference)
    right_target_key = {
        "safe_orientation": "right_safe_orientation_waypoint_ee_world_xyzw",
        "orientation_midpoint": (
            "right_orientation_midpoint_waypoint_ee_world_xyzw"
        ),
        "clearance": "right_clearance_waypoint_ee_world_xyzw",
        "observation": "right_observation_ee_world_xyzw",
        "continuous_observation": "right_observation_ee_world_xyzw",
    }[args.target_kind]
    targets = {
        "left": tuple(float(v) for v in reference["left_safe_ee_world_xyzw"]),
        "right": tuple(
            float(v) for v in reference[right_target_key]
        ),
    }
    base_target = tuple(float(v) for v in reference["base_xyyaw"])
    spine_target = float(reference["spine_command_m"])
    rclpy.init()
    node = ObservationStager()
    signal.signal(signal.SIGINT, lambda *_: setattr(node, "stop_requested", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True))
    wall_started = time.monotonic()
    last_sim_time: float | None = None
    transition_started: float | None = None
    initial: dict[str, tuple[float, ...]] | None = None
    duration_s: float | None = None
    stable_since: float | None = None
    stable_origin: dict[str, tuple[float, ...]] = {}
    right_pose_path: list[tuple[float, ...]] | None = None
    previous_joint_sample: tuple[float, tuple[float, ...]] | None = None
    transition_max_joint_speed = 0.0
    settle_max_joint_speed = 0.0
    settle_max_ee_drift = 0.0
    errors: dict[str, dict[str, float]] = {}
    base_errors: dict[str, float] = {}
    reason = "host_watchdog_timeout"
    try:
        while not node.stop_requested and time.monotonic() - wall_started < args.max_duration_s:
            rclpy.spin_once(node, timeout_sec=0.02)
            if not node.fresh(args.maximum_skew_s) or not node.subscribers_ready():
                continue
            conflicts = node.conflicts()
            if any(conflicts.values()):
                reason = f"publisher_contention:{conflicts}"
                break
            assert node.sim_time is not None and node.base is not None
            if last_sim_time is not None and node.sim_time < last_sim_time:
                reason = "simulator_clock_reset"
                break
            if last_sim_time is not None and node.sim_time == last_sim_time:
                continue
            last_sim_time = node.sim_time
            positions = node.arm_positions()
            sample_speed = 0.0
            if previous_joint_sample is not None:
                dt = node.sim_time - previous_joint_sample[0]
                if dt > 1.0e-6:
                    sample_speed = max(
                        abs(a - b) / dt
                        for a, b in zip(positions, previous_joint_sample[1], strict=True)
                    )
            previous_joint_sample = (node.sim_time, positions)
            if initial is None:
                dx = node.base[0] - base_target[0]
                dy = node.base[1] - base_target[1]
                dyaw = math.atan2(
                    math.sin(node.base[2] - base_target[2]),
                    math.cos(node.base[2] - base_target[2]),
                )
                base_errors = {
                    "position_m": math.hypot(dx, dy),
                    "yaw_rad": abs(dyaw),
                }
                if base_errors["position_m"] > args.base_position_tolerance_m:
                    reason = f"base_position_gate:{base_errors['position_m']:.6f}"
                    break
                if base_errors["yaw_rad"] > args.base_yaw_tolerance_rad:
                    reason = f"base_yaw_gate:{base_errors['yaw_rad']:.6f}"
                    break
                spine = resolve_joint(node.joints, SPINE_JOINT)
                if abs(spine - spine_target) > args.spine_tolerance_m:
                    reason = f"spine_gate:{spine:.6f}"
                    break
                open_fractions = {
                    "left": gripper_open_fraction(
                        resolve_joint(node.joints, LEFT_GRIPPER_DRIVER)
                    ),
                    "right": gripper_open_fraction(
                        resolve_joint(node.joints, RIGHT_GRIPPER_DRIVER)
                    ),
                }
                if min(open_fractions.values()) < args.gripper_min_open_fraction:
                    reason = f"gripper_open_gate:{open_fractions}"
                    break
                initial = dict(node.ee)
                duration_s = transition_duration_s(
                    initial,
                    targets,
                    max_linear_speed_m_s=args.max_linear_speed_m_s,
                    max_angular_speed_deg_s=args.max_angular_speed_deg_s,
                    minimum_s=args.minimum_transition_s,
                    maximum_s=args.maximum_transition_s,
                )
                if args.target_kind == "continuous_observation":
                    right_pose_path = [
                        initial["right"],
                        tuple(
                            float(value) for value in reference[
                                "right_safe_orientation_waypoint_ee_world_xyzw"
                            ]
                        ),
                        tuple(
                            float(value) for value in reference[
                                "right_clearance_waypoint_ee_world_xyzw"
                            ]
                        ),
                        targets["right"],
                    ]
                    duration_s = continuous_path_duration_s(
                        right_pose_path,
                        max_linear_speed_m_s=args.max_linear_speed_m_s,
                        max_angular_speed_deg_s=args.max_angular_speed_deg_s,
                        minimum_s=args.minimum_transition_s,
                        maximum_s=args.maximum_transition_s,
                    )
                transition_started = node.sim_time
            assert initial is not None and transition_started is not None
            assert duration_s is not None
            raw_fraction = min(1.0, max(0.0, (node.sim_time - transition_started) / duration_s))
            fraction = minimum_jerk_fraction(raw_fraction)
            commanded = {
                side: (
                    *(
                        initial[side][index]
                        + fraction * (targets[side][index] - initial[side][index])
                        for index in range(3)
                    ),
                    *_slerp(initial[side][3:7], targets[side][3:7], fraction),
                )
                for side in ("left", "right")
            }
            if right_pose_path is not None:
                commanded["right"] = continuous_landmark_pose(
                    right_pose_path, fraction
                )
            node.publish(commanded, spine_target)
            errors = {
                side: {
                    "position_m": _position_error(node.ee[side], targets[side]),
                    "orientation_deg": _orientation_error_deg(node.ee[side], targets[side]),
                }
                for side in ("left", "right")
            }
            if raw_fraction < 1.0:
                transition_max_joint_speed = max(transition_max_joint_speed, sample_speed)
                stable_since = None
                stable_origin.clear()
                continue
            within_pose = all(
                value["position_m"] <= args.position_tolerance_m
                and value["orientation_deg"] <= args.orientation_tolerance_deg
                for value in errors.values()
            )
            if not within_pose or sample_speed > args.settle_max_joint_speed_rad_s:
                stable_since = None
                stable_origin.clear()
                settle_max_joint_speed = max(settle_max_joint_speed, sample_speed)
                continue
            if stable_since is None:
                stable_since = node.sim_time
                stable_origin = dict(node.ee)
                settle_max_joint_speed = sample_speed
                settle_max_ee_drift = 0.0
            else:
                settle_max_joint_speed = max(settle_max_joint_speed, sample_speed)
                settle_max_ee_drift = max(
                    settle_max_ee_drift,
                    *(
                        _position_error(node.ee[side], stable_origin[side])
                        for side in ("left", "right")
                    ),
                )
                if settle_max_ee_drift > args.settle_max_ee_drift_m:
                    stable_since = None
                    stable_origin.clear()
                    continue
                if node.sim_time - stable_since >= args.stable_dwell_s:
                    reason = "stable_rmpflow_observation_pose"
                    break
    finally:
        success = reason == "stable_rmpflow_observation_pose"
        result = {
            "schema_version": 1,
            "success": success,
            "reason": reason,
            "control": "simulator_side_rmpflow_pose_staging",
            "target_kind": args.target_kind,
            "ground_truth_subscriptions": [],
            "reference": reference,
            "initial_ee_world": initial,
            "target_ee_world": targets,
            "final_ee_world": node.ee,
            "final_hold_command_action_3_20": (
                [
                    *(resolve_joint(node.joints, name) for name in LEFT_JOINTS),
                    *(resolve_joint(node.joints, name) for name in RIGHT_JOINTS),
                    gripper_open_fraction(
                        resolve_joint(node.joints, LEFT_GRIPPER_DRIVER)
                    ),
                    gripper_open_fraction(
                        resolve_joint(node.joints, RIGHT_GRIPPER_DRIVER)
                    ),
                    spine_target,
                ]
                if success
                else None
            ),
            "final_errors": errors,
            "base_errors": base_errors,
            "transition_duration_s": duration_s,
            "continuous_right_path_landmarks": (
                ["initial", "safe_orientation", "clearance", "observation"]
                if right_pose_path is not None
                else None
            ),
            "transition_max_measured_joint_speed_rad_s": transition_max_joint_speed,
            "settle_max_measured_joint_speed_rad_s": settle_max_joint_speed,
            "settle_max_ee_drift_m": settle_max_ee_drift,
            "command_publications": node.publish_count,
            "tolerances": {
                "position_m": args.position_tolerance_m,
                "orientation_deg": args.orientation_tolerance_deg,
                "settle_max_joint_speed_rad_s": args.settle_max_joint_speed_rad_s,
                "settle_max_ee_drift_m": args.settle_max_ee_drift_m,
                "stable_dwell_s": args.stable_dwell_s,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
