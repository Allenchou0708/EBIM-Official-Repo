#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stage audited Task 2 camera-ready EE poses from live ground truth."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from task2_isaacsim.common.state_contract import (
    LEFT_GRIPPER_DRIVER,
    RIGHT_GRIPPER_DRIVER,
    SPINE_JOINT,
    gripper_open_fraction,
    resolve_joint,
)
from task2_isaacsim.scripts.topics import load_topics

TOPICS = load_topics()
EE_TARGET_TOPICS = TOPICS["cartesian_control"]["ee_target"]


def _stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _normalize_quaternion(quaternion: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1.0e-9 or not math.isfinite(norm):
        raise ValueError("invalid zero or non-finite quaternion")
    return tuple(value / norm for value in quaternion)


def _quaternion_multiply_xyzw(
    first: tuple[float, ...], second: tuple[float, ...]
) -> tuple[float, ...]:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return _normalize_quaternion(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def _yaw_from_wxyz(quaternion: tuple[float, ...]) -> float:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1.0e-9 or not math.isfinite(norm):
        raise ValueError("invalid zero or non-finite quaternion")
    qw, qx, qy, qz = (value / norm for value in quaternion)
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _rotate_z(vector: tuple[float, ...], angle: float) -> tuple[float, ...]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
        vector[2],
    )


def _slerp(
    start: tuple[float, ...], target: tuple[float, ...], fraction: float
) -> tuple[float, ...]:
    start = _normalize_quaternion(start)
    target = _normalize_quaternion(target)
    dot = sum(a * b for a, b in zip(start, target, strict=True))
    if dot < 0.0:
        target = tuple(-value for value in target)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(
            tuple(
                a + fraction * (b - a)
                for a, b in zip(start, target, strict=True)
            )
        )
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return tuple(
        (
            math.sin((1.0 - fraction) * angle) * a
            + math.sin(fraction * angle) * b
        )
        / denominator
        for a, b in zip(start, target, strict=True)
    )


def _position_error(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(first[:3], second[:3], strict=True))
    )


def _orientation_error_deg(
    first: tuple[float, ...], second: tuple[float, ...]
) -> float:
    first_q = _normalize_quaternion(first[3:7])
    second_q = _normalize_quaternion(second[3:7])
    dot = abs(sum(a * b for a, b in zip(first_q, second_q, strict=True)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


class GroundTruthPregraspNode(Node):
    def __init__(self) -> None:
        super().__init__("task2_ground_truth_pregrasp")
        self.sim_time: float | None = None
        self.ee: dict[str, tuple[float, ...]] = {}
        self.ee_time: dict[str, float] = {}
        self.objects: dict[str, tuple[float, ...]] = {}
        self.objects_time: float | None = None
        self.base: tuple[float, float, float] | None = None
        self.base_time: float | None = None
        self.joints: dict[str, float] = {}
        self.joints_time: float | None = None
        self.stop_requested = False
        self.publish_count = 0
        self._target_publishers = {
            side: self.create_publisher(PoseStamped, topic, 10)
            for side, topic in EE_TARGET_TOPICS.items()
        }
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
            self.create_subscription(
                String,
                TOPICS["ground_truth"]["object_poses"],
                self._on_objects,
                10,
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

    def _on_joints(self, message: JointState) -> None:
        self.joints = {
            str(name): float(value)
            for name, value in zip(message.name, message.position)
        }
        self.joints_time = _stamp_seconds(message.header.stamp)

    def _on_objects(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            objects = {
                str(name): tuple(float(value) for value in pose)
                for name, pose in payload["objects"].items()
            }
            sim_time = float(payload["sim_time"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if all(
            len(pose) == 7 and all(math.isfinite(value) for value in pose)
            for pose in objects.values()
        ):
            self.objects = objects
            self.objects_time = sim_time

    def fresh(self, maximum_skew_s: float) -> bool:
        if self.sim_time is None:
            return False
        sample_times = [
            self.objects_time,
            self.base_time,
            self.joints_time,
            *(self.ee_time.get(side) for side in ("left", "right")),
        ]
        return all(
            sample_time is not None
            and 0.0 <= self.sim_time - sample_time <= maximum_skew_s
            for sample_time in sample_times
        )

    def conflicts(self) -> dict[str, int]:
        return {
            side: max(
                0,
                len(self.get_publishers_info_by_topic(topic)) - 1,
            )
            for side, topic in EE_TARGET_TOPICS.items()
        }

    def subscribers_ready(self) -> bool:
        return all(
            publisher.get_subscription_count() == 1
            for publisher in self._target_publishers.values()
        )

    def publish(self, targets: dict[str, tuple[float, ...]]) -> None:
        if self.sim_time is None:
            raise RuntimeError("simulator clock unavailable")
        stamp = Time(nanoseconds=max(0, int(self.sim_time * 1.0e9))).to_msg()
        for side, target in targets.items():
            message = PoseStamped()
            message.header.stamp = stamp
            message.header.frame_id = "world"
            message.pose.position.x = target[0]
            message.pose.position.y = target[1]
            message.pose.position.z = target[2]
            message.pose.orientation.x = target[3]
            message.pose.orientation.y = target[4]
            message.pose.orientation.z = target[5]
            message.pose.orientation.w = target[6]
            self._target_publishers[side].publish(message)
            self.publish_count += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-current-only", action="store_true")
    parser.add_argument(
        "--base-target",
        nargs=3,
        type=float,
        default=(2.100026845932007, 3.0529046058654785, -1.5706931352615356),
    )
    parser.add_argument("--base-position-tolerance-m", type=float, default=0.04)
    parser.add_argument("--base-yaw-tolerance-rad", type=float, default=0.08)
    parser.add_argument("--spine-tolerance-m", type=float, default=0.02)
    parser.add_argument("--gripper-min-open-fraction", type=float, default=0.95)
    parser.add_argument("--reference-pad-yaw-deg", type=float, default=90.0)
    parser.add_argument("--transition-duration-s", type=float, default=6.0)
    parser.add_argument("--stable-dwell-s", type=float, default=1.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.04)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=12.0)
    parser.add_argument("--maximum-skew-s", type=float, default=0.10)
    parser.add_argument("--max-duration-s", type=float, default=45.0)
    return parser


def _actual_targets(
    audit: dict[str, Any],
    objects: dict[str, tuple[float, ...]],
    pad_yaw: float,
    reference_pad_yaw: float,
) -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
    final = audit["final_target"]["measured_reference"]
    pad = objects["thermalpad"]
    yaw_delta = pad_yaw - reference_pad_yaw
    offset = tuple(float(value) for value in final["right_ee_relative_to_thermalpad_m"])
    rotated_offset = _rotate_z(offset, yaw_delta)
    reference_right = tuple(float(value) for value in final["right_ee"])
    reference_left = tuple(float(value) for value in final["left_ee"])
    yaw_quaternion = (0.0, 0.0, math.sin(yaw_delta / 2.0), math.cos(yaw_delta / 2.0))
    right_orientation = _quaternion_multiply_xyzw(
        yaw_quaternion, reference_right[3:7]
    )
    targets = {
        "left": reference_left,
        "right": (
            pad[0] + rotated_offset[0],
            pad[1] + rotated_offset[1],
            pad[2] + rotated_offset[2],
            *right_orientation,
        ),
    }
    provenance = {
        "dataset_episode": audit["selection"]["episode"],
        "dataset_frame": audit["selection"]["frame"],
        "reference_pad_position_m": final["thermalpad_position_m"],
        "live_pad_pose_wxyz": pad,
        "reference_right_ee_pad_offset_m": offset,
        "rotated_right_ee_pad_offset_m": rotated_offset,
        "pad_yaw_delta_deg": math.degrees(yaw_delta),
        "guessed_ik_used": False,
    }
    return targets, provenance


def main() -> int:
    args = build_parser().parse_args()
    if not args.smoke_current_only and args.audit is None:
        raise SystemExit("--audit is required unless --smoke-current-only is used")
    audit = (
        json.loads(args.audit.read_text(encoding="utf-8")) if args.audit else None
    )
    rclpy.init()
    node = GroundTruthPregraspNode()
    signal.signal(
        signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True)
    )
    signal.signal(signal.SIGINT, lambda *_: setattr(node, "stop_requested", True))
    wall_started = time.monotonic()
    last_sim_time: float | None = None
    started_sim: float | None = None
    stable_since: float | None = None
    initial: dict[str, tuple[float, ...]] | None = None
    targets: dict[str, tuple[float, ...]] | None = None
    provenance: dict[str, Any] = {}
    errors: dict[str, dict[str, float]] = {}
    reason = "host_watchdog_timeout"
    try:
        while (
            not node.stop_requested
            and time.monotonic() - wall_started < args.max_duration_s
        ):
            rclpy.spin_once(node, timeout_sec=0.02)
            if not node.fresh(args.maximum_skew_s) or not node.subscribers_ready():
                continue
            conflicts = node.conflicts()
            if any(conflicts.values()):
                reason = f"publisher_contention:{conflicts}"
                break
            assert node.sim_time is not None
            if last_sim_time is not None and node.sim_time < last_sim_time:
                reason = "simulator_clock_reset"
                break
            if last_sim_time is not None and node.sim_time == last_sim_time:
                continue
            last_sim_time = node.sim_time
            if initial is None:
                initial = dict(node.ee)
                if args.smoke_current_only:
                    targets = dict(initial)
                    provenance = {"mode": "hold_current_world_pose"}
                else:
                    assert audit is not None
                    assert node.base is not None
                    position_error = math.hypot(
                        node.base[0] - args.base_target[0],
                        node.base[1] - args.base_target[1],
                    )
                    yaw_error = abs(
                        math.atan2(
                            math.sin(node.base[2] - args.base_target[2]),
                            math.cos(node.base[2] - args.base_target[2]),
                        )
                    )
                    spine_target = float(
                        audit["final_target"]["measured_reference"]["spine_m"]
                    )
                    spine = resolve_joint(node.joints, SPINE_JOINT)
                    left_open = gripper_open_fraction(
                        resolve_joint(node.joints, LEFT_GRIPPER_DRIVER)
                    )
                    right_open = gripper_open_fraction(
                        resolve_joint(node.joints, RIGHT_GRIPPER_DRIVER)
                    )
                    if position_error > args.base_position_tolerance_m:
                        reason = f"base_position_gate:{position_error:.6f}"
                        break
                    if yaw_error > args.base_yaw_tolerance_rad:
                        reason = f"base_yaw_gate:{yaw_error:.6f}"
                        break
                    if abs(spine - spine_target) > args.spine_tolerance_m:
                        reason = f"spine_gate:{spine:.6f}"
                        break
                    if min(left_open, right_open) < args.gripper_min_open_fraction:
                        reason = f"gripper_open_gate:{left_open:.6f},{right_open:.6f}"
                        break
                    pad = node.objects.get("thermalpad")
                    if pad is None:
                        reason = "thermalpad_ground_truth_missing"
                        break
                    targets, provenance = _actual_targets(
                        audit,
                        node.objects,
                        _yaw_from_wxyz(pad[3:7]),
                        math.radians(args.reference_pad_yaw_deg),
                    )
                started_sim = node.sim_time
            assert initial is not None and targets is not None
            assert started_sim is not None
            fraction = min(
                1.0,
                max(
                    0.0,
                    (node.sim_time - started_sim) / args.transition_duration_s,
                ),
            )
            commanded = {}
            for side in ("left", "right"):
                commanded[side] = (
                    *(
                        initial[side][index]
                        + fraction * (targets[side][index] - initial[side][index])
                        for index in range(3)
                    ),
                    *_slerp(initial[side][3:7], targets[side][3:7], fraction),
                )
            node.publish(commanded)
            errors = {
                side: {
                    "position_m": _position_error(node.ee[side], targets[side]),
                    "orientation_deg": _orientation_error_deg(
                        node.ee[side], targets[side]
                    ),
                }
                for side in ("left", "right")
            }
            within_tolerance = all(
                value["position_m"] <= args.position_tolerance_m
                and value["orientation_deg"] <= args.orientation_tolerance_deg
                for value in errors.values()
            )
            if fraction < 1.0 or not within_tolerance:
                stable_since = None
            elif stable_since is None:
                stable_since = node.sim_time
            elif node.sim_time - stable_since >= args.stable_dwell_s:
                reason = (
                    "stable_current_pose_smoke"
                    if args.smoke_current_only
                    else "stable_dataset_ground_truth_pregrasp"
                )
                break
    finally:
        success = reason in {
            "stable_current_pose_smoke",
            "stable_dataset_ground_truth_pregrasp",
        }
        result = {
            "schema_version": 1,
            "success": success,
            "reason": reason,
            "clock": "simulator",
            "clock_topic": TOPICS["clock"],
            "host_clock_use": "process_watchdog_only",
            "command_header_clock": "simulator",
            "maximum_sample_skew_s": args.maximum_skew_s,
            "command_publications": node.publish_count,
            "initial_ee_world": initial,
            "target_ee_world": targets,
            "final_ee_world": node.ee,
            "final_base_xyyaw": node.base,
            "final_joints": node.joints,
            "handoff_sim_time": node.sim_time,
            "final_sample_sim_times": {
                "objects": node.objects_time,
                "base": node.base_time,
                "joints": node.joints_time,
                "ee": node.ee_time,
            },
            "final_errors": errors,
            "provenance": provenance,
            "tolerances": {
                "position_m": args.position_tolerance_m,
                "orientation_deg": args.orientation_tolerance_deg,
                "stable_dwell_sim_s": args.stable_dwell_s,
            },
            "elapsed_sim_s": (
                node.sim_time - started_sim
                if node.sim_time is not None and started_sim is not None
                else None
            ),
            "elapsed_wall_s": time.monotonic() - wall_started,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
