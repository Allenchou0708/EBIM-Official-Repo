#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Drive the Task 2 base to a fixed or live-GT grasp pose and settle."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from task2_isaacsim.baselines.pi05.live.ground_truth_joint_lift import (
    REFERENCE_PAD_XYYAW,
    anchored_base_pose,
    deepen_grasp_base_pose,
)
from task2_isaacsim.scripts.topics import load_topics


TOPICS = load_topics()


def angular_error(target: float, actual: float) -> float:
    return math.atan2(math.sin(target - actual), math.cos(target - actual))


class FixedBaseStager(Node):
    def __init__(
        self, *, command_topic: str, odom_topic: str, clock_topic: str
    ):
        super().__init__("pi05_fixed_base_stager")
        self.publisher = self.create_publisher(String, command_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(
            Clock, clock_topic, self._on_clock, qos_profile_sensor_data
        )
        self.create_subscription(
            String,
            TOPICS["ground_truth"]["object_poses"],
            self._on_objects,
            10,
        )
        self.pose: tuple[float, float, float] | None = None
        self.velocity = (0.0, 0.0, 0.0)
        self.speed = math.inf
        self.sim_time: float | None = None
        self.thermalpad_pose: tuple[float, ...] | None = None
        self.objects_time: float | None = None
        self.stop_requested = False

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
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
        self.pose = (pose.position.x, pose.position.y, yaw)
        self.velocity = (
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
        )
        self.speed = math.sqrt(
            twist.linear.x**2 + twist.linear.y**2 + twist.angular.z**2
        )

    def _on_clock(self, message: Clock) -> None:
        stamp = message.clock
        self.sim_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _on_objects(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            thermalpad = tuple(
                float(value) for value in payload["objects"]["thermalpad"]
            )
            sim_time = float(payload["sim_time"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if len(thermalpad) == 7 and all(
            math.isfinite(value) for value in thermalpad
        ):
            self.thermalpad_pose = thermalpad
            self.objects_time = sim_time

    def publish(self, token: str) -> None:
        self.publisher.publish(String(data=token))

    def stop(self, duration_sim_s: float = 0.10) -> None:
        started_sim = self.sim_time
        started_wall = time.monotonic()
        last_publish_sim: float | None = None
        while (
            started_sim is None
            or self.sim_time is None
            or self.sim_time - started_sim < duration_sim_s
        ):
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.sim_time is not None and (
                last_publish_sim is None or self.sim_time > last_publish_sim
            ):
                self.publish("NONE")
                last_publish_sim = self.sim_time
            if time.monotonic() - started_wall >= 2.0:
                break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target", nargs=3, type=float)
    target.add_argument(
        "--target-from-live-pad",
        action="store_true",
        help="derive the grasp base from the simulator thermal-pad GT pose",
    )
    parser.add_argument("--grasp-depth-bias-m", type=float, default=0.005)
    parser.add_argument("--grasp-yaw-limit-deg", type=float, default=5.0)
    parser.add_argument("--maximum-object-skew-s", type=float, default=0.10)
    parser.add_argument("--command-topic", default="/pedal/state")
    parser.add_argument("--odom-topic", default="/isaac/odom")
    parser.add_argument("--clock-topic", default="/isaac/clock")
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.04)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.08)
    parser.add_argument("--velocity-threshold", type=float, default=0.02)
    parser.add_argument("--settle-duration-s", type=float, default=1.0)
    parser.add_argument("--stop-duration-s", type=float, default=0.10)
    parser.add_argument("--correction-pulse-s", type=float, default=0.05)
    parser.add_argument(
        "--straight-correction-pulse-s", type=float, default=0.10
    )
    parser.add_argument("--max-duration-s", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    return parser


def correction_token(
    pose: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    yaw_tolerance_rad: float = 0.08,
) -> str:
    x, y, yaw = pose
    dx, dy = target[0] - x, target[1] - y
    yaw_error = angular_error(target[2], yaw)
    if abs(yaw_error) > yaw_tolerance_rad:
        return "A+C" if yaw_error > 0.0 else "B+C"
    forward_error = math.cos(yaw) * dx + math.sin(yaw) * dy
    left_error = -math.sin(yaw) * dx + math.cos(yaw) * dy
    if abs(forward_error) >= abs(left_error):
        return "FWD" if forward_error > 0.0 else "BACK"
    return "A" if left_error > 0.0 else "B"


def brake_token(velocity: tuple[float, float, float]) -> str:
    vx, vy, wz = velocity
    axis = max(range(3), key=lambda index: abs(velocity[index]))
    if axis == 0:
        return "BACK" if vx > 0.0 else "FWD"
    if axis == 1:
        return "B" if vy > 0.0 else "A"
    return "B+C" if wz > 0.0 else "A+C"


def main() -> int:
    args = build_parser().parse_args()
    target = tuple(args.target) if args.target is not None else None
    target_source = "fixed_cli" if target is not None else "live_thermalpad_gt"
    period_s = 1.0 / args.rate_hz
    rclpy.init()
    node = FixedBaseStager(
        command_topic=args.command_topic,
        odom_topic=args.odom_topic,
        clock_topic=args.clock_topic,
    )
    signal.signal(
        signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True)
    )
    signal.signal(
        signal.SIGINT, lambda *_: setattr(node, "stop_requested", True)
    )
    phase = "WAIT_ODOM"
    phase_since: float | None = None
    wall_started = time.monotonic()
    sim_started: float | None = None
    active_token = "NONE"
    command_until: float | None = None
    next_correction_at: float | None = None
    route: list[dict[str, object]] = []
    last_sim_time: float | None = None

    def enter(next_phase: str) -> None:
        nonlocal phase, phase_since
        assert node.sim_time is not None
        phase = next_phase
        phase_since = node.sim_time
        route.append(
            {"phase": phase, "pose": node.pose, "sim_time": node.sim_time}
        )
        print(f"base_stage phase={phase} pose={node.pose}", flush=True)

    try:
        while (
            not node.stop_requested
            and time.monotonic() - wall_started < args.max_duration_s
        ):
            rclpy.spin_once(node, timeout_sec=period_s)
            now = node.sim_time
            if node.pose is None or now is None:
                continue
            if target is None:
                if (
                    node.thermalpad_pose is None
                    or node.objects_time is None
                    or not 0.0
                    <= now - node.objects_time
                    <= args.maximum_object_skew_s
                ):
                    continue
                target = deepen_grasp_base_pose(
                    anchored_base_pose(
                        node.thermalpad_pose,
                        REFERENCE_PAD_XYYAW,
                        max_yaw_delta_rad=math.radians(
                            args.grasp_yaw_limit_deg
                        ),
                    ),
                    node.thermalpad_pose,
                    args.grasp_depth_bias_m,
                )
                print(
                    "base_stage live_gt_target="
                    f"{target} thermalpad={node.thermalpad_pose}",
                    flush=True,
                )
            if last_sim_time is not None and now < last_sim_time:
                phase = "SIMULATOR_CLOCK_RESET"
                break
            if last_sim_time is not None and now == last_sim_time:
                continue
            last_sim_time = now
            if sim_started is None:
                sim_started = now
                phase_since = now
                command_until = now
                next_correction_at = now
            if node.publisher.get_subscription_count() < 1:
                continue
            if len(node.get_publishers_info_by_topic(args.command_topic)) > 1:
                raise RuntimeError("another /pedal/state publisher is active")
            x, y, yaw = node.pose
            if phase == "WAIT_ODOM":
                position_error = math.hypot(x - target[0], y - target[1])
                yaw_error = abs(angular_error(target[2], yaw))
                if (
                    position_error <= args.position_tolerance_m
                    and yaw_error <= args.yaw_tolerance_rad
                ):
                    enter("SETTLE")
                elif position_error <= 0.5:
                    enter("ODOMETRY_CORRECTION")
                else:
                    enter("BACK")
            elif phase == "BACK":
                node.publish("BACK")
                if y >= target[1] - 0.08:
                    node.stop()
                    enter("STOP_AFTER_BACK")
            elif phase == "STOP_AFTER_BACK":
                node.publish("NONE")
                assert phase_since is not None
                if now - phase_since >= args.stop_duration_s:
                    enter("STRAFE_RIGHT")
            elif phase == "STRAFE_RIGHT":
                node.publish("B")
                if x <= target[0] + 0.35:
                    enter("BRAKE_RIGHT")
            elif phase == "BRAKE_RIGHT":
                if node.velocity[1] < -0.10:
                    node.publish("A")
                else:
                    node.stop()
                    enter("STOP_AFTER_RIGHT")
            elif phase == "STOP_AFTER_RIGHT":
                node.publish("NONE")
                assert phase_since is not None
                if now - phase_since >= args.stop_duration_s:
                    enter("ODOMETRY_CORRECTION")
            elif phase == "ODOMETRY_CORRECTION":
                position_error = math.hypot(x - target[0], y - target[1])
                yaw_error = abs(angular_error(target[2], yaw))
                if (
                    position_error <= args.position_tolerance_m
                    and yaw_error <= args.yaw_tolerance_rad
                    and node.speed <= args.velocity_threshold
                ):
                    node.stop()
                    enter("SETTLE")
                    continue
                assert command_until is not None
                assert next_correction_at is not None
                if now < command_until:
                    node.publish(active_token)
                elif now < next_correction_at:
                    node.publish("NONE")
                else:
                    within_pose = (
                        position_error <= args.position_tolerance_m
                        and yaw_error <= args.yaw_tolerance_rad
                    )
                    active_token = (
                        brake_token(node.velocity)
                        if within_pose
                        else correction_token(
                            node.pose,
                            target,
                            yaw_tolerance_rad=args.yaw_tolerance_rad,
                        )
                    )
                    pulse_s = (
                        args.straight_correction_pulse_s
                        if not within_pose and active_token in ("FWD", "BACK")
                        else args.correction_pulse_s
                    )
                    command_until = now + pulse_s
                    next_correction_at = command_until + args.stop_duration_s
                    node.publish(active_token)
                    print(
                        "base_stage correction="
                        f"{active_token} pose={node.pose} "
                        f"velocity={node.velocity}",
                        flush=True,
                    )
            elif phase == "SETTLE":
                node.publish("NONE")
                assert phase_since is not None
                position_error = math.hypot(x - target[0], y - target[1])
                yaw_error = abs(angular_error(target[2], yaw))
                within_pose = (
                    position_error <= args.position_tolerance_m
                    and yaw_error <= args.yaw_tolerance_rad
                )
                if not within_pose:
                    enter("ODOMETRY_CORRECTION")
                elif node.speed > args.velocity_threshold:
                    # Stay command-free while residual velocity decays.  Do
                    # not turn a successful pose into a brake-pulse limit
                    # cycle; restart only the stable dwell clock.
                    phase_since = now
                elif now - phase_since >= args.settle_duration_s:
                    enter("COMPLETE")
                    break
        success = phase == "COMPLETE"
    finally:
        node.stop()
        final_pose = node.pose
        result = {
            "success": phase == "COMPLETE",
            "phase": phase,
            "target": target,
            "target_source": target_source,
            "thermalpad_pose_wxyz": node.thermalpad_pose,
            "final_pose": final_pose,
            "final_speed": node.speed,
            "elapsed_s": time.monotonic() - wall_started,
            "elapsed_sim_s": (
                node.sim_time - sim_started
                if node.sim_time is not None and sim_started is not None
                else None
            ),
            "clock": "simulator",
            "clock_topic": args.clock_topic,
            "route": route,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
