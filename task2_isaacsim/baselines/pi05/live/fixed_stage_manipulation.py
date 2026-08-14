#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Execute and verify the audited dataset-derived pre-inference joint route."""

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
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from task2_isaacsim.baselines.pi05.live.staging import (
    interpolate_staging_command,
    staging_entry_duration_s,
    staging_feedback,
    validate_staging_audit,
)
from task2_isaacsim.common.state_contract import (
    GRIPPER_CLOSED_RAD,
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
COMMAND_ENTRIES = TOPICS["bridge"]["joint_groups"]
SPINE_COMMAND_TOPIC = TOPICS["teleop"]["spine_target"]


class ManipulationStager(Node):
    def __init__(self) -> None:
        super().__init__("pi05_dataset_manipulation_stager")
        self.sim_time: float | None = None
        self.joints: dict[str, float] = {}
        self.ee: dict[str, tuple[float, ...]] = {}
        self.stop_requested = False
        self.publish_count = 0
        self.publishers: dict[str, Any] = {
            group: self.create_publisher(JointState, entry["command"], 10)
            for group, entry in COMMAND_ENTRIES.items()
        }
        self.publishers["spine"] = self.create_publisher(
            Float64, SPINE_COMMAND_TOPIC, 10
        )
        self.create_subscription(
            Clock, TOPICS["clock"], self._on_clock, qos_profile_sensor_data
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

    def _on_clock(self, message: Clock) -> None:
        self.sim_time = (
            float(message.clock.sec) + float(message.clock.nanosec) * 1e-9
        )

    def _on_joints(self, message: JointState) -> None:
        self.joints = {
            str(name): float(value)
            for name, value in zip(message.name, message.position)
        }

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

    def publisher_conflicts(self) -> dict[str, int]:
        topics = {
            **{
                group: entry["command"]
                for group, entry in COMMAND_ENTRIES.items()
            },
            "spine": SPINE_COMMAND_TOPIC,
        }
        return {
            group: max(0, len(self.get_publishers_info_by_topic(topic)) - 1)
            for group, topic in topics.items()
        }

    def samples_ready(self) -> bool:
        names = (
            *LEFT_JOINTS,
            *RIGHT_JOINTS,
            LEFT_GRIPPER_DRIVER,
            RIGHT_GRIPPER_DRIVER,
            SPINE_JOINT,
        )
        return (
            self.sim_time is not None
            and all(math.isfinite(resolve_joint(self.joints, name)) for name in names)
            and all(
                side in self.ee
                and len(self.ee[side]) == 7
                and all(math.isfinite(value) for value in self.ee[side])
                for side in ("left", "right")
            )
        )

    def publish_command(self, command: list[float]) -> None:
        if self.sim_time is None:
            raise RuntimeError("simulator clock unavailable")
        if len(command) != 17:
            raise ValueError("staging command must contain action indices 3:20")
        stamp = Time(nanoseconds=max(0, int(self.sim_time * 1e9))).to_msg()
        groups = {
            "left_arm": (LEFT_JOINTS, command[0:7]),
            "right_arm": (RIGHT_JOINTS, command[7:14]),
            "left_gripper": (
                (LEFT_GRIPPER_DRIVER,),
                ((1.0 - command[14]) * GRIPPER_CLOSED_RAD,),
            ),
            "right_gripper": (
                (RIGHT_GRIPPER_DRIVER,),
                ((1.0 - command[15]) * GRIPPER_CLOSED_RAD,),
            ),
        }
        for group, (names, positions) in groups.items():
            message = JointState()
            message.header.stamp = stamp
            message.name = list(names)
            message.position = list(positions)
            self.publishers[group].publish(message)
        self.publishers["spine"].publish(Float64(data=command[16]))
        self.publish_count += 5

    def measured_command(self) -> tuple[float, ...]:
        return (
            *(resolve_joint(self.joints, name) for name in LEFT_JOINTS),
            *(resolve_joint(self.joints, name) for name in RIGHT_JOINTS),
            gripper_open_fraction(
                resolve_joint(self.joints, LEFT_GRIPPER_DRIVER)
            ),
            gripper_open_fraction(
                resolve_joint(self.joints, RIGHT_GRIPPER_DRIVER)
            ),
            resolve_joint(self.joints, SPINE_JOINT),
        )

    def feedback(self, audit: dict[str, Any]) -> dict[str, Any]:
        return staging_feedback(
            audit=audit,
            left_arm=tuple(resolve_joint(self.joints, name) for name in LEFT_JOINTS),
            right_arm=tuple(
                resolve_joint(self.joints, name) for name in RIGHT_JOINTS
            ),
            spine_m=resolve_joint(self.joints, SPINE_JOINT),
            left_gripper_open=gripper_open_fraction(
                resolve_joint(self.joints, LEFT_GRIPPER_DRIVER)
            ),
            right_gripper_open=gripper_open_fraction(
                resolve_joint(self.joints, RIGHT_GRIPPER_DRIVER)
            ),
            left_ee=self.ee["left"],
            right_ee=self.ee["right"],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration-s", type=float, default=300.0)
    parser.add_argument("--spin-timeout-s", type=float, default=0.02)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit = validate_staging_audit(
        json.loads(args.audit.read_text(encoding="utf-8"))
    )
    route = audit["trajectory"]
    dwell_s = float(audit["tolerances"]["stable_dwell_sim_s"])
    rclpy.init()
    node = ManipulationStager()
    signal.signal(
        signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True)
    )
    signal.signal(
        signal.SIGINT, lambda *_: setattr(node, "stop_requested", True)
    )
    wall_started = time.monotonic()
    sim_started: float | None = None
    last_sim_time: float | None = None
    initial_command: tuple[float, ...] | None = None
    entry_duration_s: float | None = None
    stable_since: float | None = None
    route_index = 0
    feedback: dict[str, Any] | None = None
    reason = "host_watchdog_timeout"
    try:
        while (
            not node.stop_requested
            and time.monotonic() - wall_started < args.max_duration_s
        ):
            rclpy.spin_once(node, timeout_sec=args.spin_timeout_s)
            if not node.samples_ready():
                continue
            conflicts = node.publisher_conflicts()
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
            if sim_started is None:
                sim_started = node.sim_time
                initial_command = node.measured_command()
                try:
                    entry_duration_s = staging_entry_duration_s(
                        audit, initial_command, tuple(route[0]["command"])
                    )
                except (KeyError, TypeError, ValueError) as error:
                    reason = f"invalid_measured_entry:{error}"
                    break
            elapsed_sim = node.sim_time - sim_started
            assert initial_command is not None
            assert entry_duration_s is not None
            if elapsed_sim < entry_duration_s:
                fraction = (
                    elapsed_sim / entry_duration_s
                    if entry_duration_s > 0.0
                    else 1.0
                )
                node.publish_command(
                    interpolate_staging_command(
                        initial_command, tuple(route[0]["command"]), fraction
                    )
                )
                continue
            route_elapsed_sim = elapsed_sim - entry_duration_s
            if route_index < len(route):
                row = route[route_index]
                if route_elapsed_sim >= float(row["scheduled_at_s"]):
                    node.publish_command(row["command"])
                    route_index += 1
                continue
            node.publish_command(route[-1]["command"])
            feedback = node.feedback(audit)
            if not feedback["within_tolerance"]:
                stable_since = None
            elif stable_since is None:
                stable_since = node.sim_time
            elif node.sim_time - stable_since >= dwell_s:
                reason = "stable_dataset_pregrasp"
                break
    finally:
        success = reason == "stable_dataset_pregrasp"
        result = {
            "schema_version": 1,
            "success": success,
            "reason": reason,
            "clock": "simulator",
            "clock_topic": TOPICS["clock"],
            "host_clock_use": "process_watchdog_only",
            "command_header_clock": "simulator",
            "selection": audit["selection"],
            "final_target": audit["final_target"],
            "tolerances": audit["tolerances"],
            "route_frames_completed": route_index,
            "route_frames_total": len(route),
            "measured_entry_command": initial_command,
            "entry_transition_duration_sim_s": entry_duration_s,
            "scheduled_duration_sim_s": route[-1]["scheduled_at_s"],
            "elapsed_sim_s": (
                node.sim_time - sim_started
                if node.sim_time is not None and sim_started is not None
                else None
            ),
            "elapsed_wall_s": time.monotonic() - wall_started,
            "command_publications": node.publish_count,
            "feedback": feedback,
            "guessed_ik_used": False,
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
