#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Command and verify the fixed Task 2 pre-manipulation spine height."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from task2_isaacsim.common.state_contract import SPINE_JOINT


class FixedSpineStager(Node):
    def __init__(self, *, command_topic: str, state_topic: str, clock_topic: str):
        super().__init__("pi05_fixed_spine_stager")
        self.publisher = self.create_publisher(Float64, command_topic, 10)
        self.create_subscription(JointState, state_topic, self._on_state, 10)
        self.create_subscription(
            Clock, clock_topic, self._on_clock, qos_profile_sensor_data
        )
        self.position: float | None = None
        self.previous_position: float | None = None
        self.previous_time: float | None = None
        self.velocity = math.inf
        self.stop_requested = False
        self.sim_time: float | None = None

    def _on_clock(self, message: Clock) -> None:
        self.sim_time = (
            float(message.clock.sec) + float(message.clock.nanosec) * 1e-9
        )

    def _on_state(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        if SPINE_JOINT not in positions:
            return
        stamp = message.header.stamp
        now = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        position = float(positions[SPINE_JOINT])
        if self.previous_position is not None and self.previous_time is not None:
            elapsed = now - self.previous_time
            if elapsed > 0.0:
                self.velocity = abs(position - self.previous_position) / elapsed
        self.position = position
        self.previous_position = position
        self.previous_time = now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-m", type=float, default=0.50)
    parser.add_argument("--measured-target-m", type=float, default=0.4857)
    parser.add_argument("--tolerance-m", type=float, default=0.015)
    parser.add_argument("--velocity-threshold-mps", type=float, default=0.003)
    parser.add_argument("--settle-duration-s", type=float, default=1.0)
    parser.add_argument("--max-duration-s", type=float, default=40.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--command-topic", default="/isaac/spine_target")
    parser.add_argument("--state-topic", default="/isaac/joint_states_full")
    parser.add_argument("--clock-topic", default="/isaac/clock")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rclpy.init()
    node = FixedSpineStager(
        command_topic=args.command_topic,
        state_topic=args.state_topic,
        clock_topic=args.clock_topic,
    )
    signal.signal(
        signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True)
    )
    signal.signal(
        signal.SIGINT, lambda *_: setattr(node, "stop_requested", True)
    )
    started = time.monotonic()
    stable_since: float | None = None
    period_s = 1.0 / args.rate_hz
    success = False
    reason = "timeout"
    last_sim_time: float | None = None
    try:
        while (
            not node.stop_requested
            and time.monotonic() - started < args.max_duration_s
        ):
            rclpy.spin_once(node, timeout_sec=period_s)
            now = node.sim_time
            if now is None:
                stable_since = None
                continue
            if last_sim_time is not None and now < last_sim_time:
                reason = "simulator_clock_reset"
                break
            if last_sim_time is not None and now == last_sim_time:
                continue
            last_sim_time = now
            if node.publisher.get_subscription_count() < 1:
                stable_since = None
                continue
            if len(node.get_publishers_info_by_topic(args.command_topic)) > 1:
                raise RuntimeError("another spine target publisher is active")
            node.publisher.publish(Float64(data=args.target_m))
            if node.position is None:
                continue
            stable = (
                abs(node.position - args.measured_target_m) <= args.tolerance_m
                and node.velocity <= args.velocity_threshold_mps
            )
            if not stable:
                stable_since = None
            elif stable_since is None:
                stable_since = now
            elif now - stable_since >= args.settle_duration_s:
                success = True
                reason = "stable"
                break
    except RuntimeError as error:
        reason = str(error)
    finally:
        result = {
            "success": success,
            "reason": reason,
            "command_target_m": args.target_m,
            "measured_target_m": args.measured_target_m,
            "final_position_m": node.position,
            "final_velocity_mps": node.velocity,
            "elapsed_s": time.monotonic() - started,
            "clock": "simulator",
            "clock_topic": args.clock_topic,
            "stable_dwell_clock": "simulator",
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
