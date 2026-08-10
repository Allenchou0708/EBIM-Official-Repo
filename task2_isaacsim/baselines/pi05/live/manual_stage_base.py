#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Publish one bounded operator base-staging pulse, then an explicit stop.

This helper is intended to run with the ROS Python environment bundled in the
Isaac Sim container.  It is deliberately separate from the live runner: the
runner never publishes base commands.
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TOKENS = ("FWD", "BACK", "A", "B", "A+C", "B+C")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token", choices=TOKENS)
    parser.add_argument("duration_s", type=float)
    parser.add_argument("--topic", default="/pedal/state")
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--discovery-timeout-s", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if args.rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")

    rclpy.init()
    node = Node("pi05_manual_base_stager")
    publisher = node.create_publisher(String, args.topic, 10)
    try:
        discovery_deadline = time.monotonic() + args.discovery_timeout_s
        while publisher.get_subscription_count() < 1:
            if time.monotonic() >= discovery_deadline:
                raise RuntimeError(f"no subscriber discovered on {args.topic}")
            rclpy.spin_once(node, timeout_sec=0.1)

        period_s = 1.0 / args.rate_hz
        deadline = time.monotonic() + args.duration_s
        command = String(data=args.token)
        while time.monotonic() < deadline:
            publisher.publish(command)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period_s)
    finally:
        stop = String(data="NONE")
        for _ in range(10):
            publisher.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
