#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Capture one simulator-clock-fresh ROS Image as PNG evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image

from task2_isaacsim.scripts.topics import load_topics


def _seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


class CameraCapture(Node):
    def __init__(self, topic: str, maximum_skew_s: float) -> None:
        super().__init__("task2_camera_capture")
        self.maximum_skew_s = maximum_skew_s
        self.sim_time: float | None = None
        self.message: Image | None = None
        self.skew_s: float | None = None
        self._clock_subscription = self.create_subscription(
            Clock,
            load_topics()["clock"],
            self._on_clock,
            qos_profile_sensor_data,
        )
        self._image_subscription = self.create_subscription(
            Image, topic, self._on_image, qos_profile_sensor_data
        )

    def _on_clock(self, message: Clock) -> None:
        self.sim_time = _seconds(message.clock)

    def _on_image(self, message: Image) -> None:
        if self.sim_time is None:
            return
        skew = self.sim_time - _seconds(message.header.stamp)
        if 0.0 <= skew <= self.maximum_skew_s:
            self.message = message
            self.skew_s = skew


def _to_pillow(message: Image) -> PILImage.Image:
    encodings = {
        "rgb8": ("RGB", "RGB"),
        "bgr8": ("RGB", "BGR"),
        "rgba8": ("RGBA", "RGBA"),
        "bgra8": ("RGBA", "BGRA"),
        "mono8": ("L", "L"),
    }
    if message.encoding not in encodings:
        raise ValueError(f"unsupported image encoding: {message.encoding}")
    mode, raw_mode = encodings[message.encoding]
    return PILImage.frombytes(
        mode,
        (message.width, message.height),
        bytes(message.data),
        "raw",
        raw_mode,
        message.step,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-skew-s", type=float, default=0.10)
    parser.add_argument("--max-duration-s", type=float, default=20.0)
    args = parser.parse_args()
    rclpy.init()
    node = CameraCapture(args.topic, args.maximum_skew_s)
    started = time.monotonic()
    reason = "host_watchdog_timeout"
    try:
        while (
            node.message is None
            and time.monotonic() - started < args.max_duration_s
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.message is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _to_pillow(node.message).save(args.output)
            reason = "fresh_image_saved"
    finally:
        message = node.message
        manifest = {
            "success": reason == "fresh_image_saved",
            "reason": reason,
            "topic": args.topic,
            "output": str(args.output),
            "clock": "simulator",
            "maximum_skew_s": args.maximum_skew_s,
            "capture_skew_s": node.skew_s,
            "stamp_s": _seconds(message.header.stamp) if message else None,
            "width": message.width if message else None,
            "height": message.height if message else None,
            "encoding": message.encoding if message else None,
            "command_publications": 0,
        }
        manifest_path = args.output.with_suffix(".json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, sort_keys=True), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if reason == "fresh_image_saved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
