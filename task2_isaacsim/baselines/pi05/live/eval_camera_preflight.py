#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reject a blank Task 2 eval camera before consuming a live attempt."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from task2_isaacsim.scripts.topics import camera_topic, load_topics

TASK_CLASS_NAMES = ("thermalpad", "liner", "target")


class EvalCameraPreflight(Node):
    def __init__(self):
        super().__init__("task2_eval_camera_preflight")
        topics = load_topics()
        entry = topics["cameras"]["eval"]
        self.rgb: Image | None = None
        self.semantic_labels: str | None = None
        self.bbox_labels: str | None = None
        self.stop_requested = False
        self.create_subscription(
            Image,
            camera_topic(topics, entry["namespace"], "image"),
            self._on_rgb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, entry["semantic_labels"], self._on_semantic, 10
        )
        self.create_subscription(
            String, entry["bbox_tight_labels"], self._on_bbox, 10
        )

    def _on_rgb(self, message: Image) -> None:
        self.rgb = message

    def _on_semantic(self, message: String) -> None:
        self.semantic_labels = message.data

    def _on_bbox(self, message: String) -> None:
        self.bbox_labels = message.data


def _class_names(payload: str | None) -> list[str]:
    if not payload:
        return []
    lowered = payload.lower()
    return [name for name in TASK_CLASS_NAMES if name in lowered]


def _rgb_metrics(message: Image | None) -> dict[str, object]:
    if message is None or not message.data:
        return {"received": False, "not_all_white": False}
    array = np.frombuffer(message.data, dtype=np.uint8)
    return {
        "received": True,
        "shape": [message.height, message.width],
        "minimum": int(array.min()),
        "maximum": int(array.max()),
        "mean": float(array.mean()),
        "not_all_white": bool(np.any(array < 250)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = EvalCameraPreflight()
    signal.signal(
        signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True)
    )
    signal.signal(
        signal.SIGINT, lambda *_: setattr(node, "stop_requested", True)
    )
    started = time.monotonic()
    while (
        not node.stop_requested
        and time.monotonic() - started < args.timeout_s
        and (
            node.rgb is None
            or node.semantic_labels is None
            or node.bbox_labels is None
        )
    ):
        rclpy.spin_once(node, timeout_sec=0.05)
    rgb = _rgb_metrics(node.rgb)
    semantic_classes = _class_names(node.semantic_labels)
    bbox_classes = _class_names(node.bbox_labels)
    passed = bool(
        rgb["not_all_white"] and semantic_classes and bbox_classes
    )
    result = {
        "passed": passed,
        "rgb": rgb,
        "semantic_classes": semantic_classes,
        "bbox_classes": bbox_classes,
        "semantic_payload": node.semantic_labels,
        "bbox_payload": node.bbox_labels,
        "elapsed_s": time.monotonic() - started,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
