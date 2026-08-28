#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""External-only simulator GT audit for a report_hybrid shadow result."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from task2_isaacsim.baselines.report_hybrid.core import wrap_angle
from task2_isaacsim.scripts.topics import load_topics


def yaw_from_wxyz(pose: list[float]) -> float:
    _, _, _, w, x, y, z = pose
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Capture(Node):
    def __init__(self):
        super().__init__("task2_report_hybrid_external_audit")
        self.payload = None
        self.create_subscription(
            String,
            load_topics()["ground_truth"]["object_poses"],
            self._on_objects,
            10,
        )

    def _on_objects(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if (
                "thermalpad" in payload["objects"]
                and "board_target" in payload["objects"]
            ):
                self.payload = payload
        except (KeyError, TypeError, json.JSONDecodeError):
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()
    shadow = json.loads(args.shadow.read_text(encoding="utf-8"))
    if shadow.get("mode") != "zero_publication_shadow" or not shadow.get(
        "success"
    ):
        raise SystemExit("shadow must pass before external audit")
    rclpy.init()
    node = Capture()
    started = time.monotonic()
    while node.payload is None and time.monotonic() - started < args.timeout_s:
        rclpy.spin_once(node, timeout_sec=0.05)
    payload = node.payload
    node.destroy_node()
    rclpy.shutdown()
    if payload is None:
        result = {"success": False, "reason": "ground_truth_timeout"}
    else:
        perception = shadow["perception"]
        measurements = {}
        for key, object_name in (
            ("pad", "thermalpad"),
            ("target", "board_target"),
        ):
            predicted = perception[f"{key}_center_xyz"]
            actual = payload["objects"][object_name]
            translation = math.dist(predicted, actual[:3])
            yaw_error = abs(
                math.degrees(
                    wrap_angle(
                        perception[f"{key}_yaw_rad"] - yaw_from_wxyz(actual)
                    )
                )
            )
            # Rectangular axes are pi-periodic.
            yaw_error = min(yaw_error, abs(180.0 - yaw_error))
            measurements[key] = {
                "translation_m": translation,
                "yaw_error_deg": yaw_error,
            }
        translations = [
            value["translation_m"] for value in measurements.values()
        ]
        yaws = [value["yaw_error_deg"] for value in measurements.values()]
        result = {
            "schema_version": 1,
            "mode": "external_gt_audit_not_policy_input",
            "measurements": measurements,
            "median_translation_m": statistics.median(translations),
            "median_yaw_error_deg": statistics.median(yaws),
        }
        result["success"] = (
            result["median_translation_m"] <= 0.015
            and result["median_yaw_error_deg"] <= 8.0
        )
        result["thresholds"] = {
            "median_translation_m": 0.015,
            "median_yaw_error_deg": 8.0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
