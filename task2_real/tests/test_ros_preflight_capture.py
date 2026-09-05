from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from task2_real.ros_preflight_capture import (
    COMMAND_KEYS,
    REQUIRED_STREAMS,
    _load_topics,
    summarize_image,
    summarize_joint_state,
    summarize_laser_scan,
)


ROOT = Path(__file__).resolve().parents[1]


def stamp(seconds: int = 12, nanoseconds: int = 500_000_000) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=seconds, nanosec=nanoseconds),
            frame_id="frame",
        )
    )


class RosPreflightCaptureTest(unittest.TestCase):
    def test_tool_source_contains_no_publisher_creation(self) -> None:
        source = (ROOT / "ros_preflight_capture.py").read_text(encoding="utf-8")
        self.assertNotIn("create_" + "publisher(", source)

    def test_contract_covers_every_read_and_command_topic(self) -> None:
        topics = _load_topics(ROOT / "contract.json")
        self.assertEqual(
            set(REQUIRED_STREAMS) | set(COMMAND_KEYS),
            (set(REQUIRED_STREAMS) | set(COMMAND_KEYS)) & set(topics),
        )

    def test_joint_state_preserves_names_order_and_all_numeric_fields(self) -> None:
        message = stamp()
        message.name = ["joint_b", "joint_a"]
        message.position = [2.0, 1.0]
        message.velocity = [0.2, 0.1]
        message.effort = [4.0, 3.0]
        result = summarize_joint_state(message)
        self.assertEqual(result["names"], ["joint_b", "joint_a"])
        self.assertEqual(result["position"], [2.0, 1.0])
        self.assertEqual(result["velocity"], [0.2, 0.1])
        self.assertEqual(result["effort"], [4.0, 3.0])
        self.assertEqual(result["stamp_s"], 12.5)

    def test_image_records_shape_without_copying_payload(self) -> None:
        message = stamp()
        message.height = 480
        message.width = 640
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = 1920
        message.data = bytes(480 * 640 * 3)
        result = summarize_image(message)
        self.assertEqual(result["data_bytes"], 480 * 640 * 3)
        self.assertNotIn("data", result)

    def test_lidar_reports_nonfinite_counts_without_serializing_nan(self) -> None:
        message = stamp()
        message.ranges = [0.5, math.inf, math.nan, 2.0]
        message.range_min = 0.1
        message.range_max = 10.0
        result = summarize_laser_scan(message)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["finite_samples"], 2)
        self.assertEqual(result["observed_min"], 0.5)
        self.assertEqual(result["observed_max"], 2.0)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
