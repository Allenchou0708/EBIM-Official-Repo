"""Capture Phase II real-robot ROS interface evidence without publishers.

This tool is intentionally separate from policy execution.  It records message
shapes, joint names/order, numeric fields, receive freshness, and command-topic
publisher discovery so site-specific calibration can be completed from facts.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable


REQUIRED_STREAMS = (
    "base_pose",
    "base_twist",
    "base_odom",
    "spine_state",
    "left_arm_state",
    "right_arm_state",
    "right_external_joint_torques",
    "right_external_wrench",
    "left_gripper_state",
    "right_gripper_state",
    "head_rgb",
    "right_wrist_rgb",
    "lidar_front",
    "lidar_rear",
)
COMMAND_KEYS = (
    "base_command",
    "spine_command",
    "left_arm_command",
    "right_arm_command",
    "left_gripper_command",
    "right_gripper_command",
)


def _finite_or_none(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _stamp_seconds(message: Any) -> float | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return seconds if math.isfinite(seconds) else None


def summarize_joint_state(message: Any) -> dict[str, Any]:
    return {
        "stamp_s": _stamp_seconds(message),
        "names": [str(value) for value in message.name],
        "position": [_finite_or_none(value) for value in message.position],
        "velocity": [_finite_or_none(value) for value in message.velocity],
        "effort": [_finite_or_none(value) for value in message.effort],
    }


def summarize_image(message: Any) -> dict[str, Any]:
    return {
        "stamp_s": _stamp_seconds(message),
        "height": int(message.height),
        "width": int(message.width),
        "encoding": str(message.encoding),
        "is_bigendian": int(message.is_bigendian),
        "step": int(message.step),
        "data_bytes": len(message.data),
    }


def summarize_pose_stamped(message: Any) -> dict[str, Any]:
    pose = message.pose
    return {
        "stamp_s": _stamp_seconds(message),
        "frame_id": str(message.header.frame_id),
        "position": [
            _finite_or_none(pose.position.x),
            _finite_or_none(pose.position.y),
            _finite_or_none(pose.position.z),
        ],
        "orientation_xyzw": [
            _finite_or_none(pose.orientation.x),
            _finite_or_none(pose.orientation.y),
            _finite_or_none(pose.orientation.z),
            _finite_or_none(pose.orientation.w),
        ],
    }


def _twist_payload(twist: Any) -> dict[str, Any]:
    return {
        "linear_xyz": [
            _finite_or_none(twist.linear.x),
            _finite_or_none(twist.linear.y),
            _finite_or_none(twist.linear.z),
        ],
        "angular_xyz": [
            _finite_or_none(twist.angular.x),
            _finite_or_none(twist.angular.y),
            _finite_or_none(twist.angular.z),
        ],
    }


def summarize_twist_stamped(message: Any) -> dict[str, Any]:
    return {
        "stamp_s": _stamp_seconds(message),
        "frame_id": str(message.header.frame_id),
        **_twist_payload(message.twist),
    }


def summarize_odometry(message: Any) -> dict[str, Any]:
    pose = message.pose.pose
    return {
        "stamp_s": _stamp_seconds(message),
        "frame_id": str(message.header.frame_id),
        "child_frame_id": str(message.child_frame_id),
        "position": [
            _finite_or_none(pose.position.x),
            _finite_or_none(pose.position.y),
            _finite_or_none(pose.position.z),
        ],
        "orientation_xyzw": [
            _finite_or_none(pose.orientation.x),
            _finite_or_none(pose.orientation.y),
            _finite_or_none(pose.orientation.z),
            _finite_or_none(pose.orientation.w),
        ],
        **_twist_payload(message.twist.twist),
    }


def summarize_wrench(message: Any) -> dict[str, Any]:
    wrench = message.wrench
    return {
        "stamp_s": _stamp_seconds(message),
        "frame_id": str(message.header.frame_id),
        "force_xyz": [
            _finite_or_none(wrench.force.x),
            _finite_or_none(wrench.force.y),
            _finite_or_none(wrench.force.z),
        ],
        "torque_xyz": [
            _finite_or_none(wrench.torque.x),
            _finite_or_none(wrench.torque.y),
            _finite_or_none(wrench.torque.z),
        ],
    }


def summarize_laser_scan(message: Any) -> dict[str, Any]:
    ranges = [_finite_or_none(value) for value in message.ranges]
    finite_ranges = [value for value in ranges if value is not None]
    return {
        "stamp_s": _stamp_seconds(message),
        "frame_id": str(message.header.frame_id),
        "samples": len(ranges),
        "finite_samples": len(finite_ranges),
        "range_min": _finite_or_none(message.range_min),
        "range_max": _finite_or_none(message.range_max),
        "observed_min": min(finite_ranges) if finite_ranges else None,
        "observed_max": max(finite_ranges) if finite_ranges else None,
    }


def _load_topics(contract_path: Path) -> dict[str, list[str]]:
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    topics = payload["ros2_topics"]
    required = set(REQUIRED_STREAMS) | set(COMMAND_KEYS)
    missing = sorted(required - set(topics))
    if missing:
        raise ValueError(f"contract lacks required ROS topics: {missing}")
    return topics


def run_capture(contract_path: Path, output_path: Path, duration_s: float) -> dict[str, Any]:
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration must be finite and positive")

    import rclpy
    from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState, LaserScan

    topics = _load_topics(contract_path)
    message_specs: dict[str, tuple[type, Callable[[Any], dict[str, Any]], Any]] = {
        "base_pose": (PoseStamped, summarize_pose_stamped, 10),
        "base_twist": (TwistStamped, summarize_twist_stamped, 10),
        "base_odom": (Odometry, summarize_odometry, 10),
        "spine_state": (JointState, summarize_joint_state, 10),
        "left_arm_state": (JointState, summarize_joint_state, 10),
        "right_arm_state": (JointState, summarize_joint_state, 10),
        "right_external_joint_torques": (JointState, summarize_joint_state, 10),
        "right_external_wrench": (WrenchStamped, summarize_wrench, 10),
        "left_gripper_state": (JointState, summarize_joint_state, 10),
        "right_gripper_state": (JointState, summarize_joint_state, 10),
        "head_rgb": (Image, summarize_image, qos_profile_sensor_data),
        "right_wrist_rgb": (Image, summarize_image, qos_profile_sensor_data),
        "lidar_front": (LaserScan, summarize_laser_scan, qos_profile_sensor_data),
        "lidar_rear": (LaserScan, summarize_laser_scan, qos_profile_sensor_data),
    }

    class ReadOnlyPreflight(Node):
        def __init__(self) -> None:
            super().__init__("task2_real_read_only_preflight")
            self.samples: dict[str, dict[str, Any]] = {}
            self.counts = {key: 0 for key in REQUIRED_STREAMS}
            self._owned_subscription_refs = []
            for key, (message_type, summarizer, qos) in message_specs.items():
                topic = topics[key][0]
                subscription = self.create_subscription(
                    message_type,
                    topic,
                    lambda message, key=key, summarizer=summarizer: self._record(
                        key, summarizer(message)
                    ),
                    qos,
                )
                self._owned_subscription_refs.append(subscription)

        def _record(self, key: str, summary: dict[str, Any]) -> None:
            self.counts[key] += 1
            self.samples[key] = {
                **summary,
                "received_monotonic_s": time.monotonic(),
            }

        def publisher_counts(self) -> dict[str, int]:
            return {
                key: len(self.get_publishers_info_by_topic(topics[key][0]))
                for key in COMMAND_KEYS
            }

        def publisher_types(self) -> dict[str, list[str]]:
            return {
                key: sorted(
                    {
                        str(endpoint.topic_type)
                        for endpoint in self.get_publishers_info_by_topic(
                            topics[key][0]
                        )
                    }
                )
                for key in REQUIRED_STREAMS
            }

    rclpy.init()
    node: ReadOnlyPreflight | None = None
    try:
        node = ReadOnlyPreflight()
        started = time.monotonic()
        while rclpy.ok() and time.monotonic() - started < duration_s:
            rclpy.spin_once(node, timeout_sec=0.05)
        finished = time.monotonic()
        missing = [key for key in REQUIRED_STREAMS if node.counts[key] == 0]
        result = {
            "success": not missing,
            "scope": "read-only ROS interface evidence; not a handoff or task-success gate",
            "duration_s": finished - started,
            "missing_streams": missing,
            "message_counts": dict(node.counts),
            "last_samples": node.samples,
            "stream_publisher_types": node.publisher_types(),
            "command_publisher_counts": node.publisher_counts(),
            "command_publications": 0,
        }
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("contract.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=10.0)
    args = parser.parse_args()
    result = run_capture(args.contract, args.output, args.duration_s)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
