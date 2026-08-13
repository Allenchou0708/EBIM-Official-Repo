#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Replay one organizer Task 2 raw action trajectory over the ROS bridge."""

from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from task2_isaacsim.baselines.pi05.contract import (
    ACTION_SIZE,
    STATE_SIZE,
    validate_absolute_action_bounds,
)

FPS = 30.0
RIGHT_CLOSE_THRESHOLD = 0.5
SPINE_DEMO_MIN_M = 0.0
SPINE_DEMO_MAX_M = 0.6
READY_ARMS = (
    0.0,
    -0.7854,
    0.0,
    -2.3562,
    0.0,
    1.5708,
    0.7854,
) * 2


def _finite_vector(values: Sequence[float], width: int, name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if len(vector) != width:
        raise ValueError(f"{name} must contain {width} values, got {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _parquet_rows(dataset_root: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to read the organizer dataset"
        ) from error

    paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not paths:
        raise ValueError(f"no parquet files below {dataset_root / 'data'}")
    rows: list[dict[str, Any]] = []
    columns = ["index", "episode_index", "frame_index", "action", "observation.state"]
    for path in paths:
        rows.extend(parquet.read_table(path, columns=columns).to_pylist())
    return rows


def load_episode(
    dataset_root: Path,
    episode: int,
    *,
    row_reader: Callable[[Path], list[dict[str, Any]]] = _parquet_rows,
) -> list[dict[str, Any]]:
    chosen = [
        row for row in row_reader(dataset_root) if int(row["episode_index"]) == episode
    ]
    chosen.sort(key=lambda row: int(row["frame_index"]))
    if not chosen:
        raise ValueError(f"episode {episode} is absent from {dataset_root}")
    if [int(row["frame_index"]) for row in chosen] != list(range(len(chosen))):
        raise ValueError(f"episode {episode} frame indices are not contiguous")

    normalized: list[dict[str, Any]] = []
    for row in chosen:
        action = _finite_vector(row["action"], ACTION_SIZE, "raw action")
        state = _finite_vector(row["observation.state"], STATE_SIZE, "recorded state")
        validate_absolute_action_bounds(action)
        if not SPINE_DEMO_MIN_M <= action[19] <= SPINE_DEMO_MAX_M:
            raise ValueError(
                f"spine command {action[19]} is outside demonstrated "
                f"[{SPINE_DEMO_MIN_M}, {SPINE_DEMO_MAX_M}] m range"
            )
        normalized.append(
            {
                "index": int(row["index"]),
                "episode_index": episode,
                "frame_index": int(row["frame_index"]),
                "action": action,
                "state": state,
            }
        )
    return normalized


def select_episode(
    dataset_root: Path, audit_report: Path
) -> tuple[int, list[dict[str, Any]]]:
    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    train = {int(value) for value in audit["split"]["train"]}
    rows = _parquet_rows(dataset_root)
    frame_zero = {
        int(row["episode_index"]): row
        for row in rows
        if int(row["frame_index"]) == 0 and int(row["episode_index"]) in train
    }
    if set(frame_zero) != train:
        raise ValueError("train split and frame-0 episode set disagree")

    def score(item: tuple[int, dict[str, Any]]) -> tuple[float, int]:
        episode, row = item
        state = _finite_vector(row["observation.state"], STATE_SIZE, "frame-0 state")
        arm_l2 = math.sqrt(
            sum((state[14 + index] - READY_ARMS[index]) ** 2 for index in range(14))
        )
        base_pose_error = math.sqrt(
            (state[31] - 2.10) ** 2
            + (state[32] - 3.05) ** 2
            + (state[33] + 1.571) ** 2
        )
        return arm_l2 + abs(state[28]) + base_pose_error, episode

    selected = min(frame_zero.items(), key=score)[0]
    return selected, load_episode(dataset_root, selected)


def _first_at_least(values: Sequence[float], threshold: float) -> int | None:
    return next(
        (index for index, value in enumerate(values) if value >= threshold),
        None,
    )


def summarize_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [row["action"] for row in rows]
    states = [row["state"] for row in rows]
    arm_columns = list(zip(*(action[3:17] for action in actions), strict=True))
    extras_path = str(rows[0].get("extras_path", "")) or None
    return {
        "episode": int(rows[0]["episode_index"]),
        "frames": len(rows),
        "fps": FPS,
        "duration_s": len(rows) / FPS,
        "frame0_state": list(states[0]),
        "frame0_base_pose": list(states[0][31:34]),
        "frame0_arm_positions": list(states[0][14:28]),
        "frame0_grippers": list(states[0][29:31]),
        "frame0_spine_m": states[0][28],
        "spine_first_0_10_frame": _first_at_least(
            [action[19] for action in actions], 0.10
        ),
        "spine_first_0_30_frame": _first_at_least(
            [action[19] for action in actions], 0.30
        ),
        "recorded_spine_first_0_10_frame": _first_at_least(
            [state[28] for state in states], 0.10
        ),
        "recorded_spine_first_0_30_frame": _first_at_least(
            [state[28] for state in states], 0.30
        ),
        "right_gripper_first_close_frame": next(
            (
                index
                for index, action in enumerate(actions)
                if action[18] < RIGHT_CLOSE_THRESHOLD
            ),
            None,
        ),
        "arm_action_range": [
            {"minimum": min(values), "maximum": max(values)}
            for values in arm_columns
        ],
        "raw_actions": True,
        "mapped_relative_actions": False,
        "extras_path": extras_path,
    }


def _extras_summary(dataset_root: Path, episode: int) -> dict[str, Any]:
    path = dataset_root / "task2_extras" / f"episode_{episode:06d}.npz"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    try:
        import numpy as np

        with np.load(path) as payload:
            names = [str(value) for value in payload["object_names"].tolist()]
            shape = list(payload["object_poses"].shape)
    except (KeyError, OSError, ValueError) as error:
        return {"available": False, "path": str(path), "error": str(error)}
    required = {"board_target", "thermalpad", "thermalpad_base"}
    return {
        "available": required.issubset(names) and len(shape) == 3,
        "path": str(path),
        "object_names": names,
        "object_poses_shape": shape,
        "runtime_pose_injection_implemented": False,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _angular_error(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def run_ros_replay(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    align_only: bool,
    max_frames: int | None,
    alignment_timeout_s: float,
) -> dict[str, Any]:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64

    from task2_isaacsim.common.state_contract import (
        GRIPPER_CLOSED_RAD,
        LEFT_GRIPPER_DRIVER,
        LEFT_JOINTS,
        RIGHT_GRIPPER_DRIVER,
        RIGHT_JOINTS,
        SPINE_JOINT,
        assemble_state,
        finite_state,
    )
    from task2_isaacsim.scripts.topics import load_topics

    topics = load_topics()
    groups = topics["bridge"]["joint_groups"]

    class ReplayNode(Node):
        def __init__(self) -> None:
            super().__init__("task2_pi05_dataset_replay")
            self.joints: dict[str, float] = {}
            self.odom: tuple[float, ...] | None = None
            self.ee = {"left": None, "right": None}
            self.stop_requested = False
            self.create_subscription(
                JointState,
                topics["recording"]["joint_states_full"],
                self._on_joints,
                10,
            )
            self.create_subscription(
                Odometry, topics["recording"]["odom"], self._on_odom, 10
            )
            for side, topic in topics["recording"]["ee_pose"].items():
                self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda message, side=side: self._on_ee(side, message),
                    10,
                )
            command_topics = {
                **{name: entry["command"] for name, entry in groups.items()},
                "spine": topics["teleop"]["spine_target"],
            }
            conflicts = {
                name: len(self.get_publishers_info_by_topic(topic))
                for name, topic in command_topics.items()
            }
            if any(conflicts.values()):
                raise RuntimeError(
                    f"another trajectory publisher is active: {conflicts}"
                )
            self.command_publishers = {
                name: self.create_publisher(JointState, entry["command"], 10)
                for name, entry in groups.items()
            }
            self.spine_publisher = self.create_publisher(
                Float64, topics["teleop"]["spine_target"], 10
            )

        def _on_joints(self, message: JointState) -> None:
            self.joints = {
                str(name): float(value)
                for name, value in zip(message.name, message.position, strict=False)
            }

        def _on_odom(self, message: Odometry) -> None:
            pose, twist = message.pose.pose, message.twist.twist
            self.odom = (
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.z,
            )

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

        def state(self) -> tuple[float, ...] | None:
            try:
                return finite_state(
                    assemble_state(
                        {
                            "ee_poses": self.ee,
                            "joint_states": self.joints,
                            "odom": self.odom,
                        }
                    )
                )
            except ValueError:
                return None

        def alignment_state(self) -> tuple[float, ...] | None:
            """Return only the measured fields needed by the frame-0 gate."""

            state = assemble_state({"joint_states": self.joints, "odom": self.odom})
            if not all(math.isfinite(value) for value in state[14:34]):
                return None
            return state

        def publish_target(self, action: Sequence[float]) -> None:
            targets = {
                "left_arm": (LEFT_JOINTS, action[3:10]),
                "right_arm": (RIGHT_JOINTS, action[10:17]),
                "left_gripper": (
                    (LEFT_GRIPPER_DRIVER,),
                    ((1.0 - action[17]) * GRIPPER_CLOSED_RAD,),
                ),
                "right_gripper": (
                    (RIGHT_GRIPPER_DRIVER,),
                    ((1.0 - action[18]) * GRIPPER_CLOSED_RAD,),
                ),
            }
            for name, (joint_names, positions) in targets.items():
                message = JointState()
                message.header.stamp = self.get_clock().now().to_msg()
                message.name = list(joint_names)
                message.position = list(positions)
                self.command_publishers[name].publish(message)
            self.spine_publisher.publish(Float64(data=float(action[19])))

    rclpy.init()
    node = ReplayNode()
    signal.signal(signal.SIGINT, lambda *_: setattr(node, "stop_requested", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True))
    frame0 = rows[0]["state"]
    align_target = [0.0] * ACTION_SIZE
    align_target[3:17] = frame0[14:28]
    align_target[17:19] = frame0[29:31]
    align_target[19] = frame0[28]
    validate_absolute_action_bounds(align_target)
    alignment_started = time.monotonic()
    stable_since: float | None = None
    alignment: dict[str, Any] = {"success": False, "reason": "timeout"}
    while (
        not node.stop_requested
        and time.monotonic() - alignment_started < alignment_timeout_s
    ):
        rclpy.spin_once(node, timeout_sec=1.0 / FPS)
        node.publish_target(align_target)
        live = node.alignment_state()
        if live is None:
            continue
        arm_max = max(abs(live[14 + index] - frame0[14 + index]) for index in range(14))
        spine_error = abs(live[28] - frame0[28])
        gripper_error = max(
            abs(live[29 + index] - frame0[29 + index]) for index in range(2)
        )
        base_position_error = math.hypot(live[31] - frame0[31], live[32] - frame0[32])
        base_yaw_error = abs(_angular_error(live[33], frame0[33]))
        errors = {
            "arm_max_abs_rad": arm_max,
            "spine_abs_m": spine_error,
            "gripper_max_abs_fraction": gripper_error,
            "base_position_m": base_position_error,
            "base_yaw_rad": base_yaw_error,
        }
        within = (
            arm_max <= 0.03
            and spine_error <= 0.015
            and gripper_error <= 0.08
            and base_position_error <= 0.03
            and base_yaw_error <= 0.08
        )
        if not within:
            stable_since = None
        elif stable_since is None:
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= 1.0:
            alignment = {"success": True, "reason": "stable", "errors": errors}
            break
    if not alignment["success"]:
        alignment["last_state_available"] = node.alignment_state() is not None
        node.destroy_node()
        rclpy.shutdown()
        return {"alignment": alignment, "replay_started": False}
    if align_only:
        node.destroy_node()
        rclpy.shutdown()
        return {"alignment": alignment, "replay_started": False, "align_only": True}

    limit = len(rows) if max_frames is None else min(len(rows), max_frames)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for frame, row in enumerate(rows[:limit]):
        if node.stop_requested:
            break
        target_at = started + frame / FPS
        while time.monotonic() < target_at and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=min(0.005, target_at - time.monotonic()))
        node.publish_target(row["action"])
        rclpy.spin_once(node, timeout_sec=0.0)
        live = node.state()
        records.append(
            {
                "frame": frame,
                "wall_time_s": time.monotonic() - started,
                "reference_state": list(row["state"]),
                "live_state": list(live) if live is not None else None,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "live_state.jsonl"
    state_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    valid = [record for record in records if record["live_state"] is not None]
    left_errors = [
        math.sqrt(
            sum(
                (
                    record["live_state"][14 + i]
                    - record["reference_state"][14 + i]
                )
                ** 2
                for i in range(7)
            )
        )
        for record in valid
    ]
    right_errors = [
        math.sqrt(
            sum(
                (
                    record["live_state"][21 + i]
                    - record["reference_state"][21 + i]
                )
                ** 2
                for i in range(7)
            )
        )
        for record in valid
    ]
    spine_errors = [
        abs(record["live_state"][28] - record["reference_state"][28])
        for record in valid
    ]
    command_close = next(
        (
            index
            for index, row in enumerate(rows[:limit])
            if row["action"][18] < RIGHT_CLOSE_THRESHOLD
        ),
        None,
    )
    recorded_close = next(
        (
            index
            for index, row in enumerate(rows[:limit])
            if row["state"][30] < RIGHT_CLOSE_THRESHOLD
        ),
        None,
    )
    live_close = next(
        (
            record["frame"]
            for record in valid
            if record["live_state"][30] < RIGHT_CLOSE_THRESHOLD
        ),
        None,
    )

    def stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "mean": statistics.mean(values),
            "p95": ordered[round((len(ordered) - 1) * 0.95)],
            "maximum": max(values),
        }

    result = {
        "alignment": alignment,
        "replay_started": True,
        "requested_frames": limit,
        "published_frames": len(records),
        "interrupted": node.stop_requested,
        "elapsed_s": time.monotonic() - started,
        "command_publications": len(records) * 5,
        "base_command_publications": 0,
        "live_state_records": len(valid),
        "left_arm_l2_rad": stats(left_errors),
        "right_arm_l2_rad": stats(right_errors),
        "spine_abs_error_m": stats(spine_errors),
        "raw_command_right_close_frame": command_close,
        "recorded_right_close_frame": recorded_close,
        "live_right_close_frame": live_close,
        "live_vs_recorded_right_close_frame_offset": (
            None
            if recorded_close is None or live_close is None
            else live_close - recorded_close
        ),
        "live_vs_command_right_close_frame_offset": (
            None
            if command_close is None or live_close is None
            else live_close - command_close
        ),
        "recorded_spine_first_0_10_frame": _first_at_least(
            [row["state"][28] for row in rows[:limit]], 0.10
        ),
        "recorded_spine_first_0_30_frame": _first_at_least(
            [row["state"][28] for row in rows[:limit]], 0.30
        ),
        "live_spine_first_0_10_frame": _first_at_least(
            [record["live_state"][28] for record in valid], 0.10
        ),
        "live_spine_first_0_30_frame": _first_at_least(
            [record["live_state"][28] for record in valid], 0.30
        ),
        "live_state_path": str(state_path),
    }
    node.destroy_node()
    rclpy.shutdown()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", default="auto")
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--align-only", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--alignment-timeout-s", type=float, default=40.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.episode == "auto":
            if args.audit_report is None:
                raise ValueError("--audit-report is required with --episode auto")
            episode, rows = select_episode(args.dataset_root, args.audit_report)
            selection = "closest_train_frame0_to_live_ready_pose"
        else:
            episode = int(args.episode)
            rows = load_episode(args.dataset_root, episode)
            selection = "explicit"
        extras = _extras_summary(args.dataset_root, episode)
        for row in rows:
            row["extras_path"] = extras["path"] if extras["available"] else ""
        summary = {
            **summarize_trajectory(rows),
            "selection": selection,
            "object_pose_metadata": extras,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "trajectory_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if args.summary_only:
            return 0
        result = run_ros_replay(
            rows,
            output_dir=args.output_dir,
            align_only=args.align_only,
            max_frames=args.max_frames,
            alignment_timeout_s=args.alignment_timeout_s,
        )
        report = {**summary, **result}
        _write_json(args.output_dir / "replay_report.json", report)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["alignment"]["success"] else 2
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
