#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""ROS 2 live PI0.5 runner; shadow by default, simulator publication opt-in."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import signal
import statistics
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

from task2_isaacsim.baselines.pi05.contract import PI05_CONTRACT
from task2_isaacsim.baselines.pi05.live.core import (
    BaseReadinessGate,
    FreshnessConfig,
    ReadinessConfig,
    RunnerPhase,
    freshness_metrics,
    safe_action,
    validate_live_state,
    validate_rgb_frame,
)
from task2_isaacsim.baselines.pi05.live.policy import LivePi05Policy
from task2_isaacsim.common.state_contract import (
    GRIPPER_CLOSED_RAD,
    LEFT_GRIPPER_DRIVER,
    LEFT_JOINTS,
    RIGHT_GRIPPER_DRIVER,
    RIGHT_JOINTS,
    SPINE_JOINT,
    assemble_state,
)
from task2_isaacsim.scripts.topics import camera_topic, load_topics

TOPICS = load_topics()
CAMERA_ENTRIES = TOPICS["cameras"]["robot"]
COMMAND_ENTRIES = TOPICS["bridge"]["joint_groups"]


def _image_array(message: Image) -> np.ndarray:
    if message.encoding.lower() not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported RGB encoding: {message.encoding}")
    data = np.frombuffer(message.data, dtype=np.uint8)
    array = data.reshape(message.height, message.step)[:, : message.width * 3]
    array = array.reshape(message.height, message.width, 3)
    if message.encoding.lower() == "bgr8":
        array = array[:, :, ::-1]
    return np.ascontiguousarray(array)


class LiveObservationNode(Node):
    """Latest-value ROS cache; command publishers exist only when armed."""

    def __init__(self, *, publish: bool, gate: BaseReadinessGate):
        super().__init__("task2_pi05_live_runner")
        self.publish_enabled = publish
        self.gate = gate
        self.images: dict[str, np.ndarray] = {}
        self.image_times: dict[str, float] = {}
        self.image_sequences: dict[str, int] = {}
        self.joints: dict[str, float] = {}
        self.ee_poses: dict[str, tuple[float, ...] | None] = {
            "left": None,
            "right": None,
        }
        self.odom: tuple[float, ...] | None = None
        self.state_time = -math.inf
        self.reset_count = 0
        self.stop_requested = False
        self.base_input_count = 0
        self.publish_count = 0
        self._command_publishers: dict[str, object] = {}

        for key, entry in CAMERA_ENTRIES.items():
            topic = camera_topic(TOPICS, entry["namespace"], "image")
            self.create_subscription(
                Image,
                topic,
                lambda msg, key=key: self._on_image(key, msg),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            JointState,
            TOPICS["recording"]["joint_states_full"],
            self._on_joints,
            10,
        )
        self.create_subscription(
            Odometry, TOPICS["recording"]["odom"], self._on_odom, 10
        )
        self.create_subscription(
            Twist,
            TOPICS["recording"]["cmd_vel_applied"],
            self._on_base_command,
            10,
        )
        for side, topic in TOPICS["recording"]["ee_pose"].items():
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, side=side: self._on_ee(side, msg),
                10,
            )
        self.create_subscription(
            String,
            TOPICS["ground_truth"]["scene_reset"],
            self._on_reset,
            10,
        )

        if publish:
            conflicts = self.command_publisher_counts()
            if any(conflicts.values()):
                raise RuntimeError(
                    f"command publisher contention: {conflicts}"
                )
            for group, entry in COMMAND_ENTRIES.items():
                self._command_publishers[group] = self.create_publisher(
                    JointState, entry["command"], 10
                )

    def _on_image(self, key: str, message: Image) -> None:
        try:
            image = _image_array(message)
            validate_rgb_frame(key, image)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        self.images[key] = image
        self.image_times[key] = time.monotonic()
        self.image_sequences[key] = self.image_sequences.get(key, 0) + 1

    def _on_joints(self, message: JointState) -> None:
        self.joints = {
            str(name): float(position)
            for name, position in zip(message.name, message.position)
        }
        self.state_time = time.monotonic()

    def _on_ee(self, side: str, message: PoseStamped) -> None:
        pose = message.pose
        self.ee_poses[side] = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.state_time = time.monotonic()

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
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
        self.state_time = time.monotonic()

    def _on_base_command(self, message: Twist) -> None:
        velocity = (
            message.linear.x,
            message.linear.y,
            message.angular.z,
        )
        if any(abs(value) > 1e-6 for value in velocity):
            self.base_input_count += 1
            self.gate.note_base_input()

    def _on_reset(self, _message: String) -> None:
        self.reset_count += 1
        self.images.clear()
        self.image_times.clear()
        self.gate.reset()

    def snapshot(self) -> tuple[dict[str, np.ndarray], tuple[float, ...]]:
        state = assemble_state(
            {
                "ee_poses": self.ee_poses,
                "joint_states": self.joints,
                "odom": self.odom,
            }
        )
        return {key: value.copy() for key, value in self.images.items()}, state

    def update_readiness(self, now: float) -> bool:
        if self.odom is None or SPINE_JOINT not in self.joints:
            return False
        x, y, _, qx, qy, qz, qw, vx, vy, _, wz = self.odom
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return self.gate.update(
            now,
            (x, y, yaw),
            (vx, vy, wz),
            self.joints[SPINE_JOINT],
        )

    def command_publisher_counts(self) -> dict[str, int]:
        return {
            group: len(self.get_publishers_info_by_topic(entry["command"]))
            for group, entry in COMMAND_ENTRIES.items()
        }

    def publish_action(self, action: tuple[float, ...]) -> None:
        if not self.publish_enabled:
            raise RuntimeError("shadow runner cannot publish")
        groups = {
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
        for group, (names, positions) in groups.items():
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = list(names)
            message.position = list(positions)
            self._command_publishers[group].publish(message)
        self.publish_count += 4


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-repo-id", default="hermanprawiro/task2_fixpos_v1"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-target", nargs=3, type=float, required=True)
    parser.add_argument("--base-coordinate-frame", required=True)
    parser.add_argument("--confirm-fixed-staging", action="store_true")
    parser.add_argument("--arm-simulator", action="store_true")
    parser.add_argument("--warmup-decisions", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=5)
    parser.add_argument("--max-publish-actions", type=int, default=50)
    parser.add_argument("--queue-refill-actions", type=int, default=30)
    parser.add_argument("--max-duration-s", type=float, default=300.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.05)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.10)
    parser.add_argument("--velocity-threshold", type=float, default=0.02)
    parser.add_argument("--settle-duration-s", type=float, default=1.0)
    parser.add_argument("--spine-ready-height-m", type=float, default=0.4857)
    parser.add_argument("--spine-tolerance-m", type=float, default=0.015)
    parser.add_argument("--camera-max-age-s", type=float, default=0.25)
    parser.add_argument("--camera-max-skew-s", type=float, default=0.10)
    parser.add_argument("--state-max-age-s", type=float, default=0.10)
    parser.add_argument("--action-queue-max-age-s", type=float, default=2.0)
    parser.add_argument("--action-rate-hz", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_fixed_staging:
        print("FAIL: --confirm-fixed-staging is required")
        return 2
    if args.arm_simulator and args.max_publish_actions <= 0:
        print("FAIL: --max-publish-actions must be positive")
        return 2
    if not 0 < args.queue_refill_actions <= PI05_CONTRACT.chunk_size:
        print("FAIL: --queue-refill-actions must be within the policy chunk")
        return 2
    readiness = ReadinessConfig(
        *args.base_target,
        spine_target_m=args.spine_ready_height_m,
        spine_tolerance_m=args.spine_tolerance_m,
        position_tolerance_m=args.position_tolerance_m,
        yaw_tolerance_rad=args.yaw_tolerance_rad,
        velocity_threshold=args.velocity_threshold,
        settle_duration_s=args.settle_duration_s,
    )
    freshness = FreshnessConfig(
        camera_max_age_s=args.camera_max_age_s,
        camera_max_skew_s=args.camera_max_skew_s,
        state_max_age_s=args.state_max_age_s,
        action_queue_max_age_s=args.action_queue_max_age_s,
    )
    gate = BaseReadinessGate(readiness)
    gate.reset("startup")
    rclpy.init()
    try:
        node = LiveObservationNode(publish=args.arm_simulator, gate=gate)
    except Exception as error:
        rclpy.shutdown()
        print(f"FAIL: {error}")
        return 2
    signal.signal(
        signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True)
    )
    signal.signal(
        signal.SIGINT, lambda *_: setattr(node, "stop_requested", True)
    )
    mode = "simulator_publish" if args.arm_simulator else "shadow"
    print(f"Mode: {mode}")
    print(
        "Enabled command topics: "
        + (
            ", ".join(entry["command"] for entry in COMMAND_ENTRIES.values())
            if args.arm_simulator
            else "<none>"
        )
    )
    policy = LivePi05Policy(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        instruction=PI05_CONTRACT.task_instruction,
        seed=args.seed,
    )
    started = time.monotonic()
    last_sequences: dict[str, int] = {}
    last_reset_count = node.reset_count
    events: list[dict] = []
    warmup_latencies: list[float] = []
    latencies: list[float] = []
    invalid_actions = 0
    stale_observations = 0
    reset_recoveries = 0
    queue: deque[tuple[tuple[float, ...], float, int]] = deque()
    future: Future | None = None
    future_context: dict | None = None
    published_actions = 0
    next_publish_at = time.monotonic()
    publish_blocked: str | None = None
    manipulation_latched = False
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05")

    # Warm up CUDA/model kernels before steady-state measurements. No action is
    # published here and the same valid observation can be reused safely.
    warmup_images: dict[str, np.ndarray] | None = None
    warmup_state: tuple[float, ...] | None = None
    while (
        warmup_images is None
        and time.monotonic() - started < args.max_duration_s
        and not node.stop_requested
    ):
        rclpy.spin_once(node, timeout_sec=0.02)
        now = time.monotonic()
        if not node.update_readiness(now):
            continue
        try:
            warmup_images, warmup_state = node.snapshot()
            validate_live_state(warmup_state)
            freshness_metrics(
                now=now,
                camera_times=node.image_times,
                camera_sequences=node.image_sequences,
                state_time=node.state_time,
                last_camera_sequences={},
                config=freshness,
            )
        except ValueError:
            warmup_images = None
            stale_observations += 1
    if warmup_images is not None and warmup_state is not None:
        for _ in range(args.warmup_decisions):
            _, latency = policy.predict_chunk(
                images=warmup_images, state=warmup_state
            )
            warmup_latencies.append(latency)
        policy.reset()

    try:
        while (
            time.monotonic() - started < args.max_duration_s
            and not node.stop_requested
        ):
            rclpy.spin_once(node, timeout_sec=0.005)
            now = time.monotonic()
            if node.reset_count != last_reset_count:
                if args.arm_simulator and manipulation_latched:
                    gate.stop("scene_reset")
                    publish_blocked = "scene_reset"
                    queue.clear()
                    break
                policy.reset()
                queue.clear()
                last_sequences.clear()
                last_reset_count = node.reset_count
                reset_recoveries += 1
                continue

            ready = node.update_readiness(now)
            if args.arm_simulator and manipulation_latched and not ready:
                publish_blocked = str(
                    gate.last_evidence.get("reason", "readiness_revoked")
                )
                queue.clear()
                break

            if future is not None and future.done():
                validation_index = 0
                latency: float | None = None
                assert future_context is not None
                state = future_context["state"]
                try:
                    chunk, latency = future.result()
                    latencies.append(latency)
                    created_at = time.monotonic()
                    validated = []
                    for validation_index, action in enumerate(chunk):
                        validated.append(
                            safe_action(action, spine_hold=state[28])
                        )
                except (RuntimeError, ValueError) as error:
                    invalid_actions += 1
                    events.append(
                        {
                            "decision": len(events),
                            "valid": False,
                            "error": str(error),
                            "invalid_action_index": validation_index,
                            "inference_latency_s": latency,
                        }
                    )
                    publish_blocked = "invalid_policy_action"
                    if args.arm_simulator:
                        gate.stop(publish_blocked)
                    queue.clear()
                    future = None
                    future_context = None
                    break

                event_index = len(events)
                refill_window_s = (
                    args.queue_refill_actions / args.action_rate_hz
                )
                event = {
                    "decision": event_index,
                    "valid": True,
                    "fixed_staging": True,
                    "readiness": future_context["readiness"],
                    "freshness": future_context["freshness"],
                    "inference_latency_s": latency,
                    "queue_refill_window_s": refill_window_s,
                    "latency_within_refill_window": latency <= refill_window_s,
                    "raw_actions": [list(raw) for raw, _ in validated],
                    "effective_actions": [
                        list(effective) for _, effective in validated
                    ],
                    "published_actions": 0,
                }
                events.append(event)
                first_raw, first_effective = validated[0]
                print(
                    json.dumps(
                        {
                            "decision": event_index,
                            "latency_s": latency,
                            "raw_gripper": first_raw[17:19],
                            "effective_gripper": first_effective[17:19],
                            "first_arm_target": first_effective[3:17],
                        }
                    ),
                    flush=True,
                )
                if args.arm_simulator:
                    for _, effective in validated:
                        queue.append((effective, created_at, event_index))
                future = None
                future_context = None

            if args.arm_simulator and queue and now >= next_publish_at:
                if not ready:
                    gate.stop("readiness_revoked_before_publish")
                    publish_blocked = "readiness_revoked_before_publish"
                    queue.clear()
                    break
                try:
                    freshness_metrics(
                        now=now,
                        camera_times=node.image_times,
                        camera_sequences=node.image_sequences,
                        state_time=node.state_time,
                        last_camera_sequences={
                            key: sequence - 1
                            for key, sequence in node.image_sequences.items()
                        },
                        config=freshness,
                    )
                except ValueError:
                    stale_observations += 1
                    if published_actions > 0:
                        gate.stop("live_stream_stale")
                        publish_blocked = "live_stream_stale"
                        queue.clear()
                        break
                    else:
                        continue
                if gate.phase == RunnerPhase.MANIPULATION_READY:
                    gate.arm()
                    manipulation_latched = True
                effective, created_at, event_index = queue.popleft()
                if now - created_at > freshness.action_queue_max_age_s:
                    publish_blocked = "action_queue_watchdog"
                elif publish_blocked is None:
                    counts = node.command_publisher_counts()
                    if any(count > 1 for count in counts.values()):
                        publish_blocked = f"command_contention:{counts}"
                    else:
                        node.publish_action(effective)
                        published_actions += 1
                        events[event_index]["published_actions"] += 1
                        next_publish_at = now + 1.0 / args.action_rate_hz
                if publish_blocked:
                    queue.clear()
                    break
                if published_actions >= args.max_publish_actions:
                    queue.clear()
                    break

            decisions_started = len(events) + int(future is not None)
            needs_inference = (
                decisions_started < args.max_decisions
                and future is None
                and ready
                and (
                    not args.arm_simulator
                    or len(queue) <= args.queue_refill_actions
                )
            )
            if needs_inference:
                try:
                    images, state = node.snapshot()
                    validate_live_state(state)
                    metrics = freshness_metrics(
                        now=now,
                        camera_times=node.image_times,
                        camera_sequences=node.image_sequences,
                        state_time=node.state_time,
                        last_camera_sequences=last_sequences,
                        config=freshness,
                    )
                except ValueError:
                    stale_observations += 1
                else:
                    last_sequences = dict(node.image_sequences)
                    future_context = {
                        "state": state,
                        "readiness": dict(gate.last_evidence),
                        "freshness": metrics,
                    }
                    if args.arm_simulator and not manipulation_latched:
                        gate.arm()
                        manipulation_latched = True
                        future_context["readiness"] = dict(gate.last_evidence)
                    future = worker.submit(
                        policy.predict_chunk, images=images, state=state
                    )

            if not args.arm_simulator:
                if len(events) >= args.max_decisions and future is None:
                    break
            elif (
                len(events) >= args.max_decisions
                and future is None
                and not queue
            ):
                break
    finally:
        worker.shutdown(wait=True, cancel_futures=True)

    elapsed = time.monotonic() - started
    refill_window_s = args.queue_refill_actions / args.action_rate_hz
    latency_contract_pass = bool(latencies) and all(
        value <= refill_window_s for value in latencies
    )
    summary = {
        "schema_version": 2,
        "mode": mode,
        "fixed_staging": True,
        "manipulation_only": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "task_instruction": PI05_CONTRACT.task_instruction,
        "chunk_size": PI05_CONTRACT.chunk_size,
        "n_action_steps": PI05_CONTRACT.n_action_steps,
        "warmup_decisions": args.warmup_decisions,
        "warmup_latency_s": warmup_latencies,
        "seed": args.seed,
        "base_coordinate_frame": args.base_coordinate_frame,
        "readiness_config": readiness.__dict__,
        "freshness_config": freshness.__dict__,
        "final_readiness": gate.last_evidence,
        "decisions": len(events),
        "valid_decisions": sum(event.get("valid", False) for event in events),
        "invalid_actions": invalid_actions,
        "stale_observation_rejections": stale_observations,
        "reset_events": node.reset_count,
        "reset_recoveries": reset_recoveries,
        "command_publications": node.publish_count,
        "published_actions": published_actions,
        "publish_blocked": publish_blocked,
        "base_input_events": node.base_input_count,
        "elapsed_s": elapsed,
        "inference_latency_s": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "queue_refill_actions": args.queue_refill_actions,
        "queue_refill_window_s": refill_window_s,
        "latency_contract_pass": latency_contract_pass,
        "completed": invalid_actions == 0
        and not node.stop_requested
        and publish_blocked is None
        and (
            (
                not args.arm_simulator
                and len(events) == args.max_decisions
            )
            or (
                args.arm_simulator
                and bool(events)
                and published_actions == args.max_publish_actions
            )
        ),
        "ros_publication": args.arm_simulator and node.publish_count > 0,
        "events": events,
    }
    output = args.output_dir / "live_runner_manifest.json"
    _write_json(output, summary)
    node.destroy_node()
    rclpy.shutdown()
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "mode",
                    "decisions",
                    "valid_decisions",
                    "invalid_actions",
                    "command_publications",
                    "latency_contract_pass",
                    "completed",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if summary["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
