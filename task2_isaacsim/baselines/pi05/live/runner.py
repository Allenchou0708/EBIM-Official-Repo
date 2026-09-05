#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""ROS 2 live PI0.5 runner; shadow by default, simulator publication opt-in."""

from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, String

from task2_isaacsim.baselines.pi05.contract import PI05_CONTRACT
from task2_isaacsim.baselines.pi05.live.core import (
    BaseReadinessGate,
    FreshnessConfig,
    FreshnessError,
    QueuedAction,
    ReadinessConfig,
    RightGraspGuard,
    RunnerPhase,
    align_action_chunk,
    apply_right_only_policy_ownership,
    freshness_metrics,
    gripper_open_fraction_command,
    hard5_action_window,
    hard5_hold_action,
    policy_command_topics,
    project_fr3_joint_step,
    replace_action_queue,
    right_ee_within_demonstrated_workspace,
    right_wrist_grasp_evidence_within_development_envelope,
    safe_action,
    startup_inventory,
    validate_live_state,
    validate_rgb_frame,
)
from task2_isaacsim.baselines.pi05.live.policy import LivePi05Policy
from task2_isaacsim.baselines.pi05.live.fixed_stage_observation import (
    RMPFLOW_STAGE_PLAN,
    load_reference as load_observation_reference,
)
from task2_isaacsim.baselines.pi05.live.language_gt import (
    LANGUAGE_GT_MAX_WINDOW,
    LANGUAGE_GT_TOKENIZER_MAX_LENGTH,
    SOURCE_END_EXCLUSIVE,
    SOURCE_EPISODE,
    SOURCE_START_FRAME,
    format_language_gt_window,
    load_episode_19_actions,
)
from task2_isaacsim.baselines.pi05.live.staging import validate_staging_audit
from task2_isaacsim.common.state_contract import (
    LEFT_GRIPPER_DRIVER,
    LEFT_JOINTS,
    RIGHT_GRIPPER_DRIVER,
    RIGHT_JOINTS,
    SPINE_JOINT,
    assemble_state,
    gripper_open_fraction,
    resolve_joint,
)
from task2_isaacsim.scripts.topics import camera_topic, load_topics

TOPICS = load_topics()
CAMERA_ENTRIES = TOPICS["cameras"]["robot"]
COMMAND_ENTRIES = TOPICS["bridge"]["joint_groups"]
SPINE_COMMAND_TOPIC = TOPICS["teleop"]["spine_target"]
POLICY_COMMAND_TOPICS = policy_command_topics(TOPICS)
EE_TARGET_TOPICS = TOPICS["cartesian_control"]["ee_target"]


def _stamp_seconds(message: object) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _image_array(message: Image) -> np.ndarray:
    if message.encoding.lower() not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported RGB encoding: {message.encoding}")
    data = np.frombuffer(message.data, dtype=np.uint8)
    array = data.reshape(message.height, message.step)[:, : message.width * 3]
    array = array.reshape(message.height, message.width, 3)
    if message.encoding.lower() == "bgr8":
        array = array[:, :, ::-1]
    return np.ascontiguousarray(array)


def _right_wrist_pad_signature(image: np.ndarray) -> dict[str, float | int]:
    """Return resolution-independent blue-pad geometry in the robot camera."""

    rgb = np.asarray(image, dtype=np.uint8)
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    mask = (blue >= 90) & (blue - red >= 35) & (blue - green >= 15)
    rows, columns = np.nonzero(mask)
    if len(rows) < 100:
        return {"pixel_count": int(len(rows))}
    height, width = mask.shape
    area_fraction = float(len(rows) / mask.size)
    return {
        "pixel_count": int(len(rows)),
        "centroid_u_fraction": float(np.median(columns) / width),
        "centroid_v_fraction": float(np.median(rows) / height),
        "log_area_fraction": math.log(area_fraction),
    }


class LiveObservationNode(Node):
    """Latest-value ROS cache; command publishers exist only when armed."""

    def __init__(self, *, publish: bool, gate: BaseReadinessGate):
        super().__init__("task2_pi05_live_runner")
        self.publish_enabled = publish
        self.gate = gate
        self.images: dict[str, np.ndarray] = {}
        self.image_times: dict[str, float] = {}
        self.image_capture_times: dict[str, float] = {}
        self.image_sequences: dict[str, int] = {}
        self.joints: dict[str, float] = {}
        self.ee_poses: dict[str, tuple[float, ...] | None] = {
            "left": None,
            "right": None,
        }
        self.odom: tuple[float, ...] | None = None
        self.state_time = -math.inf
        self.state_times: dict[str, float] = {}
        self.state_capture_times: dict[str, float] = {}
        self.sim_time: float | None = None
        self.sim_clock_time = -math.inf
        self.reset_count = 0
        self.stop_requested = False
        self.base_input_count = 0
        self.base_input_after_manipulation_count = 0
        self.publish_count = 0
        self.ee_handoff_publish_count = 0
        self._command_publishers: dict[str, object] = {}
        self._ee_handoff_publishers: dict[str, object] = {}

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
        self.create_subscription(
            Clock,
            TOPICS["clock"],
            self._on_clock,
            qos_profile_sensor_data,
        )

        if publish:
            self.activate_publishers()

    def activate_publishers(self) -> None:
        """Create policy publishers only after the staging process exits."""

        if self._command_publishers:
            self.publish_enabled = True
            return
        conflicts = self.command_publisher_counts()
        if any(conflicts.values()):
            raise RuntimeError(f"command publisher contention: {conflicts}")
        for group, entry in COMMAND_ENTRIES.items():
            self._command_publishers[group] = self.create_publisher(
                JointState, entry["command"], 10
            )
        self._command_publishers["spine"] = self.create_publisher(
            Float64, SPINE_COMMAND_TOPIC, 10
        )
        for side, topic in EE_TARGET_TOPICS.items():
            self._ee_handoff_publishers[side] = self.create_publisher(
                PoseStamped, topic, 10
            )
        self.publish_enabled = True

    def deactivate_publishers(self) -> None:
        """Release command ownership before a subprocess takes over."""

        for publisher in self._command_publishers.values():
            self.destroy_publisher(publisher)
        for publisher in self._ee_handoff_publishers.values():
            self.destroy_publisher(publisher)
        self._command_publishers.clear()
        self._ee_handoff_publishers.clear()
        self.publish_enabled = False

    def _on_image(self, key: str, message: Image) -> None:
        received_at = time.monotonic()
        try:
            image = _image_array(message)
            validate_rgb_frame(key, image)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        self.images[key] = image
        self.image_times[key] = received_at
        self.image_capture_times[key] = _stamp_seconds(message)
        self.image_sequences[key] = self.image_sequences.get(key, 0) + 1

    def _on_joints(self, message: JointState) -> None:
        self.joints = {
            str(name): float(position)
            for name, position in zip(message.name, message.position)
        }
        self._record_state_time("joints", message)

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
        self._record_state_time(f"ee_{side}", message)

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
        self._record_state_time("odom", message)

    def _record_state_time(self, key: str, message: object) -> None:
        self.state_times[key] = time.monotonic()
        self.state_capture_times[key] = _stamp_seconds(message)
        # Freshness must describe the oldest component used by assemble_state,
        # not whichever ROS callback happened to arrive last.
        self.state_time = min(self.state_times.values())

    def _on_base_command(self, message: Twist) -> None:
        velocity = (
            message.linear.x,
            message.linear.y,
            message.angular.z,
        )
        if any(abs(value) > 1e-6 for value in velocity):
            self.base_input_count += 1
            if self.gate.phase == RunnerPhase.PI05_MANIPULATION:
                self.base_input_after_manipulation_count += 1
            self.gate.note_base_input()

    def _on_reset(self, _message: String) -> None:
        self.reset_count += 1
        self.images.clear()
        self.image_times.clear()
        self.image_capture_times.clear()
        self.state_times.clear()
        self.state_capture_times.clear()
        self.state_time = -math.inf
        self.gate.reset()

    def _on_clock(self, message: Clock) -> None:
        stamp = message.clock
        self.sim_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        self.sim_clock_time = time.monotonic()

    def snapshot(self) -> tuple[dict[str, np.ndarray], tuple[float, ...]]:
        state = assemble_state(
            {
                "ee_poses": self.ee_poses,
                "joint_states": self.joints,
                "odom": self.odom,
            }
        )
        return {key: value.copy() for key, value in self.images.items()}, state

    def discard_staging_observations(self) -> None:
        """Require fresh samples after pre-manipulation staging."""

        self.images.clear()
        self.image_times.clear()
        self.image_capture_times.clear()
        self.image_sequences.clear()
        self.state_times.clear()
        self.state_capture_times.clear()
        self.state_time = -math.inf

    def startup_status(self) -> dict[str, object]:
        _images, state = self.snapshot()
        status = startup_inventory(
            camera_sequences=self.image_sequences,
            joint_names=set(self.joints),
            ee_available={
                side: pose is not None for side, pose in self.ee_poses.items()
            },
            odom_available=self.odom is not None,
            state=state,
        )
        status["sim_clock_sample"] = self.sim_time is not None
        if self.sim_time is None:
            status["all_required_samples"] = False
            status["missing_inputs"] = [
                *status["missing_inputs"],
                TOPICS["clock"],
            ]
        return status

    def update_readiness(self, now: float) -> bool:
        if self.odom is None or SPINE_JOINT not in self.joints:
            self.gate.last_evidence = {
                "ready": False,
                "reason": "startup_inputs_missing",
                **self.startup_status(),
            }
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
            group: len(self.get_publishers_info_by_topic(topic))
            for group, topic in POLICY_COMMAND_TOPICS.items()
        }

    def publish_action(self, action: tuple[float, ...]) -> None:
        if not self.publish_enabled:
            raise RuntimeError("shadow runner cannot publish")
        if self.sim_time is None:
            raise RuntimeError("simulator clock unavailable for command stamp")
        command_stamp = Time(
            nanoseconds=max(int(self.sim_time * 1e9), 0)
        ).to_msg()
        groups = {
            "left_arm": (LEFT_JOINTS, action[3:10]),
            "right_arm": (RIGHT_JOINTS, action[10:17]),
            "left_gripper": (
                (LEFT_GRIPPER_DRIVER,),
                gripper_open_fraction_command(action[17]),
            ),
            "right_gripper": (
                (RIGHT_GRIPPER_DRIVER,),
                gripper_open_fraction_command(action[18]),
            ),
        }
        for group, (names, positions) in groups.items():
            message = JointState()
            message.header.stamp = command_stamp
            message.name = list(names)
            message.position = list(positions)
            self._command_publishers[group].publish(message)
        self._command_publishers["spine"].publish(Float64(data=action[19]))
        self.publish_count += 5

    def publish_ee_handoff_target(
        self, side: str, pose_world_xyzw: tuple[float, ...]
    ) -> None:
        """Latch RMPflow at the measured pose after joint-command ownership."""

        if not self.publish_enabled or self.sim_time is None:
            raise RuntimeError("cannot publish EE handoff without live control")
        if side not in self._ee_handoff_publishers:
            raise ValueError(f"unsupported EE handoff side: {side}")
        if len(pose_world_xyzw) != 7 or not all(
            math.isfinite(value) for value in pose_world_xyzw
        ):
            raise ValueError("EE handoff pose must contain seven finite values")
        x, y, z, qx, qy, qz, qw = pose_world_xyzw
        message = PoseStamped()
        message.header.stamp = Time(
            nanoseconds=max(int(self.sim_time * 1e9), 0)
        ).to_msg()
        message.header.frame_id = "world"
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.position.z = z
        message.pose.orientation.x = qx
        message.pose.orientation.y = qy
        message.pose.orientation.z = qz
        message.pose.orientation.w = qw
        self._ee_handoff_publishers[side].publish(message)
        self.ee_handoff_publish_count += 1


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_ppm(path: Path, image: np.ndarray) -> None:
    """Persist an RGB gate frame without adding an image-codec dependency."""

    height, width, channels = image.shape
    if channels != 3 or image.dtype != np.uint8:
        raise ValueError("PPM evidence requires uint8 RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        stream.write(image.tobytes())


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
    parser.add_argument("--confirm-fixed-base-staging", action="store_true")
    parser.add_argument("--stage-base-after-policy-load", action="store_true")
    parser.add_argument(
        "--restage-base-after-manipulation", action="store_true"
    )
    parser.add_argument("--stage-spine-after-base", action="store_true")
    parser.add_argument("--stage-manipulation-after-base", action="store_true")
    parser.add_argument("--staging-audit", type=Path)
    parser.add_argument(
        "--staging-mode",
        choices=("dataset_replay", "rmpflow_observation"),
        default="dataset_replay",
    )
    parser.add_argument("--observation-reference", type=Path)
    parser.add_argument(
        "--confirm-right-wrist-pad-visible", action="store_true"
    )
    parser.add_argument(
        "--base-stage-max-duration-s", type=float, default=90.0
    )
    parser.add_argument(
        "--spine-stage-max-duration-s", type=float, default=90.0
    )
    parser.add_argument(
        "--manipulation-stage-max-duration-s", type=float, default=300.0
    )
    parser.add_argument("--arm-simulator", action="store_true")
    parser.add_argument("--right-only-policy-after-staging", action="store_true")
    parser.add_argument("--whole-body-policy-after-staging", action="store_true")
    parser.add_argument("--hybrid-rmpflow-transport", action="store_true")
    parser.add_argument("--code-policy-acquire", action="store_true")
    parser.add_argument(
        "--with-language-gt",
        "--with_language_gt",
        action="store_true",
        help="condition PI0.5 on rolling numeric episode-19 GT actions",
    )
    parser.add_argument(
        "--runtime-mode", choices=("legacy", "hard5"), default="hard5"
    )
    parser.add_argument("--warmup-decisions", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=5)
    parser.add_argument("--max-publish-actions", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=5)
    parser.add_argument("--queue-refill-actions", type=int, default=24)
    parser.add_argument("--max-duration-s", type=float, default=300.0)
    parser.add_argument("--startup-timeout-s", type=float, default=10.0)
    parser.add_argument("--startup-retries", type=int, default=1)
    parser.add_argument("--position-tolerance-m", type=float, default=0.05)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.10)
    # Odom reports linear m/s and angular rad/s in one legacy norm.  The GUI
    # bridge shows a stable-pose angular floor around 0.022 rad/s, so retain a
    # conservative 0.03 threshold while position/yaw gates remain unchanged.
    parser.add_argument("--velocity-threshold", type=float, default=0.03)
    parser.add_argument("--settle-duration-s", type=float, default=1.0)
    parser.add_argument("--camera-max-age-s", type=float, default=0.65)
    parser.add_argument("--camera-max-skew-s", type=float, default=0.10)
    parser.add_argument("--state-max-age-s", type=float, default=0.15)
    parser.add_argument("--observation-max-skew-s", type=float, default=0.10)
    parser.add_argument("--action-queue-max-age-s", type=float, default=2.0)
    parser.add_argument("--action-rate-hz", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> int:  # noqa: C901 - one bounded live control loop
    args = parse_args()
    if (
        args.right_only_policy_after_staging
        and args.whole_body_policy_after_staging
    ):
        print("FAIL: choose exactly one post-staging policy ownership mode")
        return 2
    if args.hybrid_rmpflow_transport and (
        not args.arm_simulator or not args.right_only_policy_after_staging
    ):
        print(
            "FAIL: --hybrid-rmpflow-transport requires simulator publication "
            "with right-only policy ownership"
        )
        return 2
    if args.code_policy_acquire and not args.hybrid_rmpflow_transport:
        print("FAIL: --code-policy-acquire requires hybrid RMPflow transport")
        return 2
    if args.code_policy_acquire and args.with_language_gt:
        print("FAIL: --with-language-gt requires PI0.5 inference")
        return 2
    if not args.confirm_fixed_base_staging:
        print("FAIL: --confirm-fixed-base-staging is required")
        return 2
    if args.arm_simulator and args.max_publish_actions <= 0:
        print("FAIL: --max-publish-actions must be positive")
        return 2
    if not args.stage_manipulation_after_base:
        print(
            "FAIL: --stage-manipulation-after-base is required"
        )
        return 2
    if args.staging_mode == "dataset_replay" and args.staging_audit is None:
        print("FAIL: --staging-audit is required for dataset_replay staging")
        return 2
    if (
        args.staging_mode == "rmpflow_observation"
        and args.observation_reference is None
    ):
        print(
            "FAIL: --observation-reference is required for "
            "rmpflow_observation staging"
        )
        return 2
    if args.staging_mode == "rmpflow_observation" and (
        not args.stage_base_after_policy_load
        or not args.stage_spine_after_base
    ):
        print(
            "FAIL: RMPflow staging requires base then spine staging before "
            "the post-spine base recheck"
        )
        return 2
    if (
        args.staging_mode == "rmpflow_observation"
        and args.position_tolerance_m > 0.020
    ):
        print("FAIL: RMPflow staging requires a base tolerance <= 0.020 m")
        return 2
    if args.staging_mode == "rmpflow_observation" and args.restage_base_after_manipulation:
        print("FAIL: base restaging is incompatible with the latched RMPflow base hold")
        return 2
    if args.arm_simulator and not args.confirm_right_wrist_pad_visible:
        print(
            "FAIL: formal publication requires "
            "--confirm-right-wrist-pad-visible after shadow evidence review"
        )
        return 2
    try:
        if args.staging_mode == "dataset_replay":
            assert args.staging_audit is not None
            staging_audit = validate_staging_audit(
                json.loads(args.staging_audit.read_text(encoding="utf-8"))
            )
            staged_spine_command = float(
                staging_audit["final_target"]["spine_command_m"]
            )
            staged_spine_target = float(
                staging_audit["final_target"]["measured_reference"]["spine_m"]
            )
            staged_spine_tolerance = float(
                staging_audit["tolerances"]["spine_abs_m"]
            )
            deterministic_hold_command: tuple[float, ...] | None = tuple(
                float(value)
                for value in staging_audit["trajectory"][-1]["command"]
            )
            observation_reference = None
        else:
            assert args.observation_reference is not None
            observation_reference = load_observation_reference(
                args.observation_reference
            )
            staging_audit = None
            staged_spine_command = float(
                observation_reference["spine_command_m"]
            )
            staged_spine_target = float(
                observation_reference["spine_measured_target_m"]
            )
            staged_spine_tolerance = float(
                observation_reference["spine_tolerance_m"]
            )
            deterministic_hold_command = None
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"FAIL: invalid staging configuration: {error}")
        return 2
    if not 0 < args.queue_refill_actions <= PI05_CONTRACT.chunk_size:
        print("FAIL: --queue-refill-actions must be within the policy chunk")
        return 2
    if not 0 < args.execution_horizon <= PI05_CONTRACT.chunk_size:
        print("FAIL: --execution-horizon must be within the policy chunk")
        return 2
    if args.with_language_gt and args.runtime_mode != "hard5":
        print("FAIL: --with-language-gt requires hard5 execution")
        return 2
    if args.with_language_gt and args.execution_horizon > LANGUAGE_GT_MAX_WINDOW:
        print(
            "FAIL: --with-language-gt execution horizon must not exceed "
            f"{LANGUAGE_GT_MAX_WINDOW}"
        )
        return 2
    language_gt_trajectory = None
    if args.with_language_gt:
        try:
            language_gt_trajectory = load_episode_19_actions(args.dataset_root)
        except (OSError, TypeError, ValueError) as error:
            print(f"FAIL: cannot load numeric language GT: {error}")
            return 2
        if args.max_publish_actions > len(language_gt_trajectory):
            print(
                "FAIL: --with-language-gt cannot publish beyond its "
                f"{len(language_gt_trajectory)} GT frames"
            )
            return 2
    if args.startup_timeout_s <= 0.0 or args.startup_retries < 0:
        print("FAIL: startup timeout must be positive and retries non-negative")
        return 2
    readiness = ReadinessConfig(
        *args.base_target,
        initial_spine_target_m=staged_spine_target,
        initial_spine_max_abs_m=staged_spine_tolerance,
        position_tolerance_m=args.position_tolerance_m,
        yaw_tolerance_rad=args.yaw_tolerance_rad,
        velocity_threshold=args.velocity_threshold,
        settle_duration_s=args.settle_duration_s,
    )
    freshness = FreshnessConfig(
        camera_max_age_s=args.camera_max_age_s,
        camera_max_skew_s=args.camera_max_skew_s,
        state_max_age_s=args.state_max_age_s,
        observation_max_skew_s=args.observation_max_skew_s,
        action_queue_max_age_s=args.action_queue_max_age_s,
    )
    gate = BaseReadinessGate(readiness)
    gate.reset("startup")
    rclpy.init()
    try:
        node = LiveObservationNode(publish=False, gate=gate)
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
    hard5 = args.runtime_mode == "hard5"
    print(f"Mode: {mode}")
    print(f"Runtime mode: {args.runtime_mode}")
    print(
        "Enabled command topics: "
        + (
            ", ".join(POLICY_COMMAND_TOPICS.values())
            if args.arm_simulator
            else "<none>"
        )
    )
    startup_attempts: list[dict[str, object]] = []
    startup_status = node.startup_status()
    for attempt in range(args.startup_retries + 1):
        attempt_started = time.monotonic()
        while (
            time.monotonic() - attempt_started < args.startup_timeout_s
            and not node.stop_requested
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.sim_time is not None:
                node.update_readiness(node.sim_time)
            startup_status = node.startup_status()
            if startup_status["all_required_samples"]:
                break
        attempt_result = {
            "attempt": attempt + 1,
            "elapsed_s": time.monotonic() - attempt_started,
            **startup_status,
        }
        startup_attempts.append(attempt_result)
        print(json.dumps({"startup": attempt_result}, sort_keys=True), flush=True)
        if startup_status["all_required_samples"] or node.stop_requested:
            break
        if attempt < args.startup_retries:
            node.destroy_node()
            rclpy.shutdown()
            time.sleep(0.5)
            gate.reset("dds_participant_retry")
            rclpy.init()
            node = LiveObservationNode(publish=False, gate=gate)

    if not startup_status["all_required_samples"]:
        failure = {
            "schema_version": 5,
            "mode": mode,
            "completed": False,
            "decisions": 0,
            "valid_decisions": 0,
            "command_publications": 0,
            "published_actions": 0,
            "publish_blocked": "startup_inputs_missing",
            "startup": {
                "reason": "startup_inputs_missing",
                "attempts": startup_attempts,
                "final": startup_status,
            },
            "final_readiness": dict(gate.last_evidence),
        }
        _write_json(args.output_dir / "live_runner_manifest.json", failure)
        node.destroy_node()
        rclpy.shutdown()
        print(json.dumps(failure["startup"], sort_keys=True), flush=True)
        return 2

    language_gt_initial_frames: tuple[int, ...] = ()
    if language_gt_trajectory is not None:
        task_instruction, language_gt_initial_frames = format_language_gt_window(
            language_gt_trajectory,
            executed_actions=0,
            window=args.execution_horizon,
        )
    else:
        task_instruction = PI05_CONTRACT.task_instruction
    print(
        json.dumps(
            {
                "startup": {
                    "with_language_gt": args.with_language_gt,
                    "language_gt_source_episode": (
                        SOURCE_EPISODE if args.with_language_gt else None
                    ),
                    "language_gt_frames": list(language_gt_initial_frames),
                    "task_instruction": task_instruction,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    policy: LivePi05Policy | None = None
    policy_load_s = 0.0
    if not args.code_policy_acquire:
        policy_load_started = time.monotonic()
        policy = LivePi05Policy(
            checkpoint=args.checkpoint,
            instruction=task_instruction,
            seed=args.seed,
            tokenizer_max_length=(
                LANGUAGE_GT_TOKENIZER_MAX_LENGTH
                if args.with_language_gt
                else None
            ),
        )
        if args.with_language_gt and policy.action_size != 8:
            print("FAIL: numeric language GT requires the 8-D right-only policy")
            node.destroy_node()
            rclpy.shutdown()
            return 2
        if args.whole_body_policy_after_staging and policy.action_size != 20:
            print("FAIL: whole-body policy ownership requires a 20-D checkpoint")
            node.destroy_node()
            rclpy.shutdown()
            return 2
        policy_load_s = time.monotonic() - policy_load_started
    print(
        json.dumps({"startup": {"policy_loaded_s": policy_load_s}}),
        flush=True,
    )
    base_stage_s: float | None = None
    base_stage_output: Path | None = None
    if args.stage_base_after_policy_load:
        base_stage_settle_duration_s = args.settle_duration_s
        if args.staging_mode == "rmpflow_observation":
            # The mobile base is not pose-held until the first Cartesian
            # target latches the simulator-side base hold.  A long unheld
            # dwell can coast back across the position threshold in GUI
            # runs; the stricter post-spine recheck and RMPflow readiness
            # gate still run before any arm motion.
            base_stage_settle_duration_s = min(
                base_stage_settle_duration_s, 0.10
            )
        base_stage_output = (
            args.output_dir / "fixed_base_after_policy_load.json"
        )
        base_stage_started = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "task2_isaacsim.baselines.pi05.live.fixed_stage_base",
                "--target",
                *(str(value) for value in args.base_target),
                "--position-tolerance-m",
                str(args.position_tolerance_m),
                "--yaw-tolerance-rad",
                str(args.yaw_tolerance_rad),
                "--velocity-threshold",
                str(args.velocity_threshold),
                "--settle-duration-s",
                str(base_stage_settle_duration_s),
                "--max-duration-s",
                str(args.base_stage_max_duration_s),
                "--output",
                str(base_stage_output),
            ],
            check=False,
        )
        base_stage_s = time.monotonic() - base_stage_started
        if result.returncode != 0:
            failure = {
                "schema_version": 6,
                "mode": mode,
                "completed": False,
                "decisions": 0,
                "valid_decisions": 0,
                "command_publications": 0,
                "published_actions": 0,
                "publish_blocked": "base_stage_after_policy_load_failed",
                "startup_timing": {
                    "policy_load_s": policy_load_s,
                    "base_stage_s": base_stage_s,
                    "base_stage_order": "after_policy_load",
                    "base_stage_output": str(base_stage_output),
                },
            }
            _write_json(
                args.output_dir / "live_runner_manifest.json", failure
            )
            node.destroy_node()
            rclpy.shutdown()
            print("FAIL: base staging after policy load failed", flush=True)
            return 2
    spine_stage_s: float | None = None
    spine_stage_output: Path | None = None
    if args.stage_spine_after_base:
        spine_stage_output = args.output_dir / "fixed_spine_after_base.json"
        spine_stage_started = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "task2_isaacsim.baselines.pi05.live.fixed_stage_spine",
                "--target-m",
                str(staged_spine_command),
                "--measured-target-m",
                str(staged_spine_target),
                "--tolerance-m",
                str(staged_spine_tolerance),
                "--max-duration-s",
                str(args.spine_stage_max_duration_s),
                "--output",
                str(spine_stage_output),
            ],
            check=False,
        )
        spine_stage_s = time.monotonic() - spine_stage_started
        spine_stage_result: dict[str, object] = {}
        if spine_stage_output.is_file():
            spine_stage_result = json.loads(
                spine_stage_output.read_text(encoding="utf-8")
            )
        if result.returncode != 0 or spine_stage_result.get("success") is not True:
            failure = {
                "schema_version": 9,
                "mode": mode,
                "completed": False,
                "decisions": 0,
                "valid_decisions": 0,
                "command_publications": 0,
                "published_actions": 0,
                "publish_blocked": "spine_stage_after_base_failed",
                "spine_stage": spine_stage_result,
                "startup_timing": {
                    "policy_load_s": policy_load_s,
                    "base_stage_s": base_stage_s,
                    "spine_stage_s": spine_stage_s,
                },
            }
            _write_json(args.output_dir / "live_runner_manifest.json", failure)
            node.destroy_node()
            rclpy.shutdown()
            print("FAIL: spine staging after base failed", flush=True)
            return 2
    post_spine_base_stage_s: float | None = None
    post_spine_base_stage_output: Path | None = None
    if args.staging_mode == "rmpflow_observation":
        post_spine_base_stage_output = (
            args.output_dir / "fixed_base_after_spine.json"
        )
        # GUI rendering lowers the real-time factor enough that the wheeled
        # base coasts about 1.5 mm during the stable dwell.  A 10 mm control
        # threshold therefore chatters across its boundary even though the
        # downstream RMPflow readiness gate is 15 mm.  Reuse that exact gate:
        # stricter 12 and 14 mm inner gates both chattered indefinitely just
        # outside their boundary despite satisfying the formal readiness
        # contract and stable dwell requirement.
        post_spine_control_position_tolerance_m = args.position_tolerance_m
        post_spine_base_started = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "task2_isaacsim.baselines.pi05.live.fixed_stage_base",
                "--target",
                *(str(value) for value in args.base_target),
                "--position-tolerance-m",
                str(post_spine_control_position_tolerance_m),
                "--yaw-tolerance-rad",
                str(args.yaw_tolerance_rad),
                "--velocity-threshold",
                str(args.velocity_threshold),
                "--settle-duration-s",
                # Finish the corrective pulse with the same measured settle
                # dwell as the initial route.  Exiting while the base is still
                # coasting can move it back outside the 15 mm gate before the
                # RMPflow process has published its first base-hold target.
                str(args.settle_duration_s),
                "--max-duration-s",
                str(args.base_stage_max_duration_s),
                "--output",
                str(post_spine_base_stage_output),
            ],
            check=False,
        )
        post_spine_base_stage_s = time.monotonic() - post_spine_base_started
        post_spine_base_result: dict[str, object] = {}
        if post_spine_base_stage_output.is_file():
            post_spine_base_result = json.loads(
                post_spine_base_stage_output.read_text(encoding="utf-8")
            )
        if (
            result.returncode != 0
            or post_spine_base_result.get("success") is not True
        ):
            failure = {
                "schema_version": 10,
                "mode": mode,
                "completed": False,
                "decisions": 0,
                "valid_decisions": 0,
                "command_publications": 0,
                "published_actions": 0,
                "publish_blocked": "post_spine_base_recheck_failed",
                "post_spine_base_stage": post_spine_base_result,
                "startup_timing": {
                    "policy_load_s": policy_load_s,
                    "base_stage_s": base_stage_s,
                    "spine_stage_s": spine_stage_s,
                    "post_spine_base_stage_s": post_spine_base_stage_s,
                },
            }
            _write_json(args.output_dir / "live_runner_manifest.json", failure)
            node.destroy_node()
            rclpy.shutdown()
            print("FAIL: post-spine base recheck failed", flush=True)
            return 2
    manipulation_stage_started = time.monotonic()
    rmpflow_stage_results: list[dict[str, object]] = []
    if args.staging_mode == "dataset_replay":
        assert args.staging_audit is not None and staging_audit is not None
        manipulation_stage_output = (
            args.output_dir / "dataset_pregrasp_staging_manifest.json"
        )
        staging_command = [
            sys.executable,
            "-m",
            "task2_isaacsim.baselines.pi05.live.fixed_stage_manipulation",
            "--audit",
            str(args.staging_audit),
            "--output",
            str(manipulation_stage_output),
            "--max-duration-s",
            str(args.manipulation_stage_max_duration_s),
            "--hold-spine-command-m",
            str(staged_spine_command),
            *(
                ["--defer-pad-relative-gate"]
                if args.restage_base_after_manipulation
                else []
            ),
        ]
        result = subprocess.run(staging_command, check=False)
        stage_result: dict = {}
        if manipulation_stage_output.is_file():
            stage_result = json.loads(
                manipulation_stage_output.read_text(encoding="utf-8")
            )
        stage_feedback = stage_result.get("feedback") or {}
        stage_feedback_pass = stage_feedback.get("within_tolerance") is True
        staging_returncode = result.returncode
    else:
        assert args.observation_reference is not None
        stage_result = {}
        staging_returncode = 0
        for target_kind, angular_speed in RMPFLOW_STAGE_PLAN:
            manipulation_stage_output = (
                args.output_dir
                / f"rmpflow_{target_kind}_staging_manifest.json"
            )
            staging_command = [
                sys.executable,
                "-m",
                "task2_isaacsim.baselines.pi05.live.fixed_stage_observation",
                "--reference",
                str(args.observation_reference),
                "--target-kind",
                target_kind,
                "--output",
                str(manipulation_stage_output),
                "--max-duration-s",
                str(args.manipulation_stage_max_duration_s),
                "--max-linear-speed-m-s",
                "0.10",
                "--max-angular-speed-deg-s",
                str(angular_speed),
                "--maximum-transition-s",
                "8.0",
                "--minimum-transition-s",
                "2.0",
                "--position-tolerance-m",
                (
                    "0.015"
                    if target_kind in {"observation", "continuous_observation"}
                    else "0.025"
                ),
                "--base-position-tolerance-m",
                str(args.position_tolerance_m),
                "--base-yaw-tolerance-rad",
                str(args.yaw_tolerance_rad),
            ]
            result = subprocess.run(staging_command, check=False)
            stage_result = {}
            if manipulation_stage_output.is_file():
                stage_result = json.loads(
                    manipulation_stage_output.read_text(encoding="utf-8")
                )
            rmpflow_stage_results.append(
                {
                    "target_kind": target_kind,
                    "manifest": str(manipulation_stage_output),
                    "returncode": result.returncode,
                    "result": stage_result,
                }
            )
            if (
                result.returncode != 0
                or stage_result.get("success") is not True
                or stage_result.get("reason")
                != "stable_rmpflow_observation_pose"
            ):
                staging_returncode = result.returncode or 2
                break
        stage_feedback_pass = (
            len(rmpflow_stage_results) == len(RMPFLOW_STAGE_PLAN)
            and all(
                item["result"].get("success") is True
                for item in rmpflow_stage_results
            )
        )
    manipulation_stage_s = time.monotonic() - manipulation_stage_started
    if (
        staging_returncode != 0
        or stage_result.get("success") is not True
        or not stage_feedback_pass
    ):
        failure = {
            "schema_version": 10,
            "mode": mode,
            "completed": False,
            "decisions": 0,
            "valid_decisions": 0,
            "command_publications": 0,
            "published_actions": 0,
            "publish_blocked": f"{args.staging_mode}_staging_failed",
            "staging": (
                {"stages": rmpflow_stage_results}
                if args.staging_mode == "rmpflow_observation"
                else stage_result
            ),
            "startup_timing": {
                "policy_load_s": policy_load_s,
                "base_stage_s": base_stage_s,
                "spine_stage_s": spine_stage_s,
                "post_spine_base_stage_s": post_spine_base_stage_s,
                "manipulation_stage_s": manipulation_stage_s,
            },
        }
        _write_json(args.output_dir / "live_runner_manifest.json", failure)
        node.destroy_node()
        rclpy.shutdown()
        print(f"FAIL: {args.staging_mode} staging failed", flush=True)
        return 2
    if args.staging_mode == "rmpflow_observation":
        deterministic_hold_command = tuple(
            float(value)
            for value in stage_result["final_hold_command_action_3_20"]
        )
        if len(deterministic_hold_command) != 17:
            print("FAIL: invalid measured RMPflow staging hold", flush=True)
            node.destroy_node()
            rclpy.shutdown()
            return 2
    if args.code_policy_acquire:
        assert args.observation_reference is not None
        code_policy_output = args.output_dir / "code_policy_transport.json"
        node.destroy_node()
        rclpy.shutdown()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "task2_isaacsim.baselines.pi05.live.fixed_hybrid_transport",
                "--reference",
                str(args.observation_reference),
                "--output",
                str(code_policy_output),
                "--max-duration-s",
                str(args.manipulation_stage_max_duration_s),
                "--code-policy-acquire",
            ],
            check=False,
        )
        payload: dict[str, object] = {}
        if code_policy_output.is_file():
            payload = json.loads(
                code_policy_output.read_text(encoding="utf-8")
            )
        success = result.returncode == 0 and payload.get("success") is True
        summary = {
            "schema_version": 10,
            "mode": mode,
            "completed": success,
            "code_policy_acquire": True,
            "policy_inference_decisions": 0,
            "command_publications": payload.get("command_publications", 0),
            "published_actions": 0,
            "publish_blocked": (
                None if success else "code_policy_transport_failed"
            ),
            "startup_timing": {
                "policy_load_s": policy_load_s,
                "base_stage_s": base_stage_s,
                "spine_stage_s": spine_stage_s,
                "post_spine_base_stage_s": post_spine_base_stage_s,
                "manipulation_stage_s": manipulation_stage_s,
            },
            "code_policy_transport": {
                "returncode": result.returncode,
                "manifest": str(code_policy_output),
                "result": payload,
            },
        }
        _write_json(args.output_dir / "live_runner_manifest.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if success else 2
    final_base_stage_s: float | None = None
    final_base_stage_output: Path | None = None
    if args.restage_base_after_manipulation:
        final_base_stage_output = (
            args.output_dir / "fixed_base_after_manipulation.json"
        )
        final_base_stage_started = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "task2_isaacsim.baselines.pi05.live.fixed_stage_base",
                "--target",
                *(str(value) for value in args.base_target),
                "--position-tolerance-m",
                str(args.position_tolerance_m),
                "--yaw-tolerance-rad",
                str(args.yaw_tolerance_rad),
                "--velocity-threshold",
                str(args.velocity_threshold),
                "--settle-duration-s",
                str(args.settle_duration_s),
                "--max-duration-s",
                str(args.base_stage_max_duration_s),
                "--output",
                str(final_base_stage_output),
            ],
            check=False,
        )
        final_base_stage_s = time.monotonic() - final_base_stage_started
        final_base_stage_result: dict[str, object] = {}
        if final_base_stage_output.is_file():
            final_base_stage_result = json.loads(
                final_base_stage_output.read_text(encoding="utf-8")
            )
        if (
            result.returncode != 0
            or final_base_stage_result.get("success") is not True
        ):
            failure = {
                "schema_version": 9,
                "mode": mode,
                "completed": False,
                "decisions": 0,
                "valid_decisions": 0,
                "command_publications": 0,
                "published_actions": 0,
                "publish_blocked": "base_restage_after_manipulation_failed",
                "base_restage": final_base_stage_result,
                "startup_timing": {
                    "policy_load_s": policy_load_s,
                    "base_stage_s": base_stage_s,
                    "spine_stage_s": spine_stage_s,
                    "manipulation_stage_s": manipulation_stage_s,
                    "final_base_stage_s": final_base_stage_s,
                },
            }
            _write_json(args.output_dir / "live_runner_manifest.json", failure)
            node.destroy_node()
            rclpy.shutdown()
            print("FAIL: base restaging after manipulation failed", flush=True)
            return 2
    if args.arm_simulator:
        try:
            node.activate_publishers()
        except RuntimeError as error:
            print(f"FAIL: {error}", flush=True)
            node.destroy_node()
            rclpy.shutdown()
            return 2
    node.discard_staging_observations()
    gate.reset(f"{args.staging_mode}_staging_complete")
    started = time.monotonic()
    last_sequences: dict[str, int] = {}
    last_reset_count = node.reset_count
    events: list[dict] = []
    warmup_latencies: list[float] = []
    latencies: list[float] = []
    invalid_actions = 0
    stale_observations = 0
    reset_recoveries = 0
    queue_replacements = 0
    replaced_residual_actions = 0
    queue: deque[QueuedAction] = deque()
    future: Future | None = None
    future_context: dict | None = None
    published_actions = 0
    publish_blocked: str | None = None
    queue_underflow = False
    queue_pause_count = 0
    waiting_for_fresh_observation = False
    capture_to_ready_host_latencies: list[float] = []
    capture_to_ready_sim_latencies: list[float] = []
    discarded_prefix_actions = 0
    manipulation_latched = False
    last_freshness_rejection: dict | None = None
    initial_spine_position_m: float | None = None
    last_policy_action: tuple[float, ...] | None = None
    next_hold_sim_time: float | None = None
    hold_action_publications = 0
    post_action_hold_publications = 0
    spine_trajectory: list[dict[str, float | int]] = []
    right_ee_trajectory: list[dict[str, object]] = []
    grasp_guard = (
        RightGraspGuard() if args.right_only_policy_after_staging else None
    )
    right_workspace_stop_evidence: dict[str, object] | None = None
    hybrid_grasp_latched = False
    first_published_sim_time: float | None = None
    last_published_sim_time: float | None = None
    last_actuator_publication_sim_time: float | None = None
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05")

    # Warm up CUDA/model kernels before steady-state measurements. No action is
    # published here and the same valid observation can be reused safely.
    warmup_images: dict[str, np.ndarray] | None = None
    warmup_state: tuple[float, ...] | None = None
    staging_observation_images: dict[str, str] = {}
    while (
        warmup_images is None
        and time.monotonic() - started < args.max_duration_s
        and not node.stop_requested
    ):
        rclpy.spin_once(node, timeout_sec=0.02)
        now = time.monotonic()
        if node.sim_time is None or not node.update_readiness(node.sim_time):
            continue
        try:
            warmup_images, warmup_state = node.snapshot()
            validate_live_state(warmup_state)
            freshness_metrics(
                now=now,
                capture_now=node.sim_time,
                camera_times=node.image_times,
                camera_capture_times=node.image_capture_times,
                camera_sequences=node.image_sequences,
                state_time=node.state_time,
                state_times=node.state_times,
                state_capture_times=node.state_capture_times,
                last_camera_sequences={},
                config=freshness,
            )
        except ValueError:
            warmup_images = None
            stale_observations += 1
    if warmup_images is not None and warmup_state is not None:
        initial_spine_position_m = warmup_state[28]
        for key, image in warmup_images.items():
            path = args.output_dir / f"settled_fresh_{key}.ppm"
            _write_ppm(path, image)
            staging_observation_images[key] = str(path)
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

            ready = (
                node.sim_time is not None
                and node.update_readiness(node.sim_time)
            )
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
                    if node.sim_time is None:
                        raise RuntimeError("simulator clock unavailable")
                    ready_sim_at = node.sim_time
                    capture_to_ready_latency = (
                        created_at - future_context["capture_at"]
                    )
                    capture_to_ready_sim_s = max(
                        0.0,
                        ready_sim_at - future_context["capture_sim_at"],
                    )
                    capture_to_ready_host_latencies.append(
                        capture_to_ready_latency
                    )
                    capture_to_ready_sim_latencies.append(
                        capture_to_ready_sim_s
                    )
                    validated = []
                    previous_right_target = tuple(
                        last_policy_action[10:17]
                        if last_policy_action is not None
                        else state[21:28]
                    )
                    previous_left_target = tuple(
                        last_policy_action[3:10]
                        if last_policy_action is not None
                        else state[14:21]
                    )
                    left_slew_projected_actions = 0
                    left_slew_max_correction_rad = 0.0
                    right_slew_projected_actions = 0
                    right_slew_max_correction_rad = 0.0
                    for validation_index, action in enumerate(chunk):
                        raw, effective = safe_action(action)
                        if args.right_only_policy_after_staging:
                            assert deterministic_hold_command is not None
                            effective = apply_right_only_policy_ownership(
                                effective,
                                deterministic_hold_command,
                            )
                            requested_right = tuple(effective[10:17])
                            projected_right = project_fr3_joint_step(
                                previous_right_target,
                                requested_right,
                                action_rate_hz=args.action_rate_hz,
                            )
                            corrections = [
                                abs(after - before)
                                for before, after in zip(
                                    requested_right,
                                    projected_right,
                                    strict=True,
                                )
                            ]
                            if any(value > 1e-9 for value in corrections):
                                right_slew_projected_actions += 1
                                right_slew_max_correction_rad = max(
                                    right_slew_max_correction_rad,
                                    max(corrections),
                                )
                            effective = (
                                *effective[:10],
                                *projected_right,
                                *effective[17:],
                            )
                            previous_right_target = projected_right
                        elif args.whole_body_policy_after_staging:
                            requested_left = tuple(effective[3:10])
                            requested_right = tuple(effective[10:17])
                            projected_left = project_fr3_joint_step(
                                previous_left_target,
                                requested_left,
                                action_rate_hz=args.action_rate_hz,
                            )
                            projected_right = project_fr3_joint_step(
                                previous_right_target,
                                requested_right,
                                action_rate_hz=args.action_rate_hz,
                            )
                            left_corrections = [
                                abs(after - before)
                                for before, after in zip(
                                    requested_left,
                                    projected_left,
                                    strict=True,
                                )
                            ]
                            right_corrections = [
                                abs(after - before)
                                for before, after in zip(
                                    requested_right,
                                    projected_right,
                                    strict=True,
                                )
                            ]
                            if any(value > 1e-9 for value in left_corrections):
                                left_slew_projected_actions += 1
                                left_slew_max_correction_rad = max(
                                    left_slew_max_correction_rad,
                                    max(left_corrections),
                                )
                            if any(value > 1e-9 for value in right_corrections):
                                right_slew_projected_actions += 1
                                right_slew_max_correction_rad = max(
                                    right_slew_max_correction_rad,
                                    max(right_corrections),
                                )
                            effective = (
                                *effective[:3],
                                *projected_left,
                                *projected_right,
                                *effective[17:],
                            )
                            previous_left_target = projected_left
                            previous_right_target = projected_right
                        validated.append((raw, effective))
                    effective_actions = [
                        effective for _, effective in validated
                    ]
                    if hard5:
                        discarded = 0
                        aligned = hard5_action_window(
                            effective_actions,
                            ready_at=ready_sim_at,
                            action_rate_hz=args.action_rate_hz,
                            execution_horizon=args.execution_horizon,
                            max_actions=(
                                args.max_publish_actions - published_actions
                                if args.arm_simulator
                                else 5
                            ),
                        )
                    else:
                        discarded, aligned = align_action_chunk(
                            effective_actions,
                            capture_at=future_context["capture_sim_at"],
                            ready_at=ready_sim_at,
                            action_rate_hz=args.action_rate_hz,
                            execution_started=published_actions > 0,
                        )
                    discarded_prefix_actions += discarded
                except (RuntimeError, ValueError) as error:
                    invalid_actions += 1
                    events.append(
                        {
                            "decision": len(events),
                            "valid": False,
                            "error": str(error),
                            "invalid_action_index": validation_index,
                            "policy_compute_host_s": latency,
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
                    "runtime_mode": args.runtime_mode,
                    "valid": True,
                    "fixed_base_staging": True,
                    "policy_controlled_spine": not args.right_only_policy_after_staging,
                    "policy_owned_groups": (
                        ["right_arm", "right_gripper"]
                        if args.right_only_policy_after_staging
                        else [
                            "left_arm",
                            "right_arm",
                            "left_gripper",
                            "right_gripper",
                            "spine",
                        ]
                    ),
                    "deterministic_hold_groups": (
                        ["left_arm", "left_gripper", "spine"]
                        if args.right_only_policy_after_staging
                        else []
                    ),
                    "readiness": future_context["readiness"],
                    "freshness": future_context["freshness"],
                    "observation_state": list(state),
                    "language_gt_frames": future_context.get(
                        "language_gt_frames", []
                    ),
                    "language_gt_reference_right_joints": future_context.get(
                        "language_gt_reference_right_joints"
                    ),
                    "language_prompt": future_context.get("language_prompt"),
                    "inference_latency_s": capture_to_ready_sim_s,
                    "inference_latency_clock": "simulator",
                    "policy_compute_host_s": latency,
                    "capture_to_ready_host_s": capture_to_ready_latency,
                    "capture_to_ready_sim_s": capture_to_ready_sim_s,
                    "discarded_prefix_actions": discarded,
                    "aligned_actions": len(aligned),
                    "first_aligned_chunk_index": (
                        aligned[0][2] if aligned else None
                    ),
                    "last_aligned_chunk_index": (
                        aligned[-1][2] if aligned else None
                    ),
                    "first_executed_chunk_index": None,
                    "last_executed_chunk_index": None,
                    "policy_indices": [item[2] for item in aligned],
                    "hold_action_publications": future_context[
                        "hold_action_publications"
                    ],
                    "queue_underflow": False,
                    "queue_refill_window_s": refill_window_s,
                    "latency_within_refill_window": (
                        capture_to_ready_sim_s <= refill_window_s
                    ),
                    "raw_actions": [list(raw) for raw, _ in validated],
                    "effective_actions": [
                        list(effective) for _, effective in validated
                    ],
                    "right_slew_projected_actions": (
                        right_slew_projected_actions
                    ),
                    "right_slew_max_correction_rad": (
                        right_slew_max_correction_rad
                    ),
                    "left_slew_projected_actions": (
                        left_slew_projected_actions
                    ),
                    "left_slew_max_correction_rad": (
                        left_slew_max_correction_rad
                    ),
                    "published_actions": 0,
                }
                events.append(event)
                first_raw, first_effective = validated[discarded]
                print(
                    json.dumps(
                        {
                            "decision": event_index,
                            "policy_compute_host_s": latency,
                            "capture_to_ready_host_s": (
                                capture_to_ready_latency
                            ),
                            "capture_to_ready_sim_s": capture_to_ready_sim_s,
                            "discarded_prefix_actions": discarded,
                            "first_aligned_chunk_index": (
                                aligned[0][2] if aligned else None
                            ),
                            "raw_gripper": first_raw[17:19],
                            "effective_gripper": first_effective[17:19],
                            "raw_spine_target_m": first_raw[19],
                            "effective_spine_target_m": first_effective[19],
                            "first_arm_target": first_effective[3:17],
                        }
                    ),
                    flush=True,
                )
                if args.arm_simulator:
                    if not aligned:
                        publish_blocked = "entire_action_chunk_expired"
                        gate.stop(publish_blocked)
                        queue.clear()
                        future = None
                        future_context = None
                        break
                    residual_actions = replace_action_queue(
                        queue,
                        aligned,
                        completed_at=ready_sim_at,
                        event_index=event_index,
                    )
                    if residual_actions:
                        queue_replacements += 1
                        replaced_residual_actions += residual_actions
                        event["replaced_residual_actions"] = residual_actions
                    if hard5 and residual_actions:
                        publish_blocked = "hard5_residual_queue"
                        gate.stop(publish_blocked)
                        queue.clear()
                        future = None
                        future_context = None
                        break
                future = None
                future_context = None

            if (
                args.arm_simulator
                and queue
                and node.sim_time is not None
                and node.sim_time >= queue[0][1]
                and (
                    last_actuator_publication_sim_time is None
                    or node.sim_time > last_actuator_publication_sim_time
                )
            ):
                if not ready:
                    gate.stop("readiness_revoked_before_publish")
                    publish_blocked = "readiness_revoked_before_publish"
                    queue.clear()
                    break
                if gate.phase == RunnerPhase.MANIPULATION_READY:
                    gate.arm()
                    manipulation_latched = True
                (
                    effective,
                    _target_at,
                    created_at,
                    event_index,
                    chunk_index,
                ) = queue.popleft()
                if (
                    node.sim_time - created_at
                    > freshness.action_queue_max_age_s
                ):
                    publish_blocked = "action_queue_watchdog"
                elif publish_blocked is None:
                    counts = node.command_publisher_counts()
                    if any(count > 1 for count in counts.values()):
                        publish_blocked = f"command_contention:{counts}"
                    else:
                        if grasp_guard is not None:
                            right_pose = node.ee_poses["right"]
                            if (
                                right_pose is None
                                or not right_ee_within_demonstrated_workspace(
                                    right_pose
                                )
                            ):
                                right_workspace_stop_evidence = {
                                    "published_action_attempt": published_actions + 1,
                                    "right_ee_world_xyzw": (
                                        list(right_pose)
                                        if right_pose is not None
                                        else None
                                    ),
                                    "grasp_guard": grasp_guard.summary(),
                                }
                                publish_blocked = (
                                    "right_ee_outside_demonstrated_workspace"
                                )
                                gate.stop(publish_blocked)
                            else:
                                wrist_evidence = _right_wrist_pad_signature(
                                    node.images["wrist_right"]
                                )
                                wrist_ready = (
                                    right_wrist_grasp_evidence_within_development_envelope(
                                        wrist_evidence
                                    )
                                )
                                guarded_open, guard_event = grasp_guard.apply(
                                    effective[18],
                                    right_pose,
                                    camera_grasp_ready=wrist_ready,
                                    camera_grasp_evidence=wrist_evidence,
                                )
                                if (
                                    guard_event["intervened"]
                                    or guard_event["phase_changed"]
                                ):
                                    guard_event["published_action_attempt"] = (
                                        published_actions + 1
                                    )
                                    events[event_index].setdefault(
                                        "grasp_guard_events", []
                                    ).append(guard_event)
                                effective = (
                                    *effective[:18],
                                    guarded_open,
                                    effective[19],
                                )
                                if (
                                    args.hybrid_rmpflow_transport
                                    and guard_event["phase_changed"]
                                    and guard_event["phase_after"] == "latched"
                                ):
                                    hybrid_grasp_latched = True
                        if publish_blocked is None:
                            node.publish_action(effective)
                        else:
                            queue.clear()
                            break
                        published_actions += 1
                        last_actuator_publication_sim_time = node.sim_time
                        last_policy_action = effective
                        if first_published_sim_time is None:
                            first_published_sim_time = node.sim_time
                        last_published_sim_time = node.sim_time
                        event = events[event_index]
                        event["published_actions"] += 1
                        if event["first_executed_chunk_index"] is None:
                            event["first_executed_chunk_index"] = chunk_index
                        event["last_executed_chunk_index"] = chunk_index
                        spine_trajectory.append(
                            {
                                "published_action": published_actions,
                                "sim_time": float(node.sim_time),
                                "measured_m": float(node.joints[SPINE_JOINT]),
                                "target_m": effective[19],
                            }
                        )
                        if node.ee_poses["right"] is not None:
                            right_ee_trajectory.append(
                                {
                                    "published_action": published_actions,
                                    "sim_time": float(node.sim_time),
                                    "pose_world_xyzw_before_publish": list(
                                        node.ee_poses["right"]
                                    ),
                                }
                            )
                        if hybrid_grasp_latched:
                            queue.clear()
                if publish_blocked:
                    queue.clear()
                    break
                if published_actions >= args.max_publish_actions:
                    queue.clear()
                    break
                if not queue:
                    if hard5:
                        next_hold_sim_time = node.sim_time
                    else:
                        queue_underflow = True
                        queue_pause_count += 1
                        events[event_index]["queue_underflow"] = True

            if hybrid_grasp_latched:
                break

            hold_target = hard5_hold_action(
                last_policy_action,
                inference_pending=hard5 and future is not None,
                queue=queue,
            )
            if (
                args.arm_simulator
                and hold_target is not None
                and node.sim_time is not None
                and next_hold_sim_time is not None
                and node.sim_time >= next_hold_sim_time
                and (
                    last_actuator_publication_sim_time is None
                    or node.sim_time > last_actuator_publication_sim_time
                )
            ):
                if not ready:
                    publish_blocked = "readiness_revoked_before_hold"
                    gate.stop(publish_blocked)
                    break
                counts = node.command_publisher_counts()
                if any(count > 1 for count in counts.values()):
                    publish_blocked = f"command_contention:{counts}"
                    gate.stop(publish_blocked)
                    break
                node.publish_action(hold_target)
                last_actuator_publication_sim_time = node.sim_time
                hold_action_publications += 1
                assert future_context is not None
                future_context["hold_action_publications"] += 1
                next_hold_sim_time += 1.0 / args.action_rate_hz

            decisions_started = len(events) + int(future is not None)
            needs_inference = (
                decisions_started < args.max_decisions
                and future is None
                and ready
                and (
                    not args.arm_simulator
                    or (
                        not queue
                        if hard5
                        else len(queue) <= args.queue_refill_actions
                    )
                )
            )
            if needs_inference:
                try:
                    capture_at = time.monotonic()
                    if node.sim_time is None:
                        raise FreshnessError(
                            "simulator clock unavailable",
                            {"offending_streams": [TOPICS["clock"]]},
                        )
                    capture_sim_at = node.sim_time
                    images, state = node.snapshot()
                    validate_live_state(state)
                    metrics = freshness_metrics(
                        now=capture_at,
                        capture_now=capture_sim_at,
                        camera_times=node.image_times,
                        camera_capture_times=node.image_capture_times,
                        camera_sequences=node.image_sequences,
                        state_time=node.state_time,
                        state_times=node.state_times,
                        state_capture_times=node.state_capture_times,
                        last_camera_sequences=last_sequences,
                        config=freshness,
                    )
                except FreshnessError as error:
                    stale_observations += 1
                    last_freshness_rejection = {
                        **error.evidence,
                        "runner_phase": gate.phase.value,
                        "check": "inference_snapshot",
                    }
                    if (
                        args.arm_simulator
                        and published_actions > 0
                        and not queue
                        and not waiting_for_fresh_observation
                    ):
                        waiting_for_fresh_observation = True
                        print(
                            json.dumps(
                                {
                                    "pause": "waiting_for_fresh_observation",
                                    **last_freshness_rejection,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                except ValueError:
                    stale_observations += 1
                else:
                    if waiting_for_fresh_observation:
                        waiting_for_fresh_observation = False
                        print(
                            json.dumps(
                                {"resume": "fresh_observation_received"}
                            ),
                            flush=True,
                        )
                    last_sequences = dict(node.image_sequences)
                    inference_instruction = task_instruction
                    language_gt_frames: tuple[int, ...] = ()
                    if language_gt_trajectory is not None:
                        (
                            inference_instruction,
                            language_gt_frames,
                        ) = format_language_gt_window(
                            language_gt_trajectory,
                            executed_actions=published_actions,
                            window=args.execution_horizon,
                            reference_joint_state=tuple(state[21:28]),
                        )
                    future_context = {
                        "state": state,
                        "capture_at": capture_at,
                        "capture_sim_at": capture_sim_at,
                        "readiness": dict(gate.last_evidence),
                        "freshness": metrics,
                        "hold_action_publications": 0,
                        "language_gt_frames": list(language_gt_frames),
                        "language_gt_reference_right_joints": (
                            list(state[21:28])
                            if language_gt_trajectory is not None
                            else None
                        ),
                        "language_prompt": (
                            inference_instruction
                            if language_gt_trajectory is not None
                            else None
                        ),
                    }
                    if args.arm_simulator and not manipulation_latched:
                        gate.arm()
                        manipulation_latched = True
                        future_context["readiness"] = dict(gate.last_evidence)
                    future = worker.submit(
                        policy.predict_chunk,
                        images=images,
                        state=state,
                        instruction=inference_instruction,
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

    post_action_handoff: dict[str, object] = {
        "attempted": False,
        "success": not args.arm_simulator or published_actions == 0,
        "reason": "not_required",
        "hold_publications": 0,
        "ee_target_publications": 0,
    }
    safety_joint_hold = False
    if (
        publish_blocked is not None
        and args.right_only_policy_after_staging
        and last_policy_action is not None
    ):
        measured_right = tuple(
            node.joints.get(name, math.nan) for name in RIGHT_JOINTS
        )
        if all(math.isfinite(value) for value in measured_right):
            hold_open_fraction = last_policy_action[18]
            if grasp_guard is not None:
                if grasp_guard.phase == "latched":
                    hold_open_fraction = 0.0
                elif grasp_guard.phase == "released":
                    hold_open_fraction = 1.0
            last_policy_action = (
                *last_policy_action[:10],
                *measured_right,
                last_policy_action[17],
                hold_open_fraction,
                last_policy_action[19],
            )
            safety_joint_hold = True
    if (
        args.arm_simulator
        and published_actions > 0
        and last_policy_action is not None
        and node.sim_time is not None
    ):
        post_action_handoff.update(
            {"attempted": True, "success": False, "reason": "in_progress"}
        )
        handoff_host_deadline = time.monotonic() + 20.0
        # A policy-side close latch means that the close command was accepted,
        # not that the physical gripper has already reached the pad.  Keep the
        # final closed command under joint-command ownership until the measured
        # driver is closed and stable before handing ownership to RMPflow.
        hybrid_close_confirmed = not (
            args.hybrid_rmpflow_transport and hybrid_grasp_latched
        )
        hybrid_close_stable_since: float | None = None
        hybrid_close_deadline_sim = (
            node.sim_time + 2.0 if not hybrid_close_confirmed else None
        )
        hold_until_sim = (
            hybrid_close_deadline_sim
            if hybrid_close_deadline_sim is not None
            else node.sim_time + 0.20
        )
        next_post_hold_sim = node.sim_time
        while (
            node.sim_time is not None
            and node.sim_time < hold_until_sim
            and time.monotonic() < handoff_host_deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.01)
            if (
                node.sim_time is not None
                and node.sim_time >= next_post_hold_sim
            ):
                node.publish_action(last_policy_action)
                hold_action_publications += 1
                post_action_hold_publications += 1
                next_post_hold_sim += 1.0 / args.action_rate_hz
            if not hybrid_close_confirmed and node.sim_time is not None:
                measured_open = gripper_open_fraction(
                    resolve_joint(node.joints, RIGHT_GRIPPER_DRIVER)
                )
                if math.isfinite(measured_open) and measured_open <= 0.25:
                    if hybrid_close_stable_since is None:
                        hybrid_close_stable_since = node.sim_time
                    elif node.sim_time - hybrid_close_stable_since >= 0.25:
                        hybrid_close_confirmed = True
                        hold_until_sim = node.sim_time
                        break
                else:
                    hybrid_close_stable_since = None

        pose_before_handoff = node.ee_poses["right"]
        fresh_ee_after_last_action = bool(
            last_published_sim_time is not None
            and node.state_capture_times.get("ee_right", -math.inf)
            > last_published_sim_time
        )
        post_action_handoff.update(
            {
                "hold_publications": post_action_hold_publications,
                "fresh_ee_after_last_action": fresh_ee_after_last_action,
                "pose_before_handoff_world_xyzw": (
                    list(pose_before_handoff)
                    if pose_before_handoff is not None
                    else None
                ),
                "hybrid_close_confirmed": hybrid_close_confirmed,
                "measured_right_gripper_open_fraction": gripper_open_fraction(
                    resolve_joint(node.joints, RIGHT_GRIPPER_DRIVER)
                ),
            }
        )
        if safety_joint_hold:
            post_action_handoff.update(
                {
                    "success": True,
                    "reason": "measured_joint_safety_hold",
                    "safety_stop": publish_blocked,
                    "ee_target_publications": 0,
                }
            )
        elif not hybrid_close_confirmed:
            post_action_handoff["reason"] = "hybrid_gripper_close_timeout"
        elif (
            node.sim_time is None
            or node.sim_time < hold_until_sim
            or pose_before_handoff is None
            or not fresh_ee_after_last_action
        ):
            post_action_handoff["reason"] = "fresh_post_action_pose_unavailable"
        else:
            latch_until_sim = node.sim_time + 0.50
            next_latch_sim = node.sim_time
            while (
                node.sim_time is not None
                and node.sim_time < latch_until_sim
                and time.monotonic() < handoff_host_deadline
            ):
                rclpy.spin_once(node, timeout_sec=0.01)
                if node.sim_time is not None and node.sim_time >= next_latch_sim:
                    node.publish_ee_handoff_target(
                        "right", pose_before_handoff
                    )
                    next_latch_sim += 0.10
            pose_after_handoff = node.ee_poses["right"]
            handoff_drift_m = (
                math.dist(
                    pose_before_handoff[:3], pose_after_handoff[:3]
                )
                if pose_after_handoff is not None
                else math.inf
            )
            handoff_success = bool(
                node.sim_time is not None
                and node.sim_time >= latch_until_sim
                and pose_after_handoff is not None
                and math.isfinite(handoff_drift_m)
                and handoff_drift_m <= 0.005
            )
            post_action_handoff.update(
                {
                    "success": handoff_success,
                    "reason": (
                        "current_pose_latched"
                        if handoff_success
                        else "rmpflow_handoff_drift_or_timeout"
                    ),
                    "ee_target_publications": node.ee_handoff_publish_count,
                    "pose_after_handoff_world_xyzw": (
                        list(pose_after_handoff)
                        if pose_after_handoff is not None
                        else None
                    ),
                    "position_drift_m": handoff_drift_m,
                    "position_drift_limit_m": 0.005,
                }
            )
        if post_action_handoff["success"] is not True and publish_blocked is None:
            publish_blocked = "post_action_rmpflow_handoff_failed"

    hybrid_transport: dict[str, object] = {
        "attempted": False,
        "success": not args.hybrid_rmpflow_transport,
        "reason": "not_requested",
    }
    if args.hybrid_rmpflow_transport:
        hybrid_output = args.output_dir / "hybrid_rmpflow_transport.json"
        hybrid_transport = {
            "attempted": False,
            "success": False,
            "reason": "grasp_latch_not_reached",
            "manifest": str(hybrid_output),
        }
        if hybrid_grasp_latched and post_action_handoff["success"] is True:
            assert args.observation_reference is not None
            node.deactivate_publishers()
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "task2_isaacsim.baselines.pi05.live.fixed_hybrid_transport",
                    "--reference",
                    str(args.observation_reference),
                    "--output",
                    str(hybrid_output),
                ],
                check=False,
            )
            payload: dict[str, object] = {}
            if hybrid_output.is_file():
                payload = json.loads(hybrid_output.read_text(encoding="utf-8"))
            hybrid_transport = {
                "attempted": True,
                "success": result.returncode == 0
                and payload.get("success") is True,
                "reason": payload.get("reason", "transport_process_failed"),
                "returncode": result.returncode,
                "manifest": str(hybrid_output),
                "result": payload,
            }
            if hybrid_transport["success"] is not True and publish_blocked is None:
                publish_blocked = "hybrid_rmpflow_transport_failed"

    elapsed = time.monotonic() - started
    refill_window_s = args.queue_refill_actions / args.action_rate_hz
    latency_contract_pass = bool(capture_to_ready_sim_latencies) and all(
        value <= refill_window_s for value in capture_to_ready_sim_latencies
    )
    summary = {
        "schema_version": 10,
        "mode": mode,
        "runtime_mode": args.runtime_mode,
        "execution_horizon": (
            args.execution_horizon if hard5 else "asynchronous_refill"
        ),
        "fixed_base_staging": True,
        "right_only_policy_after_staging": args.right_only_policy_after_staging,
        "whole_body_policy_after_staging": (
            args.whole_body_policy_after_staging
        ),
        "hybrid_rmpflow_transport_requested": args.hybrid_rmpflow_transport,
        "hybrid_grasp_latched": hybrid_grasp_latched,
        "hybrid_transport": hybrid_transport,
        "policy_owned_groups": (
            ["right_arm", "right_gripper"]
            if args.right_only_policy_after_staging
            else [
                "left_arm",
                "right_arm",
                "left_gripper",
                "right_gripper",
                "spine",
            ]
        ),
        "deterministic_hold_command_action_3_20": (
            list(deterministic_hold_command)
            if args.right_only_policy_after_staging
            and deterministic_hold_command is not None
            else None
        ),
        "manipulation_only": True,
        "policy_controlled_groups": (
            ["right_arm", "right_gripper"]
            if args.right_only_policy_after_staging
            else list(POLICY_COMMAND_TOPICS)
        ),
        "command_publication_groups": list(POLICY_COMMAND_TOPICS),
        "forbidden_groups": (
            ["base", "left_arm", "left_gripper", "spine"]
            if args.right_only_policy_after_staging
            else ["base"]
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_relative_action_state_indices": (
            list(policy.relative_action_state_indices)
            if policy.relative_action_state_indices is not None
            else None
        ),
        "ignored_legacy_checkpoint_config_fields": list(
            policy.ignored_legacy_config_fields
        ),
        "manual_mapped_relative_action_inverse": (
            policy.manual_relative_inverse
        ),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "task_instruction": task_instruction,
        "with_language_gt": args.with_language_gt,
        "language_gt": (
            {
                "source_episode": SOURCE_EPISODE,
                "source_frame_start": SOURCE_START_FRAME,
                "source_frame_end_inclusive": SOURCE_END_EXCLUSIVE - 1,
                "trajectory_frames": len(language_gt_trajectory),
                "rolling_window": args.execution_horizon,
                "tokenizer_max_length": LANGUAGE_GT_TOKENIZER_MAX_LENGTH,
            }
            if language_gt_trajectory is not None
            else None
        ),
        "chunk_size": policy.chunk_size,
        "n_action_steps": policy.n_action_steps,
        "warmup_decisions": args.warmup_decisions,
        "warmup_policy_compute_host_s": warmup_latencies,
        "seed": args.seed,
        "base_coordinate_frame": args.base_coordinate_frame,
        "readiness_config": readiness.__dict__,
        "freshness_config": freshness.__dict__,
        "startup": {
            "reason": "all_required_samples_received",
            "attempts": startup_attempts,
            "final": startup_status,
        },
        "startup_timing": {
            "policy_load_s": policy_load_s,
            "base_stage_s": base_stage_s,
            "spine_stage_s": spine_stage_s,
            "post_spine_base_stage_s": post_spine_base_stage_s,
            "manipulation_stage_s": manipulation_stage_s,
            "base_stage_order": (
                "after_policy_load"
                if args.stage_base_after_policy_load
                else "before_runner_start"
            ),
            "base_stage_output": (
                str(base_stage_output)
                if base_stage_output is not None
                else None
            ),
            "spine_stage_output": (
                str(spine_stage_output)
                if spine_stage_output is not None
                else None
            ),
            "post_spine_base_stage_output": (
                str(post_spine_base_stage_output)
                if post_spine_base_stage_output is not None
                else None
            ),
            "final_base_stage_s": final_base_stage_s,
            "final_base_stage_output": (
                str(final_base_stage_output)
                if final_base_stage_output is not None
                else None
            ),
        },
        "staging": {
            "mode": args.staging_mode,
            "audit": (
                str(args.staging_audit.resolve())
                if args.staging_audit is not None
                else None
            ),
            "observation_reference": (
                str(args.observation_reference.resolve())
                if args.observation_reference is not None
                else None
            ),
            "execution_manifest": str(manipulation_stage_output),
            "result": stage_result,
            "rmpflow_waypoints": rmpflow_stage_results,
            "observations_discarded_after_staging": True,
            "settled_fresh_observation_images": staging_observation_images,
            "right_wrist_pad_visible_operator_attested": (
                args.confirm_right_wrist_pad_visible
            ),
        },
        "final_readiness": gate.last_evidence,
        "decisions": len(events),
        "valid_decisions": sum(event.get("valid", False) for event in events),
        "invalid_actions": invalid_actions,
        "stale_observation_rejections": stale_observations,
        "reset_events": node.reset_count,
        "reset_recoveries": reset_recoveries,
        "queue_replacements": queue_replacements,
        "replaced_residual_actions": replaced_residual_actions,
        "time_alignment": {
            "clock": "simulator",
            "clock_topic": TOPICS["clock"],
            "freshness_clock": "host_monotonic",
            "capture_alignment_clock": "simulator_message_headers",
            "action_command_header_clock": "simulator",
            "action_queue_watchdog_clock": "simulator",
            "first_published_sim_time": first_published_sim_time,
            "last_published_sim_time": last_published_sim_time,
            "published_sim_duration_s": (
                last_published_sim_time - first_published_sim_time
                if first_published_sim_time is not None
                and last_published_sim_time is not None
                else None
            ),
            "capture_to_ready_sim_latency_s": {
                "p50": (
                    statistics.median(capture_to_ready_sim_latencies)
                    if capture_to_ready_sim_latencies
                    else None
                ),
                "p95": (
                    _percentile(capture_to_ready_sim_latencies, 0.95)
                    if capture_to_ready_sim_latencies
                    else None
                ),
                "max": (
                    max(capture_to_ready_sim_latencies)
                    if capture_to_ready_sim_latencies
                    else None
                ),
            },
            "capture_to_ready_host_latency_s": {
                "p50": (
                    statistics.median(capture_to_ready_host_latencies)
                    if capture_to_ready_host_latencies
                    else None
                ),
                "p95": (
                    _percentile(capture_to_ready_host_latencies, 0.95)
                    if capture_to_ready_host_latencies
                    else None
                ),
                "max": (
                    max(capture_to_ready_host_latencies)
                    if capture_to_ready_host_latencies
                    else None
                ),
            },
            "discarded_prefix_actions": discarded_prefix_actions,
            "first_executed_chunk_index": min(
                (
                    event["first_executed_chunk_index"]
                    for event in events
                    if event.get("first_executed_chunk_index") is not None
                ),
                default=None,
            ),
            "last_executed_chunk_index": max(
                (
                    event["last_executed_chunk_index"]
                    for event in events
                    if event.get("last_executed_chunk_index") is not None
                ),
                default=None,
            ),
            "queue_underflow": queue_underflow,
            "queue_pause_count": queue_pause_count,
        },
        "command_publications": node.publish_count,
        "hold_action_publications": hold_action_publications,
        "post_action_handoff": post_action_handoff,
        "published_actions": published_actions,
        "publish_blocked": publish_blocked,
        "freshness_stop_evidence": (
            last_freshness_rejection
            if publish_blocked == "live_stream_stale"
            else None
        ),
        "last_freshness_rejection": last_freshness_rejection,
        "base_input_events": node.base_input_count,
        "base_input_events_after_manipulation_latch": (
            node.base_input_after_manipulation_count
        ),
        "spine_control": {
            "policy_controlled": not args.right_only_policy_after_staging,
            "command_topic": SPINE_COMMAND_TOPIC,
            "initial_measured_m": initial_spine_position_m,
            "final_measured_m": (
                float(node.joints[SPINE_JOINT])
                if SPINE_JOINT in node.joints
                else None
            ),
            "target_min_m": (
                min(sample["target_m"] for sample in spine_trajectory)
                if spine_trajectory
                else None
            ),
            "target_max_m": (
                max(sample["target_m"] for sample in spine_trajectory)
                if spine_trajectory
                else None
            ),
            "measured_min_m": (
                min(sample["measured_m"] for sample in spine_trajectory)
                if spine_trajectory
                else None
            ),
            "measured_max_m": (
                max(sample["measured_m"] for sample in spine_trajectory)
                if spine_trajectory
                else None
            ),
            "trajectory": spine_trajectory,
        },
        "right_ee_trajectory": right_ee_trajectory,
        "grasp_guard": grasp_guard.summary() if grasp_guard is not None else None,
        "right_workspace_stop_evidence": right_workspace_stop_evidence,
        "elapsed_s": elapsed,
        "inference_latency_clock": "simulator",
        "inference_latency_s": {
            "p50": (
                statistics.median(capture_to_ready_sim_latencies)
                if capture_to_ready_sim_latencies
                else None
            ),
            "p95": (
                _percentile(capture_to_ready_sim_latencies, 0.95)
                if capture_to_ready_sim_latencies
                else None
            ),
            "max": (
                max(capture_to_ready_sim_latencies)
                if capture_to_ready_sim_latencies
                else None
            ),
        },
        "policy_compute_host_s": {
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
            (not args.arm_simulator and len(events) == args.max_decisions)
            or (
                args.arm_simulator
                and args.hybrid_rmpflow_transport
                and hybrid_grasp_latched
                and post_action_handoff["success"] is True
                and hybrid_transport["success"] is True
            )
            or (
                args.arm_simulator
                and bool(events)
                and published_actions == args.max_publish_actions
                and post_action_handoff["success"] is True
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
