# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light state, freshness, readiness, and action safety gates."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from task2_isaacsim.baselines.pi05.contract import (
    ACTION_SIZE,
    EXPECTED_CAMERA_SHAPES,
    apply_fixed_mobile_axes,
    project_arm_action_bounds,
    validate_absolute_action_bounds,
)
from task2_isaacsim.common.state_contract import (
    LEFT_JOINTS,
    RIGHT_JOINTS,
    SPINE_JOINT,
    finite_state,
)

ROBOT_CAMERA_KEYS = ("head", "wrist_left", "wrist_right")
SPINE_POLICY_MIN_M = 0.0
SPINE_POLICY_MAX_M = 0.6
TimedAction = tuple[tuple[float, ...], float, int]
QueuedAction = tuple[tuple[float, ...], float, float, int, int]


def policy_command_topics(topics: dict[str, Any]) -> dict[str, str]:
    """Return the fixed-base policy publication contract."""

    return {
        **{
            group: entry["command"]
            for group, entry in topics["bridge"]["joint_groups"].items()
        },
        "spine": topics["teleop"]["spine_target"],
    }


def align_action_chunk(
    actions: list[tuple[float, ...]],
    *,
    capture_at: float,
    ready_at: float,
    action_rate_hz: float,
    execution_started: bool = True,
) -> tuple[int, list[TimedAction]]:
    """Align a chunk to action progress, preserving index 0 while idle."""

    if action_rate_hz <= 0.0:
        raise ValueError("action_rate_hz must be positive")
    if ready_at < capture_at:
        raise ValueError("ready_at must not precede capture_at")
    if not execution_started:
        return 0, [
            (action, ready_at + index / action_rate_hz, index)
            for index, action in enumerate(actions)
        ]
    elapsed_steps = (ready_at - capture_at) * action_rate_hz
    discarded = min(len(actions), math.ceil(max(0.0, elapsed_steps - 1e-9)))
    aligned = [
        (action, capture_at + index / action_rate_hz, index)
        for index, action in enumerate(actions)
        if index >= discarded
    ]
    if any(target_at < ready_at - 1e-9 for _, target_at, _ in aligned):
        raise ValueError("aligned action target precedes inference completion")
    if any(
        right[1] <= left[1]
        for left, right in zip(aligned, aligned[1:], strict=False)
    ):
        raise ValueError("aligned action targets must increase monotonically")
    return discarded, aligned


def replace_action_queue(
    queue: deque[QueuedAction],
    actions: list[TimedAction],
    *,
    completed_at: float,
    event_index: int,
) -> int:
    """Replace an obsolete residual with one aligned future action chunk."""

    residual_actions = len(queue)
    queue.clear()
    queue.extend(
        (action, target_at, completed_at, event_index, chunk_index)
        for action, target_at, chunk_index in actions
    )
    return residual_actions


class RunnerPhase(str, Enum):
    RESET = "RESET"
    BASE_PREPOSITION = "BASE_PREPOSITION"
    SETTLE = "SETTLE"
    MANIPULATION_READY = "MANIPULATION_READY"
    PI05_MANIPULATION = "PI05_MANIPULATION"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class FreshnessConfig:
    camera_max_age_s: float = 0.65
    camera_max_skew_s: float = 0.55
    state_max_age_s: float = 0.15
    action_queue_max_age_s: float = 0.25


class FreshnessError(ValueError):
    """Freshness rejection carrying compact, machine-readable evidence."""

    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True)
class ReadinessConfig:
    target_x: float
    target_y: float
    target_yaw: float
    initial_spine_max_abs_m: float = 0.01
    position_tolerance_m: float = 0.05
    yaw_tolerance_rad: float = 0.10
    velocity_threshold: float = 0.02
    settle_duration_s: float = 1.0


def angular_error(left: float, right: float) -> float:
    """Smallest signed angular difference."""

    return math.atan2(math.sin(left - right), math.cos(left - right))


class BaseReadinessGate:
    """Track manual staging and invalidate readiness on reset/base input."""

    def __init__(self, config: ReadinessConfig):
        self.config = config
        self.phase = RunnerPhase.RESET
        self._settle_since: float | None = None
        self.last_evidence: dict[str, Any] = {}

    def reset(self, reason: str = "scene_reset") -> None:
        self.phase = RunnerPhase.BASE_PREPOSITION
        self._settle_since = None
        self.last_evidence = {"ready": False, "reason": reason}

    def note_base_input(self) -> None:
        if self.phase == RunnerPhase.PI05_MANIPULATION:
            self.stop("new_base_input")
        else:
            self.reset("new_base_input")

    def stop(self, reason: str) -> None:
        self.phase = RunnerPhase.STOPPED
        self._settle_since = None
        self.last_evidence = {"ready": False, "reason": reason}

    def update(
        self,
        now: float,
        pose: tuple[float, float, float],
        velocity: tuple[float, float, float],
        spine_position: float,
    ) -> bool:
        cfg = self.config
        dx = float(pose[0]) - cfg.target_x
        dy = float(pose[1]) - cfg.target_y
        position_error = math.hypot(dx, dy)
        yaw_error = abs(angular_error(float(pose[2]), cfg.target_yaw))
        speed = math.sqrt(sum(float(value) ** 2 for value in velocity))
        spine_abs = abs(float(spine_position))
        pose_within = (
            position_error <= cfg.position_tolerance_m
            and yaw_error <= cfg.yaw_tolerance_rad
        )
        initial_spine_within = spine_abs <= cfg.initial_spine_max_abs_m
        within = (
            pose_within
            and initial_spine_within
            and speed <= cfg.velocity_threshold
        )
        if self.phase == RunnerPhase.PI05_MANIPULATION:
            if pose_within:
                self.last_evidence = {
                    "ready": True,
                    "phase": self.phase.value,
                    "latched": True,
                    "actual_pose": list(pose),
                    "target_pose": [
                        cfg.target_x,
                        cfg.target_y,
                        cfg.target_yaw,
                    ],
                    "position_error_m": position_error,
                    "yaw_error_rad": yaw_error,
                    "speed": speed,
                    "spine_position_m": float(spine_position),
                    "spine_policy_controlled": True,
                }
                return True
            self.stop("base_pose_out_of_tolerance")
            return False
        if self.phase == RunnerPhase.STOPPED:
            return False
        if not within:
            self.phase = RunnerPhase.BASE_PREPOSITION
            self._settle_since = None
        elif self._settle_since is None:
            self.phase = RunnerPhase.SETTLE
            self._settle_since = now
        elif now - self._settle_since >= cfg.settle_duration_s:
            self.phase = RunnerPhase.MANIPULATION_READY
        else:
            self.phase = RunnerPhase.SETTLE
        self.last_evidence = {
            "ready": self.phase == RunnerPhase.MANIPULATION_READY,
            "phase": self.phase.value,
            "actual_pose": list(pose),
            "target_pose": [cfg.target_x, cfg.target_y, cfg.target_yaw],
            "position_error_m": position_error,
            "yaw_error_rad": yaw_error,
            "speed": speed,
            "spine_position_m": float(spine_position),
            "initial_spine_max_abs_m": cfg.initial_spine_max_abs_m,
            "initial_spine_within": initial_spine_within,
            "settled_for_s": 0.0
            if self._settle_since is None
            else max(0.0, now - self._settle_since),
        }
        return bool(self.last_evidence["ready"])

    def arm(self) -> None:
        if self.phase != RunnerPhase.MANIPULATION_READY:
            raise RuntimeError("base is not MANIPULATION_READY")
        self.phase = RunnerPhase.PI05_MANIPULATION
        self.last_evidence = {
            **self.last_evidence,
            "ready": True,
            "phase": self.phase.value,
            "latched": True,
        }


def validate_rgb_frame(key: str, array: Any) -> None:
    """Validate policy RGB shape without accepting evaluator camera keys."""

    if key not in ROBOT_CAMERA_KEYS:
        raise ValueError(f"unsupported policy camera: {key}")
    expected = tuple(EXPECTED_CAMERA_SHAPES[f"observation.images.{key}"])
    actual = tuple(int(value) for value in array.shape)
    if actual != expected:
        raise ValueError(f"{key} image shape {actual} != {expected}")
    if str(array.dtype) != "uint8":
        raise ValueError(f"{key} image dtype must be uint8, got {array.dtype}")


def freshness_metrics(
    *,
    now: float,
    camera_times: dict[str, float],
    camera_sequences: dict[str, int],
    state_time: float,
    last_camera_sequences: dict[str, int],
    config: FreshnessConfig,
) -> dict[str, Any]:
    """Require one new, fresh frame from each robot camera and fresh state."""

    missing = [key for key in ROBOT_CAMERA_KEYS if key not in camera_times]
    if missing:
        raise FreshnessError(
            f"missing robot camera frames: {missing}",
            {"offending_streams": missing, "missing_camera_frames": missing},
        )
    stale_sequences = [
        key
        for key in ROBOT_CAMERA_KEYS
        if camera_sequences.get(key, -1) <= last_camera_sequences.get(key, -1)
    ]
    ages = {
        key: max(0.0, now - camera_times[key]) for key in ROBOT_CAMERA_KEYS
    }
    skew = max(camera_times.values()) - min(camera_times.values())
    state_age = max(0.0, now - state_time)
    metrics = {
        "frame_age_s": ages,
        "inter_camera_skew_s": skew,
        "state_age_s": state_age,
        "camera_sequences": dict(camera_sequences),
    }
    offending = list(stale_sequences)
    offending.extend(
        key
        for key, age in ages.items()
        if age > config.camera_max_age_s and key not in offending
    )
    if skew > config.camera_max_skew_s:
        offending.append("camera_skew")
    if state_age > config.state_max_age_s:
        offending.append("state")
    if offending:
        evidence = {
            **metrics,
            "offending_streams": offending,
            "reused_camera_frames": stale_sequences,
        }
        raise FreshnessError(
            "freshness threshold exceeded: " + ", ".join(offending),
            evidence,
        )
    return metrics


def safe_action(
    raw_action: Any,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Project actuator targets while fixing base and preserving the spine.

    Gripper and spine outputs remain absolute policy dimensions. The simulator
    adapter clips their finite raw values to the demonstrated pilot ranges.
    Non-finite values are rejected before anything can be published.
    """

    raw = tuple(float(value) for value in raw_action)
    if len(raw) != ACTION_SIZE:
        raise ValueError(f"policy action must contain {ACTION_SIZE} values")
    if not all(math.isfinite(raw[index]) for index in (17, 18, 19)):
        raise ValueError("policy gripper and spine actions must be finite")
    spine_target = min(SPINE_POLICY_MAX_M, max(SPINE_POLICY_MIN_M, raw[19]))
    effective = list(apply_fixed_mobile_axes(raw, spine_height=spine_target))
    effective[17] = min(1.0, max(0.0, effective[17]))
    effective[18] = min(1.0, max(0.0, effective[18]))
    effective = list(project_arm_action_bounds(effective))
    validate_absolute_action_bounds(effective)
    return raw, tuple(effective)


class ActionWatchdog:
    """Reject stale action queues and hold instead of publishing targets."""

    def __init__(self, maximum_age_s: float):
        self.maximum_age_s = maximum_age_s

    def valid(self, *, now: float, created_at: float) -> bool:
        return 0.0 <= now - created_at <= self.maximum_age_s


def validate_live_state(state: Any) -> tuple[float, ...]:
    return finite_state(state)


def startup_inventory(
    *,
    camera_sequences: dict[str, int],
    joint_names: set[str],
    ee_available: dict[str, bool],
    odom_available: bool,
    state: Any,
) -> dict[str, Any]:
    """Describe whether every input required for one inference is present."""

    cameras = {
        key: camera_sequences.get(key, 0) > 0 for key in ROBOT_CAMERA_KEYS
    }
    required_joints = set((*LEFT_JOINTS, *RIGHT_JOINTS, SPINE_JOINT))
    missing_joints = sorted(required_joints - joint_names)
    vector = tuple(float(value) for value in state)
    invalid_state_indices = [
        index for index, value in enumerate(vector) if not math.isfinite(value)
    ]
    missing_inputs = [key for key, available in cameras.items() if not available]
    missing_inputs.extend(
        f"{side}_ee_pose"
        for side in ("left", "right")
        if not ee_available.get(side, False)
    )
    if not odom_available:
        missing_inputs.append("odom")
    if missing_joints:
        missing_inputs.append("joint_states")
    if invalid_state_indices:
        missing_inputs.append("finite_37d_state")
    return {
        "all_required_samples": not missing_inputs,
        "missing_inputs": missing_inputs,
        "camera_samples": cameras,
        "ee_pose_samples": dict(ee_available),
        "odom_sample": odom_available,
        "missing_joints": missing_joints,
        "invalid_state_indices": invalid_state_indices,
    }
