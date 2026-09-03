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
    FR3_JOINT_LIMITS,
    apply_fixed_mobile_axes,
    project_arm_action_bounds,
    validate_absolute_action_bounds,
)
from task2_isaacsim.common.state_contract import (
    GRIPPER_CLOSED_RAD,
    LEFT_JOINTS,
    RIGHT_JOINTS,
    SPINE_JOINT,
    finite_state,
)

ROBOT_CAMERA_KEYS = ("head", "wrist_left", "wrist_right")
HARD5_EXECUTION_HORIZON = 5
SPINE_POLICY_MIN_M = 0.0
SPINE_POLICY_MAX_M = 0.6
FR3_JOINT_VELOCITY_LIMITS_RAD_S = (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)
RIGHT_ARM_SLEW_VELOCITY_FRACTION = 0.50

# Development-only envelopes measured over all 90,028 manipulation frames
# from the 180-episode split.  The workspace adds roughly 3 cm beyond the
# observed extrema; the acquisition/release gates add margin around their
# corresponding event distributions.  They are simulator-world coordinates,
# so the real-robot adapter must derive equivalent gates in its task frame.
RIGHT_EE_WORKSPACE_MIN_XYZ = (1.70, 1.86, 0.82)
RIGHT_EE_WORKSPACE_MAX_XYZ = (2.21, 2.23, 1.17)
RIGHT_GRASP_GATE_MIN_XYZ = (1.70, 2.10, 0.82)
RIGHT_GRASP_GATE_MAX_XYZ = (1.80, 2.22, 0.93)
RIGHT_RELEASE_GATE_MIN_XYZ = (2.08, 1.97, 0.87)
RIGHT_RELEASE_GATE_MAX_XYZ = (2.19, 2.12, 0.98)
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


def gripper_open_fraction_command(value: float) -> tuple[float]:
    """Convert semantic open fraction to the direct driver-joint command.

    The live runner publishes ``/isaac/*_robotiq_joint_commands``, whose
    payload is a joint position in radians.  It does not publish the separate
    cartesian-control ``*_gripper_open_fraction_target`` interface.
    """

    target = float(value)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("gripper open fraction must be within [0, 1]")
    return ((1.0 - target) * GRIPPER_CLOSED_RAD,)


def _xyz_in_box(
    xyz: Any,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> bool:
    values = tuple(float(value) for value in xyz[:3])
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        return False
    return all(
        lower <= value <= upper
        for value, lower, upper in zip(values, minimum, maximum, strict=True)
    )


def right_ee_within_demonstrated_workspace(pose: Any) -> bool:
    """Return whether the current right EE remains near demonstrated motion."""

    return _xyz_in_box(
        pose,
        RIGHT_EE_WORKSPACE_MIN_XYZ,
        RIGHT_EE_WORKSPACE_MAX_XYZ,
    )


@dataclass
class RightGraspGuard:
    """Latch one grasp and allow one release only in demonstrated regions."""

    close_confirm_actions: int = 3
    minimum_hold_actions: int = 60
    release_confirm_actions: int = 3
    close_threshold: float = 0.20
    open_threshold: float = 0.80
    phase: str = "pregrasp"
    close_evidence_actions: int = 0
    held_actions: int = 0
    release_evidence_actions: int = 0
    interventions: int = 0

    def apply(
        self,
        requested_open_fraction: float,
        ee_pose: Any,
    ) -> tuple[float, dict[str, Any]]:
        requested = float(requested_open_fraction)
        if not math.isfinite(requested) or not 0.0 <= requested <= 1.0:
            raise ValueError("requested gripper open fraction must be in [0, 1]")
        xyz = tuple(float(value) for value in ee_pose[:3])
        if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
            raise ValueError("right EE pose must provide three finite coordinates")

        before = self.phase
        in_grasp_gate = _xyz_in_box(
            xyz, RIGHT_GRASP_GATE_MIN_XYZ, RIGHT_GRASP_GATE_MAX_XYZ
        )
        in_release_gate = _xyz_in_box(
            xyz, RIGHT_RELEASE_GATE_MIN_XYZ, RIGHT_RELEASE_GATE_MAX_XYZ
        )
        effective = requested
        reason = "policy_command_allowed"

        if self.phase == "pregrasp":
            if requested <= self.close_threshold:
                if in_grasp_gate:
                    self.close_evidence_actions += 1
                    if self.close_evidence_actions >= self.close_confirm_actions:
                        self.phase = "latched"
                        self.held_actions = 1
                else:
                    self.close_evidence_actions = 0
                    effective = 1.0
                    reason = "close_blocked_outside_grasp_gate"
            else:
                self.close_evidence_actions = 0
        elif self.phase == "latched":
            self.held_actions += 1
            if (
                requested >= self.open_threshold
                and self.held_actions >= self.minimum_hold_actions
                and in_release_gate
            ):
                self.release_evidence_actions += 1
                if self.release_evidence_actions >= self.release_confirm_actions:
                    self.phase = "released"
                    effective = 1.0
                    reason = "release_confirmed_in_gate"
                else:
                    effective = 0.0
                    reason = "release_confirmation_pending"
            else:
                self.release_evidence_actions = 0
                effective = 0.0
                if requested > self.close_threshold:
                    reason = "open_blocked_while_latched"
        elif self.phase == "released":
            effective = 1.0
            if requested < self.open_threshold:
                reason = "reclose_blocked_after_release"
        else:
            raise RuntimeError(f"unknown grasp guard phase: {self.phase}")

        intervened = abs(effective - requested) > 1e-9
        if intervened:
            self.interventions += 1
        return effective, {
            "phase_before": before,
            "phase_after": self.phase,
            "phase_changed": before != self.phase,
            "requested_open_fraction": requested,
            "effective_open_fraction": effective,
            "intervened": intervened,
            "reason": reason,
            "ee_world_xyz": list(xyz),
            "in_grasp_gate": in_grasp_gate,
            "in_release_gate": in_release_gate,
            "held_actions": self.held_actions,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "close_evidence_actions": self.close_evidence_actions,
            "held_actions": self.held_actions,
            "release_evidence_actions": self.release_evidence_actions,
            "interventions": self.interventions,
            "workspace_min_xyz": list(RIGHT_EE_WORKSPACE_MIN_XYZ),
            "workspace_max_xyz": list(RIGHT_EE_WORKSPACE_MAX_XYZ),
            "grasp_gate_min_xyz": list(RIGHT_GRASP_GATE_MIN_XYZ),
            "grasp_gate_max_xyz": list(RIGHT_GRASP_GATE_MAX_XYZ),
            "release_gate_min_xyz": list(RIGHT_RELEASE_GATE_MIN_XYZ),
            "release_gate_max_xyz": list(RIGHT_RELEASE_GATE_MAX_XYZ),
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


def hard5_action_window(
    actions: list[tuple[float, ...]],
    *,
    ready_at: float,
    action_rate_hz: float,
    execution_horizon: int = HARD5_EXECUTION_HORIZON,
    max_actions: int = HARD5_EXECUTION_HORIZON,
) -> list[TimedAction]:
    """Schedule one bounded synchronous execution window."""

    if action_rate_hz <= 0.0:
        raise ValueError("action_rate_hz must be positive")
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    if not 0 < execution_horizon <= len(actions):
        raise ValueError("execution_horizon must be within the action chunk")
    count = min(execution_horizon, max_actions, len(actions))
    if count == 0:
        raise ValueError("hard5 requires at least one action")
    return [
        (actions[index], ready_at + index / action_rate_hz, index)
        for index in range(count)
    ]


def hard5_hold_action(
    last_policy_action: tuple[float, ...] | None,
    *,
    inference_pending: bool,
    queue: deque[QueuedAction],
) -> tuple[float, ...] | None:
    """Hold the last absolute target while synchronous inference runs."""

    if not inference_pending or queue or last_policy_action is None:
        return None
    return last_policy_action


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
    camera_max_skew_s: float = 0.10
    state_max_age_s: float = 0.15
    observation_max_skew_s: float = 0.10
    action_queue_max_age_s: float = 0.25


class FreshnessError(ValueError):
    """Freshness rejection carrying compact, machine-readable evidence."""

    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


REQUIRED_STATE_STREAMS = ("joints", "odom", "ee_left", "ee_right")


@dataclass(frozen=True)
class ReadinessConfig:
    target_x: float
    target_y: float
    target_yaw: float
    initial_spine_target_m: float = 0.0
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
        spine_error = abs(float(spine_position) - cfg.initial_spine_target_m)
        pose_within = (
            position_error <= cfg.position_tolerance_m
            and yaw_error <= cfg.yaw_tolerance_rad
        )
        initial_spine_within = spine_error <= cfg.initial_spine_max_abs_m
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
            "initial_spine_target_m": cfg.initial_spine_target_m,
            "initial_spine_error_m": spine_error,
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
    capture_now: float | None = None,
    camera_times: dict[str, float],
    camera_capture_times: dict[str, float] | None = None,
    camera_sequences: dict[str, int],
    state_time: float,
    last_camera_sequences: dict[str, int],
    config: FreshnessConfig,
    state_times: dict[str, float] | None = None,
    state_capture_times: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Require new cameras and a capture-aligned, fresh robot state.

    The live runner gates ages and alignment on message-header simulator time.
    Host arrival ages remain diagnostic only. The legacy path without
    ``capture_now`` remains for dependency-light callers and tests.
    """

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
    arrival_ages = {
        key: max(0.0, now - camera_times[key]) for key in ROBOT_CAMERA_KEYS
    }
    capture_times = camera_capture_times or camera_times
    missing_capture_times = [
        key for key in ROBOT_CAMERA_KEYS if key not in capture_times
    ]
    if missing_capture_times:
        raise FreshnessError(
            f"missing robot camera capture times: {missing_capture_times}",
            {
                "offending_streams": missing_capture_times,
                "missing_camera_capture_times": missing_capture_times,
            },
        )
    ages = (
        {
            key: max(0.0, capture_now - capture_times[key])
            for key in ROBOT_CAMERA_KEYS
        }
        if capture_now is not None
        else arrival_ages
    )
    arrival_skew = max(camera_times.values()) - min(camera_times.values())
    skew = max(capture_times.values()) - min(capture_times.values())
    state_arrival_ages = (
        {
            key: max(0.0, now - received_at)
            for key, received_at in state_times.items()
        }
        if state_times is not None
        else {"state": max(0.0, now - state_time)}
    )
    missing_state_times = (
        [key for key in REQUIRED_STATE_STREAMS if key not in state_times]
        if state_times is not None
        else []
    )
    missing_state_capture_times = (
        [
            key
            for key in REQUIRED_STATE_STREAMS
            if key not in state_capture_times
        ]
        if state_capture_times is not None
        else []
    )
    state_ages = (
        {
            key: max(0.0, capture_now - captured_at)
            for key, captured_at in state_capture_times.items()
        }
        if capture_now is not None and state_capture_times is not None
        else state_arrival_ages
    )
    state_age = max(state_ages.values(), default=math.inf)
    state_capture_skew = None
    observation_capture_skew = None
    if state_capture_times is not None:
        state_capture_skew = (
            max(state_capture_times.values())
            - min(state_capture_times.values())
            if state_capture_times
            else math.inf
        )
        all_capture_times = {
            **{f"camera.{key}": value for key, value in capture_times.items()},
            **{
                f"state.{key}": value
                for key, value in state_capture_times.items()
            },
        }
        observation_capture_skew = (
            max(all_capture_times.values()) - min(all_capture_times.values())
            if all_capture_times
            else math.inf
        )
    metrics = {
        "frame_age_s": ages,
        "frame_arrival_age_s": arrival_ages,
        "inter_camera_skew_s": skew,
        "inter_camera_capture_skew_s": skew,
        "inter_camera_arrival_skew_s": arrival_skew,
        "camera_capture_times_s": {
            key: capture_times[key] for key in ROBOT_CAMERA_KEYS
        },
        "state_age_s": state_age,
        "state_age_by_stream_s": state_ages,
        "state_arrival_age_by_stream_s": state_arrival_ages,
        "state_capture_times_s": dict(state_capture_times or {}),
        "missing_state_times": missing_state_times,
        "missing_state_capture_times": missing_state_capture_times,
        "inter_state_capture_skew_s": state_capture_skew,
        "observation_capture_skew_s": observation_capture_skew,
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
    if missing_state_times or missing_state_capture_times:
        offending.append("state_streams_missing")
    if (
        observation_capture_skew is not None
        and observation_capture_skew > config.observation_max_skew_s
    ):
        offending.append("observation_skew")
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


def project_fr3_joint_step(
    previous: Any,
    target: Any,
    *,
    action_rate_hz: float,
    velocity_fraction: float = RIGHT_ARM_SLEW_VELOCITY_FRACTION,
) -> tuple[float, ...]:
    """Bound one seven-joint absolute target by FR3 position and slew limits."""

    before = tuple(float(value) for value in previous)
    requested = tuple(float(value) for value in target)
    if len(before) != 7 or len(requested) != 7:
        raise ValueError("FR3 joint step requires seven previous and target values")
    if not all(math.isfinite(value) for value in (*before, *requested)):
        raise ValueError("FR3 joint step must be finite")
    if action_rate_hz <= 0.0:
        raise ValueError("action_rate_hz must be positive")
    if not 0.0 < velocity_fraction <= 1.0:
        raise ValueError("velocity_fraction must be within (0, 1]")

    projected = []
    for current, desired, position_bounds, velocity_limit in zip(
        before,
        requested,
        FR3_JOINT_LIMITS,
        FR3_JOINT_VELOCITY_LIMITS_RAD_S,
        strict=True,
    ):
        desired = min(position_bounds[1], max(position_bounds[0], desired))
        maximum_step = velocity_limit * velocity_fraction / action_rate_hz
        projected.append(
            min(current + maximum_step, max(current - maximum_step, desired))
        )
    return tuple(projected)


def apply_right_only_policy_ownership(
    effective_action: Any,
    staging_command: Any,
) -> tuple[float, ...]:
    """Keep deterministic staging owners fixed while PI0.5 owns the right side.

    ``staging_command`` follows the existing manipulation-stager action[3:20]
    layout. Only right arm action[10:17] and right gripper action[18] survive
    from the policy output.
    """

    action = tuple(float(value) for value in effective_action)
    hold = tuple(float(value) for value in staging_command)
    if len(action) != 20:
        raise ValueError("effective action must contain 20 values")
    if len(hold) != 17:
        raise ValueError("staging hold command must contain action[3:20]")
    if not all(math.isfinite(value) for value in (*action, *hold)):
        raise ValueError("right-only ownership inputs must be finite")
    owned = list(action)
    owned[0:3] = [0.0, 0.0, 0.0]
    owned[3:10] = hold[0:7]
    owned[17] = hold[14]
    owned[19] = hold[16]
    validate_absolute_action_bounds(owned)
    return tuple(owned)


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
