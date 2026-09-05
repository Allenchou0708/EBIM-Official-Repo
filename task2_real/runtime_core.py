"""Dependency-light Phase II real-robot handoff and action safety gates."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable


POLICY_COMMAND_TOPICS = {
    "right_arm": "/right/gello/joint_states",
    "right_gripper": "/right/gripper/gripper_client/target_gripper_width_percent",
}
DETERMINISTIC_COMMAND_TOPICS = {
    "base": "/swerve_drive_controller/cmd_vel",
    "spine": "/spine/target_height",
    "left_arm": "/left/gello/joint_states",
    "left_gripper": "/left/gripper/gripper_client/target_gripper_width_percent",
}
PRE_POLICY_PUBLISHER_COUNTS = {
    "base": 1,
    "spine": 1,
    "left_arm": 1,
    "left_gripper": 1,
    "right_arm": 0,
    "right_gripper": 0,
}
REQUIRED_CAPTURE_STREAMS = ("head", "wrist_right", "right_joints", "right_gripper")
RIGHT_ACTION_SIZE = 8
RIGHT_JOINT_COUNT = 7
MAX_POLICY_HORIZON = 5


class CalibrationError(ValueError):
    """The site profile is incomplete or cannot authorize actuation."""


class ActionSafetyError(ValueError):
    """A decoded right-only policy action is unsafe to publish."""


def _finite_vector(name: str, values: Iterable[Any], size: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _norm(values: Iterable[Any]) -> float:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        return math.inf
    return math.sqrt(sum(value * value for value in vector))


@dataclass(frozen=True)
class ActionLimits:
    joint_lower_rad: tuple[float, ...]
    joint_upper_rad: tuple[float, ...]
    max_joint_step_rad: float
    maximum_action_age_s: float
    external_joint_torque_max_abs_nm: float
    force_norm_max_n: float
    torque_norm_max_nm: float


def validate_site_profile(
    payload: dict[str, Any], *, require_verified: bool = True
) -> dict[str, Any]:
    """Validate one site profile without accepting simulator coordinates."""

    payload = copy.deepcopy(payload)
    if payload.get("schema_version") != 1:
        raise CalibrationError("site profile must use schema_version 1")
    if not str(payload.get("site", "")).strip():
        raise CalibrationError("site name is required")

    calibration = payload["calibration"]
    missing: list[str] = []
    base = calibration["base"]
    if not base.get("verified"):
        missing.append("base")
    waypoints = base.get("relative_waypoints") or []
    if base.get("verified") and not waypoints:
        raise CalibrationError("verified base calibration requires relative waypoints")
    for index, waypoint in enumerate(waypoints):
        _finite_vector(f"base.relative_waypoints[{index}]", waypoint, 3)
    if base.get("verified"):
        _finite_vector("base.target_relative", base["target_relative"], 3)

    spine = calibration["spine"]
    if not spine.get("verified") or not str(spine.get("unit", "")).strip():
        missing.append("spine")
    if spine.get("verified"):
        _finite_vector("spine.target", [spine["target"]], 1)
        _positive("spine.tolerance", spine["tolerance"])

    staging = calibration["staging"]
    _finite_vector(
        "staging.left_safe_hold_joint_rad",
        staging["left_safe_hold_joint_rad"],
        7,
    )
    _finite_vector(
        "staging.right_observation_joint_rad",
        staging["right_observation_joint_rad"],
        7,
    )
    for key in ("left_gripper_open_percent", "right_gripper_open_percent"):
        value = float(staging[key])
        if not 0.0 <= value <= 1.0:
            raise CalibrationError(f"staging.{key} must be within [0, 1]")
    if not staging.get("collision_checked"):
        missing.append("staging_collision_check")

    limits = payload["safety"]
    lower = limits.get("right_joint_lower_rad")
    upper = limits.get("right_joint_upper_rad")
    if lower is None or upper is None:
        missing.append("right_joint_limits")
    else:
        lower_vector = _finite_vector("safety.right_joint_lower_rad", lower, 7)
        upper_vector = _finite_vector("safety.right_joint_upper_rad", upper, 7)
        if any(
            low >= high
            for low, high in zip(lower_vector, upper_vector, strict=True)
        ):
            raise CalibrationError("every right joint lower limit must be below its upper limit")
    for key in (
        "max_right_joint_step_rad",
        "maximum_action_age_s",
        "external_joint_torque_max_abs_nm",
        "force_norm_max_n",
        "torque_norm_max_nm",
    ):
        if limits.get(key) is None:
            missing.append(key)
        else:
            _positive(f"safety.{key}", limits[key])

    runtime = payload["runtime"]
    for key in (
        "base_position_tolerance_m",
        "base_yaw_tolerance_rad",
        "base_linear_speed_max_mps",
        "base_angular_speed_max_rps",
        "base_settle_duration_s",
        "spine_velocity_max",
        "joint_position_tolerance_rad",
        "camera_max_age_s",
        "camera_max_skew_s",
        "state_max_age_s",
    ):
        _positive(f"runtime.{key}", runtime[key])

    if require_verified and missing:
        raise CalibrationError(
            "unverified site calibration: " + ", ".join(sorted(set(missing)))
        )
    payload["calibration_blockers"] = sorted(set(missing))
    return payload


def action_limits(profile: dict[str, Any]) -> ActionLimits:
    validated = validate_site_profile(profile, require_verified=True)
    safety = validated["safety"]
    return ActionLimits(
        joint_lower_rad=_finite_vector(
            "right_joint_lower_rad", safety["right_joint_lower_rad"], 7
        ),
        joint_upper_rad=_finite_vector(
            "right_joint_upper_rad", safety["right_joint_upper_rad"], 7
        ),
        max_joint_step_rad=float(safety["max_right_joint_step_rad"]),
        maximum_action_age_s=float(safety["maximum_action_age_s"]),
        external_joint_torque_max_abs_nm=float(safety["external_joint_torque_max_abs_nm"]),
        force_norm_max_n=float(safety["force_norm_max_n"]),
        torque_norm_max_nm=float(safety["torque_norm_max_nm"]),
    )


def force_stop_reasons(snapshot: dict[str, Any], limits: ActionLimits) -> list[str]:
    """Evaluate independent torque/wrench stop signals outside PI0.5."""

    reasons: list[str] = []
    external = _finite_vector(
        "right_external_joint_torques",
        snapshot["right_external_joint_torques_nm"],
        7,
    )
    force = _finite_vector(
        "right_external_force", snapshot["right_external_force_n"], 3
    )
    torque = _finite_vector(
        "right_external_torque", snapshot["right_external_torque_nm"], 3
    )
    if max(abs(value) for value in external) > limits.external_joint_torque_max_abs_nm:
        reasons.append("external_joint_torque_limit")
    if _norm(force) > limits.force_norm_max_n:
        reasons.append("external_force_limit")
    if _norm(torque) > limits.torque_norm_max_nm:
        reasons.append("external_torque_limit")
    return reasons


def evaluate_handoff(
    profile: dict[str, Any], snapshot: dict[str, Any], *, now: float
) -> dict[str, Any]:
    """Require a settled, fresh, contention-free handoff to right-only PI0.5."""

    validated = validate_site_profile(profile, require_verified=True)
    limits = action_limits(validated)
    runtime = validated["runtime"]
    staging = validated["calibration"]["staging"]
    spine = validated["calibration"]["spine"]
    reasons: list[str] = []

    scalar_checks = {
        "base_position": float(snapshot["base_position_error_m"])
        <= float(runtime["base_position_tolerance_m"]),
        "base_yaw": abs(float(snapshot["base_yaw_error_rad"]))
        <= float(runtime["base_yaw_tolerance_rad"]),
        "base_linear_speed": abs(float(snapshot["base_linear_speed_mps"]))
        <= float(runtime["base_linear_speed_max_mps"]),
        "base_angular_speed": abs(float(snapshot["base_angular_speed_rps"]))
        <= float(runtime["base_angular_speed_max_rps"]),
        "base_settled": float(snapshot["base_settled_for_s"])
        >= float(runtime["base_settle_duration_s"]),
        "spine_position": abs(float(snapshot["spine_position"]) - float(spine["target"]))
        <= float(spine["tolerance"]),
        "spine_velocity": abs(float(snapshot["spine_velocity"]))
        <= float(runtime["spine_velocity_max"]),
        "left_gripper_open": float(snapshot["left_gripper_open_fraction"]) >= 0.90,
        "right_gripper_open": float(snapshot["right_gripper_open_fraction"]) >= 0.90,
        "post_settle_observation": bool(snapshot["post_settle_observation"]),
    }
    reasons.extend(name for name, passed in scalar_checks.items() if not passed)

    tolerance = float(runtime["joint_position_tolerance_rad"])
    left = _finite_vector("left_joint_positions", snapshot["left_joint_positions_rad"], 7)
    right = _finite_vector("right_joint_positions", snapshot["right_joint_positions_rad"], 7)
    left_target = _finite_vector("left_safe_hold", staging["left_safe_hold_joint_rad"], 7)
    right_target = _finite_vector("right_observation", staging["right_observation_joint_rad"], 7)
    left_error = max(abs(a - b) for a, b in zip(left, left_target, strict=True))
    right_error = max(abs(a - b) for a, b in zip(right, right_target, strict=True))
    if left_error > tolerance:
        reasons.append("left_safe_hold")
    if right_error > tolerance:
        reasons.append("right_observation_pose")

    capture_times = {
        key: float(snapshot["capture_times_s"].get(key, -math.inf))
        for key in REQUIRED_CAPTURE_STREAMS
    }
    ages = {key: float(now) - value for key, value in capture_times.items()}
    camera_times = [capture_times["head"], capture_times["wrist_right"]]
    state_times = [capture_times["right_joints"], capture_times["right_gripper"]]
    camera_skew = max(camera_times) - min(camera_times)
    camera_age_limit = float(runtime["camera_max_age_s"])
    if any(
        not math.isfinite(value) or not 0.0 <= ages[key] <= camera_age_limit
        for key, value in (
            ("head", camera_times[0]),
            ("wrist_right", camera_times[1]),
        )
    ):
        reasons.append("camera_freshness")
    if camera_skew > float(runtime["camera_max_skew_s"]):
        reasons.append("camera_skew")
    state_age_limit = float(runtime["state_max_age_s"])
    if any(
        not math.isfinite(value) or not 0.0 <= ages[key] <= state_age_limit
        for key, value in (
            ("right_joints", state_times[0]),
            ("right_gripper", state_times[1]),
        )
    ):
        reasons.append("state_freshness")

    publisher_counts = {
        key: int(value) for key, value in snapshot["publisher_counts"].items()
    }
    contention = {
        key: {"expected": expected, "actual": publisher_counts.get(key, -1)}
        for key, expected in PRE_POLICY_PUBLISHER_COUNTS.items()
        if publisher_counts.get(key) != expected
    }
    if contention:
        reasons.append("publisher_contention")
    reasons.extend(force_stop_reasons(snapshot, limits))

    return {
        "ready": not reasons,
        "reasons": sorted(set(reasons)),
        "metrics": {
            "left_joint_max_error_rad": left_error,
            "right_joint_max_error_rad": right_error,
            "capture_age_s": ages,
            "camera_skew_s": camera_skew,
            "publisher_contention": contention,
        },
        "policy_command_topics": dict(POLICY_COMMAND_TOPICS),
        "deterministic_command_topics": dict(DETERMINISTIC_COMMAND_TOPICS),
    }


def safe_right_action_window(
    raw_actions: Iterable[Iterable[Any]],
    *,
    measured_joints_rad: Iterable[Any],
    created_at: float,
    now: float,
    limits: ActionLimits,
) -> list[tuple[float, ...]]:
    """Validate at most five postprocessed absolute right-arm actions."""

    age = float(now) - float(created_at)
    if not 0.0 <= age <= limits.maximum_action_age_s:
        raise ActionSafetyError("stale policy action window")
    actions = list(raw_actions)
    if not actions:
        raise ActionSafetyError("empty policy action window")
    actions = actions[:MAX_POLICY_HORIZON]
    previous = _finite_vector("measured_joints_rad", measured_joints_rad, 7)
    validated: list[tuple[float, ...]] = []
    for index, raw in enumerate(actions):
        action = _finite_vector(f"action[{index}]", raw, RIGHT_ACTION_SIZE)
        joints = action[:RIGHT_JOINT_COUNT]
        gripper = action[7]
        if not 0.0 <= gripper <= 1.0:
            raise ActionSafetyError(f"action[{index}] gripper target outside [0, 1]")
        for joint, (target, low, high, prior) in enumerate(
            zip(
                joints,
                limits.joint_lower_rad,
                limits.joint_upper_rad,
                previous,
                strict=True,
            )
        ):
            if not low <= target <= high:
                raise ActionSafetyError(f"action[{index}] joint {joint} outside calibrated limits")
            if abs(target - prior) > limits.max_joint_step_rad:
                raise ActionSafetyError(f"action[{index}] joint {joint} step exceeds limit")
        validated.append(action)
        previous = joints
    return validated
