# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light validation and feedback checks for startup staging."""

from __future__ import annotations

import math
from typing import Any

from task2_isaacsim.baselines.pi05.contract import FR3_JOINT_LIMITS
from task2_isaacsim.baselines.pi05.pregrasp_pose_audit import (
    quaternion_angle_deg,
)


def _validate_command(command: tuple[float, ...], *, context: str) -> None:
    if len(command) != 17:
        raise ValueError(f"{context} must contain action indices 3:20")
    if not all(math.isfinite(value) for value in command):
        raise ValueError(f"{context} contains non-finite values")
    for side, values in (("left", command[0:7]), ("right", command[7:14])):
        for joint_index, (value, bounds) in enumerate(
            zip(values, FR3_JOINT_LIMITS, strict=True), start=1
        ):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f"{context} {side} joint {joint_index} is outside "
                    "FR3 bounds"
                )
    if not all(0.0 <= value <= 1.0 for value in command[14:16]):
        raise ValueError(f"{context} gripper is outside [0, 1]")
    if not 0.0 <= command[16] <= 0.6:
        raise ValueError(f"{context} spine is outside [0, 0.6] m")


def staging_entry_duration_s(
    audit: dict[str, Any], current: tuple[float, ...], target: tuple[float, ...]
) -> float:
    """Bound the measured-state-to-dataset-route transition in sim time."""

    _validate_command(current, context="measured staging entry")
    _validate_command(target, context="dataset staging entry target")
    risk = audit["velocity_risk"]
    arm_limits = tuple(float(value) for value in risk["arm_limits_rad_s"])
    if len(arm_limits) != 14 or not all(value > 0.0 for value in arm_limits):
        raise ValueError("staging audit must contain 14 positive arm limits")
    arm_fraction = float(risk["staging_arm_velocity_fraction"])
    spine_rate = float(risk["staging_spine_velocity_m_s"])
    gripper_rate = float(risk["staging_gripper_velocity_fraction_s"])
    if not 0.0 < arm_fraction <= 1.0:
        raise ValueError("staging arm velocity fraction must be within (0, 1]")
    if spine_rate <= 0.0 or gripper_rate <= 0.0:
        raise ValueError("staging spine and gripper rates must be positive")
    durations = [
        abs(after - before) / (limit * arm_fraction)
        for before, after, limit in zip(
            current[:14], target[:14], arm_limits, strict=True
        )
    ]
    durations.extend(
        abs(after - before) / gripper_rate
        for before, after in zip(current[14:16], target[14:16], strict=True)
    )
    durations.append(abs(target[16] - current[16]) / spine_rate)
    return max(durations, default=0.0)


def interpolate_staging_command(
    current: tuple[float, ...], target: tuple[float, ...], fraction: float
) -> list[float]:
    bounded = min(1.0, max(0.0, float(fraction)))
    return [
        before + (after - before) * bounded
        for before, after in zip(current, target, strict=True)
    ]


def staging_command_within_tolerance(
    audit: dict[str, Any],
    current: tuple[float, ...],
    target: tuple[float, ...],
) -> bool:
    """Check measured joint-space feedback before entry calibration."""

    _validate_command(current, context="measured staging command")
    _validate_command(target, context="dataset staging command")
    tolerance = audit["tolerances"]
    return bool(
        max(abs(a - b) for a, b in zip(current[:14], target[:14]))
        <= float(tolerance["arm_max_abs_rad"])
        and max(abs(a - b) for a, b in zip(current[14:16], target[14:16]))
        <= float(tolerance["gripper_open_fraction"])
        and abs(current[16] - target[16])
        <= float(tolerance["spine_abs_m"])
    )


def validate_staging_audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 2:
        raise ValueError("staging audit must use pad-relative schema version 2")
    if payload.get("guessed_ik_used") is not False:
        raise ValueError("staging audit must explicitly forbid guessed IK")
    selection = payload.get("selection", {})
    if selection.get("episode") is None or selection.get("frame") is None:
        raise ValueError("staging audit is missing episode/frame provenance")
    target = payload.get("final_target", {})
    left = tuple(float(value) for value in target.get("left_arm_rad", []))
    right = tuple(float(value) for value in target.get("right_arm_rad", []))
    if len(left) != 7 or len(right) != 7:
        raise ValueError("staging target must contain two 7-D arms")
    for side, values in (("left", left), ("right", right)):
        for index, (value, bounds) in enumerate(
            zip(values, FR3_JOINT_LIMITS, strict=True), start=1
        ):
            if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f"{side} staging joint {index} is outside FR3 bounds"
                )
    for key in (
        "left_gripper_open_fraction",
        "right_gripper_open_fraction",
    ):
        value = float(target[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be within [0, 1]")
    spine = float(target["spine_command_m"])
    if not 0.0 <= spine <= 0.6:
        raise ValueError("staging spine target must be within [0, 0.6] m")
    reference = target.get("measured_reference", {})
    pad_offset = tuple(
        float(value)
        for value in reference.get("right_ee_relative_to_thermalpad_m", [])
    )
    if len(pad_offset) != 3 or not all(math.isfinite(v) for v in pad_offset):
        raise ValueError(
            "staging audit must contain a finite right-EE-to-pad offset"
        )
    entry_reference = target.get("entry_calibration_reference", {})
    entry_pad_offset = tuple(
        float(value)
        for value in entry_reference.get(
            "right_ee_relative_to_thermalpad_m", []
        )
    )
    if len(entry_pad_offset) != 3 or not all(
        math.isfinite(v) for v in entry_pad_offset
    ):
        raise ValueError(
            "staging audit must contain a finite entry right-EE-to-pad offset"
        )
    pad_tolerance = float(
        payload.get("tolerances", {}).get(
            "right_ee_relative_to_thermalpad_m", math.nan
        )
    )
    if not math.isfinite(pad_tolerance) or pad_tolerance <= 0.0:
        raise ValueError(
            "staging audit must contain a positive pad-relative tolerance"
        )
    route = payload.get("trajectory", [])
    if not route:
        raise ValueError("staging trajectory is empty")
    scheduled = [float(row["scheduled_at_s"]) for row in route]
    if scheduled[0] != 0.0 or any(
        right_time <= left_time
        for left_time, right_time in zip(scheduled, scheduled[1:], strict=False)
    ):
        raise ValueError("staging trajectory schedule must increase from zero")
    if int(route[-1]["frame"]) != int(selection["frame"]):
        raise ValueError("staging route does not end at the selected frame")
    for route_index, row in enumerate(route):
        command = tuple(float(value) for value in row.get("command", []))
        _validate_command(command, context=f"staging route row {route_index}")
    return payload


def staging_feedback(
    *,
    audit: dict[str, Any],
    left_arm: tuple[float, ...],
    right_arm: tuple[float, ...],
    spine_m: float,
    left_gripper_open: float,
    right_gripper_open: float,
    left_ee: tuple[float, ...],
    right_ee: tuple[float, ...],
    thermalpad_position_m: tuple[float, ...],
    right_ee_pad_relative_calibration_m: tuple[float, ...],
) -> dict[str, Any]:
    reference = audit["final_target"]["measured_reference"]
    tolerance = audit["tolerances"]
    left_error = max(
        abs(actual - float(expected))
        for actual, expected in zip(
            left_arm, reference["left_arm_rad"], strict=True
        )
    )
    right_error = max(
        abs(actual - float(expected))
        for actual, expected in zip(
            right_arm, reference["right_arm_rad"], strict=True
        )
    )
    spine_error = abs(spine_m - float(reference["spine_m"]))
    left_gripper_error = abs(
        left_gripper_open - float(reference["left_gripper_open_fraction"])
    )
    right_gripper_error = abs(
        right_gripper_open - float(reference["right_gripper_open_fraction"])
    )
    left_ee_z_error = abs(left_ee[2] - float(reference["left_ee"][2]))
    actual_right_ee_relative_to_thermalpad = tuple(
        actual - pad
        for actual, pad in zip(
            right_ee[:3], thermalpad_position_m, strict=True
        )
    )
    expected_right_ee_relative_to_thermalpad = tuple(
        float(expected) + calibration
        for expected, calibration in zip(
            reference["right_ee_relative_to_thermalpad_m"],
            right_ee_pad_relative_calibration_m,
            strict=True,
        )
    )
    right_pad_relative_error = math.sqrt(
        sum(
            (actual - float(expected)) ** 2
            for actual, expected in zip(
                actual_right_ee_relative_to_thermalpad,
                expected_right_ee_relative_to_thermalpad,
                strict=True,
            )
        )
    )
    right_orientation_error = quaternion_angle_deg(
        list(right_ee[3:7]), list(reference["right_ee"][3:7])
    )
    groups = {
        "left_arm": left_error <= tolerance["arm_max_abs_rad"],
        "right_arm": right_error <= tolerance["arm_max_abs_rad"],
        "spine": spine_error <= tolerance["spine_abs_m"],
        "left_gripper": (
            left_gripper_error <= tolerance["gripper_open_fraction"]
        ),
        "right_gripper": (
            right_gripper_error <= tolerance["gripper_open_fraction"]
        ),
        "left_ee_height": left_ee_z_error <= tolerance["left_ee_z_m"],
        "right_camera_ready_pad_relative_position": (
            right_pad_relative_error
            <= tolerance["right_ee_relative_to_thermalpad_m"]
        ),
        "right_camera_ready_orientation": (
            right_orientation_error <= tolerance["right_ee_orientation_deg"]
        ),
    }
    return {
        "within_tolerance": all(groups.values()),
        "groups": groups,
        "errors": {
            "left_arm_max_abs_rad": left_error,
            "right_arm_max_abs_rad": right_error,
            "spine_abs_m": spine_error,
            "left_gripper_open_fraction": left_gripper_error,
            "right_gripper_open_fraction": right_gripper_error,
            "left_ee_z_m": left_ee_z_error,
            "right_ee_relative_to_thermalpad_m": right_pad_relative_error,
            "right_ee_orientation_deg": right_orientation_error,
        },
        "actual_right_ee_relative_to_thermalpad_m": list(
            actual_right_ee_relative_to_thermalpad
        ),
        "expected_right_ee_relative_to_thermalpad_m": list(
            expected_right_ee_relative_to_thermalpad
        ),
        "entry_pad_relative_calibration_m": list(
            right_ee_pad_relative_calibration_m
        ),
    }
