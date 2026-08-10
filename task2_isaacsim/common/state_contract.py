# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Import-safe assembly of the official Task 2 37-D robot state."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

LEFT_JOINTS = tuple(f"left_fr3v2_joint{i}" for i in range(1, 8))
RIGHT_JOINTS = tuple(f"right_fr3v2_joint{i}" for i in range(1, 8))
SPINE_JOINT = "franka_spine_vertical_joint"
LEFT_GRIPPER_DRIVER = "left_right_finger_joint"
RIGHT_GRIPPER_DRIVER = "right_right_finger_joint"
GRIPPER_CLOSED_RAD = 0.8
STATE_DIM = 37


def candidate_joint_names(name: str):
    """Yield the canonical joint name followed by known USD aliases."""

    yield name
    if "fr3v2_joint" in name:
        yield name.replace("fr3v2_joint", "fr3v2_1_joint")
    if name == LEFT_GRIPPER_DRIVER:
        yield "left_fr3v2_finger_joint1"
    if name == RIGHT_GRIPPER_DRIVER:
        yield "right_fr3v2_finger_joint1"


def resolve_joint(
    joint_map: Mapping[str, float], name: str, default: float = math.nan
) -> float:
    """Resolve a finite joint value through the official alias contract."""

    for candidate in candidate_joint_names(name):
        value = joint_map.get(candidate)
        if value is not None and math.isfinite(value):
            return float(value)
    return float(default)


def gripper_open_fraction(driver_position_rad: float) -> float:
    """Convert the Robotiq driver position to the official open fraction."""

    if not math.isfinite(driver_position_rad):
        return math.nan
    return min(1.0, max(0.0, 1.0 - driver_position_rad / GRIPPER_CLOSED_RAD))


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return yaw for a ROS xyzw quaternion."""

    return math.atan2(
        2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
    )


def assemble_state(snapshot: Mapping[str, object]) -> tuple[float, ...]:
    """Assemble the recorder/live-runner 37-D state from a common snapshot.

    ``odom`` follows the recorder tuple contract:
    ``x,y,z,qx,qy,qz,qw,vx,vy,vz,wz``.
    """

    state = [math.nan] * STATE_DIM
    ee_poses = snapshot.get("ee_poses") or {}
    assert isinstance(ee_poses, Mapping)
    for offset, side in ((0, "left"), (7, "right")):
        pose = ee_poses.get(side)
        if pose is not None:
            values = tuple(float(value) for value in pose)
            if len(values) != 7:
                raise ValueError(f"{side} EE pose must contain 7 values")
            state[offset : offset + 7] = values

    measured = snapshot.get("joint_states") or {}
    assert isinstance(measured, Mapping)
    for index, name in enumerate((*LEFT_JOINTS, *RIGHT_JOINTS)):
        state[14 + index] = resolve_joint(measured, name)
    state[28] = resolve_joint(measured, SPINE_JOINT, 0.0)
    state[29] = gripper_open_fraction(
        resolve_joint(measured, LEFT_GRIPPER_DRIVER, 0.0)
    )
    state[30] = gripper_open_fraction(
        resolve_joint(measured, RIGHT_GRIPPER_DRIVER, 0.0)
    )

    odom = snapshot.get("odom")
    if odom is not None:
        values = tuple(float(value) for value in odom)
        if len(values) != 11:
            raise ValueError("odom snapshot must contain 11 values")
        x, y, _, qx, qy, qz, qw, vx, vy, _, wz = values
        state[31:34] = (x, y, quat_to_yaw(qx, qy, qz, qw))
        state[34:37] = (vx, vy, wz)
    return tuple(state)


def finite_state(values: Sequence[float]) -> tuple[float, ...]:
    """Validate the exact 37-D finite live state boundary."""

    vector = tuple(float(value) for value in values)
    if len(vector) != STATE_DIM:
        raise ValueError(f"Task 2 state must contain {STATE_DIM} values")
    invalid = [i for i, value in enumerate(vector) if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"Task 2 state has non-finite indices {invalid}")
    return vector
