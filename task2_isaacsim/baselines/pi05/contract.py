# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light Task 2 dataset and PI05 boundary contract.

This module deliberately imports only the Python standard library. It can be
tested on a developer workstation before the GPU LeRobot environment exists.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ACTION_SIZE = 20
STATE_SIZE = 37
PI05_ACTION_SIZE = 32
PI05_MODEL_REVISION = "338b5c22c12dbdd0d2ab19046802de2eb7696a6b"

BASE_ACTION_SLICE = slice(0, 3)
SPINE_ACTION_INDEX = 19

# Franka Research 3 joint limits from the official mobile FR3 Duo URDF. Both
# arms share the same seven limits. The small tolerance is only for floating
# point round-off at the policy/runtime boundary.
FR3_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.9007400167, 2.9007400167),
    (-1.8360900167, 1.8360900167),
    (-2.9007400167, 2.9007400167),
    (-3.0770200167, -0.1169370833),
    (-2.87630335, 2.87630335),
    (0.43982265, 4.62163335),
    (-3.05083335, 3.05083335),
)

# Task 2's action and state vectors are not index-aligned. ``None`` means the
# action is already expressed in the command space used at runtime and must
# remain absolute. Integer entries identify the state element used as the
# reference for both training-time delta conversion and inference-time inverse
# conversion.
RELATIVE_ACTION_STATE_INDICES: tuple[int | None, ...] = (
    None,
    None,
    None,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    None,
    None,
    28,
)

ACTION_NAMES = (
    "base.vx",
    "base.vy",
    "base.wz",
    "left_fr3v2_joint1.target",
    "left_fr3v2_joint2.target",
    "left_fr3v2_joint3.target",
    "left_fr3v2_joint4.target",
    "left_fr3v2_joint5.target",
    "left_fr3v2_joint6.target",
    "left_fr3v2_joint7.target",
    "right_fr3v2_joint1.target",
    "right_fr3v2_joint2.target",
    "right_fr3v2_joint3.target",
    "right_fr3v2_joint4.target",
    "right_fr3v2_joint5.target",
    "right_fr3v2_joint6.target",
    "right_fr3v2_joint7.target",
    "left_gripper.open_fraction.target",
    "right_gripper.open_fraction.target",
    "spine.height.target",
)

STATE_NAMES = (
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.qx",
    "left_ee.qy",
    "left_ee.qz",
    "left_ee.qw",
    "right_ee.x",
    "right_ee.y",
    "right_ee.z",
    "right_ee.qx",
    "right_ee.qy",
    "right_ee.qz",
    "right_ee.qw",
    "left_fr3v2_joint1.pos",
    "left_fr3v2_joint2.pos",
    "left_fr3v2_joint3.pos",
    "left_fr3v2_joint4.pos",
    "left_fr3v2_joint5.pos",
    "left_fr3v2_joint6.pos",
    "left_fr3v2_joint7.pos",
    "right_fr3v2_joint1.pos",
    "right_fr3v2_joint2.pos",
    "right_fr3v2_joint3.pos",
    "right_fr3v2_joint4.pos",
    "right_fr3v2_joint5.pos",
    "right_fr3v2_joint6.pos",
    "right_fr3v2_joint7.pos",
    "spine.height",
    "left_gripper.open_fraction",
    "right_gripper.open_fraction",
    "base.odom.x",
    "base.odom.y",
    "base.odom.yaw",
    "base.vel.vx",
    "base.vel.vy",
    "base.vel.wz",
)

POLICY_CAMERA_RENAME_MAP = {
    "observation.images.head": "observation.images.base_0_rgb",
    "observation.images.wrist_left": "observation.images.left_wrist_0_rgb",
    "observation.images.wrist_right": "observation.images.right_wrist_0_rgb",
}
EVAL_CAMERA_KEY = "observation.images.eval_camera"

EXPECTED_CAMERA_SHAPES = {
    "observation.images.head": [720, 1280, 3],
    "observation.images.wrist_left": [480, 848, 3],
    "observation.images.wrist_right": [480, 848, 3],
    EVAL_CAMERA_KEY: [720, 1280, 3],
}


@dataclass(frozen=True)
class Pi05Task2Contract:
    """Configuration choices shared by training and rollout."""

    policy_path: str = "lerobot/pi05_base"
    policy_revision: str = PI05_MODEL_REVISION
    task_instruction: str = (
        "Pick up the thermal pad and place it on the target RAM board."
    )
    max_state_dim: int = STATE_SIZE
    max_action_dim: int = PI05_ACTION_SIZE
    chunk_size: int = 50
    n_action_steps: int = 5
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    train_expert_only: bool = False
    freeze_vision_encoder: bool = False
    use_relative_actions: bool = True
    relative_action_state_indices: tuple[int | None, ...] = (
        RELATIVE_ACTION_STATE_INDICES
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FULL_FINETUNE_CONTRACT = Pi05Task2Contract()
SMOKE_EXPERT_CONTRACT = Pi05Task2Contract(
    train_expert_only=True,
    freeze_vision_encoder=True,
)
EXPERT_FINETUNE_CONTRACT = Pi05Task2Contract(
    train_expert_only=True,
    freeze_vision_encoder=True,
)

# The organizer-data baseline uses the action expert on the available 32 GiB
# GPU. Full fine-tuning remains an explicit 80 GB-class ablation.
PI05_CONTRACT = EXPERT_FINETUNE_CONTRACT


def _finite_vector(
    values: Sequence[float],
    expected_size: int,
    name: str,
) -> list[float]:
    if len(values) != expected_size:
        raise ValueError(
            f"{name} must have {expected_size} values, got {len(values)}"
        )

    vector = [float(value) for value in values]
    invalid = [
        index for index, value in enumerate(vector) if not math.isfinite(value)
    ]
    if invalid:
        raise ValueError(
            f"{name} contains non-finite values at indices {invalid}"
        )
    return vector


def pad_action(action: Sequence[float]) -> tuple[float, ...]:
    """Pad one official 20-D action to the PI05 32-D model boundary."""

    vector = _finite_vector(action, ACTION_SIZE, "Task 2 action")
    vector.extend([0.0] * (PI05_ACTION_SIZE - ACTION_SIZE))
    return tuple(vector)


def unpad_action(pi05_action: Sequence[float]) -> tuple[float, ...]:
    """Crop a PI05 32-D output back to the official Task 2 20-D action."""

    vector = _finite_vector(pi05_action, PI05_ACTION_SIZE, "PI05 action")
    return tuple(vector[:ACTION_SIZE])


def validate_relative_action_state_indices(
    indices: Sequence[int | None],
) -> tuple[int | None, ...]:
    """Validate and freeze an explicit action-to-state mapping."""

    if len(indices) != ACTION_SIZE:
        raise ValueError(
            "relative action mapping must have "
            f"{ACTION_SIZE} entries, got {len(indices)}"
        )

    validated: list[int | None] = []
    for action_index, state_index in enumerate(indices):
        if state_index is None:
            validated.append(None)
            continue
        if isinstance(state_index, bool) or not isinstance(state_index, int):
            raise ValueError(
                "relative action mapping entries must be integer state "
                f"indices or null; action {action_index} got {state_index!r}"
            )
        if not 0 <= state_index < STATE_SIZE:
            raise ValueError(
                "relative action mapping for action "
                f"{action_index} references "
                f"state {state_index}, outside 0..{STATE_SIZE - 1}"
            )
        validated.append(state_index)
    return tuple(validated)


def to_relative_action(
    action: Sequence[float],
    state: Sequence[float],
    *,
    state_indices: Sequence[int | None] = RELATIVE_ACTION_STATE_INDICES,
) -> tuple[float, ...]:
    """Convert one official 20-D Task 2 command to the mapped delta space."""

    action_vector = _finite_vector(action, ACTION_SIZE, "Task 2 action")
    state_vector = _finite_vector(state, STATE_SIZE, "Task 2 state")
    mapping = validate_relative_action_state_indices(state_indices)
    for action_index, state_index in enumerate(mapping):
        if state_index is not None:
            action_vector[action_index] -= state_vector[state_index]
    return tuple(action_vector)


def to_absolute_action(
    relative_action: Sequence[float],
    state: Sequence[float],
    *,
    state_indices: Sequence[int | None] = RELATIVE_ACTION_STATE_INDICES,
) -> tuple[float, ...]:
    """Invert :func:`to_relative_action` for policy output publication."""

    action_vector = _finite_vector(
        relative_action, ACTION_SIZE, "Task 2 relative action"
    )
    state_vector = _finite_vector(state, STATE_SIZE, "Task 2 state")
    mapping = validate_relative_action_state_indices(state_indices)
    for action_index, state_index in enumerate(mapping):
        if state_index is not None:
            action_vector[action_index] += state_vector[state_index]
    return tuple(action_vector)


def apply_fixed_mobile_axes(
    action: Sequence[float],
    *,
    spine_height: float,
) -> tuple[float, ...]:
    """Hold base and spine fixed for the first safe closed-loop pilot.

    Arm and gripper targets remain unchanged. The spine value must be the
    current or reset target supplied by the runtime, not a hard-coded height.
    """

    vector = _finite_vector(action, ACTION_SIZE, "Task 2 action")
    if not math.isfinite(spine_height):
        raise ValueError("spine_height must be finite")
    vector[BASE_ACTION_SLICE] = [0.0, 0.0, 0.0]
    vector[SPINE_ACTION_INDEX] = float(spine_height)
    return tuple(vector)


def validate_absolute_action_bounds(
    action: Sequence[float],
    *,
    tolerance: float = 1e-5,
) -> tuple[float, ...]:
    """Validate the publishable 20-D Task 2 command boundary."""

    vector = _finite_vector(action, ACTION_SIZE, "Task 2 action")
    for arm_offset, arm_name in ((3, "left"), (10, "right")):
        for joint_index, (lower, upper) in enumerate(FR3_JOINT_LIMITS):
            value = vector[arm_offset + joint_index]
            if not lower - tolerance <= value <= upper + tolerance:
                raise ValueError(
                    f"{arm_name} joint {joint_index + 1} command {value} "
                    f"is outside [{lower}, {upper}]"
                )
    for index, name in ((17, "left"), (18, "right")):
        value = vector[index]
        if not -tolerance <= value <= 1.0 + tolerance:
            raise ValueError(
                f"{name} gripper command {value} is outside [0, 1]"
            )
    return tuple(vector)


def _validate_feature(
    features: dict[str, Any],
    key: str,
    *,
    dtype: str,
    shape: list[int],
    names: tuple[str, ...] | None = None,
) -> list[str]:
    errors: list[str] = []
    feature = features.get(key)
    if not isinstance(feature, dict):
        return [f"missing feature: {key}"]
    if feature.get("dtype") != dtype:
        errors.append(
            f"{key}.dtype must be {dtype!r}, got {feature.get('dtype')!r}"
        )
    if feature.get("shape") != shape:
        errors.append(
            f"{key}.shape must be {shape}, got {feature.get('shape')!r}"
        )
    if names is not None and feature.get("names") != list(names):
        errors.append(
            f"{key}.names do not match the official ordered contract"
        )
    return errors


def validate_info(info: dict[str, Any]) -> list[str]:
    """Return every Task 2/PI05 contract violation found in ``info.json``."""

    errors: list[str] = []
    if info.get("codebase_version") != "v3.0":
        errors.append("codebase_version must be 'v3.0'")
    if info.get("fps") != 30:
        errors.append("fps must be 30")
    if info.get("robot_type") != "fr3duo_mobile_task2":
        errors.append("robot_type must be 'fr3duo_mobile_task2'")

    features = info.get("features")
    if not isinstance(features, dict):
        return [*errors, "features must be a JSON object"]

    errors.extend(
        _validate_feature(
            features,
            "action",
            dtype="float32",
            shape=[ACTION_SIZE],
            names=ACTION_NAMES,
        )
    )
    errors.extend(
        _validate_feature(
            features,
            "observation.state",
            dtype="float32",
            shape=[STATE_SIZE],
            names=STATE_NAMES,
        )
    )
    for key, shape in EXPECTED_CAMERA_SHAPES.items():
        errors.extend(
            _validate_feature(features, key, dtype="video", shape=shape)
        )

    policy_inputs = set(POLICY_CAMERA_RENAME_MAP)
    if EVAL_CAMERA_KEY in policy_inputs:
        errors.append("eval_camera must not be a policy input")
    camera_destinations = set(POLICY_CAMERA_RENAME_MAP.values())
    if len(camera_destinations) != len(POLICY_CAMERA_RENAME_MAP):
        errors.append("policy camera destinations must be unique")
    return errors


def validate_dataset_root(
    dataset_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    """Load and validate ``meta/info.json`` from a local LeRobot dataset."""

    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return {}, [f"missing dataset metadata: {info_path}"]

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"unable to read {info_path}: {error}"]
    if not isinstance(info, dict):
        return {}, [f"{info_path} must contain a JSON object"]
    errors = validate_info(info)
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        return info, [*errors, f"missing dataset statistics: {stats_path}"]

    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return info, [*errors, f"unable to read {stats_path}: {error}"]
    if not isinstance(stats, dict):
        return info, [*errors, f"{stats_path} must contain a JSON object"]

    for key, expected_size in (
        ("action", ACTION_SIZE),
        ("observation.state", STATE_SIZE),
    ):
        feature_stats = stats.get(key)
        if not isinstance(feature_stats, dict):
            errors.append(f"missing quantile statistics: {key}")
            continue
        quantiles: dict[str, list[float]] = {}
        for quantile in ("q01", "q99"):
            raw_values = feature_stats.get(quantile)
            if not isinstance(raw_values, list):
                errors.append(f"missing quantile statistics: {key}.{quantile}")
                continue
            try:
                values = [float(value) for value in raw_values]
            except (TypeError, ValueError):
                errors.append(f"{key}.{quantile} must contain numeric values")
                continue
            if len(values) != expected_size:
                errors.append(
                    f"{key}.{quantile} must have {expected_size} values, "
                    f"got {len(values)}"
                )
                continue
            if not all(math.isfinite(value) for value in values):
                errors.append(f"{key}.{quantile} contains non-finite values")
                continue
            quantiles[quantile] = values
        if quantiles.keys() >= {"q01", "q99"} and any(
            low > high
            for low, high in zip(
                quantiles["q01"], quantiles["q99"], strict=True
            )
        ):
            errors.append(f"{key} has q01 values greater than q99")
    return info, errors
