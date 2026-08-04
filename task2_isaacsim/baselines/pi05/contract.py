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

BASE_ACTION_SLICE = slice(0, 3)
SPINE_ACTION_INDEX = 19

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
    task_instruction: str = (
        "Pick up the thermal pad and place it on the target RAM board."
    )
    max_state_dim: int = STATE_SIZE
    max_action_dim: int = PI05_ACTION_SIZE
    chunk_size: int = 50
    n_action_steps: int = 5
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    train_expert_only: bool = True
    use_relative_actions: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PI05_CONTRACT = Pi05Task2Contract()


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
        index
        for index, value in enumerate(vector)
        if not math.isfinite(value)
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
    return info, validate_info(info)
