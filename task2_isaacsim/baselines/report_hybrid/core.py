# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure geometry, policy-boundary, and FSM logic for report_hybrid.

This module intentionally has no ROS or simulator dependency.  The runtime
node uses these routines, while focused tests can disprove the safety and
retarget contracts without starting Isaac Sim.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REFERENCE_PATH = Path(__file__).with_name("reference_library_v1.json")

FORBIDDEN_TOPIC_FRAGMENTS = (
    "/isaac/eval_camera",
    "/evaluate_task2",
    "/isaac/task2/object_poses",
    "/isaac/task2/pad_points",
    "deformable",
    "ground_truth",
)

PHASES = (
    "observe",
    "approach",
    "insert",
    "acquire",
    "peel_lift",
    "transfer_place",
    "release",
    "retreat",
    "done",
)


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data["source"]
    if source["support_unique_episodes"] < 180:
        raise ValueError(
            "reference must be supported by 180 development episodes"
        )
    if len(source["support_by_collection"]) < 4:
        raise ValueError(
            "reference must cover all four development collections"
        )
    if source["episode_19_special_case"] or source["legacy_duplicates_used"]:
        raise ValueError(
            "episode 19 and legacy duplicates may not own the reference"
        )
    required = set(PHASES[1:-1]) - {"transfer_place"}
    required.add("transfer")
    required.add("place")
    if not required.issubset(data["right_ee_landmarks_xyzw"]):
        raise ValueError("reference landmark library is incomplete")
    return data


def assert_policy_topics(topics: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(topic) for topic in topics)
    bad = [
        topic
        for topic in values
        if any(
            fragment in topic.lower() for fragment in FORBIDDEN_TOPIC_FRAGMENTS
        )
    ]
    if bad:
        raise ValueError(f"forbidden runtime policy topics: {sorted(bad)}")
    return values


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compose_initial_relative_xyyaw(
    initial: Sequence[float], relative: Sequence[float]
) -> tuple[float, float, float]:
    """Compose a base target in the latched initial odometry frame."""
    x, y, yaw = (float(value) for value in initial)
    dx, dy, dyaw = (float(value) for value in relative)
    return (
        x + math.cos(yaw) * dx - math.sin(yaw) * dy,
        y + math.sin(yaw) * dx + math.cos(yaw) * dy,
        wrap_angle(yaw + dyaw),
    )


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((x, y, z, w)))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_multiply_xyzw(
    a: Sequence[float], b: Sequence[float]
) -> tuple[float, ...]:
    ax, ay, az, aw = (float(value) for value in a)
    bx, by, bz, bw = (float(value) for value in b)
    result = np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )
    result /= np.linalg.norm(result)
    return tuple(float(value) for value in result)


def rotate_quaternion_about_world_z(
    quaternion: Sequence[float], yaw_delta: float
) -> tuple[float, ...]:
    half = 0.5 * float(yaw_delta)
    return quaternion_multiply_xyzw(
        (0.0, 0.0, math.sin(half), math.cos(half)), quaternion
    )


def deproject_masked_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: Sequence[float],
    camera_pose_xyzw: Sequence[float],
    *,
    minimum_depth_m: float,
    maximum_depth_m: float,
    stride: int = 2,
) -> np.ndarray:
    """Return masked world points using ROS optical-axis conventions."""
    depth = np.asarray(depth_m, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or selected.shape != depth.shape:
        raise ValueError("depth and mask must be matching HxW arrays")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if min(fx, fy) <= 0.0:
        raise ValueError("invalid camera intrinsics")
    valid = (
        selected
        & np.isfinite(depth)
        & (depth >= minimum_depth_m)
        & (depth <= maximum_depth_m)
    )
    vv, uu = np.nonzero(valid)
    if stride > 1:
        vv, uu = vv[::stride], uu[::stride]
    if len(uu) < 8:
        raise ValueError("insufficient masked depth pixels")
    z = depth[vv, uu]
    optical = np.column_stack(((uu - cx) * z / fx, (vv - cy) * z / fy, z))
    pose = np.asarray(camera_pose_xyzw, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError("camera pose must be xyz+xyzw")
    rotation = quaternion_xyzw_to_matrix(pose[3:])
    return optical @ rotation.T + pose[:3]


def robust_center_yaw(points_world: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError("at least eight XYZ points are required")
    lower = np.quantile(points, 0.10, axis=0)
    upper = np.quantile(points, 0.90, axis=0)
    clipped = points[np.all((points >= lower) & (points <= upper), axis=1)]
    if len(clipped) < 8:
        clipped = points
    center = np.median(clipped, axis=0)
    xy = clipped[:, :2] - center[:2]
    covariance = xy.T @ xy / max(1, len(xy) - 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    yaw = math.atan2(float(axis[1]), float(axis[0]))
    # A rectangle has pi ambiguity; select the benchmark's +Y-ish axis.
    if abs(wrap_angle(yaw - math.pi / 2)) > abs(wrap_angle(yaw + math.pi / 2)):
        yaw = wrap_angle(yaw + math.pi)
    return center, yaw


def red_target_mask(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("RGB image must be HxWx3")
    red, green, blue = [
        image[..., index].astype(np.int16) for index in range(3)
    ]
    return (red >= 95) & (red - green >= 25) & (red - blue >= 25)


def dark_pad_mask(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("RGB image must be HxWx3")
    channels = image[..., :3].astype(np.int16)
    maximum = channels.max(axis=2)
    minimum = channels.min(axis=2)
    height, width = maximum.shape
    yy, xx = np.ogrid[:height, :width]
    roi = (
        (xx >= int(0.12 * width))
        & (xx < int(0.88 * width))
        & (yy >= int(0.12 * height))
        & (yy < int(0.92 * height))
    )
    return roi & (maximum <= 105) & (maximum - minimum <= 42)


def pose_distance(
    a: Sequence[float], b: Sequence[float]
) -> tuple[float, float]:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    translation = float(np.linalg.norm(first[:3] - second[:3]))
    qa = first[3:] / np.linalg.norm(first[3:])
    qb = second[3:] / np.linalg.norm(second[3:])
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return translation, math.degrees(2.0 * math.acos(dot))


def retarget_landmarks(
    reference: Mapping,
    pad_center_xyz: Sequence[float],
    target_center_xyz: Sequence[float],
    pad_yaw: float,
    target_yaw: float,
) -> dict[str, tuple[float, ...]]:
    """Rigidly retarget the demonstrated contact order to camera anchors."""
    anchors = reference["reference_anchors"]
    pad_delta = np.asarray(pad_center_xyz, dtype=float) - np.asarray(
        anchors["pad_center_xyz"], dtype=float
    )
    target_delta = np.asarray(target_center_xyz, dtype=float) - np.asarray(
        anchors["target_center_xyz"], dtype=float
    )
    limits = reference["limits"]
    maximum_translation = float(limits["maximum_retarget_translation_m"])
    if np.linalg.norm(pad_delta[:2]) > maximum_translation:
        raise ValueError("pad translation exceeds retarget bound")
    if np.linalg.norm(target_delta[:2]) > maximum_translation:
        raise ValueError("target translation exceeds retarget bound")
    max_yaw = math.radians(float(limits["maximum_retarget_yaw_deg"]))
    # Fixed training scene is +Y oriented.  Preserve all demonstrated wrist
    # orientations and apply only the bounded in-plane correction.
    reference_yaw = math.pi / 2
    pad_yaw_delta = wrap_angle(float(pad_yaw) - reference_yaw)
    target_yaw_delta = wrap_angle(float(target_yaw) - reference_yaw)
    if abs(pad_yaw_delta) > max_yaw or abs(target_yaw_delta) > max_yaw:
        raise ValueError("yaw correction exceeds retarget bound")

    result: dict[str, tuple[float, ...]] = {}
    for name, raw_pose in reference["right_ee_landmarks_xyzw"].items():
        pose = np.asarray(raw_pose, dtype=float).copy()
        if name in {"approach", "insert", "acquire", "peel_lift"}:
            pose[:3] += pad_delta
            pose[3:] = rotate_quaternion_about_world_z(pose[3:], pad_yaw_delta)
        elif name == "transfer":
            pose[:3] += 0.5 * (pad_delta + target_delta)
            pose[3:] = rotate_quaternion_about_world_z(
                pose[3:], 0.5 * (pad_yaw_delta + target_yaw_delta)
            )
        else:
            pose[:3] += target_delta
            pose[3:] = rotate_quaternion_about_world_z(
                pose[3:], target_yaw_delta
            )
        minimum = np.asarray(limits["world_workspace_min_xyz"], dtype=float)
        maximum = np.asarray(limits["world_workspace_max_xyz"], dtype=float)
        if np.any(pose[:3] < minimum) or np.any(pose[:3] > maximum):
            raise ValueError(f"{name} target is outside bounded workspace")
        result[name] = tuple(float(value) for value in pose)
    return result


@dataclass
class ForwardOnlyFSM:
    phase: str = "observe"
    close_count: int = 0
    open_count: int = 0
    manipulation_latched: bool = False

    def advance(self, next_phase: str) -> None:
        current_index = PHASES.index(self.phase)
        next_index = PHASES.index(next_phase)
        if next_index != current_index + 1:
            raise ValueError(
                f"non-forward transition {self.phase}->{next_phase}"
            )
        self.phase = next_phase
        if next_phase == "acquire":
            self.close_count += 1
            self.manipulation_latched = True
        if next_phase == "release":
            if not self.manipulation_latched:
                raise ValueError("release before acquire")
            self.open_count += 1

    @property
    def right_gripper_open_fraction(self) -> float:
        if self.phase in {"acquire", "peel_lift", "transfer_place"}:
            return 0.0
        return 1.0

    @property
    def base_and_spine_locked(self) -> bool:
        return self.manipulation_latched

    def validate_terminal(self) -> None:
        if (
            self.phase != "done"
            or self.close_count != 1
            or self.open_count != 1
        ):
            raise ValueError("FSM did not complete exactly one close/open")
