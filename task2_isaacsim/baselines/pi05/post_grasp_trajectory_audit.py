#!/usr/bin/env python3
"""Audit early post-grasp motion from development GT trajectories.

This is offline design evidence only.  Runtime controllers must not subscribe
to simulator task-object state or evaluator output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FPS = 30
HELD_OUT_EPISODES = set(range(7, 200, 10))
JOINT_SPLINE_FRACTIONS = tuple(index / 10.0 for index in range(11))


def _quantiles(values: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "q10": float(np.quantile(array, 0.10)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _vector_quantiles(rows: list[list[float]]) -> dict[str, list[float]]:
    import numpy as np

    array = np.asarray(rows, dtype=np.float64)
    return {
        "q10": np.quantile(array, 0.10, axis=0).tolist(),
        "q50": np.quantile(array, 0.50, axis=0).tolist(),
        "q90": np.quantile(array, 0.90, axis=0).tolist(),
    }


def _markley_pose_mean(rows: list[list[float]]) -> list[float]:
    """Average xyz directly and xyzw quaternions on the unit sphere."""
    import numpy as np

    array = np.asarray(rows, dtype=np.float64)
    quaternions = array[:, 3:7]
    accumulator = quaternions.T @ quaternions
    _, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, -1]
    reference = quaternions[0] / np.linalg.norm(quaternions[0])
    if float(quaternion @ reference) < 0.0:
        quaternion = -quaternion
    return [*np.mean(array[:, :3], axis=0).tolist(), *quaternion.tolist()]


def _tool_z_elevation_deg(
    pose: list[float] | Any,
    robot_forward_xy: Any,
) -> float:
    """Elevation of tool-local +Z relative to robot-forward horizontal.

    Positive is upward and negative is downward.  Naming the actual tool axis
    avoids ambiguous visual descriptions such as wrist "forward" or "back".
    """
    import numpy as np

    quaternion = np.asarray(pose[3:7], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    tool_z_world = np.asarray(
        (
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dtype=np.float64,
    )
    forward_component = float(tool_z_world[:2] @ robot_forward_xy)
    return math.degrees(math.atan2(float(tool_z_world[2]), forward_component))


def _quaternion_error_deg(first: Any, second: Any) -> float:
    import numpy as np

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    first_array /= np.linalg.norm(first_array)
    second_array /= np.linalg.norm(second_array)
    cosine = min(1.0, abs(float(first_array @ second_array)))
    return math.degrees(2.0 * math.acos(cosine))


def build_audit(dataset_root: Path, landmark_audit: Path) -> dict[str, Any]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as parquet

    tables = [
        parquet.read_table(
            path,
            columns=["episode_index", "frame_index", "observation.state"],
        )
        for path in sorted((dataset_root / "data").glob("**/*.parquet"))
    ]
    if not tables:
        raise ValueError("dataset contains no parquet rows")
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    episode_column = np.asarray(table["episode_index"], dtype=np.int64)
    frame_column = np.asarray(table["frame_index"], dtype=np.int64)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    landmarks = json.loads(landmark_audit.read_text(encoding="utf-8"))

    time_offsets_s = (0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
    z_thresholds_m = (0.005, 0.010, 0.015, 0.020, 0.030)
    forward_thresholds_m = (0.005, 0.010, 0.020, 0.030, 0.040)
    left_thresholds_m = (0.005, 0.010, 0.020)
    by_time = {str(value): [] for value in time_offsets_s}
    by_z = {str(value): [] for value in z_thresholds_m}
    by_forward = {str(value): [] for value in forward_thresholds_m}
    by_left = {str(value): [] for value in left_thresholds_m}
    episode_rows: list[dict[str, Any]] = []
    latch_poses: list[list[float]] = []
    close_to_latch_frames: list[float] = []
    first_left_frames: list[float] = []
    close_to_first_left_frames: list[float] = []
    first_left_to_transfer_frames: list[float] = []
    transfer_to_place_frames: list[float] = []
    close_to_place_frames: list[float] = []
    relative_joint_spline = {
        f"{fraction:.1f}": [] for fraction in JOINT_SPLINE_FRACTIONS
    }
    lateral_average_speeds: list[float] = []
    lateral_peak_speeds: list[float] = []
    orientation_landmark_poses: dict[str, list[list[float]]] = {
        "stable_latch": [],
        "first_z_15mm": [],
        "first_forward_40mm": [],
        "retained_lift": [],
        "first_left_5mm": [],
        "transfer": [],
    }
    orientation_landmark_tilts: dict[str, list[float]] = {
        key: [] for key in orientation_landmark_poses
    }
    orientation_tilt_changes: dict[str, list[float]] = {
        key: [] for key in orientation_landmark_poses if key != "stable_latch"
    }
    release_time_offsets_s = (0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
    release_by_time = {str(value): [] for value in release_time_offsets_s}
    release_to_open_frames: list[float] = []
    approach_fractions = (0.0, 0.25, 0.50, 0.75, 1.0)
    approach_by_fraction = {str(value): [] for value in approach_fractions}
    approach_seconds_before_place = (3.0, 2.0, 1.0, 0.5, 0.25, 0.0)
    approach_before_place = {
        str(value): [] for value in approach_seconds_before_place
    }

    for record in landmarks["episode_records"]:
        episode = int(record["episode_index"])
        if episode in HELD_OUT_EPISODES or not record.get("phase_complete"):
            continue
        positions = np.flatnonzero(episode_column == episode)
        order = np.argsort(frame_column[positions])
        episode_states = states[positions[order]]
        frames = frame_column[positions[order]]
        phase = record["phase_frames"]
        grasp_frame = int(phase["grasp"])
        retained_frame = int(phase["retained_lift"])
        grasp_index = int(np.searchsorted(frames, grasp_frame))
        retained_index = int(np.searchsorted(frames, retained_frame))
        transfer_index = int(np.searchsorted(frames, int(phase["transfer"])))
        place_index = int(np.searchsorted(frames, int(phase["place"])))
        release_index = int(np.searchsorted(frames, int(phase["release"])))
        if retained_index <= grasp_index + 2:
            continue

        # Align to three consecutive measured closed-gripper frames rather
        # than the earlier close-command edge.
        latch_index = None
        for index in range(grasp_index, retained_index - 1):
            if np.all(episode_states[index : index + 3, 30] <= 0.10):
                latch_index = index
                break
        if latch_index is None:
            continue

        origin = episode_states[latch_index, 7:10]
        latch_poses.append(episode_states[latch_index, 7:14].tolist())
        base_yaw = float(episode_states[latch_index, 33])
        forward = np.asarray(
            [math.cos(base_yaw), math.sin(base_yaw)], dtype=np.float64
        )
        left = np.asarray([-math.sin(base_yaw), math.cos(base_yaw)])

        place_pose = episode_states[place_index, 7:14]

        def approach_relative_to_place(index: int) -> list[float]:
            delta = episode_states[index, 7:10] - place_pose[:3]
            return [
                float(delta[:2] @ forward),
                float(delta[2]),
                float(delta[:2] @ left),
                float(np.linalg.norm(delta[:2])),
                _quaternion_error_deg(
                    episode_states[index, 10:14], place_pose[3:7]
                ),
            ]

        approach_span = max(1, place_index - transfer_index)
        for fraction in approach_fractions:
            index = transfer_index + int(round(fraction * approach_span))
            approach_by_fraction[str(fraction)].append(
                approach_relative_to_place(index)
            )
        for seconds in approach_seconds_before_place:
            index = max(
                transfer_index,
                place_index - int(round(seconds * FPS)),
            )
            approach_before_place[str(seconds)].append(
                approach_relative_to_place(index)
            )

        release_origin = episode_states[release_index, 7:10]

        def release_displacement(index: int) -> list[float]:
            delta = episode_states[index, 7:10] - release_origin
            return [
                float(delta[:2] @ forward),
                float(delta[2]),
                float(delta[:2] @ left),
                float(np.linalg.norm(delta[:2])),
                float(episode_states[index, 30]),
            ]

        open_matches = np.flatnonzero(
            episode_states[release_index:, 30] >= 0.95
        )
        if not len(open_matches):
            raise ValueError(f"episode {episode} never reaches open gripper")
        open_index = release_index + int(open_matches[0])
        release_to_open_frames.append(float(open_index - release_index))
        for seconds in release_time_offsets_s:
            index = min(
                len(episode_states) - 1,
                release_index + int(round(seconds * FPS)),
            )
            release_by_time[str(seconds)].append(release_displacement(index))

        def displacement(index: int) -> list[float]:
            delta = episode_states[index, 7:10] - origin
            return [
                float(delta[:2] @ forward),
                float(delta[2]),
                float(delta[:2] @ left),
                float(np.linalg.norm(delta[:2])),
            ]

        for seconds in time_offsets_s:
            index = min(
                retained_index,
                latch_index + int(round(seconds * FPS)),
            )
            by_time[str(seconds)].append(displacement(index))

        trajectory = np.asarray(
            [displacement(index) for index in range(latch_index, transfer_index + 1)],
            dtype=np.float64,
        )
        first_left_matches = np.flatnonzero(trajectory[:, 2] >= 0.005)
        if not len(first_left_matches):
            continue
        first_left_index = latch_index + int(first_left_matches[0])
        z_15_matches = np.flatnonzero(trajectory[:, 1] >= 0.015)
        forward_40_matches = np.flatnonzero(trajectory[:, 0] >= 0.040)
        if not len(z_15_matches) or not len(forward_40_matches):
            continue

        # Preserve the demonstrated joint-space coupling through contact.
        # Align at the close-command edge (not the later measured latch),
        # normalize each successful segment to the first 5 mm robot-left
        # displacement, and express every knot relative to its own initial
        # right-arm state. Runtime can therefore transplant the motion onto a
        # camera-retargeted insert without replaying absolute simulator state.
        joint_segment = episode_states[
            grasp_index : first_left_index + 1, 21:28
        ]
        joint_origin = joint_segment[0]
        for fraction in JOINT_SPLINE_FRACTIONS:
            sample = fraction * (len(joint_segment) - 1)
            lower = int(math.floor(sample))
            upper = min(len(joint_segment) - 1, lower + 1)
            weight = sample - lower
            interpolated = (
                (1.0 - weight) * joint_segment[lower]
                + weight * joint_segment[upper]
            )
            relative_joint_spline[f"{fraction:.1f}"].append(
                (interpolated - joint_origin).tolist()
            )
        orientation_indices = {
            "stable_latch": latch_index,
            "first_z_15mm": latch_index + int(z_15_matches[0]),
            "first_forward_40mm": latch_index + int(forward_40_matches[0]),
            "retained_lift": retained_index,
            "first_left_5mm": first_left_index,
            "transfer": transfer_index,
        }
        latch_tilt = None
        for name, index in orientation_indices.items():
            pose = episode_states[index, 7:14].tolist()
            tilt = _tool_z_elevation_deg(pose, forward)
            orientation_landmark_poses[name].append(pose)
            orientation_landmark_tilts[name].append(tilt)
            if name == "stable_latch":
                latch_tilt = tilt
            else:
                assert latch_tilt is not None
                orientation_tilt_changes[name].append(tilt - latch_tilt)
        lateral_displacements = np.asarray(
            [
                displacement(index)[2]
                for index in range(first_left_index, transfer_index + 1)
            ],
            dtype=np.float64,
        )
        lateral_duration_s = (transfer_index - first_left_index) / FPS
        close_to_latch_frames.append(float(latch_index - grasp_index))
        first_left_frames.append(float(first_left_index - latch_index))
        close_to_first_left_frames.append(
            float(first_left_index - grasp_index)
        )
        first_left_to_transfer_frames.append(
            float(transfer_index - first_left_index)
        )
        transfer_to_place_frames.append(float(place_index - transfer_index))
        close_to_place_frames.append(float(place_index - grasp_index))
        lateral_average_speeds.append(
            float(
                (lateral_displacements[-1] - lateral_displacements[0])
                / max(lateral_duration_s, 1.0 / FPS)
            )
        )
        lateral_peak_speeds.append(
            float(np.max(np.diff(lateral_displacements)) * FPS)
        )
        for threshold in z_thresholds_m:
            matches = np.flatnonzero(trajectory[:, 1] >= threshold)
            if len(matches):
                by_z[str(threshold)].append(trajectory[int(matches[0])].tolist())
        for threshold in forward_thresholds_m:
            matches = np.flatnonzero(trajectory[:, 0] >= threshold)
            if len(matches):
                by_forward[str(threshold)].append(
                    trajectory[int(matches[0])].tolist()
                )
        for threshold in left_thresholds_m:
            matches = np.flatnonzero(trajectory[:, 2] >= threshold)
            if len(matches):
                by_left[str(threshold)].append(
                    trajectory[int(matches[0])].tolist()
                )
        episode_rows.append(
            {
                "episode": episode,
                "latch_frame": int(frames[latch_index]),
                "retained_lift_frame": retained_frame,
                "latch_to_retained_frames": retained_index - latch_index,
                "close_command_to_latch_frames": latch_index - grasp_index,
                "latch_to_first_left_5mm_frames": (
                    first_left_index - latch_index
                ),
                "close_command_to_first_left_5mm_frames": (
                    first_left_index - grasp_index
                ),
                "first_left_5mm_to_transfer_frames": (
                    transfer_index - first_left_index
                ),
                "lateral_average_speed_m_s": lateral_average_speeds[-1],
                "lateral_peak_one_frame_speed_m_s": lateral_peak_speeds[-1],
                "retained_displacement_forward_z_planar_m": displacement(
                    retained_index
                ),
            }
        )

    if len(episode_rows) != 180:
        raise ValueError(f"expected 180 development episodes, got {len(episode_rows)}")
    return {
        "schema_version": 1,
        "source_dataset": str(dataset_root.resolve()),
        "source_landmark_audit": str(landmark_audit.resolve()),
        "split": "development_only",
        "held_out_episode_rule": "range(7, 200, 10)",
        "episode_count": len(episode_rows),
        "alignment": "first_three_measured_right_gripper_frames_le_0.10",
        "displacement_order": [
            "robot_forward_m",
            "world_z_m",
            "robot_left_m",
            "planar_m",
        ],
        "at_seconds_after_latch": {
            key: {"count": len(rows), **_vector_quantiles(rows)}
            for key, rows in by_time.items()
        },
        "at_first_world_z_threshold_m": {
            key: {"count": len(rows), **_vector_quantiles(rows)}
            for key, rows in by_z.items()
        },
        "at_first_forward_threshold_m": {
            key: {"count": len(rows), **_vector_quantiles(rows)}
            for key, rows in by_forward.items()
        },
        "at_first_left_threshold_m": {
            key: {"count": len(rows), **_vector_quantiles(rows)}
            for key, rows in by_left.items()
        },
        "latch_to_retained_frames": _quantiles(
            [float(row["latch_to_retained_frames"]) for row in episode_rows]
        ),
        "close_command_to_stable_latch_frames": _quantiles(
            close_to_latch_frames
        ),
        "stable_latch_to_first_left_5mm_frames": _quantiles(
            first_left_frames
        ),
        "close_command_to_clear_joint_spline": {
            "alignment": "phase_grasp_close_command_edge",
            "endpoint": "first_robot_left_displacement_ge_0.005m_after_latch",
            "joint_names": [
                f"right_fr3v2_joint{index}" for index in range(1, 8)
            ],
            "fractions": list(JOINT_SPLINE_FRACTIONS),
            "duration_frames": _quantiles(close_to_first_left_frames),
            "duration_s": {
                key: value / FPS
                for key, value in _quantiles(
                    close_to_first_left_frames
                ).items()
            },
            "relative_joint_positions_rad": {
                key: {"count": len(rows), **_vector_quantiles(rows)}
                for key, rows in relative_joint_spline.items()
            },
        },
        "first_left_to_transfer_lateral_speed_m_s": {
            "duration_frames": _quantiles(first_left_to_transfer_frames),
            "duration_s": {
                key: value / FPS
                for key, value in _quantiles(
                    first_left_to_transfer_frames
                ).items()
            },
            "average": _quantiles(lateral_average_speeds),
            "peak_one_frame": _quantiles(lateral_peak_speeds),
        },
        "stable_latch_ee_world_xyzw": _vector_quantiles(latch_poses),
        "release_motion": {
            "alignment": "phase_release_command_edge",
            "displacement_order": [
                "robot_forward_m",
                "world_z_m",
                "robot_left_m",
                "planar_m",
                "measured_right_gripper_open_fraction",
            ],
            "release_to_measured_open_0.95_frames": _quantiles(
                release_to_open_frames
            ),
            "at_seconds_after_release": {
                key: {"count": len(rows), **_vector_quantiles(rows)}
                for key, rows in release_by_time.items()
            },
        },
        "transfer_to_place_motion": {
            "duration_frames": _quantiles(transfer_to_place_frames),
            "duration_s": {
                key: value / FPS
                for key, value in _quantiles(transfer_to_place_frames).items()
            },
            "close_command_to_place_duration_s": {
                key: value / FPS
                for key, value in _quantiles(close_to_place_frames).items()
            },
            "displacement_order": [
                "robot_forward_from_place_m",
                "world_z_from_place_m",
                "robot_left_from_place_m",
                "planar_from_place_m",
                "orientation_error_to_place_deg",
            ],
            "at_normalized_fraction": {
                key: {"count": len(rows), **_vector_quantiles(rows)}
                for key, rows in approach_by_fraction.items()
            },
            "at_seconds_before_place": {
                key: {"count": len(rows), **_vector_quantiles(rows)}
                for key, rows in approach_before_place.items()
            },
        },
        "right_tool_local_z_elevation_relative_robot_forward_deg": {
            "definition": (
                "atan2(tool_local_positive_z_world_z, "
                "tool_local_positive_z_dot_robot_forward_xy); positive=up, "
                "negative=down"
            ),
            "landmarks": {
                key: {
                    "count": len(orientation_landmark_poses[key]),
                    "angle_quantiles_deg": _quantiles(
                        orientation_landmark_tilts[key]
                    ),
                    "markley_mean_pose_world_xyzw": _markley_pose_mean(
                        orientation_landmark_poses[key]
                    ),
                }
                for key in orientation_landmark_poses
            },
            "change_from_stable_latch_quantiles_deg": {
                key: _quantiles(values)
                for key, values in orientation_tilt_changes.items()
            },
        },
        "retained_lift_displacement": _vector_quantiles(
            [row["retained_displacement_forward_z_planar_m"] for row in episode_rows]
        ),
        "episodes": episode_rows,
        "runtime_ground_truth_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--landmark-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_audit(args.dataset_root, args.landmark_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
