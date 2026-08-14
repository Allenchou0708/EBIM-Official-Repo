#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audit Task 2 wrist-height compensation and right pre-grasp orientation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def normalize_quaternion(
    values: list[float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("quaternion must have a finite non-zero norm")
    normalized = tuple(float(value) / norm for value in values)
    return normalized  # type: ignore[return-value]


def quaternion_angle_deg(left: list[float], right: list[float]) -> float:
    """Return the sign-invariant geodesic angle between xyzw quaternions."""

    lhs = normalize_quaternion(left)
    rhs = normalize_quaternion(right)
    dot = min(1.0, max(-1.0, sum(a * b for a, b in zip(lhs, rhs))))
    return math.degrees(2.0 * math.acos(abs(dot)))


def local_axis_world_z_abs(
    quaternion: list[float],
) -> tuple[float, float, float]:
    """Return |world-z dot local-axis| for local x, y, and z."""

    x, y, z, w = normalize_quaternion(quaternion)
    return (
        abs(2.0 * (x * z - w * y)),
        abs(2.0 * (y * z + w * x)),
        abs(1.0 - 2.0 * (x * x + y * y)),
    )


def _mean_quaternion(rows: Any) -> list[float]:
    import numpy as np

    values = np.asarray(rows, dtype=np.float64)
    reference = values[-1]
    signs = np.where(values @ reference < 0.0, -1.0, 1.0)
    mean = np.mean(values * signs[:, None], axis=0)
    return list(normalize_quaternion(mean.tolist()))


def _first_sustained_below(
    values: list[float], *, threshold: float, count: int, start: int, end: int
) -> int | None:
    run = 0
    for index in range(start, end):
        run = run + 1 if values[index] <= threshold else 0
        if run >= count:
            return index - count + 1
    return None


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    import numpy as np

    if not values:
        return {"count": 0, "q10": None, "q50": None, "q90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "q10": float(np.quantile(array, 0.10)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _vector_quantiles(rows: list[list[float]]) -> dict[str, list[float] | int]:
    import numpy as np

    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("vector quantiles require a non-empty matrix")
    return {
        "count": len(values),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
    }


def build_pregrasp_pose_audit(
    *,
    dataset_root: Path,
    phase_manifest: Path,
    orientation_threshold_deg: float = 12.0,
    sustained_frames: int = 8,
) -> dict[str, Any]:
    """Build a non-mutating state, action, event, and video audit."""

    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("numpy and pyarrow are required") from error

    root = dataset_root.resolve()
    manifest = json.loads(phase_manifest.read_text(encoding="utf-8"))
    tables = [
        parquet.read_table(
            path,
            columns=[
                "episode_index",
                "frame_index",
                "timestamp",
                "action",
                "observation.state",
            ],
        )
        for path in sorted((root / "data").glob("**/*.parquet"))
    ]
    if not tables:
        raise ValueError(f"no parquet data under {root / 'data'}")
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    episode_column = np.asarray(table["episode_index"], dtype=np.int64)
    frame_column = np.asarray(table["frame_index"], dtype=np.int64)
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    states = np.asarray(
        table["observation.state"].to_pylist(), dtype=np.float64
    )

    episode_tables = [
        parquet.read_table(path)
        for path in sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    ]
    if not episode_tables:
        raise ValueError("episode video metadata is missing")
    episode_table = (
        episode_tables[0]
        if len(episode_tables) == 1
        else pa.concat_tables(episode_tables)
    )
    video_metadata = {
        int(row["episode_index"]): row for row in episode_table.to_pylist()
    }

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for phase_record in manifest["episodes"]:
        episode = int(phase_record["episode"])
        positions = np.flatnonzero(episode_column == episode)
        if not len(positions):
            skipped.append({"episode": episode, "reason": "missing_frames"})
            continue
        state = states[positions]
        action = actions[positions]
        frame = frame_column[positions]
        timestamp = timestamps[positions]
        events = phase_record["events"]
        spine_high = int(events["spine_high"])
        right_close = int(events["right_close"])
        if not 0 <= spine_high < right_close <= len(positions):
            skipped.append({"episode": episode, "reason": "invalid_events"})
            continue
        reference_start = max(spine_high, right_close - 15)
        preclose_reference = _mean_quaternion(
            state[reference_start:right_close, 10:14]
        )
        orientation_error = [
            quaternion_angle_deg(row[10:14].tolist(), preclose_reference)
            for row in state
        ]
        orientation_entry = _first_sustained_below(
            orientation_error,
            threshold=orientation_threshold_deg,
            count=sustained_frames,
            start=spine_high,
            end=right_close,
        )
        if orientation_entry is None:
            orientation_entry = reference_start

        preclose = max(spine_high, right_close - 1)
        start_right_joints = state[spine_high, 21:28]
        preclose_right_joints = state[preclose, 21:28]
        joint_delta = preclose_right_joints - start_right_joints
        video = video_metadata[episode]
        video_file = int(
            video["videos/observation.images.wrist_right/file_index"]
        )
        video_chunk = int(
            video["videos/observation.images.wrist_right/chunk_index"]
        )
        video_offset = float(
            video["videos/observation.images.wrist_right/from_timestamp"]
        )

        def landmark(local_index: int) -> dict[str, Any]:
            return {
                "frame": int(frame[local_index]),
                "episode_time_s": float(timestamp[local_index]),
                "wrist_right_video_time_s": float(
                    video_offset + timestamp[local_index]
                ),
                "right_ee": state[local_index, 7:14].tolist(),
                "right_joints": state[local_index, 21:28].tolist(),
                "left_ee_z_m": float(state[local_index, 2]),
                "right_ee_z_m": float(state[local_index, 9]),
                "spine_m": float(state[local_index, 28]),
                "right_gripper_action": float(action[local_index, 18]),
                "orientation_error_to_preclose_deg": float(
                    orientation_error[local_index]
                ),
                "local_axis_world_z_abs": list(
                    local_axis_world_z_abs(state[local_index, 10:14].tolist())
                ),
            }

        records.append(
            {
                "episode": episode,
                "split": (
                    "held_out"
                    if episode in set(manifest["held_out_episodes"])
                    else "train"
                ),
                "right_wrist_video": (
                    "videos/observation.images.wrist_right/"
                    f"chunk-{video_chunk:03d}/file-{video_file:03d}.mp4"
                ),
                "events": events,
                "orientation_entry_frame": int(frame[orientation_entry]),
                "orientation_entry_to_close_frames": int(
                    right_close - orientation_entry
                ),
                "orientation_shift_spine_high_to_preclose_deg": float(
                    orientation_error[spine_high]
                ),
                "right_joint_delta_spine_high_to_preclose": (
                    joint_delta.tolist()
                ),
                "right_joint_delta_l2_rad": float(np.linalg.norm(joint_delta)),
                "spine_rise_m": float(
                    state[spine_high, 28] - state[0, 28]
                ),
                "left_world_z_change_during_spine_rise_m": float(
                    state[spine_high, 2] - state[0, 2]
                ),
                "right_world_z_change_during_spine_rise_m": float(
                    state[spine_high, 9] - state[0, 9]
                ),
                "landmarks": {
                    "start": landmark(0),
                    "spine_high": landmark(spine_high),
                    "orientation_entry": landmark(orientation_entry),
                    "preclose": landmark(preclose),
                    "right_close": landmark(right_close),
                },
            }
        )

    if not records:
        raise ValueError("no auditable episodes")
    preclose_axes = [
        record["landmarks"]["preclose"]["local_axis_world_z_abs"]
        for record in records
    ]
    median_axes = np.median(np.asarray(preclose_axes), axis=0)
    axis_names = ("local_x", "local_y", "local_z")
    dominant_axis = int(np.argmax(median_axes))
    global_preclose_quaternion = _mean_quaternion(
        [
            record["landmarks"]["preclose"]["right_ee"][3:7]
            for record in records
        ]
    )
    global_preclose_orientation_errors = [
        quaternion_angle_deg(
            record["landmarks"]["preclose"]["right_ee"][3:7],
            global_preclose_quaternion,
        )
        for record in records
    ]
    aggregate = {
        "episodes": len(records),
        "train_episodes": sum(
            record["split"] == "train" for record in records
        ),
        "held_out_episodes": sum(
            record["split"] == "held_out" for record in records
        ),
        "spine_rise_m": _quantiles(
            [record["spine_rise_m"] for record in records]
        ),
        "left_world_z_change_during_spine_rise_m": _quantiles(
            [
                record["left_world_z_change_during_spine_rise_m"]
                for record in records
            ]
        ),
        "right_world_z_change_during_spine_rise_m": _quantiles(
            [
                record["right_world_z_change_during_spine_rise_m"]
                for record in records
            ]
        ),
        "orientation_shift_spine_high_to_preclose_deg": _quantiles(
            [
                record["orientation_shift_spine_high_to_preclose_deg"]
                for record in records
            ]
        ),
        "orientation_entry_to_close_frames": _quantiles(
            [record["orientation_entry_to_close_frames"] for record in records]
        ),
        "right_joint_delta_l2_rad": _quantiles(
            [record["right_joint_delta_l2_rad"] for record in records]
        ),
        "right_joint_delta_rad": _vector_quantiles(
            [
                record["right_joint_delta_spine_high_to_preclose"]
                for record in records
            ]
        ),
        "orientation_entry_right_ee_position_m": _vector_quantiles(
            [
                record["landmarks"]["orientation_entry"]["right_ee"][:3]
                for record in records
            ]
        ),
        "preclose_right_ee_position_m": _vector_quantiles(
            [
                record["landmarks"]["preclose"]["right_ee"][:3]
                for record in records
            ]
        ),
        "global_preclose_right_ee_quaternion_xyzw": global_preclose_quaternion,
        "global_preclose_orientation_error_deg": _quantiles(
            global_preclose_orientation_errors
        ),
        "median_preclose_local_axis_world_z_abs": median_axes.tolist(),
        "dominant_preclose_vertical_axis": axis_names[dominant_axis],
    }
    return {
        "schema_version": 1,
        "dataset_root": str(root),
        "phase_manifest": str(phase_manifest.resolve()),
        "orientation_entry_definition": {
            "reference": (
                "mean right-EE quaternion over final 15 pre-close frames"
            ),
            "threshold_deg": orientation_threshold_deg,
            "sustained_frames": sustained_frames,
        },
        "aggregate": aggregate,
        "episodes": records,
        "skipped": skipped,
        "raw_dataset_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--phase-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--orientation-threshold-deg", type=float, default=12.0
    )
    parser.add_argument("--sustained-frames", type=int, default=8)
    args = parser.parse_args()
    report = build_pregrasp_pose_audit(
        dataset_root=args.dataset_root,
        phase_manifest=args.phase_manifest,
        orientation_threshold_deg=args.orientation_threshold_deg,
        sustained_frames=args.sustained_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
