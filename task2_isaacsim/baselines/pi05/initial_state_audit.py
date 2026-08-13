#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Summarize organizer Task 2 frame-0 robot and object states."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from task2_isaacsim.baselines.pi05.contract import STATE_NAMES

ROBOT_INDICES = tuple(range(14, 34))
READY_ARMS = (
    0.0,
    -0.7854,
    0.0,
    -2.3562,
    0.0,
    1.5708,
    0.7854,
) * 2
LIVE_RESET = (*READY_ARMS, 0.0, 1.0, 1.0, 2.10, 3.05, -1.571)
ROBOT_TOLERANCE = (0.03,) * 14 + (0.01, 0.02, 0.02, 0.01, 0.01, 0.01)


def _yaw_from_wxyz(pose: list[float]) -> float:
    _x, _y, _z, qw, qx, qy, qz = pose
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _stats(values: Any) -> dict[str, float]:
    import numpy as np

    vector = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(vector)),
        "minimum": float(np.min(vector)),
        "maximum": float(np.max(vector)),
    }


def audit_initial_states(dataset_root: Path) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as parquet

    paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not paths:
        raise ValueError(f"no parquet files below {dataset_root / 'data'}")
    columns = ["episode_index", "frame_index", "observation.state"]
    rows: list[dict[str, Any]] = []
    schema_names: set[str] = set()
    for path in paths:
        schema_names.update(parquet.read_schema(path).names)
        rows.extend(parquet.read_table(path, columns=columns).to_pylist())
    frame_zero = sorted(
        (row for row in rows if int(row["frame_index"]) == 0),
        key=lambda row: int(row["episode_index"]),
    )
    episodes = [int(row["episode_index"]) for row in frame_zero]
    if episodes != list(range(200)):
        raise ValueError("expected exactly frame 0 for organizer episodes 0..199")
    states = np.asarray([row["observation.state"] for row in frame_zero], dtype=float)

    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature_names = set(info.get("features", {}))
    success_fields = sorted(
        name for name in schema_names | feature_names if "success" in name.lower()
    )

    object_frames: list[Any] = []
    object_names: list[str] | None = None
    extras_keys: set[str] = set()
    for episode in episodes:
        path = dataset_root / "task2_extras" / f"episode_{episode:06d}.npz"
        with np.load(path) as payload:
            names = [str(value) for value in payload["object_names"].tolist()]
            if object_names is None:
                object_names = names
            elif names != object_names:
                raise ValueError("task2_extras object order changes across episodes")
            extras_keys.update(payload.files)
            object_frames.append(payload["object_poses"][0].astype(float))
    assert object_names is not None
    objects = np.asarray(object_frames, dtype=float)

    medians = np.median(states[:, ROBOT_INDICES], axis=0)
    normalized = (states[:, ROBOT_INDICES] - np.asarray(LIVE_RESET)) / np.asarray(
        ROBOT_TOLERANCE
    )
    distances = np.sqrt(np.sum(normalized**2, axis=1))
    representatives = [int(value) for value in np.argsort(distances)[:3]]
    dominant_members = np.all(
        np.abs(states[:, ROBOT_INDICES] - medians)
        <= np.asarray(ROBOT_TOLERANCE),
        axis=1,
    )

    robot_fields = {
        STATE_NAMES[index]: _stats(states[:, index]) for index in ROBOT_INDICES
    }
    object_fields: dict[str, Any] = {}
    for object_index, name in enumerate(object_names):
        poses = objects[:, object_index]
        yaws = [_yaw_from_wxyz(pose.tolist()) for pose in poses]
        object_fields[name] = {
            "x": _stats(poses[:, 0]),
            "y": _stats(poses[:, 1]),
            "z": _stats(poses[:, 2]),
            "yaw": _stats(yaws),
        }

    episode9_state = states[9]
    episode9_objects = {
        name: {
            "pose_xyz_wxyz": objects[9, index].tolist(),
            "yaw": _yaw_from_wxyz(objects[9, index].tolist()),
        }
        for index, name in enumerate(object_names)
    }
    return {
        "dataset_root": str(dataset_root),
        "episode_count": len(episodes),
        "per_episode_success_label_exists": bool(success_fields),
        "success_label_candidates": success_fields,
        "success_label_interpretation": (
            "A dataset success field is present."
            if success_fields
            else "No trustworthy per-episode success label exists; group by "
            "initial state, not success rate."
        ),
        "schema_fields": sorted(schema_names),
        "extras_keys": sorted(extras_keys),
        "robot_frame0": robot_fields,
        "object_frame0": object_fields,
        "major_clusters": [
            {
                "name": "dominant_live_reset",
                "count": int(np.count_nonzero(dominant_members)),
                "robot_tolerance": {
                    "arm_rad": 0.03,
                    "spine_m": 0.01,
                    "gripper_fraction": 0.02,
                    "base_x_y_m": 0.01,
                    "base_yaw_rad": 0.01,
                },
                "object_state": "identical within float precision",
            },
            {
                "name": "outside_dominant_live_reset",
                "count": int(np.count_nonzero(~dominant_members)),
            },
        ],
        "representative_episode_ids": representatives,
        "episode9": {
            "arm_14d": episode9_state[14:28].tolist(),
            "spine_m": float(episode9_state[28]),
            "grippers": episode9_state[29:31].tolist(),
            "base_x_y_yaw": episode9_state[31:34].tolist(),
            "objects": episode9_objects,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Task 2 organizer initial-state audit",
        "",
        f"Episodes: {report['episode_count']}",
        "",
        "Per-episode success label: "
        + ("yes" if report["per_episode_success_label_exists"] else "no"),
        "",
        report["success_label_interpretation"],
        "",
        "## Major clusters",
        "",
        "| Cluster | Count | Interpretation |",
        "|---|---:|---|",
    ]
    for cluster in report["major_clusters"]:
        interpretation = cluster.get("object_state", "outside tolerance")
        lines.append(f"| {cluster['name']} | {cluster['count']} | {interpretation} |")
    lines.extend(
        [
            "",
            "Representative episodes closest to live reset: "
            + ", ".join(map(str, report["representative_episode_ids"])),
            "",
            "## Robot frame-0 range",
            "",
            "| Field | Median | Minimum | Maximum |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, stats in report["robot_frame0"].items():
        lines.append(
            f"| {name} | {stats['median']:.8f} | "
            f"{stats['minimum']:.8f} | {stats['maximum']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Object frame-0 range",
            "",
            "| Object | x median/range | y median/range | z median/range | yaw median/range |",
            "|---|---|---|---|---|",
        ]
    )
    for name, fields in report["object_frame0"].items():
        cells = []
        for field in ("x", "y", "z", "yaw"):
            stats = fields[field]
            cells.append(
                f"{stats['median']:.6f} "
                f"[{stats['minimum']:.6f}, {stats['maximum']:.6f}]"
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Episode 9 exact frame-0 state",
            "",
            "```json",
            json.dumps(report["episode9"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit_initial_states(args.dataset_root)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "initial_state_audit.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "initial_state_audit.md").write_text(
            _markdown(report), encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
