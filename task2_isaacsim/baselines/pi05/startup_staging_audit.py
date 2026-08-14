#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Select and audit a dataset-derived Task 2 pre-inference staging route."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from task2_isaacsim.baselines.pi05.contract import (
    ACTION_NAMES,
    FR3_JOINT_LIMITS,
    STATE_NAMES,
)
from task2_isaacsim.baselines.pi05.pregrasp_pose_audit import (
    quaternion_angle_deg,
)

ARM_VELOCITY_LIMITS_RAD_S = (
    2.62,
    2.62,
    2.62,
    2.62,
    5.26,
    4.18,
    5.26,
) * 2
ARM_STAGING_VELOCITY_FRACTION = 0.50
SPINE_STAGING_VELOCITY_M_S = 0.05
GRIPPER_STAGING_VELOCITY_FRACTION_S = 1.0


def _quantiles(values: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "count": int(array.shape[0]),
        "min": np.min(array, axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q10": np.quantile(array, 0.10, axis=0).tolist(),
        "q50": np.quantile(array, 0.50, axis=0).tolist(),
        "q90": np.quantile(array, 0.90, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
    }


def _robust_scores(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    median = np.median(array, axis=0)
    q25 = np.quantile(array, 0.25, axis=0)
    q75 = np.quantile(array, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    return np.sqrt(np.mean(((array - median) / scale) ** 2, axis=1))


def _scheduled_interval_s(previous: Any, current: Any, recorded_dt: float) -> float:
    import numpy as np

    delta = np.abs(np.asarray(current, dtype=np.float64) - previous)
    arm_limits = np.asarray(ARM_VELOCITY_LIMITS_RAD_S, dtype=np.float64)
    arm_dt = float(
        np.max(delta[3:17] / (arm_limits * ARM_STAGING_VELOCITY_FRACTION))
    )
    gripper_dt = float(
        np.max(delta[17:19]) / GRIPPER_STAGING_VELOCITY_FRACTION_S
    )
    spine_dt = float(delta[19] / SPINE_STAGING_VELOCITY_M_S)
    return max(float(recorded_dt), arm_dt, gripper_dt, spine_dt, 1e-4)


def build_startup_staging_audit(
    *, dataset_root: Path, pregrasp_audit: Path
) -> dict[str, Any]:
    """Return distribution, risk, representative target, and route evidence."""

    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("numpy and pyarrow are required") from error

    root = dataset_root.resolve()
    pose_audit = json.loads(pregrasp_audit.read_text(encoding="utf-8"))
    if Path(pose_audit["dataset_root"]).resolve() != root:
        raise ValueError("pregrasp audit dataset root does not match")
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

    pose_records = {
        int(record["episode"]): record for record in pose_audit["episodes"]
    }
    segments: dict[str, list[Any]] = {
        "startup_action": [],
        "startup_state": [],
        "pregrasp_action": [],
        "pregrasp_state": [],
    }
    candidate_rows: list[dict[str, Any]] = []
    velocity_records: list[dict[str, Any]] = []
    for episode, pose_record in pose_records.items():
        positions = np.flatnonzero(episode_column == episode)
        if not len(positions):
            continue
        episode_actions = actions[positions]
        episode_states = states[positions]
        episode_times = timestamps[positions]
        events = pose_record["events"]
        spine_high = int(events["spine_high"])
        orientation_entry = int(pose_record["orientation_entry_frame"])
        right_close = int(events["right_close"])
        preclose = right_close - 1
        segments["startup_action"].append(
            episode_actions[: spine_high + 1, 3:20]
        )
        segments["startup_state"].append(
            episode_states[: spine_high + 1, 14:31]
        )
        segments["pregrasp_action"].append(
            episode_actions[orientation_entry:right_close, 3:20]
        )
        segments["pregrasp_state"].append(
            episode_states[orientation_entry:right_close, 14:31]
        )
        final_state = episode_states[preclose]
        final_action = episode_actions[preclose]
        candidate_rows.append(
            {
                "episode": episode,
                "split": pose_record["split"],
                "frame": int(frame_column[positions[preclose]]),
                "local_index": preclose,
                "feature": [
                    *final_state[14:31].tolist(),
                    float(final_state[2]),
                    *final_state[7:10].tolist(),
                ],
                "right_quaternion": final_state[10:14].tolist(),
                "action": final_action.tolist(),
                "state": final_state.tolist(),
                "timestamp": float(episode_times[preclose]),
            }
        )
        for name, start, end in (
            ("startup", 0, spine_high + 1),
            ("pregrasp", orientation_entry, right_close),
        ):
            action = episode_actions[start:end]
            times = episode_times[start:end]
            if len(action) < 2:
                continue
            dt = np.diff(times)
            delta = np.diff(action[:, 3:20], axis=0)
            arm_rate = np.abs(delta[:, :14] / dt[:, None])
            arm_limits = np.asarray(ARM_VELOCITY_LIMITS_RAD_S)
            violations = np.argwhere(arm_rate > arm_limits[None, :])
            velocity_records.append(
                {
                    "episode": episode,
                    "segment": name,
                    "frames": int(len(action)),
                    "median_dt_s": float(np.median(dt)),
                    "max_arm_rate_rad_s": float(np.max(arm_rate)),
                    "arm_limit_violation_count": int(len(violations)),
                    "max_spine_rate_m_s": float(
                        np.max(np.abs(delta[:, 16] / dt))
                    ),
                    "max_gripper_rate_fraction_s": float(
                        np.max(np.abs(delta[:, 14:16] / dt[:, None]))
                    ),
                }
            )

    if not candidate_rows:
        raise ValueError("no pregrasp candidates")
    train_candidates = [
        row for row in candidate_rows if row["split"] == "train"
    ]
    features = np.asarray(
        [row["feature"] for row in train_candidates], dtype=np.float64
    )
    scores = _robust_scores(features)
    global_quaternion = pose_audit["aggregate"][
        "global_preclose_right_ee_quaternion_xyzw"
    ]
    quaternion_errors = np.asarray(
        [
            quaternion_angle_deg(row["right_quaternion"], global_quaternion)
            for row in train_candidates
        ],
        dtype=np.float64,
    )
    combined = scores + quaternion_errors / 12.0
    selected_index = int(np.argmin(combined))
    selected = train_candidates[selected_index]
    selected["robust_distance_score"] = float(scores[selected_index])
    selected["global_orientation_error_deg"] = float(
        quaternion_errors[selected_index]
    )

    episode = int(selected["episode"])
    positions = np.flatnonzero(episode_column == episode)
    end = int(selected["local_index"])
    route_actions = actions[positions[: end + 1]]
    route_states = states[positions[: end + 1]]
    route_frames = frame_column[positions[: end + 1]]
    route_times = timestamps[positions[: end + 1]]
    route = []
    scheduled_at = 0.0
    for index, (action, state, frame, timestamp) in enumerate(
        zip(
            route_actions,
            route_states,
            route_frames,
            route_times,
            strict=True,
        )
    ):
        interval = 0.0
        if index:
            interval = _scheduled_interval_s(
                route_actions[index - 1], action, timestamp - route_times[index - 1]
            )
            scheduled_at += interval
        route.append(
            {
                "frame": int(frame),
                "episode_time_s": float(timestamp),
                "scheduled_at_s": scheduled_at,
                "scheduled_interval_s": interval,
                "command": action[3:20].tolist(),
                "measured_reference": state[14:31].tolist(),
            }
        )

    final_action = selected["action"]
    final_state = selected["state"]
    arm_bounds_ok = all(
        lower <= value <= upper
        for values in (final_action[3:10], final_action[10:17])
        for value, (lower, upper) in zip(values, FR3_JOINT_LIMITS, strict=True)
    )
    segment_stats = {}
    for name in ("startup", "pregrasp"):
        action_values = np.concatenate(segments[f"{name}_action"], axis=0)
        state_values = np.concatenate(segments[f"{name}_state"], axis=0)
        segment_stats[name] = {
            "action_names": list(ACTION_NAMES[3:20]),
            "action": _quantiles(action_values),
            "state_names": list(STATE_NAMES[14:31]),
            "state": _quantiles(state_values),
            "robust_outlier_frames": int(
                np.sum(_robust_scores(state_values) > 6.0)
            ),
        }
    episode_scores = [
        {
            "episode": int(row["episode"]),
            "split": row["split"],
            "robust_distance_score": float(score),
        }
        for row, score in zip(train_candidates, scores, strict=True)
    ]
    episode_scores.sort(key=lambda row: row["robust_distance_score"], reverse=True)
    return {
        "schema_version": 1,
        "dataset_root": str(root),
        "pregrasp_audit": str(pregrasp_audit.resolve()),
        "selection": {
            "episode": episode,
            "split": selected["split"],
            "frame": int(selected["frame"]),
            "episode_time_s": float(selected["timestamp"]),
            "reason": (
                "train-split pre-close frame nearest the robust multivariate "
                "median of both arms, spine, grippers, left wrist height, "
                "right EE position, and audited vertical orientation"
            ),
            "robust_distance_score": selected["robust_distance_score"],
            "global_orientation_error_deg": selected[
                "global_orientation_error_deg"
            ],
        },
        "final_target": {
            "command_action_indices": list(range(3, 20)),
            "left_arm_rad": final_action[3:10],
            "right_arm_rad": final_action[10:17],
            "left_gripper_open_fraction": final_action[17],
            "right_gripper_open_fraction": final_action[18],
            "spine_command_m": final_action[19],
            "measured_reference": {
                "left_arm_rad": final_state[14:21],
                "right_arm_rad": final_state[21:28],
                "spine_m": final_state[28],
                "left_gripper_open_fraction": final_state[29],
                "right_gripper_open_fraction": final_state[30],
                "left_ee": final_state[0:7],
                "right_ee": final_state[7:14],
            },
            "arm_joint_bounds_ok": arm_bounds_ok,
            "right_pregrasp_vertical_axis": pose_audit["aggregate"][
                "dominant_preclose_vertical_axis"
            ],
        },
        "tolerances": {
            "arm_max_abs_rad": 0.04,
            "spine_abs_m": 0.02,
            "gripper_open_fraction": 0.05,
            "left_ee_z_m": 0.04,
            "right_ee_position_m": 0.04,
            "right_ee_orientation_deg": 12.0,
            "stable_dwell_sim_s": 1.0,
        },
        "segment_distribution": segment_stats,
        "velocity_risk": {
            "arm_limits_rad_s": list(ARM_VELOCITY_LIMITS_RAD_S),
            "audited_segment_count": len(velocity_records),
            "raw_arm_limit_violation_count": sum(
                row["arm_limit_violation_count"] for row in velocity_records
            ),
            "segments_with_arm_limit_violations": [
                row
                for row in velocity_records
                if row["arm_limit_violation_count"]
            ],
            "selected_episode_segments": [
                row
                for row in velocity_records
                if row["episode"] == episode
            ],
            "largest_arm_rate_segments": sorted(
                velocity_records,
                key=lambda row: row["max_arm_rate_rad_s"],
                reverse=True,
            )[:10],
            "largest_spine_rate_segments": sorted(
                velocity_records,
                key=lambda row: row["max_spine_rate_m_s"],
                reverse=True,
            )[:10],
            "staging_arm_velocity_fraction": ARM_STAGING_VELOCITY_FRACTION,
            "staging_spine_velocity_m_s": SPINE_STAGING_VELOCITY_M_S,
            "staging_gripper_velocity_fraction_s": (
                GRIPPER_STAGING_VELOCITY_FRACTION_S
            ),
            "scheduled_duration_sim_s": scheduled_at,
        },
        "largest_episode_outliers": episode_scores[:10],
        "trajectory": route,
        "raw_dataset_modified": False,
        "guessed_ik_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pregrasp-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_startup_staging_audit(
        dataset_root=args.dataset_root,
        pregrasp_audit=args.pregrasp_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selection": report["selection"],
                "final_target": report["final_target"],
                "velocity_risk": {
                    key: report["velocity_risk"][key]
                    for key in (
                        "audited_segment_count",
                        "raw_arm_limit_violation_count",
                        "scheduled_duration_sim_s",
                    )
                },
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
