# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Rolling numeric GT prompts for the PI0.5 language-conditioning ablation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


SOURCE_EPISODE = 19
SOURCE_START_FRAME = 370
SOURCE_END_EXCLUSIVE = 950
RIGHT_ONLY_ACTION_INDICES = (*range(10, 17), 18)
LANGUAGE_GT_MAX_WINDOW = 15
LANGUAGE_GT_TOKENIZER_MAX_LENGTH = 1024


@dataclass(frozen=True)
class TrajectoryRow:
    frame: int
    right_joint_state: tuple[float, ...]
    absolute_action: tuple[float, ...]


def load_episode_19_actions(dataset_root: Path) -> tuple[TrajectoryRow, ...]:
    """Load source states and absolute right-arm/gripper GT outputs."""

    import pyarrow as pa
    import pyarrow.parquet as parquet

    paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not paths:
        raise ValueError(f"dataset contains no parquet action data: {dataset_root}")
    tables = [
        parquet.read_table(
            path,
            columns=[
                "episode_index",
                "frame_index",
                "observation.state",
                "action",
            ],
        )
        for path in paths
    ]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    payload = table.to_pydict()
    rows: list[TrajectoryRow] = []
    for episode, frame, state, action in zip(
        payload["episode_index"],
        payload["frame_index"],
        payload["observation.state"],
        payload["action"],
        strict=True,
    ):
        frame = int(frame)
        if (
            int(episode) != SOURCE_EPISODE
            or frame < SOURCE_START_FRAME
            or frame >= SOURCE_END_EXCLUSIVE
        ):
            continue
        if len(action) != 20:
            raise ValueError("source Task 2 action must contain 20 values")
        if len(state) != 37:
            raise ValueError("source Task 2 state must contain 37 values")
        right_joint_state = tuple(float(value) for value in state[21:28])
        right = tuple(float(action[index]) for index in RIGHT_ONLY_ACTION_INDICES)
        if not all(math.isfinite(value) for value in (*right_joint_state, *right)):
            raise ValueError(f"non-finite GT state/action at source frame {frame}")
        rows.append(
            TrajectoryRow(
                frame=frame,
                right_joint_state=right_joint_state,
                absolute_action=right,
            )
        )

    expected_frames = list(range(SOURCE_START_FRAME, SOURCE_END_EXCLUSIVE))
    if [row.frame for row in rows] != expected_frames:
        raise ValueError(
            "episode-19 GT action frames are missing, duplicated, or out of order"
        )
    return tuple(rows)


def format_language_gt_window(
    trajectory: tuple[TrajectoryRow, ...],
    *,
    executed_actions: int,
    window: int,
    reference_joint_state: tuple[float, ...] | None = None,
) -> tuple[str, tuple[int, ...]]:
    """Format GT targets in the checkpoint's relative joint-action space."""

    if not trajectory:
        raise ValueError("language GT trajectory is empty")
    if executed_actions < 0:
        raise ValueError("executed_actions must be non-negative")
    if not 1 <= window <= LANGUAGE_GT_MAX_WINDOW:
        raise ValueError(
            f"language GT window must be within 1..{LANGUAGE_GT_MAX_WINDOW}"
        )
    start = min(executed_actions, len(trajectory) - 1)
    selected = trajectory[start : start + window]
    reference = (
        selected[0].right_joint_state
        if reference_joint_state is None
        else tuple(float(value) for value in reference_joint_state)
    )
    if len(reference) != 7 or not all(math.isfinite(value) for value in reference):
        raise ValueError("language GT reference must contain seven finite joints")
    rows = "".join(
        f" F{row.frame}=[{','.join(f'{value:.3f}' for value in relative)}];"
        for row in selected
        for relative in (
            tuple(
                row.absolute_action[index] - reference[index]
                for index in range(7)
            )
            + (row.absolute_action[7],),
        )
    )
    prompt = (
        "GT outputs for the next frames. Joint deltas are relative to the "
        "current right-arm state at this inference start. Row format: "
        "[delta_joint1_rad,delta_joint2_rad,"
        "delta_joint3_rad,delta_joint4_rad,delta_joint5_rad,delta_joint6_rad,"
        "delta_joint7_rad,gripper_open_fraction_absolute]."
        f"{rows} Use the current head-camera and right-wrist-camera observations "
        "together with this ground-truth demonstration to generate a more "
        "precise action for every frame."
    )
    return prompt, tuple(row.frame for row in selected)
