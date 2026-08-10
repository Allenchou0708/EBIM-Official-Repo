#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Extract an auditable initial base target from organizer train episodes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-repo-id", default="hermanprawiro/task2_fixpos_v1"
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = sorted({int(value) for value in args.episodes.split(",")})
    dataset = LeRobotDataset(
        args.dataset_repo_id, root=args.dataset_root, episodes=episodes
    )
    selected = dataset.hf_dataset.select_columns(
        ["episode_index", "observation.state"]
    )
    first: dict[int, list[float]] = {}
    for row in selected:
        episode = int(row["episode_index"])
        if episode not in first:
            first[episode] = [
                float(value) for value in row["observation.state"]
            ]
    missing = sorted(set(episodes) - set(first))
    if missing:
        raise ValueError(f"episodes missing from dataset: {missing}")
    initial = {str(episode): first[episode][31:34] for episode in episodes}
    initial_spine = {str(episode): first[episode][28] for episode in episodes}
    axes = list(zip(*initial.values(), strict=True))
    target = [statistics.median(axis) for axis in axes]
    ranges = [max(axis) - min(axis) for axis in axes]
    reset_relative = (
        max(abs(value) for value in target) < 0.02 and max(ranges) < 0.02
    )
    report = {
        "schema_version": 1,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root.resolve()),
        "episodes": episodes,
        "state_indices": [31, 32, 33],
        "initial_base_pose_by_episode": initial,
        "median_target": target,
        "axis_ranges": ranges,
        "spine_state_index": 28,
        "initial_spine_by_episode": initial_spine,
        "median_spine_height_m": statistics.median(initial_spine.values()),
        "spine_height_range_m": max(initial_spine.values())
        - min(initial_spine.values()),
        "coordinate_interpretation": (
            "reset_relative_odom; use the reproducible room-scene reset "
            "pose as "
            "the staging marker, not these values as a world/table transform"
            if reset_relative
            else "dataset odometry frame; verify against the room-scene "
            "transform"
        ),
        "reset_relative_detected": reset_relative,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
