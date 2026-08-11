#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Create a non-destructive Task 2 dataset view with relative action stats."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contract import (
    ACTION_SIZE,
    RELATIVE_ACTION_STATE_INDICES,
    STATE_SIZE,
    to_relative_action,
    validate_dataset_root,
)


def _finite_rows(
    rows: Sequence[Sequence[float]], expected_size: int, name: str
) -> list[tuple[float, ...]]:
    result: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        if len(row) != expected_size:
            raise ValueError(
                f"{name} row {row_index} must have {expected_size} values, "
                f"got {len(row)}"
            )
        values = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"{name} row {row_index} contains non-finite values"
            )
        result.append(values)
    return result


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile from no values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def compute_vector_stats(
    rows: Sequence[Sequence[float]],
) -> dict[str, list[float] | int]:
    """Compute exact per-dimension stats for a bounded pilot dataset."""

    if not rows:
        raise ValueError("no relative action rows were produced")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("relative action rows have inconsistent widths")

    columns = [
        sorted(float(row[index]) for row in rows) for index in range(width)
    ]
    means = [sum(column) / len(column) for column in columns]
    return {
        "min": [column[0] for column in columns],
        "max": [column[-1] for column in columns],
        "mean": means,
        "std": [
            math.sqrt(
                sum((value - mean) ** 2 for value in column) / len(column)
            )
            for column, mean in zip(columns, means, strict=True)
        ],
        "q01": [_quantile(column, 0.01) for column in columns],
        "q10": [_quantile(column, 0.10) for column in columns],
        "q50": [_quantile(column, 0.50) for column in columns],
        "q90": [_quantile(column, 0.90) for column in columns],
        "q99": [_quantile(column, 0.99) for column in columns],
        "count": [len(rows)],
    }


def compute_relative_action_stats(
    actions: Sequence[Sequence[float]],
    states: Sequence[Sequence[float]],
    episode_indices: Sequence[int],
    *,
    chunk_size: int,
    included_episodes: Iterable[int] | None = None,
) -> tuple[dict[str, list[float] | int], int]:
    """Compute stats over every valid same-episode action chunk."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (len(actions) == len(states) == len(episode_indices)):
        raise ValueError(
            "actions, states, and episode_indices must have equal length"
        )
    action_rows = _finite_rows(actions, ACTION_SIZE, "action")
    state_rows = _finite_rows(states, STATE_SIZE, "state")
    episodes = [int(value) for value in episode_indices]
    selected = (
        set(episodes)
        if included_episodes is None
        else {int(value) for value in included_episodes}
    )
    if not selected:
        raise ValueError("included_episodes must not be empty")

    relative_rows: list[tuple[float, ...]] = []
    valid_chunks = 0
    for start in range(0, len(episodes) - chunk_size + 1):
        episode = episodes[start]
        if episode not in selected:
            continue
        chunk_episodes = episodes[start : start + chunk_size]
        if any(value != episode for value in chunk_episodes):
            continue
        reference_state = state_rows[start]
        valid_chunks += 1
        for offset in range(chunk_size):
            relative_rows.append(
                to_relative_action(
                    action_rows[start + offset], reference_state
                )
            )

    if not valid_chunks:
        raise ValueError(
            "no valid same-episode action chunks; collect longer episodes "
            "or lower chunk_size"
        )
    return compute_vector_stats(relative_rows), valid_chunks


def load_numeric_frames(
    dataset_root: Path,
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    """Load the three numeric columns needed for relative statistics."""

    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise ValueError(
            f"no parquet episodes found under {dataset_root / 'data'}"
        )
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to prepare the relative dataset view"
        ) from error

    actions: list[list[float]] = []
    states: list[list[float]] = []
    episodes: list[int] = []
    for path in parquet_files:
        table = parquet.read_table(
            path, columns=["action", "observation.state", "episode_index"]
        )
        actions.extend(table.column("action").to_pylist())
        states.extend(table.column("observation.state").to_pylist())
        episodes.extend(
            int(value) for value in table.column("episode_index").to_pylist()
        )
    return actions, states, episodes


def _canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _copy_dataset_tree(source: Path, destination: Path) -> tuple[int, int]:
    linked = 0
    copied = 0
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if relative.as_posix() == "meta/stats.json":
            continue
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, destination_path)
            linked += 1
        except OSError:
            shutil.copy2(source_path, destination_path, follow_symlinks=True)
            copied += 1
    return linked, copied


def materialize_relative_dataset_view(
    source_root: Path,
    destination_root: Path,
    *,
    included_episodes: Sequence[int],
    chunk_size: int = 50,
) -> dict[str, Any]:
    """Hard-link/copy a dataset and replace only the view's action stats."""

    source = source_root.resolve()
    destination = destination_root.resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError(
            "source and destination dataset roots must not contain one another"
        )
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    _, errors = validate_dataset_root(source)
    if errors:
        raise ValueError(
            "source dataset contract failed: " + "; ".join(errors)
        )

    source_stats_path = source / "meta" / "stats.json"
    source_stats = json.loads(source_stats_path.read_text(encoding="utf-8"))
    actions, states, episode_indices = load_numeric_frames(source)
    relative_action_stats, valid_chunks = compute_relative_action_stats(
        actions,
        states,
        episode_indices,
        chunk_size=chunk_size,
        included_episodes=included_episodes,
    )
    derived_stats = dict(source_stats)
    derived_stats["action"] = relative_action_stats
    stats_bytes = _canonical_json_bytes(derived_stats)

    staging = destination.with_name(
        f"{destination.name}.partial-{os.getpid()}"
    )
    if staging.exists():
        raise ValueError(f"staging destination already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        linked, copied = _copy_dataset_tree(source, staging)
        (staging / "meta").mkdir(parents=True, exist_ok=True)
        (staging / "meta" / "stats.json").write_bytes(stats_bytes)
        (staging / "task2_relative_stats.json").write_bytes(
            _canonical_json_bytes(relative_action_stats)
        )
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset_root": str(source),
            "relative_action_state_indices": list(
                RELATIVE_ACTION_STATE_INDICES
            ),
            "included_episodes": sorted(
                {int(value) for value in included_episodes}
            ),
            "chunk_size": chunk_size,
            "valid_chunks": valid_chunks,
            "relative_rows": int(relative_action_stats["count"][0]),
            "hardlinked_files": linked,
            "copied_files": copied,
            "raw_dataset_modified": False,
        }
        (staging / "meta" / "task2_relative_manifest.json").write_bytes(
            _canonical_json_bytes(manifest)
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def parse_episode_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise ValueError(
            "episodes must be a unique comma-separated list of "
            "non-negative integers"
        )
    return sorted(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        required=True,
        help="Training episodes only, e.g. 0,1,2,3,4,5,6,7",
    )
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    try:
        manifest = materialize_relative_dataset_view(
            args.dataset_root,
            args.output_root,
            included_episodes=parse_episode_list(args.episodes),
            chunk_size=args.chunk_size,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}")
        return 2
    print("PASS: relative Task 2 dataset view prepared")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
