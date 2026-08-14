#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build and consume the event-balanced sampler used by Task 2 PI0.5 V2."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

PHASE_RATIOS = {
    "startup_rise": 20,
    "approach": 20,
    "grasp_acquisition": 20,
    "lift_transfer": 15,
    "lower_place": 15,
    "release_retreat": 10,
}
CHUNK_SIZE = 50


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _first_sustained(
    mask: Sequence[bool], *, count: int, start: int = 0
) -> int | None:
    run = 0
    for index in range(start, len(mask)):
        run = run + 1 if bool(mask[index]) else 0
        if run >= count:
            return index - count + 1
    return None


def build_phase_manifest(
    *, dataset_root: Path, audit_report: Path
) -> dict[str, Any]:
    """Return full-horizon training starts grouped by physical task event."""

    try:
        import numpy as np
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("numpy and pyarrow are required") from error

    audit = _read_json(audit_report)
    train_episodes = {int(value) for value in audit["split"]["train"]}
    held_out = {int(value) for value in audit["split"]["held_out"]}
    tables = [
        parquet.read_table(
            path,
            columns=["episode_index", "frame_index", "action", "observation.state"],
        )
        for path in sorted((dataset_root / "data").glob("**/*.parquet"))
    ]
    if not tables:
        raise ValueError(f"no parquet data under {dataset_root / 'data'}")
    table = (
        tables[0]
        if len(tables) == 1
        else __import__("pyarrow").concat_tables(tables)
    )
    episode_column = np.asarray(table["episode_index"], dtype=np.int64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    states = np.asarray(
        table["observation.state"].to_pylist(), dtype=np.float64
    )

    groups: dict[str, list[int]] = {name: [] for name in PHASE_RATIOS}
    episode_records: list[dict[str, Any]] = []
    skipped: list[int] = []
    for episode in sorted(train_episodes | held_out):
        positions = np.flatnonzero(episode_column == episode)
        if not len(positions):
            skipped.append(episode)
            continue
        action = actions[positions]
        state = states[positions]
        length = len(positions)
        extras = np.load(
            dataset_root / "task2_extras" / f"episode_{episode:06d}.npz"
        )
        sim_time = np.asarray(extras["sim_time"], dtype=np.float64)
        pad_time = np.asarray(extras["pad_sim_time"], dtype=np.float64)
        pad_centroid = np.nanmean(
            np.asarray(extras["pad_points"], dtype=np.float64), axis=1
        )
        order = np.argsort(pad_time)
        pad_time = pad_time[order]
        pad_centroid = pad_centroid[order]
        initial_pad = np.median(pad_centroid[: min(10, len(pad_centroid))], axis=0)
        pad_move_sample = _first_sustained(
            np.linalg.norm(pad_centroid[:, :2] - initial_pad[:2], axis=1)
            > 0.01,
            count=3,
        )
        target_sample = _first_sustained(
            pad_centroid[:, 0] > 2.10, count=3
        )

        def pad_frame(sample: int | None) -> int | None:
            if sample is None:
                return None
            return int(np.argmin(np.abs(sim_time - pad_time[sample])))

        spine_high = _first_sustained(state[:, 28] >= 0.44, count=3)
        close = _first_sustained(action[:, 18] < 0.5, count=5)
        pad_move = pad_frame(pad_move_sample)
        target = pad_frame(target_sample)
        release = (
            None
            if close is None
            else _first_sustained(
                action[:, 18] > 0.5, count=5, start=close + 5
            )
        )
        boundaries = [0, spine_high, close, pad_move, target, release, length]
        if any(value is None for value in boundaries):
            skipped.append(episode)
            continue
        boundaries = [int(value) for value in boundaries]
        if any(right < left for left, right in zip(boundaries, boundaries[1:])):
            skipped.append(episode)
            continue
        events = dict(
            zip(
                (
                    "start",
                    "spine_high",
                    "right_close",
                    "pad_move",
                    "target_arrival",
                    "right_release",
                    "end",
                ),
                boundaries,
                strict=True,
            )
        )
        episode_records.append(
            {"episode": episode, "length": length, "events": events}
        )
        if episode not in train_episodes:
            continue
        maximum_start = length - CHUNK_SIZE
        for name, start, end in zip(
            PHASE_RATIOS, boundaries[:-1], boundaries[1:], strict=True
        ):
            for local_index in range(start, min(end, maximum_start + 1)):
                groups[name].append(int(positions[local_index]))

    empty = [name for name, values in groups.items() if not values]
    if empty:
        raise ValueError(f"empty phase sampling groups: {empty}")
    return {
        "schema_version": 2,
        "dataset_root": str(dataset_root.resolve()),
        "chunk_size": CHUNK_SIZE,
        "train_episodes": sorted(train_episodes),
        "held_out_episodes": sorted(held_out),
        "sampling_ratios_percent": PHASE_RATIOS,
        "train_phase_frame_counts": {
            name: len(values) for name, values in groups.items()
        },
        "train_sampling_groups": groups,
        "episodes": episode_records,
        "skipped_episodes": skipped,
        "raw_dataset_modified": False,
    }


def balanced_epoch_indices(
    manifest: dict[str, Any], *, length: int, seed: int, epoch: int = 0
) -> list[int]:
    """Sample one deterministic balanced epoch, with replacement."""

    groups = manifest["train_sampling_groups"]
    ratios = manifest["sampling_ratios_percent"]
    rng = random.Random(int(seed) + 1_000_003 * int(epoch))
    counts: dict[str, int] = {}
    assigned = 0
    names = list(ratios)
    for name in names[:-1]:
        count = round(length * int(ratios[name]) / 100)
        counts[name] = count
        assigned += count
    counts[names[-1]] = length - assigned
    sampled = [
        rng.choice(groups[name])
        for name in names
        for _ in range(counts[name])
    ]
    rng.shuffle(sampled)
    return sampled


class PhaseBalancedSampler:
    """Drop-in replacement for LeRobot's EpisodeAwareSampler."""

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        del dataset_from_indices, dataset_to_indices, episode_indices_to_use
        del drop_n_first_frames, drop_n_last_frames, shuffle
        path = os.environ.get("EBIM_PHASE_MANIFEST", "").strip()
        if not path:
            raise ValueError("EBIM_PHASE_MANIFEST is required")
        self.manifest = _read_json(Path(path))
        self.seed = int(seed)
        self._epoch = 0
        self._start_index = 0
        self._mapping = absolute_to_relative_idx
        absolute = set().union(
            *(
                set(int(value) for value in values)
                for values in self.manifest[
                    "train_sampling_groups"
                ].values()
            )
        )
        if self._mapping is not None:
            missing = absolute - set(self._mapping)
            if missing:
                raise ValueError(
                    f"phase manifest has {len(missing)} frames outside training dataset"
                )
        self._length = len(absolute)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._epoch = int(state["epoch"])
        self._start_index = int(state["start_index"])

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        absolute = balanced_epoch_indices(
            self.manifest, length=self._length, seed=self.seed, epoch=epoch
        )
        for index in absolute[start:]:
            yield self._mapping[index] if self._mapping is not None else index

    def __len__(self) -> int:
        return self._length


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_phase_manifest(
        dataset_root=args.dataset_root.resolve(),
        audit_report=args.audit_report.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "phase_counts": manifest["train_phase_frame_counts"],
                "sampling_ratios_percent": PHASE_RATIOS,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
