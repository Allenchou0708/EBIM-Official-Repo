#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build a non-destructive seven-prompt Task 2 PI0.5 dataset view."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .phase_balance import CHUNK_SIZE

PHASE_PROMPTS = {
    "startup_rise": (
        "Raise the spine while keeping both wrists at table height."
    ),
    "approach": "Move the open right gripper above the thermal pad.",
    "orient_pregrasp": (
        "Rotate the right gripper vertical and center the thermal pad in the "
        "right wrist view."
    ),
    "grasp_acquisition": (
        "Close the right gripper on the thermal pad and lift it."
    ),
    "lift_transfer": "Carry the thermal pad to the target RAM board.",
    "lower_place": "Lower the thermal pad onto the target RAM board.",
    "release_retreat": "Open the right gripper and retreat safely.",
}

PHASE_RATIOS = {
    "startup_rise": 20,
    "approach": 15,
    "orient_pregrasp": 20,
    "grasp_acquisition": 20,
    "lift_transfer": 10,
    "lower_place": 10,
    "release_retreat": 5,
}


def phase_for_frame(
    frame: int, *, events: dict[str, int], orientation_entry: int
) -> str:
    boundaries = (
        ("startup_rise", int(events["spine_high"])),
        ("approach", int(orientation_entry)),
        ("orient_pregrasp", int(events["right_close"])),
        ("grasp_acquisition", int(events["pad_move"])),
        ("lift_transfer", int(events["target_arrival"])),
        ("lower_place", int(events["right_release"])),
        ("release_retreat", int(events["end"])),
    )
    if not 0 <= frame < int(events["end"]):
        raise ValueError(f"frame {frame} outside episode")
    previous = 0
    for phase, end in boundaries:
        if end < previous:
            raise ValueError("phase boundaries are not monotonic")
        if frame < end:
            return phase
        previous = end
    raise AssertionError("unreachable phase lookup")


def _copy_unchanged_tree(source: Path, destination: Path) -> None:
    replaced_prefixes = ("data/", "meta/episodes/")
    replaced_files = {"meta/info.json", "meta/tasks.parquet"}
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        relative_name = relative.as_posix()
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if relative_name in replaced_files or relative_name.startswith(
            replaced_prefixes
        ):
            continue
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, destination_path)
        except OSError:
            shutil.copy2(source_path, destination_path, follow_symlinks=True)


def materialize_phase_conditioned_view(
    *, source_root: Path, pose_audit: Path, destination_root: Path
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required") from error

    source = source_root.resolve()
    destination = destination_root.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("destination must be outside the source dataset")
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    audit = json.loads(pose_audit.read_text(encoding="utf-8"))
    records = {int(row["episode"]): row for row in audit["episodes"]}
    train_episodes = {
        episode
        for episode, row in records.items()
        if row["split"] == "train"
    }
    held_out_episodes = {
        episode
        for episode, row in records.items()
        if row["split"] == "held_out"
    }
    prompt_names = list(PHASE_PROMPTS)
    task_indices = {name: index for index, name in enumerate(prompt_names)}
    fallback_task_index = len(prompt_names)
    original_task = (
        "Pick up the thermal pad and place it on the target RAM board."
    )
    groups: dict[str, list[int]] = {name: [] for name in prompt_names}
    task_counts = {name: 0 for name in prompt_names}
    fallback_episodes: set[int] = set()
    global_position = 0

    staging = destination.with_name(
        f"{destination.name}.partial-{os.getpid()}"
    )
    if staging.exists():
        raise ValueError(f"staging destination already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        _copy_unchanged_tree(source, staging)
        for source_path in sorted((source / "data").glob("**/*.parquet")):
            table = parquet.read_table(source_path)
            episodes = table["episode_index"].to_pylist()
            frames = table["frame_index"].to_pylist()
            derived_indices: list[int] = []
            for row_offset, (episode_value, frame_value) in enumerate(
                zip(episodes, frames, strict=True)
            ):
                episode = int(episode_value)
                frame = int(frame_value)
                record = records.get(episode)
                if record is None:
                    fallback_episodes.add(episode)
                    derived_indices.append(fallback_task_index)
                    continue
                phase = phase_for_frame(
                    frame,
                    events=record["events"],
                    orientation_entry=int(record["orientation_entry_frame"]),
                )
                derived_indices.append(task_indices[phase])
                task_counts[phase] += 1
                if (
                    episode in train_episodes
                    and frame <= int(record["events"]["end"]) - CHUNK_SIZE
                ):
                    groups[phase].append(global_position + row_offset)
            relative = source_path.relative_to(source)
            output_path = staging / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            task_field = table.schema.field("task_index")
            table = table.set_column(
                table.column_names.index("task_index"),
                task_field,
                pa.array(derived_indices, type=task_field.type),
            )
            parquet.write_table(table, output_path)
            global_position += len(table)

        tasks = [
            {"task_index": task_indices[name], "task": PHASE_PROMPTS[name]}
            for name in prompt_names
        ]
        tasks.append(
            {"task_index": fallback_task_index, "task": original_task}
        )
        parquet.write_table(
            pa.Table.from_pylist(tasks), staging / "meta" / "tasks.parquet"
        )

        for source_path in sorted(
            (source / "meta" / "episodes").glob("**/*.parquet")
        ):
            table = parquet.read_table(source_path)
            episode_tasks = []
            for episode_value in table["episode_index"].to_pylist():
                episode = int(episode_value)
                episode_tasks.append(
                    [PHASE_PROMPTS[name] for name in prompt_names]
                    if episode in records
                    else [original_task]
                )
            tasks_field = table.schema.field("tasks")
            table = table.set_column(
                table.column_names.index("tasks"),
                tasks_field,
                pa.array(episode_tasks, type=tasks_field.type),
            )
            relative = source_path.relative_to(source)
            output_path = staging / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_table(table, output_path)

        info = json.loads((source / "meta" / "info.json").read_text())
        info["total_tasks"] = len(tasks)
        (staging / "meta" / "info.json").write_text(
            json.dumps(info, indent=4) + "\n", encoding="utf-8"
        )
        empty_groups = [name for name, values in groups.items() if not values]
        if empty_groups:
            raise ValueError(f"empty V4 sampling groups: {empty_groups}")
        manifest = {
            "schema_version": 1,
            "source_root": str(source),
            "destination_root": str(destination),
            "pose_audit": str(pose_audit.resolve()),
            "phase_prompts": PHASE_PROMPTS,
            "task_indices": task_indices,
            "sampling_ratios_percent": PHASE_RATIOS,
            "train_episodes": sorted(train_episodes),
            "held_out_episodes": sorted(held_out_episodes),
            "fallback_episodes_excluded_from_training": sorted(
                fallback_episodes
            ),
            "task_string_frame_counts": task_counts,
            "train_phase_frame_counts": {
                name: len(values) for name, values in groups.items()
            },
            "train_sampling_groups": groups,
            "raw_dataset_modified": False,
        }
        (staging / "phase_conditioned_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pose-audit", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize_phase_conditioned_view(
        source_root=args.source_root,
        pose_audit=args.pose_audit,
        destination_root=args.destination_root,
    )
    print(
        json.dumps(
            {
                "destination_root": manifest["destination_root"],
                "phase_counts": manifest["train_phase_frame_counts"],
                "task_string_counts": manifest["task_string_frame_counts"],
                "excluded": manifest[
                    "fallback_episodes_excluded_from_training"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
