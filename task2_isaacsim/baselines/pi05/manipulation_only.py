#!/usr/bin/env python3
"""Build the event-cropped, right-only Task 2 PI0.5 dataset views.

The deterministic controller owns everything before the retained pre-close
observation.  This adapter keeps one second of open-gripper context followed
by the complete grasp, transport, placement, release, and retreat trajectory.
The immutable episode split is inherited from the existing multi-episode
audit; held-out episodes never contribute rows or statistics to the train
view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FPS = 30
CHUNK_SIZE = 50
PRE_CLOSE_FRAMES = 30
DECODE_BLOCK_FRAMES = 64
CAMERA_KEYS = (
    "observation.images.head",
    "observation.images.wrist_right",
)
CANONICAL_SHAPES_HWC = {
    "observation.images.head": (224, 400, 3),
    "observation.images.wrist_right": (224, 396, 3),
}
STATE_INDICES = (*range(21, 28), 30)
ACTION_INDICES = (*range(10, 17), 18)
PHASE_RATIOS = {
    "pre_close": 25,
    "grasp_acquisition": 25,
    "lift_transfer": 15,
    "lower_place": 15,
    "release_retreat": 20,
}
TASK = "Pick up the thermal pad and place it on the target RAM board."


def crop_record(record: dict[str, Any]) -> dict[str, Any]:
    """Translate source event frames into one cropped episode contract."""

    events = {name: int(value) for name, value in record["events"].items()}
    length = int(record["length"])
    start = events["right_close"] - PRE_CLOSE_FRAMES
    end = length
    if start < 0 or end - start < CHUNK_SIZE:
        raise ValueError(f"episode {record['episode']} has no valid crop")
    ordered = (
        start,
        events["right_close"],
        events["pad_move"],
        events["target_arrival"],
        events["right_release"],
        end,
    )
    if any(right < left for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"episode {record['episode']} event order is invalid")
    names = (*PHASE_RATIOS, "end")
    derived_events = {
        name: int(value - start) for name, value in zip(names, ordered, strict=True)
    }
    return {
        "source_episode_index": int(record["episode"]),
        "source_start_frame": start,
        "source_end_exclusive": end,
        "frames": end - start,
        "derived_events": derived_events,
    }


def phase_groups_for_record(
    record: dict[str, Any], *, derived_global_start: int
) -> dict[str, list[int]]:
    """Return full-horizon training starts for each physical phase."""

    events = record["derived_events"]
    maximum_start = int(record["frames"]) - CHUNK_SIZE
    boundaries = [int(events[name]) for name in PHASE_RATIOS] + [int(events["end"])]
    groups: dict[str, list[int]] = {}
    for name, start, end in zip(PHASE_RATIOS, boundaries[:-1], boundaries[1:], strict=True):
        last = min(end - 1, maximum_start)
        groups[name] = (
            list(range(derived_global_start + start, derived_global_start + last + 1))
            if last >= start
            else []
        )
    return groups


def source_records(landmark_audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Normalize the 200-episode audit into the event vocabulary used here."""

    normalized = []
    for row in landmark_audit["episode_records"]:
        phases = row["phase_frames"]
        # These two independently detected landmarks cross once (episode 15).
        # Their sorted order still gives two observable, non-overlapping motion
        # intervals without dropping or relabeling the episode outcome.
        middle_events = sorted(
            (int(phases["retained_lift"]), int(phases["transfer"]))
        )
        normalized.append(
            {
                "episode": int(row["episode_index"]),
                "length": int(row["frames"]),
                "events": {
                    "right_close": int(phases["grasp"]),
                    "pad_move": middle_events[0],
                    "target_arrival": middle_events[1],
                    "right_release": int(phases["release"]),
                },
            }
        )
    records = {int(row["episode"]): row for row in normalized}
    if set(records) != set(range(200)):
        raise ValueError("landmark audit must contain exactly source episodes 0..199")
    return records


def _split_source_ids(split: str) -> list[int]:
    held_out = list(range(7, 200, 10))
    return held_out if split == "held_out" else [i for i in range(200) if i not in held_out]


def materialize(
    *,
    source_root: Path,
    destination_root: Path,
    landmark_audit_path: Path,
    split: str,
    repo_id: str,
) -> dict[str, Any]:
    """Create one independent train or held-out LeRobot v3 view."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as parquet
    import torch.nn.functional as functional
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames_pyav

    if destination_root.exists():
        raise ValueError(f"destination already exists: {destination_root}")
    landmark_audit = json.loads(landmark_audit_path.read_text(encoding="utf-8"))
    records = source_records(landmark_audit)
    source_ids = _split_source_ids(split)
    source_meta = LeRobotDatasetMetadata(
        "local/task2_fixpos_200",
        root=source_root,
    )
    source_from = [int(value) for value in source_meta.episodes["dataset_from_index"]]
    state_names = source_meta.features["observation.state"]["names"]
    action_names = source_meta.features["action"]["names"]
    tables = [parquet.read_table(path) for path in sorted((source_root / "data").glob("**/*.parquet"))]
    if not tables:
        raise ValueError("source dataset contains no parquet data")
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    source_states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    source_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    source_timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    features = {
        key: {
            "dtype": "video",
            "shape": CANONICAL_SHAPES_HWC[key],
            "names": ["height", "width", "channels"],
        }
        for key in CAMERA_KEYS
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": [state_names[index] for index in STATE_INDICES],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": [action_names[index] for index in ACTION_INDICES],
            },
        }
    )
    destination = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=destination_root,
        robot_type="ebim_task2_sim_right_manipulation_only",
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=2,
    )

    episode_records: list[dict[str, Any]] = []
    phase_groups = {name: [] for name in PHASE_RATIOS}
    total_frames = 0
    for derived_episode, source_episode in enumerate(source_ids):
        crop = crop_record(records[source_episode])
        source_global_start = source_from[source_episode] + crop["source_start_frame"]
        episode_meta = source_meta.episodes[source_episode]
        for block_start in range(0, crop["frames"], DECODE_BLOCK_FRAMES):
            block_end = min(crop["frames"], block_start + DECODE_BLOCK_FRAMES)
            global_start = source_global_start + block_start
            global_end = source_global_start + block_end
            local_timestamps = source_timestamps[global_start:global_end]
            decoded = {}
            for key in CAMERA_KEYS:
                chunk = int(episode_meta[f"videos/{key}/chunk_index"])
                file_index = int(episode_meta[f"videos/{key}/file_index"])
                timestamp_offset = float(episode_meta[f"videos/{key}/from_timestamp"])
                path = (
                    source_root
                    / "videos"
                    / key
                    / f"chunk-{chunk:03d}"
                    / f"file-{file_index:03d}.mp4"
                )
                queries = (local_timestamps + timestamp_offset).tolist()
                frames = decode_video_frames_pyav(
                    path,
                    queries,
                    tolerance_s=1e-4,
                    return_uint8=True,
                )
                target_height, target_width, _ = CANONICAL_SHAPES_HWC[key]
                if tuple(frames.shape[-2:]) != (target_height, target_width):
                    original_dtype = frames.dtype
                    frames = functional.interpolate(
                        frames.float(),
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    ).round().clamp(0, 255).to(dtype=original_dtype)
                decoded[key] = frames.permute(0, 2, 3, 1).numpy()

            for block_offset, global_index in enumerate(range(global_start, global_end)):
                state = source_states[global_index, list(STATE_INDICES)]
                action = source_actions[global_index, list(ACTION_INDICES)]
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError(f"non-finite row in source episode {source_episode}")
                destination.add_frame(
                    {
                        **{key: decoded[key][block_offset] for key in CAMERA_KEYS},
                        "observation.state": state,
                        "action": action,
                        "task": TASK,
                    }
                )
        destination.save_episode()
        crop["derived_episode_index"] = derived_episode
        crop["derived_global_start"] = total_frames
        local_groups = phase_groups_for_record(crop, derived_global_start=total_frames)
        for name, values in local_groups.items():
            phase_groups[name].extend(values)
        episode_records.append(crop)
        total_frames += crop["frames"]
        print(
            f"{split} {derived_episode + 1}/{len(source_ids)}: "
            f"source={source_episode} frames={crop['frames']} total={total_frames}",
            flush=True,
        )
    destination.finalize()

    if any(not values for values in phase_groups.values()):
        raise ValueError("one or more manipulation phase groups are empty")
    report = {
        "schema_version": 1,
        "split": split,
        "source_root": str(source_root.resolve()),
        "destination_root": str(destination_root.resolve()),
        "repo_id": repo_id,
        "source_landmark_audit": str(landmark_audit_path.resolve()),
        "source_episode_indices": source_ids,
        "episodes": len(source_ids),
        "frames": total_frames,
        "fps": FPS,
        "chunk_size": CHUNK_SIZE,
        "pre_close_frames": PRE_CLOSE_FRAMES,
        "policy_state_source_indices": list(STATE_INDICES),
        "policy_action_source_indices": list(ACTION_INDICES),
        "camera_keys": list(CAMERA_KEYS),
        "canonical_image_shapes_hwc": {
            key: list(value) for key, value in CANONICAL_SHAPES_HWC.items()
        },
        "task": TASK,
        "sampling_ratios_percent": PHASE_RATIOS,
        "phase_frame_counts": {name: len(values) for name, values in phase_groups.items()},
        "train_sampling_groups": phase_groups if split == "train" else {},
        "episode_records": episode_records,
        "normalization_scope": "this derived split only",
        "raw_dataset_modified": False,
    }
    (destination_root / "meta" / "ebim_manipulation_only.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--landmark-audit", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "held_out"), required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        source_root=args.source_root.resolve(),
        destination_root=args.destination_root.resolve(),
        landmark_audit_path=args.landmark_audit.resolve(),
        split=args.split,
        repo_id=args.repo_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": result["episodes"], "frames": result["frames"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
