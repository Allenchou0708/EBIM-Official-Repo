"""Materialize the immutable Phase II valid-only right-arm LeRobot datasets.

The train and held-out roots are intentionally separate.  Each source episode
is copied in the exact order frozen by Gate A, while head and right-wrist RGB
are sampled independently at every parquet timestamp.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from task2_real.contract import validate_contract
from task2_real.smoke_dataset import SequentialNearestVideo


CAMERA_KEYS = ("observation.images.head", "observation.images.wrist_right")
CANONICAL_SHAPES_HWC = {
    "observation.images.head": (224, 400, 3),
    "observation.images.wrist_right": (480, 640, 3),
}
TIMESTAMP_ERROR_LIMITS_S = {
    "observation.images.head": 0.11,
    "observation.images.wrist_right": 0.04,
}
TAIL_REPEAT_CAP_S = 1.05
EXPECTED_SPLIT_COUNTS = {
    "train": (119, 71420),
    "held_out": (30, 15024),
}


def timestamp_sample_policy(
    *, query_s: float, selected_s: float, decoder_exhausted: bool, camera_key: str
) -> tuple[str, float]:
    """Classify an ordinary nearest sample versus an explicit bounded EOS repeat."""

    if camera_key not in TIMESTAMP_ERROR_LIMITS_S:
        raise ValueError(f"unknown camera key: {camera_key}")
    error_s = abs(float(selected_s) - float(query_s))
    is_tail_repeat = bool(decoder_exhausted and query_s > selected_s)
    if is_tail_repeat:
        if error_s > TAIL_REPEAT_CAP_S:
            raise ValueError(
                f"{camera_key} tail-repeat gap {error_s:.6f}s exceeds "
                f"the explicit {TAIL_REPEAT_CAP_S:.2f}s cap"
            )
        return "tail_repeat", error_s
    if error_s > TIMESTAMP_ERROR_LIMITS_S[camera_key]:
        raise ValueError(
            f"{camera_key} nearest timestamp error {error_s:.6f}s exceeds "
            f"{TIMESTAMP_ERROR_LIMITS_S[camera_key]:.6f}s"
        )
    return "nearest", error_s


def load_frozen_split(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    split = json.loads(path.read_text(encoding="utf-8"))
    if split.get("dataset_repo_id") != contract["dataset"]["repo_id"]:
        raise ValueError("split dataset repo differs from the locked contract")
    if split.get("dataset_revision") != contract["dataset"]["revision"]:
        raise ValueError("split revision differs from the locked contract")
    train = [int(value) for value in split.get("train", [])]
    held_out = [int(value) for value in split.get("held_out", [])]
    if len(train) != EXPECTED_SPLIT_COUNTS["train"][0]:
        raise ValueError("frozen train split no longer contains 119 episodes")
    if len(held_out) != EXPECTED_SPLIT_COUNTS["held_out"][0]:
        raise ValueError("frozen held-out split no longer contains 30 episodes")
    if len(set(train)) != len(train) or len(set(held_out)) != len(held_out):
        raise ValueError("frozen split contains duplicate episode IDs")
    if set(train) & set(held_out):
        raise ValueError("frozen train and held-out episode IDs overlap")
    if int(split.get("train_frames", -1)) != EXPECTED_SPLIT_COUNTS["train"][1]:
        raise ValueError("frozen train frame count changed")
    if int(split.get("held_out_frames", -1)) != EXPECTED_SPLIT_COUNTS["held_out"][1]:
        raise ValueError("frozen held-out frame count changed")
    return split


def _source_paths(source_root: Path, episode_index: int) -> tuple[Path, dict[str, Path]]:
    chunk = episode_index // 1000
    parquet_path = (
        source_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    )
    video_paths = {
        key: (
            source_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / key
            / f"episode_{episode_index:06d}.mp4"
        )
        for key in CAMERA_KEYS
    }
    return parquet_path, video_paths


def materialize_split(
    source_root: Path,
    destination_root: Path,
    contract: dict[str, Any],
    split_manifest_path: Path,
    split_name: str,
    repo_id: str,
) -> dict[str, Any]:
    try:
        import numpy as np
        import pyarrow.parquet as parquet
        from PIL import Image
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:  # pragma: no cover - exercised in Docker
        raise RuntimeError("numpy, pyarrow, and LeRobot are required") from error

    validate_contract(contract)
    split = load_frozen_split(split_manifest_path, contract)
    if split_name not in EXPECTED_SPLIT_COUNTS:
        raise ValueError("split name must be train or held_out")
    source_episode_ids = [int(value) for value in split[split_name]]
    expected_episodes, expected_frames = EXPECTED_SPLIT_COUNTS[split_name]
    if destination_root.exists():
        raise ValueError(f"destination already exists: {destination_root}")

    state_indices = [int(value) for value in contract["policy_view"]["state"]["raw_indices"]]
    action_indices = [int(value) for value in contract["policy_view"]["action"]["raw_indices"]]
    if state_indices != list(range(21, 29)) or action_indices != list(range(8, 16)):
        raise ValueError("full adapter requires state[21:29] and action[8:16]")

    for episode_index in source_episode_ids:
        parquet_path, video_paths = _source_paths(source_root, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        for path in video_paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)

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
                "names": list(contract["policy_view"]["state"]["names"]),
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": list(contract["policy_view"]["action"]["names"]),
            },
        }
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(contract["dataset"]["fps"]),
        features=features,
        root=destination_root,
        robot_type="ebim_task2_real_right_only",
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=2,
    )

    instruction = str(contract["policy_view"]["instruction"])
    episode_records: list[dict[str, Any]] = []
    total_frames = 0
    maximum_timestamp_errors = {key: 0.0 for key in CAMERA_KEYS}
    tail_repeat_rows = {key: 0 for key in CAMERA_KEYS}
    maximum_tail_repeat_gaps = {key: 0.0 for key in CAMERA_KEYS}
    source_resolution_counts: dict[str, Counter[tuple[int, int, int]]] = {
        key: Counter() for key in CAMERA_KEYS
    }
    for derived_episode_index, source_episode_index in enumerate(source_episode_ids):
        parquet_path, video_paths = _source_paths(source_root, source_episode_index)
        table = parquet.read_table(
            parquet_path,
            columns=[
                "episode_index",
                "frame_index",
                "timestamp",
                "observation.state",
                "action",
                "annotation.human.validity",
            ],
        )
        rows = table.num_rows
        if {int(value) for value in table["episode_index"].to_pylist()} != {
            source_episode_index
        }:
            raise ValueError(f"source episode index mismatch: {source_episode_index}")
        if table["frame_index"].to_pylist() != list(range(rows)):
            raise ValueError(f"source frame index is not contiguous: {source_episode_index}")
        if {int(value) for value in table["annotation.human.validity"].to_pylist()} != {1}:
            raise ValueError(f"source episode is not fully valid: {source_episode_index}")

        timestamps = [float(value) for value in table["timestamp"].to_pylist()]
        raw_states = table["observation.state"].to_pylist()
        raw_actions = table["action"].to_pylist()
        decoders = {key: SequentialNearestVideo(path) for key, path in video_paths.items()}
        episode_source_shapes: dict[str, list[int]] = {}
        try:
            for row_index, timestamp in enumerate(timestamps):
                images: dict[str, Any] = {}
                for key, decoder in decoders.items():
                    image, selected_timestamp = decoder.sample(timestamp)
                    if row_index == 0:
                        original_shape = tuple(int(value) for value in image.shape)
                        source_resolution_counts[key][original_shape] += 1
                        episode_source_shapes[key] = list(original_shape)
                    target_height, target_width, _ = CANONICAL_SHAPES_HWC[key]
                    if image.shape != CANONICAL_SHAPES_HWC[key]:
                        image = np.array(
                            Image.fromarray(image).resize(
                                (target_width, target_height), resample=Image.Resampling.BILINEAR
                            ),
                            dtype=np.uint8,
                            copy=True,
                        )
                    images[key] = np.asarray(image, dtype=np.uint8)
                    try:
                        sampling_method, error_s = timestamp_sample_policy(
                            query_s=timestamp,
                            selected_s=float(selected_timestamp),
                            decoder_exhausted=decoder.exhausted,
                            camera_key=key,
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"{error} in episode {source_episode_index} row {row_index}"
                        ) from error
                    if sampling_method == "tail_repeat":
                        tail_repeat_rows[key] += 1
                        maximum_tail_repeat_gaps[key] = max(
                            maximum_tail_repeat_gaps[key], error_s
                        )
                    else:
                        maximum_timestamp_errors[key] = max(
                            maximum_timestamp_errors[key], error_s
                        )
                state = np.asarray(
                    [raw_states[row_index][index] for index in state_indices], dtype=np.float32
                )
                action = np.asarray(
                    [raw_actions[row_index][index] for index in action_indices], dtype=np.float32
                )
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError(f"non-finite policy row in episode {source_episode_index}")
                dataset.add_frame(
                    {
                        **images,
                        "observation.state": state,
                        "action": action,
                        "task": instruction,
                    }
                )
            dataset.save_episode()
        finally:
            for decoder in decoders.values():
                decoder.close()

        episode_records.append(
            {
                "derived_episode_index": derived_episode_index,
                "source_episode_index": source_episode_index,
                "frames": rows,
                "global_start": total_frames,
                "global_end_exclusive": total_frames + rows,
                "source_video_shapes_hwc": episode_source_shapes,
            }
        )
        total_frames += rows
        print(
            f"materialized {split_name} {derived_episode_index + 1}/{expected_episodes}: "
            f"source={source_episode_index} frames={rows} total={total_frames}",
            flush=True,
        )

    dataset.finalize()
    if len(episode_records) != expected_episodes or total_frames != expected_frames:
        raise ValueError(
            f"materialized {len(episode_records)}/{total_frames}, "
            f"expected {expected_episodes}/{expected_frames}"
        )
    provenance = {
        "source_repo_id": contract["dataset"]["repo_id"],
        "source_revision": contract["dataset"]["revision"],
        "source_split_manifest": str(split_manifest_path.resolve()),
        "source_split_name": split_name,
        "source_episode_indices": source_episode_ids,
        "source_policy_state_raw_indices": state_indices,
        "source_policy_action_raw_indices": action_indices,
        "raw_action_index_16_excluded": True,
        "source_timestamp_sampling": (
            "nearest frame independently per camera using each parquet timestamp"
        ),
        "end_of_stream_policy": {
            "method": "repeat final decoded frame after decoder exhaustion",
            "hard_cap_s": TAIL_REPEAT_CAP_S,
            "tail_repeat_rows": tail_repeat_rows,
            "maximum_tail_repeat_gap_s": maximum_tail_repeat_gaps,
        },
        "maximum_non_tail_timestamp_error_s": maximum_timestamp_errors,
        "canonical_image_shapes_hwc": {
            key: list(shape) for key, shape in CANONICAL_SHAPES_HWC.items()
        },
        "source_resolution_episode_counts_hwc": {
            key: {
                "x".join(str(value) for value in shape): count
                for shape, count in sorted(source_resolution_counts[key].items())
            }
            for key in CAMERA_KEYS
        },
        "canonicalization_reason": (
            "canonicalize mixed head resolutions at 224x400 before encoding so high-resolution "
            "episodes retain at least PI0.5 target-scale height; PI0.5 subsequently maps policy "
            "inputs to its 224x224 visual target"
        ),
        "episode_records": episode_records,
        "total_frames": total_frames,
    }
    provenance_path = destination_root / "meta" / "ebim_source.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "destination_root": str(destination_root.resolve()),
        "destination_repo_id": repo_id,
        "split": split_name,
        "episodes": len(episode_records),
        "frames": total_frames,
        "source_episode_indices": source_episode_ids,
        "maximum_source_timestamp_error_s": maximum_timestamp_errors,
        "maximum_source_timestamp_error_limit_s": TIMESTAMP_ERROR_LIMITS_S,
        "tail_repeat_rows": tail_repeat_rows,
        "maximum_tail_repeat_gap_s": maximum_tail_repeat_gaps,
        "tail_repeat_hard_cap_s": TAIL_REPEAT_CAP_S,
        "source_resolution_episode_counts_hwc": {
            key: {
                "x".join(str(value) for value in shape): count
                for shape, count in sorted(source_resolution_counts[key].items())
            }
            for key in CAMERA_KEYS
        },
        "provenance": str(provenance_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED_SPLIT_COUNTS), required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=Path(__file__).with_name("contract.json")
    )
    args = parser.parse_args()
    contract = validate_contract(json.loads(args.contract.read_text(encoding="utf-8")))
    report = materialize_split(
        args.source_root.resolve(),
        args.destination_root.resolve(),
        contract,
        args.split_manifest.resolve(),
        args.split,
        args.repo_id,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: materialized {report['split']} episodes={report['episodes']} "
        f"frames={report['frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
