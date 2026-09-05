"""CPU gate for the immutable full Phase II right-only train/held-out views."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from task2_real.full_dataset import CAMERA_KEYS, EXPECTED_SPLIT_COUNTS


TRAIN_REPO_ID = "local/phase2_real_right_train"
HELDOUT_REPO_ID = "local/phase2_real_right_heldout"
RENAME_MAP = {
    "observation.images.head": "observation.images.base_0_rgb",
    "observation.images.wrist_right": "observation.images.right_wrist_0_rgb",
}
BASE_REVISION = "338b5c22c12dbdd0d2ab19046802de2eb7696a6b"
EVAL_QUANTILES = (0.10, 0.35, 0.60, 0.85)
LANDMARK_SOURCE_EPISODES = (210, 229, 251)
TASK_PROMPT = "Pick up the thermal pad and place it on the target RAM board"


def _json_stats(values: Any, np: Any) -> dict[str, list[float]]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute_train_relative_action_stats(
    train_root: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Fit action stats from full-horizon train chunks only, never held-out rows."""

    try:
        import numpy as np
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - exercised in Docker
        raise RuntimeError("numpy and pyarrow are required for the data gate") from error

    tables = [
        parquet.read_table(path, columns=["episode_index", "observation.state", "action"])
        for path in sorted((train_root / "data").glob("**/*.parquet"))
    ]
    if not tables:
        raise ValueError("train root contains no data parquet")
    episode_indices = np.concatenate(
        [np.asarray(table["episode_index"].to_pylist(), dtype=np.int64) for table in tables]
    )
    states = np.concatenate(
        [np.asarray(table["observation.state"].to_pylist(), dtype=np.float32) for table in tables]
    )
    actions = np.concatenate(
        [np.asarray(table["action"].to_pylist(), dtype=np.float32) for table in tables]
    )
    if states.shape != (EXPECTED_SPLIT_COUNTS["train"][1], 8):
        raise ValueError(f"unexpected train state shape for relative stats: {states.shape}")
    if actions.shape != states.shape:
        raise ValueError(f"unexpected train action shape for relative stats: {actions.shape}")

    relative_chunks: list[Any] = []
    valid_chunks = 0
    for record in provenance["episode_records"]:
        episode_index = int(record["derived_episode_index"])
        positions = np.flatnonzero(episode_indices == episode_index)
        expected_frames = int(record["frames"])
        if len(positions) != expected_frames or not np.array_equal(
            positions, np.arange(positions[0], positions[0] + expected_frames)
        ):
            raise ValueError(f"derived episode rows are not contiguous: {episode_index}")
        episode_states = states[positions]
        episode_actions = actions[positions]
        for start in range(expected_frames - 50 + 1):
            chunk = episode_actions[start : start + 50].copy()
            chunk[:, :7] -= episode_states[start, :7]
            relative_chunks.append(chunk)
        valid_chunks += max(0, expected_frames - 50 + 1)
    relative = np.concatenate(relative_chunks, axis=0)
    if not np.isfinite(relative).all():
        raise ValueError("train-derived relative action stats contain non-finite values")
    result = _json_stats(relative, np)
    result.update(
        {
            "fit_scope": "train split only; all full 50-step same-episode chunks",
            "valid_chunks": valid_chunks,
            "relative_rows": int(relative.shape[0]),
            "relative_action_state_indices": [0, 1, 2, 3, 4, 5, 6, None],
            "gripper_remains_absolute": True,
        }
    )
    return result


def _tail_first_rows(video_audit: dict[str, Any]) -> dict[int, dict[str, int | None]]:
    result: dict[int, dict[str, int | None]] = {}
    for record in video_audit["episode_records"]:
        result[int(record["episode_index"])] = {
            key: record["cameras"][key]["end_of_stream_first_row"] for key in CAMERA_KEYS
        }
    return result


def _make_eval_plan(
    heldout_provenance: dict[str, Any], video_audit: dict[str, Any]
) -> list[dict[str, Any]]:
    tail_first = _tail_first_rows(video_audit)
    plan: list[dict[str, Any]] = []
    for record in heldout_provenance["episode_records"]:
        source_episode = int(record["source_episode_index"])
        frames = int(record["frames"])
        safe_last = frames - 50
        for camera_first in tail_first[source_episode].values():
            if camera_first is not None:
                safe_last = min(safe_last, int(camera_first) - 1)
        if safe_last < 3:
            raise ValueError(f"held-out episode {source_episode} has no safe eval horizon")
        local_rows = [int(round(probability * safe_last)) for probability in EVAL_QUANTILES]
        if len(set(local_rows)) != len(local_rows):
            raise ValueError(f"held-out episode {source_episode} quantiles are not unique")
        for probability, local_row in zip(EVAL_QUANTILES, local_rows, strict=True):
            plan.append(
                {
                    "source_episode_index": source_episode,
                    "derived_episode_index": int(record["derived_episode_index"]),
                    "episode_frame_index": local_row,
                    "dataset_index": int(record["global_start"]) + local_row,
                    "temporal_quantile": probability,
                    "full_action_horizon": local_row + 49 < frames,
                    "observation_before_camera_tail_repeat": all(
                        first is None or local_row < int(first)
                        for first in tail_first[source_episode].values()
                    ),
                }
            )
    if len(plan) != 120:
        raise ValueError(f"fixed held-out plan contains {len(plan)} points, expected 120")
    return plan


def _lifecycle(values: list[float]) -> dict[str, int]:
    seen_open = False
    close_start: int | None = None
    held: int | None = None
    for index, value in enumerate(values):
        if not seen_open:
            seen_open = value >= 0.90
            continue
        if close_start is None and value <= 0.25:
            close_start = index
        if close_start is not None and held is None:
            if value > 0.25:
                close_start = None
            elif index - close_start >= 9:
                held = close_start + 9
        if held is not None and value >= 0.90:
            pre_close_candidates = [
                candidate
                for candidate in range(close_start or 0)
                if values[candidate] >= 0.90
            ]
            if not pre_close_candidates:
                raise ValueError("lifecycle has no open pre-close landmark")
            return {
                "pre_close": pre_close_candidates[-1],
                "closed": held,
                "reopen": index,
            }
    raise ValueError("episode lacks complete close-hold-reopen lifecycle")


def _make_landmarks(
    source_root: Path,
    heldout_provenance: dict[str, Any],
    video_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - exercised in Docker
        raise RuntimeError("pyarrow is required for landmark extraction") from error
    by_source = {
        int(record["source_episode_index"]): record
        for record in heldout_provenance["episode_records"]
    }
    tail_first = _tail_first_rows(video_audit)
    landmarks = []
    for source_episode in LANDMARK_SOURCE_EPISODES:
        record = by_source[source_episode]
        path = source_root / "data/chunk-000" / f"episode_{source_episode:06d}.parquet"
        actions = parquet.read_table(path, columns=["action"])["action"].to_pylist()
        phases = _lifecycle([float(row[15]) for row in actions])
        for phase, local_row in phases.items():
            before_tail = all(
                first is None or local_row < int(first)
                for first in tail_first[source_episode].values()
            )
            if not before_tail:
                raise ValueError(
                    f"landmark {source_episode}/{phase} falls in camera tail-repeat region"
                )
            frames = int(record["frames"])
            landmarks.append(
                {
                    "source_episode_index": source_episode,
                    "derived_episode_index": int(record["derived_episode_index"]),
                    "phase": phase,
                    "episode_frame_index": local_row,
                    "dataset_index": int(record["global_start"]) + local_row,
                    "observation_before_camera_tail_repeat": True,
                    "available_target_actions": min(50, frames - local_row),
                }
            )
    return landmarks


def _stats_match(left: dict[str, Any], right: dict[str, Any], np: Any) -> bool:
    for feature in ("observation.state", "action"):
        for name in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            if not np.allclose(np.asarray(left[feature][name]), np.asarray(right[feature][name])):
                return False
    return True


def run(
    train_root: Path,
    heldout_root: Path,
    source_root: Path,
    base_snapshot: Path,
    video_audit_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    train_provenance = json.loads((train_root / "meta/ebim_source.json").read_text())
    heldout_provenance = json.loads((heldout_root / "meta/ebim_source.json").read_text())
    if train_provenance["source_split_name"] != "train":
        raise ValueError("train root provenance is not train")
    if heldout_provenance["source_split_name"] != "held_out":
        raise ValueError("held-out root provenance is not held_out")
    train_ids = set(train_provenance["source_episode_indices"])
    heldout_ids = set(heldout_provenance["source_episode_indices"])
    if train_ids & heldout_ids:
        raise ValueError("derived train/held-out source IDs overlap")

    train_meta = LeRobotDatasetMetadata(TRAIN_REPO_ID, root=train_root)
    heldout_meta = LeRobotDatasetMetadata(HELDOUT_REPO_ID, root=heldout_root)
    raw_datasets = {
        "train": LeRobotDataset(
            TRAIN_REPO_ID,
            root=train_root,
            download_videos=False,
            video_backend="pyav",
            return_uint8=True,
        ),
        "held_out": LeRobotDataset(
            HELDOUT_REPO_ID,
            root=heldout_root,
            download_videos=False,
            video_backend="pyav",
            return_uint8=True,
        ),
    }
    readback: dict[str, Any] = {}
    for split_name, dataset in raw_datasets.items():
        expected_episodes, expected_frames = EXPECTED_SPLIT_COUNTS[split_name]
        provenance = train_provenance if split_name == "train" else heldout_provenance
        if len(dataset) != expected_frames or len(provenance["episode_records"]) != expected_episodes:
            raise ValueError(f"{split_name} row/episode count changed")
        checked = [0, len(dataset) // 2, len(dataset) - 1]
        shapes = []
        for index in checked:
            sample = dataset[index]
            camera_keys = sorted(
                key for key in sample if key.startswith("observation.images.")
            )
            if camera_keys != sorted(CAMERA_KEYS):
                raise ValueError(f"{split_name} camera schema changed: {camera_keys}")
            if tuple(sample["observation.state"].shape) != (8,):
                raise ValueError(f"{split_name} state is not 8-D")
            if tuple(sample["action"].shape) != (8,):
                raise ValueError(f"{split_name} action is not 8-D")
            if sample.get("task") != TASK_PROMPT:
                raise ValueError(f"{split_name} task prompt changed")
            shapes.append({key: list(sample[key].shape) for key in CAMERA_KEYS})
        readback[split_name] = {"indices": checked, "camera_shapes_chw": shapes}

    relative_stats = compute_train_relative_action_stats(train_root, train_provenance)
    processor_stats = copy.deepcopy(train_meta.stats)
    processor_stats["action"] = {
        key: np.asarray(value)
        for key, value in relative_stats.items()
        if key in {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
    }
    config = PreTrainedConfig.from_pretrained(base_snapshot, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError("base checkpoint did not decode as PI05Config")
    config.device = "cpu"
    config.use_relative_actions = True
    config.relative_exclude_joints = ["right_gripper_target_percent"]
    config.relative_action_state_indices = [0, 1, 2, 3, 4, 5, 6, None]
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config, pretrained_path=None, dataset_stats=processor_stats
    )
    normalizers = [step for step in preprocessor.steps if step.__class__.__name__ == "NormalizerProcessorStep"]
    if len(normalizers) != 1:
        raise RuntimeError("canonical PI0.5 preprocessor no longer has one normalizer")
    normalizer = normalizers[0]
    if not bool(getattr(normalizer, "_stats_explicitly_provided", False)):
        raise RuntimeError("PI0.5 normalizer did not retain explicitly supplied train stats")
    expected_stats = {
        "observation.state": processor_stats["observation.state"],
        "action": processor_stats["action"],
    }
    actual_stats = {
        "observation.state": normalizer.stats["observation.state"],
        "action": normalizer.stats["action"],
    }
    if not _stats_match(actual_stats, expected_stats, np):
        raise RuntimeError("PI0.5 normalizer stats differ from explicit train-only stats")

    video_audit = json.loads(video_audit_path.read_text())
    eval_plan = _make_eval_plan(heldout_provenance, video_audit)
    delta_timestamps = resolve_delta_timestamps(config, heldout_meta)
    heldout_chunked = LeRobotDataset(
        HELDOUT_REPO_ID,
        root=heldout_root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    for point in eval_plan:
        sample = heldout_chunked[int(point["dataset_index"])]
        if tuple(sample["action"].shape) != (50, 8):
            raise ValueError("fixed eval sample action chunk is not 50x8")
        if "action_is_pad" not in sample or bool(sample["action_is_pad"].any()):
            raise ValueError("fixed eval sample uses canonical action padding")
        if not point["full_action_horizon"] or not point["observation_before_camera_tail_repeat"]:
            raise ValueError("fixed eval plan violates full-horizon/non-tail policy")

    landmarks = _make_landmarks(source_root, heldout_provenance, video_audit)
    schema_text = json.dumps(train_meta.features, sort_keys=True).lower()
    forbidden = ("left", "spine", "raw_action_16")
    if any(token in schema_text for token in forbidden):
        raise ValueError("trainable schema contains a forbidden owner")
    return {
        "success": True,
        "source_revision": train_provenance["source_revision"],
        "counts": {
            "train_episodes": len(train_provenance["episode_records"]),
            "train_frames": len(raw_datasets["train"]),
            "heldout_episodes": len(heldout_provenance["episode_records"]),
            "heldout_frames": len(raw_datasets["held_out"]),
            "source_id_overlap": sorted(train_ids & heldout_ids),
        },
        "schema": {
            "camera_keys": list(CAMERA_KEYS),
            "state_shape": [8],
            "action_shape": [8],
            "action_chunk_shape": [50, 8],
            "raw_action_index_16_excluded": True,
            "task_prompt": TASK_PROMPT,
        },
        "readback": readback,
        "normalization_proof": {
            "normalizer_stats_explicitly_provided": True,
            "observation_stats_source": "train root meta/stats.json",
            "action_stats_source": "relative actions fit from full-horizon train chunks only",
            "heldout_root_stats_loaded_for_comparison_only": True,
            "heldout_root_stats_passed_to_preprocessor": False,
            "heldout_transform_uses_train_processor_stats": True,
            "train_and_heldout_raw_state_stats_differ": not np.allclose(
                np.asarray(train_meta.stats["observation.state"]["mean"]),
                np.asarray(heldout_meta.stats["observation.state"]["mean"]),
            ),
        },
        "train_relative_action_stats": relative_stats,
        "heldout_eval": {
            "samples": len(eval_plan),
            "episodes": len({point["source_episode_index"] for point in eval_plan}),
            "points_per_episode": 4,
            "determinism": "fixed dataset indices; training runner resets identical noise/timestep seed",
            "all_full_action_horizon": True,
            "all_observations_before_camera_tail_repeat": True,
            "plan": eval_plan,
        },
        "landmarks": landmarks,
        "heldout_root_descriptive_stats_ignored": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--video-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_root,
        args.heldout_root,
        args.source_root,
        args.base_snapshot,
        args.video_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: train={result['counts']['train_frames']} heldout="
        f"{result['counts']['heldout_frames']} eval={result['heldout_eval']['samples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
