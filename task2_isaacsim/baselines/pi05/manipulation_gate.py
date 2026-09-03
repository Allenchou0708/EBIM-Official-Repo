#!/usr/bin/env python3
"""Validate and finalize the train-only statistics for manipulation-only PI0.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .manipulation_only import CHUNK_SIZE, PHASE_RATIOS


EXPECTED = {
    "train": (180, 90028),
    "held_out": (20, 10302),
}


def numeric_stats(values: Any, np: Any) -> dict[str, list[float]]:
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


def phase_sample_counts(manifest: dict[str, Any], *, epoch_size: int) -> dict[str, int]:
    assigned = 0
    result = {}
    names = list(PHASE_RATIOS)
    for name in names[:-1]:
        count = round(epoch_size * PHASE_RATIOS[name] / 100)
        result[name] = count
        assigned += count
    result[names[-1]] = epoch_size - assigned
    if sum(result.values()) != epoch_size:
        raise AssertionError("phase sample counts do not sum to epoch size")
    if any(not manifest["train_sampling_groups"][name] for name in names):
        raise ValueError("phase manifest contains an empty sampling group")
    return result


def _load_arrays(root: Path, np: Any, parquet: Any) -> tuple[Any, Any, Any]:
    tables = [
        parquet.read_table(path, columns=["episode_index", "observation.state", "action"])
        for path in sorted((root / "data").glob("**/*.parquet"))
    ]
    if not tables:
        raise ValueError(f"no parquet data under {root}")
    episodes = np.concatenate(
        [np.asarray(table["episode_index"].to_pylist(), dtype=np.int64) for table in tables]
    )
    states = np.concatenate(
        [np.asarray(table["observation.state"].to_pylist(), dtype=np.float32) for table in tables]
    )
    actions = np.concatenate(
        [np.asarray(table["action"].to_pylist(), dtype=np.float32) for table in tables]
    )
    return episodes, states, actions


def _video_readback(root: Path, repo_id: str) -> list[dict[str, Any]]:
    """Decode deterministic first/middle/last samples from a finalized view."""

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id,
        root=root,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    records = []
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        record = {"index": index}
        for key in ("observation.images.head", "observation.images.wrist_right"):
            value = sample[key]
            if value.dtype != torch.uint8 or value.ndim != 3:
                raise ValueError(f"{key} readback contract failed at index {index}")
            record[key] = {"shape_chw": list(value.shape), "dtype": str(value.dtype)}
        for key in ("observation.state", "action"):
            value = sample[key]
            if tuple(value.shape) != (8,) or not torch.isfinite(value).all():
                raise ValueError(f"{key} readback contract failed at index {index}")
            record[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        records.append(record)
    return records


def compute_relative_action_stats(
    states: Any, actions: Any, manifest: dict[str, Any], np: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    starts = sorted(
        set().union(
            *(set(int(value) for value in values) for values in manifest["train_sampling_groups"].values())
        )
    )
    chunks = np.empty((len(starts), CHUNK_SIZE, 8), dtype=np.float32)
    for output_index, start in enumerate(starts):
        chunks[output_index] = actions[start : start + CHUNK_SIZE]
    chunks[:, :, :7] -= states[np.asarray(starts), :7, None].transpose(0, 2, 1)
    flattened = chunks.reshape(-1, 8)
    if not np.isfinite(flattened).all():
        raise ValueError("relative action statistics contain non-finite values")
    gripper = actions[np.asarray(starts), 7]
    lifecycle = {
        "eligible_chunk_starts": len(starts),
        "relative_action_rows": int(flattened.shape[0]),
        "first_action_gripper": {
            "closed_le_0_25": int((gripper <= 0.25).sum()),
            "transition": int(((gripper > 0.25) & (gripper < 0.90)).sum()),
            "open_ge_0_90": int((gripper >= 0.90).sum()),
        },
    }
    return numeric_stats(flattened, np), lifecycle


def run(train_root: Path, heldout_root: Path, *, patch_train_stats: bool) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as parquet
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    manifests = {
        split: json.loads((root / "meta" / "ebim_manipulation_only.json").read_text())
        for split, root in (("train", train_root), ("held_out", heldout_root))
    }
    for split, manifest in manifests.items():
        if (int(manifest["episodes"]), int(manifest["frames"])) != EXPECTED[split]:
            raise ValueError(f"{split} count drift: {manifest['episodes']}/{manifest['frames']}")
    train_sources = set(manifests["train"]["source_episode_indices"])
    held_sources = set(manifests["held_out"]["source_episode_indices"])
    if train_sources & held_sources or train_sources | held_sources != set(range(200)):
        raise ValueError("train/held-out source split is not an exact partition")

    train_episodes, train_states, train_actions = _load_arrays(train_root, np, parquet)
    held_episodes, held_states, held_actions = _load_arrays(heldout_root, np, parquet)
    if train_states.shape != (EXPECTED["train"][1], 8) or train_actions.shape != train_states.shape:
        raise ValueError("train numeric shape differs from the 8-D contract")
    if held_states.shape != (EXPECTED["held_out"][1], 8) or held_actions.shape != held_states.shape:
        raise ValueError("held-out numeric shape differs from the 8-D contract")
    if len(set(train_episodes.tolist())) != 180 or len(set(held_episodes.tolist())) != 20:
        raise ValueError("derived episode cardinality mismatch")
    if not np.isfinite(train_states).all() or not np.isfinite(train_actions).all():
        raise ValueError("train numeric data contain non-finite values")
    if not np.isfinite(held_states).all() or not np.isfinite(held_actions).all():
        raise ValueError("held-out numeric data contain non-finite values")

    # The materializer records groups, while this gate owns the final sampling ratios.
    manifests["train"]["sampling_ratios_percent"] = PHASE_RATIOS
    manifests["train"]["train_episodes"] = list(range(180))
    relative_stats, lifecycle = compute_relative_action_stats(
        train_states, train_actions, manifests["train"], np
    )
    epoch_size = lifecycle["eligible_chunk_starts"]
    epoch_counts = phase_sample_counts(manifests["train"], epoch_size=epoch_size)
    phase_gripper = {}
    balanced_gripper = {"closed": 0.0, "transition": 0.0, "open": 0.0}
    for name, ratio in PHASE_RATIOS.items():
        indices = np.asarray(manifests["train"]["train_sampling_groups"][name], dtype=np.int64)
        values = train_actions[indices, 7]
        counts = {
            "closed": int((values <= 0.25).sum()),
            "transition": int(((values > 0.25) & (values < 0.90)).sum()),
            "open": int((values >= 0.90).sum()),
        }
        phase_gripper[name] = {"eligible": len(values), **counts}
        for phase in balanced_gripper:
            balanced_gripper[phase] += ratio / 100 * counts[phase] / len(values)

    stats_path = train_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    absolute_action_stats = manifests["train"].get(
        "absolute_action_stats_before_relative_patch", stats["action"]
    )
    if patch_train_stats:
        stats["action"] = relative_stats
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        manifests["train"][
            "absolute_action_stats_before_relative_patch"
        ] = absolute_action_stats
        manifests["train"]["action_stats_representation"] = (
            "relative right joint targets; absolute right gripper"
        )
        (train_root / "meta" / "ebim_manipulation_only.json").write_text(
            json.dumps(manifests["train"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    train_meta = LeRobotDatasetMetadata(
        manifests["train"]["repo_id"], root=train_root
    )
    heldout_meta = LeRobotDatasetMetadata(
        manifests["held_out"]["repo_id"], root=heldout_root
    )
    expected_keys = {
        "observation.images.head",
        "observation.images.wrist_right",
        "observation.state",
        "action",
    }
    if set(train_meta.features) - {"timestamp", "frame_index", "episode_index", "index", "task_index"} != expected_keys:
        raise ValueError("train feature contract contains an unexpected policy feature")
    if tuple(train_meta.features["observation.state"]["shape"]) != (8,) or tuple(
        train_meta.features["action"]["shape"]
    ) != (8,):
        raise ValueError("train feature contract is not 8-D")

    return {
        "success": True,
        "counts": {
            "train_episodes": 180,
            "train_frames": len(train_states),
            "heldout_episodes": 20,
            "heldout_frames": len(held_states),
            "source_overlap": sorted(train_sources & held_sources),
        },
        "train_only_normalization": {
            "patched": patch_train_stats,
            "absolute_action_stats_before_patch": absolute_action_stats,
            "relative_action_stats": relative_stats,
            "heldout_stats_used": False,
            "train_and_heldout_state_means_differ": not np.allclose(
                np.asarray(train_meta.stats["observation.state"]["mean"]),
                np.asarray(heldout_meta.stats["observation.state"]["mean"]),
            ),
        },
        "lifecycle": lifecycle,
        "phase_sampling": {
            "ratios_percent": PHASE_RATIOS,
            "eligible_counts": manifests["train"]["phase_frame_counts"],
            "samples_per_balanced_epoch": epoch_counts,
            "gripper_labels_by_phase": phase_gripper,
            "expected_balanced_gripper_fraction": balanced_gripper,
        },
        "video_readback": {
            "train": _video_readback(
                train_root, manifests["train"]["repo_id"]
            ),
            "held_out": _video_readback(
                heldout_root, manifests["held_out"]["repo_id"]
            ),
        },
        "raw_source_modified": False,
        "task_success": False,
        "generalization_validated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-train-stats", action="store_true")
    args = parser.parse_args()
    result = run(
        args.train_root.resolve(),
        args.heldout_root.resolve(),
        patch_train_stats=args.patch_train_stats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"success": True, **result["counts"], **result["lifecycle"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
