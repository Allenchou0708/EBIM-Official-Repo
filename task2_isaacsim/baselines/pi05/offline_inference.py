#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run checkpoint inference without publishing ROS commands."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .contract import (
    ACTION_SIZE,
    POLICY_CAMERA_RENAME_MAP,
    RELATIVE_ACTION_STATE_INDICES,
    validate_absolute_action_bounds,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for pattern in ("*.safetensors", "config.json", "train_config.json"):
        for path in sorted(checkpoint.glob(pattern)):
            hashes[path.name] = _sha256_file(path)
    return hashes


def _sample_indices_by_episode(
    dataset: Any, samples_per_episode: int
) -> list[tuple[int, int]]:
    raw = dataset.select_columns("episode_index")
    positions: dict[int, list[int]] = {}
    for index, value in enumerate(raw["episode_index"]):
        positions.setdefault(int(value), []).append(index)
    selected: list[tuple[int, int]] = []
    for episode, indices in sorted(positions.items()):
        if samples_per_episode >= len(indices):
            chosen = indices
        elif samples_per_episode == 1:
            chosen = [indices[len(indices) // 2]]
        else:
            chosen = [
                indices[
                    round(
                        slot * (len(indices) - 1) / (samples_per_episode - 1)
                    )
                ]
                for slot in range(samples_per_episode)
            ]
        selected.extend((episode, index) for index in chosen)
    return selected


def run_offline_inference(
    *,
    checkpoint: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    episodes: list[int],
    output: Path,
    samples_per_episode: int,
    seed: int,
    rollout_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a local checkpoint and produce deterministic shadow commands."""

    try:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies import make_policy, make_pre_post_processors
    except ImportError as error:
        raise RuntimeError(
            "run this command inside the pinned Task 2 PI0.5 image"
        ) from error

    if samples_per_episode <= 0:
        raise ValueError("samples_per_episode must be positive")
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=episodes,
    )
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.pretrained_path = str(checkpoint)
    config.pretrained_revision = None
    config.device = "cuda"
    policy = make_policy(
        config, ds_meta=dataset.meta, rename_map=POLICY_CAMERA_RENAME_MAP
    )
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {
                "rename_map": POLICY_CAMERA_RENAME_MAP
            },
            "relative_actions_processor": {
                "enabled": True,
                "exclude_joints": [],
                "action_names": list(
                    dataset.meta.features["action"].get("names", [])
                ),
                "state_indices": list(RELATIVE_ACTION_STATE_INDICES),
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
            "absolute_actions_processor": {"enabled": True},
        },
    )

    constraints = dict(rollout_constraints or {})
    base_clamp = constraints.get("base_action_variance") is not None and (
        constraints.get("base_vx_vy_wz_clamp_to_zero") is True
    )
    spine_hold = constraints.get("spine_hold_dataset_median")
    if spine_hold is not None:
        spine_hold = float(spine_hold)
        if not math.isfinite(spine_hold):
            raise ValueError("spine hold value must be finite")

    def predict(dataset_index: int) -> list[float]:
        policy.reset()
        fixed_seed = seed + dataset_index
        torch.manual_seed(fixed_seed)
        torch.cuda.manual_seed_all(fixed_seed)
        frame = dict(dataset[dataset_index])
        frame.pop("action", None)
        with torch.inference_mode():
            processed = preprocessor(frame)
            action = postprocessor(policy.select_action(processed))
        return action.detach().cpu().reshape(-1).tolist()

    records: list[dict[str, Any]] = []
    for episode, dataset_index in _sample_indices_by_episode(
        dataset, samples_per_episode
    ):
        values = predict(dataset_index)
        replay = predict(dataset_index)
        if len(values) != ACTION_SIZE:
            raise ValueError(
                f"checkpoint produced {len(values)} values, "
                f"expected {ACTION_SIZE}"
            )
        if len(replay) != ACTION_SIZE:
            raise ValueError(
                "checkpoint replay produced the wrong action size"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("checkpoint produced a non-finite action")
        maximum_replay_error = max(
            abs(left - right)
            for left, right in zip(values, replay, strict=True)
        )
        if maximum_replay_error > 1e-5:
            raise ValueError(
                "checkpoint reload/replay is not deterministic: "
                f"max error={maximum_replay_error}"
            )
        validate_absolute_action_bounds(values)
        effective = list(values)
        if base_clamp:
            effective[:3] = [0.0, 0.0, 0.0]
        if spine_hold is not None:
            effective[19] = spine_hold
        validate_absolute_action_bounds(effective)
        records.append(
            {
                "episode_index": episode,
                "dataset_index": dataset_index,
                "model_absolute_action": values,
                "effective_absolute_action": effective,
                "maximum_replay_error": maximum_replay_error,
                "reproducible": True,
                "base_nonzero": any(abs(value) > 1e-6 for value in values[:3]),
                "effective_base_nonzero": any(
                    abs(value) > 1e-6 for value in effective[:3]
                ),
                "published": False,
            }
        )

    report = {
        "schema_version": 1,
        "mode": "offline_shadow",
        "image_digest": os.environ.get("EBIM_PI05_IMAGE_DIGEST", "unrecorded"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_hashes": _checkpoint_hashes(checkpoint.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_stats_sha256": _sha256_file(
            dataset_root.resolve() / "meta" / "stats.json"
        ),
        "episodes": episodes,
        "relative_action_state_indices": list(RELATIVE_ACTION_STATE_INDICES),
        "samples_per_episode": samples_per_episode,
        "seed": seed,
        "finite_20d_outputs": True,
        "joint_and_gripper_bounds_valid": True,
        "checkpoint_replay_reproducible": True,
        "rollout_constraints": constraints,
        "base_nonzero_records": sum(
            1 for record in records if record["base_nonzero"]
        ),
        "effective_base_nonzero_records": sum(
            1 for record in records if record["effective_base_nonzero"]
        ),
        "ros_publication": False,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del policy, dataset, preprocessor, postprocessor
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="local/task2_pi05_shadow")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    episodes = sorted(
        {int(value) for value in args.episodes.split(",") if value.strip()}
    )
    try:
        report = run_offline_inference(
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            dataset_repo_id=args.dataset_repo_id,
            episodes=episodes,
            output=args.output,
            samples_per_episode=args.samples_per_episode,
            seed=args.seed,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}")
        return 2
    print(
        f"PASS: {len(report['records'])} finite 20-D shadow outputs; "
        f"base_nonzero={report['base_nonzero_records']} ROS publication=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
