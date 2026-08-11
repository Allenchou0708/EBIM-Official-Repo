#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate PI0.5 checkpoints on a fixed held-out frame sample."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any

from .contract import POLICY_CAMERA_RENAME_MAP, RELATIVE_ACTION_STATE_INDICES
from .offline_inference import run_offline_inference


def deterministic_frame_indices(length: int, maximum: int) -> list[int]:
    if length <= 0 or maximum <= 0:
        raise ValueError("dataset length and maximum frames must be positive")
    if length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [length // 2]
    return sorted(
        {round(slot * (length - 1) / (maximum - 1)) for slot in range(maximum)}
    )


def run_heldout_loss(
    *,
    checkpoint: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    episodes: list[int],
    seed: int,
    max_frames: int,
) -> dict[str, Any]:
    """Compute deterministic LeRobot-native loss without updating weights."""

    try:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.collate import lerobot_collate_fn
    except ImportError as error:
        raise RuntimeError(
            "run held-out evaluation inside the pinned PI0.5 image"
        ) from error

    checkpoint = checkpoint.resolve()
    dataset_root = dataset_root.resolve()
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.pretrained_path = str(checkpoint)
    config.pretrained_revision = None
    config.device = "cuda"
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    policy = make_policy(
        config,
        ds_meta=dataset.meta,
        rename_map=POLICY_CAMERA_RENAME_MAP,
    )
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
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
    )

    selected = deterministic_frame_indices(len(dataset), max_frames)
    subset = torch.utils.data.Subset(dataset, selected)
    collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    losses: list[float] = []
    with torch.no_grad():
        for sample_number, batch in enumerate(loader):
            fixed_seed = seed + selected[sample_number]
            torch.manual_seed(fixed_seed)
            torch.cuda.manual_seed_all(fixed_seed)
            for camera in dataset.meta.camera_keys:
                if camera in batch and batch[camera].dtype == torch.uint8:
                    batch[camera] = batch[camera].to(torch.float32) / 255.0
            processed = preprocessor(batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(processed)
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise ValueError(
                    f"non-finite held-out loss at sample {sample_number}"
                )
            losses.append(value)

    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "episodes": episodes,
        "seed": seed,
        "selected_dataset_indices": selected,
        "evaluated_frames": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "minimum_loss": min(losses),
        "maximum_loss": max(losses),
        "finite": True,
        "loss_contract": "lerobot_native_20d",
    }
    del policy, dataset, loader, subset
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _checkpoint_step(checkpoint: Path) -> int:
    candidates = (checkpoint.name, checkpoint.parent.name)
    for candidate in candidates:
        if candidate.isdigit():
            return int(candidate)
    raise ValueError(f"cannot infer training step from {checkpoint}")


def checkpoint_sweep(
    *,
    checkpoints_root: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    episodes: list[int],
    seed: int,
    max_frames: int,
    report_directory: Path,
    rollout_constraints: dict[str, Any],
) -> dict[str, Any]:
    checkpoints = sorted(
        checkpoints_root.glob("*/pretrained_model"),
        key=_checkpoint_step,
    )
    if not checkpoints:
        raise ValueError(
            f"no */pretrained_model checkpoints below {checkpoints_root}"
        )
    report_directory.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        step = _checkpoint_step(checkpoint)
        loss_report = run_heldout_loss(
            checkpoint=checkpoint,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            episodes=episodes,
            seed=seed,
            max_frames=max_frames,
        )
        shadow_path = report_directory / f"step_{step:06d}_shadow.json"
        shadow = run_offline_inference(
            checkpoint=checkpoint,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            episodes=episodes,
            output=shadow_path,
            samples_per_episode=1,
            seed=seed,
            rollout_constraints=rollout_constraints,
        )
        results.append(
            {
                "step": step,
                **loss_report,
                "offline_shadow_report": str(shadow_path),
                "finite_20d_outputs": shadow["finite_20d_outputs"],
                "joint_and_gripper_bounds_valid": shadow[
                    "joint_and_gripper_bounds_valid"
                ],
                "checkpoint_replay_reproducible": shadow[
                    "checkpoint_replay_reproducible"
                ],
            }
        )
    valid = [result for result in results if result["finite"]]
    ranked = sorted(valid, key=lambda result: result["mean_loss"])
    candidates = [
        {
            "step": result["step"],
            "checkpoint": result["checkpoint"],
            "mean_loss": result["mean_loss"],
        }
        for result in ranked[:2]
    ]
    best = ranked[0]
    return {
        "schema_version": 1,
        "seed": seed,
        "held_out_episodes": episodes,
        "results": results,
        "selected_checkpoint": best["checkpoint"],
        "selected_step": best["step"],
        "selected_mean_loss": best["mean_loss"],
        "selected_candidates": candidates,
    }
