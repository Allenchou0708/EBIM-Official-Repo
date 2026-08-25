#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run LeRobot PI0.5 training with Task 2's phase-balanced sampler."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from .phase_balance import PhaseBalancedSampler
from .phase_conditioned_dataset import PHASE_PROMPTS


ACTION_LOSS_WEIGHTS_ENV = "EBIM_ACTION_LOSS_WEIGHTS"


def parse_action_loss_weights(raw: str, *, action_dim: int = 20) -> list[float]:
    """Parse positive per-action loss weights from the training environment."""

    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != action_dim:
        raise ValueError(
            f"{ACTION_LOSS_WEIGHTS_ENV} must contain {action_dim} weights"
        )
    weights = [float(item) for item in value]
    if any(not math.isfinite(item) or item <= 0.0 for item in weights):
        raise ValueError("action loss weights must all be finite and positive")
    return weights


def install_weighted_pi05_forward(policy_class, weights: list[float]) -> None:
    """Weight PI0.5 flow losses by action dimension without patching LeRobot."""

    import torch
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    def weighted_forward(self, batch, reduction="mean"):
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        losses = self.model.forward(
            images, img_masks, tokens, masks, actions, noise, time
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        if original_action_dim != len(weights):
            raise ValueError(
                "configured action dimension does not match weighted loss"
            )
        dimension_weights = torch.as_tensor(
            weights, dtype=losses.dtype, device=losses.device
        )
        dimension_weights = dimension_weights / dimension_weights.mean()
        weighted = losses * dimension_weights.view(1, 1, -1)
        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1])
            .detach()
            .cpu()
            .numpy()
            .tolist(),
            "action_loss_weights": list(weights),
        }
        if reduction == "none":
            per_sample_loss = weighted.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        loss = weighted.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    policy_class.forward = weighted_forward


def disable_local_dataset_hub_fallback(dataset_class) -> None:
    """Build the reader cache from a mounted local dataset, never the Hub."""

    from lerobot.datasets import lerobot_dataset as dataset_module

    def local_only_download(self, download_videos: bool = True) -> None:
        del download_videos
        if self._requested_root is None or not self._requested_root.is_dir():
            raise FileNotFoundError("local-only training dataset root is missing")

    dataset_class._download = local_only_download
    dataset_module.get_safe_version = lambda repo_id, revision: revision


def resolve_phase_task(sample: dict, prompts: list[str]) -> dict:
    """Replace a scalar task index with its V4 prompt before collation."""

    value = sample.get("task")
    if isinstance(value, str):
        return sample
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"phase task must be a scalar index, got {value!r}")
    if not 0 <= value < len(prompts):
        raise ValueError(f"phase task index {value} outside 0..{len(prompts) - 1}")
    resolved = dict(sample)
    resolved["task"] = prompts[value]
    return resolved


def ordered_phase_prompts(manifest: dict) -> list[str]:
    """Return task-index order, independent of JSON key serialization."""

    if manifest.get("phase_prompts") != PHASE_PROMPTS:
        raise ValueError("phase-conditioned manifest prompt strings drifted")
    return list(PHASE_PROMPTS.values())


def main() -> None:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts import lerobot_train
    from lerobot.utils.collate import lerobot_collate_fn

    raw_weights = os.environ.get(ACTION_LOSS_WEIGHTS_ENV, "").strip()
    if raw_weights:
        install_weighted_pi05_forward(
            PI05Policy, parse_action_loss_weights(raw_weights)
        )
    disable_local_dataset_hub_fallback(LeRobotDataset)

    manifest_path = os.environ.get("EBIM_PHASE_MANIFEST", "").strip()
    if not manifest_path:
        raise ValueError("EBIM_PHASE_MANIFEST is required")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    prompt_map = manifest.get("phase_prompts")
    if prompt_map is not None:
        prompts = ordered_phase_prompts(manifest)
        if len(prompts) != 7:
            raise ValueError("phase-conditioned training requires seven prompts")

        def phase_collate(batch):
            return lerobot_collate_fn(
                [resolve_phase_task(sample, prompts) for sample in batch]
            )

        # The organizer schema represents language through task/task_index,
        # not the newer language_events columns.  Force this V4-only process
        # onto the language-aware collate path after resolving those indices.
        LeRobotDatasetMetadata.has_language_columns = property(
            lambda self: True
        )
        lerobot_train.lerobot_collate_fn = phase_collate
    if not any(
        argument.startswith("--dataset.episodes") for argument in sys.argv[1:]
    ):
        sys.argv.append(
            "--dataset.episodes="
            + json.dumps(manifest["train_episodes"], separators=(",", ":"))
        )
    lerobot_train.EpisodeAwareSampler = PhaseBalancedSampler
    lerobot_train.main()


if __name__ == "__main__":
    main()
