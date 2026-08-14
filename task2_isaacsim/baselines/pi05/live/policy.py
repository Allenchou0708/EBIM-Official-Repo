# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed PI0.5 action-chunk inference for live observations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from task2_isaacsim.baselines.pi05.contract import (
    ACTION_NAMES,
    POLICY_CAMERA_RENAME_MAP,
    checkpoint_action_state_indices,
)


class LivePi05Policy:
    """Load checkpoint processors once and preserve their relative inverse."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        dataset_root: Path,
        dataset_repo_id: str,
        instruction: str,
        seed: int,
    ):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies import make_policy, make_pre_post_processors

        self.torch = torch
        self.instruction = instruction
        self.seed = seed
        self.decision_index = 0
        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        config.pretrained_path = str(checkpoint)
        config.pretrained_revision = None
        config.device = "cuda"
        metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
        self.policy = make_policy(
            config,
            ds_meta=metadata,
            rename_map=POLICY_CAMERA_RENAME_MAP,
        )
        self.policy.eval()
        self.action_state_indices = checkpoint_action_state_indices(
            self.policy.config
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(checkpoint),
            dataset_stats=metadata.stats,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "normalizer_processor": {
                    "stats": metadata.stats,
                    "features": {
                        **self.policy.config.input_features,
                        **self.policy.config.output_features,
                    },
                    "norm_map": self.policy.config.normalization_mapping,
                },
                "rename_observations_processor": {
                    "rename_map": POLICY_CAMERA_RENAME_MAP
                },
                "relative_actions_processor": {
                    "enabled": True,
                    "exclude_joints": [],
                    "action_names": list(ACTION_NAMES),
                    "state_indices": list(self.action_state_indices),
                },
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": metadata.stats,
                    "features": self.policy.config.output_features,
                    "norm_map": self.policy.config.normalization_mapping,
                },
                "absolute_actions_processor": {"enabled": True},
            },
        )

    def reset(self) -> None:
        self.policy.reset()
        self.decision_index = 0

    def predict_chunk(
        self, *, images: dict[str, Any], state: tuple[float, ...]
    ) -> tuple[list[list[float]], float]:
        """Return one complete postprocessed absolute action chunk."""

        torch = self.torch
        fixed_seed = self.seed + self.decision_index
        self.decision_index += 1
        torch.manual_seed(fixed_seed)
        torch.cuda.manual_seed_all(fixed_seed)
        observation: dict[str, Any] = {
            "observation.state": torch.tensor(state, dtype=torch.float32),
            "task": self.instruction,
        }
        for key, image in images.items():
            observation[f"observation.images.{key}"] = (
                torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            )
        started = time.monotonic()
        with torch.inference_mode():
            processed = self.preprocessor(observation)
            relative_chunk = self.policy.predict_action_chunk(processed)
            absolute_chunk = self.postprocessor(relative_chunk)
        torch.cuda.synchronize()
        latency = time.monotonic() - started
        values = absolute_chunk.detach().cpu().reshape(-1, 20).tolist()
        return values, latency
