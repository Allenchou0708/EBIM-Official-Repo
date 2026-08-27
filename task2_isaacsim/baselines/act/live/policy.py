# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed ACT inference with the PI0.5-compatible 20-D output."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from task2_isaacsim.baselines.act.contract import ACT_CAMERA_KEYS, validate_action_chunk


class LiveACTPolicy:
    """Expose ACT through the exact interface used by the safe Task 2 runner."""

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
        import torch.nn.functional as functional
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies import make_policy, make_pre_post_processors

        del instruction, seed
        self.torch = torch
        self.functional = functional
        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        config.pretrained_path = str(checkpoint)
        config.pretrained_revision = None
        config.device = "cuda"
        metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
        self.policy = make_policy(config, ds_meta=metadata)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
        )
        first_camera = next(iter(config.image_features.values()))
        self.image_size = (int(first_camera.shape[-2]), int(first_camera.shape[-1]))
        self.chunk_size = int(config.chunk_size)
        # The runner records this field for PI0.5 relative-action provenance.
        # ACT is trained directly in the absolute command space.
        self.action_state_indices = (None,) * 20

    def reset(self) -> None:
        self.policy.reset()

    def predict_chunk(
        self, *, images: dict[str, Any], state: tuple[float, ...]
    ) -> tuple[list[list[float]], float]:
        torch = self.torch
        observation: dict[str, Any] = {
            "observation.state": torch.tensor(state, dtype=torch.float32),
        }
        for full_key in ACT_CAMERA_KEYS:
            short_key = full_key.removeprefix("observation.images.")
            image = torch.from_numpy(images[short_key].copy()).permute(2, 0, 1).float() / 255.0
            image = self.functional.interpolate(
                image.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            observation[full_key] = image
        started = time.monotonic()
        with torch.inference_mode():
            processed = self.preprocessor(observation)
            normalized = self.policy.predict_action_chunk(processed)
            absolute = self.postprocessor(normalized)
        torch.cuda.synchronize()
        latency = time.monotonic() - started
        values = absolute.detach().cpu().reshape(-1, 20).tolist()
        return validate_action_chunk(values), latency
