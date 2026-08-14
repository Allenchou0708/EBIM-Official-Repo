#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run LeRobot PI0.5 training with Task 2's phase-balanced sampler."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .phase_balance import PhaseBalancedSampler
from .phase_conditioned_dataset import PHASE_PROMPTS


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
    from lerobot.scripts import lerobot_train
    from lerobot.utils.collate import lerobot_collate_fn

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
