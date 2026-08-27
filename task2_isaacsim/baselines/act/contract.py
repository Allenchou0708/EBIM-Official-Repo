# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light ACT/Task-2 interface contract."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from task2_isaacsim.baselines.pi05.contract import (
    ACTION_NAMES,
    ACTION_SIZE,
    STATE_NAMES,
    STATE_SIZE,
)

LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
DATASET_REPO_ID = "hermanprawiro/task2_fixpos_200"
DATASET_REVISION = "46ab41f16fe836ee8ca791c7afaade44783eefe6"
SPLIT_SEED = 20260812
ACT_CAMERA_KEYS = (
    "observation.images.head",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)
EVALUATOR_CAMERA_KEY = "observation.images.eval_camera"


@dataclass(frozen=True)
class ACTTask2Contract:
    state_dim: int = STATE_SIZE
    action_dim: int = ACTION_SIZE
    chunk_size: int = 100
    n_action_steps: int = 5
    kl_weight: float = 10.0
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    batch_size: int = 8
    learning_rate: float = 1e-5
    paper_epochs: int = 2000
    checkpoint_every: int = 2300
    image_height: int = 480
    image_width: int = 640
    train_episodes: int = 180
    validation_episodes: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ACT_CONTRACT = ACTTask2Contract()


def deterministic_episode_split(
    total_episodes: int = 200,
    *,
    train_episodes: int = ACT_CONTRACT.train_episodes,
    seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """Return the fixed episode-level split shared with the Task 2 baseline."""

    if total_episodes <= 1:
        raise ValueError("total_episodes must be greater than one")
    if not 0 < train_episodes < total_episodes:
        raise ValueError("train_episodes must be between zero and total_episodes")
    episodes = list(range(total_episodes))
    random.Random(seed).shuffle(episodes)
    return sorted(episodes[:train_episodes]), sorted(episodes[train_episodes:])


def validate_action_chunk(values: Sequence[Sequence[float]]) -> list[list[float]]:
    """Validate the exact absolute 20-D action boundary consumed by ROS."""

    chunk: list[list[float]] = []
    for row_index, row in enumerate(values):
        if len(row) != ACTION_SIZE:
            raise ValueError(
                f"ACT action {row_index} has {len(row)} values; expected {ACTION_SIZE}"
            )
        vector = [float(value) for value in row]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"ACT action {row_index} contains a non-finite value")
        chunk.append(vector)
    if not chunk:
        raise ValueError("ACT returned an empty action chunk")
    return chunk


def load_split_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"split manifest is not a JSON object: {path}")
    train = payload.get("train_episodes")
    validation = payload.get("validation_episodes")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("split manifest requires train_episodes and validation_episodes")
    if len(train) != 180 or len(validation) != 20:
        raise ValueError(
            "Task 2 ACT split must contain exactly 180 train and 20 "
            "validation episodes"
        )
    if set(train) & set(validation):
        raise ValueError("training and validation episodes overlap")
    return payload


def contract_manifest() -> dict[str, Any]:
    return {
        "contract": ACT_CONTRACT.to_dict(),
        "action_names": list(ACTION_NAMES),
        "state_names": list(STATE_NAMES),
        "camera_keys": list(ACT_CAMERA_KEYS),
        "excluded_evaluator_camera": EVALUATOR_CAMERA_KEY,
        "action_semantics": "absolute_task2_ros_command",
        "loss": "l1 + kl_weight * kld",
    }
