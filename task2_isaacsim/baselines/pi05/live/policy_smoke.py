#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the live observation path on one organizer train frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from task2_isaacsim.baselines.pi05.contract import PI05_CONTRACT
from task2_isaacsim.baselines.pi05.live.core import safe_action
from task2_isaacsim.baselines.pi05.live.policy import LivePi05Policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "hermanprawiro/task2_fixpos_v1",
        root=args.dataset_root,
        episodes=[args.episode],
    )
    frame = dataset[len(dataset) // 2]
    state = tuple(float(value) for value in frame["observation.state"])
    images = {}
    for key in ("head", "wrist_left", "wrist_right"):
        tensor = frame[f"observation.images.{key}"]
        array = tensor.detach().cpu().permute(1, 2, 0).numpy()
        images[key] = np.rint(array * 255.0).clip(0, 255).astype(np.uint8)
    policy = LivePi05Policy(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        dataset_repo_id="hermanprawiro/task2_fixpos_v1",
        instruction=PI05_CONTRACT.task_instruction,
        seed=1000,
    )
    chunk, latency = policy.predict_chunk(images=images, state=state)
    validated = []
    invalid = []
    for index, action in enumerate(chunk):
        try:
            validated.append(safe_action(action, spine_hold=state[28]))
        except ValueError as error:
            invalid.append({"action_index": index, "error": str(error)})
    report = {
        "schema_version": 1,
        "episode": args.episode,
        "dataset_frame": len(dataset) // 2,
        "chunk_actions": len(chunk),
        "action_dimensions": sorted({len(action) for action in chunk}),
        "finite": all(
            math.isfinite(value) for action in chunk for value in action
        ),
        "saved_postprocessor_relative_inverse": True,
        "invalid_actions": invalid,
        "invalid_action_hold": bool(invalid),
        "effective_base_zero": all(
            effective[:3] == (0.0, 0.0, 0.0) for _, effective in validated
        ),
        "spine_hold": all(
            effective[19] == state[28] for _, effective in validated
        ),
        "inference_latency_s": latency,
        "raw_actions": chunk,
        "effective_actions": [effective for _, effective in validated],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "chunk_actions",
                    "action_dimensions",
                    "finite",
                    "effective_base_zero",
                    "spine_hold",
                    "invalid_actions",
                    "inference_latency_s",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
