#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the live observation path on one organizer train frame."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from task2_isaacsim.baselines.pi05.contract import PI05_CONTRACT
from task2_isaacsim.baselines.pi05.live.core import (
    align_action_chunk,
    safe_action,
)
from task2_isaacsim.baselines.pi05.live.policy import LivePi05Policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--dataset-repo-id", default="hermanprawiro/task2_fixpos_200"
    )
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        args.dataset_repo_id,
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
        dataset_repo_id=args.dataset_repo_id,
        instruction=PI05_CONTRACT.task_instruction,
        seed=1000,
    )
    capture_at = time.monotonic()
    chunk, latency = policy.predict_chunk(images=images, state=state)
    ready_at = time.monotonic()
    validated = []
    invalid = []
    for index, action in enumerate(chunk):
        try:
            validated.append(safe_action(action))
        except ValueError as error:
            invalid.append({"action_index": index, "error": str(error)})
    discarded, aligned = align_action_chunk(
        [effective for _, effective in validated],
        capture_at=capture_at,
        ready_at=ready_at,
        action_rate_hz=30.0,
    )
    aligned_target_times = [target_at for _, target_at, _ in aligned]
    report = {
        "schema_version": 2,
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
        "spine_policy_controlled": all(
            effective[19] == min(0.6, max(0.0, raw[19]))
            for raw, effective in validated
        ),
        "spine_targets_in_range": all(
            0.0 <= effective[19] <= 0.6 for _, effective in validated
        ),
        "observation_spine_m": state[28],
        "spine_not_held_at_observation": any(
            abs(effective[19] - state[28]) > 1e-6 for _, effective in validated
        ),
        "raw_spine_targets_m": [raw[19] for raw, _ in validated],
        "effective_spine_targets_m": [
            effective[19] for _, effective in validated
        ],
        "inference_latency_s": latency,
        "capture_to_ready_latency_s": ready_at - capture_at,
        "discarded_prefix_actions": discarded,
        "aligned_actions": len(aligned),
        "first_aligned_chunk_index": aligned[0][2] if aligned else None,
        "last_aligned_chunk_index": aligned[-1][2] if aligned else None,
        "aligned_target_times_monotonic": all(
            right > left
            for left, right in zip(
                aligned_target_times,
                aligned_target_times[1:],
                strict=False,
            )
        ),
        "immediate_queue_underflow": not aligned,
        "raw_actions": chunk,
        "effective_actions": [effective for _, effective in validated],
        "aligned_effective_actions": [
            {
                "chunk_index": chunk_index,
                "target_offset_s": target_at - capture_at,
                "action": effective,
            }
            for effective, target_at, chunk_index in aligned
        ],
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
                    "spine_policy_controlled",
                    "spine_targets_in_range",
                    "spine_not_held_at_observation",
                    "capture_to_ready_latency_s",
                    "discarded_prefix_actions",
                    "aligned_actions",
                    "first_aligned_chunk_index",
                    "last_aligned_chunk_index",
                    "aligned_target_times_monotonic",
                    "immediate_queue_underflow",
                    "invalid_actions",
                    "inference_latency_s",
                )
            },
            sort_keys=True,
        )
    )
    passed = (
        not invalid
        and report["finite"]
        and report["effective_base_zero"]
        and report["spine_policy_controlled"]
        and report["spine_targets_in_range"]
        and report["spine_not_held_at_observation"]
        and report["discarded_prefix_actions"] > 0
        and report["aligned_target_times_monotonic"]
        and not report["immediate_queue_underflow"]
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
