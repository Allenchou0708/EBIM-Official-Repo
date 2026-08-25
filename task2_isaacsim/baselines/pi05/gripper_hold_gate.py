#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Held-out hard5 gate for Task 2 grasp, retention, and release behavior."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

from .contract import (
    ACTION_SIZE,
    POLICY_CAMERA_RENAME_MAP,
    V2_RELATIVE_ACTION_STATE_INDICES,
    checkpoint_action_state_indices,
    project_arm_action_bounds,
    validate_absolute_action_bounds,
)


HARD5_ACTIONS = 5
RIGHT_GRIPPER = 18
TASK_PROMPT = "Pick up the thermal pad and place it on the target RAM board."
DATASET_REPO_ID = "local/task2_fixpos_v2_expert_30k_phase_absolute_spine"
LANDMARK_EXPECTED_OPEN = {
    "approach_open": True,
    "grasp_close": False,
    "hold_early": False,
    "hold_mid": False,
    "hold_late": False,
    "release_open": True,
}


def landmark_frames(record: dict[str, Any]) -> dict[str, int]:
    events = record["events"]
    close = int(events["right_close"])
    move = int(events["pad_move"])
    target = int(events["target_arrival"])
    release = int(events["right_release"])
    return {
        "approach_open": max(0, close - HARD5_ACTIONS),
        "grasp_close": close,
        "hold_early": move,
        "hold_mid": move + max(0, target - move) // 2,
        "hold_late": max(target, release - 2 * HARD5_ACTIONS),
        "release_open": release,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def evaluate_gripper_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("gripper hold gate requires prediction records")
    expected_landmarks = set(LANDMARK_EXPECTED_OPEN)
    present = {str(record["landmark"]) for record in records}
    if present != expected_landmarks:
        raise ValueError(
            f"records must cover all gripper landmarks: {sorted(present)}"
        )
    landmark_success: dict[str, float] = {}
    landmark_mae: dict[str, float] = {}
    for landmark, expected_open in LANDMARK_EXPECTED_OPEN.items():
        selected = [record for record in records if record["landmark"] == landmark]
        landmark_success[landmark] = _mean(
            [
                float(
                    all(
                        (float(action[RIGHT_GRIPPER]) > 0.5) == expected_open
                        for action in record["predictions"]
                    )
                )
                for record in selected
            ]
        )
        landmark_mae[landmark] = _mean(
            [
                abs(
                    float(predicted[RIGHT_GRIPPER])
                    - float(reference[RIGHT_GRIPPER])
                )
                for record in selected
                for predicted, reference in zip(
                    record["predictions"],
                    record["reference_actions"],
                    strict=True,
                )
            ]
        )
    hold_closed = _mean(
        [
            landmark_success[name]
            for name in ("hold_early", "hold_mid", "hold_late")
        ]
    )
    safe_fraction = _mean(
        [
            float(bool(record["effective_actions_safe"]))
            for record in records
        ]
    )
    metrics = {
        "landmark_all_hard5_correct_fraction": landmark_success,
        "landmark_right_gripper_mae": landmark_mae,
        "hold_closed_fraction": hold_closed,
        "effective_actions_safe_fraction": safe_fraction,
    }
    checks = {
        "effective_action_safety": safe_fraction == 1.0,
        "approach_stays_open": landmark_success["approach_open"] >= 0.90,
        "grasp_closes": landmark_success["grasp_close"] >= 0.80,
        "grasp_is_retained": hold_closed >= 0.90,
        "release_opens": landmark_success["release_open"] >= 0.90,
    }
    return {"metrics": metrics, "checks": checks, "go": all(checks.values())}


def run_gripper_hold_gate(
    *,
    checkpoint: Path,
    dataset_root: Path,
    phase_manifest: Path,
    output: Path,
    seed: int = 1000,
    maximum_episodes: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies import make_policy, make_pre_post_processors
    except ImportError as error:
        raise RuntimeError("run the gripper gate inside the pinned PI0.5 image") from error

    from .phase_train import disable_local_dataset_hub_fallback

    disable_local_dataset_hub_fallback(LeRobotDataset)

    manifest = json.loads(phase_manifest.read_text(encoding="utf-8"))
    held_out = {int(value) for value in manifest["held_out_episodes"]}
    episode_records = [
        record
        for record in manifest["episodes"]
        if int(record["episode"]) in held_out
    ]
    if maximum_episodes is not None:
        episode_records = episode_records[:maximum_episodes]
    episodes = [int(record["episode"]) for record in episode_records]
    if not episodes:
        raise ValueError("phase manifest contains no held-out gate episodes")
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_root,
        episodes=episodes,
    )
    raw = dataset.hf_dataset.select_columns(["episode_index", "frame_index"])
    positions = {
        (int(episode), int(frame)): index
        for index, (episode, frame) in enumerate(
            zip(raw["episode_index"], raw["frame_index"], strict=True)
        )
    }

    checkpoint = checkpoint.resolve()
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.pretrained_path = str(checkpoint)
    config.pretrained_revision = None
    config.device = "cuda"
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID, root=dataset_root
    )
    policy = make_policy(
        config, ds_meta=metadata, rename_map=POLICY_CAMERA_RENAME_MAP
    )
    policy.eval()
    mapping = checkpoint_action_state_indices(policy.config)
    if mapping != V2_RELATIVE_ACTION_STATE_INDICES:
        raise ValueError("gripper gate requires the V2 absolute-spine contract")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        dataset_stats=metadata.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "stats": metadata.stats,
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
                    metadata.features["action"].get("names", [])
                ),
                "state_indices": list(mapping),
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": metadata.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
            "absolute_actions_processor": {"enabled": True},
        },
    )

    records: list[dict[str, Any]] = []
    for episode_record in episode_records:
        episode = int(episode_record["episode"])
        for landmark, frame in landmark_frames(episode_record).items():
            dataset_index = positions[(episode, frame)]
            item = dataset[dataset_index]
            reference_actions = []
            for offset in range(HARD5_ACTIONS):
                row = dataset.hf_dataset[dataset_index + offset]
                if int(row["episode_index"]) != episode:
                    raise ValueError("hard5 reference crosses an episode boundary")
                reference_actions.append([float(value) for value in row["action"]])
            observation = {
                key: value
                for key, value in item.items()
                if key == "observation.state" or key in POLICY_CAMERA_RENAME_MAP
            }
            observation["task"] = TASK_PROMPT
            policy.reset()
            fixed_seed = seed + episode * 100_000 + frame
            torch.manual_seed(fixed_seed)
            torch.cuda.manual_seed_all(fixed_seed)
            with torch.inference_mode():
                processed = preprocessor(observation)
                relative_chunk = policy.predict_action_chunk(processed)
                absolute_chunk = postprocessor(relative_chunk)
            values = absolute_chunk.detach().cpu().reshape(-1, ACTION_SIZE)
            predictions = values[:HARD5_ACTIONS].tolist()
            safe = True
            for predicted in predictions:
                effective = list(predicted)
                effective[17] = min(1.0, max(0.0, effective[17]))
                effective[18] = min(1.0, max(0.0, effective[18]))
                effective[19] = min(0.6, max(0.0, effective[19]))
                try:
                    validate_absolute_action_bounds(
                        project_arm_action_bounds(effective)
                    )
                except ValueError:
                    safe = False
            records.append(
                {
                    "episode": episode,
                    "frame": frame,
                    "landmark": landmark,
                    "expected_open": LANDMARK_EXPECTED_OPEN[landmark],
                    "reference_actions": reference_actions,
                    "predictions": predictions,
                    "effective_actions_safe": safe,
                }
            )

    evaluation = evaluate_gripper_records(records)
    report = {
        "schema_version": 1,
        "mode": "v5_gripper_hold_heldout_hard5",
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root.resolve()),
        "phase_manifest": str(phase_manifest.resolve()),
        "held_out_episodes": episodes,
        "relative_action_state_indices": list(mapping),
        "hard5_actions": HARD5_ACTIONS,
        "ros_publication": False,
        **evaluation,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    del policy, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--phase-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--maximum-episodes", type=int)
    args = parser.parse_args()
    try:
        report = run_gripper_hold_gate(
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            phase_manifest=args.phase_manifest,
            output=args.output,
            seed=args.seed,
            maximum_episodes=args.maximum_episodes,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}")
        return 2
    print(json.dumps({"go": report["go"], "checks": report["checks"]}, indent=2))
    return 0 if report["go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
