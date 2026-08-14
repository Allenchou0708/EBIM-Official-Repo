#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Held-out, prompt-contrastive offline gate for Task 2 PI0.5 V4."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .contract import (
    ACTION_SIZE,
    FR3_JOINT_LIMITS,
    POLICY_CAMERA_RENAME_MAP,
    V2_RELATIVE_ACTION_STATE_INDICES,
    checkpoint_action_state_indices,
    project_arm_action_bounds,
    validate_absolute_action_bounds,
)
from .phase_conditioned_dataset import PHASE_PROMPTS


HARD5_ACTIONS = 5
RIGHT_GRIPPER = 18
RIGHT_JOINT4 = 13
PHASE_ACTION_INDICES = {
    "startup_rise": tuple(range(3, 17)) + (19,),
    "approach": tuple(range(10, 19)),
    "orient_pregrasp": tuple(range(10, 19)),
    "grasp_acquisition": tuple(range(10, 19)),
    "lift_transfer": tuple(range(10, 19)),
    "lower_place": tuple(range(10, 19)),
    "release_retreat": tuple(range(10, 19)),
}


@dataclass(frozen=True)
class V4GateThresholds:
    effective_predictions_safe_fraction: float = 1.0
    startup_arm_mae_max_rad: float = 0.12
    startup_direction_cosine_median_min: float = 0.40
    approach_open_fraction_min: float = 0.95
    orient_open_fraction_min: float = 0.95
    orient_right_arm_mae_max_rad: float = 0.12
    orient_to_preclose_cosine_median_min: float = 0.50
    orient_joint4_saturation_fraction_max: float = 0.02
    grasp_close_recall_min: float = 0.80
    release_open_recall_min: float = 0.90
    correct_prompt_best_fraction_min: float = 0.45
    correct_prompt_margin_median_min: float = 0.0
    prompt_delta_l2_median_min: float = 0.02


def phase_landmark_frames(record: dict[str, Any]) -> dict[str, int]:
    events = record["events"]
    return {
        "startup_rise": max(0, int(events["spine_high"]) - HARD5_ACTIONS),
        "approach": int(events["spine_high"]),
        "orient_pregrasp": int(record["orientation_entry_frame"]),
        "grasp_acquisition": int(events["right_close"]),
        "lift_transfer": int(events["pad_move"]),
        "lower_place": int(events["target_arrival"]),
        "release_retreat": int(events["right_release"]),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _mae(
    prediction: list[list[float]],
    reference: list[list[float]],
    indices: tuple[int, ...],
) -> float:
    values = [
        abs(predicted[index] - expected[index])
        for predicted, expected in zip(prediction, reference, strict=True)
        for index in indices
    ]
    return _mean(values)


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def _direction_cosine(
    *, current: list[float], predicted: list[float], target: list[float]
) -> float:
    return _cosine(
        [value - origin for value, origin in zip(predicted, current, strict=True)],
        [value - origin for value, origin in zip(target, current, strict=True)],
    )


def evaluate_phase_records(
    records: list[dict[str, Any]],
    thresholds: V4GateThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate already-generated hard5 prompt predictions."""

    cfg = thresholds or V4GateThresholds()
    if not records:
        raise ValueError("phase offline gate requires prediction records")
    prompts = tuple(PHASE_PROMPTS)
    expected_phases = set(prompts)
    present = {str(record["phase"]) for record in records}
    if present != expected_phases:
        raise ValueError(
            f"phase records must cover all seven phases: {sorted(present)}"
        )

    all_predictions = [
        action
        for record in records
        for chunk in record["predictions"].values()
        for action in chunk
    ]
    effective_safe = 0
    raw_arm_valid = 0
    raw_gripper_valid = 0
    raw_spine_valid = 0
    for action in all_predictions:
        arm_valid = True
        for offset in (3, 10):
            for joint, (lower, upper) in enumerate(FR3_JOINT_LIMITS):
                value = float(action[offset + joint])
                arm_valid = arm_valid and lower <= value <= upper
        raw_arm_valid += int(arm_valid)
        raw_gripper_valid += int(
            0.0 <= float(action[17]) <= 1.0
            and 0.0 <= float(action[18]) <= 1.0
        )
        raw_spine_valid += int(0.0 <= float(action[19]) <= 0.6)
        try:
            effective = list(action)
            effective[17] = min(1.0, max(0.0, effective[17]))
            effective[18] = min(1.0, max(0.0, effective[18]))
            effective[19] = min(0.6, max(0.0, effective[19]))
            effective = list(project_arm_action_bounds(effective))
            validate_absolute_action_bounds(effective)
        except ValueError:
            continue
        effective_safe += 1
    prediction_count = len(all_predictions)

    phase_mae: dict[str, float] = {}
    correct_best: list[float] = []
    prompt_margins: list[float] = []
    prompt_deltas: list[float] = []
    for phase in prompts:
        phase_records = [record for record in records if record["phase"] == phase]
        indices = PHASE_ACTION_INDICES[phase]
        errors = []
        for record in phase_records:
            reference = record["reference_actions"]
            correct = record["predictions"][phase]
            correct_error = _mae(correct, reference, indices)
            errors.append(correct_error)
            wrong_errors = [
                _mae(record["predictions"][other], reference, indices)
                for other in prompts
                if other != phase
            ]
            best_wrong = min(wrong_errors)
            correct_best.append(float(correct_error < best_wrong))
            prompt_margins.append(best_wrong - correct_error)
            correct_flat = [value for row in correct for value in row]
            for other in prompts:
                if other == phase:
                    continue
                other_flat = [
                    value for row in record["predictions"][other] for value in row
                ]
                prompt_deltas.append(
                    math.sqrt(
                        sum(
                            (a - b) ** 2
                            for a, b in zip(correct_flat, other_flat, strict=True)
                        )
                    )
                )
        phase_mae[phase] = _mean(errors)

    startup = [record for record in records if record["phase"] == "startup_rise"]
    startup_cosines = [
        _direction_cosine(
            current=record["state"][14:28],
            predicted=record["predictions"]["startup_rise"][-1][3:17],
            target=record["reference_actions"][-1][3:17],
        )
        for record in startup
    ]
    orient = [
        record for record in records if record["phase"] == "orient_pregrasp"
    ]
    orient_cosines = [
        _direction_cosine(
            current=record["state"][21:28],
            predicted=record["predictions"]["orient_pregrasp"][-1][10:17],
            target=record["preclose_right_joints"],
        )
        for record in orient
    ]
    lower4, upper4 = FR3_JOINT_LIMITS[3]
    joint4_saturated = sum(
        1
        for record in orient
        for action in record["predictions"]["orient_pregrasp"]
        if action[RIGHT_JOINT4] <= lower4 + 0.02
        or action[RIGHT_JOINT4] >= upper4 - 0.02
    )
    orient_actions = len(orient) * HARD5_ACTIONS

    def open_record_fraction(phase: str) -> float:
        selected = [record for record in records if record["phase"] == phase]
        return _mean(
            [
                float(
                    all(
                        action[RIGHT_GRIPPER] > 0.5
                        for action in record["predictions"][phase]
                    )
                )
                for record in selected
            ]
        )

    grasp_records = [
        record for record in records if record["phase"] == "grasp_acquisition"
    ]
    release_records = [
        record for record in records if record["phase"] == "release_retreat"
    ]
    metrics = {
        "effective_predictions_safe_fraction": effective_safe / prediction_count,
        "raw_arm_in_bounds_fraction": raw_arm_valid / prediction_count,
        "raw_gripper_in_bounds_fraction": raw_gripper_valid / prediction_count,
        "raw_spine_in_bounds_fraction": raw_spine_valid / prediction_count,
        "phase_action_mae": phase_mae,
        "startup_arm_mae_rad": _mean(
            [
                _mae(
                    record["predictions"]["startup_rise"],
                    record["reference_actions"],
                    tuple(range(3, 17)),
                )
                for record in startup
            ]
        ),
        "startup_direction_cosine_median": median(startup_cosines),
        "approach_open_fraction": open_record_fraction("approach"),
        "orient_open_fraction": open_record_fraction("orient_pregrasp"),
        "orient_right_arm_mae_rad": _mean(
            [
                _mae(
                    record["predictions"]["orient_pregrasp"],
                    record["reference_actions"],
                    tuple(range(10, 17)),
                )
                for record in orient
            ]
        ),
        "orient_to_preclose_cosine_median": median(orient_cosines),
        "orient_joint4_saturation_fraction": joint4_saturated / orient_actions,
        "grasp_close_recall": _mean(
            [
                float(
                    any(
                        action[RIGHT_GRIPPER] < 0.5
                        for action in record["predictions"]["grasp_acquisition"]
                    )
                )
                for record in grasp_records
            ]
        ),
        "release_open_recall": _mean(
            [
                float(
                    any(
                        action[RIGHT_GRIPPER] > 0.5
                        for action in record["predictions"]["release_retreat"]
                    )
                )
                for record in release_records
            ]
        ),
        "correct_prompt_best_fraction": _mean(correct_best),
        "correct_prompt_margin_median": median(prompt_margins),
        "prompt_delta_l2_median": median(prompt_deltas),
    }
    checks = {
        "effective_action_safety": metrics[
            "effective_predictions_safe_fraction"
        ]
        >= cfg.effective_predictions_safe_fraction,
        "startup_wrist_compensation_proxy": (
            metrics["startup_arm_mae_rad"] <= cfg.startup_arm_mae_max_rad
            and metrics["startup_direction_cosine_median"]
            >= cfg.startup_direction_cosine_median_min
        ),
        "approach_stays_open": metrics["approach_open_fraction"]
        >= cfg.approach_open_fraction_min,
        "orient_pregrasp": (
            metrics["orient_open_fraction"] >= cfg.orient_open_fraction_min
            and metrics["orient_right_arm_mae_rad"]
            <= cfg.orient_right_arm_mae_max_rad
            and metrics["orient_to_preclose_cosine_median"]
            >= cfg.orient_to_preclose_cosine_median_min
            and metrics["orient_joint4_saturation_fraction"]
            <= cfg.orient_joint4_saturation_fraction_max
        ),
        "grasp_closes": metrics["grasp_close_recall"]
        >= cfg.grasp_close_recall_min,
        "release_opens": metrics["release_open_recall"]
        >= cfg.release_open_recall_min,
        "prompt_discriminability": (
            metrics["correct_prompt_best_fraction"]
            >= cfg.correct_prompt_best_fraction_min
            and metrics["correct_prompt_margin_median"]
            >= cfg.correct_prompt_margin_median_min
            and metrics["prompt_delta_l2_median"]
            >= cfg.prompt_delta_l2_median_min
        ),
    }
    return {
        "thresholds": asdict(cfg),
        "metrics": metrics,
        "checks": checks,
        "go": all(checks.values()),
    }


def run_phase_offline_gate(
    *,
    checkpoint: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    pose_audit: Path,
    output: Path,
    seed: int = 1000,
    maximum_episodes: int | None = None,
) -> dict[str, Any]:
    """Generate prompt-contrastive predictions without ROS publication."""

    try:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies import make_policy, make_pre_post_processors
    except ImportError as error:
        raise RuntimeError("run the V4 gate inside the pinned PI0.5 image") from error

    audit = json.loads(pose_audit.read_text(encoding="utf-8"))
    audit_records = [
        record for record in audit["episodes"] if record["split"] == "held_out"
    ]
    if maximum_episodes is not None:
        audit_records = audit_records[:maximum_episodes]
    episodes = [int(record["episode"]) for record in audit_records]
    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, episodes=episodes)
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
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    policy = make_policy(config, ds_meta=metadata, rename_map=POLICY_CAMERA_RENAME_MAP)
    policy.eval()
    mapping = checkpoint_action_state_indices(policy.config)
    if mapping != V2_RELATIVE_ACTION_STATE_INDICES:
        raise ValueError("V4 gate requires the V2 absolute-spine checkpoint contract")
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
                "action_names": list(metadata.features["action"].get("names", [])),
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
    for audit_record in audit_records:
        episode = int(audit_record["episode"])
        preclose_joints = audit_record["landmarks"]["preclose"]["right_joints"]
        for phase, frame in phase_landmark_frames(audit_record).items():
            dataset_index = positions[(episode, frame)]
            item = dataset[dataset_index]
            state = item["observation.state"].detach().cpu().tolist()
            reference_actions = []
            for offset in range(HARD5_ACTIONS):
                row = dataset.hf_dataset[dataset_index + offset]
                if int(row["episode_index"]) != episode:
                    raise ValueError("hard5 reference crosses an episode boundary")
                reference_actions.append([float(value) for value in row["action"]])
            base_observation = {
                key: value
                for key, value in item.items()
                if key == "observation.state"
                or key in POLICY_CAMERA_RENAME_MAP
            }
            predictions: dict[str, list[list[float]]] = {}
            fixed_seed = seed + episode * 100_000 + frame
            for prompt_name, prompt in PHASE_PROMPTS.items():
                policy.reset()
                torch.manual_seed(fixed_seed)
                torch.cuda.manual_seed_all(fixed_seed)
                observation = {**base_observation, "task": prompt}
                with torch.inference_mode():
                    processed = preprocessor(observation)
                    relative_chunk = policy.predict_action_chunk(processed)
                    absolute_chunk = postprocessor(relative_chunk)
                values = absolute_chunk.detach().cpu().reshape(-1, ACTION_SIZE)
                if len(values) < HARD5_ACTIONS:
                    raise ValueError("checkpoint returned fewer than five actions")
                predictions[prompt_name] = values[:HARD5_ACTIONS].tolist()
            records.append(
                {
                    "episode": episode,
                    "frame": frame,
                    "phase": phase,
                    "state": state,
                    "preclose_right_joints": preclose_joints,
                    "reference_actions": reference_actions,
                    "predictions": predictions,
                }
            )

    evaluation = evaluate_phase_records(records)
    report = {
        "schema_version": 1,
        "mode": "v4_prompt_contrastive_heldout_hard5",
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root.resolve()),
        "pose_audit": str(pose_audit.resolve()),
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
    parser.add_argument("--dataset-repo-id", default="local/task2_pi05_v4")
    parser.add_argument("--pose-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--maximum-episodes", type=int)
    args = parser.parse_args()
    try:
        report = run_phase_offline_gate(
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            dataset_repo_id=args.dataset_repo_id,
            pose_audit=args.pose_audit,
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
