#!/usr/bin/env python3
"""Held-out lifecycle evaluation for an 8-D Task 2 PI0.5 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LANDMARKS = (
    ("pre_close", "open"),
    ("grasp_acquisition", "closed"),
    ("lift_transfer", "closed"),
    ("lower_place", "closed"),
    ("release_retreat", "open"),
)

# Frozen before evaluating either checkpoint.  These gates are intentionally
# stricter than an always-open/always-closed policy and must all pass before a
# GUI simulator actuation run is allowed.
OFFLINE_GO_THRESHOLDS = {
    "gripper_phase_accuracy_min": 0.60,
    "full_lifecycle_accuracy_min": 0.20,
    "each_landmark_phase_accuracy_min": 0.40,
    "mean_joint_direction_cosine_min_exclusive": 0.0,
    "action_bound_violation_fraction_max": 0.05,
}

# V2 evaluates the commands the live right-only runner would actually publish:
# FR3 position projection, gripper clipping, and a 50%-of-rated-speed slew
# envelope at 30 Hz.  Projection burden is compared with the held-out expert
# target under the same rule instead of treating the narrower train min/max as
# a physical robot limit.
RUNTIME_SAFETY_THRESHOLDS = {
    "gripper_phase_accuracy_min": 0.60,
    "full_lifecycle_accuracy_min": 0.20,
    "each_landmark_phase_accuracy_min": 0.40,
    "effective_joint_direction_cosine_min_exclusive": 0.0,
    "effective_physical_violation_fraction_max": 0.0,
    "effective_slew_violation_fraction_max": 0.0,
    "projected_joint_fraction_excess_over_target_max": 0.10,
    "mean_projection_correction_excess_rad_max": 0.02,
    "max_projection_correction_excess_rad_max": 0.10,
}
RUNTIME_ACTION_RATE_HZ = 30.0
RUNTIME_EXECUTION_HORIZON = 5


def gripper_phase(value: float) -> str:
    if value <= 0.25:
        return "closed"
    if value >= 0.90:
        return "open"
    return "transition"


def run(
    checkpoint: Path,
    heldout_root: Path,
    train_gate_path: Path,
    *,
    seeds: list[int],
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate

    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.feature_utils import dataset_to_policy_features

    from task2_isaacsim.baselines.pi05.contract import FR3_JOINT_LIMITS
    from task2_isaacsim.baselines.pi05.live.core import (
        FR3_JOINT_VELOCITY_LIMITS_RAD_S,
        RIGHT_ARM_SLEW_VELOCITY_FRACTION,
        project_fr3_joint_step,
    )

    manifest = json.loads(
        (heldout_root / "meta" / "ebim_manipulation_only.json").read_text()
    )
    gate = json.loads(train_gate_path.read_text(encoding="utf-8"))
    if not gate.get("success") or gate["counts"]["source_overlap"]:
        raise ValueError("train-only data gate is absent or failed")
    meta = LeRobotDatasetMetadata(manifest["repo_id"], root=heldout_root)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError("checkpoint is not PI0.5")
    if tuple(config.output_features["action"].shape) != (8,):
        raise ValueError("checkpoint is not the right-only 8-D contract")
    # The derived videos are encoded smaller for storage, while PI0.5's
    # preprocessor maps every visual stream to its fixed 224x224 model input.
    # Reconcile only the declared raw image shapes so the same checkpoint can
    # be compared fairly without copying or rewriting its weights/config.
    dataset_features = dataset_to_policy_features(meta.features)
    config.input_features = {
        key: value for key, value in dataset_features.items() if key != "action"
    }
    config.pretrained_path = str(checkpoint)
    config.device = "cuda"
    config.dtype = "bfloat16"
    dataset = LeRobotDataset(
        manifest["repo_id"],
        root=heldout_root,
        delta_timestamps=resolve_delta_timestamps(config, meta),
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    policy = make_policy(config, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
    )
    lower = torch.as_tensor(
        gate["train_only_normalization"]["absolute_action_stats_before_patch"]["min"]
    )
    upper = torch.as_tensor(
        gate["train_only_normalization"]["absolute_action_stats_before_patch"]["max"]
    )
    physical_lower = torch.as_tensor(
        [bounds[0] for bounds in FR3_JOINT_LIMITS] + [0.0]
    )
    physical_upper = torch.as_tensor(
        [bounds[1] for bounds in FR3_JOINT_LIMITS] + [1.0]
    )

    def runtime_projection(chunk: Any, initial_state: Any) -> tuple[Any, dict[str, float]]:
        horizon = min(RUNTIME_EXECUTION_HORIZON, int(chunk.shape[0]))
        raw_chunk = chunk[:horizon].clone()
        effective = raw_chunk.clone()
        previous = tuple(float(value) for value in initial_state[:7])
        corrections = []
        projected_joints = 0
        for index in range(horizon):
            requested = tuple(float(value) for value in raw_chunk[index, :7])
            projected = project_fr3_joint_step(
                previous,
                requested,
                action_rate_hz=RUNTIME_ACTION_RATE_HZ,
            )
            for joint, (before, after) in enumerate(
                zip(requested, projected, strict=True)
            ):
                correction = abs(after - before)
                corrections.append(correction)
                projected_joints += int(correction > 1e-9)
                effective[index, joint] = after
            effective[index, 7] = effective[index, 7].clamp(0.0, 1.0)
            previous = projected

        effective_physical_outside = (
            (effective < physical_lower) | (effective > physical_upper)
        )
        effective_slew_violations = 0
        previous = tuple(float(value) for value in initial_state[:7])
        for row in effective:
            for before, after, limit in zip(
                previous,
                row[:7],
                FR3_JOINT_VELOCITY_LIMITS_RAD_S,
                strict=True,
            ):
                maximum_step = (
                    limit
                    * RIGHT_ARM_SLEW_VELOCITY_FRACTION
                    / RUNTIME_ACTION_RATE_HZ
                )
                effective_slew_violations += int(
                    abs(float(after) - before) > maximum_step + 1e-6
                )
            previous = tuple(float(value) for value in row[:7])
        return effective, {
            "projected_joint_fraction": projected_joints / (horizon * 7),
            "mean_projection_correction_rad": sum(corrections) / len(corrections),
            "max_projection_correction_rad": max(corrections, default=0.0),
            "effective_physical_violation_fraction": float(
                effective_physical_outside.float().mean()
            ),
            "effective_slew_violation_fraction": (
                effective_slew_violations / (horizon * 7)
            ),
        }

    records = []
    policy.eval()
    with torch.no_grad():
        for seed in seeds:
            for episode in manifest["episode_records"]:
                base = int(episode["derived_global_start"])
                for landmark, expected in LANDMARKS:
                    offset = int(episode["derived_events"][landmark])
                    batch = default_collate([dataset[base + offset]])
                    target = batch["action"].clone()
                    state = batch["observation.state"].clone()
                    valid = ~batch["action_is_pad"].clone()
                    for key in tuple(batch):
                        if key.startswith("observation.images.") and batch[key].dtype == torch.uint8:
                            batch[key] = batch[key].to(torch.float32) / 255.0
                    torch.manual_seed(seed + int(episode["source_episode_index"]) * 101 + offset)
                    torch.cuda.manual_seed_all(seed + int(episode["source_episode_index"]) * 101 + offset)
                    processed = preprocessor(batch)
                    predicted = policy.predict_action_chunk(processed)
                    predicted = postprocessor(predicted)
                    if isinstance(predicted, dict):
                        predicted = predicted["action"]
                    predicted = predicted.detach().cpu()
                    target = target.detach().cpu()
                    state = state.detach().cpu()
                    valid = valid.detach().cpu()
                    predicted_valid = predicted[0, valid[0]]
                    target_valid = target[0, valid[0]]
                    runtime_predicted, runtime_predicted_projection = runtime_projection(
                        predicted_valid,
                        state[0],
                    )
                    runtime_target = target_valid[:RUNTIME_EXECUTION_HORIZON]
                    runtime_target_projected, runtime_target_projection = runtime_projection(
                        runtime_target,
                        state[0],
                    )
                    first_gripper = float(predicted[0, 0, 7])
                    target_gripper = float(target[0, 0, 7])
                    pred_delta = (predicted_valid[:, :7] - state[0, :7]).reshape(-1)
                    target_delta = (target_valid[:, :7] - state[0, :7]).reshape(-1)
                    denominator = float(pred_delta.norm() * target_delta.norm())
                    cosine = (
                        float(torch.dot(pred_delta, target_delta) / denominator)
                        if denominator > 1e-9
                        else None
                    )
                    runtime_predicted_delta = (
                        runtime_predicted[:, :7] - state[0, :7]
                    ).reshape(-1)
                    runtime_target_delta = (
                        runtime_target[:, :7] - state[0, :7]
                    ).reshape(-1)
                    runtime_denominator = float(
                        runtime_predicted_delta.norm() * runtime_target_delta.norm()
                    )
                    runtime_cosine = (
                        float(
                            torch.dot(runtime_predicted_delta, runtime_target_delta)
                            / runtime_denominator
                        )
                        if runtime_denominator > 1e-9
                        else None
                    )
                    outside = (predicted < lower) | (predicted > upper)
                    records.append(
                        {
                            "seed": seed,
                            "source_episode_index": int(episode["source_episode_index"]),
                            "landmark": landmark,
                            "expected_phase": expected,
                            "target_phase": gripper_phase(target_gripper),
                            "predicted_phase": gripper_phase(first_gripper),
                            "predicted_gripper_first": first_gripper,
                            "target_gripper_first": target_gripper,
                            "joint_mae": float(
                                (predicted_valid[:, :7] - target_valid[:, :7]).abs().mean()
                            ),
                            "gripper_mae": float(
                                (predicted_valid[:, 7] - target_valid[:, 7]).abs().mean()
                            ),
                            "joint_direction_cosine": cosine,
                            "action_bound_violation_fraction": float(outside.float().mean()),
                            "runtime_effective_joint_mae": float(
                                (
                                    runtime_predicted[:, :7]
                                    - runtime_target[:, :7]
                                )
                                .abs()
                                .mean()
                            ),
                            "runtime_effective_joint_direction_cosine": runtime_cosine,
                            "runtime_model_projection": runtime_predicted_projection,
                            "runtime_target_projection": runtime_target_projection,
                            "runtime_target_projection_joint_mae": float(
                                (
                                    runtime_target_projected[:, :7]
                                    - runtime_target[:, :7]
                                )
                                .abs()
                                .mean()
                            ),
                            "valid_target_actions": int(valid.sum()),
                        }
                    )

    exact = [row["predicted_phase"] == row["expected_phase"] for row in records]
    lifecycle = []
    for seed in seeds:
        for source_episode in manifest["source_episode_indices"]:
            selected = [
                row
                for row in records
                if row["seed"] == seed and row["source_episode_index"] == source_episode
            ]
            lifecycle.append(all(row["predicted_phase"] == row["expected_phase"] for row in selected))
    cosines = [row["joint_direction_cosine"] for row in records if row["joint_direction_cosine"] is not None]
    result = {
        "success": True,
        "scope": "offline held-out action lifecycle; not simulator task success",
        "checkpoint": str(checkpoint.resolve()),
        "heldout_root": str(heldout_root.resolve()),
        "seeds": seeds,
        "episodes": len(manifest["source_episode_indices"]),
        "predictions": len(records),
        "gripper_phase_accuracy": sum(exact) / len(exact),
        "full_lifecycle_accuracy": sum(lifecycle) / len(lifecycle),
        "mean_joint_mae": sum(row["joint_mae"] for row in records) / len(records),
        "mean_gripper_mae": sum(row["gripper_mae"] for row in records) / len(records),
        "mean_joint_direction_cosine": sum(cosines) / len(cosines),
        "action_bound_violation_fraction": sum(
            row["action_bound_violation_fraction"] for row in records
        )
        / len(records),
        "by_landmark": {},
        "records": records,
        "task_success": False,
        "generalization_validated": False,
    }
    runtime_cosines = [
        row["runtime_effective_joint_direction_cosine"]
        for row in records
        if row["runtime_effective_joint_direction_cosine"] is not None
    ]
    result["runtime_safety_projection"] = {
        "action_rate_hz": RUNTIME_ACTION_RATE_HZ,
        "execution_horizon": RUNTIME_EXECUTION_HORIZON,
        "velocity_fraction": RIGHT_ARM_SLEW_VELOCITY_FRACTION,
        "mean_effective_joint_mae": sum(
            row["runtime_effective_joint_mae"] for row in records
        )
        / len(records),
        "mean_effective_joint_direction_cosine": sum(runtime_cosines)
        / len(runtime_cosines),
        "model": {
            key: sum(row["runtime_model_projection"][key] for row in records)
            / len(records)
            for key in (
                "projected_joint_fraction",
                "mean_projection_correction_rad",
                "effective_physical_violation_fraction",
                "effective_slew_violation_fraction",
            )
        },
        "target": {
            key: sum(row["runtime_target_projection"][key] for row in records)
            / len(records)
            for key in (
                "projected_joint_fraction",
                "mean_projection_correction_rad",
                "effective_physical_violation_fraction",
                "effective_slew_violation_fraction",
            )
        },
        "model_max_projection_correction_rad": max(
            row["runtime_model_projection"]["max_projection_correction_rad"]
            for row in records
        ),
        "target_max_projection_correction_rad": max(
            row["runtime_target_projection"]["max_projection_correction_rad"]
            for row in records
        ),
    }
    for landmark, _ in LANDMARKS:
        selected = [row for row in records if row["landmark"] == landmark]
        result["by_landmark"][landmark] = {
            "phase_accuracy": sum(
                row["predicted_phase"] == row["expected_phase"] for row in selected
            )
            / len(selected),
            "mean_joint_mae": sum(row["joint_mae"] for row in selected) / len(selected),
            "mean_gripper_mae": sum(row["gripper_mae"] for row in selected) / len(selected),
        }
    checks = {
        "gripper_phase_accuracy": result["gripper_phase_accuracy"]
        >= OFFLINE_GO_THRESHOLDS["gripper_phase_accuracy_min"],
        "full_lifecycle_accuracy": result["full_lifecycle_accuracy"]
        >= OFFLINE_GO_THRESHOLDS["full_lifecycle_accuracy_min"],
        "each_landmark_phase_accuracy": all(
            values["phase_accuracy"]
            >= OFFLINE_GO_THRESHOLDS["each_landmark_phase_accuracy_min"]
            for values in result["by_landmark"].values()
        ),
        "mean_joint_direction_cosine": result["mean_joint_direction_cosine"]
        > OFFLINE_GO_THRESHOLDS["mean_joint_direction_cosine_min_exclusive"],
        "action_bound_violation_fraction": result["action_bound_violation_fraction"]
        <= OFFLINE_GO_THRESHOLDS["action_bound_violation_fraction_max"],
    }
    result["offline_gui_gate"] = {
        "thresholds_frozen_before_evaluation": OFFLINE_GO_THRESHOLDS,
        "checks": checks,
        "pass": all(checks.values()),
    }
    runtime = result["runtime_safety_projection"]
    runtime_checks = {
        "gripper_phase_accuracy": result["gripper_phase_accuracy"]
        >= RUNTIME_SAFETY_THRESHOLDS["gripper_phase_accuracy_min"],
        "full_lifecycle_accuracy": result["full_lifecycle_accuracy"]
        >= RUNTIME_SAFETY_THRESHOLDS["full_lifecycle_accuracy_min"],
        "each_landmark_phase_accuracy": all(
            values["phase_accuracy"]
            >= RUNTIME_SAFETY_THRESHOLDS["each_landmark_phase_accuracy_min"]
            for values in result["by_landmark"].values()
        ),
        "effective_joint_direction_cosine": runtime[
            "mean_effective_joint_direction_cosine"
        ]
        > RUNTIME_SAFETY_THRESHOLDS[
            "effective_joint_direction_cosine_min_exclusive"
        ],
        "effective_physical_violation_fraction": runtime["model"][
            "effective_physical_violation_fraction"
        ]
        <= RUNTIME_SAFETY_THRESHOLDS[
            "effective_physical_violation_fraction_max"
        ],
        "effective_slew_violation_fraction": runtime["model"][
            "effective_slew_violation_fraction"
        ]
        <= RUNTIME_SAFETY_THRESHOLDS["effective_slew_violation_fraction_max"],
        "projected_joint_fraction_vs_target": (
            runtime["model"]["projected_joint_fraction"]
            - runtime["target"]["projected_joint_fraction"]
            <= RUNTIME_SAFETY_THRESHOLDS[
                "projected_joint_fraction_excess_over_target_max"
            ]
        ),
        "mean_projection_correction_vs_target": (
            runtime["model"]["mean_projection_correction_rad"]
            - runtime["target"]["mean_projection_correction_rad"]
            <= RUNTIME_SAFETY_THRESHOLDS[
                "mean_projection_correction_excess_rad_max"
            ]
        ),
        "max_projection_correction_vs_target": (
            runtime["model_max_projection_correction_rad"]
            - runtime["target_max_projection_correction_rad"]
            <= RUNTIME_SAFETY_THRESHOLDS[
                "max_projection_correction_excess_rad_max"
            ]
        ),
    }
    result["runtime_gui_gate_v2"] = {
        "thresholds_frozen_before_evaluation": RUNTIME_SAFETY_THRESHOLDS,
        "checks": runtime_checks,
        "pass": all(runtime_checks.values()),
    }
    if not all(math.isfinite(float(result[key])) for key in (
        "gripper_phase_accuracy",
        "full_lifecycle_accuracy",
        "mean_joint_mae",
        "mean_gripper_mae",
        "mean_joint_direction_cosine",
        "action_bound_violation_fraction",
    )):
        raise RuntimeError("offline evaluation produced a non-finite aggregate")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="20260902,20260903")
    args = parser.parse_args()
    result = run(
        args.checkpoint.resolve(),
        args.heldout_root.resolve(),
        args.train_gate.resolve(),
        seeds=[int(value) for value in args.seeds.split(",")],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "predictions",
        "gripper_phase_accuracy",
        "full_lifecycle_accuracy",
        "mean_joint_mae",
        "mean_gripper_mae",
        "mean_joint_direction_cosine",
        "action_bound_violation_fraction",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
