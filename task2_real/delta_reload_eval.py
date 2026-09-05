"""Fresh-base, inference-only validation of an expert-only PI0.5 delta."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from task2_real.expert_baseline import (
    LEROBOT_COMMIT,
    SOURCE_REVISION,
    _json_safe,
    _parameter_evidence,
    _phase,
    _prepare_raw_batch,
    _processor_stats,
)
from task2_real.full_dataset_gate import (
    BASE_REVISION,
    HELDOUT_REPO_ID,
    RENAME_MAP,
    TRAIN_REPO_ID,
)


NOISE_SEEDS = tuple(range(20263901, 20263909))
MAX_GPU_SECONDS = 10 * 60


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        confusion[row["target_gripper_phase"]][row["predicted_gripper_phase"]] += 1
        by_phase[row["phase"]].append(row)

    def mean(key: str, values: list[dict[str, Any]]) -> float:
        return sum(float(row[key]) for row in values) / len(values)

    return {
        "predictions": len(rows),
        "noise_seeds": list(NOISE_SEEDS),
        "gripper_phase_accuracy": sum(
            row["target_gripper_phase"] == row["predicted_gripper_phase"] for row in rows
        )
        / len(rows),
        "gripper_phase_correct": sum(
            row["target_gripper_phase"] == row["predicted_gripper_phase"] for row in rows
        ),
        "gripper_phase_confusion": {
            target: dict(predicted) for target, predicted in confusion.items()
        },
        "per_landmark_phase": {
            phase: {
                "predictions": len(values),
                "accuracy": sum(
                    row["target_gripper_phase"] == row["predicted_gripper_phase"]
                    for row in values
                )
                / len(values),
                "joint_action_mae": mean("joint_action_mae", values),
                "gripper_action_mae": mean("gripper_action_mae", values),
                "predicted_chunk_mean_abs_delta": mean(
                    "predicted_chunk_mean_abs_delta", values
                ),
                "empirical_train_action_envelope_violation_fraction": mean(
                    "empirical_train_action_envelope_violation_fraction", values
                ),
            }
            for phase, values in sorted(by_phase.items())
        },
        "joint_action_mae": mean("joint_action_mae", rows),
        "gripper_action_mae": mean("gripper_action_mae", rows),
        "predicted_chunk_mean_abs_delta": mean("predicted_chunk_mean_abs_delta", rows),
        "empirical_train_action_envelope_violation_fraction": mean(
            "empirical_train_action_envelope_violation_fraction", rows
        ),
        "envelope_metric_warning": (
            "train min/max is an empirical data envelope, not a physical safety bound"
        ),
    }


def _evaluate(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    cached_batches: list[tuple[dict[str, Any], dict[str, Any]]],
    action_bounds: tuple[Any, Any],
    run_started: float,
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy.eval()
    lower, upper = action_bounds
    rows = []
    started = time.monotonic()
    with torch.no_grad():
        for landmark, cached_batch in cached_batches:
            for seed in NOISE_SEEDS:
                if time.monotonic() - run_started >= MAX_GPU_SECONDS:
                    raise TimeoutError("ten GPU minute inference cap reached")
                batch = copy.deepcopy(cached_batch)
                target = batch["action"].clone()
                valid = ~batch["action_is_pad"].clone()
                batch = _prepare_raw_batch(batch, torch)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                processed = preprocessor(batch)
                predicted_relative = policy.predict_action_chunk(processed)
                predicted = postprocessor(predicted_relative)
                if isinstance(predicted, dict):
                    predicted = predicted["action"]
                predicted = predicted.detach().cpu()
                target = target.detach().cpu()
                valid = valid.detach().cpu()
                if not bool(torch.isfinite(predicted).all()):
                    raise RuntimeError("non-finite action prediction")
                valid_predictions = predicted[0, valid[0]]
                valid_targets = target[0, valid[0]]
                outside = (predicted < lower) | (predicted > upper)
                predicted_gripper = float(predicted[0, 0, 7])
                target_gripper = float(target[0, 0, 7])
                rows.append(
                    {
                        **landmark,
                        "noise_seed": seed,
                        "target_gripper_first": target_gripper,
                        "predicted_gripper_first": predicted_gripper,
                        "target_gripper_phase": _phase(target_gripper),
                        "predicted_gripper_phase": _phase(predicted_gripper),
                        "joint_action_mae": float(
                            (valid_predictions[:, :7] - valid_targets[:, :7]).abs().mean()
                        ),
                        "gripper_action_mae": float(
                            (valid_predictions[:, 7] - valid_targets[:, 7]).abs().mean()
                        ),
                        "predicted_chunk_mean_abs_delta": float(
                            (predicted[:, 1:] - predicted[:, :-1]).abs().mean()
                        ),
                        "empirical_train_action_envelope_violation_fraction": float(
                            outside.float().mean()
                        ),
                        "valid_target_actions": int(valid.sum()),
                    }
                )
    torch.cuda.synchronize()
    return rows, {
        **_summarize(rows),
        "seconds": time.monotonic() - started,
    }


def _strict_load_delta(policy: Any, delta_path: Path, torch: Any) -> dict[str, Any]:
    from safetensors import safe_open

    intended = {
        name: parameter for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    if not intended:
        raise RuntimeError("fresh model exposes no intended trainable parameters")
    with safe_open(delta_path, framework="pt", device="cpu") as stream:
        saved_keys = set(stream.keys())
        metadata = stream.metadata()
        if saved_keys != set(intended):
            missing = sorted(set(intended) - saved_keys)
            unexpected = sorted(saved_keys - set(intended))
            raise RuntimeError(
                f"delta key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        witness = None
        for name in sorted(saved_keys):
            tensor = stream.get_tensor(name)
            parameter = intended[name]
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise RuntimeError(f"delta shape mismatch for {name}")
            if tensor.dtype != parameter.dtype:
                raise RuntimeError(
                    f"delta dtype mismatch for {name}: {tensor.dtype} != {parameter.dtype}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"non-finite delta tensor: {name}")
            if witness is None:
                difference = (
                    parameter.detach().cpu().to(torch.float32) - tensor.to(torch.float32)
                ).abs()
                if bool((difference > 0).any()):
                    witness = {
                        "parameter": name,
                        "base_to_delta_max_abs_change": float(difference.max()),
                        "base_to_delta_mean_abs_change": float(difference.mean()),
                    }
    if witness is None:
        raise RuntimeError("delta caused no witnessed parameter change")

    with torch.no_grad(), safe_open(delta_path, framework="pt", device="cpu") as stream:
        for name in sorted(saved_keys):
            intended[name].copy_(stream.get_tensor(name).to(device=intended[name].device))
    with safe_open(delta_path, framework="pt", device="cpu") as stream:
        expected = stream.get_tensor(witness["parameter"])
    applied = intended[witness["parameter"]].detach().cpu()
    if not torch.equal(applied, expected):
        raise RuntimeError("witnessed parameter does not exactly equal delta after load")
    return {
        "success": True,
        "strict_key_set_match": True,
        "shape_compatible": True,
        "dtype_compatible": True,
        "all_tensors_finite": True,
        "key_count": len(saved_keys),
        "file_size_bytes": delta_path.stat().st_size,
        "metadata": metadata,
        "witnessed_parameter_change": witness,
        "witness_exact_after_load": True,
        "fresh_model_delta_reload_validated": True,
        "runtime_packaging_load_path_validated": False,
        "deployable": False,
    }


def run(
    train_root: Path,
    heldout_root: Path,
    base_snapshot: Path,
    gate_path: Path,
    delta_path: Path,
    output_path: Path,
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

    if output_path.exists():
        raise ValueError(f"follow-up output already exists: {output_path}")
    run_started = time.monotonic()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("success") or len(gate["landmarks"]) != 9:
        raise ValueError("exact CPU gate with nine landmarks is required")

    config = PreTrainedConfig.from_pretrained(base_snapshot, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError("base checkpoint did not decode as PI05Config")
    config.pretrained_path = str(base_snapshot)
    config.pretrained_revision = BASE_REVISION
    config.device = "cuda"
    config.dtype = "bfloat16"
    config.train_expert_only = True
    config.freeze_vision_encoder = True
    config.gradient_checkpointing = True
    config.compile_model = False
    config.use_relative_actions = True
    config.relative_exclude_joints = ["right_gripper_target_percent"]
    config.relative_action_state_indices = [0, 1, 2, 3, 4, 5, 6, None]
    config.n_action_steps = 5

    train_meta = LeRobotDatasetMetadata(TRAIN_REPO_ID, root=train_root)
    heldout_meta = LeRobotDatasetMetadata(HELDOUT_REPO_ID, root=heldout_root)
    heldout_dataset = LeRobotDataset(
        HELDOUT_REPO_ID,
        root=heldout_root,
        delta_timestamps=resolve_delta_timestamps(config, heldout_meta),
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    policy = make_policy(config, ds_meta=train_meta, rename_map=RENAME_MAP)
    parameter_evidence = _parameter_evidence(policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=None,
        dataset_stats=_processor_stats(train_meta, gate, np),
    )
    rename_step = preprocessor.steps[0]
    if rename_step.__class__.__name__ != "RenameObservationsProcessorStep":
        raise RuntimeError("canonical PI0.5 preprocessor no longer starts with rename")
    rename_step.rename_map = RENAME_MAP
    cached_batches = [
        (
            landmark,
            default_collate([heldout_dataset[int(landmark["dataset_index"])]]),
        )
        for landmark in gate["landmarks"]
    ]
    action_bounds = (
        torch.as_tensor(train_meta.stats["action"]["min"], dtype=torch.float32),
        torch.as_tensor(train_meta.stats["action"]["max"], dtype=torch.float32),
    )
    result: dict[str, Any] = {
        "success": False,
        "scope": "inference-only stochastic landmark comparison; not task success",
        "base_revision": BASE_REVISION,
        "source_revision": SOURCE_REVISION,
        "lerobot_commit": LEROBOT_COMMIT,
        "model_loads": 1,
        "noise_seeds": list(NOISE_SEEDS),
        "landmarks": gate["landmarks"],
        "parameter_evidence": parameter_evidence,
        "before_delta": None,
        "delta_load_gate": None,
        "after_delta": None,
        "hard_cap_seconds": MAX_GPU_SECONDS,
        "non_claims": [
            "empirical train min/max is not a physical safety bound",
            "offline action metrics do not establish task success",
            "checkpoint is not deployable until a runtime packaging/load path is validated",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before_rows, before_summary = _evaluate(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        cached_batches=cached_batches,
        action_bounds=action_bounds,
        run_started=run_started,
        torch=torch,
    )
    result["before_delta"] = {"summary": before_summary, "predictions": before_rows}
    output_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n")
    result["delta_load_gate"] = _strict_load_delta(policy, delta_path, torch)
    after_rows, after_summary = _evaluate(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        cached_batches=cached_batches,
        action_bounds=action_bounds,
        run_started=run_started,
        torch=torch,
    )
    result["after_delta"] = {"summary": after_summary, "predictions": after_rows}
    result["total_seconds"] = time.monotonic() - run_started
    if result["total_seconds"] > MAX_GPU_SECONDS:
        raise TimeoutError("ten GPU minute inference cap exceeded")
    for section in ("before_delta", "after_delta"):
        for value in result[section]["summary"].values():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"non-finite summary metric in {section}")
    result["success"] = True
    output_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_root,
        args.heldout_root,
        args.base_snapshot,
        args.gate,
        args.delta,
        args.output,
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
