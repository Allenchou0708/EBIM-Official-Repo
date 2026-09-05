"""Bounded expert-only PI0.5 baseline on the immutable Phase II real split."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from task2_real.full_dataset_gate import (
    BASE_REVISION,
    HELDOUT_REPO_ID,
    RENAME_MAP,
    TRAIN_REPO_ID,
)


LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
SOURCE_REVISION = "495ebb7b56fb9e2f3952398a63d86f08cacb9531"
SEED = 20260901
EVAL_NOISE_SEED = 20261901
LANDMARK_NOISE_SEED = 20262901
MAX_STEPS = 200
CHECKPOINT_STEP = 50
MAX_GPU_SECONDS = 2 * 60 * 60


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except ImportError:  # pragma: no cover
        np = None
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return float(value.item()) if value.numel() == 1 else value.tolist()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _prepare_raw_batch(batch: dict[str, Any], torch: Any) -> dict[str, Any]:
    for key in tuple(batch):
        if key.startswith("observation.images."):
            if batch[key].dtype != torch.uint8:
                raise ValueError(f"expected uint8 camera before PI0.5 processor: {key}")
            batch[key] = batch[key].to(dtype=torch.float32) / 255.0
    return batch


def _processor_stats(train_meta: Any, gate: dict[str, Any], np: Any) -> dict[str, Any]:
    stats = copy.deepcopy(train_meta.stats)
    relative = gate["train_relative_action_stats"]
    stats["action"] = {
        key: np.asarray(relative[key])
        for key in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
    }
    return stats


def _parameter_evidence(policy: Any) -> dict[str, Any]:
    named = list(policy.named_parameters())
    projection_tokens = (
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    )
    categories = {
        "vlm_backbone": [item for item in named if ".paligemma." in item[0]],
        "action_expert": [item for item in named if ".gemma_expert." in item[0]],
        "action_projections": [
            item for item in named if any(token in item[0] for token in projection_tokens)
        ],
    }
    if any(not values for values in categories.values()):
        raise RuntimeError("PI0.5 parameter categories could not be identified")
    evidence = {}
    for category, values in categories.items():
        evidence[category] = {
            "parameters": sum(parameter.numel() for _, parameter in values),
            "trainable_parameters": sum(
                parameter.numel() for _, parameter in values if parameter.requires_grad
            ),
            "trainable_tensors": sum(parameter.requires_grad for _, parameter in values),
        }
    if evidence["vlm_backbone"]["trainable_parameters"] != 0:
        raise RuntimeError("expert-only baseline left VLM parameters trainable")
    if evidence["action_expert"]["trainable_parameters"] == 0:
        raise RuntimeError("expert-only baseline froze the action expert")
    if evidence["action_projections"]["trainable_parameters"] == 0:
        raise RuntimeError("expert-only baseline froze the action projections")
    allowed_ids = {
        id(parameter)
        for category in ("action_expert", "action_projections")
        for _, parameter in categories[category]
        if parameter.requires_grad
    }
    unexpected = [name for name, parameter in named if parameter.requires_grad and id(parameter) not in allowed_ids]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters outside expert/projections: {unexpected[:5]}")
    return evidence


def _evaluate_loss(
    *,
    policy: Any,
    preprocessor: Any,
    dataset: Any,
    eval_plan: list[dict[str, Any]],
    torch: Any,
    default_collate: Any,
) -> dict[str, Any]:
    policy.eval()
    torch.manual_seed(EVAL_NOISE_SEED)
    torch.cuda.manual_seed_all(EVAL_NOISE_SEED)
    losses: list[float] = []
    loss_per_dim: list[list[float]] = []
    started = time.monotonic()
    with torch.no_grad():
        for point in eval_plan:
            batch = default_collate([dataset[int(point["dataset_index"])]])
            if bool(batch["action_is_pad"].any()):
                raise RuntimeError("fixed held-out evaluation unexpectedly contains action padding")
            batch = _prepare_raw_batch(batch, torch)
            processed = preprocessor(batch)
            loss, details = policy(processed)
            value = float(loss.detach().item())
            if not math.isfinite(value):
                raise RuntimeError("held-out diffusion loss is non-finite")
            if not all(math.isfinite(float(item)) for item in details["loss_per_dim"]):
                raise RuntimeError("held-out per-dimension diffusion loss is non-finite")
            losses.append(value)
            loss_per_dim.append([float(item) for item in details["loss_per_dim"]])
    torch.cuda.synchronize()
    dimensions = len(loss_per_dim[0])
    return {
        "loss": sum(losses) / len(losses),
        "loss_per_dim": [
            sum(values[dimension] for values in loss_per_dim) / len(loss_per_dim)
            for dimension in range(dimensions)
        ],
        "samples": len(losses),
        "episodes": len({point["source_episode_index"] for point in eval_plan}),
        "seconds": time.monotonic() - started,
        "fixed_noise_seed": EVAL_NOISE_SEED,
    }


def _phase(value: float) -> str:
    if value <= 0.25:
        return "closed"
    if value >= 0.90:
        return "open"
    return "transition"


def _evaluate_landmarks(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    landmarks: list[dict[str, Any]],
    action_bounds: tuple[Any, Any],
    torch: Any,
    default_collate: Any,
) -> list[dict[str, Any]]:
    policy.eval()
    lower, upper = action_bounds
    results = []
    with torch.no_grad():
        for landmark_index, landmark in enumerate(landmarks):
            batch = default_collate([dataset[int(landmark["dataset_index"])]])
            target = batch["action"].clone()
            valid = ~batch["action_is_pad"].clone()
            batch = _prepare_raw_batch(batch, torch)
            torch.manual_seed(LANDMARK_NOISE_SEED + landmark_index)
            torch.cuda.manual_seed_all(LANDMARK_NOISE_SEED + landmark_index)
            processed = preprocessor(batch)
            predicted_relative = policy.predict_action_chunk(processed)
            predicted = postprocessor(predicted_relative)
            if isinstance(predicted, dict):
                predicted = predicted["action"]
            predicted = predicted.detach().cpu()
            target = target.detach().cpu()
            valid = valid.detach().cpu()
            valid_predictions = predicted[0, valid[0]]
            valid_targets = target[0, valid[0]]
            joint_mae = float((valid_predictions[:, :7] - valid_targets[:, :7]).abs().mean())
            gripper_mae = float((valid_predictions[:, 7] - valid_targets[:, 7]).abs().mean())
            smoothness = float((predicted[:, 1:] - predicted[:, :-1]).abs().mean())
            outside = (predicted < lower) | (predicted > upper)
            first_predicted_gripper = float(predicted[0, 0, 7])
            first_target_gripper = float(target[0, 0, 7])
            results.append(
                {
                    **landmark,
                    "target_phase_label": landmark["phase"],
                    "target_gripper_phase": _phase(first_target_gripper),
                    "predicted_gripper_phase": _phase(first_predicted_gripper),
                    "predicted_gripper_first": first_predicted_gripper,
                    "target_gripper_first": first_target_gripper,
                    "joint_action_mae": joint_mae,
                    "gripper_action_mae": gripper_mae,
                    "predicted_chunk_mean_abs_delta": smoothness,
                    "predicted_action_bound_violation_fraction": float(outside.float().mean()),
                    "valid_target_actions": int(valid.sum()),
                    "fixed_noise_seed": LANDMARK_NOISE_SEED + landmark_index,
                }
            )
    return results


def _save_trainable_delta(
    policy: Any, path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    if not tensors:
        raise RuntimeError("cannot save an empty expert-only delta")
    expected_keys = set(tensors)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".temporary.safetensors")
    save_file(tensors, temporary, metadata={key: str(value) for key, value in metadata.items()})
    os.replace(temporary, path)
    del tensors
    with safe_open(path, framework="pt", device="cpu") as stream:
        saved_keys = set(stream.keys())
        saved_metadata = stream.metadata()
    if saved_keys != expected_keys:
        raise RuntimeError("saved expert delta keys differ from trainable parameter keys")
    expected_metadata = {key: str(value) for key, value in metadata.items()}
    if saved_metadata != expected_metadata:
        raise RuntimeError("saved expert delta metadata differs from requested metadata")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "header_validation": "safe_open_pass",
        "key_count": len(saved_keys),
        "file_size_bytes": path.stat().st_size,
        "metadata": saved_metadata,
        "fresh_model_reload_validated": False,
        "deployable": False,
    }


def run(
    train_root: Path,
    heldout_root: Path,
    base_snapshot: Path,
    gate_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate

    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    if output_dir.exists():
        raise ValueError(f"baseline output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    run_started = time.monotonic()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("success") or gate["counts"] != {
        "train_episodes": 119,
        "train_frames": 71420,
        "heldout_episodes": 30,
        "heldout_frames": 15024,
        "source_id_overlap": [],
    }:
        raise ValueError("CPU data gate is absent or no longer exact")
    if gate.get("source_revision") != SOURCE_REVISION:
        raise ValueError("CPU gate source revision changed")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
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
    train_dataset = LeRobotDataset(
        TRAIN_REPO_ID,
        root=train_root,
        delta_timestamps=resolve_delta_timestamps(config, train_meta),
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    heldout_dataset = LeRobotDataset(
        HELDOUT_REPO_ID,
        root=heldout_root,
        delta_timestamps=resolve_delta_timestamps(config, heldout_meta),
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    policy = make_policy(config, ds_meta=train_meta, rename_map=RENAME_MAP)
    parameter_evidence = _parameter_evidence(policy)
    stats = _processor_stats(train_meta, gate, np)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=None,
        dataset_stats=stats,
    )
    rename_step = preprocessor.steps[0]
    if rename_step.__class__.__name__ != "RenameObservationsProcessorStep":
        raise RuntimeError("canonical PI0.5 preprocessor no longer starts with rename")
    rename_step.rename_map = RENAME_MAP
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    eval_plan = gate["heldout_eval"]["plan"]
    landmarks = gate["landmarks"]
    action_lower = torch.as_tensor(train_meta.stats["action"]["min"], dtype=torch.float32)
    action_upper = torch.as_tensor(train_meta.stats["action"]["max"], dtype=torch.float32)

    result: dict[str, Any] = {
        "success": False,
        "scope": "bounded expert-only offline baseline; not real task success",
        "base": {
            "repo_id": "lerobot/pi05_base",
            "revision": BASE_REVISION,
            "snapshot": str(base_snapshot),
        },
        "lerobot_commit": LEROBOT_COMMIT,
        "source_revision": SOURCE_REVISION,
        "configuration": {
            "seed": SEED,
            "batch_size": 1,
            "max_steps": MAX_STEPS,
            "max_gpu_seconds": MAX_GPU_SECONDS,
            "dtype": "bfloat16",
            "train_expert_only": True,
            "freeze_vision_encoder": True,
            "gradient_checkpointing": True,
            "gradient_clip_norm": float(config.optimizer_grad_clip_norm),
            "use_relative_actions": True,
            "relative_action_state_indices": config.relative_action_state_indices,
            "heldout_samples": len(eval_plan),
            "heldout_episodes": 30,
            "optimizer": optimizer.__class__.__name__,
            "optimizer_learning_rates": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
        },
        "parameter_evidence": parameter_evidence,
        "heldout": {},
        "landmarks": {},
        "train_curve": [],
        "best_delta": None,
        "stop_reason": None,
    }
    result_path = output_dir / "result.json"
    train_log_path = output_dir / "train_metrics.jsonl"

    step0 = _evaluate_loss(
        policy=policy,
        preprocessor=preprocessor,
        dataset=heldout_dataset,
        eval_plan=eval_plan,
        torch=torch,
        default_collate=default_collate,
    )
    result["heldout"]["step0"] = step0
    result["landmarks"]["step0"] = _evaluate_landmarks(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=heldout_dataset,
        landmarks=landmarks,
        action_bounds=(action_lower, action_upper),
        torch=torch,
        default_collate=default_collate,
    )
    result_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n")

    iterator = iter(train_loader)
    best_loss = step0["loss"]
    best_step: int | None = None
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, MAX_STEPS + 1):
        if time.monotonic() - run_started >= MAX_GPU_SECONDS:
            result["stop_reason"] = "two_gpu_hour_cap"
            break
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _prepare_raw_batch(batch, torch)
        policy.train()
        step_started = time.monotonic()
        processed = preprocessor(batch)
        loss, details = policy(processed)
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite train loss at step {step}")
        if not all(math.isfinite(float(item)) for item in details["loss_per_dim"]):
            raise RuntimeError(f"non-finite train per-dimension loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), config.optimizer_grad_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        metric = {
            "step": step,
            "loss": float(loss.detach().item()),
            "loss_per_dim": details["loss_per_dim"],
            "gradient_norm_before_clip": float(grad_norm.detach().item()),
            "gradient_clipped": bool(float(grad_norm) > config.optimizer_grad_clip_norm),
            "step_seconds": time.monotonic() - step_started,
            "samples_per_second": 1.0 / (time.monotonic() - step_started),
            "elapsed_gpu_seconds": time.monotonic() - run_started,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        if not math.isfinite(metric["gradient_norm_before_clip"]):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        result["train_curve"].append(metric)
        with train_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_safe(metric), sort_keys=True) + "\n")

        should_evaluate = step == CHECKPOINT_STEP or step == MAX_STEPS
        if should_evaluate:
            evaluation = _evaluate_loss(
                policy=policy,
                preprocessor=preprocessor,
                dataset=heldout_dataset,
                eval_plan=eval_plan,
                torch=torch,
                default_collate=default_collate,
            )
            result["heldout"][f"step{step}"] = evaluation
            result["landmarks"][f"step{step}"] = _evaluate_landmarks(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=heldout_dataset,
                landmarks=landmarks,
                action_bounds=(action_lower, action_upper),
                torch=torch,
                default_collate=default_collate,
            )
            if step == CHECKPOINT_STEP and evaluation["loss"] > 1.10 * step0["loss"]:
                result["stop_reason"] = "step50_heldout_worsened_more_than_10_percent"
                result_path.write_text(
                    json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n"
                )
                break
            if evaluation["loss"] < best_loss:
                best_loss = evaluation["loss"]
                best_step = step
                delta_path = output_dir / "best_expert_delta.safetensors"
                delta_validation = _save_trainable_delta(
                    policy,
                    delta_path,
                    {
                        "base_revision": BASE_REVISION,
                        "source_revision": SOURCE_REVISION,
                        "step": step,
                        "heldout_loss": evaluation["loss"],
                    },
                )
                result["best_delta"] = {
                    "path": str(delta_path),
                    "step": step,
                    "heldout_loss": evaluation["loss"],
                    "improvement_vs_step0": step0["loss"] - evaluation["loss"],
                    "validation": delta_validation,
                }
            result_path.write_text(
                json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n"
            )
    else:
        result["stop_reason"] = "max_200_steps"

    completed_steps = len(result["train_curve"])
    if completed_steps not in (CHECKPOINT_STEP, MAX_STEPS) and f"step{completed_steps}" not in result["heldout"]:
        final_evaluation = _evaluate_loss(
            policy=policy,
            preprocessor=preprocessor,
            dataset=heldout_dataset,
            eval_plan=eval_plan,
            torch=torch,
            default_collate=default_collate,
        )
        result["heldout"][f"step{completed_steps}"] = final_evaluation
        result["landmarks"][f"step{completed_steps}"] = _evaluate_landmarks(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=heldout_dataset,
            landmarks=landmarks,
            action_bounds=(action_lower, action_upper),
            torch=torch,
            default_collate=default_collate,
        )
        if final_evaluation["loss"] < best_loss:
            best_loss = final_evaluation["loss"]
            best_step = completed_steps
            delta_path = output_dir / "best_expert_delta.safetensors"
            delta_validation = _save_trainable_delta(
                policy,
                delta_path,
                {
                    "base_revision": BASE_REVISION,
                    "source_revision": SOURCE_REVISION,
                    "step": completed_steps,
                    "heldout_loss": final_evaluation["loss"],
                },
            )
            result["best_delta"] = {
                "path": str(delta_path),
                "step": completed_steps,
                "heldout_loss": final_evaluation["loss"],
                "improvement_vs_step0": step0["loss"] - final_evaluation["loss"],
                "validation": delta_validation,
            }
    result["completed_steps"] = completed_steps
    result["best_step"] = best_step
    result["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "total_run_seconds": time.monotonic() - run_started,
    }
    result["success"] = bool(
        completed_steps >= CHECKPOINT_STEP
        and result["stop_reason"] != "step50_heldout_worsened_more_than_10_percent"
    )
    result["non_claims"] = [
        "offline diffusion loss and action metrics do not establish task success",
        "no simulator or real robot command was published",
        "no LoRA or full VLM fine-tuning was performed",
    ]
    result_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_root,
        args.heldout_root,
        args.base_snapshot,
        args.gate,
        args.output_dir,
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
