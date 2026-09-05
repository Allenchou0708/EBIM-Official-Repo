"""Run one expert-only PI0.5 optimizer step on the real episode-0 adapter.

This is an interface/gradient smoke test, not a training entrypoint.  It never
writes a checkpoint and refuses any dataset other than the immutable one-episode
right-only derived view produced by ``smoke_dataset.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BASE_REPO_ID = "lerobot/pi05_base"
BASE_REVISION = "338b5c22c12dbdd0d2ab19046802de2eb7696a6b"
LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
DATASET_REPO_ID = "local/phase2_real_right_smoke"
SOURCE_DATASET_REVISION = "495ebb7b56fb9e2f3952398a63d86f08cacb9531"
RAW_STATE_INDICES = list(range(21, 29))
RAW_ACTION_INDICES = list(range(8, 16))
CAMERA_KEYS = {
    "observation.images.head",
    "observation.images.wrist_right",
}
RENAME_MAP = {
    "observation.images.head": "observation.images.base_0_rgb",
    "observation.images.wrist_right": "observation.images.right_wrist_0_rgb",
}


def _gradient_summary(named_parameters: list[tuple[str, Any]]) -> dict[str, Any]:
    tensors_with_grad = 0
    finite_tensors = 0
    nonzero_tensors = 0
    maximum_absolute_gradient = 0.0
    parameter_count = 0
    trainable_parameter_count = 0
    for _, parameter in named_parameters:
        parameter_count += parameter.numel()
        if parameter.requires_grad:
            trainable_parameter_count += parameter.numel()
        if parameter.grad is None:
            continue
        tensors_with_grad += 1
        finite = bool(parameter.grad.isfinite().all().item())
        finite_tensors += int(finite)
        maximum = float(parameter.grad.detach().abs().max().item())
        maximum_absolute_gradient = max(maximum_absolute_gradient, maximum)
        nonzero_tensors += int(maximum > 0.0)
    return {
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "tensors_with_grad": tensors_with_grad,
        "finite_tensors_with_grad": finite_tensors,
        "nonzero_gradient_tensors": nonzero_tensors,
        "maximum_absolute_gradient": maximum_absolute_gradient,
    }


def _json_safe(value: Any) -> Any:
    """Keep diagnostic output small and serializable after the optimizer step."""
    try:
        import torch
    except ImportError:  # pragma: no cover - this script requires torch
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() == 1:
            return float(detached.item())
        return {
            "shape": list(detached.shape),
            "finite": bool(detached.isfinite().all().item()),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def run(dataset_root: Path, base_snapshot: Path, output: Path) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the authorized PI0.5 one-step smoke")
    provenance = json.loads((dataset_root / "meta" / "ebim_source.json").read_text())
    if provenance != {
        "source_episode_index": 0,
        "source_policy_action_raw_indices": RAW_ACTION_INDICES,
        "source_policy_state_raw_indices": RAW_STATE_INDICES,
        "source_repo_id": "ebim-benchmark/ebim_task2_realrobotdata",
        "source_revision": SOURCE_DATASET_REVISION,
        "source_timestamp_sampling": (
            "nearest frame independently per camera using each parquet timestamp"
        ),
    }:
        raise ValueError("derived dataset provenance is not the pinned right-only episode-0 view")
    if not (base_snapshot / "model.safetensors").is_file():
        raise FileNotFoundError("pinned PI0.5 model.safetensors is absent")

    torch.manual_seed(20260831)
    torch.cuda.manual_seed_all(20260831)
    # Importing PI05Config above registers the ``pi05`` choice used by the
    # generic checkpoint decoder.  Calling the subclass decoder directly would
    # incorrectly reject the checkpoint's required ``type`` discriminator.
    config = PreTrainedConfig.from_pretrained(base_snapshot, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError(f"pinned checkpoint decoded as unexpected config: {type(config).__name__}")
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

    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, root=dataset_root)
    feature_keys = set(metadata.features)
    image_feature_keys = {key for key in feature_keys if key.startswith("observation.images.")}
    if image_feature_keys != CAMERA_KEYS:
        raise ValueError(f"unexpected derived image schema: {sorted(image_feature_keys)}")
    if tuple(metadata.features["observation.state"]["shape"]) != (8,):
        raise ValueError("derived state schema is not 8-D")
    if tuple(metadata.features["action"]["shape"]) != (8,):
        raise ValueError("derived action schema is not 8-D")
    forbidden = ("left", "spine", "base", "raw_action_16")
    schema_text = json.dumps(metadata.features, sort_keys=True).lower()
    if any(token in schema_text for token in forbidden):
        raise ValueError("derived trainable schema contains forbidden ownership names")

    delta_timestamps = resolve_delta_timestamps(config, metadata)
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_root,
        episodes=[0],
        delta_timestamps=delta_timestamps,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    if len(dataset) != 1281:
        raise ValueError(f"derived dataset length changed: {len(dataset)}")
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    batch_image_keys = {key for key in batch if key.startswith("observation.images.")}
    if batch_image_keys != CAMERA_KEYS:
        raise ValueError(f"real batch image keys changed: {sorted(batch_image_keys)}")
    if tuple(batch["observation.state"].shape) != (1, 8):
        raise ValueError("real batch state is not (1, 8)")
    if tuple(batch["action"].shape) != (1, 50, 8):
        raise ValueError(f"real action chunk is not (1, 50, 8): {tuple(batch['action'].shape)}")
    if not bool(batch["observation.state"].isfinite().all()):
        raise ValueError("real batch state is non-finite")
    if not bool(batch["action"].isfinite().all()):
        raise ValueError("real batch action is non-finite")
    raw_batch_shapes = {key: list(batch[key].shape) for key in sorted(batch_image_keys)}
    for key in batch_image_keys:
        if batch[key].dtype != torch.uint8:
            raise ValueError(f"expected uint8 real camera batch before processor: {key}")
        batch[key] = batch[key].to(dtype=torch.float32) / 255.0

    torch.cuda.reset_peak_memory_stats()
    policy = make_policy(config, ds_meta=dataset.meta, rename_map=RENAME_MAP)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=None,
        dataset_stats=dataset.meta.stats,
    )
    rename_step = preprocessor.steps[0]
    if rename_step.__class__.__name__ != "RenameObservationsProcessorStep":
        raise RuntimeError("canonical PI0.5 preprocessor no longer starts with rename")
    rename_step.rename_map = RENAME_MAP
    processed = preprocessor(batch)
    processed_image_keys = {key for key in processed if key.startswith("observation.images.")}
    if processed_image_keys != set(RENAME_MAP.values()):
        raise ValueError(f"processor emitted unexpected camera keys: {sorted(processed_image_keys)}")
    if tuple(processed["observation.state"].shape) != (1, 8):
        raise ValueError("processed state is not (1, 8)")
    if tuple(processed["action"].shape) != (1, 50, 8):
        raise ValueError("processed action is not (1, 50, 8)")

    named = list(policy.named_parameters())
    vlm = [(name, parameter) for name, parameter in named if ".paligemma." in name]
    expert = [(name, parameter) for name, parameter in named if ".gemma_expert." in name]
    projection_tokens = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    projections = [
        (name, parameter) for name, parameter in named if any(token in name for token in projection_tokens)
    ]
    if not vlm or not expert or not projections:
        raise RuntimeError("PI0.5 parameter categories could not be identified")
    if any(parameter.requires_grad for _, parameter in vlm):
        raise RuntimeError("VLM backbone is not frozen in expert-only mode")
    if not any(parameter.requires_grad for _, parameter in expert):
        raise RuntimeError("action expert has no trainable parameters")
    if not any(parameter.requires_grad for _, parameter in projections):
        raise RuntimeError("action projections have no trainable parameters")

    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    optimizer.zero_grad(set_to_none=True)
    policy.train()
    loss, loss_details = policy(processed)
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"one-step loss is not a finite scalar: {loss}")
    loss.backward()

    categories = {
        "vlm_backbone": _gradient_summary(vlm),
        "action_expert": _gradient_summary(expert),
        "action_projections": _gradient_summary(projections),
    }
    if categories["vlm_backbone"]["tensors_with_grad"] != 0:
        raise RuntimeError("frozen VLM backbone accumulated gradients")
    for category in ("action_expert", "action_projections"):
        summary = categories[category]
        if summary["tensors_with_grad"] == 0:
            raise RuntimeError(f"{category} accumulated no gradients")
        if summary["finite_tensors_with_grad"] != summary["tensors_with_grad"]:
            raise RuntimeError(f"{category} has non-finite gradients")
        if summary["nonzero_gradient_tensors"] == 0:
            raise RuntimeError(f"{category} has only zero gradients")

    witness_name, witness_parameter = next(
        (name, parameter)
        for name, parameter in projections
        if parameter.grad is not None and float(parameter.grad.detach().abs().max().item()) > 0.0
    )
    witness_flat_index = int(witness_parameter.grad.detach().abs().reshape(-1).argmax().item())
    witness_before = float(witness_parameter.detach().reshape(-1)[witness_flat_index].item())
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), config.optimizer_grad_clip_norm, error_if_nonfinite=True
    )
    optimizer.step()
    torch.cuda.synchronize()
    witness_after = float(witness_parameter.detach().reshape(-1)[witness_flat_index].item())
    if witness_before == witness_after:
        raise RuntimeError(
            "optimizer step state advanced but the maximum-gradient projection witness did not change"
        )
    optimizer_step_values = []
    for state in optimizer.state.values():
        if "step" in state:
            step = state["step"]
            optimizer_step_values.append(float(step.item() if hasattr(step, "item") else step))
    if not optimizer_step_values or max(optimizer_step_values) < 1.0:
        raise RuntimeError("optimizer has no completed step state")

    trainable_groups = {
        "action_expert": sum(parameter.numel() for _, parameter in expert if parameter.requires_grad),
        "action_projections": sum(
            parameter.numel() for _, parameter in projections if parameter.requires_grad
        ),
    }
    result = {
        "success": True,
        "scope": "expert-only one-batch/one-step smoke; not strategy validation",
        "base_checkpoint": {
            "repo_id": BASE_REPO_ID,
            "revision": BASE_REVISION,
            "snapshot_path": str(base_snapshot),
            "model_bytes": (base_snapshot / "model.safetensors").stat().st_size,
        },
        "lerobot_commit": LEROBOT_COMMIT,
        "derived_dataset": {
            "root": str(dataset_root),
            "repo_id": metadata.repo_id,
            "source_revision": SOURCE_DATASET_REVISION,
            "source_episode_index": 0,
            "rows": len(dataset),
            "raw_state_indices": RAW_STATE_INDICES,
            "raw_action_indices": RAW_ACTION_INDICES,
            "raw_action_index_16_excluded": True,
            "batch_image_keys_before_processor": sorted(batch_image_keys),
            "batch_image_shapes_before_processor": raw_batch_shapes,
            "batch_state_shape": [1, 8],
            "batch_action_shape": [1, 50, 8],
        },
        "configuration": {
            "batch_size": 1,
            "max_steps": 1,
            "dtype": config.dtype,
            "train_expert_only": config.train_expert_only,
            "freeze_vision_encoder": config.freeze_vision_encoder,
            "gradient_checkpointing": config.gradient_checkpointing,
            "use_relative_actions": config.use_relative_actions,
            "relative_action_state_indices": config.relative_action_state_indices,
            "rename_map": RENAME_MAP,
        },
        "loss": float(loss.detach().item()),
        "loss_details": _json_safe(loss_details),
        "gradient_norm_before_clip": float(grad_norm.detach().item()),
        "parameter_groups": trainable_groups,
        "gradient_evidence": categories,
        "optimizer": {
            "class": optimizer.__class__.__name__,
            "step_state_min": min(optimizer_step_values),
            "step_state_max": max(optimizer_step_values),
            "witness_parameter": witness_name,
            "witness_flat_index": witness_flat_index,
            "witness_before": witness_before,
            "witness_after": witness_after,
            "witness_changed": True,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "no_checkpoint_written": True,
    }
    if not math.isfinite(result["gradient_norm_before_clip"]):
        raise RuntimeError("gradient norm is non-finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset_root, args.base_snapshot, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
