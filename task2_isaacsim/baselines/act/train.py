#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Train Task 2 ACT, alternating one train epoch and one validation epoch."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from .contract import ACT_CAMERA_KEYS, ACT_CONTRACT, load_split_manifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_imports() -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.utils.feature_utils import dataset_to_policy_features
    except ImportError as error:
        raise RuntimeError(
            "ACT environment is incomplete; create it from environment.yml first"
        ) from error
    return locals()


class EpisodeSampleDataset:
    """Match original ACT: one randomly selected timestep per episode/epoch."""

    def __init__(
        self,
        base: Any,
        episodes: list[int],
        *,
        validation: bool,
        seed: int = 1000,
    ):
        self.base = base
        self.episodes = episodes
        self.validation = validation
        self.seed = seed
        self.epoch = 0
        metadata = base.meta.episodes
        self.ranges = [
            (
                int(metadata["dataset_from_index"][episode]),
                int(metadata["dataset_to_index"][episode]),
            )
            for episode in episodes
        ]

    def __len__(self) -> int:
        return len(self.episodes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        start, stop = self.ranges[index]
        if self.validation:
            generator = random.Random(
                self.seed + self.epoch * 10_000 + self.episodes[index]
            )
            absolute = generator.randrange(start, stop)
        else:
            absolute = random.randrange(start, stop)
        relative = self.base.absolute_to_relative_idx[absolute]
        item = self.base[relative]
        return {
            key: value
            for key, value in item.items()
            if key != "observation.images.eval_camera"
        }


def _load_stats(path: Path, base_stats: dict[str, Any], torch: Any) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = {key: dict(value) for key, value in base_stats.items()}
    for key in ("action", "observation.state"):
        if key not in payload:
            raise ValueError(f"training-only stats missing {key}")
        stats[key] = {
            name: torch.tensor(value, dtype=torch.float32)
            if isinstance(value, list)
            else value
            for name, value in payload[key].items()
        }
    return stats


def build_datasets(
    dataset_root: Path,
    manifest: dict[str, Any],
    chunk_size: int,
    *,
    seed: int = 1000,
) -> tuple[Any, Any, Any]:
    modules = _runtime_imports()
    metadata_cls = modules["LeRobotDatasetMetadata"]
    dataset_cls = modules["LeRobotDataset"]
    metadata = metadata_cls(manifest["dataset_repo_id"], root=dataset_root)
    delta_timestamps = {"action": [index / metadata.fps for index in range(chunk_size)]}

    def make(episodes: list[int]) -> Any:
        return dataset_cls(
            manifest["dataset_repo_id"],
            root=dataset_root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            return_uint8=True,
        )

    train_base = make(manifest["train_episodes"])
    validation_base = make(manifest["validation_episodes"])
    return (
        EpisodeSampleDataset(
            train_base,
            manifest["train_episodes"],
            validation=False,
            seed=seed,
        ),
        EpisodeSampleDataset(
            validation_base,
            manifest["validation_episodes"],
            validation=True,
            seed=seed,
        ),
        train_base.meta,
    )


def build_policy(meta: Any, args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    modules = _runtime_imports()
    torch = modules["torch"]
    feature_type = modules["FeatureType"]
    policy_feature = modules["PolicyFeature"]
    features = modules["dataset_to_policy_features"](meta.features)
    input_features = {
        "observation.state": features["observation.state"],
        **{
            key: policy_feature(
                type=feature_type.VISUAL,
                shape=(3, args.image_height, args.image_width),
            )
            for key in ACT_CAMERA_KEYS
        },
    }
    output_features = {"action": features["action"]}
    config = modules["ACTConfig"](
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        push_to_hub=False,
        chunk_size=args.chunk_size,
        n_action_steps=ACT_CONTRACT.n_action_steps,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        dim_model=ACT_CONTRACT.hidden_dim,
        n_heads=8,
        dim_feedforward=ACT_CONTRACT.dim_feedforward,
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        kl_weight=ACT_CONTRACT.kl_weight,
        optimizer_lr=args.learning_rate,
        optimizer_lr_backbone=args.learning_rate,
        optimizer_weight_decay=1e-4,
    )
    policy = modules["ACTPolicy"](config).to(args.device)
    train_stats = _load_stats(
        args.dataset_root / "act_train_stats.json", meta.stats, torch
    )
    preprocessor, postprocessor = modules["make_pre_post_processors"](
        policy_cfg=config,
        dataset_stats=train_stats,
    )
    return policy, preprocessor, postprocessor, train_stats


def _prepare_batch(
    batch: dict[str, Any],
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> dict[str, Any]:
    torch = modules["torch"]
    functional = modules["functional"]
    for key in ACT_CAMERA_KEYS:
        image = batch[key]
        if image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32) / 255.0
        batch[key] = functional.interpolate(
            image,
            size=(args.image_height, args.image_width),
            mode="bilinear",
            align_corners=False,
        )
    return batch


def _mean_metrics(sums: dict[str, float], samples: int) -> dict[str, float]:
    return {key: value / max(samples, 1) for key, value in sums.items()}


def _set_loss_evaluation_mode(policy: Any, torch: Any) -> None:
    """Enable ACT's VAE posterior for loss evaluation without dropout noise.

    LeRobot's ACT only computes the posterior and KL term while the module is in
    training mode.  Keep that loss path active, disable Dropout explicitly, and
    let the caller use inference_mode so validation never updates parameters.
    """

    policy.train()
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def _argument_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _save_checkpoint(
    output: Path,
    step: int,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    optimizer: Any,
    history: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    torch = _runtime_imports()["torch"]
    checkpoint = output / "checkpoints" / f"{step:06d}"
    pretrained = checkpoint / "pretrained_model"
    policy.save_pretrained(pretrained)
    preprocessor.save_pretrained(pretrained)
    postprocessor.save_pretrained(pretrained)
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    _write_json(
        checkpoint / "training_state.json",
        {
            "step": step,
            "epoch": len(history["train_loss"]),
            "arguments": _argument_dict(args),
        },
    )
    _write_json(output / "loss_history.json", history)
    last = output / "checkpoints" / "last"
    if last.is_symlink():
        last.unlink()
    if not last.exists():
        last.symlink_to(checkpoint.name, target_is_directory=True)
    return checkpoint


def train(args: argparse.Namespace) -> dict[str, Any]:
    modules = _runtime_imports()
    torch = modules["torch"]
    manifest = load_split_manifest(args.dataset_root / "act_split.json")
    if args.output_path.exists() and any(args.output_path.iterdir()):
        raise FileExistsError(f"output path is not empty: {args.output_path}")
    args.output_path.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds, validation_ds, meta = build_datasets(
        args.dataset_root,
        manifest,
        args.chunk_size,
        seed=args.seed,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    policy, preprocessor, postprocessor, _ = build_policy(meta, args)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    history: dict[str, Any] = {
        "train_loss": [],
        "train_l1_loss": [],
        "train_kl_loss": [],
        "validation_loss": [],
        "validation_l1_loss": [],
        "validation_kl_loss": [],
        "steps": [],
    }
    _write_json(
        args.output_path / "run_config.json",
        {**_argument_dict(args), "split": manifest},
    )
    step = 0
    epoch = 0
    started = time.monotonic()
    while step < args.steps:
        epoch += 1
        policy.train()
        train_sums = {"loss": 0.0, "l1_loss": 0.0, "kld_loss": 0.0}
        train_samples = 0
        for batch in train_loader:
            batch = _prepare_batch(batch, args, modules)
            batch = preprocessor(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            size = int(batch["action"].shape[0])
            train_samples += size
            train_sums["loss"] += float(loss.item()) * size
            train_sums["l1_loss"] += float(loss_dict["l1_loss"]) * size
            train_sums["kld_loss"] += float(loss_dict["kld_loss"]) * size
            step += 1
            if step % args.log_every == 0:
                print(f"step={step}/{args.steps} loss={loss.item():.6f}", flush=True)
            if step % args.checkpoint_every == 0:
                _save_checkpoint(
                    args.output_path,
                    step,
                    policy,
                    preprocessor,
                    postprocessor,
                    optimizer,
                    history,
                    args,
                )
            if step >= args.steps:
                break

        _set_loss_evaluation_mode(policy, torch)
        validation_ds.set_epoch(epoch)
        torch.manual_seed(args.seed + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + epoch)
        validation_sums = {"loss": 0.0, "l1_loss": 0.0, "kld_loss": 0.0}
        validation_samples = 0
        with torch.inference_mode():
            for batch in validation_loader:
                batch = _prepare_batch(batch, args, modules)
                batch = preprocessor(batch)
                loss, loss_dict = policy.forward(batch)
                size = int(batch["action"].shape[0])
                validation_samples += size
                validation_sums["loss"] += float(loss.item()) * size
                validation_sums["l1_loss"] += float(loss_dict["l1_loss"]) * size
                validation_sums["kld_loss"] += float(loss_dict["kld_loss"]) * size

        train_metrics = _mean_metrics(train_sums, train_samples)
        validation_metrics = _mean_metrics(validation_sums, validation_samples)
        history["train_loss"].append(train_metrics["loss"])
        history["train_l1_loss"].append(train_metrics["l1_loss"])
        history["train_kl_loss"].append(train_metrics["kld_loss"])
        history["validation_loss"].append(validation_metrics["loss"])
        history["validation_l1_loss"].append(validation_metrics["l1_loss"])
        history["validation_kl_loss"].append(validation_metrics["kld_loss"])
        history["steps"].append(step)
        _write_json(args.output_path / "loss_history.json", history)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": history["train_loss"],
                    "validation_loss": history["validation_loss"],
                    "train_l1_loss": history["train_l1_loss"],
                    "train_kl_loss": history["train_kl_loss"],
                    "validation_l1_loss": history["validation_l1_loss"],
                    "validation_kl_loss": history["validation_kl_loss"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    final_checkpoint = _save_checkpoint(
        args.output_path,
        step,
        policy,
        preprocessor,
        postprocessor,
        optimizer,
        history,
        args,
    )
    result = {
        "completed": True,
        "steps": step,
        "epochs": epoch,
        "elapsed_s": time.monotonic() - started,
        "checkpoint": str(final_checkpoint / "pretrained_model"),
        "loss_history": str(args.output_path / "loss_history.json"),
    }
    _write_json(args.output_path / "train_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=46000)
    parser.add_argument("--batch-size", type=int, default=ACT_CONTRACT.batch_size)
    parser.add_argument("--learning-rate", type=float, default=ACT_CONTRACT.learning_rate)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=ACT_CONTRACT.checkpoint_every,
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=ACT_CONTRACT.chunk_size)
    parser.add_argument("--image-height", type=int, default=ACT_CONTRACT.image_height)
    parser.add_argument("--image-width", type=int, default=ACT_CONTRACT.image_width)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    for name in ("steps", "batch_size", "checkpoint_every", "log_every", "chunk_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    args.dataset_root = args.dataset_root.resolve()
    args.output_path = args.output_path.resolve()
    return args


def main() -> int:
    args = parse_args()
    try:
        result = train(args)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
