#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compute held-out ACT L1/KL losses on the fixed 20 Task 2 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contract import load_split_manifest
from .train import (
    _mean_metrics,
    _prepare_batch,
    _runtime_imports,
    _set_loss_evaluation_mode,
    _write_json,
    build_datasets,
)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    modules = _runtime_imports()
    torch = modules["torch"]
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    config.pretrained_path = str(args.checkpoint)
    config.pretrained_revision = None
    config.device = args.device
    first_camera = next(iter(config.image_features.values()))
    args.image_height = int(first_camera.shape[-2])
    args.image_width = int(first_camera.shape[-1])
    args.chunk_size = int(config.chunk_size)
    manifest = load_split_manifest(args.dataset_root / "act_split.json")
    _, validation_ds, meta = build_datasets(args.dataset_root, manifest, args.chunk_size)
    loader = torch.utils.data.DataLoader(
        validation_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    policy = make_policy(config, ds_meta=meta)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
    )
    _set_loss_evaluation_mode(policy, torch)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sums = {"loss": 0.0, "l1_loss": 0.0, "kld_loss": 0.0}
    samples = 0
    with torch.inference_mode():
        for batch in loader:
            batch = _prepare_batch(batch, args, modules)
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            size = int(batch["action"].shape[0])
            samples += size
            sums["loss"] += float(loss.item()) * size
            sums["l1_loss"] += float(loss_dict["l1_loss"]) * size
            sums["kld_loss"] += float(loss_dict["kld_loss"]) * size
    means = _mean_metrics(sums, samples)
    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "validation_episodes": manifest["validation_episodes"],
        "validation_samples": samples,
        "batch_size": args.batch_size,
        "validation_loss": means["loss"],
        "validation_l1_loss": means["l1_loss"],
        "validation_kl_loss": means["kld_loss"],
        "official_valid_placement_iou": None,
        "official_metric_note": (
            "Offline demonstrations cannot produce the official placement score; "
            "run one Isaac Sim rollout and then ./run_act.sh evaluate."
        ),
    }
    _write_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    args.checkpoint = args.checkpoint.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.output = args.output.resolve()
    return args


def main() -> int:
    args = parse_args()
    try:
        result = evaluate(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
