#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded V2 continuation with extra right-gripper supervision."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contract import V2_RELATIVE_ACTION_STATE_INDICES


RIGHT_GRIPPER_ACTION_INDEX = 18
RIGHT_GRIPPER_LOSS_WEIGHT = 8.0
ACTION_LOSS_WEIGHTS = [
    RIGHT_GRIPPER_LOSS_WEIGHT if index == RIGHT_GRIPPER_ACTION_INDEX else 1.0
    for index in range(20)
]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_v5_inputs(
    *, checkpoint: Path, dataset_root: Path, phase_manifest: Path
) -> dict[str, Any]:
    config = _load_json(checkpoint / "config.json")
    mapping = config.get("relative_action_state_indices")
    if mapping != list(V2_RELATIVE_ACTION_STATE_INDICES):
        raise ValueError("V5 init checkpoint does not use the V2 action contract")
    if int(config.get("n_action_steps", -1)) != 5:
        raise ValueError("V5 init checkpoint must preserve n_action_steps=5")
    manifest = _load_json(phase_manifest)
    if manifest.get("raw_dataset_modified") is not False:
        raise ValueError("phase manifest does not preserve the raw dataset")
    if int(manifest.get("chunk_size", -1)) != 50:
        raise ValueError("V5 requires the original 50-action training chunks")
    expected_phases = {
        "startup_rise",
        "approach",
        "grasp_acquisition",
        "lift_transfer",
        "lower_place",
        "release_retreat",
    }
    groups = manifest.get("train_sampling_groups", {})
    if set(groups) != expected_phases or any(not values for values in groups.values()):
        raise ValueError("phase manifest groups are missing or empty")
    train = {int(value) for value in manifest["train_episodes"]}
    held_out = {int(value) for value in manifest["held_out_episodes"]}
    if not train or not held_out or train & held_out:
        raise ValueError("phase manifest train/held-out split is invalid")
    info = _load_json(dataset_root / "meta" / "info.json")
    if int(info.get("total_frames", -1)) != 174719:
        raise ValueError("V5 dataset frame count drifted")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "phase_manifest": str(phase_manifest.resolve()),
        "train_episodes": sorted(train),
        "held_out_episodes": sorted(held_out),
        "sampling_ratios_percent": manifest["sampling_ratios_percent"],
        "relative_action_state_indices": mapping,
        "action_loss_weights": ACTION_LOSS_WEIGHTS,
    }


def run_v5_training(
    *,
    checkpoint: Path,
    dataset_root: Path,
    phase_manifest: Path,
    output_dir: Path,
    profile: Path,
    steps: int,
    execute: bool,
) -> int:
    if steps <= 0:
        raise ValueError("V5 steps must be positive")
    inputs = validate_v5_inputs(
        checkpoint=checkpoint.resolve(),
        dataset_root=dataset_root.resolve(),
        phase_manifest=phase_manifest.resolve(),
    )
    training_root = output_dir.resolve() / "training"
    if training_root.exists():
        raise ValueError(f"V5 training output already exists: {training_root}")
    command = [
        sys.executable,
        "-m",
        "task2_isaacsim.baselines.pi05.phase_train",
        f"--config_path={profile.resolve()}",
        f"--dataset.root={dataset_root.resolve()}",
        "--dataset.episodes="
        + json.dumps(inputs["train_episodes"], separators=(",", ":")),
        f"--output_dir={training_root}",
        f"--steps={steps}",
    ]
    smoke = steps == 1
    if smoke:
        command.extend(("--save_checkpoint=false", "--log_freq=1"))
    else:
        command.append(f"--save_freq={steps}")
    final_checkpoint = (
        None
        if smoke
        else str(training_root / f"checkpoints/{steps:06d}/pretrained_model")
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image": os.environ.get("EBIM_PI05_IMAGE", "unspecified-local-image"),
        "profile": str(profile.resolve()),
        "inputs": inputs,
        "steps": steps,
        "smoke": smoke,
        "command": command,
        "returncode": None,
        "final_checkpoint": final_checkpoint,
    }
    print("PASS: V5 weighted-gripper training preflight")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not execute:
        print("DRY RUN: no V5 training output was created")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    log_path = output_dir / "train.log"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    environment = dict(os.environ)
    environment["EBIM_PHASE_MANIFEST"] = inputs["phase_manifest"]
    environment["EBIM_ACTION_LOSS_WEIGHTS"] = json.dumps(
        ACTION_LOSS_WEIGHTS, separators=(",", ":")
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()
    manifest["returncode"] = returncode
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if returncode or smoke:
        return returncode
    checkpoint_root = Path(final_checkpoint)
    required = ("config.json", "model.safetensors", "train_config.json")
    missing = [name for name in required if not (checkpoint_root / name).is_file()]
    if missing:
        print(f"FAIL: incomplete V5 checkpoint: {missing}")
        return 3
    print(f"PASS: V5 final-only checkpoint: {checkpoint_root}")
    return 0
