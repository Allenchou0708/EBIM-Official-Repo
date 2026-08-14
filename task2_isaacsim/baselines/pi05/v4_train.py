#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate and run the single Task 2 PI0.5 V4 continuation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contract import V2_RELATIVE_ACTION_STATE_INDICES
from .phase_conditioned_dataset import PHASE_PROMPTS


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_v4_inputs(
    *, checkpoint: Path, dataset_root: Path, pose_audit: Path
) -> dict[str, Any]:
    config = _load_json(checkpoint / "config.json")
    mapping = config.get("relative_action_state_indices")
    if mapping != list(V2_RELATIVE_ACTION_STATE_INDICES):
        raise ValueError("V4 init checkpoint does not use the V2 spine contract")
    if int(config.get("n_action_steps", -1)) != 5:
        raise ValueError("V4 init checkpoint must preserve n_action_steps=5")

    manifest_path = dataset_root / "phase_conditioned_manifest.json"
    manifest = _load_json(manifest_path)
    audit = _load_json(pose_audit)
    info = _load_json(dataset_root / "meta" / "info.json")
    if manifest.get("phase_prompts") != PHASE_PROMPTS:
        raise ValueError("V4 manifest prompt strings/order drifted")
    if manifest.get("raw_dataset_modified") is not False:
        raise ValueError("V4 manifest does not preserve the raw dataset")
    if int(info.get("total_frames", -1)) != 174719:
        raise ValueError("V4 dataset frame count drifted")
    if int(info.get("total_episodes", -1)) != 200:
        raise ValueError("V4 dataset episode count drifted")
    if int(info.get("total_tasks", -1)) != 8:
        raise ValueError("V4 dataset must contain seven prompts plus fallback")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required for the V4 task-order gate") from error
    task_rows = parquet.read_table(
        dataset_root / "meta" / "tasks.parquet"
    ).sort_by("task_index").to_pylist()
    expected_tasks = list(PHASE_PROMPTS.values()) + [
        "Pick up the thermal pad and place it on the target RAM board."
    ]
    if [row["task"] for row in task_rows] != expected_tasks:
        raise ValueError("V4 tasks.parquet task-index order drifted")
    if [int(row["task_index"]) for row in task_rows] != list(range(8)):
        raise ValueError("V4 tasks.parquet indices must be contiguous 0..7")
    audit_train = {
        int(row["episode"])
        for row in audit["episodes"]
        if row["split"] == "train"
    }
    audit_heldout = {
        int(row["episode"])
        for row in audit["episodes"]
        if row["split"] == "held_out"
    }
    manifest_train = {int(value) for value in manifest["train_episodes"]}
    manifest_heldout = {int(value) for value in manifest["held_out_episodes"]}
    if manifest_train != audit_train or manifest_heldout != audit_heldout:
        raise ValueError("V4 manifest split differs from the pre-grasp audit")
    groups = manifest["train_sampling_groups"]
    if set(groups) != set(PHASE_PROMPTS) or any(not group for group in groups.values()):
        raise ValueError("V4 sampling groups are missing or empty")
    fallback = manifest["fallback_episodes_excluded_from_training"]
    if fallback != [132]:
        raise ValueError("only physical-event fallback episode 132 may be excluded")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "phase_manifest": str(manifest_path.resolve()),
        "pose_audit": str(pose_audit.resolve()),
        "train_episodes": sorted(manifest_train),
        "held_out_episodes": sorted(manifest_heldout),
        "fallback_episodes_excluded_from_training": fallback,
        "phase_counts": manifest["train_phase_frame_counts"],
        "sampling_ratios_percent": manifest["sampling_ratios_percent"],
        "task_index_to_prompt": {
            int(row["task_index"]): row["task"] for row in task_rows
        },
        "relative_action_state_indices": mapping,
    }


def run_v4_training(
    *,
    checkpoint: Path,
    dataset_root: Path,
    pose_audit: Path,
    output_dir: Path,
    profile: Path,
    execute: bool,
) -> int:
    inputs = validate_v4_inputs(
        checkpoint=checkpoint.resolve(),
        dataset_root=dataset_root.resolve(),
        pose_audit=pose_audit.resolve(),
    )
    training_root = output_dir.resolve() / "training"
    if training_root.exists():
        raise ValueError(f"V4 training output already exists: {training_root}")
    command = [
        sys.executable,
        "-m",
        "task2_isaacsim.baselines.pi05.phase_train",
        f"--config_path={profile.resolve()}",
        f"--dataset.root={dataset_root.resolve()}",
        "--dataset.episodes="
        + json.dumps(inputs["train_episodes"], separators=(",", ":")),
        f"--output_dir={training_root}",
    ]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image": os.environ.get("EBIM_PI05_IMAGE", "unspecified-local-image"),
        "profile": str(profile.resolve()),
        "inputs": inputs,
        "command": command,
        "returncode": None,
        "final_checkpoint": str(
            training_root / "checkpoints/003000/pretrained_model"
        ),
    }
    print("PASS: V4 training preflight")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not execute:
        print("DRY RUN: no V4 training output was created")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    log_path = output_dir / "train.log"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    environment = dict(os.environ)
    environment["EBIM_PHASE_MANIFEST"] = inputs["phase_manifest"]
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
    if returncode:
        return returncode
    checkpoint_root = training_root / "checkpoints/003000/pretrained_model"
    required = ("config.json", "model.safetensors", "train_config.json")
    missing = [name for name in required if not (checkpoint_root / name).is_file()]
    if missing:
        print(f"FAIL: incomplete V4 checkpoint: {missing}")
        return 3
    print(f"PASS: V4 final-only checkpoint: {checkpoint_root}")
    return 0
