#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify the selected V1 and V2 030000 checkpoints without a sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from task2_isaacsim.baselines.pi05.contract import (
    PI05_MODEL_REVISION,
    RELATIVE_ACTION_STATE_INDICES,
    V2_RELATIVE_ACTION_STATE_INDICES,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_mapping(preprocessor: dict[str, Any]) -> list[int | None]:
    steps = preprocessor.get("steps", [])
    matches = [
        step
        for step in steps
        if step.get("registry_name") == "relative_actions_processor"
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one relative_actions_processor")
    return list(matches[0]["config"]["state_indices"])


def verify_checkpoint(checkpoint: Path, *, contract: str) -> dict[str, Any]:
    root = checkpoint.resolve()
    expected_mapping = list(
        RELATIVE_ACTION_STATE_INDICES
        if contract == "v1"
        else V2_RELATIVE_ACTION_STATE_INDICES
    )
    model_root = root / "pretrained_model"
    state_root = root / "training_state"
    paths = {
        "model": model_root / "model.safetensors",
        "optimizer": state_root / "optimizer_state.safetensors",
        "train_config": model_root / "train_config.json",
        "policy_config": model_root / "config.json",
        "preprocessor": model_root / "policy_preprocessor.json",
        "postprocessor": model_root / "policy_postprocessor.json",
        "training_step": state_root / "training_step.json",
    }
    try:
        train_config = _read_json(paths["train_config"])
        policy_config = _read_json(paths["policy_config"])
        preprocessor = _read_json(paths["preprocessor"])
        training_step = _read_json(paths["training_step"])
        mapping = _checkpoint_mapping(preprocessor)
        policy = train_config["policy"]
        checks = {
            "directory_named_030000": root.name == "030000",
            "model_present": (
                paths["model"].is_file()
                and paths["model"].stat().st_size > 1_000_000_000
            ),
            "optimizer_present": (
                paths["optimizer"].is_file()
                and paths["optimizer"].stat().st_size > 1_000_000_000
            ),
            "processor_configs_present": (
                paths["preprocessor"].is_file()
                and paths["postprocessor"].is_file()
            ),
            "training_step_30000": int(training_step["step"]) == 30000,
            "batch_size_recorded": (
                int(train_config["batch_size"])
                == int(training_step["batch_size"])
            ),
            "steps_30000": train_config.get("steps") == 30000,
            "base_model_revision": (
                policy.get("pretrained_revision") == PI05_MODEL_REVISION
            ),
            "expert_only": policy.get("train_expert_only") is True,
            "vision_encoder_frozen": (
                policy.get("freeze_vision_encoder") is True
            ),
            "n_action_steps_5": policy.get("n_action_steps") == 5,
            "action_size_20": policy_config.get("output_features", {})
            .get("action", {})
            .get("shape")
            == [20],
            "expected_action_mapping": mapping == expected_mapping,
            "expected_spine_mapping": mapping[19]
            == (28 if contract == "v1" else None),
        }
        return {
            "valid": all(checks.values()),
            "contract": contract,
            "checkpoint": str(root),
            "batch_size": train_config.get("batch_size"),
            "mapping": mapping,
            "checks": checks,
            "sizes_bytes": {
                "model": paths["model"].stat().st_size
                if paths["model"].is_file()
                else None,
                "optimizer": paths["optimizer"].stat().st_size
                if paths["optimizer"].is_file()
                else None,
            },
            "errors": [],
        }
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        return {
            "valid": False,
            "contract": contract,
            "checkpoint": str(root),
            "checks": {"readable_complete_checkpoint": False},
            "errors": [str(error)],
        }


def verify_pair(v1_checkpoint: Path, v2_checkpoint: Path) -> dict[str, Any]:
    reports = [
        verify_checkpoint(v1_checkpoint, contract="v1"),
        verify_checkpoint(v2_checkpoint, contract="v2"),
    ]
    checks = {
        "only_v1_v2_030000_selected": all(
            report["checkpoint"].endswith("/030000") for report in reports
        ),
        "checkpoints_distinct": (
            reports[0]["checkpoint"] != reports[1]["checkpoint"]
        ),
        "both_checkpoints_valid": all(report["valid"] for report in reports),
    }
    return {
        "schema_version": 1,
        "valid": all(checks.values()),
        "scope": ["v1:030000", "v2:030000"],
        "checks": checks,
        "checkpoints": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_pair(args.v1_checkpoint, args.v2_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
