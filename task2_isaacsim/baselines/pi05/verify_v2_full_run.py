#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify a V2 30k manifest, log, configs, and two checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from task2_isaacsim.baselines.pi05.contract import (
    PI05_MODEL_REVISION,
    V2_RELATIVE_ACTION_STATE_INDICES,
)

EXPECTED_STEPS = (15000, 30000)
EXPECTED_MODES = {
    "v2_full_30k": False,
    "v2_expert_30k": True,
}


def verify_run(run_root: Path, *, expected_mode: str) -> dict[str, Any]:
    expected_expert_only = EXPECTED_MODES[expected_mode]
    root = run_root.resolve()
    errors: list[str] = []
    manifest_path = root / "run_manifest.json"
    log_path = root / "train.log"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"valid": False, "run_root": str(root), "errors": [str(error)]}
    profile = manifest.get("profile", {})
    policy_profile = profile.get("policy", {})
    expected_mapping = list(V2_RELATIVE_ACTION_STATE_INDICES)
    checks = {
        "returncode_zero": manifest.get("returncode") == 0,
        "expected_v2_mode": profile.get("mode") == expected_mode,
        "phase_balanced": profile.get("_ebim_phase_balanced") is True,
        "expected_parameter_mode": (
            policy_profile.get("train_expert_only") is expected_expert_only
        ),
        "vision_encoder_frozen": (
            policy_profile.get("freeze_vision_encoder") is True
        ),
        "manifest_v2_mapping": (
            manifest.get("relative_action_state_indices") == expected_mapping
        ),
        "log_present": log_path.is_file(),
    }
    checkpoints_root = root / "training" / "checkpoints"
    numbered = sorted(
        path.name
        for path in checkpoints_root.iterdir()
        if path.is_dir() and re.fullmatch(r"[0-9]{6}", path.name)
    ) if checkpoints_root.is_dir() else []
    checks["numbered_checkpoints_exact"] = numbered == ["015000", "030000"]
    checkpoint_reports = []
    for step in EXPECTED_STEPS:
        checkpoint = checkpoints_root / f"{step:06d}"
        model = checkpoint / "pretrained_model" / "model.safetensors"
        train_config = checkpoint / "pretrained_model" / "train_config.json"
        optimizer = checkpoint / "training_state" / "optimizer_state.safetensors"
        training_step = checkpoint / "training_state" / "training_step.json"
        report: dict[str, Any] = {
            "step": step,
            "path": str(checkpoint),
            "model_bytes": model.stat().st_size if model.is_file() else None,
            "optimizer_bytes": (
                optimizer.stat().st_size if optimizer.is_file() else None
            ),
        }
        try:
            config = json.loads(train_config.read_text(encoding="utf-8"))
            policy = config["policy"]
            step_payload = json.loads(training_step.read_text(encoding="utf-8"))
            report["checks"] = {
                "model_present": (
                    model.is_file() and model.stat().st_size > 1_000_000_000
                ),
                "optimizer_present": (
                    optimizer.is_file() and optimizer.stat().st_size > 1_000_000_000
                ),
                "training_step": int(step_payload["step"]) == step,
                "base_model_revision": (
                    policy.get("pretrained_revision") == PI05_MODEL_REVISION
                ),
                "expected_parameter_mode": (
                    policy.get("train_expert_only") is expected_expert_only
                ),
                "vision_encoder_frozen": (
                    policy.get("freeze_vision_encoder") is True
                ),
                "n_action_steps_5": policy.get("n_action_steps") == 5,
                "v2_absolute_spine_mapping": (
                    policy.get("relative_action_state_indices")
                    == expected_mapping
                ),
                "steps_30000": config.get("steps") == 30000,
                "save_freq_15000": config.get("save_freq") == 15000,
            }
        except (KeyError, OSError, TypeError, ValueError) as error:
            report["checks"] = {"readable_complete_config": False}
            report["error"] = str(error)
        if not all(report["checks"].values()):
            errors.append(f"checkpoint {step:06d} failed: {report['checks']}")
        checkpoint_reports.append(report)
    if log_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        learnable = re.findall(r"num_learnable_params=([0-9]+)", log)
        total = re.findall(r"num_total_params=([0-9]+)", log)
        losses = [
            float(value)
            for value in re.findall(r"\bloss[:=]\s*([0-9.eE+-]+)", log)
        ]
        log_report = {
            "trainable_parameters": int(learnable[-1]) if learnable else None,
            "total_parameters": int(total[-1]) if total else None,
            "loss_samples": len(losses),
            "final_loss": losses[-1] if losses else None,
            "finite_final_loss": bool(losses and math.isfinite(losses[-1])),
            "step_30000_reported": bool(
                re.search(r"\bstep:30000\b|30000/30000", log)
            ),
        }
        trainable = log_report["trainable_parameters"]
        checks["expected_trainable_parameters"] = bool(
            trainable
            and (
                500_000_000 <= trainable <= 1_000_000_000
                if expected_expert_only
                else trainable >= 3_500_000_000
            )
        )
        checks["finite_training_loss"] = log_report["finite_final_loss"]
        checks["step_30000_reported"] = log_report["step_30000_reported"]
    else:
        log_report = {}
    errors.extend(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 1,
        "valid": not errors,
        "expected_mode": expected_mode,
        "run_root": str(root),
        "manifest": str(manifest_path),
        "log": str(log_path),
        "checks": checks,
        "numbered_checkpoints": numbered,
        "checkpoint_reports": checkpoint_reports,
        "log_report": log_report,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=tuple(EXPECTED_MODES), default="v2_expert_30k"
    )
    args = parser.parse_args()
    report = verify_run(args.run_root, expected_mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
