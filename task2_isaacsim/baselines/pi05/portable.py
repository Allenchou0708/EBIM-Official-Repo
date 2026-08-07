#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable Docker/Apptainer entry point for Task 2 PI0.5 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from task2_isaacsim.baselines.pi05.contract import (  # noqa: E402
    ACTION_SIZE,
    PI05_ACTION_SIZE,
    RELATIVE_ACTION_STATE_INDICES,
    to_absolute_action,
)
from task2_isaacsim.baselines.pi05.dataset_audit import (  # noqa: E402
    ORGANIZER_DATASET_REVISION,
    audit_organizer_dataset,
    dataset_tree_checksums,
    download_organizer_dataset,
)
from task2_isaacsim.baselines.pi05.heldout_evaluation import (  # noqa: E402
    checkpoint_sweep,
)
from task2_isaacsim.baselines.pi05.loss_parity import (
    loss_parity_report,  # noqa: E402
)
from task2_isaacsim.baselines.pi05.offline_inference import (
    run_offline_inference,  # noqa: E402
)
from task2_isaacsim.baselines.pi05.relative_dataset import (  # noqa: E402
    materialize_relative_dataset_view,
    parse_episode_list,
)
from task2_isaacsim.baselines.pi05.train_smoke import (
    load_episode_labels,  # noqa: E402
)

LEROBOT_SOURCE_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
PROFILE_DIRECTORY = Path(__file__).resolve().parent / "profiles"
PATCH_PATH = (
    Path(__file__).resolve().parent
    / "patches"
    / "lerobot-v0.6.0-task2-relative-map.patch"
)
FULL_PROFILE_MIN_GIB = 70.0
SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_external_output(path: Path) -> Path:
    """Keep datasets, checkpoints, and logs outside the Git checkout."""

    resolved = path.resolve()
    if resolved == SOURCE_ROOT or SOURCE_ROOT in resolved.parents:
        raise ValueError(
            f"output must be outside the competition repository: {resolved}"
        )
    return resolved


def _require_image_digest() -> str:
    digest = os.environ.get("EBIM_PI05_IMAGE_DIGEST", "").strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(
            "EBIM_PI05_IMAGE_DIGEST must be an immutable sha256 digest "
            "before an executable run"
        )
    return digest


def _profile_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate.resolve()
    names = {
        "smoke": "smoke_expert.yaml",
        "expert": "expert_finetune.yaml",
        "expert_finetune": "expert_finetune.yaml",
        "full": "full_finetune.yaml",
    }
    path = PROFILE_DIRECTORY / names.get(name_or_path, name_or_path)
    if not path.is_file():
        raise ValueError(f"unknown training profile: {name_or_path}")
    return path.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required inside the training image"
        ) from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"profile must contain a YAML object: {path}")
    return payload


def validate_profile(path: Path) -> dict[str, Any]:
    profile = _load_yaml(path)
    policy = profile.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("profile.policy must be an object")
    expected_mapping = list(RELATIVE_ACTION_STATE_INDICES)
    expected_modes = {
        "smoke_expert.yaml": ("expert_smoke", False, True, True),
        "expert_finetune.yaml": ("expert_finetune", True, True, True),
        "full_finetune.yaml": ("full_finetune", True, False, False),
    }
    if path.name not in expected_modes:
        raise ValueError(
            "profile filename must be smoke_expert.yaml, "
            "expert_finetune.yaml, or full_finetune.yaml"
        )
    mode, formal, train_expert_only, freeze_vision_encoder = expected_modes[
        path.name
    ]
    required = {
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "freeze_vision_encoder": freeze_vision_encoder,
        "train_expert_only": train_expert_only,
        "use_relative_actions": True,
        "max_state_dim": 37,
        "max_action_dim": 32,
    }
    for key, value in required.items():
        if policy.get(key) != value:
            raise ValueError(f"profile.policy.{key} must be {value!r}")
    if policy.get("relative_action_state_indices") != expected_mapping:
        raise ValueError(
            "profile uses the wrong Task 2 relative action mapping"
        )
    if policy.get("push_to_hub") is not False:
        raise ValueError("profile must keep policy.push_to_hub=false")
    return {**profile, "_ebim_mode": mode, "_ebim_formal": formal}


def inspect_gpu() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "total_memory_gib": props.total_memory / 1024**3,
        "cuda_version": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def inspect_video_runtime() -> dict[str, Any]:
    try:
        from torchcodec.decoders import VideoDecoder
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "TorchCodec video backend is unavailable; verify FFmpeg and "
            "the Python shared runtime in the image"
        ) from error
    return {
        "backend": "torchcodec",
        "decoder": VideoDecoder.__name__,
    }


def doctor(profile_path: Path) -> tuple[dict[str, Any], list[str]]:
    profile = validate_profile(profile_path)
    gpu = inspect_gpu()
    video_runtime = inspect_video_runtime()
    errors: list[str] = []
    if not gpu["bf16_supported"]:
        errors.append("selected GPU does not support bfloat16")
    mode = profile["_ebim_mode"]
    is_full = mode == "full_finetune"
    if is_full and gpu["total_memory_gib"] < FULL_PROFILE_MIN_GIB:
        errors.append(
            "full_finetune requires a single 80 GB-class GPU; "
            f"detected {gpu['total_memory_gib']:.1f} GiB. No automatic "
            "LoRA or expert-only downgrade is allowed."
        )
    return {
        "profile": str(profile_path),
        "profile_sha256": _sha256_file(profile_path),
        "mode": mode,
        "formal": profile["_ebim_formal"],
        "gpu": gpu,
        "video_runtime": video_runtime,
        "lerobot_source_commit": LEROBOT_SOURCE_COMMIT,
        "lerobot_patch_sha256": _sha256_file(PATCH_PATH),
    }, errors


def deterministic_split(
    total_episodes: int, held_out: int = 2
) -> tuple[list[int], list[int]]:
    if total_episodes <= held_out or held_out <= 0:
        raise ValueError(
            "total_episodes must be greater than held_out, and held_out "
            "must be positive"
        )
    cutoff = total_episodes - held_out
    return list(range(cutoff)), list(range(cutoff, total_episodes))


def _checkpoint_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for pattern in (
        "**/*.safetensors",
        "**/training_step.json",
        "**/train_config.json",
    ):
        for path in sorted(output_dir.glob(pattern)):
            hashes[str(path.relative_to(output_dir))] = _sha256_file(path)
    return hashes


def _trainable_counts(log_path: Path) -> dict[str, int | None]:
    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    result: dict[str, int | None] = {
        "trainable_parameters": None,
        "total_parameters": None,
    }
    patterns = {
        "trainable_parameters": r"num_learnable_params=([0-9]+)",
        "total_parameters": r"num_total_params=([0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = int(match.group(1))
    return result


def _training_metrics(log_path: Path) -> dict[str, Any]:
    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    patterns = {
        "loss": r"\bloss[:=]\s*([0-9.eE+-]+)",
        "gradient_norm": r"\bgrdn[:=]\s*([0-9.eE+-]+)",
        "memory_gib": r"\bmem(?:_gb)?[:=]\s*([0-9.eE+-]+)",
    }
    result: dict[str, Any] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        result[key] = float(matches[-1]) if matches else None
    losses = [float(value) for value in re.findall(patterns["loss"], text)]
    window = max(1, min(20, len(losses) // 5)) if losses else 0
    result["loss_samples"] = len(losses)
    result["loss_window"] = window
    result["initial_loss_mean"] = (
        sum(losses[:window]) / window if window else None
    )
    result["final_loss_mean"] = (
        sum(losses[-window:]) / window if window else None
    )
    result["loss_improved"] = (
        result["final_loss_mean"] < result["initial_loss_mean"]
        if len(losses) >= 2
        else None
    )
    return result


def _verify_parameter_mode(
    mode: str, counts: dict[str, int | None]
) -> list[str]:
    trainable = counts["trainable_parameters"]
    total = counts["total_parameters"]
    errors: list[str] = []
    if trainable is None or total is None:
        return [
            "training log did not report trainable and total parameter counts"
        ]
    if not 3_500_000_000 <= total <= 5_000_000_000:
        errors.append(f"unexpected PI0.5 total parameter count: {total}")
    if mode == "full_finetune" and trainable < 3_500_000_000:
        errors.append(
            f"full profile trained only {trainable} parameters; "
            "expected approximately 4B"
        )
    if (
        mode in {"expert_smoke", "expert_finetune"}
        and not 500_000_000 <= trainable <= 1_000_000_000
    ):
        errors.append(
            f"expert-only profile trained {trainable} parameters; "
            "expected the action expert range"
        )
    return errors


def _read_formal_audit(audit_path: Path) -> dict[str, Any]:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset audit report must be a JSON object")
    if payload.get("dataset_revision") != ORGANIZER_DATASET_REVISION:
        raise ValueError("dataset audit uses the wrong organizer revision")
    if payload.get("audit_pass") is not True:
        raise ValueError("dataset audit did not pass")
    if payload.get("formal_training_allowed") is not True:
        raise ValueError("dataset audit permits smoke only")
    split = payload.get("split")
    if not isinstance(split, dict):
        raise ValueError("dataset audit is missing its episode split")
    return payload


def _load_formal_audit(
    audit_path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    payload = _read_formal_audit(audit_path)
    critical = payload.get("critical_file_sha256")
    if not isinstance(critical, dict):
        raise ValueError("dataset audit is missing critical file hashes")
    for relative, expected in critical.items():
        path = dataset_root / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"dataset changed after audit: {relative}")
    current_tree = dataset_tree_checksums(dataset_root)["tree_sha256"]
    expected_tree = payload.get("checksums", {}).get("tree_sha256")
    if current_tree != expected_tree:
        raise ValueError("dataset tree checksum changed after audit")
    return payload


def _run_and_tee(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def command_train(args: argparse.Namespace) -> int:
    profile_path = _profile_path(args.profile)
    try:
        environment, errors = doctor(profile_path)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: doctor: {error}")
        return 2
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 2

    try:
        output_dir = _require_external_output(args.output_dir)
    except ValueError as error:
        print(f"FAIL: {error}")
        return 2
    if output_dir.exists():
        print(f"FAIL: output directory already exists: {output_dir}")
        return 2
    train_episodes = parse_episode_list(args.episodes)
    if not train_episodes:
        print("FAIL: at least one training episode is required")
        return 2
    try:
        labels = {
            item["episode_index"]: item["success"]
            for item in load_episode_labels(args.dataset_root.resolve())
        }
    except ValueError as error:
        print(f"FAIL: episode labels: {error}")
        return 2
    missing_labels = [
        episode for episode in train_episodes if episode not in labels
    ]
    if missing_labels:
        print(f"FAIL: selected episodes have no QA labels: {missing_labels}")
        return 2
    unsuccessful = [
        episode for episode in train_episodes if not labels[episode]
    ]
    is_formal = bool(environment["formal"])
    if args.allow_unsuccessful_smoke_data and is_formal:
        print(
            "FAIL: --allow-unsuccessful-smoke-data is limited to expert_smoke"
        )
        return 2
    if unsuccessful and is_formal:
        print(f"FAIL: {environment['mode']} refuses success=false episodes")
        return 2
    if unsuccessful and not args.allow_unsuccessful_smoke_data:
        print(
            "FAIL: success=false episodes require "
            "--allow-unsuccessful-smoke-data "
            "and remain disposable engineering evidence"
        )
        return 2

    audit_evidence: dict[str, Any] | None = None
    if is_formal:
        if args.audit_report is None:
            print("FAIL: formal training requires --audit-report")
            return 2
        try:
            audit = _load_formal_audit(
                args.audit_report.resolve(), args.dataset_root.resolve()
            )
        except (OSError, ValueError) as error:
            print(f"FAIL: formal dataset audit: {error}")
            return 2
        audit_train = [int(value) for value in audit["split"]["train"]]
        eligible = {
            int(item["episode_index"])
            for item in audit.get("episodes", [])
            if item.get("eligible") is True
        }
        selected = set(train_episodes)
        if not selected <= eligible:
            print(
                "FAIL: selected episodes include organizer data that did "
                "not pass QA"
            )
            return 2
        if not selected <= set(audit_train):
            print("FAIL: selected episodes overlap the held-out split")
            return 2
        if selected != set(audit_train) and not args.allow_train_subset:
            print(
                "FAIL: formal training must use the complete train split; "
                "use --allow-train-subset only for one-step or overfit gates"
            )
            return 2
        audit_evidence = {
            "report": str(args.audit_report.resolve()),
            "report_sha256": _sha256_file(args.audit_report.resolve()),
            "dataset_repo_id": audit["dataset_repo_id"],
            "dataset_revision": audit["dataset_revision"],
            "dataset_license": audit["dataset_license"],
            "dataset_tree_sha256": audit["checksums"]["tree_sha256"],
            "eligible_episodes": sorted(eligible),
            "train_split": audit_train,
            "held_out_split": [
                int(value) for value in audit["split"]["held_out"]
            ],
            "selected_train_episodes": train_episodes,
            "training_scope": (
                "complete_train_split"
                if selected == set(audit_train)
                else "explicit_train_subset"
            ),
            "rollout_constraints": audit["rollout_constraints"],
        }
    prepared_root = output_dir / "relative_dataset"
    training_root = output_dir / "training"
    command = [
        "lerobot-train",
        f"--config_path={profile_path}",
        f"--dataset.root={prepared_root}",
        "--dataset.episodes="
        + json.dumps(train_episodes, separators=(",", ":")),
        f"--output_dir={training_root}",
    ]
    for override in args.override:
        if not override.startswith("--"):
            print(f"FAIL: override must begin with --: {override}")
            return 2
        protected = (
            "--policy.train_expert_only=",
            "--policy.freeze_vision_encoder=",
            "--policy.use_relative_actions=",
            "--policy.relative_action_state_indices=",
            "--policy.max_state_dim=",
            "--policy.max_action_dim=",
            "--policy.push_to_hub=",
            "--dataset.root=",
            "--dataset.episodes=",
            "--output_dir=",
        )
        if override.startswith(protected):
            print(
                f"FAIL: protected contract override is not allowed: {override}"
            )
            return 2
        command.append(override)

    print("PASS: portable training preflight")
    print(" ".join(command))
    if not args.execute:
        print("DRY RUN: no dataset view, output, or checkpoint was created")
        return 0

    try:
        image_digest = _require_image_digest()
    except ValueError as error:
        print(f"FAIL: {error}")
        return 2

    output_dir.mkdir(parents=True)
    try:
        relative_manifest = materialize_relative_dataset_view(
            args.dataset_root,
            prepared_root,
            included_episodes=train_episodes,
            chunk_size=50,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: relative dataset preparation: {error}")
        return 2

    manifest_path = output_dir / "run_manifest.json"
    log_path = output_dir / "train.log"
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": environment,
        "relative_dataset": relative_manifest,
        "relative_action_state_indices": list(RELATIVE_ACTION_STATE_INDICES),
        "selected_success_false_episodes": unsuccessful,
        "engineering_only": bool(unsuccessful),
        "dataset_audit": audit_evidence,
        "image_digest": image_digest,
        "command": command,
        "returncode": None,
        "checkpoint_hashes": {},
    }
    _write_json(manifest_path, manifest)
    returncode = _run_and_tee(command, log_path)
    manifest["returncode"] = returncode
    manifest["parameter_counts"] = _trainable_counts(log_path)
    manifest["training_metrics"] = _training_metrics(log_path)
    manifest["parameter_mode_errors"] = _verify_parameter_mode(
        environment["mode"], manifest["parameter_counts"]
    )
    manifest["checkpoint_hashes"] = _checkpoint_hashes(training_root)
    _write_json(manifest_path, manifest)
    if returncode:
        print(f"FAIL: lerobot-train exited with {returncode}")
        return returncode
    if manifest["parameter_mode_errors"]:
        for error in manifest["parameter_mode_errors"]:
            print(f"FAIL: {error}")
        return 3
    if manifest["training_metrics"]["loss"] is None:
        print("FAIL: successful training log did not contain a finite loss")
        return 3
    if (
        args.require_loss_improvement
        and manifest["training_metrics"]["loss_improved"] is not True
    ):
        print(
            "FAIL: overfit gate requires the final loss window to be lower "
            "than the initial window"
        )
        return 3
    print(f"PASS: training command completed; manifest={manifest_path}")
    return 0


def command_resume(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.resolve()
    train_config = checkpoint / "pretrained_model" / "train_config.json"
    if not train_config.is_file():
        print(f"FAIL: missing checkpoint train config: {train_config}")
        return 2
    try:
        output_dir = _require_external_output(args.output_dir)
    except ValueError as error:
        print(f"FAIL: {error}")
        return 2
    command = [
        "lerobot-train",
        f"--config_path={train_config}",
        "--resume=true",
        f"--output_dir={output_dir}",
        f"--steps={args.steps}",
    ]
    print(" ".join(command))
    if not args.execute:
        print("DRY RUN: resume command was not launched")
        return 0
    try:
        image_digest = _require_image_digest()
    except ValueError as error:
        print(f"FAIL: {error}")
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "resume_manifest.json"
    if manifest_path.exists():
        print(f"FAIL: resume manifest already exists: {manifest_path}")
        return 2
    log_path = output_dir / "resume.log"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_digest": image_digest,
        "lerobot_source_commit": LEROBOT_SOURCE_COMMIT,
        "lerobot_patch_sha256": _sha256_file(PATCH_PATH),
        "relative_action_state_indices": list(RELATIVE_ACTION_STATE_INDICES),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_hashes": _checkpoint_hashes(checkpoint),
        "target_steps": args.steps,
        "command": command,
        "returncode": None,
        "resumed_checkpoint": None,
        "resumed_checkpoint_hashes": {},
    }
    _write_json(manifest_path, manifest)
    returncode = _run_and_tee(command, log_path)
    resumed_checkpoint = output_dir / "checkpoints" / f"{args.steps:06d}"
    manifest["returncode"] = returncode
    manifest["training_metrics"] = _training_metrics(log_path)
    manifest["resumed_checkpoint"] = str(resumed_checkpoint)
    manifest["resumed_checkpoint_hashes"] = _checkpoint_hashes(
        resumed_checkpoint
    )
    _write_json(manifest_path, manifest)
    if returncode:
        print(f"FAIL: resume exited with {returncode}")
        return returncode
    resumed_config = (
        resumed_checkpoint / "pretrained_model" / "train_config.json"
    )
    if not resumed_config.is_file():
        print(
            "FAIL: resume returned zero but did not create expected "
            f"checkpoint: {resumed_checkpoint}"
        )
        return 3
    print(f"PASS: resume checkpoint verified; manifest={manifest_path}")
    return 0


def command_validate_shadow(args: argparse.Namespace) -> int:
    try:
        image_digest = _require_image_digest()
        output = _require_external_output(args.output)
    except ValueError as error:
        print(f"FAIL: {error}")
        return 2
    records = []
    for line_number, line in enumerate(
        args.input.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        state = item["state"]
        relative = item["relative_action"]
        if len(relative) == PI05_ACTION_SIZE:
            relative = relative[:ACTION_SIZE]
        absolute = to_absolute_action(relative, state)
        if not all(0.0 <= absolute[index] <= 1.0 for index in (17, 18)):
            raise ValueError(
                f"line {line_number}: gripper output outside [0, 1]"
            )
        records.append(
            {
                "line": line_number,
                "absolute_action": list(absolute),
                "published": False,
            }
        )
    if not records:
        print("FAIL: shadow input contains no records")
        return 2
    _write_json(
        output,
        {
            "mode": "shadow",
            "image_digest": image_digest,
            "relative_action_state_indices": list(
                RELATIVE_ACTION_STATE_INDICES
            ),
            "ros_publication": False,
            "records": records,
        },
    )
    print(
        f"PASS: {len(records)} shadow actions validated; "
        "no command was published"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--profile", default="smoke")

    download_parser = subparsers.add_parser("download-organizer")
    download_parser.add_argument("--destination", type=Path, required=True)
    download_parser.add_argument("--source-manifest", type=Path)

    audit_parser = subparsers.add_parser("audit-dataset")
    audit_parser.add_argument("--dataset-root", type=Path, required=True)
    audit_parser.add_argument("--source-manifest", type=Path)
    audit_parser.add_argument("--organizer-use-attestation", type=Path)
    audit_parser.add_argument("--output", type=Path, required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--total-episodes", type=int, required=True)
    split_parser.add_argument("--held-out", type=int, default=2)

    parity_parser = subparsers.add_parser("loss-parity")
    parity_parser.add_argument("--output", type=Path)

    prepare_parser = subparsers.add_parser("prepare-relative")
    prepare_parser.add_argument("--dataset-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--episodes", required=True)
    prepare_parser.add_argument("--chunk-size", type=int, default=50)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--profile", default="smoke")
    train_parser.add_argument("--dataset-root", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--episodes", required=True)
    train_parser.add_argument("--audit-report", type=Path)
    train_parser.add_argument("--allow-train-subset", action="store_true")
    train_parser.add_argument(
        "--require-loss-improvement", action="store_true"
    )
    train_parser.add_argument("--override", action="append", default=[])
    train_parser.add_argument(
        "--allow-unsuccessful-smoke-data", action="store_true"
    )
    train_parser.add_argument("--execute", action="store_true")

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--checkpoint", type=Path, required=True)
    resume_parser.add_argument("--output-dir", type=Path, required=True)
    resume_parser.add_argument("--steps", type=int, required=True)
    resume_parser.add_argument("--execute", action="store_true")

    shadow_parser = subparsers.add_parser("validate-shadow")
    shadow_parser.add_argument("--input", type=Path, required=True)
    shadow_parser.add_argument("--output", type=Path, required=True)

    inference_parser = subparsers.add_parser("offline-inference")
    inference_parser.add_argument("--checkpoint", type=Path, required=True)
    inference_parser.add_argument("--dataset-root", type=Path, required=True)
    inference_parser.add_argument(
        "--dataset-repo-id", default="local/task2_pi05_shadow"
    )
    inference_parser.add_argument("--episodes", required=True)
    inference_parser.add_argument("--output", type=Path, required=True)
    inference_parser.add_argument("--audit-report", type=Path)
    inference_parser.add_argument("--samples-per-episode", type=int, default=3)
    inference_parser.add_argument("--seed", type=int, default=1000)

    sweep_parser = subparsers.add_parser("checkpoint-sweep")
    sweep_parser.add_argument("--checkpoints-root", type=Path, required=True)
    sweep_parser.add_argument("--dataset-root", type=Path, required=True)
    sweep_parser.add_argument(
        "--dataset-repo-id", default="local/task2_pi05_heldout"
    )
    sweep_parser.add_argument("--audit-report", type=Path, required=True)
    sweep_parser.add_argument("--output", type=Path, required=True)
    sweep_parser.add_argument("--max-frames", type=int, default=128)
    sweep_parser.add_argument("--seed", type=int, default=1000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        try:
            report, errors = doctor(_profile_path(args.profile))
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAIL: {error}")
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        for error in errors:
            print(f"FAIL: {error}")
        return 2 if errors else 0
    if args.command == "download-organizer":
        try:
            destination = _require_external_output(args.destination)
            report = download_organizer_dataset(
                destination,
                source_manifest=args.source_manifest,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAIL: {error}")
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "audit-dataset":
        try:
            output = _require_external_output(args.output)
            dataset_root = args.dataset_root.resolve()
            if output == dataset_root or dataset_root in output.parents:
                raise ValueError("audit output must be outside dataset root")
            report = audit_organizer_dataset(
                dataset_root,
                source_manifest=args.source_manifest,
                organizer_use_attestation=args.organizer_use_attestation,
            )
            _write_json(output, report)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAIL: {error}")
            return 2
        print(
            f"{'PASS' if report['audit_pass'] else 'FAIL'}: "
            f"eligible={report['eligible_episode_count']} "
            f"train={len(report['split']['train'])} "
            f"held_out={len(report['split']['held_out'])} "
            f"formal={report['formal_training_allowed']} output={output}"
        )
        return 0 if report["audit_pass"] else 3
    if args.command == "split":
        train, held_out = deterministic_split(
            args.total_episodes, args.held_out
        )
        print(
            json.dumps({"train": train, "held_out": held_out}, sort_keys=True)
        )
        return 0
    if args.command == "loss-parity":
        report = loss_parity_report()
        if args.output:
            _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-relative":
        try:
            output_root = _require_external_output(args.output_root)
        except ValueError as error:
            print(f"FAIL: {error}")
            return 2
        manifest = materialize_relative_dataset_view(
            args.dataset_root,
            output_root,
            included_episodes=parse_episode_list(args.episodes),
            chunk_size=args.chunk_size,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        return command_train(args)
    if args.command == "resume":
        return command_resume(args)
    if args.command == "validate-shadow":
        return command_validate_shadow(args)
    if args.command == "checkpoint-sweep":
        try:
            image_digest = _require_image_digest()
            output = _require_external_output(args.output)
            audit = _read_formal_audit(args.audit_report.resolve())
            held_out = [int(value) for value in audit["split"]["held_out"]]
            report = checkpoint_sweep(
                checkpoints_root=args.checkpoints_root,
                dataset_root=args.dataset_root,
                dataset_repo_id=args.dataset_repo_id,
                episodes=held_out,
                seed=args.seed,
                max_frames=args.max_frames,
                report_directory=output.with_suffix("").with_name(
                    output.stem + "_checkpoint_reports"
                ),
                rollout_constraints=audit["rollout_constraints"],
            )
            report["image_digest"] = image_digest
            report["dataset_audit"] = {
                "path": str(args.audit_report.resolve()),
                "sha256": _sha256_file(args.audit_report.resolve()),
                "revision": audit["dataset_revision"],
                "held_out_split": held_out,
            }
            _write_json(output, report)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAIL: checkpoint sweep: {error}")
            return 2
        print(
            "PASS: checkpoint sweep selected "
            f"step={report['selected_step']} "
            f"loss={report['selected_mean_loss']:.6f} "
            f"resume_to_step={report['resume_to_step']} output={output}"
        )
        return 0
    if args.command == "offline-inference":
        try:
            _require_image_digest()
            output = _require_external_output(args.output)
            episodes = parse_episode_list(args.episodes)
            constraints = None
            if args.audit_report is not None:
                audit = _read_formal_audit(args.audit_report.resolve())
                held_out = {int(value) for value in audit["split"]["held_out"]}
                if not set(episodes) <= held_out:
                    raise ValueError(
                        "formal offline inference episodes must come from "
                        "the immutable held-out split"
                    )
                constraints = audit["rollout_constraints"]
            report = run_offline_inference(
                checkpoint=args.checkpoint,
                dataset_root=args.dataset_root,
                dataset_repo_id=args.dataset_repo_id,
                episodes=episodes,
                output=output,
                samples_per_episode=args.samples_per_episode,
                seed=args.seed,
                rollout_constraints=constraints,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAIL: offline inference: {error}")
            return 2
        print(
            f"PASS: {len(report['records'])} finite 20-D offline shadow "
            f"outputs; base_nonzero={report['base_nonzero_records']} "
            "ROS publication=false"
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
