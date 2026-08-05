#!/usr/bin/env python3
"""Build and optionally run a pinned, local-only PI05 training smoke test.

The module is dependency-light until ``--execute`` is used. This lets the
command builder and dataset selection logic run in ordinary repository CI,
while the actual LeRobot/PyTorch imports stay inside the lab GPU environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from task2_isaacsim.baselines.pi05.contract import (  # noqa: E402
    PI05_CONTRACT,
    POLICY_CAMERA_RENAME_MAP,
    RELATIVE_ACTION_STATE_INDICES,
    SMOKE_EXPERT_CONTRACT,
    validate_dataset_root,
)

LEROBOT_VERSION = "0.6.0"
LEROBOT_SOURCE_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
DEFAULT_DATASET_REPO_ID = "local/task2_pi05_smoke"
DEFAULT_MIN_FREE_GIB = 40.0
MAX_SMOKE_STEPS = 20
PALIGEMMA_TOKENIZER_REPO = "google/paligemma-3b-pt-224"
HUB_PREFLIGHT_FILES = (
    (PI05_CONTRACT.policy_path, PI05_CONTRACT.policy_revision, "config.json"),
    (PALIGEMMA_TOKENIZER_REPO, "main", "config.json"),
)


def load_episode_labels(dataset_root: Path) -> list[dict[str, Any]]:
    """Read and validate the Task 2 episode-level QA sidecar."""

    path = dataset_root / "task2_extras" / "episodes_task2.jsonl"
    if not path.is_file():
        raise ValueError(f"missing Task 2 episode labels: {path}")

    labels: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in {path} line {line_number}: {error}"
            ) from error
        if not isinstance(item, dict):
            raise ValueError(f"{path} line {line_number} must be an object")
        if not isinstance(item.get("episode_index"), int):
            raise ValueError(
                f"{path} line {line_number} has no integer episode_index"
            )
        if not isinstance(item.get("success"), bool):
            raise ValueError(
                f"{path} line {line_number} has no boolean success label"
            )
        labels.append(item)

    indices = [item["episode_index"] for item in labels]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate episode_index in {path}")
    return labels


def select_smoke_episodes(
    labels: list[dict[str, Any]],
    *,
    allow_unsuccessful: bool,
    max_episodes: int,
) -> tuple[list[int], bool]:
    """Prefer successful episodes; opt in to failed-only code smoke."""

    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    successful = sorted(
        item["episode_index"] for item in labels if item["success"]
    )
    if successful:
        return successful[:max_episodes], False

    if not allow_unsuccessful:
        raise ValueError(
            "dataset has no successful episodes; pass "
            "--allow-unsuccessful-smoke-data only for a pipeline code smoke"
        )
    available = sorted(item["episode_index"] for item in labels)
    if not available:
        raise ValueError("dataset has no episode labels")
    return available[:max_episodes], True


def build_train_command(
    *,
    dataset_root: Path,
    output_dir: Path,
    dataset_repo_id: str,
    episodes: list[int],
    steps: int,
    save_checkpoint: bool,
    executable: str = "lerobot-train",
) -> list[str]:
    """Return the exact LeRobot v0.6.0 command for a bounded smoke run."""

    if not 1 <= steps <= MAX_SMOKE_STEPS:
        raise ValueError(f"steps must be between 1 and {MAX_SMOKE_STEPS}")
    if not episodes:
        raise ValueError("at least one episode is required")
    if not dataset_repo_id.strip():
        raise ValueError("dataset_repo_id must not be empty")

    contract = SMOKE_EXPERT_CONTRACT
    return [
        executable,
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root.resolve()}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        f"--policy.path={contract.policy_path}",
        f"--policy.pretrained_revision={contract.policy_revision}",
        "--policy.device=cuda",
        f"--policy.dtype={contract.dtype}",
        f"--policy.max_state_dim={contract.max_state_dim}",
        f"--policy.max_action_dim={contract.max_action_dim}",
        f"--policy.chunk_size={contract.chunk_size}",
        f"--policy.n_action_steps={contract.n_action_steps}",
        (
            "--policy.gradient_checkpointing="
            f"{str(contract.gradient_checkpointing).lower()}"
        ),
        f"--policy.train_expert_only={str(contract.train_expert_only).lower()}",
        (
            "--policy.freeze_vision_encoder="
            f"{str(contract.freeze_vision_encoder).lower()}"
        ),
        f"--policy.use_relative_actions={str(contract.use_relative_actions).lower()}",
        "--policy.relative_action_state_indices="
        + json.dumps(RELATIVE_ACTION_STATE_INDICES, separators=(",", ":")),
        "--policy.compile_model=false",
        "--policy.push_to_hub=false",
        "--rename_map="
        + json.dumps(POLICY_CAMERA_RENAME_MAP, separators=(",", ":")),
        f"--output_dir={output_dir.resolve()}",
        "--job_name=task2_pi05_smoke",
        f"--steps={steps}",
        "--batch_size=1",
        "--num_workers=0",
        "--persistent_workers=false",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--log_freq=1",
        f"--save_checkpoint={str(save_checkpoint).lower()}",
        "--save_freq=1",
        "--save_checkpoint_to_hub=false",
        "--wandb.enable=false",
    ]


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in (
        "lerobot",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
        "torchcodec",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def verify_required_hub_access(
    download_file: Callable[..., str] | None = None,
) -> list[dict[str, str]]:
    """Fetch small config files to fail fast on Hub access requirements."""

    if download_file is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is not installed in the active environment"
            ) from error
        download_file = hf_hub_download

    dependencies: list[dict[str, str]] = []
    for repo_id, revision, filename in HUB_PREFLIGHT_FILES:
        try:
            cached_path = download_file(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
            )
        except Exception as error:
            action = "verify the pinned model repository is accessible"
            if repo_id == PALIGEMMA_TOKENIZER_REPO:
                action = (
                    "accept the Google PaliGemma usage license at "
                    "https://huggingface.co/google/"
                    "paligemma-3b-pt-224 and run `hf auth login` with the "
                    "same HF_HOME; never paste the token into logs or chat"
                )
            raise RuntimeError(
                "cannot access required Hub file "
                f"{repo_id}@{revision}/{filename}; "
                f"{action}. Original error: {error}"
            ) from error
        dependencies.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "filename": filename,
                "cached_path": str(cached_path),
            }
        )
    return dependencies


def inspect_runtime(lerobot_source_root: Path) -> dict[str, Any]:
    """Fail fast on the pinned source, LeRobot version, and CUDA/bfloat16."""

    if sys.version_info < (3, 12):
        raise RuntimeError("LeRobot v0.6.0 requires Python 3.12 or newer")

    versions = _package_versions()
    if versions["lerobot"] != LEROBOT_VERSION:
        raise RuntimeError(
            f"expected lerobot=={LEROBOT_VERSION}, got {versions['lerobot']!r}"
        )

    try:
        source_commit = subprocess.run(
            ["git", "-C", str(lerobot_source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_status = subprocess.run(
            ["git", "-C", str(lerobot_source_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "unable to inspect LeRobot source at "
            f"{lerobot_source_root}: {error}"
        ) from error
    if source_commit != LEROBOT_SOURCE_COMMIT:
        raise RuntimeError(
            f"expected LeRobot source {LEROBOT_SOURCE_COMMIT}, "
            f"got {source_commit}"
        )
    if source_status:
        raise RuntimeError(
            "LeRobot source worktree is dirty; preserve a clean v0.6.0 pin"
        )

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is not installed in the active environment"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA GPU does not support bfloat16")

    hub_dependencies = verify_required_hub_access()
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "lerobot_source_root": str(lerobot_source_root.resolve()),
        "lerobot_source_commit": source_commit,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": True,
        "cuda_bfloat16_supported": True,
        "hub_dependencies": hub_dependencies,
        "gpu": {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
        },
    }


def verify_checkpoint(output_dir: Path, step: int) -> list[str]:
    checkpoint = output_dir / "checkpoints" / f"{step:06d}"
    required = [
        checkpoint / "pretrained_model" / "config.json",
        checkpoint / "pretrained_model" / "model.safetensors",
        checkpoint / "pretrained_model" / "train_config.json",
        checkpoint / "training_state" / "training_step.json",
    ]
    return [str(path) for path in required if not path.is_file()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and run a bounded Task 2 PI05 training smoke."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help=(
            "Logical LeRobot dataset id; --dataset.root remains authoritative."
        ),
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--allow-unsuccessful-smoke-data",
        action="store_true",
        help=(
            "Allow failed-only data solely to verify the training code path; "
            "never interpret the resulting weights as a baseline."
        ),
    )
    parser.add_argument(
        "--lerobot-source-root",
        type=Path,
        help=(
            "Required with --execute; must be clean v0.6.0 at the "
            "pinned commit."
        ),
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=DEFAULT_MIN_FREE_GIB,
    )
    parser.add_argument(
        "--environment-report",
        type=Path,
        help="JSON evidence path; defaults beside the output directory.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run after validation. Without this flag, only print the command."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    info, errors = validate_dataset_root(dataset_root)
    if errors:
        print("FAIL: dataset contract")
        for error in errors:
            print(f"  - {error}")
        return 2

    try:
        labels = load_episode_labels(dataset_root)
        if len(labels) != info.get("total_episodes"):
            raise ValueError(
                "episode label count does not match meta/info.json "
                "total_episodes"
            )
        label_indices = sorted(item["episode_index"] for item in labels)
        expected_indices = list(range(info["total_episodes"]))
        if label_indices != expected_indices:
            raise ValueError(
                "episode labels must cover each dataset index exactly once"
            )
        episodes, uses_unsuccessful_data = select_smoke_episodes(
            labels,
            allow_unsuccessful=args.allow_unsuccessful_smoke_data,
            max_episodes=args.max_episodes,
        )
        command = build_train_command(
            dataset_root=dataset_root,
            output_dir=output_dir,
            dataset_repo_id=args.dataset_repo_id,
            episodes=episodes,
            steps=args.steps,
            save_checkpoint=args.save_checkpoint,
        )
    except ValueError as error:
        print(f"FAIL: {error}")
        return 2

    successful_count = sum(1 for item in labels if item["success"])
    print("PASS: Task 2 PI05 smoke inputs")
    print(
        f"  episodes={episodes} successful_labels="
        f"{successful_count}/{len(labels)}"
    )
    if uses_unsuccessful_data:
        print(
            "  WARNING: failed episode selected for pipeline "
            "verification only; "
            "discard all resulting weights"
        )
    print(
        f"  mode={'execute' if args.execute else 'dry-run'} steps={args.steps}"
    )
    print(shlex.join(command))

    if not args.execute:
        return 0
    if args.lerobot_source_root is None:
        print("FAIL: --lerobot-source-root is required with --execute")
        return 2
    if output_dir.exists():
        print(f"FAIL: output directory already exists: {output_dir}")
        return 2
    repository_root = Path(__file__).resolve().parents[3]
    if output_dir.is_relative_to(repository_root):
        print("FAIL: output directory must be outside the competition repo")
        return 2
    if args.min_free_gib < 0:
        print("FAIL: --min-free-gib must not be negative")
        return 2

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_dir.parent).free / (1024**3)
    if free_gib < args.min_free_gib:
        print(
            f"FAIL: only {free_gib:.1f} GiB free at {output_dir.parent}; "
            f"require {args.min_free_gib:.1f} GiB"
        )
        return 2

    try:
        runtime = inspect_runtime(args.lerobot_source_root.resolve())
    except RuntimeError as error:
        print(f"FAIL: runtime preflight: {error}")
        return 2

    report_path = (
        args.environment_report.resolve()
        if args.environment_report
        else output_dir.parent / f"{output_dir.name}_preflight.json"
    )
    if report_path.is_relative_to(repository_root):
        print("FAIL: environment report must be outside the competition repo")
        return 2
    if report_path.is_relative_to(output_dir):
        print(
            "FAIL: environment report must not create the training "
            "output directory"
        )
        return 2
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "root": str(dataset_root),
            "episodes": episodes,
            "successful_labels": successful_count,
            "total_labels": len(labels),
            "uses_unsuccessful_data": uses_unsuccessful_data,
        },
        "runtime": runtime,
        "command": command,
        "output_dir": str(output_dir),
        "free_gib_before": round(free_gib, 3),
        "returncode": None,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"  evidence={report_path}")

    try:
        completed = subprocess.run(command, check=False)
    except OSError as error:
        report["launch_error"] = str(error)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"FAIL: unable to launch lerobot-train: {error}")
        return 127
    report["returncode"] = completed.returncode
    if completed.returncode == 0 and args.save_checkpoint:
        report["missing_checkpoint_files"] = verify_checkpoint(
            output_dir, args.steps
        )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    if completed.returncode != 0:
        print(f"FAIL: lerobot-train exited with {completed.returncode}")
        return completed.returncode
    if args.save_checkpoint and report["missing_checkpoint_files"]:
        print("FAIL: training returned success but checkpoint is incomplete")
        for path in report["missing_checkpoint_files"]:
            print(f"  - {path}")
        return 3

    print("PASS: one-batch PI05 training path")
    if args.save_checkpoint:
        print(f"PASS: checkpoint step {args.steps:06d}")
    if uses_unsuccessful_data:
        print("DECISION: delete this smoke output; it is not a usable policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
