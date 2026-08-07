#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Download and audit the pinned organizer Task 2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contract import (
    ACTION_NAMES,
    ACTION_SIZE,
    EXPECTED_CAMERA_SHAPES,
    STATE_NAMES,
    STATE_SIZE,
    validate_dataset_root,
)

ORGANIZER_DATASET_REPO = "hermanprawiro/task2_fixpos_v1"
ORGANIZER_DATASET_REVISION = "1a7253a776b9a05d866da297789c456c2f0ed9f8"
ORGANIZER_DATASET_LICENSE = "apache-2.0"
ORGANIZER_USE_AUTHORIZATION_KIND = "organizer_provided_competition_data"
ORGANIZER_USE_SCOPE = "ebim_task2_training_and_evaluation"
ORGANIZER_EXPECTED_EPISODES = 22
ORGANIZER_EXPECTED_FRAMES = 20_869
SPLIT_SEED = 20260806
FIXED_AXIS_VARIANCE_EPSILON = 1e-10


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_manifest_path(dataset_root: Path) -> Path:
    return dataset_root.with_name(dataset_root.name + ".source.json")


def _normalise_license(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _extract_hugging_face_license(
    card_data: object,
    tags: object,
) -> tuple[str | None, str | None]:
    """Return license metadata declared by the pinned Hub revision.

    A constant in this repository is an expectation, not evidence. Only
    revision-scoped Hub card data or a Hub license tag may populate the source
    manifest.
    """

    payload: Mapping[str, object] | None = None
    if hasattr(card_data, "to_dict"):
        candidate = card_data.to_dict()
        if isinstance(candidate, Mapping):
            payload = candidate
    elif isinstance(card_data, Mapping):
        payload = card_data

    declared = payload.get("license") if payload is not None else None
    candidates = (
        declared
        if isinstance(declared, Sequence)
        and not isinstance(declared, (str, bytes))
        else [declared]
    )
    for candidate in candidates:
        normalised = _normalise_license(candidate)
        if normalised:
            return normalised, "huggingface_card_data"

    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
        for tag in tags:
            if not isinstance(tag, str) or not tag.startswith("license:"):
                continue
            normalised = _normalise_license(tag.partition(":")[2])
            if normalised:
                return normalised, "huggingface_tag"
    return None, None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def download_organizer_dataset(
    destination: Path,
    *,
    source_manifest: Path | None = None,
) -> dict[str, Any]:
    """Download exactly the approved organizer data revision.

    Authentication is delegated to the Hugging Face cache. The token is never
    read by this function and is therefore never written to the source
    manifest or command output.
    """

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required inside the PI0.5 image"
        ) from error

    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"destination already exists and is not empty: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    info = HfApi().dataset_info(
        ORGANIZER_DATASET_REPO,
        revision=ORGANIZER_DATASET_REVISION,
        files_metadata=True,
    )
    resolved_revision = str(info.sha)
    if resolved_revision != ORGANIZER_DATASET_REVISION:
        raise RuntimeError(
            "Hugging Face resolved an unexpected dataset revision: "
            f"{resolved_revision}"
        )
    declared_license, license_source = _extract_hugging_face_license(
        getattr(info, "card_data", None),
        getattr(info, "tags", None),
    )
    snapshot_download(
        repo_id=ORGANIZER_DATASET_REPO,
        repo_type="dataset",
        revision=ORGANIZER_DATASET_REVISION,
        local_dir=destination,
    )

    manifest_path = (
        source_manifest.resolve()
        if source_manifest is not None
        else _source_manifest_path(destination)
    )
    manifest = {
        "schema_version": 1,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": ORGANIZER_DATASET_REPO,
        "requested_revision": ORGANIZER_DATASET_REVISION,
        "resolved_revision": resolved_revision,
        "license": declared_license,
        "license_source": license_source,
        "expected_license": ORGANIZER_DATASET_LICENSE,
        "dataset_root": str(destination),
        "token_recorded": False,
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "source_manifest": str(manifest_path)}


def load_episode_metadata(dataset_root: Path) -> list[dict[str, Any]]:
    path = dataset_root / "task2_extras" / "episodes_task2.jsonl"
    if not path.is_file():
        raise ValueError(f"missing Task 2 episode metadata: {path}")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"metadata line {line_number} is not an object")
        episode = item.get("episode_index")
        if isinstance(episode, bool) or not isinstance(episode, int):
            raise ValueError(
                f"metadata line {line_number} has invalid episode_index"
            )
        if episode in seen:
            raise ValueError(f"duplicate episode metadata: {episode}")
        if not isinstance(item.get("success"), bool):
            raise ValueError(
                f"episode {episode} must contain a boolean success label"
            )
        seen.add(episode)
        records.append(item)
    return sorted(records, key=lambda item: item["episode_index"])


def _drop_count_is_zero(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value)) and float(value) == 0.0
    if isinstance(value, dict):
        return all(_drop_count_is_zero(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_drop_count_is_zero(item) for item in value)
    return False


def episode_eligibility(
    record: dict[str, Any],
    *,
    actual_frames: int | None,
    nonfinite_frames: int,
) -> list[str]:
    """Return stable reason codes for excluding one episode."""

    reasons: list[str] = []
    if record.get("success") is not True:
        reasons.append("success_false")
    declared_frames = record.get("frames")
    if isinstance(declared_frames, bool) or not isinstance(
        declared_frames, int
    ):
        reasons.append("invalid_declared_frames")
    elif actual_frames is None or actual_frames != declared_frames:
        reasons.append("frame_mismatch")
    if nonfinite_frames:
        reasons.append("nonfinite_action_or_state")
    if record.get("dropped_stale_frames") != 0:
        reasons.append("stale_frames_dropped")
    if not _drop_count_is_zero(record.get("encoder_dropped_frames")):
        reasons.append("encoder_frames_dropped")

    suggestion = record.get("success_suggestion")
    if not isinstance(suggestion, dict):
        reasons.append("missing_success_suggestion")
    else:
        if suggestion.get("is_orientation_correct") is not True:
            reasons.append("orientation_incorrect")
        iou = suggestion.get("iou_thermalpad_vs_target_current")
        if isinstance(iou, bool) or not isinstance(iou, (int, float)):
            reasons.append("invalid_iou")
        elif not math.isfinite(float(iou)) or float(iou) <= 0.0:
            reasons.append("iou_not_positive")
    return reasons


def organizer_split(
    eligible_episodes: Sequence[int],
    *,
    revision: str = ORGANIZER_DATASET_REVISION,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Build the immutable episode-level train/held-out split."""

    eligible = sorted({int(episode) for episode in eligible_episodes})
    if len(eligible) == ORGANIZER_EXPECTED_EPISODES:
        held_out_count = 4
    else:
        held_out_count = max(2, math.ceil(0.2 * len(eligible)))
    ranked = sorted(
        eligible,
        key=lambda episode: hashlib.sha256(
            f"{revision}:{episode}:{seed}".encode()
        ).hexdigest(),
    )
    held_out = sorted(ranked[: min(held_out_count, len(ranked))])
    held_out_set = set(held_out)
    train = [episode for episode in eligible if episode not in held_out_set]
    return {
        "algorithm": "sha256(revision:episode_index:seed)",
        "revision": revision,
        "seed": seed,
        "eligible": eligible,
        "train": train,
        "held_out": held_out,
        "formal_training_allowed": len(train) >= 10 and len(held_out) >= 2,
    }


@dataclass
class _VectorMoments:
    width: int
    count: int = 0
    minimum: list[float] = field(init=False)
    maximum: list[float] = field(init=False)
    mean: list[float] = field(init=False)
    m2: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.minimum = [math.inf] * self.width
        self.maximum = [-math.inf] * self.width
        self.mean = [0.0] * self.width
        self.m2 = [0.0] * self.width

    def update(self, values: Sequence[float]) -> bool:
        if len(values) != self.width:
            raise ValueError(
                f"expected vector width {self.width}, got {len(values)}"
            )
        vector = [float(value) for value in values]
        if not all(math.isfinite(value) for value in vector):
            return False
        self.count += 1
        for index, value in enumerate(vector):
            self.minimum[index] = min(self.minimum[index], value)
            self.maximum[index] = max(self.maximum[index], value)
            delta = value - self.mean[index]
            self.mean[index] += delta / self.count
            self.m2[index] += delta * (value - self.mean[index])
        return True

    def report(self, names: Sequence[str]) -> dict[str, Any]:
        variance = [
            value / self.count if self.count else math.nan for value in self.m2
        ]
        return {
            "count": self.count,
            "names": list(names),
            "minimum": self.minimum if self.count else [],
            "maximum": self.maximum if self.count else [],
            "mean": self.mean if self.count else [],
            "variance": variance if self.count else [],
        }


def _scan_parquet(dataset_root: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required inside the PI0.5 image for dataset audit"
        ) from error

    files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not files:
        raise ValueError("dataset contains no data parquet files")
    action_stats = _VectorMoments(ACTION_SIZE)
    state_stats = _VectorMoments(STATE_SIZE)
    episode_frames: dict[int, int] = {}
    nonfinite_frames: dict[int, int] = {}
    tracked_values = {index: [] for index in (0, 1, 2, 19)}

    for path in files:
        parquet_file = parquet.ParquetFile(path)
        required = {"episode_index", "action", "observation.state"}
        missing = required - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for batch in parquet_file.iter_batches(
            batch_size=512,
            columns=["episode_index", "action", "observation.state"],
        ):
            episodes = batch.column("episode_index").to_pylist()
            actions = batch.column("action").to_pylist()
            states = batch.column("observation.state").to_pylist()
            for episode_raw, action, state in zip(
                episodes, actions, states, strict=True
            ):
                episode = int(episode_raw)
                episode_frames[episode] = episode_frames.get(episode, 0) + 1
                action_finite = action_stats.update(action)
                state_finite = state_stats.update(state)
                if not action_finite or not state_finite:
                    nonfinite_frames[episode] = (
                        nonfinite_frames.get(episode, 0) + 1
                    )
                    continue
                for index in tracked_values:
                    tracked_values[index].append(float(action[index]))

    action_report = action_stats.report(ACTION_NAMES)
    state_report = state_stats.report(STATE_NAMES)
    medians: dict[str, float | None] = {}
    for index, values in tracked_values.items():
        ordered = sorted(values)
        if not ordered:
            medians[str(index)] = None
        else:
            midpoint = len(ordered) // 2
            if len(ordered) % 2:
                medians[str(index)] = ordered[midpoint]
            else:
                medians[str(index)] = (
                    ordered[midpoint - 1] + ordered[midpoint]
                ) / 2.0
    action_report["selected_medians"] = medians
    return {
        "parquet_files": len(files),
        "episode_frames": {
            str(key): value for key, value in sorted(episode_frames.items())
        },
        "nonfinite_frames": {
            str(key): value for key, value in sorted(nonfinite_frames.items())
        },
        "action": action_report,
        "state": state_report,
    }


def _probe_videos(dataset_root: Path, total_frames: int) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for the formal dataset audit")
    files = sorted((dataset_root / "videos").glob("**/*.mp4"))
    if not files:
        raise ValueError("dataset contains no MP4 videos")

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    frame_totals: dict[str, int] = {}
    frame_counts_complete: dict[str, bool] = {}
    for path in files:
        relative = path.relative_to(dataset_root)
        camera = relative.parts[1] if len(relative.parts) > 1 else "unknown"
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,pix_fmt,nb_frames,"
                "r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            errors.append(f"ffprobe failed for {relative}")
            continue
        streams = json.loads(result.stdout).get("streams", [])
        if len(streams) != 1:
            errors.append(f"expected one video stream in {relative}")
            continue
        stream = streams[0]
        nb_frames = stream.get("nb_frames")
        parsed_frames: int | None = None
        if isinstance(nb_frames, str) and nb_frames.isdigit():
            parsed_frames = int(nb_frames)
            frame_totals[camera] = frame_totals.get(camera, 0) + parsed_frames
            frame_counts_complete.setdefault(camera, True)
        else:
            frame_counts_complete[camera] = False
        records.append(
            {
                "path": relative.as_posix(),
                "camera": camera,
                "codec": stream.get("codec_name"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "pixel_format": stream.get("pix_fmt"),
                "frame_rate": stream.get("r_frame_rate"),
                "frames": parsed_frames,
            }
        )

    for camera, complete in sorted(frame_counts_complete.items()):
        if complete and frame_totals.get(camera) != total_frames:
            errors.append(
                f"video frame total for {camera} is "
                f"{frame_totals.get(camera)}, expected {total_frames}"
            )
    return {
        "files": records,
        "codec_names": sorted(
            {str(item["codec"]) for item in records if item.get("codec")}
        ),
        "frame_totals": frame_totals,
        "frame_counts_complete": frame_counts_complete,
        "errors": errors,
    }


def _hash_dataset_tree(dataset_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    tree = hashlib.sha256()
    paths = (item for item in dataset_root.rglob("*") if item.is_file())
    for path in sorted(paths):
        relative = path.relative_to(dataset_root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        digest = _sha256_file(path)
        size = path.stat().st_size
        relative_text = relative.as_posix()
        tree.update(f"{relative_text}\0{size}\0{digest}\n".encode())
        records.append(
            {"path": relative_text, "bytes": size, "sha256": digest}
        )
    return {
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "tree_sha256": tree.hexdigest(),
    }


def dataset_tree_checksums(dataset_root: Path) -> dict[str, Any]:
    """Return the content-addressed tree identity used by formal runs."""

    return _hash_dataset_tree(dataset_root.resolve())


def _read_card_license(dataset_root: Path) -> str | None:
    card = dataset_root / "README.md"
    if not card.is_file():
        return None
    match = re.search(
        r"(?m)^license:\s*['\"]?([^'\"\s]+)",
        card.read_text(encoding="utf-8", errors="replace"),
    )
    return _normalise_license(match.group(1)) if match else None


def _verify_license_evidence(
    source: Mapping[str, object],
    card_license: str | None,
) -> dict[str, Any]:
    source_license = _normalise_license(source.get("license")) or None
    source_license_source = source.get("license_source")
    source_verified = (
        source_license == ORGANIZER_DATASET_LICENSE
        and source_license_source
        in {"huggingface_card_data", "huggingface_tag"}
    )
    errors: list[str] = []
    if source_license is not None and (
        source_license != ORGANIZER_DATASET_LICENSE
    ):
        errors.append(
            "source manifest license must be "
            f"{ORGANIZER_DATASET_LICENSE!r}, got {source_license!r}"
        )

    card_verified = card_license == ORGANIZER_DATASET_LICENSE
    if card_license is not None and not card_verified:
        errors.append(
            "dataset card license must be "
            f"{ORGANIZER_DATASET_LICENSE!r}, got {card_license!r}"
        )
    verified = source_verified or card_verified
    if not verified:
        errors.append(
            "pinned dataset revision has no verifiable Apache-2.0 license "
            "in Hugging Face metadata or its dataset card"
        )
    return {
        "source_manifest_license": source_license,
        "source_manifest_license_source": source_license_source,
        "source_manifest_verified": source_verified,
        "dataset_card_verified": card_verified,
        "verified": verified,
        "errors": errors,
    }


def _verify_organizer_use_attestation(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "authorized": False,
            "path": None,
            "sha256": None,
            "errors": [],
        }

    resolved = path.resolve()
    payload = _read_json(resolved)
    errors: list[str] = []
    expected = {
        "schema_version": 1,
        "authorization_kind": ORGANIZER_USE_AUTHORIZATION_KIND,
        "dataset_repo_id": ORGANIZER_DATASET_REPO,
        "dataset_revision": ORGANIZER_DATASET_REVISION,
        "scope": ORGANIZER_USE_SCOPE,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"attestation {key} must be {value!r}")

    for key in ("acknowledged_by", "acknowledged_utc", "statement"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"attestation {key} must be a non-empty string")
    acknowledged_utc = payload.get("acknowledged_utc")
    if isinstance(acknowledged_utc, str) and acknowledged_utc.strip():
        try:
            parsed = datetime.fromisoformat(
                acknowledged_utc.strip().replace("Z", "+00:00")
            )
        except ValueError:
            errors.append("attestation acknowledged_utc must be ISO 8601")
        else:
            if parsed.tzinfo is None:
                errors.append(
                    "attestation acknowledged_utc must include a timezone"
                )

    return {
        "provided": True,
        "authorized": not errors,
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "authorization_kind": payload.get("authorization_kind"),
        "scope": payload.get("scope"),
        "acknowledged_by": payload.get("acknowledged_by"),
        "acknowledged_utc": payload.get("acknowledged_utc"),
        "statement": payload.get("statement"),
        "errors": errors,
    }


def audit_organizer_dataset(
    dataset_root: Path,
    *,
    source_manifest: Path | None = None,
    organizer_use_attestation: Path | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    source_path = (
        source_manifest.resolve()
        if source_manifest is not None
        else _source_manifest_path(dataset_root)
    )
    structural_errors: list[str] = []
    info, contract_errors = validate_dataset_root(dataset_root)
    structural_errors.extend(contract_errors)
    source: dict[str, Any] = {}
    if not source_path.is_file():
        structural_errors.append(f"missing source manifest: {source_path}")
    else:
        source = _read_json(source_path)
        expected_source = {
            "repo_id": ORGANIZER_DATASET_REPO,
            "requested_revision": ORGANIZER_DATASET_REVISION,
            "resolved_revision": ORGANIZER_DATASET_REVISION,
        }
        for key, expected in expected_source.items():
            if source.get(key) != expected:
                structural_errors.append(
                    f"source manifest {key} must be {expected!r}"
                )
    card_license = _read_card_license(dataset_root)
    license_evidence = _verify_license_evidence(source, card_license)
    license_verified = bool(license_evidence["verified"])
    use_attestation = _verify_organizer_use_attestation(
        organizer_use_attestation
    )
    dataset_use_authorized = license_verified or bool(
        use_attestation["authorized"]
    )

    records = load_episode_metadata(dataset_root)
    try:
        parquet = _scan_parquet(dataset_root)
    except (OSError, RuntimeError, ValueError) as error:
        parquet = {
            "error": str(error),
            "episode_frames": {},
            "nonfinite_frames": {},
        }
        structural_errors.append(f"parquet audit: {error}")
    total_frames = int(info.get("total_frames", 0))
    if total_frames != ORGANIZER_EXPECTED_FRAMES:
        structural_errors.append(
            f"organizer dataset must declare {ORGANIZER_EXPECTED_FRAMES} "
            f"frames, got {total_frames}"
        )
    actual_total = sum(parquet.get("episode_frames", {}).values())
    if actual_total != total_frames:
        structural_errors.append(
            f"parquet has {actual_total} frames, info.json declares "
            f"{total_frames}"
        )
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes != ORGANIZER_EXPECTED_EPISODES:
        structural_errors.append(
            f"organizer dataset must declare {ORGANIZER_EXPECTED_EPISODES} "
            f"episodes, got {total_episodes}"
        )
    expected_indices = set(range(total_episodes))
    metadata_indices = {int(item["episode_index"]) for item in records}
    if metadata_indices != expected_indices:
        structural_errors.append(
            "episode metadata indices do not match info.json total_episodes"
        )

    try:
        video = _probe_videos(dataset_root, total_frames)
        structural_errors.extend(video["errors"])
    except (OSError, RuntimeError, ValueError) as error:
        video = {"error": str(error), "errors": [str(error)]}
        structural_errors.append(f"video audit: {error}")
    checksums = _hash_dataset_tree(dataset_root)

    episode_reports: list[dict[str, Any]] = []
    eligible: list[int] = []
    actual_frames_by_episode = parquet.get("episode_frames", {})
    nonfinite_by_episode = parquet.get("nonfinite_frames", {})
    for item in records:
        episode = int(item["episode_index"])
        reasons = episode_eligibility(
            item,
            actual_frames=actual_frames_by_episode.get(str(episode)),
            nonfinite_frames=int(nonfinite_by_episode.get(str(episode), 0)),
        )
        if not reasons:
            eligible.append(episode)
        suggestion = item.get("success_suggestion")
        episode_reports.append(
            {
                "episode_index": episode,
                "eligible": not reasons,
                "exclusion_reasons": reasons,
                "success": item.get("success"),
                "frames_declared": item.get("frames"),
                "frames_actual": actual_frames_by_episode.get(str(episode)),
                "dropped_stale_frames": item.get("dropped_stale_frames"),
                "encoder_dropped_frames": item.get("encoder_dropped_frames"),
                "orientation_correct": (
                    suggestion.get("is_orientation_correct")
                    if isinstance(suggestion, dict)
                    else None
                ),
                "iou": (
                    suggestion.get("iou_thermalpad_vs_target_current")
                    if isinstance(suggestion, dict)
                    else None
                ),
            }
        )
    split = organizer_split(eligible)

    action_report = parquet.get("action", {})
    variance = action_report.get("variance", [])
    medians = action_report.get("selected_medians", {})
    base_variance = [
        variance[index] if len(variance) > index else None
        for index in range(3)
    ]
    spine_variance = variance[19] if len(variance) > 19 else None
    base_fixed = bool(base_variance) and all(
        value is not None and value <= FIXED_AXIS_VARIANCE_EPSILON
        for value in base_variance
    )
    spine_fixed = (
        spine_variance is not None
        and spine_variance <= FIXED_AXIS_VARIANCE_EPSILON
    )

    technical_audit_pass = not structural_errors
    audit_pass = technical_audit_pass and dataset_use_authorized
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": ORGANIZER_DATASET_REPO,
        "dataset_revision": ORGANIZER_DATASET_REVISION,
        "dataset_license": (
            ORGANIZER_DATASET_LICENSE if license_verified else None
        ),
        "expected_dataset_license": ORGANIZER_DATASET_LICENSE,
        "source_manifest": str(source_path),
        "source_manifest_sha256": (
            _sha256_file(source_path) if source_path.is_file() else None
        ),
        "dataset_card_license": card_license,
        "license_evidence": license_evidence,
        "organizer_use_attestation": use_attestation,
        "dataset_use_authorized": dataset_use_authorized,
        "info": {
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "robot_type": info.get("robot_type"),
            "camera_shapes": EXPECTED_CAMERA_SHAPES,
        },
        "critical_file_sha256": {
            relative: _sha256_file(dataset_root / relative)
            for relative in (
                "meta/info.json",
                "task2_extras/episodes_task2.jsonl",
            )
            if (dataset_root / relative).is_file()
        },
        "checksums": checksums,
        "video": video,
        "parquet": parquet,
        "episodes": episode_reports,
        "eligible_episode_count": len(eligible),
        "split": split,
        "rollout_constraints": {
            "variance_epsilon": FIXED_AXIS_VARIANCE_EPSILON,
            "base_action_variance": base_variance,
            "base_vx_vy_wz_clamp_to_zero": base_fixed,
            "spine_action_variance": spine_variance,
            "spine_hold_dataset_median": (
                medians.get("19") if spine_fixed else None
            ),
        },
        "structural_errors": structural_errors,
        "technical_audit_pass": technical_audit_pass,
        "audit_pass": audit_pass,
        "formal_training_allowed": audit_pass
        and split["formal_training_allowed"],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--destination", type=Path, required=True)
    download_parser.add_argument("--source-manifest", type=Path)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--dataset-root", type=Path, required=True)
    audit_parser.add_argument("--source-manifest", type=Path)
    audit_parser.add_argument("--organizer-use-attestation", type=Path)
    audit_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "download":
            report = download_organizer_dataset(
                args.destination,
                source_manifest=args.source_manifest,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        output = args.output.resolve()
        dataset_root = args.dataset_root.resolve()
        if output == dataset_root or dataset_root in output.parents:
            raise ValueError("audit output must be outside the dataset root")
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


if __name__ == "__main__":
    raise SystemExit(main())
