"""Audit and expose the Phase II Munich dataset as a right-arm-only view."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from task2_real.contract import validate_contract


REQUIRED_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "observation.state",
    "action",
    "annotation.human.validity",
    "observation.state.franka_robot_right_measured_joint_states",
    "observation.state.franka_robot_right_joint_states",
)

MIN_TRAIN_EPISODES = 40
MIN_HELD_OUT_EPISODES = 10
MIN_TRAIN_CLOSE_HOLD_REOPEN_EPISODES = 20
MIN_HELD_OUT_CLOSE_HOLD_REOPEN_EPISODES = 3
GRIPPER_CLOSED_THRESHOLD = 0.25
GRIPPER_OPEN_THRESHOLD = 0.90
GRIPPER_HOLD_FRAMES = 10


def _finite_vector(values: Sequence[float], expected: int, label: str) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != expected:
        raise ValueError(f"{label} must contain {expected} values, got {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} contains a non-finite value")
    return vector


def policy_vectors(
    raw_state: Sequence[float],
    raw_action: Sequence[float],
    contract: dict[str, Any],
) -> tuple[list[float], list[float]]:
    """Return only the eight right-side state and action values."""

    dataset = contract["dataset"]
    state = _finite_vector(raw_state, int(dataset["raw_state_size"]), "raw state")
    action = _finite_vector(raw_action, int(dataset["raw_action_size"]), "raw action")
    view = contract["policy_view"]
    return (
        [state[int(index)] for index in view["state"]["raw_indices"]],
        [action[int(index)] for index in view["action"]["raw_indices"]],
    )


def image_references(episode_index: int, timestamp: float) -> dict[str, dict[str, Any]]:
    """Build the two virtual video references accepted by the policy view."""

    if episode_index < 0 or not math.isfinite(float(timestamp)) or timestamp < 0:
        raise ValueError("episode index and timestamp must be non-negative")
    episode_chunk = episode_index // 1000
    result: dict[str, dict[str, Any]] = {}
    for key in ("observation.images.head", "observation.images.wrist_right"):
        result[key] = {
            "path": (
                f"videos/chunk-{episode_chunk:03d}/{key}/"
                f"episode_{episode_index:06d}.mp4"
            ),
            "timestamp": float(timestamp),
        }
    return result


def time_blocked_split(
    episode_indices: Iterable[int], held_out_fraction: float = 0.2
) -> dict[str, Any]:
    """Freeze the newest episode-index block as a chronology-proxy holdout."""

    episodes = sorted({int(index) for index in episode_indices})
    if len(episodes) < 2:
        raise ValueError("at least two eligible episodes are required")
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError("held-out fraction must be between zero and one")
    held_out_count = max(1, math.ceil(len(episodes) * held_out_fraction))
    if held_out_count >= len(episodes):
        raise ValueError("held-out block would consume all eligible episodes")
    train = episodes[:-held_out_count]
    held_out = episodes[-held_out_count:]
    payload = {
        "strategy": "episode_index_blocked_last_20_percent",
        "ordering_note": "episode_index is a chronology proxy; recording timestamps are not released",
        "held_out_fraction_requested": held_out_fraction,
        "train": train,
        "held_out": held_out,
    }
    return payload


def _longest_true_run(values: Any) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _true_runs(values: Any) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if bool(value) and start is None:
            start = index
        if not bool(value) and start is not None:
            runs.append({"start": start, "end_exclusive": index, "length": index - start})
            start = None
    if start is not None:
        end = len(values)
        runs.append({"start": start, "end_exclusive": end, "length": end - start})
    return runs


def validity_eligibility(values: Sequence[int]) -> dict[str, Any]:
    validity = [int(value) for value in values]
    unique = sorted(set(validity))
    return {
        "eligible": bool(validity) and unique == [1],
        "values": unique,
        "valid_frames": sum(value == 1 for value in validity),
        "contiguous_valid_runs": _true_runs([value == 1 for value in validity]),
    }


def continuous_gripper_lifecycle(
    values: Sequence[float],
    closed_threshold: float = GRIPPER_CLOSED_THRESHOLD,
    open_threshold: float = GRIPPER_OPEN_THRESHOLD,
    hold_frames: int = GRIPPER_HOLD_FRAMES,
) -> dict[str, Any]:
    """Detect open -> contiguous closed hold -> open with a hysteresis band."""

    if not 0.0 <= closed_threshold < open_threshold <= 1.0:
        raise ValueError("gripper thresholds must satisfy 0 <= closed < open <= 1")
    seen_open = False
    close_entries = 0
    reopen_entries = 0
    low_run = 0
    longest_low_run = 0
    held_since_close = False
    complete_cycles = 0
    for raw_value in values:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("gripper trajectory contains a non-finite value")
        if not seen_open:
            seen_open = value >= open_threshold
            continue
        if value <= closed_threshold:
            if low_run == 0 and not held_since_close:
                close_entries += 1
            low_run += 1
            longest_low_run = max(longest_low_run, low_run)
            if low_run >= hold_frames:
                held_since_close = True
            continue
        low_run = 0
        if value >= open_threshold and held_since_close:
            reopen_entries += 1
            complete_cycles += 1
            held_since_close = False
    return {
        "close_entries": close_entries,
        "reopen_after_hold_entries": reopen_entries,
        "longest_contiguous_closed_run_frames": longest_low_run,
        "complete_cycles": complete_cycles,
        "complete": complete_cycles > 0,
    }


def _summary(values: Any, np: Any) -> dict[str, Any]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q25": np.quantile(values, 0.25, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "q75": np.quantile(values, 0.75, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }


def audit_dataset(
    dataset_root: Path,
    contract: dict[str, Any],
    held_out_fraction: float = 0.2,
) -> dict[str, Any]:
    """Run the full parquet audit and return an immutable split manifest."""

    try:
        import numpy as np
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - exercised in Docker
        raise RuntimeError("numpy and pyarrow are required for the full audit") from error

    validate_contract(contract)
    state_indices = [int(index) for index in contract["policy_view"]["state"]["raw_indices"]]
    action_indices = [int(index) for index in contract["policy_view"]["action"]["raw_indices"]]
    if state_indices != list(range(21, 29)) or action_indices != list(range(8, 16)):
        raise ValueError("Phase II Gate A requires the locked right-side state[21:29]/action[8:16] mapping")
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    metadata = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata_by_episode = {int(item["episode_index"]): item for item in metadata}
    if len(metadata_by_episode) != len(metadata):
        raise ValueError("episode metadata contains duplicate indices")

    paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    files_grouped: dict[int, list[Path]] = {}
    for path in paths:
        episode_index = int(path.stem.removeprefix("episode_"))
        files_grouped.setdefault(episode_index, []).append(path)
    duplicate_file_episodes = sorted(
        index for index, grouped_paths in files_grouped.items() if len(grouped_paths) != 1
    )
    file_by_episode = {
        index: grouped_paths[0]
        for index, grouped_paths in files_grouped.items()
        if len(grouped_paths) == 1
    }
    structural_issues: list[dict[str, Any]] = []
    missing_files = sorted(set(metadata_by_episode) - set(file_by_episode))
    extra_files = sorted(set(file_by_episode) - set(metadata_by_episode))
    if missing_files:
        structural_issues.append({"code": "metadata_episode_missing_file", "episodes": missing_files})
    if extra_files:
        structural_issues.append({"code": "file_missing_metadata", "episodes": extra_files})
    if duplicate_file_episodes:
        structural_issues.append(
            {"code": "duplicate_episode_parquet", "episodes": duplicate_file_episodes}
        )

    expected_instruction = str(contract["policy_view"]["instruction"])
    task_mismatches = sorted(
        int(item["episode_index"])
        for item in metadata
        if item.get("tasks") != [expected_instruction]
    )
    if task_mismatches:
        structural_issues.append(
            {"code": "task_instruction_mismatch", "episodes": task_mismatches}
        )

    valid_annotation_frames = 0
    invalid_annotation_frames = 0
    eligible_frames = 0
    eligible_episodes: list[int] = []
    invalid_episodes: list[dict[str, Any]] = []
    mixed_validity_episodes: list[int] = []
    episode_records: list[dict[str, Any]] = []
    eligible_arrays: dict[int, dict[str, Any]] = {}

    for episode_index in sorted(set(metadata_by_episode) & set(file_by_episode)):
        path = file_by_episode[episode_index]
        schema_names = set(parquet.read_schema(path).names)
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in schema_names]
        if missing_columns:
            structural_issues.append(
                {"code": "missing_columns", "episode": episode_index, "columns": missing_columns}
            )
            continue
        table = parquet.read_table(path, columns=list(REQUIRED_COLUMNS))
        rows = table.num_rows
        expected_rows = int(metadata_by_episode[episode_index]["length"])
        episode_issue_codes: list[str] = []
        if rows != expected_rows:
            episode_issue_codes.append("metadata_length_mismatch")

        episode_column = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        frame_column = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        validity = np.asarray(
            table["annotation.human.validity"].to_pylist(), dtype=np.int64
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        right_measured = np.asarray(
            table[
                "observation.state.franka_robot_right_measured_joint_states"
            ].to_pylist(),
            dtype=np.float64,
        )
        right_gripper_state = np.asarray(
            table["observation.state.franka_robot_right_joint_states"].to_pylist(),
            dtype=np.float64,
        )

        if states.shape != (rows, int(contract["dataset"]["raw_state_size"])):
            episode_issue_codes.append("raw_state_shape")
        if actions.shape != (rows, int(contract["dataset"]["raw_action_size"])):
            episode_issue_codes.append("raw_action_shape")
        if episode_column.shape != (rows,) or not np.all(episode_column == episode_index):
            episode_issue_codes.append("episode_index_mismatch")
        if frame_column.shape != (rows,) or not np.array_equal(frame_column, np.arange(rows)):
            episode_issue_codes.append("frame_index_not_contiguous")
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
            episode_issue_codes.append("timestamp_invalid")
        if not np.all(np.isfinite(states)):
            episode_issue_codes.append("raw_state_non_finite")
        if not np.all(np.isfinite(actions)):
            episode_issue_codes.append("raw_action_non_finite")
        validity_result = validity_eligibility(validity.tolist())
        validity_values = validity_result["values"]
        if not set(validity_values) <= {0, 1}:
            episode_issue_codes.append("validity_outside_binary")
        if states.shape == (rows, 42) and right_measured.shape == (rows, 7) and right_gripper_state.shape == (rows, 1):
            component_right = np.concatenate((right_measured, right_gripper_state), axis=1)
            if not np.allclose(states[:, 21:29], component_right, rtol=0.0, atol=0.0):
                episode_issue_codes.append("right_state_flat_component_mismatch")
        else:
            episode_issue_codes.append("right_state_component_shape")

        annotation_valid = int(validity_result["valid_frames"])
        annotation_invalid = int(rows - annotation_valid)
        valid_runs = validity_result["contiguous_valid_runs"]
        valid_annotation_frames += annotation_valid
        invalid_annotation_frames += annotation_invalid
        if len(validity_values) > 1:
            mixed_validity_episodes.append(episode_index)

        eligible = not episode_issue_codes and bool(validity_result["eligible"])
        record: dict[str, Any] = {
            "episode_index": episode_index,
            "frames": rows,
            "validity_values": validity_values,
            "valid_frames": annotation_valid,
            "contiguous_valid_runs": valid_runs,
            "eligible": eligible,
            "issue_codes": episode_issue_codes,
        }
        if eligible:
            eligible_episodes.append(episode_index)
            eligible_frames += rows
            selected_state = states[:, 21:29]
            selected_action = actions[:, 8:16]
            eligible_arrays[episode_index] = {
                "state": selected_state,
                "action": selected_action,
                "raw_state": states,
                "raw_action": actions,
            }
            gripper = selected_action[:, 7]
            record.update(
                {
                    "right_gripper_min": float(np.min(gripper)),
                    "right_gripper_max": float(np.max(gripper)),
                    "right_gripper_range": float(np.ptp(gripper)),
                    "right_gripper_max_abs_delta": float(
                        np.max(np.abs(np.diff(gripper))) if rows > 1 else 0.0
                    ),
                    "right_gripper_value_changes": int(
                        np.count_nonzero(np.diff(gripper) != 0.0)
                    ),
                    "right_joint_max_range_rad": float(
                        np.max(np.ptp(selected_action[:, :7], axis=0))
                    ),
                    "left_joint_max_range_rad": float(
                        np.max(np.ptp(actions[:, :7], axis=0))
                    ),
                }
            )
        else:
            reasons = list(episode_issue_codes)
            if validity_values != [1]:
                reasons.append("annotation.human.validity_not_all_one")
            invalid_episodes.append(
                {
                    "episode_index": episode_index,
                    "frames": rows,
                    "valid_frames": annotation_valid,
                    "reasons": reasons,
                }
            )
        if episode_issue_codes:
            structural_issues.append(
                {
                    "code": "episode_structural_issue",
                    "episode": episode_index,
                    "issue_codes": episode_issue_codes,
                }
            )
        episode_records.append(record)

    if not eligible_episodes:
        raise ValueError("no fully valid structurally eligible episodes")
    split = time_blocked_split(eligible_episodes, held_out_fraction)
    split["train_frames"] = sum(
        int(metadata_by_episode[index]["length"]) for index in split["train"]
    )
    split["held_out_frames"] = sum(
        int(metadata_by_episode[index]["length"]) for index in split["held_out"]
    )
    train_state = np.concatenate(
        [eligible_arrays[index]["state"] for index in split["train"]], axis=0
    )
    train_action = np.concatenate(
        [eligible_arrays[index]["action"] for index in split["train"]], axis=0
    )
    train_raw_action = np.concatenate(
        [eligible_arrays[index]["raw_action"] for index in split["train"]], axis=0
    )
    train_initial_state = np.concatenate(
        [eligible_arrays[index]["state"][:20] for index in split["train"]], axis=0
    )
    train_gripper = train_action[:, 7]
    gripper_unique = np.unique(train_gripper)
    histogram_counts, histogram_edges = np.histogram(
        train_gripper, bins=np.linspace(0.0, 1.0, 21)
    )
    low_mode_count = int(np.count_nonzero(train_gripper <= GRIPPER_CLOSED_THRESHOLD))
    high_mode_count = int(np.count_nonzero(train_gripper >= GRIPPER_OPEN_THRESHOLD))
    low_mode_fraction = low_mode_count / int(train_gripper.size)
    high_mode_fraction = high_mode_count / int(train_gripper.size)
    hysteresis_supported = bool(
        float(np.min(train_gripper)) >= 0.0
        and float(np.max(train_gripper)) <= 1.0
        and low_mode_fraction >= 0.01
        and high_mode_fraction >= 0.01
        and GRIPPER_OPEN_THRESHOLD - GRIPPER_CLOSED_THRESHOLD >= 0.5
    )
    lifecycle: dict[str, Any] = {
        "classification_allowed": hysteresis_supported,
        "reason": (
            "train-only histogram has populated <=0.25 closed and >=0.90 open modes"
            if hysteresis_supported
            else "train-only histogram does not support the fixed continuous-width hysteresis"
        ),
        "fit_scope": "split.train only",
        "closed_at_or_below": GRIPPER_CLOSED_THRESHOLD,
        "open_at_or_above": GRIPPER_OPEN_THRESHOLD,
        "closed_hold_frames": GRIPPER_HOLD_FRAMES,
        "minimum_mode_fraction": 0.01,
    }
    train_complete_episode_ids: list[int] = []
    train_transition_counts: list[int] = []
    if hysteresis_supported:
        lifecycle.update(
            {
                "episodes_with_close_transition": 0,
                "episodes_with_closed_hold_at_least_0_5s": 0,
                "episodes_with_close_hold_reopen": 0,
            }
        )
        for episode_index in split["train"]:
            gripper = eligible_arrays[episode_index]["action"][:, 7]
            detected = continuous_gripper_lifecycle(gripper.tolist())
            train_transition_counts.append(
                int(detected["close_entries"] + detected["reopen_after_hold_entries"])
            )
            lifecycle["episodes_with_close_transition"] += int(
                detected["close_entries"] > 0
            )
            lifecycle["episodes_with_closed_hold_at_least_0_5s"] += int(
                detected["longest_contiguous_closed_run_frames"] >= GRIPPER_HOLD_FRAMES
            )
            lifecycle["episodes_with_close_hold_reopen"] += int(detected["complete"])
            if detected["complete"]:
                train_complete_episode_ids.append(episode_index)
        lifecycle["transition_count_per_episode"] = _summary(
            np.asarray(train_transition_counts, dtype=np.float64)[:, None], np
        )

    held_out_lifecycle: dict[str, Any] = {
        "scope": "split.held_out evaluated with train-fixed thresholds",
        "episodes": len(split["held_out"]),
        "episodes_with_close_hold_reopen": 0,
    }
    if hysteresis_supported:
        for episode_index in split["held_out"]:
            detected = continuous_gripper_lifecycle(
                eligible_arrays[episode_index]["action"][:, 7].tolist()
            )
            held_out_lifecycle["episodes_with_close_hold_reopen"] += int(
                detected["complete"]
            )

    first_episode = eligible_episodes[0]
    first_record = next(item for item in episode_records if item["episode_index"] == first_episode)
    first_table = parquet.read_table(file_by_episode[first_episode], columns=["timestamp", "observation.state", "action"])
    first_state, first_action = policy_vectors(
        first_table["observation.state"][0].as_py(),
        first_table["action"][0].as_py(),
        contract,
    )
    source_ids = set(metadata_by_episode)
    source_index_gaps = sorted(set(range(min(source_ids), max(source_ids) + 1)) - source_ids)
    expected_images = set(contract["policy_view"]["images"])
    available_images = {
        key for key, value in info["features"].items() if value.get("dtype") == "video"
    }
    metadata_total_frames = sum(int(item["length"]) for item in metadata)
    video_fps = {
        key: int(value["video_info"]["video.fps"])
        for key, value in info["features"].items()
        if value.get("dtype") == "video"
    }
    right_gripper_names = info["features"][
        "observation.state.franka_robot_right_joint_states"
    ].get("names")
    metadata_warnings = []
    if set(video_fps.values()) != {int(info["fps"])}:
        metadata_warnings.append("global_fps_differs_from_encoded_video_fps")
    if any("left" in str(name) for name in right_gripper_names or []):
        metadata_warnings.append("right_gripper_state_name_mentions_left_knuckle")
    report: dict[str, Any] = {
        "gate": "phase2_real_gate_a",
        "dataset": {
            "repo_id": contract["dataset"]["repo_id"],
            "revision": contract["dataset"]["revision"],
            "root": str(dataset_root.resolve()),
            "metadata_episodes": len(metadata),
            "parquet_files": len(paths),
            "unique_parquet_episode_indices": len(files_grouped),
            "metadata_frames": metadata_total_frames,
            "declared_episodes": int(info["total_episodes"]),
            "declared_frames": int(info["total_frames"]),
            "source_episode_index_min": min(source_ids),
            "source_episode_index_max": max(source_ids),
            "source_index_gaps_not_present_in_release": source_index_gaps,
            "source_gap_reason": "not encoded by the released metadata",
            "global_fps": int(info["fps"]),
            "encoded_video_fps": video_fps,
            "metadata_warnings": metadata_warnings,
        },
        "validity": {
            "annotation_valid_frames": valid_annotation_frames,
            "annotation_invalid_frames": invalid_annotation_frames,
            "fully_valid_eligible_episodes": len(eligible_episodes),
            "fully_valid_eligible_frames": eligible_frames,
            "excluded_episode_count": len(invalid_episodes),
            "mixed_validity_episodes": mixed_validity_episodes,
            "adapter_policy": "reject any episode that is not structurally sound and validity=1 for every frame",
            "valid_frames_discarded_by_whole_episode_policy": valid_annotation_frames - eligible_frames,
            "excluded_episodes": invalid_episodes,
        },
        "mapping": {
            "state_raw_indices": contract["policy_view"]["state"]["raw_indices"],
            "action_raw_indices": contract["policy_view"]["action"]["raw_indices"],
            "policy_state_size": int(train_state.shape[1]),
            "policy_action_size": int(train_action.shape[1]),
            "flat_right_state_matches_component_columns": not any(
                "right_state_flat_component_mismatch" in item["issue_codes"]
                for item in episode_records
            ),
            "image_keys": sorted(expected_images),
            "image_keys_present_in_metadata": expected_images <= available_images,
        },
        "train_only_statistics": {
            "scope": "normalization and staging fits must use only split.train",
            "policy_state": _summary(train_state, np),
            "policy_action": _summary(train_action, np),
            "initial_first_20_frames_policy_state": _summary(train_initial_state, np),
            "left_measured_state_and_gripper": _summary(
                np.concatenate(
                    [eligible_arrays[index]["raw_state"][:, :8] for index in split["train"]],
                    axis=0,
                ),
                np,
            ),
            "left_action": _summary(train_raw_action[:, :8], np),
            "spine_action": _summary(train_raw_action[:, 16:17], np),
            "right_gripper_raw_distribution_before_semantic_threshold": {
                **_summary(train_gripper[:, None], np),
                "unique_count": int(gripper_unique.size),
                "unique_values": (
                    gripper_unique.tolist() if gripper_unique.size <= 20 else None
                ),
                "histogram_edges": histogram_edges.tolist(),
                "histogram_counts": histogram_counts.tolist(),
                "frames_at_or_below_0_25": low_mode_count,
                "fraction_at_or_below_0_25": low_mode_fraction,
                "frames_at_or_above_0_90": high_mode_count,
                "fraction_at_or_above_0_90": high_mode_fraction,
            },
            "right_gripper_lifecycle": lifecycle,
        },
        "held_out_fixed_threshold_metrics": {
            "right_gripper_lifecycle": held_out_lifecycle,
        },
        "video_alignment": {
            "status": "not_checked_parquet_only_gate",
            "required_before_training_smoke": True,
            "training_smoke_ready": False,
        },
        "split": split,
        "policy_view_example": {
            "episode_index": first_episode,
            "frame_index": 0,
            "split": "train",
            "state": first_state,
            "action": first_action,
            "images": image_references(
                first_episode, float(first_table["timestamp"][0].as_py())
            ),
            "eligible": first_record["eligible"],
        },
        "structural_issues": structural_issues,
        "episode_records": episode_records,
    }
    report["structural_pass"] = bool(
        not structural_issues
        and len(metadata) == int(info["total_episodes"]) == len(paths) == len(files_grouped)
        and metadata_total_frames == int(info["total_frames"])
        and valid_annotation_frames + invalid_annotation_frames == metadata_total_frames
        and eligible_frames == split["train_frames"] + split["held_out_frames"]
        and report["mapping"]["policy_state_size"] == 8
        and report["mapping"]["policy_action_size"] == 8
        and report["mapping"]["flat_right_state_matches_component_columns"]
        and report["mapping"]["image_keys_present_in_metadata"]
    )
    report["parquet_training_candidate"] = bool(
        report["structural_pass"]
        and len(split["train"]) >= MIN_TRAIN_EPISODES
        and len(split["held_out"]) >= MIN_HELD_OUT_EPISODES
        and lifecycle.get("classification_allowed", False)
        and lifecycle.get("episodes_with_close_hold_reopen", 0)
        >= MIN_TRAIN_CLOSE_HOLD_REOPEN_EPISODES
        and held_out_lifecycle.get("episodes_with_close_hold_reopen", 0)
        >= MIN_HELD_OUT_CLOSE_HOLD_REOPEN_EPISODES
    )
    report["training_readiness_requirements"] = {
        "minimum_train_episodes": MIN_TRAIN_EPISODES,
        "minimum_held_out_episodes": MIN_HELD_OUT_EPISODES,
        "minimum_train_close_hold_reopen_episodes": MIN_TRAIN_CLOSE_HOLD_REOPEN_EPISODES,
        "minimum_held_out_close_hold_reopen_episodes": MIN_HELD_OUT_CLOSE_HOLD_REOPEN_EPISODES,
        "parquet_candidate_pass": report["parquet_training_candidate"],
        "video_alignment_pass": False,
        "training_ready": False,
        "recommended_train_video_episode": (
            train_complete_episode_ids[0] if train_complete_episode_ids else None
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=Path(__file__).with_name("contract.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = validate_contract(json.loads(args.contract.read_text(encoding="utf-8")))
    report = audit_dataset(args.dataset_root, contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{'STRUCTURAL_PASS' if report['structural_pass'] else 'STRUCTURAL_FAIL'}: "
        f"eligible={report['validity']['fully_valid_eligible_episodes']} episodes/"
        f"{report['validity']['fully_valid_eligible_frames']} frames; "
        f"excluded={report['validity']['excluded_episode_count']} episodes; "
        f"train={len(report['split']['train'])}, "
        f"held_out={len(report['split']['held_out'])}"
    )
    return 0 if report["structural_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
