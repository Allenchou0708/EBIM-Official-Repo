"""Materialize one real train episode as a right-only LeRobot v3 smoke view."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from task2_real.contract import validate_contract


CAMERA_KEYS = ("observation.images.head", "observation.images.wrist_right")


class SequentialNearestVideo:
    """Sample an asynchronous video at monotonically increasing timestamps."""

    def __init__(self, path: Path) -> None:
        try:
            import av
        except ImportError as error:  # pragma: no cover - exercised in Docker
            raise RuntimeError("PyAV is required to materialize the smoke dataset") from error
        self._container = av.open(str(path))
        self._stream = self._container.streams.video[0]
        self._frames = iter(self._container.decode(self._stream))
        self._previous: tuple[float, Any] | None = None
        self._current = self._next_frame()

    def _next_frame(self) -> tuple[float, Any] | None:
        for frame in self._frames:
            if frame.pts is not None:
                return float(frame.pts * self._stream.time_base), frame
        return None

    def sample(self, timestamp: float) -> tuple[Any, float]:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("video query timestamp must be finite and non-negative")
        while self._current is not None and self._current[0] < timestamp:
            self._previous = self._current
            self._current = self._next_frame()
        candidates = [item for item in (self._previous, self._current) if item is not None]
        if not candidates:
            raise ValueError("video contains no timestamped frames")
        selected_timestamp, selected_frame = min(
            candidates, key=lambda item: abs(item[0] - timestamp)
        )
        return selected_frame.to_ndarray(format="rgb24"), selected_timestamp

    @property
    def exhausted(self) -> bool:
        """Whether sampling has advanced beyond the final decoded frame."""

        return self._current is None

    def close(self) -> None:
        self._container.close()


def materialize_smoke_dataset(
    source_root: Path,
    destination_root: Path,
    contract: dict[str, Any],
    episode_index: int,
    repo_id: str,
) -> dict[str, Any]:
    try:
        import numpy as np
        import pyarrow.parquet as parquet
        from torch.utils.data import DataLoader
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:  # pragma: no cover - exercised in Docker
        raise RuntimeError("numpy, pyarrow, and LeRobot are required") from error

    validate_contract(contract)
    if destination_root.exists():
        raise ValueError(f"destination already exists: {destination_root}")
    if episode_index < 0:
        raise ValueError("episode index must be non-negative")
    split_manifest_path = contract.get("_split_manifest_path")
    if split_manifest_path is None:
        raise ValueError("caller must attach the immutable split manifest path")
    split = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    if episode_index not in {int(index) for index in split["train"]}:
        raise ValueError("smoke episode is not in the immutable train split")

    chunk = episode_index // 1000
    parquet_path = (
        source_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = parquet.read_table(
        parquet_path,
        columns=[
            "episode_index",
            "frame_index",
            "timestamp",
            "observation.state",
            "action",
            "annotation.human.validity",
        ],
    )
    rows = table.num_rows
    if {int(value) for value in table["episode_index"].to_pylist()} != {episode_index}:
        raise ValueError("source parquet episode index mismatch")
    if table["frame_index"].to_pylist() != list(range(rows)):
        raise ValueError("source parquet frame index is not contiguous")
    if {int(value) for value in table["annotation.human.validity"].to_pylist()} != {1}:
        raise ValueError("source episode is not fully valid")

    timestamps = [float(value) for value in table["timestamp"].to_pylist()]
    raw_states = table["observation.state"].to_pylist()
    raw_actions = table["action"].to_pylist()
    state_indices = [int(index) for index in contract["policy_view"]["state"]["raw_indices"]]
    action_indices = [int(index) for index in contract["policy_view"]["action"]["raw_indices"]]
    if state_indices != list(range(21, 29)) or action_indices != list(range(8, 16)):
        raise ValueError("smoke adapter requires state[21:29] and action[8:16]")

    video_paths = {
        key: source_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / key
        / f"episode_{episode_index:06d}.mp4"
        for key in CAMERA_KEYS
    }
    decoders = {key: SequentialNearestVideo(path) for key, path in video_paths.items()}
    first_samples = {key: decoder.sample(timestamps[0]) for key, decoder in decoders.items()}
    first_images = {key: sample[0] for key, sample in first_samples.items()}
    features = {
        CAMERA_KEYS[0]: {
            "dtype": "video",
            "shape": tuple(int(value) for value in first_images[CAMERA_KEYS[0]].shape),
            "names": ["height", "width", "channels"],
        },
        CAMERA_KEYS[1]: {
            "dtype": "video",
            "shape": tuple(int(value) for value in first_images[CAMERA_KEYS[1]].shape),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(contract["policy_view"]["state"]["names"]),
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(contract["policy_view"]["action"]["names"]),
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(contract["dataset"]["fps"]),
        features=features,
        root=destination_root,
        robot_type="ebim_task2_real_right_only",
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=2,
    )
    timestamp_errors: dict[str, list[float]] = {key: [] for key in CAMERA_KEYS}
    timestamp_error_limits = {
        "observation.images.head": 0.11,
        "observation.images.wrist_right": 0.04,
    }
    instruction = str(contract["policy_view"]["instruction"])
    try:
        for row_index, timestamp in enumerate(timestamps):
            images: dict[str, Any] = {}
            for key, decoder in decoders.items():
                if row_index == 0:
                    image = first_images[key]
                    selected_timestamp = first_samples[key][1]
                else:
                    image, selected_timestamp = decoder.sample(timestamp)
                images[key] = np.asarray(image, dtype=np.uint8)
                timestamp_errors[key].append(abs(float(selected_timestamp) - timestamp))
            state = np.asarray(
                [raw_states[row_index][index] for index in state_indices], dtype=np.float32
            )
            action = np.asarray(
                [raw_actions[row_index][index] for index in action_indices], dtype=np.float32
            )
            dataset.add_frame(
                {
                    **images,
                    "observation.state": state,
                    "action": action,
                    "task": instruction,
                }
            )
        dataset.save_episode()
        dataset.finalize()
    finally:
        for decoder in decoders.values():
            decoder.close()

    maximum_timestamp_errors = {
        key: max(errors) for key, errors in timestamp_errors.items()
    }
    for key, maximum_error in maximum_timestamp_errors.items():
        if maximum_error > timestamp_error_limits[key]:
            raise ValueError(
                f"{key} nearest timestamp error {maximum_error:.6f}s exceeds "
                f"{timestamp_error_limits[key]:.6f}s"
            )

    provenance = {
        "source_repo_id": contract["dataset"]["repo_id"],
        "source_revision": contract["dataset"]["revision"],
        "source_episode_index": episode_index,
        "source_policy_state_raw_indices": state_indices,
        "source_policy_action_raw_indices": action_indices,
        "source_timestamp_sampling": "nearest frame independently per camera using each parquet timestamp",
    }
    provenance_path = destination_root / "meta" / "ebim_source.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    loaded = LeRobotDataset(
        repo_id=repo_id,
        root=destination_root,
        episodes=[0],
        download_videos=False,
        video_backend="pyav",
    )
    if len(loaded) != rows:
        raise ValueError(f"materialized dataset length is {len(loaded)}, expected {rows}")
    selected_rows = [0, rows // 2, rows - 1]
    selected_samples = [loaded[index] for index in selected_rows]
    sample = selected_samples[0]
    camera_keys = sorted(key for key in sample if key.startswith("observation.images."))
    if camera_keys != sorted(CAMERA_KEYS):
        raise ValueError(f"materialized camera keys differ from right-only contract: {camera_keys}")
    for row_index, selected_sample in zip(selected_rows, selected_samples, strict=True):
        if tuple(selected_sample["observation.state"].shape) != (8,):
            raise ValueError("materialized state is not 8-D")
        if tuple(selected_sample["action"].shape) != (8,):
            raise ValueError("materialized action is not 8-D")
        if not bool(np.all(np.isfinite(selected_sample["observation.state"].numpy()))):
            raise ValueError("materialized state contains non-finite values")
        if not bool(np.all(np.isfinite(selected_sample["action"].numpy()))):
            raise ValueError("materialized action contains non-finite values")
        if selected_sample.get("task") != instruction:
            raise ValueError("materialized task instruction changed")
        for key in CAMERA_KEYS:
            image_tensor = selected_sample[key]
            if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
                raise ValueError(f"materialized image is not CHW RGB: {key}")
            if not bool(np.all(np.isfinite(image_tensor.numpy()))):
                raise ValueError(f"materialized image contains non-finite values: {key}")
        if not np.allclose(
            selected_sample["observation.state"].numpy(),
            np.asarray(raw_states[row_index])[21:29],
        ):
            raise ValueError("materialized state does not match raw state[21:29]")
        if not np.allclose(
            selected_sample["action"].numpy(),
            np.asarray(raw_actions[row_index])[8:16],
        ):
            raise ValueError("materialized action does not match raw action[8:16]")

    batch = next(iter(DataLoader(loaded, batch_size=2, shuffle=False, num_workers=0)))
    if tuple(batch["observation.state"].shape) != (2, 8):
        raise ValueError("LeRobot batch state is not (2, 8)")
    if tuple(batch["action"].shape) != (2, 8):
        raise ValueError("LeRobot batch action is not (2, 8)")
    if list(batch.get("task", [])) != [instruction, instruction]:
        raise ValueError("LeRobot batch task instruction is unavailable")
    for key in CAMERA_KEYS:
        image_batch = batch[key]
        if image_batch.ndim != 4 or image_batch.shape[0] != 2 or image_batch.shape[1] != 3:
            raise ValueError(f"LeRobot image batch is not (2, 3, H, W): {key}")
        if not bool(np.all(np.isfinite(image_batch.numpy()))):
            raise ValueError(f"LeRobot image batch contains non-finite values: {key}")
    declared_names = [
        str(name).lower()
        for key in ("observation.state", "action")
        for name in loaded.meta.features[key].get("names") or []
    ]
    if any("left" in name or "spine" in name or "base" in name for name in declared_names):
        raise ValueError("derived numeric schema contains a forbidden owner name")
    if loaded.meta.features["observation.state"].get("names") != list(
        contract["policy_view"]["state"]["names"]
    ):
        raise ValueError("derived state feature names differ from the right-only contract")
    if loaded.meta.features["action"].get("names") != list(
        contract["policy_view"]["action"]["names"]
    ):
        raise ValueError("derived action feature names differ from the right-only contract")
    if any(
        "left" in key.lower() or "spine" in key.lower()
        for key in loaded.meta.features
    ):
        raise ValueError("derived feature keys contain a forbidden owner")

    return {
        "source_dataset_revision": contract["dataset"]["revision"],
        "source_episode_index": episode_index,
        "source_parquet_frames": rows,
        "source_timestamp_first": timestamps[0],
        "source_timestamp_last": timestamps[-1],
        "immutable_split_manifest": str(Path(split_manifest_path).resolve()),
        "derived_provenance": str(provenance_path.resolve()),
        "destination_root": str(destination_root.resolve()),
        "destination_repo_id": repo_id,
        "policy_camera_keys": list(CAMERA_KEYS),
        "policy_state_raw_indices": state_indices,
        "policy_action_raw_indices": action_indices,
        "policy_state_size": 8,
        "policy_action_size": 8,
        "raw_action_index_16_excluded": 16 not in action_indices,
        "decoded_shapes_hwc": {
            key: list(features[key]["shape"]) for key in CAMERA_KEYS
        },
        "maximum_source_timestamp_error_s": maximum_timestamp_errors,
        "maximum_source_timestamp_error_limit_s": timestamp_error_limits,
        "lerobot_readback_sample_keys": sorted(sample.keys()),
        "lerobot_readback_length": len(loaded),
        "lerobot_readback_rows_checked": selected_rows,
        "lerobot_batch_shapes": {
            "observation.state": list(batch["observation.state"].shape),
            "action": list(batch["action"].shape),
            **{key: list(batch[key].shape) for key in CAMERA_KEYS},
        },
        "lerobot_readback_pass": True,
        "ready_for_one_step_smoke": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/phase2_real_right_smoke")
    parser.add_argument(
        "--contract", type=Path, default=Path(__file__).with_name("contract.json")
    )
    args = parser.parse_args()
    contract = validate_contract(json.loads(args.contract.read_text(encoding="utf-8")))
    contract["_split_manifest_path"] = str(args.split_manifest.resolve())
    report = materialize_smoke_dataset(
        args.source_root.resolve(),
        args.destination_root.resolve(),
        contract,
        args.episode,
        args.repo_id,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"PASS: right-only LeRobot readback; episode={args.episode} "
        f"frames={report['source_parquet_frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
