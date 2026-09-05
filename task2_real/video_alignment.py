"""Verify timestamp-based decoding for one train episode and both policy cameras."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


VIDEO_KEYS = ("observation.images.head", "observation.images.wrist_right")


def nearest_timestamp_index(timestamps: Sequence[float], target: float) -> int:
    if not timestamps:
        raise ValueError("decoded video has no frame timestamps")
    if not math.isfinite(float(target)):
        raise ValueError("target timestamp must be finite")
    return min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - target))


def _decode_video(path: Path, targets: list[float]) -> dict[str, Any]:
    try:
        import av
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised in Docker
        raise RuntimeError("PyAV and numpy are required for video alignment") from error

    frame_timestamps: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_timestamps.append(float(frame.pts * stream.time_base))
        codec = str(stream.codec_context.name)
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)

    selected_indices = [nearest_timestamp_index(frame_timestamps, target) for target in targets]
    selected_set = set(selected_indices)
    decoded_means: dict[int, float] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        decoded_index = -1
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            decoded_index += 1
            if decoded_index in selected_set:
                decoded_means[decoded_index] = float(
                    np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.float64).mean()
                )

    duration = frame_timestamps[-1] - frame_timestamps[0] if len(frame_timestamps) > 1 else 0.0
    actual_fps = (len(frame_timestamps) - 1) / duration if duration > 0 else 0.0
    allowed_error = 1.0 / actual_fps + 0.005 if actual_fps > 0 else 0.0
    samples = []
    for label, target, frame_index in zip(("first", "middle", "last"), targets, selected_indices):
        decoded_timestamp = frame_timestamps[frame_index]
        samples.append(
            {
                "label": label,
                "parquet_timestamp": target,
                "decoded_frame_index": frame_index,
                "decoded_timestamp": decoded_timestamp,
                "absolute_timestamp_error_s": abs(decoded_timestamp - target),
                "rgb_mean": decoded_means[frame_index],
            }
        )
    return {
        "path": str(path),
        "codec": codec,
        "decoded_width": width,
        "decoded_height": height,
        "decoded_frames": len(frame_timestamps),
        "first_decoded_timestamp": frame_timestamps[0],
        "last_decoded_timestamp": frame_timestamps[-1],
        "actual_average_fps": actual_fps,
        "allowed_nearest_timestamp_error_s": allowed_error,
        "samples": samples,
        "decode_alignment_pass": bool(
            all(sample["absolute_timestamp_error_s"] <= allowed_error for sample in samples)
            and all(math.isfinite(sample["rgb_mean"]) for sample in samples)
        ),
    }


def audit_video_alignment(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - exercised in Docker
        raise RuntimeError("pyarrow is required for video alignment") from error

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    chunk = episode_index // 1000
    parquet_path = (
        dataset_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = parquet.read_table(
        parquet_path,
        columns=["episode_index", "frame_index", "timestamp", "annotation.human.validity"],
    )
    episode_values = {int(value) for value in table["episode_index"].to_pylist()}
    validity_values = {int(value) for value in table["annotation.human.validity"].to_pylist()}
    frame_indices = table["frame_index"].to_pylist()
    timestamps = [float(value) for value in table["timestamp"].to_pylist()]
    if episode_values != {episode_index}:
        raise ValueError("parquet episode index does not match requested episode")
    if validity_values != {1}:
        raise ValueError("video alignment episode must be fully valid")
    if frame_indices != list(range(table.num_rows)):
        raise ValueError("parquet frame indices are not contiguous")
    selected_rows = [0, table.num_rows // 2, table.num_rows - 1]
    targets = [timestamps[index] for index in selected_rows]

    camera_reports: dict[str, Any] = {}
    warnings: list[str] = []
    for key in VIDEO_KEYS:
        video_path = (
            dataset_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / key
            / f"episode_{episode_index:06d}.mp4"
        )
        report = _decode_video(video_path, targets)
        feature = info["features"][key]
        expected_height, expected_width, _ = feature["shape"]
        report.update(
            {
                "metadata_width": int(expected_width),
                "metadata_height": int(expected_height),
                "metadata_video_fps": int(feature["video_info"]["video.fps"]),
            }
        )
        if (report["decoded_height"], report["decoded_width"]) != (
            int(expected_height),
            int(expected_width),
        ):
            warnings.append(f"{key}:decoded_shape_differs_from_metadata")
        if abs(report["actual_average_fps"] - int(feature["video_info"]["video.fps"])) > 1.0:
            warnings.append(f"{key}:decoded_fps_differs_from_metadata")
        camera_reports[key] = report

    report = {
        "gate": "phase2_real_video_alignment",
        "episode_index": episode_index,
        "episode_split": "train",
        "parquet_frames": table.num_rows,
        "parquet_global_fps": int(info["fps"]),
        "selected_parquet_rows": selected_rows,
        "selected_parquet_timestamps": targets,
        "cameras": camera_reports,
        "warnings": warnings,
    }
    report["alignment_pass"] = all(
        camera["decode_alignment_pass"] for camera in camera_reports.values()
    )
    report["adapter_decision"] = (
        "use parquet timestamps to sample each asynchronous camera; resize decoded RGB in the PI0.5 processor"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_video_alignment(args.dataset_root, args.episode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{'PASS' if report['alignment_pass'] else 'FAIL'}: "
        f"episode={report['episode_index']} cameras={len(report['cameras'])} "
        f"warnings={len(report['warnings'])}"
    )
    return 0 if report["alignment_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
