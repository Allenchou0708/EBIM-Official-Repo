"""Audit end-of-stream holds and source resolutions for a frozen split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from task2_real.full_dataset import CAMERA_KEYS, load_frozen_split
from task2_real.contract import validate_contract


def _video_report(path: Path, parquet_timestamps: list[float]) -> dict[str, Any]:
    try:
        import av
    except ImportError as error:  # pragma: no cover - exercised in Docker
        raise RuntimeError("PyAV is required for the full video audit") from error

    decoded_timestamps: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        codec = str(stream.codec_context.name)
        for frame in container.decode(stream):
            if frame.pts is not None:
                decoded_timestamps.append(float(frame.pts * stream.time_base))
    if not decoded_timestamps:
        raise ValueError(f"video has no timestamped frames: {path}")
    first_timestamp = decoded_timestamps[0]
    last_timestamp = decoded_timestamps[-1]
    tail_rows = [index for index, value in enumerate(parquet_timestamps) if value > last_timestamp]
    head_rows = [index for index, value in enumerate(parquet_timestamps) if value < first_timestamp]
    return {
        "path": str(path),
        "codec": codec,
        "width": width,
        "height": height,
        "decoded_frames": len(decoded_timestamps),
        "first_decoded_timestamp_s": first_timestamp,
        "last_decoded_timestamp_s": last_timestamp,
        "parquet_first_timestamp_s": parquet_timestamps[0],
        "parquet_last_timestamp_s": parquet_timestamps[-1],
        "pre_stream_hold_rows": len(head_rows),
        "pre_stream_first_row": head_rows[0] if head_rows else None,
        "end_of_stream_hold_rows": len(tail_rows),
        "end_of_stream_first_row": tail_rows[0] if tail_rows else None,
        "end_of_stream_gap_s": max(0.0, parquet_timestamps[-1] - last_timestamp),
    }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def audit(
    dataset_root: Path,
    split_manifest_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - exercised in Docker
        raise RuntimeError("pyarrow is required for the full video audit") from error

    split = load_frozen_split(split_manifest_path, contract)
    records: list[dict[str, Any]] = []
    for split_name in ("train", "held_out"):
        for episode_index in split[split_name]:
            chunk = int(episode_index) // 1000
            parquet_path = (
                dataset_root
                / "data"
                / f"chunk-{chunk:03d}"
                / f"episode_{int(episode_index):06d}.parquet"
            )
            table = parquet.read_table(parquet_path, columns=["timestamp"])
            timestamps = [float(value) for value in table["timestamp"].to_pylist()]
            cameras = {}
            for key in CAMERA_KEYS:
                path = (
                    dataset_root
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / key
                    / f"episode_{int(episode_index):06d}.mp4"
                )
                cameras[key] = _video_report(path, timestamps)
            records.append(
                {
                    "split": split_name,
                    "episode_index": int(episode_index),
                    "parquet_rows": table.num_rows,
                    "cameras": cameras,
                }
            )
            print(
                f"audited {len(records)}/149 split={split_name} episode={episode_index}",
                flush=True,
            )

    aggregate: dict[str, Any] = {}
    for key in CAMERA_KEYS:
        reports = [record["cameras"][key] for record in records]
        gaps = [float(report["end_of_stream_gap_s"]) for report in reports]
        resolutions = Counter((report["height"], report["width"]) for report in reports)
        aggregate[key] = {
            "episodes": len(reports),
            "episodes_with_end_of_stream_hold": sum(
                int(report["end_of_stream_hold_rows"] > 0) for report in reports
            ),
            "total_end_of_stream_hold_rows": sum(
                int(report["end_of_stream_hold_rows"]) for report in reports
            ),
            "maximum_end_of_stream_hold_rows": max(
                int(report["end_of_stream_hold_rows"]) for report in reports
            ),
            "maximum_end_of_stream_gap_s": max(gaps),
            "end_of_stream_gap_quantiles_s": {
                "q50": _quantile(gaps, 0.50),
                "q90": _quantile(gaps, 0.90),
                "q95": _quantile(gaps, 0.95),
                "q99": _quantile(gaps, 0.99),
            },
            "episodes_with_pre_stream_hold": sum(
                int(report["pre_stream_hold_rows"] > 0) for report in reports
            ),
            "total_pre_stream_hold_rows": sum(
                int(report["pre_stream_hold_rows"]) for report in reports
            ),
            "source_resolution_episode_counts_hxw": {
                f"{height}x{width}": count
                for (height, width), count in sorted(resolutions.items())
            },
        }
    return {
        "source_repo_id": contract["dataset"]["repo_id"],
        "source_revision": contract["dataset"]["revision"],
        "split_manifest": str(split_manifest_path.resolve()),
        "episodes": len(records),
        "episode_records": records,
        "aggregate": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=Path(__file__).with_name("contract.json")
    )
    args = parser.parse_args()
    contract = validate_contract(json.loads(args.contract.read_text(encoding="utf-8")))
    result = audit(args.dataset_root, args.split_manifest, contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
