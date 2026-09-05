#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fit a wrist-image grasp envelope on development episodes only.

This is an offline audit.  Episode landmarks come from the frozen
multi-episode audit; held-out records are evaluated only after the envelope is
fully determined from the development split.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq


VIDEO_KEY = "observation.images.wrist_right"
METRICS = ("centroid_u_fraction", "centroid_v_fraction", "log_area_fraction")


def blue_pad_mask(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    return (blue >= 90) & (blue - red >= 35) & (blue - green >= 15)


def image_signature(rgb: np.ndarray) -> dict[str, float | int]:
    mask = blue_pad_mask(rgb)
    rows, columns = np.nonzero(mask)
    if len(rows) < 100:
        raise ValueError(f"insufficient blue pad pixels: {len(rows)}")
    height, width = mask.shape
    area_fraction = float(len(rows) / mask.size)
    return {
        "pixel_count": int(len(rows)),
        "centroid_u_fraction": float(np.median(columns) / width),
        "centroid_v_fraction": float(np.median(rows) / height),
        "area_fraction": area_fraction,
        "log_area_fraction": math.log(area_fraction),
        "bbox_width_fraction": float((columns.max() - columns.min() + 1) / width),
        "bbox_height_fraction": float((rows.max() - rows.min() + 1) / height),
    }


def decode_nearest(video_path: Path, timestamp_s: float) -> tuple[np.ndarray, float]:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        seek_s = max(0.0, timestamp_s - 0.5)
        container.seek(int(seek_s / time_base), stream=stream, backward=True)
        best_frame = None
        best_time = math.nan
        best_error = math.inf
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            error = abs(frame_time - timestamp_s)
            if error < best_error:
                best_frame = frame.to_ndarray(format="rgb24")
                best_time = frame_time
                best_error = error
            if frame_time > timestamp_s and best_frame is not None:
                break
        if best_frame is None or best_error > 0.04:
            raise RuntimeError(
                f"no wrist frame within 40 ms at {video_path}:{timestamp_s:.6f}"
            )
        return best_frame, best_time


def expanded_quantile_envelope(
    records: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Predeclared robust rule: dev q01..q99, expanded by 20% of its span."""

    envelope: dict[str, list[float]] = {}
    for metric in METRICS:
        values = np.asarray([record[metric] for record in records], dtype=np.float64)
        lower, upper = np.quantile(values, (0.01, 0.99))
        margin = max(1.0e-6, 0.20 * float(upper - lower))
        envelope[metric] = [float(lower - margin), float(upper + margin)]
    return envelope


def within_envelope(record: dict[str, Any], envelope: dict[str, list[float]]) -> bool:
    return all(
        envelope[metric][0] <= float(record[metric]) <= envelope[metric][1]
        for metric in METRICS
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--landmark-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen = json.loads(args.landmark_audit.read_text())
    episode_records = {
        int(record["episode_index"]): record
        for record in frozen["episode_records"]
        if record.get("dataset") == "task2_fixpos_200_46ab41f"
        or int(record["episode_index"]) < 200
    }
    if len(episode_records) != 200:
        raise RuntimeError(f"expected 200 primary episodes, got {len(episode_records)}")

    metadata_path = args.dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    columns = [
        "episode_index",
        f"videos/{VIDEO_KEY}/chunk_index",
        f"videos/{VIDEO_KEY}/file_index",
        f"videos/{VIDEO_KEY}/from_timestamp",
    ]
    metadata = {
        int(row["episode_index"]): row
        for row in pq.read_table(metadata_path, columns=columns).to_pylist()
    }

    requests: dict[Path, list[tuple[int, float, str]]] = defaultdict(list)
    for episode_index, record in sorted(episode_records.items()):
        row = metadata[episode_index]
        grasp_frame = int(record["phase_frames"]["grasp"])
        timestamp_s = float(row[f"videos/{VIDEO_KEY}/from_timestamp"]) + grasp_frame / 30.0
        video_path = args.dataset_root / (
            f"videos/{VIDEO_KEY}/chunk-"
            f"{int(row[f'videos/{VIDEO_KEY}/chunk_index']):03d}/file-"
            f"{int(row[f'videos/{VIDEO_KEY}/file_index']):03d}.mp4"
        )
        requests[video_path].append((episode_index, timestamp_s, str(record["split"])))

    signatures: list[dict[str, Any]] = []
    for video_path, video_requests in sorted(requests.items(), key=lambda item: str(item[0])):
        for episode_index, requested_s, split in video_requests:
            image, decoded_s = decode_nearest(video_path, requested_s)
            signatures.append(
                {
                    "episode_index": episode_index,
                    "split": split,
                    "requested_timestamp_s": requested_s,
                    "decoded_timestamp_s": decoded_s,
                    "timestamp_error_s": abs(decoded_s - requested_s),
                    **image_signature(image),
                }
            )

    development = [record for record in signatures if record["split"] == "development"]
    held_out = [record for record in signatures if record["split"] == "held_out"]
    if (len(development), len(held_out)) != (180, 20):
        raise RuntimeError(
            f"unexpected frozen split: development={len(development)}, held_out={len(held_out)}"
        )
    envelope = expanded_quantile_envelope(development)
    for record in signatures:
        record["passes_development_envelope"] = within_envelope(record, envelope)

    def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        passing = sum(bool(record["passes_development_envelope"]) for record in records)
        return {
            "episodes": len(records),
            "passing": passing,
            "coverage": passing / len(records),
            "failures": [
                int(record["episode_index"])
                for record in records
                if not record["passes_development_envelope"]
            ],
            "metrics": {
                metric: {
                    "minimum": float(min(record[metric] for record in records)),
                    "median": float(np.median([record[metric] for record in records])),
                    "maximum": float(max(record[metric] for record in records)),
                }
                for metric in METRICS
            },
        }

    result = {
        "schema_version": 1,
        "policy_input": "right_wrist_rgb_only",
        "landmark": "first_grasp_frame",
        "split_contract": "source_episode_index_mod_10_equals_7_is_held_out",
        "fit_rule": "development q01..q99 expanded by 20 percent of q-span",
        "minimum_held_out_coverage": 0.90,
        "envelope": envelope,
        "development": split_summary(development),
        "held_out": split_summary(held_out),
        "held_out_gate_pass": sum(
            bool(record["passes_development_envelope"]) for record in held_out
        )
        / len(held_out)
        >= 0.90,
        "maximum_timestamp_error_s": max(record["timestamp_error_s"] for record in signatures),
        "episode_records": sorted(signatures, key=lambda record: record["episode_index"]),
        "limitations": [
            "all 200 source episodes are successful fixed-position simulator demonstrations",
            "held-out coverage tests episode variation, not unseen task geometry or real-robot transfer",
            "color masking is simulator-specific and must be recalibrated for real camera imagery",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("envelope", "development", "held_out", "held_out_gate_pass", "maximum_timestamp_error_s")}, indent=2))
    return 0 if result["held_out_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
