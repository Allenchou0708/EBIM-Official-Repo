#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare a zero-copy LeRobot dataset view and fixed ACT 180/20 split."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    ACTION_SIZE,
    DATASET_REPO_ID,
    DATASET_REVISION,
    EVALUATOR_CAMERA_KEY,
    SPLIT_SEED,
    STATE_SIZE,
    contract_manifest,
    deterministic_episode_split,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_info(source: Path) -> dict[str, Any]:
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("total_episodes", -1)) != 200:
        raise ValueError(f"expected 200 episodes, got {info.get('total_episodes')}")
    features = info.get("features", {})
    if list(features.get("action", {}).get("shape", [])) != [ACTION_SIZE]:
        raise ValueError("dataset action feature is not Task 2 20-D")
    if list(features.get("observation.state", {}).get("shape", [])) != [STATE_SIZE]:
        raise ValueError("dataset state feature is not Task 2 37-D")
    return info


def _link_dataset(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("data", "videos", "task2_extras"):
        target = source / name
        if not target.exists():
            if name == "task2_extras":
                continue
            raise FileNotFoundError(target)
        link = output / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"existing symlink points at another dataset: {link}")
            continue
        if link.exists():
            raise FileExistsError(f"refusing to replace existing path: {link}")
        link.symlink_to(target.resolve(), target_is_directory=True)

    # Metadata is small, so keep an ACT-specific copy that excludes the static
    # evaluator camera. The videos remain zero-copy. Excluding this key prevents
    # both accidental evaluator leakage and needless decoding during training.
    meta = output / "meta"
    if not meta.exists():
        shutil.copytree(source / "meta", meta)
    info_path = meta / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.get("features", {}).pop(EVALUATOR_CAMERA_KEY, None)
    _write_json(info_path, info)
    stats_path = meta / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats.pop(EVALUATOR_CAMERA_KEY, None)
        _write_json(stats_path, stats)


def _vector_stats(source: Path, train_episodes: set[int]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to prepare ACT normalization stats") from error

    values: dict[str, list[np.ndarray]] = {"action": [], "observation.state": []}
    for parquet_path in sorted((source / "data").glob("**/*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=["episode_index", "action", "observation.state"],
        )
        episodes = np.asarray(table["episode_index"].to_numpy())
        mask = np.isin(episodes, np.fromiter(train_episodes, dtype=np.int64))
        if not mask.any():
            continue
        for key in values:
            rows = np.stack(table[key].to_pylist()).astype(np.float64, copy=False)
            values[key].append(rows[mask])

    result: dict[str, Any] = {}
    expected = {"action": ACTION_SIZE, "observation.state": STATE_SIZE}
    for key, parts in values.items():
        if not parts:
            raise ValueError(f"no training rows found for {key}")
        array = np.concatenate(parts, axis=0)
        if array.ndim != 2 or array.shape[1] != expected[key]:
            raise ValueError(f"invalid {key} shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite values found in training {key}")
        result[key] = {
            "count": int(array.shape[0]),
            "mean": array.mean(axis=0).tolist(),
            "std": np.maximum(array.std(axis=0), 1e-6).tolist(),
            "min": array.min(axis=0).tolist(),
            "max": array.max(axis=0).tolist(),
        }
    return result


def prepare(source: Path, output: Path, *, seed: int = SPLIT_SEED) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("ACT view must not overwrite the source dataset")
    info = _load_info(source)
    train, validation = deterministic_episode_split(seed=seed)
    _link_dataset(source, output)
    stats = _vector_stats(source, set(train))
    stats_path = output / "act_train_stats.json"
    _write_json(stats_path, stats)
    manifest = {
        "schema_version": 1,
        "source_dataset": str(source),
        "dataset_root": str(output),
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "source_total_frames": int(info.get("total_frames", 0)),
        "split_seed": seed,
        "train_episodes": train,
        "validation_episodes": validation,
        "train_stats": str(stats_path),
        **contract_manifest(),
    }
    _write_json(output / "act_split.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    try:
        manifest = prepare(args.source, args.output, seed=args.seed)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}")
        return 2
    print(
        f"PASS: ACT dataset view={manifest['dataset_root']} "
        f"train={len(manifest['train_episodes'])} "
        f"validation={len(manifest['validation_episodes'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
