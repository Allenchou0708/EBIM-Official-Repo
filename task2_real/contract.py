"""Validate the Phase II real-robot policy and organizer metadata boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _shape(feature: dict[str, Any]) -> int:
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"expected one-dimensional feature shape, got {shape}")
    return int(shape[0])


def validate_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Reject ownership overlap or an ambiguous raw-to-policy mapping."""

    if payload.get("schema_version") != 1:
        raise ValueError("real-robot contract must use schema version 1")

    dataset = payload["dataset"]
    revision = str(dataset["revision"])
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("dataset revision must be a 40-character lowercase SHA")

    view = payload["policy_view"]
    for key, raw_size_key in (("state", "raw_state_size"), ("action", "raw_action_size")):
        item = view[key]
        indices = [int(value) for value in item["raw_indices"]]
        names = list(item["names"])
        if int(item["size"]) != len(indices) or len(indices) != len(names):
            raise ValueError(f"{key} size, indices, and names must agree")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{key} raw indices must be unique")
        raw_size = int(dataset[raw_size_key])
        if any(index < 0 or index >= raw_size for index in indices):
            raise ValueError(f"{key} raw index is outside 0..{raw_size - 1}")

    ownership = payload["ownership"]
    deterministic = set(ownership["deterministic"])
    learned = set(ownership["pi05"])
    forbidden = set(ownership["pi05_forbidden"])
    if deterministic & learned:
        raise ValueError("deterministic and PI0.5 command ownership overlap")
    if not deterministic <= forbidden:
        raise ValueError("every deterministic group must be forbidden to PI0.5")
    if learned & forbidden:
        raise ValueError("PI0.5-owned groups cannot also be forbidden")
    if learned != {"right_arm", "right_gripper"}:
        raise ValueError("Phase II V1 policy must remain right-arm-only")
    return payload


def audit_official_metadata(
    info: dict[str, Any], modality: dict[str, Any]
) -> list[str]:
    """Return stable issue codes for the released Munich metadata."""

    features = info["features"]
    info_state = _shape(features["observation.state"])
    info_action = _shape(features["action"])
    modality_state_end = max(int(item["end"]) for item in modality["state"].values())
    modality_action_end = max(int(item["end"]) for item in modality["action"].values())

    issues: list[str] = []
    if info_state != modality_state_end:
        issues.append(f"info_state_{info_state}_vs_modality_end_{modality_state_end}")
    if info_action != modality_action_end:
        issues.append(f"info_action_{info_action}_vs_modality_end_{modality_action_end}")

    action_names = list(features["action"].get("names") or [])
    state_names = list(features["observation.state"].get("names") or [])
    if any("spine" in name for name in action_names) and not any(
        "spine" in name for name in state_names
    ):
        issues.append("spine_action_without_spine_state")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("contract.json"),
    )
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--modality", type=Path, required=True)
    parser.add_argument("--acknowledge-documented-conflicts", action="store_true")
    args = parser.parse_args()

    contract = validate_contract(json.loads(args.contract.read_text()))
    info = json.loads(args.info.read_text())
    modality = json.loads(args.modality.read_text())
    issues = audit_official_metadata(info, modality)
    documented = list(contract["dataset"]["documented_metadata_conflicts"])
    report = {
        "dataset_revision": contract["dataset"]["revision"],
        "issues": issues,
        "documented_conflicts": documented,
        "adapter_mapping_valid": issues == documented,
        "acknowledged": bool(args.acknowledge_documented_conflicts),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if issues and not args.acknowledge_documented_conflicts:
        return 2
    return 0 if issues == documented else 3


if __name__ == "__main__":
    raise SystemExit(main())
