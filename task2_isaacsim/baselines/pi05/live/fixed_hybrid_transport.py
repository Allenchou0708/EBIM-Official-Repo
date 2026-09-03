#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Finish a latched right-hand grasp with a measured RMPflow waypoint plan.

The VLA owns approach and grasp.  This controller starts only after the live
runner confirms a legal close inside the development-derived grasp region and
hands the current end-effector pose to RMPflow.  It subscribes only to robot
state and never to task objects, evaluator output, or simulator ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path
from typing import Any

import rclpy

from task2_isaacsim.baselines.pi05.live.fixed_stage_observation import (
    ObservationStager,
    load_reference,
    minimum_jerk_fraction,
    transition_duration_s,
)
from task2_isaacsim.baselines.pi05.live.ground_truth_pregrasp import (
    _orientation_error_deg,
    _position_error,
    _slerp,
)
from task2_isaacsim.common.state_contract import (
    LEFT_GRIPPER_DRIVER,
    RIGHT_GRIPPER_DRIVER,
    SPINE_JOINT,
    gripper_open_fraction,
    resolve_joint,
)

REQUIRED_LANDMARKS = ("peel_lift", "transfer", "place", "release")


def load_hybrid_reference(path: Path) -> dict[str, Any]:
    reference = load_reference(path)
    landmarks = reference.get("right_hybrid_landmarks_world_xyzw")
    if not isinstance(landmarks, dict):
        raise ValueError("hybrid reference landmarks are missing")
    for name in REQUIRED_LANDMARKS:
        values = landmarks.get(name)
        if not isinstance(values, list) or len(values) != 7:
            raise ValueError(f"invalid hybrid landmark: {name}")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite hybrid landmark: {name}")
    derivation = reference.get("right_hybrid_landmarks_derivation", {})
    if int(derivation.get("support_unique_episodes", 0)) < 20:
        raise ValueError("hybrid landmarks have insufficient episode support")
    return reference


def build_transport_plan(
    reference: dict[str, Any], initial_right: tuple[float, ...]
) -> list[dict[str, Any]]:
    """Return the fixed-layout plan; retain and clearance poses are derived."""

    landmarks = reference["right_hybrid_landmarks_world_xyzw"]
    peel = tuple(float(value) for value in landmarks["peel_lift"])
    transfer = tuple(float(value) for value in landmarks["transfer"])
    place = tuple(float(value) for value in landmarks["place"])
    release = tuple(float(value) for value in landmarks["release"])
    preplace = (place[0], place[1], place[2] + 0.08, *place[3:])
    retreat = (release[0], release[1], release[2] + 0.08, *release[3:])
    return [
        {"name": "retain", "pose": initial_right, "right_open": 0.0},
        {"name": "peel_lift", "pose": peel, "right_open": 0.0},
        {"name": "transfer", "pose": transfer, "right_open": 0.0},
        {"name": "preplace", "pose": preplace, "right_open": 0.0},
        {"name": "place", "pose": place, "right_open": 0.0},
        {"name": "release", "pose": release, "right_open": 1.0},
        {"name": "retreat", "pose": retreat, "right_open": 1.0},
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration-s", type=float, default=180.0)
    parser.add_argument("--maximum-skew-s", type=float, default=0.10)
    parser.add_argument("--max-linear-speed-m-s", type=float, default=0.10)
    parser.add_argument("--max-angular-speed-deg-s", type=float, default=35.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.025)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=8.0)
    parser.add_argument("--stable-dwell-s", type=float, default=0.30)
    parser.add_argument("--settle-max-joint-speed-rad-s", type=float, default=0.25)
    parser.add_argument("--base-position-tolerance-m", type=float, default=0.03)
    parser.add_argument("--base-yaw-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--spine-tolerance-m", type=float, default=0.03)
    return parser


def _base_errors(
    actual: tuple[float, float, float], target: tuple[float, float, float]
) -> dict[str, float]:
    return {
        "position_m": math.hypot(actual[0] - target[0], actual[1] - target[1]),
        "yaw_rad": abs(
            math.atan2(
                math.sin(actual[2] - target[2]),
                math.cos(actual[2] - target[2]),
            )
        ),
    }


def main() -> int:  # noqa: C901 - one bounded measured transport state machine
    args = build_parser().parse_args()
    reference = load_hybrid_reference(args.reference)
    base_target = tuple(float(value) for value in reference["base_xyyaw"])
    left_target = tuple(
        float(value) for value in reference["left_safe_ee_world_xyzw"]
    )
    spine_target = float(reference["spine_command_m"])

    rclpy.init()
    node = ObservationStager()
    signal.signal(signal.SIGINT, lambda *_: setattr(node, "stop_requested", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True))
    wall_started = time.monotonic()
    last_sim_time: float | None = None
    previous_joint_sample: tuple[float, tuple[float, ...]] | None = None
    plan: list[dict[str, Any]] | None = None
    stage_index = 0
    stage_initial: dict[str, tuple[float, ...]] | None = None
    stage_started: float | None = None
    stage_duration = 0.0
    stable_since: float | None = None
    handoff_started_sim: float | None = None
    handoff_close_stable_since: float | None = None
    handoff_right_pose: tuple[float, ...] | None = None
    stages: list[dict[str, Any]] = []
    reason = "host_watchdog_timeout"
    base_errors: dict[str, float] = {}
    final_grippers: dict[str, float] = {}

    try:
        while (
            not node.stop_requested
            and time.monotonic() - wall_started < args.max_duration_s
        ):
            rclpy.spin_once(node, timeout_sec=0.02)
            if not node.fresh(args.maximum_skew_s) or not node.subscribers_ready():
                continue
            conflicts = node.conflicts()
            if any(conflicts.values()):
                reason = f"publisher_contention:{conflicts}"
                break
            assert node.sim_time is not None and node.base is not None
            if last_sim_time is not None and node.sim_time < last_sim_time:
                reason = "simulator_clock_reset"
                break
            if last_sim_time is not None and node.sim_time == last_sim_time:
                continue
            last_sim_time = node.sim_time

            positions = node.arm_positions()
            sample_speed = 0.0
            if previous_joint_sample is not None:
                dt = node.sim_time - previous_joint_sample[0]
                if dt > 1.0e-6:
                    sample_speed = max(
                        abs(current - previous) / dt
                        for current, previous in zip(
                            positions, previous_joint_sample[1], strict=True
                        )
                    )
            previous_joint_sample = (node.sim_time, positions)

            if plan is None:
                base_errors = _base_errors(node.base, base_target)
                if base_errors["position_m"] > args.base_position_tolerance_m:
                    reason = f"base_position_gate:{base_errors['position_m']:.6f}"
                    break
                if base_errors["yaw_rad"] > args.base_yaw_tolerance_rad:
                    reason = f"base_yaw_gate:{base_errors['yaw_rad']:.6f}"
                    break
                measured_spine = resolve_joint(node.joints, SPINE_JOINT)
                if abs(measured_spine - spine_target) > args.spine_tolerance_m:
                    reason = f"spine_gate:{measured_spine:.6f}"
                    break
                if handoff_started_sim is None:
                    handoff_started_sim = node.sim_time
                    handoff_right_pose = tuple(node.ee["right"])
                assert handoff_right_pose is not None

                # Claim RMPflow/gripper ownership first.  The direct joint
                # command is not latched by Isaac when the parent publisher is
                # destroyed, so checking before this publication creates a
                # brief reopen gap at process handoff.
                node.publish(
                    {"left": left_target, "right": handoff_right_pose},
                    spine_target,
                    {"left": 1.0, "right": 0.0},
                )
                measured_right_open = gripper_open_fraction(
                    resolve_joint(node.joints, RIGHT_GRIPPER_DRIVER)
                )
                if measured_right_open <= 0.25:
                    if handoff_close_stable_since is None:
                        handoff_close_stable_since = node.sim_time
                    elif node.sim_time - handoff_close_stable_since >= 0.25:
                        plan = build_transport_plan(reference, handoff_right_pose)
                else:
                    handoff_close_stable_since = None
                if plan is None:
                    if node.sim_time - handoff_started_sim > 2.0:
                        reason = (
                            "right_gripper_reacquire_timeout:"
                            f"{measured_right_open:.6f}"
                        )
                        break
                    continue

            if stage_index >= len(plan):
                reason = "stable_hybrid_transport_release_and_retreat"
                break
            stage = plan[stage_index]
            target_right = tuple(stage["pose"])
            targets = {"left": left_target, "right": target_right}
            if stage_initial is None:
                stage_initial = dict(node.ee)
                stage_started = node.sim_time
                stage_duration = transition_duration_s(
                    stage_initial,
                    targets,
                    max_linear_speed_m_s=args.max_linear_speed_m_s,
                    max_angular_speed_deg_s=args.max_angular_speed_deg_s,
                    minimum_s=0.30 if stage["name"] in {"retain", "release"} else 1.0,
                    maximum_s=8.0,
                )
                stable_since = None
            assert stage_started is not None
            raw_fraction = min(
                1.0, max(0.0, (node.sim_time - stage_started) / stage_duration)
            )
            fraction = minimum_jerk_fraction(raw_fraction)
            commanded = {
                side: (
                    *(
                        stage_initial[side][index]
                        + fraction
                        * (targets[side][index] - stage_initial[side][index])
                        for index in range(3)
                    ),
                    *_slerp(
                        stage_initial[side][3:7], targets[side][3:7], fraction
                    ),
                )
                for side in ("left", "right")
            }
            node.publish(
                commanded,
                spine_target,
                {"left": 1.0, "right": float(stage["right_open"])},
            )
            errors = {
                side: {
                    "position_m": _position_error(node.ee[side], targets[side]),
                    "orientation_deg": _orientation_error_deg(
                        node.ee[side], targets[side]
                    ),
                }
                for side in ("left", "right")
            }
            final_grippers = {
                "left": gripper_open_fraction(
                    resolve_joint(node.joints, LEFT_GRIPPER_DRIVER)
                ),
                "right": gripper_open_fraction(
                    resolve_joint(node.joints, RIGHT_GRIPPER_DRIVER)
                ),
            }
            gripper_ready = (
                final_grippers["right"] <= 0.25
                if float(stage["right_open"]) < 0.5
                else final_grippers["right"] >= 0.80
            )
            within_pose = all(
                value["position_m"] <= args.position_tolerance_m
                and value["orientation_deg"] <= args.orientation_tolerance_deg
                for value in errors.values()
            )
            if (
                raw_fraction < 1.0
                or not within_pose
                or not gripper_ready
                or sample_speed > args.settle_max_joint_speed_rad_s
            ):
                stable_since = None
                continue
            if stable_since is None:
                stable_since = node.sim_time
                continue
            if node.sim_time - stable_since < args.stable_dwell_s:
                continue
            stages.append(
                {
                    "name": stage["name"],
                    "duration_sim_s": node.sim_time - stage_started,
                    "target_right_ee_world_xyzw": list(target_right),
                    "final_ee_world_xyzw": list(node.ee["right"]),
                    "final_errors": errors,
                    "right_gripper_open_fraction": final_grippers["right"],
                }
            )
            stage_index += 1
            stage_initial = None
    finally:
        success = reason == "stable_hybrid_transport_release_and_retreat"
        result = {
            "schema_version": 1,
            "success": success,
            "reason": reason,
            "control": "vla_grasp_then_rmpflow_transport",
            "ground_truth_subscriptions": [],
            "reference": str(args.reference),
            "reference_support_unique_episodes": reference["source"][
                "support_unique_episodes"
            ],
            "base_errors": base_errors,
            "stages": stages,
            "completed_stage_count": len(stages),
            "expected_stage_count": 7,
            "final_ee_world": dict(node.ee),
            "final_gripper_open_fractions": final_grippers,
            "command_publications": node.publish_count,
            "elapsed_wall_s": time.monotonic() - wall_started,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
