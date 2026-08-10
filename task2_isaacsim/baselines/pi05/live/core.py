# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light state, freshness, readiness, and action safety gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from task2_isaacsim.baselines.pi05.contract import (
    ACTION_SIZE,
    EXPECTED_CAMERA_SHAPES,
    apply_fixed_mobile_axes,
    validate_absolute_action_bounds,
)
from task2_isaacsim.common.state_contract import finite_state

ROBOT_CAMERA_KEYS = ("head", "wrist_left", "wrist_right")


class RunnerPhase(str, Enum):
    RESET = "RESET"
    BASE_PREPOSITION = "BASE_PREPOSITION"
    SETTLE = "SETTLE"
    MANIPULATION_READY = "MANIPULATION_READY"
    PI05_MANIPULATION = "PI05_MANIPULATION"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class FreshnessConfig:
    camera_max_age_s: float = 0.25
    camera_max_skew_s: float = 0.10
    state_max_age_s: float = 0.10
    action_queue_max_age_s: float = 0.25


@dataclass(frozen=True)
class ReadinessConfig:
    target_x: float
    target_y: float
    target_yaw: float
    position_tolerance_m: float = 0.05
    yaw_tolerance_rad: float = 0.10
    velocity_threshold: float = 0.02
    settle_duration_s: float = 1.0


def angular_error(left: float, right: float) -> float:
    """Smallest signed angular difference."""

    return math.atan2(math.sin(left - right), math.cos(left - right))


class BaseReadinessGate:
    """Track manual staging and invalidate readiness on reset/base input."""

    def __init__(self, config: ReadinessConfig):
        self.config = config
        self.phase = RunnerPhase.RESET
        self._settle_since: float | None = None
        self.last_evidence: dict[str, Any] = {}

    def reset(self, reason: str = "scene_reset") -> None:
        self.phase = RunnerPhase.BASE_PREPOSITION
        self._settle_since = None
        self.last_evidence = {"ready": False, "reason": reason}

    def note_base_input(self) -> None:
        self.reset("new_base_input")

    def update(
        self,
        now: float,
        pose: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> bool:
        cfg = self.config
        dx = float(pose[0]) - cfg.target_x
        dy = float(pose[1]) - cfg.target_y
        position_error = math.hypot(dx, dy)
        yaw_error = abs(angular_error(float(pose[2]), cfg.target_yaw))
        speed = math.sqrt(sum(float(value) ** 2 for value in velocity))
        within = (
            position_error <= cfg.position_tolerance_m
            and yaw_error <= cfg.yaw_tolerance_rad
            and speed <= cfg.velocity_threshold
        )
        if not within:
            self.phase = RunnerPhase.BASE_PREPOSITION
            self._settle_since = None
        elif self._settle_since is None:
            self.phase = RunnerPhase.SETTLE
            self._settle_since = now
        elif now - self._settle_since >= cfg.settle_duration_s:
            self.phase = RunnerPhase.MANIPULATION_READY
        else:
            self.phase = RunnerPhase.SETTLE
        self.last_evidence = {
            "ready": self.phase == RunnerPhase.MANIPULATION_READY,
            "phase": self.phase.value,
            "actual_pose": list(pose),
            "target_pose": [cfg.target_x, cfg.target_y, cfg.target_yaw],
            "position_error_m": position_error,
            "yaw_error_rad": yaw_error,
            "speed": speed,
            "settled_for_s": 0.0
            if self._settle_since is None
            else max(0.0, now - self._settle_since),
        }
        return bool(self.last_evidence["ready"])

    def arm(self) -> None:
        if self.phase != RunnerPhase.MANIPULATION_READY:
            raise RuntimeError("base is not MANIPULATION_READY")
        self.phase = RunnerPhase.PI05_MANIPULATION


def validate_rgb_frame(key: str, array: Any) -> None:
    """Validate policy RGB shape without accepting evaluator camera keys."""

    if key not in ROBOT_CAMERA_KEYS:
        raise ValueError(f"unsupported policy camera: {key}")
    expected = tuple(EXPECTED_CAMERA_SHAPES[f"observation.images.{key}"])
    actual = tuple(int(value) for value in array.shape)
    if actual != expected:
        raise ValueError(f"{key} image shape {actual} != {expected}")
    if str(array.dtype) != "uint8":
        raise ValueError(f"{key} image dtype must be uint8, got {array.dtype}")


def freshness_metrics(
    *,
    now: float,
    camera_times: dict[str, float],
    camera_sequences: dict[str, int],
    state_time: float,
    last_camera_sequences: dict[str, int],
    config: FreshnessConfig,
) -> dict[str, Any]:
    """Require one new, fresh frame from each robot camera and fresh state."""

    missing = [key for key in ROBOT_CAMERA_KEYS if key not in camera_times]
    if missing:
        raise ValueError(f"missing robot camera frames: {missing}")
    stale_sequences = [
        key
        for key in ROBOT_CAMERA_KEYS
        if camera_sequences.get(key, -1) <= last_camera_sequences.get(key, -1)
    ]
    if stale_sequences:
        raise ValueError(f"camera frames were reused: {stale_sequences}")
    ages = {
        key: max(0.0, now - camera_times[key]) for key in ROBOT_CAMERA_KEYS
    }
    skew = max(camera_times.values()) - min(camera_times.values())
    state_age = max(0.0, now - state_time)
    if max(ages.values()) > config.camera_max_age_s:
        raise ValueError(f"stale camera frame ages: {ages}")
    if skew > config.camera_max_skew_s:
        raise ValueError(f"inter-camera skew {skew:.6f}s exceeds threshold")
    if state_age > config.state_max_age_s:
        raise ValueError(f"state age {state_age:.6f}s exceeds threshold")
    return {
        "frame_age_s": ages,
        "inter_camera_skew_s": skew,
        "state_age_s": state_age,
        "camera_sequences": dict(camera_sequences),
    }


def safe_action(
    raw_action: Any, *, spine_hold: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate raw absolute output, then fix base and hold current spine."""

    raw = tuple(float(value) for value in raw_action)
    if len(raw) != ACTION_SIZE:
        raise ValueError(f"policy action must contain {ACTION_SIZE} values")
    validate_absolute_action_bounds(raw)
    effective = apply_fixed_mobile_axes(raw, spine_height=spine_hold)
    validate_absolute_action_bounds(effective)
    return raw, effective


class ActionWatchdog:
    """Reject stale action queues and hold instead of publishing targets."""

    def __init__(self, maximum_age_s: float):
        self.maximum_age_s = maximum_age_s

    def valid(self, *, now: float, created_at: float) -> bool:
        return 0.0 <= now - created_at <= self.maximum_age_s


def validate_live_state(state: Any) -> tuple[float, ...]:
    return finite_state(state)
