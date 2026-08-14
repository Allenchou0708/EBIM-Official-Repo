#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Observable-only phase manager for the Task 2 PI0.5 V4 policy.

This module selects language prompts.  It never creates actuator targets and
must not be used as an arm, gripper, spine, or base controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .phase_conditioned_dataset import PHASE_PROMPTS
from .pregrasp_pose_audit import (
    local_axis_world_z_abs,
    quaternion_angle_deg,
)


class V4Phase(str, Enum):
    STARTUP_RISE = "startup_rise"
    APPROACH = "approach"
    ORIENT_PREGRASP = "orient_pregrasp"
    GRASP_ACQUISITION = "grasp_acquisition"
    LIFT_TRANSFER = "lift_transfer"
    LOWER_PLACE = "lower_place"
    RELEASE_RETREAT = "release_retreat"


PHASE_ORDER = tuple(V4Phase(name) for name in PHASE_PROMPTS)


@dataclass(frozen=True)
class CameraEvidence:
    """Semantic observations produced by a separate read-only vision probe."""

    pad_visible: bool = False
    pad_centered_right_wrist: bool = False
    pad_lifted: bool = False
    target_visible: bool = False
    target_centered_right_wrist: bool = False
    pad_supported_on_target: bool = False


@dataclass(frozen=True)
class PhaseObservation:
    """One simulator-timestamped observation used only for prompt selection."""

    sim_time_s: float
    spine_m: float
    right_ee_xyzw: tuple[float, float, float, float, float, float, float]
    right_gripper_open_fraction: float
    right_gripper_command: float | None
    camera: CameraEvidence


@dataclass(frozen=True)
class PhaseManagerConfig:
    stable_observations: int = 3
    spine_ready_m: float = 0.44
    orientation_entry_position_m: tuple[float, float, float] = (
        1.8010278940200806,
        2.2869436740875244,
        0.8801376819610596,
    )
    orientation_entry_radius_m: float = 0.09
    preclose_position_m: tuple[float, float, float] = (
        1.7504082918167114,
        2.140868902206421,
        0.8722001910209656,
    )
    preclose_position_radius_m: float = 0.055
    preclose_quaternion_xyzw: tuple[float, float, float, float] = (
        -0.02928313829834448,
        0.7328308972762488,
        -0.6796555833030944,
        -0.013025432569340717,
    )
    preclose_orientation_error_deg: float = 12.0
    vertical_local_y_alignment: float = 0.98
    closed_fraction: float = 0.5
    open_fraction: float = 0.5
    maximum_phase_sim_s: float = 20.0


def _distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


class ObservablePhaseManager:
    """Forward-only, hysteretic prompt selector driven by simulator time."""

    def __init__(self, config: PhaseManagerConfig | None = None):
        self.config = config or PhaseManagerConfig()
        if self.config.stable_observations <= 0:
            raise ValueError("stable_observations must be positive")
        self.phase = V4Phase.STARTUP_RISE
        self.phase_started_sim_s: float | None = None
        self.last_sim_time_s: float | None = None
        self._transition_streak = 0
        self.complete = False
        self.stop_reason: str | None = None
        self.transitions: list[dict[str, Any]] = []

    @property
    def prompt(self) -> str:
        return PHASE_PROMPTS[self.phase.value]

    def _transition_ready(self, observation: PhaseObservation) -> bool:
        cfg = self.config
        position = observation.right_ee_xyzw[:3]
        quaternion = observation.right_ee_xyzw[3:]
        camera = observation.camera
        if self.phase == V4Phase.STARTUP_RISE:
            return observation.spine_m >= cfg.spine_ready_m
        if self.phase == V4Phase.APPROACH:
            return (
                camera.pad_visible
                and camera.pad_centered_right_wrist
                and _distance(position, cfg.orientation_entry_position_m)
                <= cfg.orientation_entry_radius_m
            )
        if self.phase == V4Phase.ORIENT_PREGRASP:
            return (
                camera.pad_visible
                and camera.pad_centered_right_wrist
                and _distance(position, cfg.preclose_position_m)
                <= cfg.preclose_position_radius_m
                and quaternion_angle_deg(
                    list(quaternion), list(cfg.preclose_quaternion_xyzw)
                )
                <= cfg.preclose_orientation_error_deg
                and local_axis_world_z_abs(list(quaternion))[1]
                >= cfg.vertical_local_y_alignment
            )
        if self.phase == V4Phase.GRASP_ACQUISITION:
            return (
                observation.right_gripper_command is not None
                and observation.right_gripper_command < cfg.closed_fraction
                and observation.right_gripper_open_fraction
                < cfg.closed_fraction
                and camera.pad_lifted
            )
        if self.phase == V4Phase.LIFT_TRANSFER:
            return (
                camera.pad_lifted
                and camera.target_visible
                and camera.target_centered_right_wrist
            )
        if self.phase == V4Phase.LOWER_PLACE:
            return (
                observation.right_gripper_open_fraction < cfg.closed_fraction
                and camera.pad_supported_on_target
            )
        if self.phase == V4Phase.RELEASE_RETREAT:
            return (
                observation.right_gripper_command is not None
                and observation.right_gripper_command > cfg.open_fraction
                and observation.right_gripper_open_fraction > cfg.open_fraction
                and camera.pad_supported_on_target
            )
        raise AssertionError(f"unsupported phase: {self.phase}")

    def update(self, observation: PhaseObservation) -> V4Phase:
        if self.stop_reason is not None or self.complete:
            return self.phase
        if not math.isfinite(observation.sim_time_s):
            raise ValueError("phase observation simulator time must be finite")
        if self.last_sim_time_s is not None and observation.sim_time_s < self.last_sim_time_s:
            raise ValueError("phase observation simulator time regressed")
        self.last_sim_time_s = observation.sim_time_s
        if self.phase_started_sim_s is None:
            self.phase_started_sim_s = observation.sim_time_s
        if (
            observation.sim_time_s - self.phase_started_sim_s
            > self.config.maximum_phase_sim_s
        ):
            self.stop_reason = f"phase_timeout:{self.phase.value}"
            return self.phase

        self._transition_streak = (
            self._transition_streak + 1
            if self._transition_ready(observation)
            else 0
        )
        if self._transition_streak < self.config.stable_observations:
            return self.phase
        self._transition_streak = 0
        if self.phase == V4Phase.RELEASE_RETREAT:
            self.complete = True
            self.transitions.append(
                {
                    "from": self.phase.value,
                    "to": "complete",
                    "sim_time_s": observation.sim_time_s,
                }
            )
            return self.phase
        next_phase = PHASE_ORDER[PHASE_ORDER.index(self.phase) + 1]
        self.transitions.append(
            {
                "from": self.phase.value,
                "to": next_phase.value,
                "sim_time_s": observation.sim_time_s,
            }
        )
        self.phase = next_phase
        self.phase_started_sim_s = observation.sim_time_s
        return self.phase
