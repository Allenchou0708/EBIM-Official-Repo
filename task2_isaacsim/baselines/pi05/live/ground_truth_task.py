#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the gated Task 2 deterministic ground-truth manipulation sequence."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from task2_isaacsim.baselines.pi05.live.ground_truth_pregrasp import (
    EE_TARGET_TOPICS,
    TOPICS,
    GroundTruthPregraspNode,
    _normalize_quaternion,
    _orientation_error_deg,
    _position_error,
    _quaternion_multiply_xyzw,
    _rotate_z,
    _slerp,
    _yaw_from_wxyz,
)
from task2_isaacsim.common.state_contract import (
    LEFT_GRIPPER_DRIVER,
    RIGHT_GRIPPER_DRIVER,
    SPINE_JOINT,
    gripper_open_fraction,
    resolve_joint,
)

GRIPPER_TARGET_TOPICS = TOPICS["cartesian_control"][
    "gripper_open_fraction_target"
]

# Compact landmarks from organizer dataset episode 19.  Positions are link8
# world poses; anchors make them follow each randomized live pad/target pose.
# The source episode and event frames are recorded in every output manifest.
REFERENCE_PAD_WXYZ = (
    1.75,
    1.9500000476837158,
    0.8500000238418579,
    0.7071099877357483,
    0.0,
    0.0,
    0.7071035504341125,
)
REFERENCE_TARGET_WXYZ = (
    2.1500000953674316,
    1.9500000476837158,
    0.75,
    0.7071099877357483,
    0.0,
    0.0,
    0.7071035504341125,
)

# Deformable-pad centroids sampled with the corresponding episode-19 link8
# landmarks.  The pad root pose remains static while the mesh is lifted, so
# lift waypoints must be expressed relative to the measured mesh rather than
# repeatedly relative to that root.
REFERENCE_PAD_CENTROIDS = {
    424: (1.7498467291543585, 1.9401436848435811, 0.8657973273369128),
    428: (1.7490971743346688, 1.935243248047229, 0.8712455671050116),
    432: (1.7477006194239366, 1.9232951098097537, 0.8832257469138224),
    436: (1.7472426638512792, 1.9151258790445422, 0.8881356428840204),
    440: (1.746401801259218, 1.9044465593354192, 0.8922001581765459),
    444: (1.7434357305962644, 1.883185849694197, 0.9051573449028228),
    448: (1.7492338788366604, 1.8752421641540147, 0.9089898976261268),
    450: (1.751831246112397, 1.871823499719064, 0.9125863845476847),
}

@dataclass(frozen=True)
class Phase:
    name: str
    frame: int
    anchor: str
    right_ee_xyzw: tuple[float, ...]
    right_gripper_open: float
    transition_sim_s: float
    stable_sim_s: float = 0.8


PHASES = (
    Phase(
        "approach_high",
        330,
        "thermalpad",
        (
            1.7535312175750732,
            2.203148365020752,
            0.8968551158905029,
            -0.034444019198417664,
            0.733759343624115,
            -0.6783731579780579,
            -0.014854952692985535,
        ),
        1.0,
        1.0,
        0.2,
    ),
    Phase(
        "approach_mid",
        350,
        "thermalpad",
        (
            1.7482216358184814,
            2.164870262145996,
            0.8882046937942505,
            -0.034079089760780334,
            0.7339746356010437,
            -0.6781599521636963,
            -0.014796532690525055,
        ),
        1.0,
        1.0,
        0.2,
    ),
    Phase(
        "approach_low",
        370,
        "thermalpad",
        (
            1.7477582693099976,
            2.1530354022979736,
            0.8714505434036255,
            -0.03360028937458992,
            0.7197021842002869,
            -0.6932889223098755,
            -0.015821928158402443,
        ),
        1.0,
        1.0,
        0.2,
    ),
    Phase(
        "approach",
        399,
        "thermalpad",
        (
            1.749876856803894,
            2.143172025680542,
            0.8659007549285889,
            -0.03354968503117561,
            0.7163382172584534,
            -0.6967585682868958,
            -0.01616767607629299,
        ),
        1.0,
        3.0,
    ),
    Phase(
        "grasp_align",
        400,
        "thermalpad",
        (
            1.749996542930603,
            2.1431241035461426,
            0.8658894300460815,
            -0.03355035558342934,
            0.7163240313529968,
            -0.6967730522155762,
            -0.016174064949154854,
        ),
        1.0,
        0.5,
        0.5,
    ),
    Phase(
        "grasp",
        410,
        "thermalpad",
        (
            1.7504874467849731,
            2.1580049991607666,
            0.8659641742706299,
            -0.03364843130111694,
            0.7162904739379883,
            -0.6968001127243042,
            -0.016287537291646004,
        ),
        0.0,
        0.4,
        # Episode 19 begins lifting roughly 0.47 s after the gripper reaches
        # fully closed.  A long dwell lets this thin deformable edge relax
        # out of the fingers before lift_start.
        0.1,
    ),
    Phase(
        "lift_start",
        424,
        "thermalpad",
        (
            1.7486478090286255,
            2.1282906532287598,
            0.9024243354797363,
            -0.03343498334288597,
            0.7160891890525818,
            -0.6970244646072388,
            -0.015977177768945694,
        ),
        0.0,
        1.0,
        0.3,
    ),
    Phase(
        "lift_428",
        428,
        "thermalpad",
        (
            1.7466994524002075,
            2.1143276691436768,
            0.9161669015884399,
            -0.03335778787732124,
            0.7160713076591492,
            -0.6970493197441101,
            -0.015852542594075203,
        ),
        0.0,
        0.5,
        0.2,
    ),
    Phase(
        "lift_432",
        432,
        "thermalpad",
        (
            1.7450852394104004,
            2.0987706184387207,
            0.9293118715286255,
            -0.03331999480724335,
            0.7160580158233643,
            -0.6970661282539368,
            -0.01579420082271099,
        ),
        0.0,
        0.5,
        0.2,
    ),
    Phase(
        "lift_436",
        436,
        "thermalpad",
        (
            1.744308590888977,
            2.0815353393554688,
            0.9401466846466064,
            -0.0332769975066185,
            0.7161064743995667,
            -0.6970203518867493,
            -0.015709569677710533,
        ),
        0.0,
        0.5,
        0.2,
    ),
    Phase(
        "lift_440",
        440,
        "thermalpad",
        (
            1.7447566986083984,
            2.0648233890533447,
            0.9496498107910156,
            -0.033250946551561356,
            0.7162041068077087,
            -0.6969231367111206,
            -0.015627533197402954,
        ),
        0.0,
        0.5,
        0.2,
    ),
    Phase(
        "lift_444",
        444,
        "thermalpad",
        (
            1.7466518878936768,
            2.049682140350342,
            0.9593536853790283,
            -0.033247582614421844,
            0.7162728905677795,
            -0.696853518486023,
            -0.015586207620799541,
        ),
        0.0,
        0.5,
        0.2,
    ),
    Phase(
        "lift_448",
        448,
        "thermalpad",
        (
            1.7480803728103638,
            2.0409555435180664,
            0.9651845693588257,
            -0.033266711980104446,
            0.7163125276565552,
            -0.6968123912811279,
            -0.015563076362013817,
        ),
        0.0,
        0.5,
        0.2,
    ),
    Phase(
        "lift",
        450,
        "thermalpad",
        (
            1.7487709522247314,
            2.03810977935791,
            0.9678614139556885,
            -0.033271025866270065,
            0.7163364291191101,
            -0.6967878341674805,
            -0.015553370118141174,
        ),
        0.0,
        3.0,
    ),
    Phase(
        "transport",
        544,
        "board_target",
        (
            2.110783100128174,
            2.0271072387695312,
            0.9782726168632507,
            -0.03588656336069107,
            0.7173925638198853,
            -0.6954920887947083,
            -0.01872878521680832,
        ),
        0.0,
        4.0,
    ),
    Phase(
        "orient",
        600,
        "board_target",
        (
            2.1442503929138184,
            1.981429934501648,
            1.076568603515625,
            -0.040783125907182693,
            0.9017878770828247,
            -0.4302130937576294,
            -0.005660881754010916,
        ),
        0.0,
        2.5,
    ),
    Phase(
        "lower_high",
        700,
        "board_target",
        (
            2.147153854370117,
            1.9999181032180786,
            0.9726579785346985,
            -0.04021255299448967,
            0.9086313843727112,
            -0.4156099557876587,
            -0.006350508891046047,
        ),
        0.0,
        3.0,
    ),
    Phase(
        "lower",
        800,
        "board_target",
        (
            2.146674156188965,
            2.0151445865631104,
            0.953220009803772,
            -0.03976678103208542,
            0.9673314094543457,
            -0.25037670135498047,
            0.00019051466370001435,
        ),
        0.0,
        3.0,
    ),
    Phase(
        "place",
        834,
        "board_target",
        (
            2.1464524269104004,
            2.016763687133789,
            0.9433668851852417,
            -0.03955800458788872,
            0.9875929951667786,
            -0.15194131433963776,
            0.003020714968442917,
        ),
        0.0,
        2.0,
    ),
    Phase(
        "release",
        835,
        "board_target",
        (
            2.146493673324585,
            2.01686954498291,
            0.9435117244720459,
            -0.03960195183753967,
            0.9879356622695923,
            -0.1496845930814743,
            0.0030537506099790335,
        ),
        1.0,
        0.5,
        1.2,
    ),
    Phase(
        "retract",
        900,
        "board_target",
        (
            2.0234971046447754,
            2.0049173831939697,
            1.0650297403335571,
            -0.03959408774971962,
            0.9905844926834106,
            -0.13100296258926392,
            0.003588818246498704,
        ),
        1.0,
        3.0,
    ),
)
PHASE_NAMES = tuple(phase.name for phase in PHASES)


def transform_landmark(
    phase: Phase,
    live_anchor_wxyz: tuple[float, ...],
) -> tuple[float, ...]:
    reference_anchor = (
        REFERENCE_PAD_WXYZ
        if phase.anchor == "thermalpad"
        else REFERENCE_TARGET_WXYZ
    )
    yaw_delta = _yaw_from_wxyz(live_anchor_wxyz[3:7]) - _yaw_from_wxyz(
        reference_anchor[3:7]
    )
    offset = tuple(
        phase.right_ee_xyzw[index] - reference_anchor[index]
        for index in range(3)
    )
    rotated = _rotate_z(offset, yaw_delta)
    yaw_quaternion_xyzw = (
        0.0,
        0.0,
        math.sin(yaw_delta / 2.0),
        math.cos(yaw_delta / 2.0),
    )
    orientation = _quaternion_multiply_xyzw(
        yaw_quaternion_xyzw, phase.right_ee_xyzw[3:7]
    )
    return (
        live_anchor_wxyz[0] + rotated[0],
        live_anchor_wxyz[1] + rotated[1],
        live_anchor_wxyz[2] + rotated[2],
        *orientation,
    )


def transform_lift_landmark(
    phase: Phase,
    live_pad_centroid: tuple[float, float, float],
    live_pad_pose_wxyz: tuple[float, ...],
) -> tuple[float, ...]:
    """Anchor a lift pose to the currently measured deformable pad mesh."""

    reference_centroid = REFERENCE_PAD_CENTROIDS[phase.frame]
    yaw_delta = _yaw_from_wxyz(live_pad_pose_wxyz[3:7]) - _yaw_from_wxyz(
        REFERENCE_PAD_WXYZ[3:7]
    )
    reference_offset = tuple(
        phase.right_ee_xyzw[index] - reference_centroid[index]
        for index in range(3)
    )
    rotated_offset = _rotate_z(reference_offset, yaw_delta)
    yaw_quaternion_xyzw = (
        0.0,
        0.0,
        math.sin(yaw_delta / 2.0),
        math.cos(yaw_delta / 2.0),
    )
    orientation = _quaternion_multiply_xyzw(
        yaw_quaternion_xyzw, phase.right_ee_xyzw[3:7]
    )
    return (
        live_pad_centroid[0] + rotated_offset[0],
        live_pad_centroid[1] + rotated_offset[1],
        live_pad_centroid[2] + reference_offset[2],
        *orientation,
    )


def tracking_corrected_position(
    nominal: tuple[float, ...],
    target: tuple[float, ...],
    measured: tuple[float, ...],
    *,
    gain: float,
    limit_m: float,
    fraction: float,
) -> tuple[float, float, float]:
    """Add bounded measured link8 feedback to an interpolated target."""

    correction = tuple(
        gain * fraction * (target[index] - measured[index])
        for index in range(3)
    )
    norm = math.sqrt(sum(value * value for value in correction))
    scale = min(1.0, limit_m / norm) if norm > 0.0 else 1.0
    return tuple(
        nominal[index] + scale * correction[index] for index in range(3)
    )


def tracking_corrected_orientation(
    nominal: tuple[float, ...],
    target: tuple[float, ...],
    measured: tuple[float, ...],
    *,
    gain: float,
    limit_deg: float,
    fraction: float,
) -> tuple[float, ...]:
    """Add bounded measured link8 orientation feedback to a target."""

    measured = _normalize_quaternion(measured)
    target = _normalize_quaternion(target)
    inverse_measured = (
        -measured[0],
        -measured[1],
        -measured[2],
        measured[3],
    )
    error = _quaternion_multiply_xyzw(target, inverse_measured)
    if error[3] < 0.0:
        error = tuple(-value for value in error)
    half_angle = math.acos(max(-1.0, min(1.0, error[3])))
    sin_half = math.sin(half_angle)
    if sin_half < 1.0e-9 or fraction <= 0.0 or gain <= 0.0:
        return _normalize_quaternion(nominal)
    axis = tuple(value / sin_half for value in error[:3])
    correction_angle = min(
        gain * fraction * 2.0 * half_angle,
        math.radians(limit_deg),
    )
    correction = (
        *(value * math.sin(correction_angle / 2.0) for value in axis),
        math.cos(correction_angle / 2.0),
    )
    return _quaternion_multiply_xyzw(correction, nominal)


class GroundTruthTaskNode(GroundTruthPregraspNode):
    def __init__(self) -> None:
        super().__init__()
        self.pad_centroid: tuple[float, float, float] | None = None
        self.pad_time: float | None = None
        self.gripper_publish_count = 0
        self._owned_subscriptions.append(
            self.create_subscription(
                Float32MultiArray,
                TOPICS["ground_truth"]["pad_points"],
                self._on_pad_points,
                10,
            )
        )
        self._gripper_publishers = {
            side: self.create_publisher(JointState, topic, 10)
            for side, topic in GRIPPER_TARGET_TOPICS.items()
        }

    def _on_pad_points(self, message: Float32MultiArray) -> None:
        data = tuple(float(value) for value in message.data)
        if len(data) < 5:
            return
        count = int(round(data[1]))
        if count <= 0 or len(data) != 2 + 3 * count:
            return
        values = data[2:]
        if not all(math.isfinite(value) for value in values):
            return
        self.pad_time = data[0]
        self.pad_centroid = tuple(
            sum(values[axis::3]) / count for axis in range(3)
        )

    def fresh_task(self, maximum_skew_s: float) -> bool:
        return (
            self.fresh(maximum_skew_s)
            and self.sim_time is not None
            and self.pad_time is not None
            and self.pad_centroid is not None
            and 0.0 <= self.sim_time - self.pad_time <= maximum_skew_s
        )

    def conflicts(self) -> dict[str, int]:
        result = super().conflicts()
        result.update(
            {
                f"{side}_gripper": max(
                    0,
                    len(self.get_publishers_info_by_topic(topic)) - 1,
                )
                for side, topic in GRIPPER_TARGET_TOPICS.items()
            }
        )
        return result

    def subscribers_ready(self) -> bool:
        return super().subscribers_ready() and all(
            publisher.get_subscription_count() == 1
            for publisher in self._gripper_publishers.values()
        )

    def publish_task(
        self,
        targets: dict[str, tuple[float, ...]],
        *,
        left_gripper_open: float,
        right_gripper_open: float,
    ) -> None:
        self.publish(targets)
        if self.sim_time is None:
            raise RuntimeError("simulator clock unavailable")
        stamp = Time(nanoseconds=max(0, int(self.sim_time * 1.0e9))).to_msg()
        values = {
            "left": left_gripper_open,
            "right": right_gripper_open,
        }
        for side, value in values.items():
            message = JointState()
            message.header.stamp = stamp
            message.name = [f"{side}_gripper_open_fraction"]
            message.position = [float(value)]
            self._gripper_publishers[side].publish(message)
            self.gripper_publish_count += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-at", choices=PHASE_NAMES, default="approach")
    parser.add_argument("--stop-after", choices=PHASE_NAMES, default="retract")
    parser.add_argument(
        "--base-target",
        nargs=3,
        type=float,
        default=(2.100026845932007, 3.0529046058654785, -1.5706931352615356),
    )
    parser.add_argument("--base-position-tolerance-m", type=float, default=0.05)
    parser.add_argument("--base-yaw-tolerance-rad", type=float, default=0.08)
    parser.add_argument("--spine-target-m", type=float, default=0.4852330982685089)
    parser.add_argument("--spine-tolerance-m", type=float, default=0.02)
    parser.add_argument("--position-tolerance-m", type=float, default=0.04)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=12.0)
    parser.add_argument("--tracking-correction-gain", type=float, default=2.0)
    parser.add_argument("--tracking-correction-limit-m", type=float, default=0.06)
    parser.add_argument(
        "--orientation-correction-gain", type=float, default=2.0
    )
    parser.add_argument(
        "--orientation-correction-limit-deg", type=float, default=15.0
    )
    parser.add_argument("--tracking-correction-ramp-s", type=float, default=1.0)
    parser.add_argument("--maximum-skew-s", type=float, default=0.12)
    parser.add_argument("--max-duration-s", type=float, default=600.0)
    return parser


def _phase_metrics(node: GroundTruthTaskNode) -> dict[str, Any]:
    assert node.pad_centroid is not None
    target = node.objects["board_target"]
    pad_xy_error = math.hypot(
        node.pad_centroid[0] - target[0],
        node.pad_centroid[1] - target[1],
    )
    return {
        "left_gripper_open_fraction": gripper_open_fraction(
            resolve_joint(node.joints, LEFT_GRIPPER_DRIVER)
        ),
        "right_gripper_open_fraction": gripper_open_fraction(
            resolve_joint(node.joints, RIGHT_GRIPPER_DRIVER)
        ),
        "pad_centroid_world_m": list(node.pad_centroid),
        "target_world_wxyz": list(target),
        "pad_target_xy_error_m": pad_xy_error,
        "pad_target_z_error_m": abs(node.pad_centroid[2] - target[2]),
        "pad_height_above_target_m": node.pad_centroid[2] - target[2],
    }


def _condition_passed(phase: Phase, metrics: dict[str, Any]) -> bool:
    right_open = float(metrics["right_gripper_open_fraction"])
    height = float(metrics["pad_height_above_target_m"])
    xy_error = float(metrics["pad_target_xy_error_m"])
    z_error = float(metrics["pad_target_z_error_m"])
    if phase.name.startswith("approach") or phase.name == "grasp_align":
        return right_open >= 0.85
    if phase.name == "grasp":
        return right_open <= 0.25
    lift_height_gates = {
        "lift_start": 0.105,
        "lift_428": 0.112,
        "lift_432": 0.120,
        "lift_436": 0.125,
        "lift_440": 0.130,
        "lift_444": 0.140,
        "lift_448": 0.145,
    }
    if phase.name in lift_height_gates:
        return right_open <= 0.25 and height >= lift_height_gates[phase.name]
    if phase.name in {"lift", "transport", "orient"}:
        return right_open <= 0.25 and height >= 0.13
    if phase.name == "lower_high":
        return right_open <= 0.25 and height >= 0.04
    if phase.name == "lower":
        return right_open <= 0.25 and xy_error <= 0.12
    if phase.name == "place":
        return right_open <= 0.25 and xy_error <= 0.06 and z_error <= 0.06
    if phase.name in {"release", "retract"}:
        return right_open >= 0.85 and xy_error <= 0.06 and z_error <= 0.06
    raise AssertionError(f"unsupported phase: {phase.name}")


def main() -> int:
    args = build_parser().parse_args()
    start_index = PHASE_NAMES.index(args.start_at)
    stop_index = PHASE_NAMES.index(args.stop_after)
    if stop_index < start_index:
        raise SystemExit("--stop-after must not precede --start-at")

    rclpy.init()
    node = GroundTruthTaskNode()
    signal.signal(
        signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True)
    )
    signal.signal(signal.SIGINT, lambda *_: setattr(node, "stop_requested", True))
    wall_started = time.monotonic()
    last_sim_time: float | None = None
    phase_index = start_index
    phase_started_sim: float | None = None
    stable_since: float | None = None
    phase_initial: dict[str, tuple[float, ...]] | None = None
    phase_target: dict[str, tuple[float, ...]] | None = None
    left_hold: tuple[float, ...] | None = None
    records: list[dict[str, Any]] = []
    errors: dict[str, dict[str, float]] = {}
    startup_validated = False
    reason = "host_watchdog_timeout"
    success = False
    try:
        while (
            not node.stop_requested
            and time.monotonic() - wall_started < args.max_duration_s
        ):
            rclpy.spin_once(node, timeout_sec=0.02)
            if not node.fresh_task(args.maximum_skew_s):
                continue
            if not node.subscribers_ready():
                continue
            conflicts = node.conflicts()
            if any(conflicts.values()):
                reason = f"publisher_contention:{conflicts}"
                break
            assert node.sim_time is not None
            if last_sim_time is not None and node.sim_time < last_sim_time:
                reason = "simulator_clock_reset"
                break
            if last_sim_time is not None and node.sim_time == last_sim_time:
                continue
            last_sim_time = node.sim_time

            assert node.base is not None
            base_position_error = math.hypot(
                node.base[0] - args.base_target[0],
                node.base[1] - args.base_target[1],
            )
            base_yaw_error = abs(
                math.atan2(
                    math.sin(node.base[2] - args.base_target[2]),
                    math.cos(node.base[2] - args.base_target[2]),
                )
            )
            spine = resolve_joint(node.joints, SPINE_JOINT)
            left_open = gripper_open_fraction(
                resolve_joint(node.joints, LEFT_GRIPPER_DRIVER)
            )
            if base_position_error > args.base_position_tolerance_m:
                reason = f"base_position_gate:{base_position_error:.6f}"
                break
            if base_yaw_error > args.base_yaw_tolerance_rad:
                reason = f"base_yaw_gate:{base_yaw_error:.6f}"
                break
            if abs(spine - args.spine_target_m) > args.spine_tolerance_m:
                reason = f"spine_gate:{spine:.6f}"
                break
            if left_open < 0.85:
                reason = f"left_gripper_open_gate:{left_open:.6f}"
                break
            if "thermalpad" not in node.objects or "board_target" not in node.objects:
                reason = "task_ground_truth_missing"
                break

            phase = PHASES[phase_index]
            if not startup_validated:
                right_open = gripper_open_fraction(
                    resolve_joint(node.joints, RIGHT_GRIPPER_DRIVER)
                )
                if (
                    phase.name.startswith("approach")
                    or phase.name in {"grasp_align", "grasp"}
                ) and right_open < 0.85:
                    reason = f"right_gripper_start_open_gate:{right_open:.6f}"
                    break
                if phase.name in {
                    "lift_start",
                    "lift_428",
                    "lift_432",
                    "lift_436",
                    "lift_440",
                    "lift_444",
                    "lift_448",
                    "lift",
                    "transport",
                    "orient",
                    "lower_high",
                    "lower",
                    "place",
                    "release",
                } and right_open > 0.25:
                    reason = f"right_gripper_start_closed_gate:{right_open:.6f}"
                    break
                if phase.name == "retract" and right_open < 0.85:
                    reason = f"right_gripper_start_open_gate:{right_open:.6f}"
                    break
                startup_validated = True
            if phase_started_sim is None:
                phase_started_sim = node.sim_time
                phase_initial = dict(node.ee)
                if left_hold is None:
                    left_hold = node.ee["left"]
                live_anchor = node.objects[phase.anchor]
                right_target = (
                    transform_lift_landmark(
                        phase, node.pad_centroid, live_anchor
                    )
                    if phase.frame in REFERENCE_PAD_CENTROIDS
                    else transform_landmark(phase, live_anchor)
                )
                phase_target = {
                    "left": left_hold,
                    "right": right_target,
                }
                stable_since = None

            assert phase_initial is not None and phase_target is not None
            fraction = min(
                1.0,
                max(
                    0.0,
                    (node.sim_time - phase_started_sim) / phase.transition_sim_s,
                ),
            )
            correction_fraction = min(
                1.0,
                max(
                    0.0,
                    (
                        node.sim_time
                        - phase_started_sim
                        - phase.transition_sim_s
                    )
                    / args.tracking_correction_ramp_s,
                ),
            )
            commanded = {}
            for side in ("left", "right"):
                nominal_position = tuple(
                    phase_initial[side][index]
                    + fraction
                    * (phase_target[side][index] - phase_initial[side][index])
                    for index in range(3)
                )
                if side == "right":
                    position = tracking_corrected_position(
                        nominal_position,
                        phase_target[side],
                        node.ee[side],
                        gain=args.tracking_correction_gain,
                        limit_m=args.tracking_correction_limit_m,
                        fraction=correction_fraction,
                    )
                else:
                    position = nominal_position
                nominal_orientation = _slerp(
                    phase_initial[side][3:7],
                    phase_target[side][3:7],
                    fraction,
                )
                if side == "right":
                    orientation = tracking_corrected_orientation(
                        nominal_orientation,
                        phase_target[side][3:7],
                        node.ee[side][3:7],
                        gain=args.orientation_correction_gain,
                        limit_deg=args.orientation_correction_limit_deg,
                        fraction=correction_fraction,
                    )
                else:
                    orientation = nominal_orientation
                commanded[side] = (
                    *position,
                    *orientation,
                )
            node.publish_task(
                commanded,
                left_gripper_open=1.0,
                right_gripper_open=phase.right_gripper_open,
            )
            errors = {
                side: {
                    "position_m": _position_error(
                        node.ee[side], phase_target[side]
                    ),
                    "orientation_deg": _orientation_error_deg(
                        node.ee[side], phase_target[side]
                    ),
                }
                for side in ("left", "right")
            }
            metrics = _phase_metrics(node)
            ee_ok = all(
                value["position_m"] <= args.position_tolerance_m
                and value["orientation_deg"] <= args.orientation_tolerance_deg
                for value in errors.values()
            )
            phase_ok = fraction >= 1.0 and ee_ok and _condition_passed(
                phase, metrics
            )
            if not phase_ok:
                stable_since = None
                continue
            if stable_since is None:
                stable_since = node.sim_time
                continue
            if node.sim_time - stable_since < phase.stable_sim_s:
                continue

            records.append(
                {
                    "phase": phase.name,
                    "source_episode": 19,
                    "source_frame": phase.frame,
                    "anchor": phase.anchor,
                    "target_basis": (
                        "live_pad_points_centroid"
                        if phase.frame in REFERENCE_PAD_CENTROIDS
                        else "object_pose"
                    ),
                    "target_right_ee_world": list(phase_target["right"]),
                    "measured_ee_world": {
                        side: list(node.ee[side]) for side in ("left", "right")
                    },
                    "right_gripper_open_fraction_target": (
                        phase.right_gripper_open
                    ),
                    "final_errors": errors,
                    "final_metrics": metrics,
                    "completed_sim_time_s": node.sim_time,
                }
            )
            if phase_index == stop_index:
                success = True
                reason = f"stable_{phase.name}"
                break
            phase_index += 1
            phase_started_sim = None
            phase_initial = None
            phase_target = None
            stable_since = None
    finally:
        result = {
            "schema_version": 1,
            "success": success,
            "reason": reason,
            "clock": "simulator",
            "host_clock_use": "process_watchdog_only",
            "start_at": args.start_at,
            "stop_after": args.stop_after,
            "source_dataset_episode": 19,
            "tracking_correction": {
                "position_gain": args.tracking_correction_gain,
                "position_limit_m": args.tracking_correction_limit_m,
                "orientation_gain": args.orientation_correction_gain,
                "orientation_limit_deg": (
                    args.orientation_correction_limit_deg
                ),
                "ramp_sim_s": args.tracking_correction_ramp_s,
            },
            "pose_command_publications": node.publish_count,
            "gripper_command_publications": node.gripper_publish_count,
            "completed_phases": records,
            "active_phase": (
                PHASES[phase_index].name
                if not success and 0 <= phase_index < len(PHASES)
                else None
            ),
            "active_target_right_ee_world": (
                list(phase_target["right"])
                if not success and phase_target is not None
                else None
            ),
            "final_ee_world": node.ee,
            "final_errors": errors,
            "final_metrics": (
                _phase_metrics(node)
                if node.pad_centroid is not None
                and "board_target" in node.objects
                and node.joints
                else None
            ),
            "elapsed_wall_s": time.monotonic() - wall_started,
            "topics": {
                "ee_targets": EE_TARGET_TOPICS,
                "gripper_open_fraction_targets": GRIPPER_TARGET_TOPICS,
            },
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
