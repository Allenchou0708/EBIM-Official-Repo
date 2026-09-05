#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Acquire or finish a right-hand grasp with a measured RMPflow waypoint plan.

The controller accepts either a VLA latch or a bounded camera-only code-policy
acquisition.  It subscribes only to robot state and robot RGB-D cameras, never
to task objects, evaluator output, or simulator ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState

from task2_isaacsim.baselines.pi05.contract import FR3_JOINT_LIMITS

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
    RIGHT_JOINTS,
    SPINE_JOINT,
    gripper_open_fraction,
    resolve_joint,
)
from task2_isaacsim.scripts.topics import camera_topic, load_topics

REQUIRED_LANDMARKS = ("peel_lift", "transfer", "place", "release")
# The lower finger can still touch the pad base immediately after closing.
# Lift only far enough to break that contact before introducing any lateral
# peel.  A longer vertical pull tensions the deformable pad and changes the
# effective grasp depth.
SHORT_DECONTACT_LIFT_M = 0.015
# Development-only medians aligned to the first three measured closed-gripper
# frames across 180 successful GT trajectories.  They define a progressively
# shallower rise: clear the wrist with a short diagonal, then move mainly
# forward.  See evidence/phase2_gt_post_grasp_trajectory_20260905/audit.json.
GT_SHORT_DIAGONAL_FORWARD_M = 0.0244850136
GT_SHORT_DIAGONAL_Z_M = 0.0315119326
GT_FORWARD_RISE_FORWARD_M = 0.0418631403
GT_FORWARD_RISE_Z_M = 0.0462560654
GT_RETAINED_FORWARD_M = 0.0736408791
GT_RETAINED_Z_M = 0.0687306821
# Median pose when the GT trajectories first begin 5 mm of robot-left motion.
# Reaching this clearance before any large lateral transfer prevents the pad
# tail from being stretched sideways against the symmetric base.
GT_PRE_LATERAL_FORWARD_M = 0.1083290609
GT_PRE_LATERAL_Z_M = 0.0958608985
# Across 180 successful development trajectories, the gripper begins closing
# at a deeper pose and retracts 14.41 mm toward the robot while the measured
# opening falls from 0.95 to the first stable latch.  RMPflow realizes only
# about 10 mm of a 14.4 mm command under pad contact; use the demonstrated
# 20 mm q10 (strong-retract) envelope so the measured motion reaches the
# 15 mm median.  This changes only the open-to-closed retract, not the final
# latch depth or height.  Closing in place left the pad easy to peel out.
GT_CLOSE_RETRACTION_M = 0.020
# Use the demonstrated strong-retract envelope for collision-free camera
# refinement.  Before moving this far toward the pad, raise the wrist enough
# to clear the base with the lower finger.  The subsequent close stage
# retracts and descends to the unchanged stable latch, preserving the deeper
# pad engagement without depending on contact rebound.
# The observation pose is already a GT-derived pre-grasp.  Runtime therefore
# makes only a small camera-relative refinement from the measured pose; the
# 20 mm value above describes the demonstrated *closing retract* envelope and
# must not be reused as an open-gripper insertion distance.
# Keep this approach shallow: a seed-1104 ablation that raised it to 8 mm
# made the lower finger strike the base and corrupted the grasp.  Cross-axis
# retention must be improved through the wrist attitude, not deeper insertion.
PREGRASP_FORWARD_REFINEMENT_M = 0.004
PREGRASP_FORWARD_REFINEMENT_MIN_M = 0.0
PREGRASP_FORWARD_REFINEMENT_MAX_M = 0.008
# Keep the camera correction inside the randomized layout envelope.  A fixed
# one-sided preload improved one trial but hurt centred trials, so acquisition
# follows the observation symmetrically and clamps only at the tested bound.
PREGRASP_CROSS_AXIS_REFINEMENT_MAX_M = 0.010
SAFE_PREINSERT_Z_OFFSET_M = 0.010
SAFE_PREINSERT_POSITION_TOLERANCE_M = 0.004
# The insert template below is now the actual 180-episode stable-latch mean.
# Contact-load telemetry showed a repeatable 3.1 mm downward tracking error,
# so compensate it here.  Four millimetres is the smallest completion bound
# that remains reachable under contact; the old 25 mm default was unsafe.
SAFE_LATCH_Z_OFFSET_M = 0.003
SAFE_LATCH_POSITION_TOLERANCE_M = 0.004
# Runtime no longer drives the fully open fingers deeper through that complete
# development envelope.  It starts closing at the measured observation pose
# and simultaneously retracts to the RGB-D-retargeted latch.  A 2026-09-05
# seed-1104 ablation that forced the full 20 mm move into the demonstrated
# 11-frame close interval dropped evaluator IoU from 0.232 to 0.000.  The
# current camera refinement is only 8--12 mm, however, and holding it for a
# full second leaves a fully closed, laterally tilted gripper near the base
# for roughly half a second.  GT begins extraction by its q90 close-to-latch
# time (14 frames); use 0.50 s and hand directly to the continuous curve.
GRASP_REFINEMENT_MINIMUM_DURATION_S = 0.50
# Successful GT trajectories do not apply a long pure vertical pull.  At the
# first 5/10/15/20 mm height thresholds their median robot-forward motion is
# already 1.3/5.3/9.9/13.9 mm.  Use these dense samples in the continuous
# code-policy curve.  Runtime uses a conservative 10 mm vertical clearance to
# protect the lower finger before the elastic pad is peeled forward.
GT_EARLY_EXTRACT_FORWARD_Z_M = (
    # The lower finger is close to the base at this edge grasp.  Preserve a
    # short pure-vertical clearance before peeling forward; the prior 5 mm
    # point was visibly insufficient in GUI trials.
    (0.0, 0.010),
    (0.0098634884, 0.0167232156),
    (0.0139179278, 0.0214935541),
)
# The demonstrated one-frame peak is about 0.22 m/s, while the median transfer
# average is 0.134 m/s.  Use the demonstrated peak to bring a weak edge grasp
# onto target support before it can creep out of the fingers.
GT_CONTINUOUS_PEAK_LINEAR_SPEED_M_S = 0.22
GT_CONTINUOUS_RAMP_FRACTION = 0.12
# Target contact was advanced after weak-grasp trials, but the endpoint was
# accidentally left at 11.88 s.  Preserve the 5.40 s first supported contact
# and the labelled GT final intervals so the last waypoint reaches place in
# 0.25 s rather than a multi-second terminal crawl.
GT_STABLE_LATCH_TO_PLACE_DURATION_S = 6.20
# Cumulative times from the first stable latch to each continuous-path
# landmark.  Cross-seed telemetry showed weak edge grasps releasing before
# the GT-median target-contact time, so target approach is advanced while
# retaining the demonstrated extraction timing and 11.88 s endpoint.  The
# target support then addresses the remaining retention window.  These remain
# one C1 curve: the values are temporal knots, not stop-and-settle stages.
GT_POST_LATCH_LANDMARK_TIMES_S = (
    0.23,  # safe_vertical_z_10mm
    0.30,  # gt_first_z_15mm
    0.36,  # gt_first_z_20mm
    0.50,  # short_diagonal_clearance
    0.65,  # forward_rising_extract
    0.93,  # retained_lift
    1.50,  # forward_clear_base
    2.40,  # transfer
    3.60,  # gt_place_fraction_025 / target overhead
    4.00,  # gt_place_fraction_050
    4.40,  # gt_place_2s_before / first supported contact
    5.20,  # gt_place_1s_before
    5.70,  # gt_place_05s_before
    5.95,  # gt_place_025s_before
    6.20,  # support_place
)
# Median relative right-arm trajectory from the first stable measured latch
# through the first 5 mm of robot-left displacement in 180 successful
# development episodes.  The earlier ablation was aligned at the close-command
# edge and then incorrectly started after runtime had already latched, so it
# replayed the 11-frame close/retract segment and hit the base.  These knots
# start at the measured latch and therefore cover only the load-sensitive
# extraction.  The held-out episode rule is range(7, 200, 10).
# Source: evidence/phase2_gt_joint_extract_spline_20260905/audit.json.
GT_LATCH_TO_CLEAR_DURATION_S = 55.0 / 30.0
# Use the demonstrated q90 duration for the first loaded lateral sweep.  This
# keeps the C1 curve below the 0.232 m/s q90 one-frame peak while the total
# close-to-place time remains essentially the 12.25 s GT median.
GT_CLEAR_TO_TRANSFER_DURATION_S = 46.1 / 30.0
GT_TRANSFER_TO_PLACE_DURATION_S = 254.0 / 30.0
GT_POST_CLEAR_TO_PLACE_DURATION_S = (
    GT_CLEAR_TO_TRANSFER_DURATION_S + GT_TRANSFER_TO_PLACE_DURATION_S
)
GT_POST_CLEAR_LANDMARK_TIMES_S = (
    GT_CLEAR_TO_TRANSFER_DURATION_S,
    GT_CLEAR_TO_TRANSFER_DURATION_S
    + 0.25 * GT_TRANSFER_TO_PLACE_DURATION_S,
    GT_CLEAR_TO_TRANSFER_DURATION_S
    + 0.50 * GT_TRANSFER_TO_PLACE_DURATION_S,
    GT_POST_CLEAR_TO_PLACE_DURATION_S - 2.0,
    GT_POST_CLEAR_TO_PLACE_DURATION_S - 1.0,
    GT_POST_CLEAR_TO_PLACE_DURATION_S - 0.5,
    GT_POST_CLEAR_TO_PLACE_DURATION_S - 0.25,
    GT_POST_CLEAR_TO_PLACE_DURATION_S,
)
GT_LATCH_TO_CLEAR_RELATIVE_RIGHT_JOINT_SPLINE_RAD = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0103190541, 0.0155656755, 0.0101314068, -0.0112090230,
     0.0269078076, 0.0180838585, -0.0082885399),
    (0.0623361468, 0.0505748987, 0.0431873322, -0.0131279469,
     0.0944004059, 0.0602128029, -0.0258063614),
    (0.1267584771, 0.0880583346, 0.0841575861, 0.0009754062,
     0.1682611197, 0.1000462174, -0.0396936059),
    (0.1944717050, 0.1250526428, 0.1218301892, 0.0272950411,
     0.2391510218, 0.1321348190, -0.0512346327),
    (0.2621864676, 0.1580794454, 0.1541960239, 0.0647160411,
     0.2941718996, 0.1561447382, -0.0590465665),
    (0.3172250688, 0.1836814761, 0.1762328625, 0.1008313656,
     0.3367739648, 0.1733514786, -0.0656338155),
    (0.3563135207, 0.2005022645, 0.1889427304, 0.1280884743,
     0.3674714118, 0.1887296915, -0.0659409702),
    (0.3728989959, 0.2109172583, 0.1945464611, 0.1324551344,
     0.3807606906, 0.1985049248, -0.0664465249),
    (0.3794912338, 0.2169686258, 0.1958151579, 0.1347805500,
     0.3871035866, 0.2034144402, -0.0645358205),
    (0.3923246264, 0.2153655291, 0.1877651215, 0.1372026205,
     0.3830761760, 0.2166250944, -0.0664793849),
)
RMPFLOW_OWNERSHIP_HANDOFF_S = 0.30
# Median motion after the release-command edge across the same 180 successful
# development trajectories.  The fingers take ten frames to become fully
# open; during those frames GT raises the wrist slightly instead of opening
# at a fixed pose.  Once open, it clears toward robot-right and upward so a
# compliant pad cannot remain hooked on a fingertip.  Values are
# (robot-forward, world-z, robot-left) displacements from the release pose.
# See evidence/phase2_gt_release_20260905/audit_release.json.
GT_RELEASE_OPEN_FORWARD_Z_LEFT_M = (-0.0024928125, 0.0060508251, 0.0000472904)
GT_RELEASE_CLEAR_FORWARD_Z_LEFT_M = (
    (-0.0004594730, 0.0233875215, -0.0042378856),
    (0.0032553531, 0.0398969352, -0.0204392700),
    (0.0075160458, 0.0505843759, -0.0491492257),
)
GT_RELEASE_OPEN_DURATION_S = 10.0 / 30.0
GT_RELEASE_CLEAR_PEAK_LINEAR_SPEED_M_S = 0.12
RELEASE_CLEAR_MINIMUM_OPEN_FRACTION = 0.995
# Loaded RMPflow endpoints track 6--8 mm below the command during opening.
# Compensate that measured sag everywhere, with another 6 mm for the distant
# board slots whose arm configuration showed the largest downward error.
BASE_RELEASE_TRACKING_CLEARANCE_M = 0.010
LARGE_TARGET_ADDITIONAL_RELEASE_CLEARANCE_M = 0.006
SUPPORTED_ALIGNMENT_MAXIMUM_RAW_CROSS_AXIS_M = 0.025
# The wrist view loses one pad edge behind the gripper near support.  Seed
# 1001 therefore overestimated an otherwise correctable 10 mm cross-axis
# error and moved a correctly supported pad too far.  Apply at most 8 mm:
# this remains conservative relative to the 25 mm reliability gate, while a
# randomized target-B trial retained the pad but scored only 0.4551 after a
# 10.7 mm measured cross-axis residual was clipped to the former 5 mm limit.
SUPPORTED_ALIGNMENT_MAXIMUM_CROSS_AXIS_M = 0.008
SUPPORTED_ALIGNMENT_MAXIMUM_YAW_DEG = 15.0
# GT target-approach position medians from 180 successful demonstrations.
# Each row is (robot-forward from place, world-z from place, robot-left from
# place, fraction of the transfer-to-place quaternion rotation completed).
# The last value is the demonstrated fraction of the transfer-to-place wrist
# rotation already completed at that landmark.  Successful GT trajectories
# rotate gradually while approaching, but retain the final, fastest part of
# the roll for distal-edge contact and flattening over the target.  These
# fractions are derived from the median orientation errors in the same 180
# demonstrations; they are not hand-tuned endpoint guesses.
GT_PLACE_APPROACH_FORWARD_Z_LEFT_ROTATION = (
    (-0.0120268956, 0.0719026923, -0.0026403578, 0.174),
    (0.0062876744, 0.0450525284, 0.0005226137, 0.453),
    (-0.0070872901, 0.0245189965, 0.0008592675, 0.515),
    (-0.0063495401, 0.0185534656, 0.0011443082, 0.739),
    (-0.0044487587, 0.0115554929, 0.0008750833, 0.864),
    (-0.0000141955, 0.0078603625, 0.0004899347, 0.935),
)
# Start the long transfer with the same forward tangent as extraction, then
# bend gradually into robot-left motion.  This removes the 90-degree command
# corner that can shear a deformable edge grasp even at modest peak speed.
TRANSFER_BLEND_FORWARD_TANGENT_M = 0.030
TRANSFER_BLEND_LEFT_TANGENT_M = 0.040
MINIMUM_SUPPORT_EE_Z_M = 0.903
SUPPORT_EE_Z_TOLERANCE_M = 0.001
BLUE_PAD_MIN_PIXELS = 200
# At target overhead the closed gripper occludes most of the retained pad.
# The independent evaluator confirmed a real grasp with 119 visible wrist
# pixels, so use a lower geometry threshold there while retaining the strict
# 200-pixel pre-grasp depth-signature gate above.
WORLD_PAD_MIN_PIXELS = 80
RED_TARGET_MIN_PIXELS = 80
GRASP_PROBE_MAX_MEDIAN_DEPTH_DRIFT_M = 0.015
GRASP_PROBE_MIN_PIXEL_AREA_RATIO = 1.20
PAD_RETENTION_MAX_WORLD_ERROR_M = 0.05
# A pad whose visible centre is already near the target/table surface but more
# than 35 mm from the requested target centre has been lost before release.
# Opening and descending after that point cannot recover the task and has
# repeatedly swept the lower finger into the target board.  This is a strict
# safety veto only; a pad still elevated with the gripper is allowed through.
PRE_RELEASE_DROPPED_PAD_MAX_SURFACE_DELTA_M = 0.035
PRE_RELEASE_DROPPED_PAD_MIN_TARGET_XY_ERROR_M = 0.035
REFERENCE_TARGET_CENTER_WORLD_M = (2.1500000954, 1.9500000477, 0.75)
REFERENCE_TARGET_YAW_RAD = math.pi / 2
# The authored Task-2 board slots are x={1.95, 2.05, 2.15, 2.25} m and the
# reference target is at x=2.15 m.  Cover the full 20 cm slot swap plus the
# launcher's bounded 1 cm XY jitter, without allowing an unconstrained
# camera detection to redirect the arm elsewhere in the cell.
MAXIMUM_TARGET_RETARGET_XY_M = 0.215
MAXIMUM_TARGET_RETARGET_Z_M = 0.03
MAXIMUM_TARGET_RETARGET_YAW_DEG = 10.0
LARGE_TARGET_RETARGET_XY_M = 0.05
LARGE_TARGET_CONTACT_TOLERANCE_M = 0.020
PAD_PLACEMENT_CENTER_Z_OFFSET_M = 0.0055
MAXIMUM_GRASP_RELATIVE_PLACE_XY_CORRECTION_M = 0.08
MAXIMUM_GRASP_RELATIVE_PLACE_Z_CORRECTION_M = 0.03
REFERENCE_VISIBLE_PAD_CENTER_WORLD_M = (
    1.7498657282,
    1.9606426936,
    0.8500623163,
)
REFERENCE_VISIBLE_PAD_YAW_RAD = math.pi / 2
REFERENCE_CODE_INSERT_POSE_WORLD_XYZW = (
    # Planar contact comes from the stable-latch mean across 180 successful
    # development episodes.  Runtime applies it as a pad-relative RGB-D
    # template, not an absolute trajectory replay.  Keep the matching Markley
    # mean carry attitude: transplanting the Formula-3 VLA handoff quaternion
    # made the pad fall about 107 mm earlier, showing that its success depended
    # on the full VLA contact state rather than that quaternion alone.
    1.7503474063,
    2.1552703977,
    # This must match the actual 180-episode Markley-mean stable latch.  The
    # previous 0.8681 m value was 5.18 mm below that statistic and caused the
    # lower finger to strike the pad base across perturbed seeds.
    0.8732815097,
    -0.0297836509,
    0.7328201582,
    -0.6796349563,
    -0.0135600884,
)
# Do not reproduce the unsafe -10 mm or -4 mm depth ablations: both visibly
# over-inserted and the latter moved the pad off-board.  Keep the camera-
# retargeted GT stable-latch depth; retention must be improved without forcing
# the wrist farther into the base.
CODE_POLICY_INSERT_DEPTH_BIAS_WORLD_Y_M = 0.0
# A +10 mm bias produced an actual latch around z=0.872 m and the compliant
# edge slid out early.  Reducing it to +5 mm moved the measured slip point
# roughly 60 mm farther toward the target, but still lost the pad in the
# second lateral segment.  Use the original z=0.8681 m command: the expected
# actual latch is near the successful Formula-3 pose and the q10 band of 180
# GT successes.  The 10 mm pure-vertical clearance above protects the lower
# finger before any forward peel.
CODE_POLICY_INSERT_Z_BIAS_M = 0.0
MAXIMUM_PAD_RETARGET_XY_M = 0.05
MAXIMUM_PAD_RETARGET_Z_M = 0.03
MAXIMUM_PAD_RETARGET_YAW_DEG = 10.0


def blue_pad_mask(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must be HxWx3")
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    return (
        (blue >= 90)
        & (blue - red >= 35)
        & (blue - green >= 15)
    )


def red_target_mask(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must be HxWx3")
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    return (red >= 95) & (red - green >= 25) & (red - blue >= 25)


def blue_pad_depth_signature(
    rgb: np.ndarray, depth_m: np.ndarray
) -> dict[str, float | int]:
    """Describe the blue pad in wrist-camera coordinates without task GT."""

    image = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != image.shape[:2]:
        raise ValueError("RGB and depth shapes must match")
    mask = (
        blue_pad_mask(image)
        & np.isfinite(depth)
        & (depth >= 0.03)
        & (depth <= 1.50)
    )
    rows, columns = np.nonzero(mask)
    if len(rows) < BLUE_PAD_MIN_PIXELS:
        raise ValueError(f"insufficient blue pad pixels: {len(rows)}")
    selected_depth = depth[mask]
    height, width = depth.shape
    return {
        "pixel_count": int(len(rows)),
        "median_depth_m": float(np.median(selected_depth)),
        "q10_depth_m": float(np.quantile(selected_depth, 0.10)),
        "q90_depth_m": float(np.quantile(selected_depth, 0.90)),
        "centroid_u_fraction": float(np.median(columns) / width),
        "centroid_v_fraction": float(np.median(rows) / height),
    }


def quaternion_xyzw_to_matrix(quaternion: tuple[float, ...]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((x, y, z, w)))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def cubic_bezier_position(
    start: tuple[float, ...],
    target: tuple[float, ...],
    start_tangent: tuple[float, ...],
    end_tangent: tuple[float, ...],
    fraction: float,
) -> tuple[float, float, float]:
    """Interpolate xyz with endpoint tangents expressed in world metres."""

    value = max(0.0, min(1.0, float(fraction)))
    p0 = np.asarray(start[:3], dtype=np.float64)
    p1 = p0 + np.asarray(start_tangent, dtype=np.float64)
    p3 = np.asarray(target[:3], dtype=np.float64)
    p2 = p3 - np.asarray(end_tangent, dtype=np.float64)
    inverse = 1.0 - value
    position = (
        inverse**3 * p0
        + 3.0 * inverse**2 * value * p1
        + 3.0 * inverse * value**2 * p2
        + value**3 * p3
    )
    return tuple(float(component) for component in position)


def continuous_landmark_pose(
    poses: list[tuple[float, ...]],
    fraction: float,
    orientation_scale_m_per_rad: float = 0.0,
    landmark_fractions: list[float] | tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Interpolate a C1 xyz curve through poses without stopping at landmarks.

    Chord length parameterisation keeps velocity approximately uniform by
    default.  Explicit temporal landmark fractions can instead reproduce the
    demonstrated non-uniform timing.  Cubic Hermite tangents are shared by
    adjacent segments, so either mode constrains geometry without turning the
    landmarks into controller stages with zero endpoint velocity.
    """

    if len(poses) < 2:
        raise ValueError("continuous pose path requires at least two poses")
    values = [np.asarray(pose, dtype=np.float64) for pose in poses]
    if any(value.shape != (7,) for value in values):
        raise ValueError("continuous pose path entries must be 7D poses")
    points = np.asarray([value[:3] for value in values], dtype=np.float64)
    linear_chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    quaternions = np.asarray([value[3:7] for value in values])
    quaternion_dots = np.abs(
        np.sum(quaternions[:-1] * quaternions[1:], axis=1)
    )
    angular_chords = 2.0 * np.arccos(np.clip(quaternion_dots, 0.0, 1.0))
    metric_chords = np.maximum(
        linear_chords,
        float(orientation_scale_m_per_rad) * angular_chords,
    )
    if np.any(metric_chords <= 1.0e-9):
        raise ValueError("continuous pose path has duplicate xyz landmarks")
    if landmark_fractions is None:
        distances = np.concatenate(([0.0], np.cumsum(metric_chords)))
    else:
        distances = np.asarray(landmark_fractions, dtype=np.float64)
        if distances.shape != (len(poses),):
            raise ValueError(
                "continuous pose path landmark fractions must match poses"
            )
        if not np.isclose(distances[0], 0.0) or not np.isclose(
            distances[-1], 1.0
        ):
            raise ValueError(
                "continuous pose path landmark fractions must span 0 to 1"
            )
        if np.any(np.diff(distances) <= 1.0e-9):
            raise ValueError(
                "continuous pose path landmark fractions must increase"
            )
    chords = np.diff(distances)
    total = float(distances[-1])
    distance = max(0.0, min(1.0, float(fraction))) * total
    segment = min(
        int(np.searchsorted(distances, distance, side="right") - 1),
        len(values) - 2,
    )
    segment = max(0, segment)
    start_distance = float(distances[segment])
    segment_length = float(chords[segment])
    local = (distance - start_distance) / segment_length

    tangents = np.empty_like(points)
    tangents[0] = (points[1] - points[0]) / chords[0]
    tangents[-1] = (points[-1] - points[-2]) / chords[-1]
    for index in range(1, len(points) - 1):
        tangents[index] = (
            points[index + 1] - points[index - 1]
        ) / (distances[index + 1] - distances[index - 1])

    local2 = local * local
    local3 = local2 * local
    h00 = 2.0 * local3 - 3.0 * local2 + 1.0
    h10 = local3 - 2.0 * local2 + local
    h01 = -2.0 * local3 + 3.0 * local2
    h11 = local3 - local2
    position = (
        h00 * points[segment]
        + h10 * segment_length * tangents[segment]
        + h01 * points[segment + 1]
        + h11 * segment_length * tangents[segment + 1]
    )
    orientation = _slerp(
        values[segment][3:7], values[segment + 1][3:7], local
    )
    return (
        *(float(component) for component in position),
        *(float(component) for component in orientation),
    )


def pose_path_length(
    poses: list[tuple[float, ...]],
    orientation_scale_m_per_rad: float = 0.0,
) -> float:
    """Return a per-segment translation/orientation-equivalent path length."""

    if len(poses) < 2:
        return 0.0
    values = np.asarray(poses, dtype=np.float64)
    linear = np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1)
    quaternion_dots = np.abs(
        np.sum(values[:-1, 3:7] * values[1:, 3:7], axis=1)
    )
    angular = 2.0 * np.arccos(np.clip(quaternion_dots, 0.0, 1.0))
    return float(
        np.maximum(
            linear,
            float(orientation_scale_m_per_rad) * angular,
        ).sum()
    )


def continuous_joint_spline(
    knots: tuple[tuple[float, ...], ...] | list[tuple[float, ...]],
    fraction: float,
) -> tuple[float, ...]:
    """Evaluate a uniform C1 cubic Hermite spline in joint space."""

    values = np.asarray(knots, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("joint spline requires at least two vector knots")
    if not np.all(np.isfinite(values)):
        raise ValueError("joint spline contains non-finite values")
    progress = max(0.0, min(1.0, float(fraction))) * (len(values) - 1)
    segment = min(int(math.floor(progress)), len(values) - 2)
    local = progress - segment
    tangents = np.empty_like(values)
    tangents[0] = values[1] - values[0]
    tangents[-1] = values[-1] - values[-2]
    tangents[1:-1] = 0.5 * (values[2:] - values[:-2])
    local2 = local * local
    local3 = local2 * local
    result = (
        (2.0 * local3 - 3.0 * local2 + 1.0) * values[segment]
        + (local3 - 2.0 * local2 + local) * tangents[segment]
        + (-2.0 * local3 + 3.0 * local2) * values[segment + 1]
        + (local3 - local2) * tangents[segment + 1]
    )
    return tuple(float(value) for value in result)


def short_ramp_fraction(fraction: float, ramp_fraction: float) -> float:
    """Integrate a raised-cosine velocity ramp with a constant-speed middle."""

    value = max(0.0, min(1.0, float(fraction)))
    ramp = float(ramp_fraction)
    if not 0.0 < ramp < 0.5:
        raise ValueError("ramp fraction must be between zero and one half")
    normalizer = 1.0 - ramp
    if value < ramp:
        area = 0.5 * (
            value - ramp / math.pi * math.sin(math.pi * value / ramp)
        )
        return area / normalizer
    if value > 1.0 - ramp:
        return 1.0 - short_ramp_fraction(1.0 - value, ramp)
    return (value - 0.5 * ramp) / normalizer


def deproject_masked_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: tuple[float, ...],
    camera_pose_xyzw: tuple[float, ...],
    *,
    minimum_depth_m: float,
    maximum_depth_m: float,
    stride: int = 2,
) -> np.ndarray:
    """Project masked ROS optical-frame depth samples into world XYZ."""

    depth = np.asarray(depth_m, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or selected.shape != depth.shape:
        raise ValueError("depth and mask must be matching HxW arrays")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if min(fx, fy) <= 0.0:
        raise ValueError("invalid camera intrinsics")
    valid = (
        selected
        & np.isfinite(depth)
        & (depth >= minimum_depth_m)
        & (depth <= maximum_depth_m)
    )
    rows, columns = np.nonzero(valid)
    if stride > 1:
        rows, columns = rows[::stride], columns[::stride]
    if len(columns) < 8:
        raise ValueError("insufficient masked depth pixels")
    z = depth[rows, columns]
    optical = np.column_stack(
        ((columns - cx) * z / fx, (rows - cy) * z / fy, z)
    )
    pose = np.asarray(camera_pose_xyzw, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError("camera pose must be xyz+xyzw")
    rotation = quaternion_xyzw_to_matrix(tuple(pose[3:]))
    return optical @ rotation.T + pose[:3]


def robust_world_surface_signature(
    points_world: np.ndarray,
) -> dict[str, float | int | list[float]]:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError("at least eight world points are required")
    lower = np.quantile(points, 0.10, axis=0)
    upper = np.quantile(points, 0.90, axis=0)
    clipped = points[np.all((points >= lower) & (points <= upper), axis=1)]
    if len(clipped) < 8:
        clipped = points
    median_center = np.median(clipped, axis=0)
    xy = clipped[:, :2] - median_center[:2]
    covariance = xy.T @ xy / max(1, len(xy) - 1)
    values, vectors = np.linalg.eigh(covariance)
    major_axis = vectors[:, int(np.argmax(values))]
    minor_axis = vectors[:, int(np.argmin(values))]
    major_projection = points[:, :2] @ major_axis
    minor_projection = points[:, :2] @ minor_axis
    major_bounds = np.quantile(major_projection, (0.02, 0.98))
    minor_bounds = np.quantile(minor_projection, (0.02, 0.98))
    geometric_xy = (
        major_axis * float(np.mean(major_bounds))
        + minor_axis * float(np.mean(minor_bounds))
    )
    geometric_center = np.array(
        [geometric_xy[0], geometric_xy[1], np.median(clipped[:, 2])]
    )
    yaw = math.atan2(float(major_axis[1]), float(major_axis[0]))
    return {
        "point_count": int(len(points)),
        "robust_point_count": int(len(clipped)),
        "center_world_m": geometric_center.tolist(),
        "visible_median_world_m": median_center.tolist(),
        "world_q02_m": np.quantile(points, 0.02, axis=0).tolist(),
        "world_q98_m": np.quantile(points, 0.98, axis=0).tolist(),
        "major_visible_extent_m": float(major_bounds[1] - major_bounds[0]),
        "minor_visible_extent_m": float(minor_bounds[1] - minor_bounds[0]),
        "long_axis_yaw_rad_mod_pi": yaw,
    }


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _rectangle_yaw_delta(actual: float, reference: float) -> float:
    delta = _wrap_angle(actual - reference)
    if delta > math.pi / 2:
        delta -= math.pi
    elif delta < -math.pi / 2:
        delta += math.pi
    return delta


def _rotate_quaternion_about_world_z(
    quaternion: tuple[float, ...], yaw_delta: float
) -> tuple[float, ...]:
    x, y, z, w = (float(value) for value in quaternion)
    half = 0.5 * yaw_delta
    rz, rw = math.sin(half), math.cos(half)
    result = np.asarray(
        (
            rw * x - rz * y,
            rw * y + rz * x,
            rw * z + rz * w,
            rw * w - rz * z,
        ),
        dtype=np.float64,
    )
    result /= np.linalg.norm(result)
    return tuple(float(value) for value in result)


def retarget_transport_plan_to_target(
    plan: list[dict[str, Any]], target: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply bounded camera-only target correction after a verified grasp."""

    center = np.asarray(target["center_world_m"], dtype=np.float64)
    reference = np.asarray(REFERENCE_TARGET_CENTER_WORLD_M, dtype=np.float64)
    delta = center - reference
    xy_norm = float(np.linalg.norm(delta[:2]))
    yaw_delta = _rectangle_yaw_delta(
        float(target["long_axis_yaw_rad_mod_pi"]), REFERENCE_TARGET_YAW_RAD
    )
    if xy_norm > MAXIMUM_TARGET_RETARGET_XY_M:
        raise ValueError(f"target_xy_retarget_exceeds_bound:{xy_norm:.6f}")
    if abs(float(delta[2])) > MAXIMUM_TARGET_RETARGET_Z_M:
        raise ValueError(
            f"target_z_retarget_exceeds_bound:{float(delta[2]):.6f}"
        )
    if abs(math.degrees(yaw_delta)) > MAXIMUM_TARGET_RETARGET_YAW_DEG:
        raise ValueError(
            "target_yaw_retarget_exceeds_bound:"
            f"{math.degrees(yaw_delta):.6f}"
        )

    weights = {
        "transfer": 0.5,
        "target_overhead": 1.0,
        "support_contact": 1.0,
        "support_precontact": 1.0,
        "support_place": 1.0,
        "release": 1.0,
        "retreat": 1.0,
    }
    retargeted: list[dict[str, Any]] = []
    for stage in plan:
        copied = dict(stage)
        name = str(stage["name"])

        def transform(
            pose_values: Any, transform_weight: float
        ) -> tuple[float, ...]:
            pose = tuple(float(value) for value in pose_values)
            return (
                pose[0] + transform_weight * float(delta[0]),
                pose[1] + transform_weight * float(delta[1]),
                pose[2] + transform_weight * float(delta[2]),
                *_rotate_quaternion_about_world_z(
                    pose[3:7], transform_weight * yaw_delta
                ),
            )

        if name == "smooth_transport_to_place":
            path_names = list(stage["path_landmark_names"])
            copied["pose"] = transform(stage["pose"], 1.0)
            copied["right_pose_path"] = [
                transform(pose, 0.5 if path_name == "transfer" else 1.0)
                for path_name, pose in zip(
                    path_names, stage["right_pose_path"], strict=True
                )
            ]
        else:
            weight = weights.get(name, 0.0)
            if weight:
                copied["pose"] = transform(stage["pose"], weight)
                if "right_pose_path" in stage:
                    copied["right_pose_path"] = [
                        transform(pose, weight)
                        for pose in stage["right_pose_path"]
                    ]
        if name in {"support_place", "release"}:
            # Preserve the measured slot displacement through the later
            # collapse into one continuous trajectory.  The furthest board
            # slot is close to the arm's contact-loaded reach boundary and
            # settles around 18 mm from its Cartesian command; nominal runs
            # keep their proven 10 mm release gate.
            copied["target_retarget_xy_m"] = xy_norm
        retargeted.append(copied)
    audit = {
        "source": "head_rgbd_optical_fk_red_target",
        "reference_target_center_world_m": list(
            REFERENCE_TARGET_CENTER_WORLD_M
        ),
        "observed_target_center_world_m": center.tolist(),
        "translation_delta_world_m": delta.tolist(),
        "translation_xy_norm_m": xy_norm,
        "yaw_delta_deg": math.degrees(yaw_delta),
        "bounds": {
            "maximum_xy_m": MAXIMUM_TARGET_RETARGET_XY_M,
            "maximum_z_m": MAXIMUM_TARGET_RETARGET_Z_M,
            "maximum_yaw_deg": MAXIMUM_TARGET_RETARGET_YAW_DEG,
        },
    }
    return retargeted, audit


def retarget_place_from_observed_grasp(
    plan: list[dict[str, Any]],
    current_ee_world_xyzw: tuple[float, ...],
    observed_pad: dict[str, Any],
    observed_target: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Center placement using the camera-measured pad-to-EE transform."""

    ee = np.asarray(current_ee_world_xyzw, dtype=np.float64)
    pad_center = np.asarray(observed_pad["center_world_m"], dtype=np.float64)
    target_center = np.asarray(
        observed_target["center_world_m"], dtype=np.float64
    )
    local_pad_offset = (
        quaternion_xyzw_to_matrix(tuple(ee[3:7])).T
        @ (pad_center - ee[:3])
    )
    pad_yaw_correction = _rectangle_yaw_delta(
        float(observed_target["long_axis_yaw_rad_mod_pi"]),
        float(observed_pad["long_axis_yaw_rad_mod_pi"]),
    )
    if abs(math.degrees(pad_yaw_correction)) > MAXIMUM_TARGET_RETARGET_YAW_DEG:
        raise ValueError(
            "grasp_relative_place_yaw_exceeds_bound:"
            f"{math.degrees(pad_yaw_correction):.6f}"
        )

    stage_by_name = {str(stage["name"]): stage for stage in plan}
    nominal_place = tuple(stage_by_name["support_place"]["pose"])
    corrected_place_orientation = _rotate_quaternion_about_world_z(
        nominal_place[3:7], pad_yaw_correction
    )
    placed_pad_offset = quaternion_xyzw_to_matrix(
        corrected_place_orientation
    ) @ local_pad_offset
    desired_pad_center = target_center + np.asarray(
        (0.0, 0.0, PAD_PLACEMENT_CENTER_Z_OFFSET_M), dtype=np.float64
    )
    rigid_corrected_place_xyz = desired_pad_center - placed_pad_offset
    rigid_correction = rigid_corrected_place_xyz - np.asarray(nominal_place[:3])
    # The flexible pad preserves neither the vertical nor planar pre-grasp
    # EE-to-pad transform after extraction and the ~68-degree placement roll.
    # Applying that rigid transform moved the seed-2217 wrist 31.5 mm away
    # from the GT placement line and the flattened pad ended 77.8 mm from the
    # target centre.  Target RGB-D retargeting already owns XYZ; retain only
    # the observed pad-to-target yaw correction here.
    correction = np.zeros(3, dtype=np.float64)
    corrected_place_xyz = np.asarray(nominal_place[:3]) + correction
    correction_xy = float(np.linalg.norm(correction[:2]))
    if correction_xy > MAXIMUM_GRASP_RELATIVE_PLACE_XY_CORRECTION_M:
        raise ValueError(
            "grasp_relative_place_xy_exceeds_bound:"
            f"{correction_xy:.6f}"
        )
    if abs(float(rigid_correction[2])) > MAXIMUM_GRASP_RELATIVE_PLACE_Z_CORRECTION_M:
        raise ValueError(
            "grasp_relative_place_z_exceeds_bound:"
            f"{float(rigid_correction[2]):.6f}"
        )

    corrected: list[dict[str, Any]] = []
    downstream = {
        "support_contact",
        "support_precontact",
        "support_place",
        "release",
        "retreat",
    }
    for stage in plan:
        copied = dict(stage)
        if str(stage["name"]) in downstream:
            def transform(pose_values: Any) -> tuple[float, ...]:
                pose = tuple(float(value) for value in pose_values)
                return (
                    pose[0] + float(correction[0]),
                    pose[1] + float(correction[1]),
                    pose[2] + float(correction[2]),
                    *_rotate_quaternion_about_world_z(
                        pose[3:7], pad_yaw_correction
                    ),
                )

            copied["pose"] = transform(stage["pose"])
            if "right_pose_path" in stage:
                copied["right_pose_path"] = [
                    transform(pose) for pose in stage["right_pose_path"]
                ]
        corrected.append(copied)
    return corrected, {
        "source": "target_rgbd_xyz_plus_pad_to_target_yaw",
        "observed_ee_world_xyzw": ee.tolist(),
        "observed_pad_center_world_m": pad_center.tolist(),
        "observed_target_center_world_m": target_center.tolist(),
        "local_pad_offset_in_ee_m": local_pad_offset.tolist(),
        "predicted_pad_offset_at_place_world_m": placed_pad_offset.tolist(),
        "desired_pad_center_world_m": desired_pad_center.tolist(),
        "nominal_place_xyz_m": list(nominal_place[:3]),
        "rigid_pad_offset_place_xyz_m": rigid_corrected_place_xyz.tolist(),
        "rigid_pad_offset_z_correction_ignored_m": float(rigid_correction[2]),
        "rigid_pad_offset_xy_correction_ignored_m": (
            rigid_correction[:2].tolist()
        ),
        "corrected_place_xyz_m": corrected_place_xyz.tolist(),
        "place_translation_correction_world_m": correction.tolist(),
        "place_translation_correction_xy_m": correction_xy,
        "pad_yaw_correction_deg": math.degrees(pad_yaw_correction),
        "bounds": {
            "maximum_xy_m": MAXIMUM_GRASP_RELATIVE_PLACE_XY_CORRECTION_M,
            "maximum_z_m": MAXIMUM_GRASP_RELATIVE_PLACE_Z_CORRECTION_M,
            "maximum_yaw_deg": MAXIMUM_TARGET_RETARGET_YAW_DEG,
        },
    }


def pad_retention_metrics(
    reference_pad_center_world_m: tuple[float, ...],
    reference_ee_world_xyzw: tuple[float, ...],
    observed_pad_center_world_m: tuple[float, ...],
    current_ee_world_xyzw: tuple[float, ...],
    maximum_world_error_m: float = PAD_RETENTION_MAX_WORLD_ERROR_M,
) -> dict[str, Any]:
    """Check that the visually measured pad still follows the gripper.

    The reference is captured immediately after the verified probe lift.  A
    rigid prediction is intentionally used only as a coarse retention gate;
    the deformable pad and changing self-occlusion make it unsuitable as a
    millimetre-accurate placement estimator.
    """

    reference_pad = np.asarray(reference_pad_center_world_m, dtype=np.float64)
    reference_ee = np.asarray(reference_ee_world_xyzw, dtype=np.float64)
    observed_pad = np.asarray(observed_pad_center_world_m, dtype=np.float64)
    current_ee = np.asarray(current_ee_world_xyzw, dtype=np.float64)
    if any(value.shape != expected for value, expected in (
        (reference_pad, (3,)),
        (reference_ee, (7,)),
        (observed_pad, (3,)),
        (current_ee, (7,)),
    )):
        raise ValueError("invalid pad retention geometry")
    reference_rotation = quaternion_xyzw_to_matrix(tuple(reference_ee[3:]))
    current_rotation = quaternion_xyzw_to_matrix(tuple(current_ee[3:]))
    pad_offset_ee = reference_rotation.T @ (
        reference_pad - reference_ee[:3]
    )
    predicted_pad = current_ee[:3] + current_rotation @ pad_offset_ee
    error_m = float(np.linalg.norm(observed_pad - predicted_pad))
    if not math.isfinite(maximum_world_error_m) or maximum_world_error_m <= 0:
        raise ValueError("invalid pad retention error bound")
    return {
        "passed": error_m <= maximum_world_error_m,
        "world_error_m": error_m,
        "maximum_world_error_m": maximum_world_error_m,
        "predicted_pad_center_world_m": predicted_pad.tolist(),
        "observed_pad_center_world_m": observed_pad.tolist(),
        "reference_pad_offset_ee_m": pad_offset_ee.tolist(),
    }


def pre_release_pad_safety(
    observed_pad: dict[str, Any],
    desired_target: dict[str, Any],
) -> dict[str, Any]:
    """Veto release when RGB-D shows the pad already dropped off target."""

    pad_center = np.asarray(observed_pad["center_world_m"], dtype=np.float64)
    target_center = np.asarray(
        desired_target["center_world_m"], dtype=np.float64
    )
    if pad_center.shape != (3,) or target_center.shape != (3,):
        raise ValueError("invalid pre-release pad geometry")
    xy_error_m = float(np.linalg.norm(pad_center[:2] - target_center[:2]))
    surface_delta_m = float(pad_center[2] - target_center[2])
    pad_is_on_surface = (
        abs(surface_delta_m)
        <= PRE_RELEASE_DROPPED_PAD_MAX_SURFACE_DELTA_M
    )
    pad_is_outside_target = (
        xy_error_m
        > PRE_RELEASE_DROPPED_PAD_MIN_TARGET_XY_ERROR_M
    )
    release_permitted = not (
        pad_is_on_surface and pad_is_outside_target
    )
    return {
        "release_permitted": release_permitted,
        "pad_is_on_surface": pad_is_on_surface,
        "pad_is_outside_target": pad_is_outside_target,
        "target_xy_error_m": xy_error_m,
        "target_surface_delta_m": surface_delta_m,
        "maximum_surface_delta_m": (
            PRE_RELEASE_DROPPED_PAD_MAX_SURFACE_DELTA_M
        ),
        "minimum_outside_target_xy_error_m": (
            PRE_RELEASE_DROPPED_PAD_MIN_TARGET_XY_ERROR_M
        ),
    }


def supported_pad_alignment(
    current_ee: tuple[float, ...],
    observed_pad: dict[str, Any],
    desired_target: dict[str, Any],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Align a supported pad from RGB-D before opening the gripper."""

    pad_center = np.asarray(observed_pad["center_world_m"], dtype=np.float64)
    target_center = np.asarray(
        desired_target["center_world_m"], dtype=np.float64
    )
    translation_xy = target_center[:2] - pad_center[:2]
    translation_norm = float(np.linalg.norm(translation_xy))
    yaw_error = _rectangle_yaw_delta(
        float(observed_pad["long_axis_yaw_rad_mod_pi"]),
        float(desired_target["long_axis_yaw_rad_mod_pi"]),
    )
    yaw_reliable = (
        int(observed_pad.get("pixel_count", 0)) >= 1000
        and float(observed_pad.get("major_visible_extent_m", 0.0)) >= 0.05
        and abs(math.degrees(yaw_error))
        <= SUPPORTED_ALIGNMENT_MAXIMUM_YAW_DEG
    )
    applied_yaw = -yaw_error if yaw_reliable else 0.0
    target_yaw = float(desired_target["long_axis_yaw_rad_mod_pi"])
    target_cross_axis = np.asarray(
        (-math.sin(target_yaw), math.cos(target_yaw)), dtype=np.float64
    )
    raw_cross_axis_m = float(translation_xy @ target_cross_axis)
    cross_axis_reliable = (
        int(observed_pad.get("pixel_count", 0)) >= 1000
        and float(observed_pad.get("major_visible_extent_m", 0.0)) >= 0.05
        # The gripper hides one long edge, so the total XY norm contains an
        # explicitly ignored longitudinal bias.  Gate the independently
        # observable perpendicular component instead.  Seed-1106 r17 had a
        # valid 11.7 mm cross-axis correction rejected only because the
        # occluded long-axis component raised the total norm to 30.5 mm.
        and abs(raw_cross_axis_m)
        <= SUPPORTED_ALIGNMENT_MAXIMUM_RAW_CROSS_AXIS_M
    )
    applied_cross_axis_m = (
        float(
            np.clip(
                raw_cross_axis_m,
                -SUPPORTED_ALIGNMENT_MAXIMUM_CROSS_AXIS_M,
                SUPPORTED_ALIGNMENT_MAXIMUM_CROSS_AXIS_M,
            )
        )
        if cross_axis_reliable
        else 0.0
    )
    applied_translation_xy = applied_cross_axis_m * target_cross_axis
    # At placement the gripper occludes one end of the pad, so the visible
    # centroid is biased along the pad long axis.  Its perpendicular component
    # remains observable from both long edges, so correct only that bounded
    # cross-axis error and reject the occluded longitudinal component.
    pose = (
        float(current_ee[0] + applied_translation_xy[0]),
        float(current_ee[1] + applied_translation_xy[1]),
        float(current_ee[2]),
        *_rotate_quaternion_about_world_z(current_ee[3:7], applied_yaw),
    )
    return pose, {
        "source": "post_place_wrist_rgbd_pad_to_pregrasp_head_rgbd_target",
        "observed_pad_center_world_m": pad_center.tolist(),
        "desired_target_center_world_m": target_center.tolist(),
        "translation_xy_m": translation_xy.tolist(),
        "translation_xy_norm_m": translation_norm,
        "target_cross_axis_xy": target_cross_axis.tolist(),
        "raw_cross_axis_translation_m": raw_cross_axis_m,
        "maximum_raw_cross_axis_translation_m": (
            SUPPORTED_ALIGNMENT_MAXIMUM_RAW_CROSS_AXIS_M
        ),
        "applied_cross_axis_translation_m": applied_cross_axis_m,
        "applied_translation_xy_m": applied_translation_xy.tolist(),
        "cross_axis_reliable": cross_axis_reliable,
        "translation_applied": bool(abs(applied_cross_axis_m) > 0.0),
        "translation_ignored_component": (
            "target_long_axis_gripper_occludes_pad_endpoint"
        ),
        "pad_yaw_error_deg": math.degrees(yaw_error),
        "yaw_reliable": yaw_reliable,
        "commanded_wrist_yaw_correction_deg": math.degrees(applied_yaw),
    }


def build_code_policy_acquire_plan(
    reference: dict[str, Any],
    perception: dict[str, Any],
    measured_pregrasp_pose: tuple[float, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a bounded visual insert/close/probe/transport program."""

    observed = np.asarray(
        perception["pad"]["center_world_m"], dtype=np.float64
    )
    baseline = np.asarray(
        REFERENCE_VISIBLE_PAD_CENTER_WORLD_M, dtype=np.float64
    )
    delta = observed - baseline
    xy_norm = float(np.linalg.norm(delta[:2]))
    yaw_delta = _rectangle_yaw_delta(
        float(perception["pad"]["long_axis_yaw_rad_mod_pi"]),
        REFERENCE_VISIBLE_PAD_YAW_RAD,
    )
    if xy_norm > MAXIMUM_PAD_RETARGET_XY_M:
        raise ValueError(f"pad_xy_retarget_exceeds_bound:{xy_norm:.6f}")
    if abs(float(delta[2])) > MAXIMUM_PAD_RETARGET_Z_M:
        raise ValueError(
            f"pad_z_retarget_exceeds_bound:{float(delta[2]):.6f}"
        )
    if abs(math.degrees(yaw_delta)) > MAXIMUM_PAD_RETARGET_YAW_DEG:
        raise ValueError(
            "pad_yaw_retarget_exceeds_bound:"
            f"{math.degrees(yaw_delta):.6f}"
        )
    measured_pregrasp: tuple[float, ...] | None = None
    if measured_pregrasp_pose is not None:
        measured_pregrasp = tuple(
            float(value) for value in measured_pregrasp_pose
        )
        if len(measured_pregrasp) != 7 or not all(
            math.isfinite(value) for value in measured_pregrasp
        ):
            raise ValueError("invalid_measured_pregrasp_pose")
    insert_reference = np.asarray(
        REFERENCE_CODE_INSERT_POSE_WORLD_XYZW, dtype=np.float64
    )
    reference_offset_xy = insert_reference[:2] - baseline[:2]
    cosine = math.cos(yaw_delta)
    sine = math.sin(yaw_delta)
    rotated_offset_xy = np.asarray(
        (
            cosine * reference_offset_xy[0]
            - sine * reference_offset_xy[1],
            sine * reference_offset_xy[0]
            + cosine * reference_offset_xy[1],
        ),
        dtype=np.float64,
    )
    close_approach_xy = -rotated_offset_xy / np.linalg.norm(rotated_offset_xy)
    grasp_cross_axis_xy = np.asarray(
        (-close_approach_xy[1], close_approach_xy[0]), dtype=np.float64
    )
    observed_cross_axis_refinement_m = float(
        delta[:2] @ grasp_cross_axis_xy
    )
    pregrasp_cross_axis_refinement_m = float(
        np.clip(
            observed_cross_axis_refinement_m,
            -PREGRASP_CROSS_AXIS_REFINEMENT_MAX_M,
            PREGRASP_CROSS_AXIS_REFINEMENT_MAX_M,
        )
    )
    cross_axis_clamp_delta_m = (
        pregrasp_cross_axis_refinement_m
        - observed_cross_axis_refinement_m
    )
    cross_axis_clamp_delta_xy = (
        cross_axis_clamp_delta_m * grasp_cross_axis_xy
    )
    insert_xy = observed[:2] + rotated_offset_xy + cross_axis_clamp_delta_xy
    insert_orientation = _rotate_quaternion_about_world_z(
        tuple(insert_reference[3:7]), yaw_delta
    )
    stable_latch_pose = (
        insert_xy[0],
        insert_xy[1] + CODE_POLICY_INSERT_DEPTH_BIAS_WORLD_Y_M,
        # Wrist-visible pad height varies with deformation/occlusion by a few
        # millimetres even at the same scene pose.  Preserve the robust
        # successful-latch height; planar retargeting carries the useful
        # cross-layout signal without turning this noise into table contact.
        insert_reference[2] + CODE_POLICY_INSERT_Z_BIAS_M,
        *insert_orientation,
    )
    safe_latch_pose = (
        stable_latch_pose[0],
        stable_latch_pose[1],
        stable_latch_pose[2] + SAFE_LATCH_Z_OFFSET_M,
        *stable_latch_pose[3:],
    )
    # Move from the robot side of the pad toward its detected centre.  This is
    # equivalent to robot-forward in the reference scene, while rotating with
    # the observed pad yaw in perturbed layouts.
    open_insert_pose = (
        stable_latch_pose[0] + GT_CLOSE_RETRACTION_M * close_approach_xy[0],
        stable_latch_pose[1] + GT_CLOSE_RETRACTION_M * close_approach_xy[1],
        stable_latch_pose[2],
        *stable_latch_pose[3:],
    )
    longitudinal_pad_delta_m = float(delta[:2] @ close_approach_xy)
    pregrasp_forward_refinement_m = float(
        np.clip(
            PREGRASP_FORWARD_REFINEMENT_M + longitudinal_pad_delta_m,
            PREGRASP_FORWARD_REFINEMENT_MIN_M,
            PREGRASP_FORWARD_REFINEMENT_MAX_M,
        )
    )
    safe_preinsert_origin = (
        measured_pregrasp if measured_pregrasp is not None else open_insert_pose
    )
    # The observation pose is nominal, while the pad can be randomized in
    # the plane.  Resolve the pad's cross-axis displacement while the fingers
    # are still open and 10 mm above the latch.  Leaving this correction for
    # ``code_grasp_retract`` makes the closing fingers drag the flexible pad
    # sideways (seed 1104 required about 9.5 mm), producing a skewed grasp
    # that later slips.  Keep the longitudinal component governed by the
    # bounded refinement above so this does not reintroduce deep insertion.
    pregrasp_cross_axis_refinement_xy = (
        pregrasp_cross_axis_refinement_m * grasp_cross_axis_xy
    )
    safe_preinsert_pose = (
        safe_preinsert_origin[0]
        + pregrasp_cross_axis_refinement_xy[0]
        + pregrasp_forward_refinement_m * close_approach_xy[0],
        safe_preinsert_origin[1]
        + pregrasp_cross_axis_refinement_xy[1]
        + pregrasp_forward_refinement_m * close_approach_xy[1],
        stable_latch_pose[2] + SAFE_PREINSERT_Z_OFFSET_M,
        *stable_latch_pose[3:],
    )
    transport = build_transport_plan(reference, tuple(safe_latch_pose))
    if "target" not in perception:
        raise ValueError("code_policy_target_perception_missing")
    transport, target_retarget_audit = retarget_transport_plan_to_target(
        transport, perception["target"]
    )
    transport, initial_place_audit = retarget_place_from_observed_grasp(
        transport,
        tuple(safe_latch_pose),
        perception["pad"],
        perception["target"],
    )
    initial_place_audit["source"] = (
        "pre_grasp_rgbd_pad_to_planned_latch_transform"
    )
    transport = collapse_code_policy_transport(transport)
    close_stage = {
        "name": "code_close",
        "pose": safe_latch_pose,
        "right_open": 0.0,
        "minimum_duration_s": 0.50,
        "maximum_right_gripper_open_fraction": 0.005,
        "continuous_transit": True,
    }
    plan = (
        [
            {
                "name": "code_safe_preinsert",
                "pose": safe_preinsert_pose,
                "right_open": 1.0,
                "minimum_duration_s": 0.50,
                "maximum_duration_s": 3.0,
                "position_tolerance_m": (
                    SAFE_PREINSERT_POSITION_TOLERANCE_M
                ),
                "continuous_transit": True,
            },
            {
                "name": "code_grasp_retract",
                "pose": safe_latch_pose,
                "right_open": 0.0,
                "minimum_duration_s": GRASP_REFINEMENT_MINIMUM_DURATION_S,
                "position_tolerance_m": SAFE_LATCH_POSITION_TOLERANCE_M,
                "maximum_right_gripper_open_fraction": 0.005,
                "continuous_transit": True,
            },
            *transport[1:],
        ]
        if measured_pregrasp is not None
        else [
            {
                "name": "code_insert",
                "pose": open_insert_pose,
                "right_open": 1.0,
            },
            close_stage,
            *transport[1:],
        ]
    )
    return plan, {
        "source": "wrist_rgbd_optical_fk_blue_pad",
        "reference_grasp_selection": (
            "markley_mean_stable_latch_planar_offset_and_orientation_"
            "from_180_successful_development_episodes"
        ),
        "reference_visible_pad_center_world_m": list(
            REFERENCE_VISIBLE_PAD_CENTER_WORLD_M
        ),
        "observed_visible_pad_center_world_m": observed.tolist(),
        "translation_delta_world_m": delta.tolist(),
        "translation_xy_norm_m": xy_norm,
        "observed_pad_yaw_rad_mod_pi": float(
            perception["pad"]["long_axis_yaw_rad_mod_pi"]
        ),
        "yaw_delta_deg": math.degrees(yaw_delta),
        "reference_grasp_offset_xy_m": reference_offset_xy.tolist(),
        "rotated_grasp_offset_xy_m": rotated_offset_xy.tolist(),
        "insert_depth_bias_world_y_m": (
            CODE_POLICY_INSERT_DEPTH_BIAS_WORLD_Y_M
        ),
        "insert_z_bias_m": CODE_POLICY_INSERT_Z_BIAS_M,
        "open_insert_pose_world_xyzw": list(open_insert_pose),
        "stable_latch_pose_world_xyzw": list(stable_latch_pose),
        "safe_latch_pose_world_xyzw": list(safe_latch_pose),
        "safe_latch_z_offset_m": SAFE_LATCH_Z_OFFSET_M,
        "safe_latch_position_tolerance_m": (
            SAFE_LATCH_POSITION_TOLERANCE_M
        ),
        "measured_pregrasp_pose_world_xyzw": (
            list(measured_pregrasp)
            if measured_pregrasp is not None
            else None
        ),
        "acquisition_mode": (
            "rgbd_bounded_pregrasp_refinement_then_close_retract"
            if measured_pregrasp is not None
            else "legacy_open_insert_then_close"
        ),
        "planned_pregrasp_to_latch_distance_m": (
            float(
                np.linalg.norm(
                    np.asarray(safe_latch_pose[:3])
                    - np.asarray(measured_pregrasp[:3])
                )
            )
            if measured_pregrasp is not None
            else None
        ),
        "safe_preinsert_pose_world_xyzw": list(safe_preinsert_pose),
        "pregrasp_forward_refinement_m": pregrasp_forward_refinement_m,
        "pregrasp_cross_axis_refinement_m": (
            pregrasp_cross_axis_refinement_m
        ),
        "observed_cross_axis_refinement_m": (
            observed_cross_axis_refinement_m
        ),
        "maximum_cross_axis_refinement_m": (
            PREGRASP_CROSS_AXIS_REFINEMENT_MAX_M
        ),
        "cross_axis_clamp_delta_m": (
            cross_axis_clamp_delta_m
        ),
        "pregrasp_cross_axis_refinement_world_xy_m": (
            pregrasp_cross_axis_refinement_xy.tolist()
        ),
        "pregrasp_forward_refinement_bounds_m": [
            PREGRASP_FORWARD_REFINEMENT_MIN_M,
            PREGRASP_FORWARD_REFINEMENT_MAX_M,
        ],
        "longitudinal_pad_delta_m": longitudinal_pad_delta_m,
        "close_retract_from_preinsert_m": float(
            np.linalg.norm(
                np.asarray(safe_latch_pose[:2])
                - np.asarray(safe_preinsert_pose[:2])
            )
        ),
        "safe_preinsert_z_offset_m": SAFE_PREINSERT_Z_OFFSET_M,
        "safe_preinsert_position_tolerance_m": (
            SAFE_PREINSERT_POSITION_TOLERANCE_M
        ),
        "close_retraction_m": GT_CLOSE_RETRACTION_M,
        "close_approach_unit_xy": close_approach_xy.tolist(),
        "initial_target_retarget": target_retarget_audit,
        "initial_grasp_relative_place": initial_place_audit,
        "post_grasp_motion": (
            "single_c1_landmark_path_with_nonzero_internal_velocity"
        ),
        "bounds": {
            "maximum_xy_m": MAXIMUM_PAD_RETARGET_XY_M,
            "maximum_z_m": MAXIMUM_PAD_RETARGET_Z_M,
            "maximum_yaw_deg": MAXIMUM_PAD_RETARGET_YAW_DEG,
        },
    }


def grasp_probe_metrics(
    pre_lift: dict[str, float | int],
    post_lift: dict[str, float | int],
) -> dict[str, float | bool]:
    """Fail-closed RGB-D evidence that the pad followed a 50 mm probe lift."""

    depth_drift = float(post_lift["median_depth_m"]) - float(
        pre_lift["median_depth_m"]
    )
    area_ratio = float(post_lift["pixel_count"]) / float(
        pre_lift["pixel_count"]
    )
    timestamp_advanced = float(post_lift["depth_stamp_s"]) > float(
        pre_lift["depth_stamp_s"]
    )
    passed = bool(
        timestamp_advanced
        and depth_drift <= GRASP_PROBE_MAX_MEDIAN_DEPTH_DRIFT_M
        and area_ratio >= GRASP_PROBE_MIN_PIXEL_AREA_RATIO
    )
    return {
        "passed": passed,
        "median_depth_drift_m": depth_drift,
        "pixel_area_ratio": area_ratio,
        "timestamp_advanced": timestamp_advanced,
        "maximum_median_depth_drift_m": (
            GRASP_PROBE_MAX_MEDIAN_DEPTH_DRIFT_M
        ),
        "minimum_pixel_area_ratio": GRASP_PROBE_MIN_PIXEL_AREA_RATIO,
    }


def _rgb_array(message: Image) -> np.ndarray:
    if message.encoding.lower() not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported RGB encoding: {message.encoding}")
    data = np.frombuffer(message.data, dtype=np.uint8)
    array = data.reshape(message.height, message.step)[:, : message.width * 3]
    array = array.reshape(message.height, message.width, 3)
    if message.encoding.lower() == "bgr8":
        array = array[:, :, ::-1]
    return np.ascontiguousarray(array)


def _depth_array(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    if encoding == "32fc1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        scale = 1.0
    elif encoding == "16uc1":
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        scale = 0.001
    else:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")
    data = np.frombuffer(message.data, dtype=dtype)
    row_values = message.step // dtype.itemsize
    return np.ascontiguousarray(
        data.reshape(message.height, row_values)[:, : message.width]
        * scale,
        dtype=np.float32,
    )


class HybridObservationStager(ObservationStager):
    """Observation stager plus policy-legal right-wrist RGB-D cache."""

    def __init__(self) -> None:
        super().__init__()
        topics = load_topics()
        self.right_joint_command_topic = topics["bridge"]["joint_groups"][
            "right_arm"
        ]["command"]
        self.right_joint_command_publisher = self.create_publisher(
            JointState, self.right_joint_command_topic, 10
        )
        entry = topics["cameras"]["robot"]["wrist_right"]
        namespace = entry["namespace"]
        self.wrist_rgb: np.ndarray | None = None
        self.wrist_depth_m: np.ndarray | None = None
        self.wrist_rgb_time: float | None = None
        self.wrist_depth_time: float | None = None
        self.head_rgb: np.ndarray | None = None
        self.head_depth_m: np.ndarray | None = None
        self.head_rgb_time: float | None = None
        self.head_depth_time: float | None = None
        self.intrinsics: dict[str, tuple[float, tuple[float, ...]]] = {}
        self.camera_pose: dict[str, tuple[float, tuple[float, ...]]] = {}
        self.world_perception_error: str | None = None
        head_entry = topics["cameras"]["robot"]["head"]
        head_namespace = head_entry["namespace"]
        self._owned_subscriptions.extend(
            (
                self.create_subscription(
                    Image,
                    camera_topic(topics, namespace, "image"),
                    self._on_wrist_rgb,
                    qos_profile_sensor_data,
                ),
                self.create_subscription(
                    Image,
                    camera_topic(topics, namespace, "depth"),
                    self._on_wrist_depth,
                    qos_profile_sensor_data,
                ),
                self.create_subscription(
                    Image,
                    camera_topic(topics, head_namespace, "image"),
                    self._on_head_rgb,
                    qos_profile_sensor_data,
                ),
                self.create_subscription(
                    Image,
                    camera_topic(topics, head_namespace, "depth"),
                    self._on_head_depth,
                    qos_profile_sensor_data,
                ),
            )
        )
        for key, camera_entry in (
            ("head", head_entry),
            ("wrist_right", entry),
        ):
            camera_namespace = camera_entry["namespace"]
            self._owned_subscriptions.extend(
                (
                    self.create_subscription(
                        CameraInfo,
                        camera_topic(topics, camera_namespace, "camera_info"),
                        lambda message, key=key: self._on_camera_info(
                            key, message
                        ),
                        qos_profile_sensor_data,
                    ),
                    self.create_subscription(
                        PoseStamped,
                        topics["recording"]["camera_pose"][key],
                        lambda message, key=key: self._on_camera_pose(
                            key, message
                        ),
                        10,
                    ),
                )
            )

    def subscribers_ready(self) -> bool:
        # Pose/gripper/spine are required from the first code-insert tick.
        # Direct joints are not needed until the later extraction stage;
        # gating the whole state machine on their DDS discovery can deadlock
        # at pre-grasp even though RMPflow is fully available.
        return super().subscribers_ready()

    def conflicts(self) -> dict[str, int]:
        # The parent runner retains a dormant right-joint publisher while it
        # synchronously waits for this child controller.  Pose/gripper/spine
        # contention remains illegal; that intentionally dormant publisher is
        # not a second active owner.
        return super().conflicts()

    def publish_right_joint_override(
        self,
        targets: dict[str, tuple[float, ...]],
        right_joint_positions: tuple[float, ...],
        spine_command_m: float,
        gripper_open_fractions: dict[str, float],
    ) -> None:
        """Publish normal holds plus one fresh direct right-arm command."""

        if len(right_joint_positions) != len(RIGHT_JOINTS):
            raise ValueError("right joint override must contain seven joints")
        self.publish(targets, spine_command_m, gripper_open_fractions)
        assert self.sim_time is not None
        message = JointState()
        message.header.stamp = Time(
            nanoseconds=max(0, int(self.sim_time * 1.0e9))
        ).to_msg()
        message.name = list(RIGHT_JOINTS)
        message.position = [float(value) for value in right_joint_positions]
        self.right_joint_command_publisher.publish(message)
        self.publish_count += 1

    def _on_wrist_rgb(self, message: Image) -> None:
        try:
            self.wrist_rgb = _rgb_array(message)
            self.wrist_rgb_time = float(message.header.stamp.sec) + float(
                message.header.stamp.nanosec
            ) * 1.0e-9
        except ValueError as error:
            self.get_logger().error(str(error))

    def _on_wrist_depth(self, message: Image) -> None:
        try:
            self.wrist_depth_m = _depth_array(message)
            self.wrist_depth_time = float(message.header.stamp.sec) + float(
                message.header.stamp.nanosec
            ) * 1.0e-9
        except ValueError as error:
            self.get_logger().error(str(error))

    def _on_head_rgb(self, message: Image) -> None:
        try:
            self.head_rgb = _rgb_array(message)
            self.head_rgb_time = float(message.header.stamp.sec) + float(
                message.header.stamp.nanosec
            ) * 1.0e-9
        except ValueError as error:
            self.get_logger().error(str(error))

    def _on_head_depth(self, message: Image) -> None:
        try:
            self.head_depth_m = _depth_array(message)
            self.head_depth_time = float(message.header.stamp.sec) + float(
                message.header.stamp.nanosec
            ) * 1.0e-9
        except ValueError as error:
            self.get_logger().error(str(error))

    def _on_camera_info(self, key: str, message: CameraInfo) -> None:
        stamp = float(message.header.stamp.sec) + float(
            message.header.stamp.nanosec
        ) * 1.0e-9
        self.intrinsics[key] = (
            stamp,
            (
                float(message.k[0]),
                float(message.k[4]),
                float(message.k[2]),
                float(message.k[5]),
            ),
        )

    def _on_camera_pose(self, key: str, message: PoseStamped) -> None:
        stamp = float(message.header.stamp.sec) + float(
            message.header.stamp.nanosec
        ) * 1.0e-9
        position = message.pose.position
        orientation = message.pose.orientation
        self.camera_pose[key] = (
            stamp,
            (
                position.x,
                position.y,
                position.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
        )

    def wrist_signature(
        self, maximum_skew_s: float
    ) -> dict[str, float | int] | None:
        if (
            self.sim_time is None
            or self.wrist_rgb is None
            or self.wrist_depth_m is None
            or self.wrist_rgb_time is None
            or self.wrist_depth_time is None
            or abs(self.wrist_rgb_time - self.wrist_depth_time)
            > maximum_skew_s
            or self.sim_time - min(self.wrist_rgb_time, self.wrist_depth_time)
            > maximum_skew_s
        ):
            return None
        try:
            signature = blue_pad_depth_signature(
                self.wrist_rgb, self.wrist_depth_m
            )
        except ValueError:
            return None
        signature["rgb_stamp_s"] = self.wrist_rgb_time
        signature["depth_stamp_s"] = self.wrist_depth_time
        return signature

    def world_perception(
        self, maximum_skew_s: float
    ) -> dict[str, Any] | None:
        """Return camera-only pad/target world geometry for audit/retarget."""

        self.world_perception_error = None
        if (
            self.sim_time is None
            or self.head_rgb is None
            or self.head_depth_m is None
            or self.wrist_rgb is None
            or self.wrist_depth_m is None
            or self.head_rgb_time is None
            or self.head_depth_time is None
            or self.wrist_rgb_time is None
            or self.wrist_depth_time is None
            or "head" not in self.intrinsics
            or "wrist_right" not in self.intrinsics
            or "head" not in self.camera_pose
            or "wrist_right" not in self.camera_pose
        ):
            self.world_perception_error = "incomplete_camera_geometry"
            return None
        stamps = [
            self.head_rgb_time,
            self.head_depth_time,
            self.wrist_rgb_time,
            self.wrist_depth_time,
            self.camera_pose["head"][0],
            self.camera_pose["wrist_right"][0],
        ]
        if (
            max(stamps) - min(stamps) > maximum_skew_s
            or self.sim_time - min(stamps) > maximum_skew_s
        ):
            self.world_perception_error = "stale_or_skewed_camera_geometry"
            return None
        try:
            target_mask = red_target_mask(self.head_rgb)
            pad_mask = blue_pad_mask(self.wrist_rgb)
            if int(target_mask.sum()) < RED_TARGET_MIN_PIXELS:
                raise ValueError(
                    f"insufficient red target pixels: {int(target_mask.sum())}"
                )
            if int(pad_mask.sum()) < WORLD_PAD_MIN_PIXELS:
                raise ValueError(
                    f"insufficient blue pad pixels: {int(pad_mask.sum())}"
                )
            target_points = deproject_masked_depth(
                self.head_depth_m,
                target_mask,
                self.intrinsics["head"][1],
                self.camera_pose["head"][1],
                minimum_depth_m=0.15,
                maximum_depth_m=5.0,
            )
            pad_points = deproject_masked_depth(
                self.wrist_depth_m,
                pad_mask,
                self.intrinsics["wrist_right"][1],
                self.camera_pose["wrist_right"][1],
                minimum_depth_m=0.03,
                maximum_depth_m=1.5,
            )
            target = robust_world_surface_signature(target_points)
            pad = robust_world_surface_signature(pad_points)
        except ValueError as error:
            self.world_perception_error = str(error)
            return None
        target["pixel_count"] = int(target_mask.sum())
        pad["pixel_count"] = int(pad_mask.sum())
        return {
            "timing_skew_s": max(stamps) - min(stamps),
            "target": target,
            "pad": pad,
        }


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
    """Return a GT-shaped decontact/extract curve and transport route."""

    landmarks = reference["right_hybrid_landmarks_world_xyzw"]
    transfer_reference = tuple(
        float(value) for value in landmarks["transfer"]
    )
    place = tuple(float(value) for value in landmarks["place"])
    short_vertical_decontact = (
        initial_right[0],
        initial_right[1],
        initial_right[2] + SHORT_DECONTACT_LIFT_M,
        *initial_right[3:],
    )
    base_yaw = float(reference["base_xyyaw"][2])
    forward_x = math.cos(base_yaw)
    forward_y = math.sin(base_yaw)
    short_diagonal_clearance = (
        initial_right[0] + GT_SHORT_DIAGONAL_FORWARD_M * forward_x,
        initial_right[1] + GT_SHORT_DIAGONAL_FORWARD_M * forward_y,
        initial_right[2] + GT_SHORT_DIAGONAL_Z_M,
        *initial_right[3:],
    )
    forward_rising_extract = (
        initial_right[0] + GT_FORWARD_RISE_FORWARD_M * forward_x,
        initial_right[1] + GT_FORWARD_RISE_FORWARD_M * forward_y,
        initial_right[2] + GT_FORWARD_RISE_Z_M,
        *initial_right[3:],
    )
    retained_lift = (
        initial_right[0] + GT_RETAINED_FORWARD_M * forward_x,
        initial_right[1] + GT_RETAINED_FORWARD_M * forward_y,
        initial_right[2] + GT_RETAINED_Z_M,
        *initial_right[3:],
    )
    forward_clear_base = (
        initial_right[0] + GT_PRE_LATERAL_FORWARD_M * forward_x,
        initial_right[1] + GT_PRE_LATERAL_FORWARD_M * forward_y,
        initial_right[2] + GT_PRE_LATERAL_Z_M,
        *initial_right[3:],
    )
    # Keep the extracted pad high while moving laterally.  Target-centre
    # retargeting below owns the final fore/aft correction; extraction is not
    # accumulated into the absolute placement pose.
    transfer = (
        transfer_reference[0],
        transfer_reference[1],
        max(transfer_reference[2], forward_clear_base[2]),
        *transfer_reference[3:],
    )
    target_overhead = (
        place[0], place[1], place[2] + 0.08, *transfer[3:]
    )
    support_contact = (*place[:3], *transfer[3:])
    # Reproduce the successful Formula-3 placement geometry without importing
    # an episode-specific waypoint: the pre-contact orientation is the
    # quaternion geodesic midpoint between the 180-development transfer and
    # place medians.  This avoids releasing in the visibly incorrect transfer
    # wrist pose while splitting the large rotation into two bounded moves.
    support_precontact = (
        *place[:3], *_slerp(transfer[3:], place[3:], 0.5)
    )
    support_place = place
    release_at_contact = tuple(
        float(value) for value in landmarks["release"]
    )
    left_x = -forward_y
    left_y = forward_x

    def release_offset_pose(offset: tuple[float, float, float]) -> tuple[float, ...]:
        forward_m, z_m, left_m = offset
        return (
            release_at_contact[0] + forward_m * forward_x + left_m * left_x,
            release_at_contact[1] + forward_m * forward_y + left_m * left_y,
            release_at_contact[2] + z_m,
            *release_at_contact[3:],
        )

    # Open while following the demonstrated small backward/upward motion.
    # Opening in place doubled the observed release interval and let the
    # compliant pad slide during the latter half of placement.
    release_open = release_offset_pose(GT_RELEASE_OPEN_FORWARD_Z_LEFT_M)
    release_clear_path = [
        release_offset_pose(offset)
        for offset in GT_RELEASE_CLEAR_FORWARD_Z_LEFT_M
    ]
    retreat = release_clear_path[-1]
    reset_clear_view = tuple(
        float(value)
        for value in reference["right_clearance_waypoint_ee_world_xyzw"]
    )
    return [
        {"name": "retain", "pose": initial_right, "right_open": 0.0},
        {
            "name": "short_vertical_decontact",
            "pose": short_vertical_decontact,
            "right_open": 0.0,
            "minimum_duration_s": 0.30,
            "continuous_transit": True,
        },
        {
            "name": "short_diagonal_clearance",
            "pose": short_diagonal_clearance,
            "right_open": 0.0,
            "minimum_duration_s": 0.30,
            "continuous_transit": True,
        },
        {
            "name": "forward_rising_extract",
            "pose": forward_rising_extract,
            "right_open": 0.0,
            "minimum_duration_s": 0.30,
            "continuous_transit": True,
        },
        {
            "name": "retained_lift",
            "pose": retained_lift,
            "right_open": 0.0,
            "minimum_duration_s": 0.30,
            "continuous_transit": True,
        },
        {
            "name": "forward_clear_base",
            "pose": forward_clear_base,
            "right_open": 0.0,
            "minimum_duration_s": 0.30,
            "continuous_transit": True,
        },
        {
            "name": "transfer",
            "pose": transfer,
            "right_open": 0.0,
            "right_bezier_start_tangent_world_m": (
                TRANSFER_BLEND_FORWARD_TANGENT_M * forward_x,
                TRANSFER_BLEND_FORWARD_TANGENT_M * forward_y,
                0.0,
            ),
            "right_bezier_end_tangent_world_m": (
                TRANSFER_BLEND_LEFT_TANGENT_M * -forward_y,
                TRANSFER_BLEND_LEFT_TANGENT_M * forward_x,
                0.0,
            ),
        },
        {
            "name": "target_overhead",
            "pose": target_overhead,
            "right_open": 0.0,
        },
        {
            "name": "support_contact",
            "pose": support_contact,
            "right_open": 0.0,
            "minimum_right_ee_z_m": MINIMUM_SUPPORT_EE_Z_M,
        },
        {
            "name": "support_precontact",
            "pose": support_precontact,
            "right_open": 0.0,
            "minimum_right_ee_z_m": MINIMUM_SUPPORT_EE_Z_M,
        },
        {
            "name": "support_place",
            "pose": support_place,
            "right_open": 0.0,
            "minimum_right_ee_z_m": MINIMUM_SUPPORT_EE_Z_M,
        },
        {
            "name": "release",
            "pose": release_open,
            "right_open": 1.0,
            "minimum_right_ee_z_m": MINIMUM_SUPPORT_EE_Z_M,
            "minimum_duration_s": GT_RELEASE_OPEN_DURATION_S,
            "minimum_right_gripper_open_fraction": (
                RELEASE_CLEAR_MINIMUM_OPEN_FRACTION
            ),
            "continuous_transit": True,
        },
        {
            "name": "retreat",
            "pose": retreat,
            # The fingers are already fully open; continue directly along
            # the demonstrated upward clear path, then reset backward/up.
            "right_pose_path": release_clear_path,
            "right_open": 1.0,
            "minimum_duration_s": 0.70,
            "maximum_duration_s": 3.0,
            "max_linear_speed_m_s": GT_RELEASE_CLEAR_PEAK_LINEAR_SPEED_M_S,
            "short_ramp_fraction": GT_CONTINUOUS_RAMP_FRACTION,
            "minimum_right_gripper_open_fraction": (
                RELEASE_CLEAR_MINIMUM_OPEN_FRACTION
            ),
            "continuous_transit": True,
        },
        {
            "name": "reset_clear_view",
            "pose": reset_clear_view,
            "right_open": 1.0,
            "minimum_duration_s": 1.0,
            "maximum_duration_s": 8.0,
            "max_linear_speed_m_s": GT_CONTINUOUS_PEAK_LINEAR_SPEED_M_S,
        },
    ]


def collapse_code_policy_transport_after_joint_extract(
    plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse post-clear transport through placement into one GT curve."""

    motion_names = (
        "short_vertical_decontact",
        "short_diagonal_clearance",
        "forward_rising_extract",
        "retained_lift",
        "forward_clear_base",
        "transfer",
        "target_overhead",
    )
    by_name = {str(stage["name"]): stage for stage in plan}
    missing = [name for name in motion_names if name not in by_name]
    if missing:
        raise ValueError(f"continuous transport landmarks missing: {missing}")
    initial = tuple(by_name["retain"]["pose"])
    forward_clear = np.asarray(
        by_name["forward_clear_base"]["pose"][:2], dtype=np.float64
    )
    forward_delta = forward_clear - np.asarray(initial[:2], dtype=np.float64)
    forward_unit = forward_delta / np.linalg.norm(forward_delta)
    place_pose = tuple(by_name["support_place"]["pose"])
    transfer_pose = tuple(by_name["transfer"]["pose"])
    left_unit = np.asarray((-forward_unit[1], forward_unit[0]))
    approach_path = []
    for forward_m, z_m, left_m, rotation_fraction in (
        GT_PLACE_APPROACH_FORWARD_Z_LEFT_ROTATION
    ):
        xy = (
            np.asarray(place_pose[:2])
            + forward_m * forward_unit
            + left_m * left_unit
        )
        approach_path.append(
            (
                float(xy[0]),
                float(xy[1]),
                place_pose[2] + z_m,
                *_slerp(
                    transfer_pose[3:], place_pose[3:], rotation_fraction
                ),
            )
        )
    path = [
        transfer_pose,
        *approach_path,
        place_pose,
    ]
    path_names = [
        "transfer",
        "gt_place_fraction_025",
        "gt_place_fraction_050",
        "gt_place_2s_before",
        "gt_place_1s_before",
        "gt_place_05s_before",
        "gt_place_025s_before",
        "support_place",
    ]
    collapsed = {
        "name": "smooth_transport_to_place",
        "pose": path[-1],
        "right_pose_path": path,
        "right_open": 0.0,
        "minimum_duration_s": GT_POST_CLEAR_TO_PLACE_DURATION_S,
        "maximum_duration_s": 13.0,
        "max_linear_speed_m_s": GT_CONTINUOUS_PEAK_LINEAR_SPEED_M_S,
        "short_ramp_fraction": GT_CONTINUOUS_RAMP_FRACTION,
        "path_landmark_names": path_names,
        "path_landmark_times_s": list(GT_POST_CLEAR_LANDMARK_TIMES_S),
        # Placement is the one endpoint where GT settles before opening.  The
        # global 15 mm transport tolerance let seed 2217 start release 8.9 mm
        # short.  A 5 mm gate was unreachable under target contact, while the
        # successful Formula-3 run settled at 7.4 mm, so use a contact-aware
        # 10 mm bound before opening.
        "position_tolerance_m": (
            LARGE_TARGET_CONTACT_TOLERANCE_M
            if float(
                by_name["support_place"].get("target_retarget_xy_m", 0.0)
            )
            > LARGE_TARGET_RETARGET_XY_M
            else 0.010
        ),
    }
    first = next(index for index, stage in enumerate(plan)
                 if stage["name"] == motion_names[0])
    last = next(index for index, stage in enumerate(plan)
                if stage["name"] == motion_names[-1])
    tail = plan[last + 1:]
    tail_by_name = {str(stage["name"]): stage for stage in tail}
    place_names = ("support_contact", "support_precontact", "support_place")
    missing_place = [name for name in place_names if name not in tail_by_name]
    if missing_place:
        raise ValueError(f"continuous place landmarks missing: {missing_place}")
    place_first = next(
        index for index, stage in enumerate(tail)
        if stage["name"] == place_names[0]
    )
    place_last = next(
        index for index, stage in enumerate(tail)
        if stage["name"] == place_names[-1]
    )
    return [
        *plan[:first],
        collapsed,
        *tail[:place_first],
        *tail[place_last + 1:],
    ]


def collapse_code_policy_transport(
    plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse extraction through placement into the proven C1 GT curve."""

    motion_names = (
        "short_vertical_decontact",
        "short_diagonal_clearance",
        "forward_rising_extract",
        "retained_lift",
        "forward_clear_base",
        "transfer",
        "target_overhead",
    )
    by_name = {str(stage["name"]): stage for stage in plan}
    missing = [name for name in motion_names if name not in by_name]
    if missing:
        raise ValueError(f"continuous transport landmarks missing: {missing}")
    initial = tuple(by_name["retain"]["pose"])
    forward_clear = np.asarray(
        by_name["forward_clear_base"]["pose"][:2], dtype=np.float64
    )
    forward_delta = forward_clear - np.asarray(initial[:2], dtype=np.float64)
    forward_unit = forward_delta / np.linalg.norm(forward_delta)
    early_path = [
        (
            initial[0] + forward_m * float(forward_unit[0]),
            initial[1] + forward_m * float(forward_unit[1]),
            initial[2] + z_m,
            *initial[3:],
        )
        for forward_m, z_m in GT_EARLY_EXTRACT_FORWARD_Z_M
    ]
    remaining_names = (
        "short_diagonal_clearance",
        "forward_rising_extract",
        "retained_lift",
        "forward_clear_base",
        "transfer",
    )
    place_pose = tuple(by_name["support_place"]["pose"])
    transfer_pose = tuple(by_name["transfer"]["pose"])
    left_unit = np.asarray((-forward_unit[1], forward_unit[0]))
    approach_path = []
    for forward_m, z_m, left_m, rotation_fraction in (
        GT_PLACE_APPROACH_FORWARD_Z_LEFT_ROTATION
    ):
        xy = (
            np.asarray(place_pose[:2])
            + forward_m * forward_unit
            + left_m * left_unit
        )
        approach_path.append(
            (
                float(xy[0]),
                float(xy[1]),
                place_pose[2] + z_m,
                *_slerp(
                    transfer_pose[3:], place_pose[3:], rotation_fraction
                ),
            )
        )
    path = [
        *early_path,
        *(
            tuple(by_name[name]["pose"])
            for name in remaining_names
        ),
        *approach_path,
        place_pose,
    ]
    path_names = [
        "safe_vertical_z_10mm",
        "gt_first_z_15mm",
        "gt_first_z_20mm",
        *remaining_names,
        "gt_place_fraction_025",
        "gt_place_fraction_050",
        "gt_place_2s_before",
        "gt_place_1s_before",
        "gt_place_05s_before",
        "gt_place_025s_before",
        "support_place",
    ]
    collapsed = {
        "name": "smooth_extract_to_place",
        "pose": path[-1],
        "right_pose_path": path,
        "right_open": 0.0,
        "minimum_duration_s": GT_STABLE_LATCH_TO_PLACE_DURATION_S,
        "maximum_duration_s": 14.0,
        "max_linear_speed_m_s": GT_CONTINUOUS_PEAK_LINEAR_SPEED_M_S,
        "short_ramp_fraction": GT_CONTINUOUS_RAMP_FRACTION,
        "path_landmark_names": path_names,
        "path_landmark_times_s": list(GT_POST_LATCH_LANDMARK_TIMES_S),
        "position_tolerance_m": (
            LARGE_TARGET_CONTACT_TOLERANCE_M
            if float(
                by_name["support_place"].get("target_retarget_xy_m", 0.0)
            )
            > LARGE_TARGET_RETARGET_XY_M
            else 0.010
        ),
    }
    first = next(
        index for index, stage in enumerate(plan)
        if stage["name"] == motion_names[0]
    )
    last = next(
        index for index, stage in enumerate(plan)
        if stage["name"] == motion_names[-1]
    )
    tail = plan[last + 1:]
    tail_by_name = {str(stage["name"]): stage for stage in tail}
    place_names = ("support_contact", "support_precontact", "support_place")
    missing_place = [name for name in place_names if name not in tail_by_name]
    if missing_place:
        raise ValueError(f"continuous place landmarks missing: {missing_place}")
    place_first = next(
        index for index, stage in enumerate(tail)
        if stage["name"] == place_names[0]
    )
    place_last = next(
        index for index, stage in enumerate(tail)
        if stage["name"] == place_names[-1]
    )
    return [
        *plan[:first],
        collapsed,
        *tail[:place_first],
        *tail[place_last + 1:],
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
    parser.add_argument(
        "--perception-only",
        action="store_true",
        help="Capture one camera-only world-geometry sample; publish nothing.",
    )
    parser.add_argument(
        "--code-policy-acquire",
        action="store_true",
        help="Use bounded visual RMPflow insert/close instead of a VLA latch.",
    )
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


def _run_perception_only(
    node: HybridObservationStager, args: argparse.Namespace
) -> int:
    started = time.monotonic()
    perception: dict[str, Any] | None = None
    while (
        not node.stop_requested
        and time.monotonic() - started < args.max_duration_s
    ):
        rclpy.spin_once(node, timeout_sec=0.02)
        perception = node.world_perception(args.maximum_skew_s)
        if perception is not None:
            break
    result = {
        "schema_version": 1,
        "success": perception is not None,
        "mode": "camera_only_rgbd_optical_fk_zero_publication",
        "reason": (
            "fresh_world_geometry"
            if perception is not None
            else node.world_perception_error or "host_watchdog_timeout"
        ),
        "perception": perception,
        "ground_truth_subscriptions": [],
        "command_publications": node.publish_count,
        "elapsed_wall_s": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if result["success"] else 2


def main() -> int:  # noqa: C901 - one bounded measured transport state machine
    args = build_parser().parse_args()
    reference = load_hybrid_reference(args.reference)
    base_target = tuple(float(value) for value in reference["base_xyyaw"])
    left_target = tuple(
        float(value) for value in reference["left_safe_ee_world_xyzw"]
    )
    spine_target = float(reference["spine_command_m"])

    rclpy.init()
    node = HybridObservationStager()
    signal.signal(signal.SIGINT, lambda *_: setattr(node, "stop_requested", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(node, "stop_requested", True))
    if args.perception_only:
        return _run_perception_only(node, args)
    wall_started = time.monotonic()
    last_sim_time: float | None = None
    previous_joint_sample: tuple[float, tuple[float, ...]] | None = None
    plan: list[dict[str, Any]] | None = None
    stage_index = 0
    stage_initial: dict[str, tuple[float, ...]] | None = None
    stage_initial_right_joints: tuple[float, ...] | None = None
    stage_started: float | None = None
    stage_duration = 0.0
    right_path_orientation_scale_m_per_rad = 0.0
    right_path_landmark_fractions: list[float] | None = None
    stable_since: float | None = None
    stage_capture_wait_started: float | None = None
    handoff_started_sim: float | None = None
    handoff_close_stable_since: float | None = None
    handoff_right_pose: tuple[float, ...] | None = None
    stages: list[dict[str, Any]] = []
    reason = "host_watchdog_timeout"
    base_errors: dict[str, float] = {}
    final_grippers: dict[str, float] = {}
    minimum_support_ee_z_m: float | None = None
    grasp_verification_pre_lift: dict[str, float | int] | None = None
    grasp_verification_post_lift: dict[str, float | int] | None = None
    grasp_verification_metrics: dict[str, float | bool] | None = None
    world_perception_pre_lift: dict[str, Any] | None = None
    world_perception_post_lift: dict[str, Any] | None = None
    world_perception_errors: list[str] = []
    visual_retarget_audit: dict[str, Any] | None = None
    grasp_relative_place_audit: dict[str, Any] | None = None
    supported_alignment_audit: dict[str, Any] | None = None
    code_policy_acquire_audit: dict[str, Any] | None = None
    retention_reference_pad: tuple[float, ...] | None = None
    retention_reference_ee: tuple[float, ...] | None = None
    retention_checks: list[dict[str, Any]] = []
    transfer_retention_gate: dict[str, Any] | None = None

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
                if args.code_policy_acquire:
                    perception = node.world_perception(args.maximum_skew_s)
                    if perception is None:
                        continue
                    try:
                        plan, code_policy_acquire_audit = (
                            build_code_policy_acquire_plan(
                                reference,
                                perception,
                                measured_pregrasp_pose=tuple(
                                    node.ee["right"]
                                ),
                            )
                        )
                        visual_retarget_audit = code_policy_acquire_audit[
                            "initial_target_retarget"
                        ]
                        grasp_relative_place_audit = (
                            code_policy_acquire_audit[
                                "initial_grasp_relative_place"
                            ]
                        )
                    except ValueError as error:
                        reason = str(error)
                        break
                    handoff_right_pose = tuple(node.ee["right"])
                    world_perception_pre_lift = perception
                    continue
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
                        grasp_verification_pre_lift = node.wrist_signature(
                            args.maximum_skew_s
                        )
                        if grasp_verification_pre_lift is not None:
                            world_perception_pre_lift = node.world_perception(
                                args.maximum_skew_s
                            )
                            if node.world_perception_error is not None:
                                world_perception_errors.append(
                                    "pre_lift:"
                                    + node.world_perception_error
                                )
                            plan = build_transport_plan(
                                reference, handoff_right_pose
                            )
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
            if stage_initial is None and bool(
                stage.get("hold_current_right_pose", False)
            ):
                stage = dict(stage)
                stage["pose"] = tuple(node.ee["right"])
                plan[stage_index] = stage
            if (
                stage["name"] == "joint_extract_spline"
                and grasp_verification_pre_lift is None
            ):
                if (
                    node.right_joint_command_publisher.get_subscription_count()
                    < 1
                ):
                    reason = "right_joint_command_subscriber_unavailable"
                    break
                grasp_verification_pre_lift = node.wrist_signature(
                    args.maximum_skew_s
                )
                if grasp_verification_pre_lift is None:
                    continue
                world_perception_pre_lift = node.world_perception(
                    args.maximum_skew_s
                )
                if world_perception_pre_lift is None and (
                    node.world_perception_error
                    in {
                        "incomplete_camera_geometry",
                        "stale_or_skewed_camera_geometry",
                    }
                ):
                    # Do not begin the direct-joint ownership window until the
                    # pre-extraction target reference and wrist pad sample are
                    # from a coherent camera pair.  Otherwise the completed
                    # spline cannot compute its incremental target retarget.
                    grasp_verification_pre_lift = None
                    continue
                if node.world_perception_error is not None:
                    world_perception_errors.append(
                        "pre_lift:" + node.world_perception_error
                    )
            target_right = tuple(stage["pose"])
            targets = {"left": left_target, "right": target_right}
            minimum_right_ee_z_m = stage.get("minimum_right_ee_z_m")
            if minimum_right_ee_z_m is not None:
                minimum_support_ee_z_m = (
                    node.ee["right"][2]
                    if minimum_support_ee_z_m is None
                    else min(minimum_support_ee_z_m, node.ee["right"][2])
                )
                if (
                    node.ee["right"][2] + SUPPORT_EE_Z_TOLERANCE_M
                    < float(minimum_right_ee_z_m)
                ):
                    reason = (
                        "unsafe_gripper_table_clearance:"
                        f"{stage['name']}:{node.ee['right'][2]:.6f}"
                    )
                    break
            if stage_initial is None:
                stage_initial = dict(node.ee)
                stage_initial_right_joints = tuple(
                    resolve_joint(node.joints, name) for name in RIGHT_JOINTS
                )
                stage_started = node.sim_time
                right_path_landmark_fractions = None
                maximum_stage_duration_s = float(
                    stage.get("maximum_duration_s", 8.0)
                )
                stage_max_linear_speed_m_s = float(
                    stage.get(
                        "max_linear_speed_m_s",
                        args.max_linear_speed_m_s,
                    )
                )
                stage_duration = transition_duration_s(
                    stage_initial,
                    targets,
                    max_linear_speed_m_s=stage_max_linear_speed_m_s,
                    max_angular_speed_deg_s=float(
                        stage.get(
                            "max_angular_speed_deg_s",
                            args.max_angular_speed_deg_s,
                        )
                    ),
                    minimum_s=float(
                        stage.get(
                            "minimum_duration_s",
                            0.30
                            if stage["name"] in {"retain", "release"}
                            else 1.0,
                        )
                    ),
                    maximum_s=maximum_stage_duration_s,
                )
                if "relative_right_joint_spline" in stage:
                    stage_duration = float(stage["fixed_duration_s"])
                    relative_knots = tuple(
                        tuple(float(value) for value in knot)
                        for knot in stage["relative_right_joint_spline"]
                    )
                    if any(
                        len(knot) != len(RIGHT_JOINTS)
                        for knot in relative_knots
                    ):
                        raise ValueError(
                            "relative right joint spline must be Nx7"
                        )
                    assert stage_initial_right_joints is not None
                    for knot in relative_knots:
                        absolute = tuple(
                            origin + delta
                            for origin, delta in zip(
                                stage_initial_right_joints, knot, strict=True
                            )
                        )
                        for value, (lower, upper) in zip(
                            absolute, FR3_JOINT_LIMITS, strict=True
                        ):
                            if not lower <= value <= upper:
                                raise ValueError(
                                    "joint extract spline exceeds FR3 limits"
                                )
                if "right_pose_path" in stage:
                    right_path = [
                        tuple(stage_initial["right"]),
                        *(
                            tuple(float(value) for value in pose)
                            for pose in stage["right_pose_path"]
                        ),
                    ]
                    ramp_fraction = stage.get("short_ramp_fraction")
                    peak_speed_ratio = (
                        1.0 / (1.0 - float(ramp_fraction))
                        if ramp_fraction is not None
                        else 1.875
                    )
                    angular_speed_rad_s = math.radians(
                        float(
                            stage.get(
                                "max_angular_speed_deg_s",
                                args.max_angular_speed_deg_s,
                            )
                        )
                    )
                    right_path_orientation_scale_m_per_rad = (
                        stage_max_linear_speed_m_s / angular_speed_rad_s
                    )
                    # Time the complete polycurve by its chord length so the
                    # command cannot exceed its GT-derived Cartesian peak
                    # merely because endpoints are closer than the route.
                    path_duration = (
                        peak_speed_ratio
                        * pose_path_length(
                            right_path,
                            right_path_orientation_scale_m_per_rad,
                        )
                        / stage_max_linear_speed_m_s
                    )
                    stage_duration = min(
                        maximum_stage_duration_s,
                        max(stage_duration, path_duration),
                    )
                    if "path_landmark_times_s" in stage:
                        landmark_times = [
                            0.0,
                            *(
                                float(value)
                                for value in stage["path_landmark_times_s"]
                            ),
                        ]
                        path_time_s = landmark_times[-1]
                        if len(landmark_times) != len(right_path):
                            raise ValueError(
                                "right path landmark times must match poses"
                            )
                        # The controller applies a global raised-cosine ramp
                        # before interpolation.  Express temporal knots in
                        # that progress domain so they are reached at their
                        # measured wall/simulation times.
                        right_path_landmark_fractions = [
                            short_ramp_fraction(
                                time_s / path_time_s,
                                float(stage["short_ramp_fraction"]),
                            )
                            for time_s in landmark_times
                        ]
                stable_since = None
            assert stage_started is not None
            raw_fraction = min(
                1.0, max(0.0, (node.sim_time - stage_started) / stage_duration)
            )
            fraction = (
                raw_fraction
                if "relative_right_joint_spline" in stage
                else short_ramp_fraction(
                    raw_fraction, float(stage["short_ramp_fraction"])
                )
                if "short_ramp_fraction" in stage
                else minimum_jerk_fraction(raw_fraction)
            )
            commanded: dict[str, tuple[float, ...]] = {}
            for side in ("left", "right"):
                if side == "right" and "right_pose_path" in stage:
                    commanded[side] = continuous_landmark_pose(
                        [
                            tuple(stage_initial[side]),
                            *(
                                tuple(float(value) for value in pose)
                                for pose in stage["right_pose_path"]
                            ),
                        ],
                        fraction,
                        right_path_orientation_scale_m_per_rad,
                        right_path_landmark_fractions,
                    )
                    continue
                position = (
                    cubic_bezier_position(
                        stage_initial[side],
                        targets[side],
                        tuple(stage["right_bezier_start_tangent_world_m"]),
                        tuple(stage["right_bezier_end_tangent_world_m"]),
                        fraction,
                    )
                    if side == "right"
                    and "right_bezier_start_tangent_world_m" in stage
                    else tuple(
                        stage_initial[side][index]
                        + fraction
                        * (
                            targets[side][index]
                            - stage_initial[side][index]
                        )
                        for index in range(3)
                    )
                )
                commanded[side] = (
                    *position,
                    *_slerp(
                        stage_initial[side][3:7],
                        targets[side][3:7],
                        fraction,
                    ),
                )
            right_joint_tracking_error_rad: float | None = None
            right_joint_target: tuple[float, ...] | None = None
            if "relative_right_joint_spline" in stage:
                assert stage_initial_right_joints is not None
                relative = continuous_joint_spline(
                    stage["relative_right_joint_spline"], fraction
                )
                right_joint_target = tuple(
                    origin + delta
                    for origin, delta in zip(
                        stage_initial_right_joints, relative, strict=True
                    )
                )
                current_right_joints = tuple(
                    resolve_joint(node.joints, name) for name in RIGHT_JOINTS
                )
                right_joint_tracking_error_rad = max(
                    abs(current - target)
                    for current, target in zip(
                        current_right_joints,
                        right_joint_target,
                        strict=True,
                    )
                )
                node.publish_right_joint_override(
                    {
                        "left": commanded["left"],
                        # Keep the dormant RMPflow target synchronized with
                        # measured FK while direct joints own this tick.
                        "right": tuple(node.ee["right"]),
                    },
                    right_joint_target,
                    spine_target,
                    {"left": 1.0, "right": float(stage["right_open"])},
                )
            else:
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
                final_grippers["right"]
                <= float(
                    stage.get("maximum_right_gripper_open_fraction", 0.25)
                )
                if float(stage["right_open"]) < 0.5
                else final_grippers["right"]
                >= float(stage.get("minimum_right_gripper_open_fraction", 0.80))
            )
            stage_position_tolerance_m = float(
                stage.get("position_tolerance_m", args.position_tolerance_m)
            )
            stage_orientation_tolerance_deg = float(
                stage.get(
                    "orientation_tolerance_deg",
                    args.orientation_tolerance_deg,
                )
            )
            within_pose = (
                errors["left"]["position_m"] <= stage_position_tolerance_m
                and errors["left"]["orientation_deg"]
                <= stage_orientation_tolerance_deg
                and right_joint_tracking_error_rad is not None
                and right_joint_tracking_error_rad <= 0.03
                if "relative_right_joint_spline" in stage
                else all(
                    value["position_m"] <= stage_position_tolerance_m
                    and value["orientation_deg"]
                    <= stage_orientation_tolerance_deg
                    for value in errors.values()
                )
            )
            if (
                raw_fraction < 1.0
                or not within_pose
                or not gripper_ready
                # During the ownership handoff an edge-held deformable pad can
                # keep unrelated arm joints above the global speed threshold.
                # The retain target is the measured handoff pose itself, so
                # bounded bilateral EE error plus a measured closed gripper is
                # the appropriate completion gate here. Later motion stages
                # retain the full joint-settle requirement.
                or (
                    stage["name"] != "retain"
                    and not bool(stage.get("continuous_transit", False))
                    and sample_speed > args.settle_max_joint_speed_rad_s
                )
            ):
                stable_since = None
                continue
            if not bool(stage.get("continuous_transit", False)):
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
                    **(
                        {
                            "target_right_joints_rad": list(right_joint_target),
                            "right_joint_tracking_error_rad": (
                                right_joint_tracking_error_rad
                            ),
                        }
                        if right_joint_target is not None
                        else {}
                    ),
                }
            )
            if stage["name"] in {
                "forward_clear_base",
                "joint_extract_spline",
            }:
                grasp_verification_post_lift = node.wrist_signature(
                    args.maximum_skew_s
                )
                world_perception_post_lift = node.world_perception(
                    args.maximum_skew_s
                )
                if world_perception_post_lift is None and (
                    node.world_perception_error
                    in {
                        "incomplete_camera_geometry",
                        "stale_or_skewed_camera_geometry",
                    }
                ):
                    # The head and wrist cameras publish at different rates.
                    # Hold this completed pose until a coherent pair arrives;
                    # do not mistake one unlucky phase sample for pad loss.
                    stages.pop()
                    continue
                if node.world_perception_error is not None:
                    world_perception_errors.append(
                        "post_lift:" + node.world_perception_error
                    )
                if grasp_verification_post_lift is None:
                    world_perception_errors.append(
                        "post_lift:wrist_signature_unavailable"
                    )
                elif grasp_verification_pre_lift is not None:
                    # A thermal pad is deformable: an edge can be retained
                    # while most visible blue pixels remain on the support.
                    # Keep the rigid RGB-D probe as telemetry, not as a hard
                    # veto of a measured, legal gripper latch.
                    grasp_verification_metrics = grasp_probe_metrics(
                        grasp_verification_pre_lift,
                        grasp_verification_post_lift,
                    )
                if world_perception_post_lift is not None:
                    retention_reference_pad = tuple(
                        float(value)
                        for value in world_perception_post_lift["pad"][
                            "center_world_m"
                        ]
                    )
                    retention_reference_ee = tuple(node.ee["right"])
                    try:
                        previous_target = (
                            world_perception_pre_lift or {}
                        ).get("target")
                        if previous_target is None:
                            raise ValueError(
                                "pre_lift_target_geometry_unavailable"
                            )
                        previous_center = np.asarray(
                            previous_target["center_world_m"],
                            dtype=np.float64,
                        )
                        current_target = world_perception_post_lift["target"]
                        current_center = np.asarray(
                            current_target["center_world_m"],
                            dtype=np.float64,
                        )
                        incremental_target = {
                            "center_world_m": (
                                np.asarray(
                                    REFERENCE_TARGET_CENTER_WORLD_M,
                                    dtype=np.float64,
                                )
                                + current_center
                                - previous_center
                            ).tolist(),
                            "long_axis_yaw_rad_mod_pi": (
                                REFERENCE_TARGET_YAW_RAD
                                + _rectangle_yaw_delta(
                                    float(
                                        current_target[
                                            "long_axis_yaw_rad_mod_pi"
                                        ]
                                    ),
                                    float(
                                        previous_target[
                                            "long_axis_yaw_rad_mod_pi"
                                        ]
                                    ),
                                )
                            ),
                        }
                        plan, incremental_audit = (
                            retarget_transport_plan_to_target(
                                plan, incremental_target
                            )
                        )
                        visual_retarget_audit = {
                            "source": "post_clear_incremental_head_rgbd",
                            "pre_lift_target_center_world_m": (
                                previous_center.tolist()
                            ),
                            "post_clear_target_center_world_m": (
                                current_center.tolist()
                            ),
                            "incremental_transform": incremental_audit,
                        }
                    except ValueError as error:
                        reason = str(error)
                        break
            elif stage["name"] in {
                "transfer",
                "target_overhead",
                "smooth_extract_to_place",
                "smooth_transport_to_place",
                "support_contact",
            }:
                at_target_overhead = stage["name"] in {
                    "target_overhead",
                    "smooth_extract_to_place",
                    "smooth_transport_to_place",
                }
                retention_perception = node.world_perception(
                    args.maximum_skew_s
                )
                runtime_place_retarget_required = (
                    at_target_overhead
                    and args.code_policy_acquire
                    and grasp_relative_place_audit is None
                )
                runtime_supported_alignment_required = (
                    at_target_overhead
                    and args.code_policy_acquire
                    and supported_alignment_audit is None
                )
                if (
                    stage["name"] == "transfer"
                    or runtime_place_retarget_required
                    or runtime_supported_alignment_required
                ) and (
                    retention_perception is None
                ):
                    if stage_capture_wait_started is None:
                        stage_capture_wait_started = node.sim_time
                    if node.sim_time - stage_capture_wait_started < 1.0:
                        stages.pop()
                        continue
                    if runtime_place_retarget_required:
                        reason = (
                            "target_overhead_place_retarget_unavailable:"
                            f"{node.world_perception_error or 'unknown'}"
                        )
                        break
                    if runtime_supported_alignment_required:
                        reason = (
                            "target_supported_alignment_unavailable:"
                            f"{node.world_perception_error or 'unknown'}"
                        )
                        break
                    world_perception_errors.append(
                        "transfer_retention_diagnostic_unavailable:"
                        f"{node.world_perception_error or 'unknown'}"
                    )
                if (
                    retention_reference_pad is None
                    or retention_reference_ee is None
                    or retention_perception is None
                ):
                    world_perception_errors.append(
                        "retention_diagnostic_unavailable:"
                        f"{stage['name']}:"
                        f"{node.world_perception_error or 'no_reference'}"
                    )
                else:
                    retention = pad_retention_metrics(
                        retention_reference_pad,
                        retention_reference_ee,
                        tuple(
                            float(value)
                            for value in retention_perception["pad"][
                                "center_world_m"
                            ]
                        ),
                        tuple(node.ee["right"]),
                        maximum_world_error_m=float(
                            stage.get(
                                "retention_max_world_error_m",
                                PAD_RETENTION_MAX_WORLD_ERROR_M,
                            )
                        ),
                    )
                    retention["stage"] = stage["name"]
                    retention["pad_pixel_count"] = retention_perception[
                        "pad"
                    ]["pixel_count"]
                    retention_checks.append(retention)
                    if stage["name"] == "transfer":
                        transfer_retention_gate = dict(retention)
                        # The pad is deformable and folds during the formal3
                        # route, so a rigid EE-to-visible-centroid transform
                        # is not a sound veto.  It also mixes in newly visible
                        # blue support pixels after lift.  Retain this metric
                        # as diagnostics and let the independent evaluator
                        # judge placement.
                        transfer_retention_gate["enforced"] = False
                        transfer_retention_gate["rationale"] = (
                            "deformable_pad_and_post_lift_blue_occlusion_"
                            "invalidate_rigid_centroid_gate"
                        )
                if (
                    stage["name"] in {
                        "smooth_extract_to_place",
                        "smooth_transport_to_place",
                    }
                ):
                    grasp_verification_post_lift = node.wrist_signature(
                        args.maximum_skew_s
                    )
                    world_perception_post_lift = retention_perception
                    if (
                        grasp_verification_pre_lift is not None
                        and grasp_verification_post_lift is not None
                    ):
                        grasp_verification_metrics = grasp_probe_metrics(
                            grasp_verification_pre_lift,
                            grasp_verification_post_lift,
                        )
                    if (
                        args.code_policy_acquire
                        and retention_perception is not None
                        and world_perception_pre_lift is not None
                    ):
                        try:
                            alignment_pose, supported_alignment_audit = (
                                supported_pad_alignment(
                                    tuple(node.ee["right"]),
                                    retention_perception["pad"],
                                    world_perception_pre_lift["target"],
                                )
                            )
                        except ValueError as error:
                            reason = str(error)
                            break
                        release_safety = pre_release_pad_safety(
                            retention_perception["pad"],
                            world_perception_pre_lift["target"],
                        )
                        supported_alignment_audit[
                            "pre_release_safety"
                        ] = release_safety
                        if not release_safety["release_permitted"]:
                            reason = (
                                "pad_dropped_outside_target_before_release:"
                                f"xy={release_safety['target_xy_error_m']:.6f}:"
                                "surface_delta="
                                f"{release_safety['target_surface_delta_m']:.6f}"
                            )
                            break
                        plan.insert(
                            stage_index + 1,
                            {
                                "name": "supported_rgbd_alignment",
                                "pose": alignment_pose,
                                "right_open": 0.0,
                                "minimum_right_ee_z_m": (
                                    MINIMUM_SUPPORT_EE_Z_M
                                ),
                                "minimum_duration_s": 0.60,
                                "maximum_duration_s": 5.0,
                                "position_tolerance_m": 0.010,
                            },
                        )
                        release_index = next(
                            index
                            for index in range(stage_index + 2, len(plan))
                            if plan[index]["name"] == "release"
                        )
                        release_stage = dict(plan[release_index])
                        base_yaw = float(base_target[2])
                        forward_xy = np.asarray(
                            (math.cos(base_yaw), math.sin(base_yaw))
                        )
                        left_xy = np.asarray(
                            (-forward_xy[1], forward_xy[0])
                        )
                        release_forward_m, release_z_m, release_left_m = (
                            GT_RELEASE_OPEN_FORWARD_Z_LEFT_M
                        )
                        # Large board-slot retargets place the arm in a less
                        # favourable contact-loaded configuration.  At slot D
                        # the measured release endpoint tracked about 7 mm
                        # below its command and an opening finger swept the
                        # target.  Preserve the demonstrated release direction
                        # while adding a bounded 6 mm upward command for those
                        # non-nominal slots; nominal runs retain the GT motion.
                        release_extra_clearance_m = (
                            BASE_RELEASE_TRACKING_CLEARANCE_M
                            + (
                                LARGE_TARGET_ADDITIONAL_RELEASE_CLEARANCE_M
                                if float(
                                    release_stage.get(
                                        "target_retarget_xy_m", 0.0
                                    )
                                )
                                > LARGE_TARGET_RETARGET_XY_M
                                else 0.0
                            )
                        )
                        release_xy = (
                            np.asarray(alignment_pose[:2])
                            + release_forward_m * forward_xy
                            + release_left_m * left_xy
                        )
                        aligned_release_pose = (
                            float(release_xy[0]),
                            float(release_xy[1]),
                            float(
                                alignment_pose[2]
                                + release_z_m
                                + release_extra_clearance_m
                            ),
                            *alignment_pose[3:],
                        )
                        release_delta = np.asarray(
                            aligned_release_pose[:3]
                        ) - np.asarray(release_stage["pose"][:3])
                        release_stage["pose"] = aligned_release_pose
                        plan[release_index] = release_stage
                        retreat_index = next(
                            index
                            for index in range(release_index + 1, len(plan))
                            if plan[index]["name"] == "retreat"
                        )
                        retreat_stage = dict(plan[retreat_index])

                        def shift_release_pose(
                            pose: tuple[float, ...],
                        ) -> tuple[float, ...]:
                            return (
                                float(pose[0] + release_delta[0]),
                                float(pose[1] + release_delta[1]),
                                float(pose[2] + release_delta[2]),
                                *pose[3:],
                            )

                        retreat_stage["pose"] = shift_release_pose(
                            tuple(retreat_stage["pose"])
                        )
                        retreat_stage["right_pose_path"] = [
                            shift_release_pose(tuple(pose))
                            for pose in retreat_stage["right_pose_path"]
                        ]
                        plan[retreat_index] = retreat_stage
                        supported_alignment_audit[
                            "release_extra_clearance_m"
                        ] = release_extra_clearance_m
                if (
                    at_target_overhead
                    and args.code_policy_acquire
                    and retention_perception is not None
                    and grasp_relative_place_audit is None
                ):
                    try:
                        plan, grasp_relative_place_audit = (
                            retarget_place_from_observed_grasp(
                                plan,
                                tuple(node.ee["right"]),
                                retention_perception["pad"],
                                retention_perception["target"],
                            )
                        )
                    except ValueError as error:
                        reason = str(error)
                        break
            stage_index += 1
            stage_initial = None
            stage_initial_right_joints = None
            stage_capture_wait_started = None
    finally:
        success = reason == "stable_hybrid_transport_release_and_retreat"
        result = {
            "schema_version": 1,
            "success": success,
            "reason": reason,
            "control": (
                "camera_code_policy_acquire_then_rmpflow_transport"
                if args.code_policy_acquire
                else "vla_grasp_then_rmpflow_transport"
            ),
            "ground_truth_subscriptions": [],
            "reference": str(args.reference),
            "reference_support_unique_episodes": reference["source"][
                "support_unique_episodes"
            ],
            "base_errors": base_errors,
            "stages": stages,
            "completed_stage_count": len(stages),
            "expected_stage_count": len(plan) if plan is not None else 10,
            "minimum_support_ee_z_m": minimum_support_ee_z_m,
            "support_ee_z_floor_m": MINIMUM_SUPPORT_EE_Z_M,
            "grasp_verification": {
                "mode": "diagnostic_blue_pad_wrist_rgbd_probe_lift",
                "enforced": False,
                "rationale": (
                    "deformable edge grasps do not preserve the visible "
                    "RGB-D centroid as a rigid body"
                ),
                "pre_lift": grasp_verification_pre_lift,
                "post_lift": grasp_verification_post_lift,
                "metrics": grasp_verification_metrics,
            },
            "pad_retention_checks": retention_checks,
            "transfer_retention_gate": transfer_retention_gate,
            "world_perception_audit": {
                "mode": "camera_only_rgbd_optical_fk_bounded_target_retarget",
                "pre_lift": world_perception_pre_lift,
                "post_lift": world_perception_post_lift,
                "errors": world_perception_errors,
                "retarget": visual_retarget_audit,
                "grasp_relative_place": grasp_relative_place_audit,
                "supported_alignment": supported_alignment_audit,
            },
            "code_policy_acquire": code_policy_acquire_audit,
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
