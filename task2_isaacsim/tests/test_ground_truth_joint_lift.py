#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from task2_isaacsim.baselines.pi05.live.ground_truth_joint_lift import (
    REFERENCE_BASE_XYYAW,
    REFERENCE_PAD_XYYAW,
    JOINT_LANDMARKS,
    anchored_base_pose,
    bounded_axis_step,
    bounded_orientation_step,
    bounded_planar_offset,
    interpolate_base_pose,
    interpolate_landmark,
    placement_release_ready,
    quaternion_error_deg,
    yaw_rotated_planar_offset,
)


class GroundTruthJointReplayTest(unittest.TestCase):
    def test_landmark_interpolation_and_release(self) -> None:
        q, grip = interpolate_landmark(834.5)
        self.assertAlmostEqual(q[0], (JOINT_LANDMARKS[-4][1][0] + JOINT_LANDMARKS[-3][1][0]) / 2.0)
        self.assertAlmostEqual(grip, 0.5)

    def test_nominal_anchor_reproduces_reference_base(self) -> None:
        pose = (
            REFERENCE_PAD_XYYAW[0],
            REFERENCE_PAD_XYYAW[1],
            0.85,
            math.cos(REFERENCE_PAD_XYYAW[2] / 2.0),
            0.0,
            0.0,
            math.sin(REFERENCE_PAD_XYYAW[2] / 2.0),
        )
        actual = anchored_base_pose(pose, REFERENCE_PAD_XYYAW)
        for observed, expected in zip(actual, REFERENCE_BASE_XYYAW, strict=True):
            self.assertAlmostEqual(observed, expected)

    def test_anchor_translation_and_rotation_transform_base(self) -> None:
        live_yaw = REFERENCE_PAD_XYYAW[2] + math.pi / 2.0
        pose = (
            REFERENCE_PAD_XYYAW[0] + 1.0,
            REFERENCE_PAD_XYYAW[1] - 0.5,
            0.85,
            math.cos(live_yaw / 2.0),
            0.0,
            0.0,
            math.sin(live_yaw / 2.0),
        )
        actual = anchored_base_pose(pose, REFERENCE_PAD_XYYAW)
        offset_x = REFERENCE_BASE_XYYAW[0] - REFERENCE_PAD_XYYAW[0]
        offset_y = REFERENCE_BASE_XYYAW[1] - REFERENCE_PAD_XYYAW[1]
        self.assertAlmostEqual(actual[0], pose[0] - offset_y)
        self.assertAlmostEqual(actual[1], pose[1] + offset_x)
        self.assertAlmostEqual(actual[2], REFERENCE_BASE_XYYAW[2] + math.pi / 2.0)

    def test_anchor_yaw_delta_can_be_clamped(self) -> None:
        live_yaw = REFERENCE_PAD_XYYAW[2] + math.radians(10.0)
        pose = (
            REFERENCE_PAD_XYYAW[0],
            REFERENCE_PAD_XYYAW[1],
            0.85,
            math.cos(live_yaw / 2.0),
            0.0,
            0.0,
            math.sin(live_yaw / 2.0),
        )
        actual = anchored_base_pose(
            pose,
            REFERENCE_PAD_XYYAW,
            max_yaw_delta_rad=math.radians(5.0),
        )
        self.assertAlmostEqual(
            actual[2],
            REFERENCE_BASE_XYYAW[2] + math.radians(5.0),
        )

    def test_base_yaw_takes_short_path(self) -> None:
        midpoint = interpolate_base_pose(
            (0.0, 0.0, math.radians(179.0)),
            (2.0, 4.0, math.radians(-179.0)),
            0.5,
        )
        self.assertAlmostEqual(midpoint[0], 1.0)
        self.assertAlmostEqual(midpoint[1], 2.0)
        self.assertAlmostEqual(abs(midpoint[2]), math.pi)

    def test_cartesian_step_is_speed_bounded(self) -> None:
        self.assertAlmostEqual(bounded_axis_step(-0.16, 0.08, 0.10), -0.008)
        self.assertAlmostEqual(bounded_axis_step(0.003, 0.08, 0.10), 0.003)
        self.assertEqual(bounded_axis_step(1.0, 0.08, -1.0), 0.0)

    def test_cartesian_planar_offset_is_bounded(self) -> None:
        self.assertEqual(bounded_planar_offset(0.03, 0.04, 0.08), (0.03, 0.04))
        x, y = bounded_planar_offset(0.3, 0.4, 0.08)
        self.assertAlmostEqual(x, 0.048)
        self.assertAlmostEqual(y, 0.064)
        self.assertEqual(bounded_planar_offset(0.3, 0.4, -1.0), (0.0, 0.0))

    def test_release_requires_near_surface_xy_and_height(self) -> None:
        ready = placement_release_ready(
            xy_error_m=0.010,
            pad_height_m=0.009,
            orientation_error_deg=2.0,
            release_xy_m=0.015,
            release_height_m=0.006,
            release_height_tolerance_m=0.008,
            orientation_tolerance_deg=3.0,
        )
        self.assertTrue(ready)
        self.assertFalse(
            placement_release_ready(
                xy_error_m=0.010,
                pad_height_m=0.16,
                orientation_error_deg=2.0,
                release_xy_m=0.015,
                release_height_m=0.006,
                release_height_tolerance_m=0.008,
                orientation_tolerance_deg=3.0,
            )
        )

    def test_randomized_release_does_not_require_target_xy(self) -> None:
        self.assertTrue(
            placement_release_ready(
                xy_error_m=0.080,
                pad_height_m=0.009,
                orientation_error_deg=2.0,
                release_xy_m=0.015,
                release_height_m=0.006,
                release_height_tolerance_m=0.008,
                orientation_tolerance_deg=3.0,
                require_target_xy=False,
            )
        )
        self.assertFalse(
            placement_release_ready(
                xy_error_m=0.020,
                pad_height_m=0.009,
                orientation_error_deg=2.0,
                release_xy_m=0.015,
                release_height_m=0.006,
                release_height_tolerance_m=0.008,
                orientation_tolerance_deg=3.0,
            )
        )
        self.assertFalse(
            placement_release_ready(
                xy_error_m=0.010,
                pad_height_m=0.009,
                orientation_error_deg=4.0,
                release_xy_m=0.015,
                release_height_m=0.006,
                release_height_tolerance_m=0.008,
                orientation_tolerance_deg=3.0,
            )
        )

    def test_orientation_step_is_angular_speed_bounded(self) -> None:
        identity = (0.0, 0.0, 0.0, 1.0)
        ninety_degrees = (
            math.sin(math.pi / 4.0),
            0.0,
            0.0,
            math.cos(math.pi / 4.0),
        )
        stepped = bounded_orientation_step(
            identity, ninety_degrees, speed_deg_s=25.0, elapsed=0.2
        )
        self.assertAlmostEqual(quaternion_error_deg(identity, stepped), 5.0)
        self.assertAlmostEqual(
            quaternion_error_deg(stepped, ninety_degrees), 85.0
        )

    def test_contact_sweep_rotates_with_target_yaw(self) -> None:
        x, y = yaw_rotated_planar_offset((0.05, 0.01), math.pi / 2.0)
        self.assertAlmostEqual(x, -0.01)
        self.assertAlmostEqual(y, 0.05)



if __name__ == "__main__":
    unittest.main()
