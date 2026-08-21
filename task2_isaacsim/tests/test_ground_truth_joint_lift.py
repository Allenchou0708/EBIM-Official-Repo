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
    interpolate_base_pose,
    interpolate_landmark,
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


if __name__ == "__main__":
    unittest.main()
