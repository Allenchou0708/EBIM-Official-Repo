#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from task2_isaacsim.baselines.pi05.live.ground_truth_task import (
    PHASES,
    REFERENCE_PAD_CENTROIDS,
    REFERENCE_PAD_WXYZ,
    REFERENCE_TARGET_WXYZ,
    tracking_corrected_orientation,
    tracking_corrected_position,
    transform_landmark,
    transform_lift_landmark,
)
from task2_isaacsim.baselines.pi05.live.ground_truth_pregrasp import (
    _yaw_from_wxyz,
)


class GroundTruthTaskTest(unittest.TestCase):
    def test_nominal_anchors_reproduce_episode_landmarks(self) -> None:
        for phase in PHASES:
            anchor = (
                REFERENCE_PAD_WXYZ
                if phase.anchor == "thermalpad"
                else REFERENCE_TARGET_WXYZ
            )
            transformed = transform_landmark(phase, anchor)
            for actual, expected in zip(
                transformed, phase.right_ee_xyzw, strict=True
            ):
                self.assertAlmostEqual(actual, expected, places=7)

    def test_pad_yaw_rotates_approach_offset_and_orientation(self) -> None:
        phase = PHASES[0]
        yaw = _yaw_from_wxyz(REFERENCE_PAD_WXYZ[3:7]) + math.pi / 2.0
        anchor = (
            REFERENCE_PAD_WXYZ[0],
            REFERENCE_PAD_WXYZ[1],
            REFERENCE_PAD_WXYZ[2],
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        )
        transformed = transform_landmark(phase, anchor)
        reference_offset = (
            phase.right_ee_xyzw[0] - REFERENCE_PAD_WXYZ[0],
            phase.right_ee_xyzw[1] - REFERENCE_PAD_WXYZ[1],
        )
        self.assertAlmostEqual(
            transformed[0] - anchor[0], -reference_offset[1], places=7
        )
        self.assertAlmostEqual(
            transformed[1] - anchor[1], reference_offset[0], places=7
        )

    def test_nominal_lift_mesh_anchors_reproduce_episode_landmarks(self) -> None:
        for phase in PHASES:
            if phase.frame not in REFERENCE_PAD_CENTROIDS:
                continue
            transformed = transform_lift_landmark(
                phase,
                REFERENCE_PAD_CENTROIDS[phase.frame],
                REFERENCE_PAD_WXYZ,
            )
            for actual, expected in zip(
                transformed, phase.right_ee_xyzw, strict=True
            ):
                self.assertAlmostEqual(actual, expected, places=7)

    def test_sequence_closes_before_lift_and_opens_before_retract(self) -> None:
        by_name = {phase.name: phase for phase in PHASES}
        self.assertEqual(
            [phase.frame for phase in PHASES if phase.name.startswith("approach")],
            [330, 350, 370, 399],
        )
        self.assertEqual(by_name["approach"].right_gripper_open, 1.0)
        self.assertEqual(by_name["grasp_align"].right_gripper_open, 1.0)
        self.assertEqual(by_name["grasp"].right_gripper_open, 0.0)
        self.assertEqual(by_name["grasp"].frame, 410)
        self.assertEqual(by_name["lift_start"].frame, 424)
        self.assertEqual(by_name["lift_start"].right_gripper_open, 0.0)
        self.assertEqual(
            [phase.frame for phase in PHASES if phase.name.startswith("lift_")],
            [424, 428, 432, 436, 440, 444, 448],
        )
        self.assertEqual(by_name["lift"].right_gripper_open, 0.0)
        self.assertEqual(by_name["release"].right_gripper_open, 1.0)
        self.assertEqual(by_name["retract"].right_gripper_open, 1.0)

    def test_tracking_correction_uses_measured_error_and_limit(self) -> None:
        corrected = tracking_corrected_position(
            (1.0, 2.0, 3.0),
            (1.1, 2.0, 3.0),
            (1.0, 2.0, 3.0),
            gain=2.0,
            limit_m=0.05,
            fraction=1.0,
        )
        self.assertEqual(corrected, (1.05, 2.0, 3.0))

        at_start = tracking_corrected_position(
            (1.0, 2.0, 3.0),
            (1.1, 2.0, 3.0),
            (1.0, 2.0, 3.0),
            gain=2.0,
            limit_m=0.05,
            fraction=0.0,
        )
        self.assertEqual(at_start, (1.0, 2.0, 3.0))

    def test_orientation_tracking_correction_is_bounded(self) -> None:
        target_angle = math.radians(20.0)
        target = (
            0.0,
            0.0,
            math.sin(target_angle / 2.0),
            math.cos(target_angle / 2.0),
        )
        corrected = tracking_corrected_orientation(
            target,
            target,
            (0.0, 0.0, 0.0, 1.0),
            gain=2.0,
            limit_deg=10.0,
            fraction=1.0,
        )
        expected_angle = math.radians(30.0)
        self.assertAlmostEqual(corrected[2], math.sin(expected_angle / 2.0))
        self.assertAlmostEqual(corrected[3], math.cos(expected_angle / 2.0))


if __name__ == "__main__":
    unittest.main()
