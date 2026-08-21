#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from task2_isaacsim.baselines.pi05.live.ground_truth_pregrasp import (
    _actual_targets,
    _orientation_error_deg,
    _yaw_from_wxyz,
)


class GroundTruthPregraspTest(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = {
            "selection": {"episode": 19, "frame": 310},
            "final_target": {
                "measured_reference": {
                    "left_ee": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                    "right_ee": [1.1, 2.2, 3.3, 0.0, 0.0, 0.0, 1.0],
                    "right_ee_relative_to_thermalpad_m": [0.1, 0.2, 0.3],
                    "thermalpad_position_m": [1.0, 2.0, 3.0],
                }
            },
        }

    def test_nominal_pad_pose_reproduces_audited_target(self) -> None:
        objects = {
            "thermalpad": (
                1.0,
                2.0,
                3.0,
                math.sqrt(0.5),
                0.0,
                0.0,
                math.sqrt(0.5),
            )
        }
        targets, provenance = _actual_targets(
            self.audit, objects, _yaw_from_wxyz(objects["thermalpad"][3:7]), math.pi / 2
        )
        self.assertEqual(targets["left"], tuple(self.audit["final_target"]["measured_reference"]["left_ee"]))
        self.assertAlmostEqual(targets["right"][0], 1.1, places=7)
        self.assertAlmostEqual(targets["right"][1], 2.2, places=7)
        self.assertAlmostEqual(targets["right"][2], 3.3, places=7)
        self.assertAlmostEqual(provenance["pad_yaw_delta_deg"], 0.0, places=6)

    def test_pad_yaw_rotates_offset_and_orientation(self) -> None:
        yaw = math.pi
        objects = {
            "thermalpad": (1.0, 2.0, 3.0, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        }
        targets, _ = _actual_targets(self.audit, objects, yaw, math.pi / 2)
        self.assertAlmostEqual(targets["right"][0], 0.8, places=7)
        self.assertAlmostEqual(targets["right"][1], 2.1, places=7)
        self.assertAlmostEqual(targets["right"][2], 3.3, places=7)
        expected = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
        self.assertAlmostEqual(
            _orientation_error_deg((*targets["right"][:3], *expected), targets["right"]),
            0.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
