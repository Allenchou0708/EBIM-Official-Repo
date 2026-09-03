# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from task2_isaacsim.baselines.pi05.live.fixed_stage_observation import (
    RMPFLOW_STAGE_PLAN,
    load_reference,
    minimum_jerk_fraction,
    transition_duration_s,
)
from task2_isaacsim.baselines.pi05.live.fixed_hybrid_transport import (
    build_transport_plan,
    load_hybrid_reference,
)


class FixedStageObservationTest(unittest.TestCase):
    def test_hybrid_plan_holds_grasp_until_stable_place(self) -> None:
        reference_path = (
            Path(__file__).parents[1]
            / "baselines"
            / "pi05"
            / "live"
            / "observation_reference_180dev_v1.json"
        )
        reference = load_hybrid_reference(reference_path)
        initial = tuple(reference["right_observation_ee_world_xyzw"])
        plan = build_transport_plan(reference, initial)
        self.assertEqual(
            [stage["name"] for stage in plan],
            [
                "retain",
                "peel_lift",
                "transfer",
                "preplace",
                "place",
                "release",
                "retreat",
            ],
        )
        self.assertTrue(
            all(stage["right_open"] == 0.0 for stage in plan[:5])
        )
        self.assertTrue(
            all(stage["right_open"] == 1.0 for stage in plan[5:])
        )
        self.assertGreater(plan[2]["pose"][0] - plan[1]["pose"][0], 0.15)
        self.assertGreater(plan[3]["pose"][2], plan[4]["pose"][2])

    def test_rmpflow_plan_ends_at_the_only_policy_observation_pose(self) -> None:
        self.assertEqual(
            [target_kind for target_kind, _ in RMPFLOW_STAGE_PLAN],
            [
                "safe_orientation",
                "clearance",
                "observation",
            ],
        )

    def test_production_reference_keeps_clearance_above_final_pregrasp(self) -> None:
        reference_path = (
            Path(__file__).parents[1]
            / "baselines"
            / "pi05"
            / "live"
            / "observation_reference_180dev_v1.json"
        )
        payload = load_reference(reference_path)
        safe_orientation = payload[
            "right_safe_orientation_waypoint_ee_world_xyzw"
        ]
        orientation_midpoint = payload[
            "right_orientation_midpoint_waypoint_ee_world_xyzw"
        ]
        clearance = payload["right_clearance_waypoint_ee_world_xyzw"]
        observation = payload["right_observation_ee_world_xyzw"]

        self.assertNotEqual(safe_orientation[3:], observation[3:])
        self.assertNotEqual(orientation_midpoint[3:], safe_orientation[3:])
        self.assertNotEqual(orientation_midpoint[3:], clearance[3:])
        self.assertGreater(clearance[2], observation[2])
        self.assertEqual(clearance[:2], observation[:2])
        self.assertEqual(clearance[3:], observation[3:])
        self.assertEqual(
            payload["right_observation_derivation"]["pose"],
            "development pregrasp median",
        )

    def test_minimum_jerk_is_bounded_and_has_zero_endpoint_slope(self) -> None:
        self.assertEqual(minimum_jerk_fraction(-1.0), 0.0)
        self.assertEqual(minimum_jerk_fraction(0.0), 0.0)
        self.assertAlmostEqual(minimum_jerk_fraction(0.5), 0.5)
        self.assertEqual(minimum_jerk_fraction(1.0), 1.0)
        self.assertEqual(minimum_jerk_fraction(2.0), 1.0)
        epsilon = 1.0e-4
        self.assertLess(minimum_jerk_fraction(epsilon) / epsilon, 1.0e-5)
        self.assertLess(
            (1.0 - minimum_jerk_fraction(1.0 - epsilon)) / epsilon,
            1.0e-5,
        )

    def test_transition_duration_uses_slowest_pose_delta(self) -> None:
        initial = {
            "left": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "right": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        }
        targets = {
            "left": (0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "right": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        }
        self.assertEqual(
            transition_duration_s(
                initial,
                targets,
                max_linear_speed_m_s=0.05,
                max_angular_speed_deg_s=20.0,
                minimum_s=6.0,
                maximum_s=18.0,
            ),
            15.0,
        )

    def test_reference_requires_development_support(self) -> None:
        payload = {
            "schema_version": 1,
            "source": {"split": "heldout", "support_unique_episodes": 180},
            "base_xyyaw": [1, 2, 3],
            "spine_command_m": 0.5,
            "left_safe_ee_world_xyzw": [1, 2, 3, 0, 0, 0, 1],
            "right_observation_ee_world_xyzw": [1, 2, 3, 0, 0, 0, 1],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "development only"):
                load_reference(path)


if __name__ == "__main__":
    unittest.main()
