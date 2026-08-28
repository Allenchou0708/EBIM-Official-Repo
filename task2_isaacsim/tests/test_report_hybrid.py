# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import math
import unittest
from pathlib import Path

import numpy as np

from task2_isaacsim.baselines.report_hybrid.core import (
    PHASES,
    ForwardOnlyFSM,
    assert_policy_topics,
    compose_initial_relative_xyyaw,
    dark_pad_mask,
    deproject_masked_depth,
    load_reference,
    red_target_mask,
    retarget_landmarks,
)


class ReportHybridTest(unittest.TestCase):
    def test_reference_has_multi_episode_support(self):
        reference = load_reference()
        source = reference["source"]
        self.assertEqual(source["support_unique_episodes"], 180)
        self.assertEqual(
            source["support_by_collection"],
            {
                "task2_fixpos_v1": 20,
                "task2_fixpos_v2": 25,
                "task2_fixpos_v3": 90,
                "task2_fixpos_v4": 45,
            },
        )
        self.assertFalse(source["episode_19_special_case"])
        self.assertFalse(source["legacy_duplicates_used"])

    def test_s0_base_is_initial_relative(self):
        reference = load_reference()
        relative = reference["s0"]["initial_relative_base_xyyaw"]
        expected = compose_initial_relative_xyyaw(
            (4.4, 2.6, -math.pi / 2), relative
        )
        np.testing.assert_allclose(
            expected, reference["s0"]["reference_world_base_xyyaw"], atol=1e-8
        )
        rotated = compose_initial_relative_xyyaw(
            (1.0, 2.0, math.pi / 2), (1, 0, 0)
        )
        np.testing.assert_allclose(rotated, (1.0, 3.0, math.pi / 2), atol=1e-8)

    def test_forbidden_runtime_topics_fail_closed(self):
        legal = (
            "/isaac/head_camera/image_raw",
            "/isaac/right_wrist_camera/depth",
            "/isaac/odom",
        )
        self.assertEqual(assert_policy_topics(legal), legal)
        for topic in (
            "/isaac/eval_camera/image_raw",
            "/isaac/task2/object_poses",
            "/isaac/task2/pad_points",
            "/evaluate_task2",
        ):
            with self.assertRaises(ValueError):
                assert_policy_topics((*legal, topic))

    def test_rgb_masks_and_rgbd_projection(self):
        rgb = np.full((12, 12, 3), 180, dtype=np.uint8)
        rgb[2:6, 2:6] = (180, 30, 20)
        rgb[7:11, 4:8] = (50, 48, 47)
        self.assertEqual(int(red_target_mask(rgb).sum()), 16)
        self.assertGreater(int(dark_pad_mask(rgb).sum()), 0)
        depth = np.ones((12, 12), dtype=np.float32)
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:10, 2:10] = True
        points = deproject_masked_depth(
            depth,
            mask,
            (10.0, 10.0, 6.0, 6.0),
            (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            minimum_depth_m=0.1,
            maximum_depth_m=2.0,
        )
        np.testing.assert_allclose(
            np.median(points, axis=0)[2], 4.0, atol=1e-8
        )

    def test_bounded_translation_and_yaw_retarget(self):
        reference = load_reference()
        anchors = reference["reference_anchors"]
        targets = retarget_landmarks(
            reference,
            np.asarray(anchors["pad_center_xyz"]) + (0.01, -0.02, 0.0),
            np.asarray(anchors["target_center_xyz"]) + (-0.015, 0.01, 0.0),
            math.pi / 2 + math.radians(5),
            math.pi / 2 - math.radians(4),
        )
        self.assertEqual(
            set(targets), set(reference["right_ee_landmarks_xyzw"])
        )
        self.assertAlmostEqual(
            targets["approach"][0]
            - reference["right_ee_landmarks_xyzw"]["approach"][0],
            0.01,
        )
        with self.assertRaises(ValueError):
            retarget_landmarks(
                reference,
                (1.95, 1.95, 0.85),
                anchors["target_center_xyz"],
                math.pi / 2,
                math.pi / 2,
            )
        with self.assertRaises(ValueError):
            retarget_landmarks(
                reference,
                anchors["pad_center_xyz"],
                anchors["target_center_xyz"],
                math.pi / 2 + math.radians(11),
                math.pi / 2,
            )

    def test_fsm_is_forward_only_and_latches_gripper(self):
        fsm = ForwardOnlyFSM()
        observed = [(fsm.phase, fsm.right_gripper_open_fraction)]
        for phase in PHASES[1:]:
            fsm.advance(phase)
            observed.append((fsm.phase, fsm.right_gripper_open_fraction))
        fsm.validate_terminal()
        self.assertEqual(fsm.close_count, 1)
        self.assertEqual(fsm.open_count, 1)
        self.assertEqual(
            [phase for phase, value in observed if value == 0.0],
            ["acquire", "peel_lift", "transfer_place"],
        )
        self.assertTrue(fsm.base_and_spine_locked)
        with self.assertRaises(ValueError):
            ForwardOnlyFSM().advance("insert")

    def test_left_hold_and_workspace_are_constant_and_bounded(self):
        reference = load_reference()
        left = reference["s0"]["left_safe_ee_xyzw"]
        self.assertEqual(len(left), 7)
        self.assertAlmostEqual(
            sum(value * value for value in left[3:]), 1.0, delta=1.0e-3
        )
        minimum = np.asarray(reference["limits"]["world_workspace_min_xyz"])
        maximum = np.asarray(reference["limits"]["world_workspace_max_xyz"])
        for pose in reference["right_ee_landmarks_xyzw"].values():
            self.assertTrue(np.all(np.asarray(pose[:3]) >= minimum))
            self.assertTrue(np.all(np.asarray(pose[:3]) <= maximum))

    def test_policy_source_does_not_import_external_audit(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "baselines/report_hybrid/policy_node.py").read_text()
        self.assertNotIn(
            "from task2_isaacsim.baselines.report_hybrid.external_audit",
            source,
        )
        self.assertNotIn('TOPICS["ground_truth"]', source)
        topics = json.loads(json.dumps(load_reference()["source"]))
        self.assertEqual(topics["support_unique_episodes"], 180)


if __name__ == "__main__":
    unittest.main()
