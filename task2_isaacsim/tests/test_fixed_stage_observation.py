# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from task2_isaacsim.baselines.pi05.live.fixed_stage_observation import (
    RMPFLOW_STAGE_PLAN,
    load_reference,
    minimum_jerk_fraction,
    transition_duration_s,
)
from task2_isaacsim.baselines.pi05.live.fixed_hybrid_transport import (
    blue_pad_depth_signature,
    build_code_policy_acquire_plan,
    build_transport_plan,
    collapse_code_policy_transport,
    continuous_joint_spline,
    continuous_landmark_pose,
    cubic_bezier_position,
    deproject_masked_depth,
    grasp_probe_metrics,
    load_hybrid_reference,
    pad_retention_metrics,
    pre_release_pad_safety,
    pose_path_length,
    retarget_place_from_observed_grasp,
    retarget_transport_plan_to_target,
    robust_world_surface_signature,
    short_ramp_fraction,
    supported_pad_alignment,
)


class FixedStageObservationTest(unittest.TestCase):
    def test_pre_release_safety_vetoes_pad_dropped_off_target(self) -> None:
        metrics = pre_release_pad_safety(
            {"center_world_m": [1.946, 1.887, 0.750]},
            {"center_world_m": [1.950, 1.941, 0.748]},
        )
        self.assertFalse(metrics["release_permitted"])
        self.assertTrue(metrics["pad_is_on_surface"])
        self.assertTrue(metrics["pad_is_outside_target"])

    def test_pre_release_safety_allows_supported_or_elevated_pad(self) -> None:
        supported = pre_release_pad_safety(
            {"center_world_m": [1.948, 1.925, 0.751]},
            {"center_world_m": [1.950, 1.941, 0.748]},
        )
        elevated = pre_release_pad_safety(
            {"center_world_m": [1.946, 1.887, 0.810]},
            {"center_world_m": [1.950, 1.941, 0.748]},
        )
        self.assertTrue(supported["release_permitted"])
        self.assertTrue(elevated["release_permitted"])

    def test_supported_alignment_corrects_only_target_cross_axis(self) -> None:
        pose, audit = supported_pad_alignment(
            (2.14, 2.02, 0.915, 0.0, 0.0, 0.0, 1.0),
            {
                "center_world_m": [2.141, 1.931, 0.750],
                "long_axis_yaw_rad_mod_pi": np.pi / 2,
                "pixel_count": 42000,
                "major_visible_extent_m": 0.089,
            },
            {
                "center_world_m": [2.150, 1.950, 0.748],
                "long_axis_yaw_rad_mod_pi": np.pi / 2,
            },
        )
        self.assertAlmostEqual(pose[0], 2.145)
        self.assertAlmostEqual(pose[1], 2.02)
        self.assertAlmostEqual(
            audit["applied_cross_axis_translation_m"], -0.005
        )
        self.assertTrue(audit["translation_applied"])

    def test_relative_joint_spline_hits_knots_and_is_c1(self) -> None:
        knots = ((0.0, 0.0), (1.0, 0.5), (1.5, 2.0))
        self.assertEqual(continuous_joint_spline(knots, 0.0), knots[0])
        self.assertTrue(
            np.allclose(continuous_joint_spline(knots, 0.5), knots[1])
        )
        self.assertEqual(continuous_joint_spline(knots, 1.0), knots[-1])
        epsilon = 1.0e-5
        left_velocity = (
            np.asarray(continuous_joint_spline(knots, 0.5))
            - np.asarray(continuous_joint_spline(knots, 0.5 - epsilon))
        ) / epsilon
        right_velocity = (
            np.asarray(continuous_joint_spline(knots, 0.5 + epsilon))
            - np.asarray(continuous_joint_spline(knots, 0.5))
        ) / epsilon
        self.assertTrue(np.allclose(left_velocity, right_velocity, atol=1e-3))

    def test_continuous_landmarks_have_no_internal_velocity_reset(self) -> None:
        poses = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ]
        knot = 1.0 / (1.0 + np.sqrt(2.0))
        self.assertTrue(
            np.allclose(continuous_landmark_pose(poses, knot)[:3], poses[1][:3])
        )
        epsilon = 1.0e-5
        at_knot = np.asarray(continuous_landmark_pose(poses, knot)[:3])
        velocity_before = (
            at_knot
            - np.asarray(continuous_landmark_pose(poses, knot - epsilon)[:3])
        ) / epsilon
        velocity_after = (
            np.asarray(continuous_landmark_pose(poses, knot + epsilon)[:3])
            - at_knot
        ) / epsilon
        self.assertGreater(float(np.linalg.norm(velocity_before)), 0.1)
        self.assertTrue(np.allclose(velocity_before, velocity_after, atol=1.0e-3))

    def test_continuous_path_reserves_time_for_local_rotation(self) -> None:
        poses = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.01, 0.0, 0.0, 0.0, 0.0, 2**-0.5, 2**-0.5),
        ]
        scale = 0.30
        metric_length = pose_path_length(poses, scale)
        self.assertAlmostEqual(metric_length, 1.0 + scale * np.pi / 2)
        first_segment_fraction = 1.0 / metric_length
        self.assertTrue(
            np.allclose(
                continuous_landmark_pose(
                    poses,
                    first_segment_fraction,
                    scale,
                )[:3],
                poses[1][:3],
            )
        )

    def test_continuous_path_accepts_nonuniform_temporal_knots(self) -> None:
        poses = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ]
        temporal_knots = [0.0, 0.75, 1.0]
        at_knot = continuous_landmark_pose(
            poses, temporal_knots[1], landmark_fractions=temporal_knots
        )
        self.assertTrue(np.allclose(at_knot[:3], poses[1][:3]))
        epsilon = 1.0e-5
        velocity_before = (
            np.asarray(at_knot[:3])
            - np.asarray(
                continuous_landmark_pose(
                    poses,
                    temporal_knots[1] - epsilon,
                    landmark_fractions=temporal_knots,
                )[:3]
            )
        ) / epsilon
        velocity_after = (
            np.asarray(
                continuous_landmark_pose(
                    poses,
                    temporal_knots[1] + epsilon,
                    landmark_fractions=temporal_knots,
                )[:3]
            )
            - np.asarray(at_knot[:3])
        ) / epsilon
        self.assertTrue(np.allclose(velocity_before, velocity_after, atol=1.0e-3))

    def test_short_ramp_reaches_cruise_without_midroute_stop(self) -> None:
        ramp = 0.12
        samples = [short_ramp_fraction(value, ramp) for value in np.linspace(0, 1, 101)]
        self.assertAlmostEqual(samples[0], 0.0)
        self.assertAlmostEqual(samples[-1], 1.0)
        self.assertTrue(all(right > left for left, right in zip(samples, samples[1:])))
        middle_speed = (
            short_ramp_fraction(0.51, ramp)
            - short_ramp_fraction(0.49, ramp)
        ) / 0.02
        self.assertAlmostEqual(middle_speed, 1.0 / (1.0 - ramp))

    def test_code_policy_acquire_uses_bounded_visible_pad_delta(self) -> None:
        reference_path = (
            Path(__file__).parents[1]
            / "baselines"
            / "pi05"
            / "live"
            / "observation_reference_180dev_v1.json"
        )
        reference = load_hybrid_reference(reference_path)
        perception = {
            "pad": {
                "center_world_m": [1.7598657282, 1.9556426936, 0.852],
                "long_axis_yaw_rad_mod_pi": np.pi / 2,
            },
            "target": {
                "center_world_m": [2.1500000954, 1.9500000477, 0.75],
                "long_axis_yaw_rad_mod_pi": np.pi / 2,
            },
        }
        plan, audit = build_code_policy_acquire_plan(reference, perception)
        self.assertEqual(plan[0]["name"], "code_insert")
        self.assertEqual(plan[1]["name"], "code_close")
        self.assertEqual(plan[2]["name"], "smooth_extract_to_place")
        self.assertEqual(
            plan[2]["path_landmark_names"],
            [
                "safe_vertical_z_10mm",
                "gt_first_z_15mm",
                "gt_first_z_20mm",
                "short_diagonal_clearance",
                "forward_rising_extract",
                "retained_lift",
                "forward_clear_base",
                "transfer",
                "gt_place_fraction_025",
                "gt_place_fraction_050",
                "gt_place_2s_before",
                "gt_place_1s_before",
                "gt_place_05s_before",
                "gt_place_025s_before",
                "support_place",
            ],
        )
        self.assertEqual(len(plan[2]["right_pose_path"]), 15)
        self.assertAlmostEqual(plan[2]["max_linear_speed_m_s"], 0.22)
        self.assertAlmostEqual(plan[2]["short_ramp_fraction"], 0.12)
        self.assertAlmostEqual(plan[2]["minimum_duration_s"], 6.20)
        self.assertEqual(len(plan[2]["path_landmark_times_s"]), 15)
        path = [plan[1]["pose"], *plan[2]["right_pose_path"]]
        timing = [0.0, *plan[2]["path_landmark_times_s"]]
        temporal_knots = [
            short_ramp_fraction(
                value / timing[-1], plan[2]["short_ramp_fraction"]
            )
            for value in timing
        ]
        sample_times = np.linspace(0.0, timing[-1], 2001)
        sample_positions = np.asarray(
            [
                continuous_landmark_pose(
                    path,
                    short_ramp_fraction(
                        value / timing[-1], plan[2]["short_ramp_fraction"]
                    ),
                    landmark_fractions=temporal_knots,
                )[:3]
                for value in sample_times
            ]
        )
        peak_linear_speed = float(
            np.max(
                np.linalg.norm(np.diff(sample_positions, axis=0), axis=1)
                / np.diff(sample_times)
            )
        )
        self.assertLess(peak_linear_speed, 0.275)
        self.assertAlmostEqual(plan[2]["position_tolerance_m"], 0.010)
        self.assertEqual(plan[3]["name"], "release")
        self.assertEqual(plan[4]["name"], "retreat")
        self.assertEqual(plan[5]["name"], "reset_clear_view")
        self.assertEqual(plan[0]["right_open"], 1.0)
        self.assertEqual(plan[1]["right_open"], 0.0)
        self.assertAlmostEqual(plan[1]["minimum_duration_s"], 0.50)
        self.assertAlmostEqual(
            plan[1]["maximum_right_gripper_open_fraction"], 0.005
        )
        self.assertTrue(plan[1]["continuous_transit"])
        self.assertAlmostEqual(audit["translation_delta_world_m"][0], 0.01)
        self.assertAlmostEqual(audit["translation_delta_world_m"][1], -0.005)
        self.assertAlmostEqual(audit["insert_depth_bias_world_y_m"], 0.0)
        self.assertAlmostEqual(audit["yaw_delta_deg"], 0.0)
        self.assertAlmostEqual(plan[0]["pose"][2], 0.8732815097)
        self.assertAlmostEqual(audit["insert_z_bias_m"], 0.0)
        self.assertAlmostEqual(audit["close_retraction_m"], 0.020)
        self.assertAlmostEqual(
            np.linalg.norm(
                np.asarray(plan[1]["pose"][:2])
                - np.asarray(plan[0]["pose"][:2])
            ),
            0.020,
        )
        self.assertIn(
            "markley_mean_stable_latch", audit["reference_grasp_selection"]
        )
        with self.assertRaisesRegex(ValueError, "pad_xy_retarget"):
            build_code_policy_acquire_plan(
                reference,
                {
                    "pad": {
                        "center_world_m": [1.90, 1.96, 0.85],
                        "long_axis_yaw_rad_mod_pi": np.pi / 2,
                    }
                },
            )

    def test_code_policy_acquire_rotates_safe_offset_with_pad_yaw(self) -> None:
        reference_path = (
            Path(__file__).parents[1]
            / "baselines"
            / "pi05"
            / "live"
            / "observation_reference_180dev_v1.json"
        )
        reference = load_hybrid_reference(reference_path)
        observed_center = np.asarray([1.7498657282, 1.9606426936, 0.85])
        yaw_delta = np.deg2rad(4.0)
        plan, audit = build_code_policy_acquire_plan(
            reference,
            {
                "pad": {
                    "center_world_m": observed_center.tolist(),
                    "long_axis_yaw_rad_mod_pi": np.pi / 2 + yaw_delta,
                },
                "target": {
                    "center_world_m": [2.1500000954, 1.9500000477, 0.75],
                    "long_axis_yaw_rad_mod_pi": np.pi / 2,
                },
            },
        )
        reference_offset = np.asarray(audit["reference_grasp_offset_xy_m"])
        expected_offset = np.asarray(
            [
                np.cos(yaw_delta) * reference_offset[0]
                - np.sin(yaw_delta) * reference_offset[1],
                np.sin(yaw_delta) * reference_offset[0]
                + np.cos(yaw_delta) * reference_offset[1],
            ]
        )
        self.assertTrue(
            np.allclose(
                np.asarray(plan[1]["pose"][:2]),
                observed_center[:2] + expected_offset,
            )
        )
        close_motion = (
            np.asarray(plan[1]["pose"][:2])
            - np.asarray(plan[0]["pose"][:2])
        )
        self.assertAlmostEqual(np.linalg.norm(close_motion), 0.020)
        self.assertGreater(float(close_motion @ expected_offset), 0.0)
        self.assertAlmostEqual(audit["yaw_delta_deg"], 4.0)
        with self.assertRaisesRegex(ValueError, "pad_yaw_retarget"):
            build_code_policy_acquire_plan(
                reference,
                {
                    "pad": {
                        "center_world_m": observed_center.tolist(),
                        "long_axis_yaw_rad_mod_pi": np.pi / 2
                        + np.deg2rad(11.0),
                    }
                },
            )

    def test_code_policy_raises_before_deep_close_and_retract(
        self,
    ) -> None:
        reference_path = (
            Path(__file__).parents[1]
            / "baselines"
            / "pi05"
            / "live"
            / "observation_reference_180dev_v1.json"
        )
        reference = load_hybrid_reference(reference_path)
        measured = tuple(reference["right_observation_ee_world_xyzw"])
        plan, audit = build_code_policy_acquire_plan(
            reference,
            {
                "pad": {
                    "center_world_m": [1.7498657282, 1.9606426936, 0.85],
                    "long_axis_yaw_rad_mod_pi": np.pi / 2,
                },
                "target": {
                    "center_world_m": [2.1500000954, 1.9500000477, 0.75],
                    "long_axis_yaw_rad_mod_pi": np.pi / 2,
                },
            },
            measured_pregrasp_pose=measured,
        )
        self.assertEqual(plan[0]["name"], "code_safe_preinsert")
        self.assertEqual(plan[0]["right_open"], 1.0)
        self.assertEqual(plan[1]["name"], "code_grasp_retract")
        self.assertEqual(plan[1]["right_open"], 0.0)
        self.assertEqual(
            plan[1]["maximum_right_gripper_open_fraction"], 0.005
        )
        self.assertAlmostEqual(plan[1]["position_tolerance_m"], 0.004)
        self.assertGreater(plan[0]["pose"][2], plan[1]["pose"][2])
        self.assertGreater(
            audit["planned_pregrasp_to_latch_distance_m"], 0.005
        )
        self.assertLess(
            audit["planned_pregrasp_to_latch_distance_m"], 0.015
        )
        self.assertEqual(
            audit["acquisition_mode"],
            "rgbd_bounded_pregrasp_refinement_then_close_retract",
        )
        self.assertAlmostEqual(audit["pregrasp_forward_refinement_m"], 0.004)
        self.assertAlmostEqual(
            audit["pregrasp_cross_axis_refinement_m"], 0.0, places=4
        )
        self.assertLessEqual(audit["pregrasp_forward_refinement_m"], 0.008)
        self.assertAlmostEqual(audit["safe_preinsert_z_offset_m"], 0.010)
        measured_xy = np.asarray(measured[:2])
        preinsert_xy = np.asarray(plan[0]["pose"][:2])
        self.assertAlmostEqual(
            np.linalg.norm(preinsert_xy - measured_xy), 0.004
        )
        self.assertLess(audit["close_retract_from_preinsert_m"], 0.015)
        self.assertAlmostEqual(
            plan[0]["position_tolerance_m"], 0.004
        )

    def test_code_policy_resolves_pad_cross_axis_before_closing(self) -> None:
        reference_path = (
            Path(__file__).parents[1]
            / "baselines"
            / "pi05"
            / "live"
            / "observation_reference_180dev_v1.json"
        )
        reference = load_hybrid_reference(reference_path)
        measured = tuple(reference["right_observation_ee_world_xyzw"])
        nominal_center = np.asarray([1.7498657282, 1.9606426936, 0.85])
        shifted_center = nominal_center + np.asarray([-0.010, 0.0, 0.0])
        plan, audit = build_code_policy_acquire_plan(
            reference,
            {
                "pad": {
                    "center_world_m": shifted_center.tolist(),
                    "long_axis_yaw_rad_mod_pi": np.pi / 2,
                },
                "target": {
                    "center_world_m": [2.1500000954, 1.9500000477, 0.75],
                    "long_axis_yaw_rad_mod_pi": np.pi / 2,
                },
            },
            measured_pregrasp_pose=measured,
        )
        preinsert_motion = (
            np.asarray(plan[0]["pose"][:2]) - np.asarray(measured[:2])
        )
        close_motion = (
            np.asarray(plan[1]["pose"][:2])
            - np.asarray(plan[0]["pose"][:2])
        )
        close_axis = np.asarray(audit["close_approach_unit_xy"])
        close_cross_axis = np.asarray((-close_axis[1], close_axis[0]))
        self.assertAlmostEqual(
            float(preinsert_motion @ close_cross_axis), -0.010, places=4
        )
        self.assertLess(
            abs(float(close_motion @ close_cross_axis)), 0.001
        )

    def test_target_rgbd_retarget_is_bounded_and_preserves_source_stages(
        self,
    ) -> None:
        reference_path = (
            Path(__file__).parents[1]
            / "baselines"
            / "pi05"
            / "live"
            / "observation_reference_180dev_v1.json"
        )
        reference = load_hybrid_reference(reference_path)
        plan = build_transport_plan(
            reference, tuple(reference["right_observation_ee_world_xyzw"])
        )
        target = {
            "center_world_m": [2.1700000954, 1.9400000477, 0.752],
            "long_axis_yaw_rad_mod_pi": np.pi / 2 + np.deg2rad(4.0),
        }
        retargeted, audit = retarget_transport_plan_to_target(plan, target)
        self.assertEqual(retargeted[1]["pose"], plan[1]["pose"])
        stage_by_name = {stage["name"]: stage for stage in plan}
        retargeted_by_name = {stage["name"]: stage for stage in retargeted}
        self.assertAlmostEqual(
            retargeted_by_name["transfer"]["pose"][0]
            - stage_by_name["transfer"]["pose"][0],
            0.01,
        )
        self.assertAlmostEqual(
            retargeted_by_name["target_overhead"]["pose"][0]
            - stage_by_name["target_overhead"]["pose"][0],
            0.02,
        )
        self.assertAlmostEqual(
            retargeted_by_name["target_overhead"]["pose"][1]
            - stage_by_name["target_overhead"]["pose"][1],
            -0.01,
        )
        self.assertAlmostEqual(audit["yaw_delta_deg"], 4.0)
        for slot_x in (1.95, 2.05, 2.15, 2.25):
            _, slot_audit = retarget_transport_plan_to_target(
                plan,
                {
                    "center_world_m": [slot_x, 1.95, 0.75],
                    "long_axis_yaw_rad_mod_pi": np.pi / 2,
                },
            )
            self.assertLessEqual(
                slot_audit["translation_xy_norm_m"], 0.201
            )
        far_target = {
            "center_world_m": [2.25, 1.95, 0.75],
            "long_axis_yaw_rad_mod_pi": np.pi / 2,
        }
        far_plan, _ = retarget_transport_plan_to_target(plan, far_target)
        far_collapsed = collapse_code_policy_transport(far_plan)
        self.assertAlmostEqual(
            next(
                stage for stage in far_collapsed
                if stage["name"] == "smooth_extract_to_place"
            )["position_tolerance_m"],
            0.020,
        )
        with self.assertRaisesRegex(ValueError, "target_xy_retarget"):
            retarget_transport_plan_to_target(
                plan,
                {
                    "center_world_m": [2.40, 1.95, 0.75],
                    "long_axis_yaw_rad_mod_pi": np.pi / 2,
                },
            )

    def test_grasp_relative_place_uses_observed_pad_to_ee_transform(self) -> None:
        reference_path = (
            Path(__file__).parents[1]
            / "baselines"
            / "pi05"
            / "live"
            / "observation_reference_180dev_v1.json"
        )
        reference = load_hybrid_reference(reference_path)
        initial = tuple(reference["right_observation_ee_world_xyzw"])
        target = {
            "center_world_m": [2.1504424941, 1.9501758865, 0.7481298505],
            "long_axis_yaw_rad_mod_pi": 1.5654055608,
        }
        plan, _ = retarget_transport_plan_to_target(
            build_transport_plan(reference, initial), target
        )
        corrected, audit = retarget_place_from_observed_grasp(
            plan,
            (
                2.1438260078,
                2.0219783783,
                0.9970651269,
                -0.0307021366,
                0.7365800181,
                -0.6754349506,
                -0.0171721636,
            ),
            {
                "center_world_m": [2.1441008906, 1.8592950909, 0.9464061688],
                "long_axis_yaw_rad_mod_pi": 1.5494056925,
            },
            target,
        )
        by_name = {stage["name"]: stage for stage in plan}
        corrected_by_name = {stage["name"]: stage for stage in corrected}
        correction = np.asarray(
            audit["place_translation_correction_world_m"]
        )
        self.assertAlmostEqual(np.linalg.norm(correction), 0.0)
        self.assertGreater(
            np.linalg.norm(
                np.asarray(
                    audit["rigid_pad_offset_xy_correction_ignored_m"]
                )
            ),
            0.04,
        )
        self.assertTrue(
            np.allclose(
                np.asarray(corrected_by_name["support_place"]["pose"][:3]),
                np.asarray(by_name["support_place"]["pose"][:3]),
            )
        )
        self.assertEqual(
            corrected_by_name["transfer"]["pose"],
            by_name["transfer"]["pose"],
        )
        for corrected_pose, original_pose in zip(
            corrected_by_name["retreat"]["right_pose_path"],
            by_name["retreat"]["right_pose_path"],
            strict=True,
        ):
            self.assertTrue(
                np.allclose(
                    np.asarray(corrected_pose[:3]),
                    np.asarray(original_pose[:3]),
                )
            )

    def test_rgbd_optical_projection_recovers_world_surface(self) -> None:
        depth = np.full((12, 16), 2.0, dtype=np.float32)
        mask = np.ones_like(depth, dtype=bool)
        points = deproject_masked_depth(
            depth,
            mask,
            (100.0, 100.0, 7.5, 5.5),
            (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            minimum_depth_m=0.1,
            maximum_depth_m=3.0,
            stride=1,
        )
        signature = robust_world_surface_signature(points)
        self.assertTrue(
            np.allclose(signature["center_world_m"], [1.0, 2.0, 5.0])
        )
        self.assertTrue(
            np.allclose(
                signature["visible_median_world_m"], [1.0, 2.0, 5.0]
            )
        )
        self.assertEqual(signature["point_count"], 192)

    def test_blue_pad_depth_signature_uses_only_blue_finite_pixels(self) -> None:
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        depth = np.full((20, 30), 0.8, dtype=np.float32)
        rgb[4:16, 5:25] = (20, 90, 190)
        depth[4:16, 5:25] = 0.31
        signature = blue_pad_depth_signature(rgb, depth)
        self.assertEqual(signature["pixel_count"], 240)
        self.assertAlmostEqual(signature["median_depth_m"], 0.31, places=5)
        self.assertAlmostEqual(signature["centroid_u_fraction"], 14.5 / 30)
        self.assertAlmostEqual(signature["centroid_v_fraction"], 9.5 / 20)

    def test_grasp_probe_rejects_depth_drift_without_area_support(self) -> None:
        pre = {
            "median_depth_m": 0.160,
            "pixel_count": 2000,
            "depth_stamp_s": 10.0,
        }
        retained = {
            "median_depth_m": 0.171,
            "pixel_count": 3000,
            "depth_stamp_s": 11.0,
        }
        unstable = {
            "median_depth_m": 0.181,
            "pixel_count": 2140,
            "depth_stamp_s": 11.0,
        }
        self.assertTrue(grasp_probe_metrics(pre, retained)["passed"])
        self.assertFalse(grasp_probe_metrics(pre, unstable)["passed"])

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
                "short_vertical_decontact",
                "short_diagonal_clearance",
                "forward_rising_extract",
                "retained_lift",
                "forward_clear_base",
                "transfer",
                "target_overhead",
                "support_contact",
                "support_precontact",
                "support_place",
                "release",
                "retreat",
                "reset_clear_view",
            ],
        )
        release_index = next(
            index for index, stage in enumerate(plan)
            if stage["name"] == "release"
        )
        self.assertTrue(all(
            stage["right_open"] == 0.0 for stage in plan[:release_index]
        ))
        self.assertTrue(all(
            stage["right_open"] == 1.0 for stage in plan[release_index:]
        ))
        self.assertEqual(plan[1]["pose"][3:], initial[3:])
        self.assertEqual(plan[1]["pose"][:2], initial[:2])
        self.assertAlmostEqual(plan[1]["pose"][2] - initial[2], 0.015)
        self.assertAlmostEqual(plan[1]["minimum_duration_s"], 0.30)
        self.assertTrue(all(
            stage.get("continuous_transit", False)
            for stage in plan[1:6]
        ))
        self.assertEqual(plan[2]["pose"][3:], initial[3:])
        base_yaw = float(reference["base_xyyaw"][2])
        self.assertAlmostEqual(
            plan[2]["pose"][0] - initial[0],
            0.0244850136 * np.cos(base_yaw),
        )
        self.assertAlmostEqual(
            plan[2]["pose"][1] - initial[1],
            0.0244850136 * np.sin(base_yaw),
        )
        self.assertAlmostEqual(plan[2]["pose"][2] - initial[2], 0.0315119326)
        self.assertEqual(plan[3]["pose"][3:], initial[3:])
        self.assertAlmostEqual(plan[3]["pose"][2] - initial[2], 0.0462560654)
        self.assertAlmostEqual(plan[4]["pose"][2] - initial[2], 0.0687306821)
        self.assertAlmostEqual(plan[5]["pose"][2] - initial[2], 0.0958608985)
        self.assertAlmostEqual(
            np.hypot(
                plan[5]["pose"][0] - initial[0],
                plan[5]["pose"][1] - initial[1],
            ),
            0.1083290609,
        )
        self.assertGreater(
            np.hypot(
                plan[3]["pose"][0] - plan[2]["pose"][0],
                plan[3]["pose"][1] - plan[2]["pose"][1],
            ),
            plan[3]["pose"][2] - plan[2]["pose"][2],
        )
        stage_by_name = {stage["name"]: stage for stage in plan}
        forward = np.asarray((np.cos(base_yaw), np.sin(base_yaw)))
        left = np.asarray((-np.sin(base_yaw), np.cos(base_yaw)))
        transfer_stage = stage_by_name["transfer"]
        start_tangent = np.asarray(
            transfer_stage["right_bezier_start_tangent_world_m"][:2]
        )
        end_tangent = np.asarray(
            transfer_stage["right_bezier_end_tangent_world_m"][:2]
        )
        self.assertAlmostEqual(float(start_tangent @ forward), 0.030)
        self.assertAlmostEqual(float(start_tangent @ left), 0.0)
        self.assertAlmostEqual(float(end_tangent @ left), 0.040)
        curve_start = stage_by_name["forward_clear_base"]["pose"]
        curve_early = cubic_bezier_position(
            curve_start,
            transfer_stage["pose"],
            transfer_stage["right_bezier_start_tangent_world_m"],
            transfer_stage["right_bezier_end_tangent_world_m"],
            0.05,
        )
        early_delta = np.asarray(curve_early[:2]) - np.asarray(curve_start[:2])
        self.assertGreater(float(early_delta @ forward), 0.0)
        self.assertEqual(
            cubic_bezier_position(
                curve_start,
                transfer_stage["pose"],
                transfer_stage["right_bezier_start_tangent_world_m"],
                transfer_stage["right_bezier_end_tangent_world_m"],
                1.0,
            ),
            transfer_stage["pose"][:3],
        )
        self.assertGreater(
            stage_by_name["transfer"]["pose"][0]
            - stage_by_name["forward_clear_base"]["pose"][0],
            0.15,
        )
        self.assertGreaterEqual(
            stage_by_name["transfer"]["pose"][2],
            stage_by_name["forward_clear_base"]["pose"][2],
        )
        self.assertEqual(
            stage_by_name["target_overhead"]["pose"][3:],
            stage_by_name["transfer"]["pose"][3:],
        )
        self.assertEqual(
            stage_by_name["target_overhead"]["pose"][:3],
            (
                *stage_by_name["support_contact"]["pose"][:2],
                stage_by_name["support_contact"]["pose"][2] + 0.08,
            ),
        )
        self.assertEqual(
            stage_by_name["support_place"]["pose"][3:],
            tuple(reference["right_hybrid_landmarks_world_xyzw"]["place"])[3:],
        )
        self.assertNotEqual(
            stage_by_name["support_contact"]["pose"][3:],
            stage_by_name["support_place"]["pose"][3:],
        )
        self.assertEqual(
            stage_by_name["retreat"]["pose"][3:],
            stage_by_name["release"]["pose"][3:],
        )
        release_origin = np.asarray(
            reference["right_hybrid_landmarks_world_xyzw"]["release"][:3]
        )
        release_delta = np.asarray(
            stage_by_name["release"]["pose"][:3]
        ) - release_origin
        self.assertAlmostEqual(
            float(release_delta[:2] @ forward), -0.0024928125
        )
        self.assertAlmostEqual(float(release_delta[2]), 0.0060508251)
        retreat_delta = np.asarray(
            stage_by_name["retreat"]["pose"][:3]
        ) - release_origin
        self.assertAlmostEqual(float(retreat_delta[:2] @ forward), 0.0075160458)
        self.assertAlmostEqual(float(retreat_delta[:2] @ left), -0.0491492257)
        self.assertAlmostEqual(float(retreat_delta[2]), 0.0505843759)
        self.assertEqual(len(stage_by_name["retreat"]["right_pose_path"]), 3)
        release_open_delta = np.asarray(
            stage_by_name["retreat"]["right_pose_path"][0][:3]
        ) - release_origin
        self.assertAlmostEqual(
            float(release_open_delta[:2] @ forward), -0.0004594730
        )
        self.assertAlmostEqual(
            float(release_open_delta[:2] @ left), -0.0042378856
        )
        self.assertAlmostEqual(float(release_open_delta[2]), 0.0233875215)
        self.assertAlmostEqual(
            stage_by_name["release"]["minimum_duration_s"], 10.0 / 30.0
        )
        self.assertEqual(
            stage_by_name["release"]["minimum_right_gripper_open_fraction"],
            0.995,
        )
        self.assertEqual(
            stage_by_name["retreat"]["minimum_right_gripper_open_fraction"],
            0.995,
        )
        self.assertTrue(stage_by_name["release"]["continuous_transit"])
        self.assertTrue(stage_by_name["retreat"]["continuous_transit"])
        self.assertEqual(
            stage_by_name["reset_clear_view"]["pose"],
            tuple(reference["right_clearance_waypoint_ee_world_xyzw"]),
        )
        self.assertEqual(
            stage_by_name["release"]["minimum_right_ee_z_m"],
            0.903,
        )

    def test_pad_retention_tracks_rigid_gripper_motion(self) -> None:
        reference_ee = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
        reference_pad = (1.0, 1.8, 2.95)
        current_ee = (1.4, 2.1, 3.2, 0.0, 0.0, 0.0, 1.0)
        retained = pad_retention_metrics(
            reference_pad,
            reference_ee,
            (1.4, 1.9, 3.15),
            current_ee,
        )
        dropped = pad_retention_metrics(
            reference_pad,
            reference_ee,
            (1.30, 1.80, 2.95),
            current_ee,
        )
        self.assertTrue(retained["passed"])
        self.assertAlmostEqual(retained["world_error_m"], 0.0)
        self.assertFalse(dropped["passed"])
        strict = pad_retention_metrics(
            reference_pad,
            reference_ee,
            (1.42, 1.9, 3.15),
            current_ee,
            maximum_world_error_m=0.015,
        )
        self.assertFalse(strict["passed"])
        self.assertEqual(strict["maximum_world_error_m"], 0.015)

    def test_rmpflow_plan_ends_at_the_only_policy_observation_pose(self) -> None:
        self.assertEqual(
            [target_kind for target_kind, _ in RMPFLOW_STAGE_PLAN],
            ["continuous_observation"],
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
