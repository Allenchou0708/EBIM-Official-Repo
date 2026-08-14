# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light tests for the Task 2 PI0.5 live safety boundary."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np

from task2_isaacsim.baselines.pi05.contract import (
    RELATIVE_ACTION_STATE_INDICES,
    STATE_NAMES,
    V2_RELATIVE_ACTION_STATE_INDICES,
)
from task2_isaacsim.baselines.pi05.live.core import (
    BaseReadinessGate,
    FreshnessConfig,
    FreshnessError,
    ReadinessConfig,
    RunnerPhase,
    align_action_chunk,
    freshness_metrics,
    hard5_action_window,
    hard5_hold_action,
    policy_command_topics,
    replace_action_queue,
    safe_action,
    startup_inventory,
    validate_rgb_frame,
)
from task2_isaacsim.baselines.pi05.live.policy import LivePi05Policy
from task2_isaacsim.baselines.pi05.live.staging import (
    interpolate_staging_command,
    staging_entry_duration_s,
    staging_feedback,
    validate_staging_audit,
)
from task2_isaacsim.baselines.pi05.verify_shadow_run import verify_shadow_run
from task2_isaacsim.common.state_contract import (
    LEFT_GRIPPER_DRIVER,
    LEFT_JOINTS,
    RIGHT_GRIPPER_DRIVER,
    RIGHT_JOINTS,
    SPINE_JOINT,
    assemble_state,
)


def valid_action() -> list[float]:
    return (
        [0.1, -0.1, 0.2]
        + [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0] * 2
        + [0.5, 0.6, 0.3]
    )


class StateContractTest(unittest.TestCase):
    def test_assembler_matches_every_official_index(self) -> None:
        joints = {
            name: 0.01 * index
            for index, name in enumerate((*LEFT_JOINTS, *RIGHT_JOINTS), 1)
        }
        joints[SPINE_JOINT] = 0.42
        joints[LEFT_GRIPPER_DRIVER] = 0.0
        joints[RIGHT_GRIPPER_DRIVER] = 0.8
        state = assemble_state(
            {
                "ee_poses": {
                    "left": (1, 2, 3, 0, 0, 0, 1),
                    "right": (4, 5, 6, 0, 0, 1, 0),
                },
                "joint_states": joints,
                "odom": (7, 8, 0, 0, 0, 0, 1, 0.1, 0.2, 0, 0.3),
            }
        )
        self.assertEqual(len(state), len(STATE_NAMES))
        self.assertEqual(
            state[:14], (1, 2, 3, 0, 0, 0, 1, 4, 5, 6, 0, 0, 1, 0)
        )
        self.assertEqual(state[14:28], tuple(0.01 * i for i in range(1, 15)))
        self.assertEqual(state[28:31], (0.42, 1.0, 0.0))
        self.assertEqual(state[31:37], (7.0, 8.0, 0.0, 0.1, 0.2, 0.3))

    def test_aliases_match_recorder_contract(self) -> None:
        state = assemble_state(
            {
                "ee_poses": {"left": (0,) * 7, "right": (0,) * 7},
                "joint_states": {
                    **{
                        name.replace("fr3v2_joint", "fr3v2_1_joint"): 0.0
                        for name in (*LEFT_JOINTS, *RIGHT_JOINTS)
                    },
                    "left_fr3v2_finger_joint1": 0.4,
                    "right_fr3v2_finger_joint1": 0.4,
                    SPINE_JOINT: 0.3,
                },
                "odom": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
            }
        )
        self.assertTrue(all(math.isfinite(value) for value in state))
        self.assertEqual(state[29:31], (0.5, 0.5))


class LiveSafetyTest(unittest.TestCase):
    def _staging_audit(self) -> dict:
        left = [0.0, -0.7, 0.1, -2.3, 0.0, 1.6, 0.8]
        right = [0.8, -1.6, -1.8, -2.4, -0.7, 3.8, -0.5]
        left_ee = [2.45, 2.2, 0.91, 0.0, 0.0, 0.0, 1.0]
        right_ee = [1.75, 2.14, 0.87, 0.0, 0.7071, -0.7071, 0.0]
        return {
            "guessed_ik_used": False,
            "selection": {"episode": 176, "frame": 408},
            "final_target": {
                "left_arm_rad": left,
                "right_arm_rad": right,
                "left_gripper_open_fraction": 1.0,
                "right_gripper_open_fraction": 1.0,
                "spine_command_m": 0.5,
                "measured_reference": {
                    "left_arm_rad": left,
                    "right_arm_rad": right,
                    "spine_m": 0.486,
                    "left_gripper_open_fraction": 1.0,
                    "right_gripper_open_fraction": 1.0,
                    "left_ee": left_ee,
                    "right_ee": right_ee,
                },
            },
            "tolerances": {
                "arm_max_abs_rad": 0.04,
                "spine_abs_m": 0.02,
                "gripper_open_fraction": 0.05,
                "left_ee_z_m": 0.04,
                "right_ee_position_m": 0.04,
                "right_ee_orientation_deg": 12.0,
                "stable_dwell_sim_s": 1.0,
            },
            "velocity_risk": {
                "arm_limits_rad_s": [2.62] * 14,
                "staging_arm_velocity_fraction": 0.5,
                "staging_spine_velocity_m_s": 0.05,
                "staging_gripper_velocity_fraction_s": 1.0,
            },
            "trajectory": [
                {
                    "frame": 0,
                    "scheduled_at_s": 0.0,
                    "command": [*left, *right, 1.0, 1.0, 0.0],
                },
                {
                    "frame": 408,
                    "scheduled_at_s": 10.0,
                    "command": [*left, *right, 1.0, 1.0, 0.5],
                },
            ],
        }

    def test_dataset_staging_requires_provenance_and_no_ik(self) -> None:
        audit = self._staging_audit()
        self.assertIs(validate_staging_audit(audit), audit)
        audit["guessed_ik_used"] = True
        with self.assertRaisesRegex(ValueError, "guessed IK"):
            validate_staging_audit(audit)

    def test_dataset_staging_rejects_unsafe_intermediate_command(self) -> None:
        audit = self._staging_audit()
        audit["trajectory"][0]["command"][0] = 99.0
        with self.assertRaisesRegex(ValueError, "outside FR3 bounds"):
            validate_staging_audit(audit)

    def test_staging_entry_uses_measured_state_and_sim_time_limits(self) -> None:
        audit = self._staging_audit()
        target = tuple(audit["trajectory"][0]["command"])
        current = (*target[:14], *target[14:16], 0.5)
        duration = staging_entry_duration_s(audit, current, target)
        self.assertEqual(duration, 10.0)
        midpoint = interpolate_staging_command(current, target, 0.5)
        self.assertAlmostEqual(midpoint[16], 0.25)

    def test_dataset_staging_feedback_covers_every_required_group(self) -> None:
        audit = self._staging_audit()
        reference = audit["final_target"]["measured_reference"]
        feedback = staging_feedback(
            audit=audit,
            left_arm=tuple(reference["left_arm_rad"]),
            right_arm=tuple(reference["right_arm_rad"]),
            spine_m=reference["spine_m"],
            left_gripper_open=1.0,
            right_gripper_open=1.0,
            left_ee=tuple(reference["left_ee"]),
            right_ee=tuple(reference["right_ee"]),
        )
        self.assertTrue(feedback["within_tolerance"])
        self.assertTrue(all(feedback["groups"].values()))
        failed = staging_feedback(
            audit=audit,
            left_arm=tuple(reference["left_arm_rad"]),
            right_arm=tuple(reference["right_arm_rad"]),
            spine_m=0.40,
            left_gripper_open=1.0,
            right_gripper_open=1.0,
            left_ee=tuple(reference["left_ee"]),
            right_ee=tuple(reference["right_ee"]),
        )
        self.assertFalse(failed["groups"]["spine"])
        self.assertFalse(failed["within_tolerance"])

    def test_hard5_two_decisions_execute_zero_to_four_and_hold_only(
        self,
    ) -> None:
        actions = [safe_action(valid_action())[1] for _ in range(50)]
        last_action = None
        for decision in range(2):
            window = hard5_action_window(
                actions,
                ready_at=100.0 + decision,
                action_rate_hz=30.0,
            )
            self.assertEqual([item[2] for item in window], list(range(5)))
            queue = deque(
                (action, target, 100.0 + decision, decision, index)
                for action, target, index in window
            )
            while queue:
                last_action = queue.popleft()[0]
            self.assertEqual(
                hard5_hold_action(
                    last_action,
                    inference_pending=True,
                    queue=queue,
                ),
                last_action,
            )
            self.assertIsNone(
                hard5_hold_action(
                    last_action,
                    inference_pending=False,
                    queue=queue,
                )
            )

    def test_shadow_gate_distinguishes_v1_and_v2_spine_contracts(self) -> None:
        groups = {
            name: True
            for name in (
                "left_arm",
                "right_arm",
                "spine",
                "left_gripper",
                "right_gripper",
                "left_ee_height",
                "right_pregrasp_position",
                "right_pregrasp_orientation",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "settled_fresh_wrist_right.ppm").write_bytes(
                b"P6\n2 2\n255\n" + b"\0" * 24
            )
            manifest = {
                "completed": True,
                "valid_decisions": 1,
                "command_publications": 0,
                "ros_publication": False,
                "staging": {
                    "result": {
                        "feedback": {
                            "within_tolerance": True,
                            "groups": groups,
                        }
                    }
                },
                "events": [
                    {
                        "valid": True,
                        "policy_indices": [0, 1, 2, 3, 4],
                        "capture_to_ready_sim_s": 0.5,
                        "freshness": {"observation_capture_skew_s": 0.02},
                    }
                ],
            }
            for contract, mapping in (
                ("v1", RELATIVE_ACTION_STATE_INDICES),
                ("v2", V2_RELATIVE_ACTION_STATE_INDICES),
            ):
                with self.subTest(contract=contract):
                    manifest[
                        "checkpoint_relative_action_state_indices"
                    ] = list(mapping)
                    (root / "live_runner_manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    self.assertTrue(
                        verify_shadow_run(root, contract=contract)["valid"]
                    )
                    wrong = "v2" if contract == "v1" else "v1"
                    self.assertFalse(
                        verify_shadow_run(root, contract=wrong)["valid"]
                    )

    def test_startup_inventory_names_missing_real_inputs(self) -> None:
        state = [0.0] * 37
        state[7] = math.nan
        status = startup_inventory(
            camera_sequences={"head": 1, "wrist_left": 1},
            joint_names={*LEFT_JOINTS, *RIGHT_JOINTS},
            ee_available={"left": True, "right": False},
            odom_available=False,
            state=state,
        )
        self.assertFalse(status["all_required_samples"])
        self.assertEqual(
            status["missing_inputs"],
            [
                "wrist_right",
                "right_ee_pose",
                "odom",
                "joint_states",
                "finite_37d_state",
            ],
        )
        self.assertEqual(status["missing_joints"], [SPINE_JOINT])
        self.assertEqual(status["invalid_state_indices"], [7])

    def test_refill_replaces_residual_with_fresh_complete_chunk(self) -> None:
        queue = deque(
            ((float(index),), 9.0, 8.5, 0, index) for index in range(24)
        )
        fresh_chunk = [
            ((100.0 + index,), 12.5 + index / 30.0, index)
            for index in range(18, 50)
        ]

        residual = replace_action_queue(
            queue,
            fresh_chunk,
            completed_at=12.5,
            event_index=1,
        )

        self.assertEqual(residual, 24)
        self.assertEqual(len(queue), 32)
        self.assertEqual(
            list(queue),
            [
                (action, target_at, 12.5, 1, chunk_index)
                for action, target_at, chunk_index in fresh_chunk
            ],
        )

    def test_time_alignment_discards_elapsed_prefix_in_monotonic_time(
        self,
    ) -> None:
        effective_actions = [safe_action(valid_action())[1] for _ in range(50)]
        for latency_s, expected_discard in ((0.5, 15), (0.65, 20)):
            with self.subTest(latency_s=latency_s):
                discarded, aligned = align_action_chunk(
                    effective_actions,
                    capture_at=100.0,
                    ready_at=100.0 + latency_s,
                    action_rate_hz=30.0,
                )
                self.assertEqual(discarded, expected_discard)
                self.assertEqual(aligned[0][2], expected_discard)
                self.assertEqual(aligned[-1][2], 49)
                self.assertTrue(
                    all(
                        right[1] > left[1]
                        for left, right in zip(
                            aligned, aligned[1:], strict=False
                        )
                    )
                )
                self.assertTrue(
                    all(
                        action[:3] == (0.0, 0.0, 0.0)
                        for action, _, _ in aligned
                    )
                )

    def test_first_chunk_starts_at_zero_when_robot_was_idle(self) -> None:
        effective_actions = [safe_action(valid_action())[1] for _ in range(50)]
        discarded, aligned = align_action_chunk(
            effective_actions,
            capture_at=100.0,
            ready_at=100.65,
            action_rate_hz=30.0,
            execution_started=False,
        )

        self.assertEqual(discarded, 0)
        self.assertEqual([item[2] for item in aligned], list(range(50)))
        self.assertAlmostEqual(aligned[0][1], 100.65)
        self.assertAlmostEqual(aligned[1][1], 100.65 + 1.0 / 30.0)

    def test_policy_reset_clears_action_queue_seed_index(self) -> None:
        class FakePolicy:
            reset_called = False

            def reset(self) -> None:
                self.reset_called = True

        live_policy = LivePi05Policy.__new__(LivePi05Policy)
        live_policy.policy = FakePolicy()
        live_policy.decision_index = 8
        live_policy.reset()
        self.assertTrue(live_policy.policy.reset_called)
        self.assertEqual(live_policy.decision_index, 0)

    def test_camera_shapes_and_eval_rejection(self) -> None:
        validate_rgb_frame("head", np.zeros((720, 1280, 3), dtype=np.uint8))
        validate_rgb_frame(
            "wrist_left", np.zeros((480, 848, 3), dtype=np.uint8)
        )
        with self.assertRaisesRegex(ValueError, "unsupported policy camera"):
            validate_rgb_frame(
                "eval_camera", np.zeros((720, 1280, 3), dtype=np.uint8)
            )
        with self.assertRaisesRegex(ValueError, "image shape"):
            validate_rgb_frame("head", np.zeros((480, 848, 3), dtype=np.uint8))

    def test_freshness_evidence_identifies_stale_stream_and_recovers(
        self,
    ) -> None:
        result = freshness_metrics(
            now=10.0,
            camera_times={
                "head": 9.95,
                "wrist_left": 9.96,
                "wrist_right": 9.97,
            },
            camera_sequences={"head": 2, "wrist_left": 3, "wrist_right": 4},
            camera_capture_times={
                "head": 20.00,
                "wrist_left": 20.02,
                "wrist_right": 20.01,
            },
            state_time=9.98,
            last_camera_sequences={
                "head": 1,
                "wrist_left": 2,
                "wrist_right": 3,
            },
            config=FreshnessConfig(),
        )
        self.assertAlmostEqual(result["inter_camera_skew_s"], 0.02)
        self.assertAlmostEqual(result["inter_camera_arrival_skew_s"], 0.02)
        delayed_transport = freshness_metrics(
            now=10.0,
            camera_times={
                "head": 9.45,
                "wrist_left": 9.96,
                "wrist_right": 9.97,
            },
            camera_capture_times={
                "head": 30.00,
                "wrist_left": 30.00,
                "wrist_right": 30.02,
            },
            camera_sequences={"head": 2, "wrist_left": 3, "wrist_right": 4},
            state_time=9.98,
            last_camera_sequences={
                "head": 1,
                "wrist_left": 2,
                "wrist_right": 3,
            },
            config=FreshnessConfig(),
        )
        self.assertAlmostEqual(
            delayed_transport["inter_camera_capture_skew_s"], 0.02
        )
        self.assertAlmostEqual(
            delayed_transport["inter_camera_arrival_skew_s"], 0.52
        )
        with self.assertRaises(FreshnessError) as caught:
            freshness_metrics(
                now=10.0,
                camera_times={
                    "head": 9.60,
                    "wrist_left": 9.96,
                    "wrist_right": 9.97,
                },
                camera_sequences={
                    "head": 1,
                    "wrist_left": 3,
                    "wrist_right": 4,
                },
                state_time=9.80,
                last_camera_sequences={"head": 1},
                config=FreshnessConfig(
                    camera_max_age_s=0.25,
                    camera_max_skew_s=0.10,
                    state_max_age_s=0.10,
                ),
            )
        evidence = caught.exception.evidence
        self.assertEqual(
            evidence["offending_streams"],
            ["head", "camera_skew", "state"],
        )
        self.assertAlmostEqual(evidence["frame_age_s"]["head"], 0.40)
        self.assertAlmostEqual(evidence["state_age_s"], 0.20)

        recovered = freshness_metrics(
            now=10.1,
            camera_times={
                "head": 10.05,
                "wrist_left": 10.06,
                "wrist_right": 10.07,
            },
            camera_sequences={"head": 3, "wrist_left": 4, "wrist_right": 5},
            state_time=10.08,
            last_camera_sequences={
                "head": 2,
                "wrist_left": 3,
                "wrist_right": 4,
            },
            config=FreshnessConfig(),
        )
        self.assertAlmostEqual(recovered["state_age_s"], 0.02)

    def test_freshness_rejects_camera_state_capture_misalignment(self) -> None:
        common = {
            "now": 10.0,
            "capture_now": 20.04,
            "camera_times": {
                "head": 9.95,
                "wrist_left": 9.96,
                "wrist_right": 9.97,
            },
            "camera_sequences": {
                "head": 2,
                "wrist_left": 2,
                "wrist_right": 2,
            },
            "camera_capture_times": {
                "head": 20.00,
                "wrist_left": 20.02,
                "wrist_right": 20.01,
            },
            "state_time": 9.98,
            "state_times": {
                "joints": 9.98,
                "odom": 9.98,
                "ee_left": 9.98,
                "ee_right": 9.98,
            },
            "last_camera_sequences": {
                "head": 1,
                "wrist_left": 1,
                "wrist_right": 1,
            },
            "config": FreshnessConfig(observation_max_skew_s=0.10),
        }
        aligned = freshness_metrics(
            **common,
            state_capture_times={
                "joints": 20.02,
                "odom": 20.02,
                "ee_left": 20.02,
                "ee_right": 20.02,
            },
        )
        self.assertAlmostEqual(aligned["observation_capture_skew_s"], 0.02)
        self.assertAlmostEqual(aligned["state_age_s"], 0.02)

        with self.assertRaises(FreshnessError) as caught:
            freshness_metrics(
                **common,
                state_capture_times={
                    "joints": 20.20,
                    "odom": 20.20,
                    "ee_left": 20.20,
                    "ee_right": 20.20,
                },
            )
        self.assertIn(
            "observation_skew", caught.exception.evidence["offending_streams"]
        )

        with self.assertRaises(FreshnessError) as caught:
            freshness_metrics(
                **{**common, "state_times": {"joints": 9.98}},
                state_capture_times={"joints": 20.02},
            )
        self.assertIn(
            "state_streams_missing",
            caught.exception.evidence["offending_streams"],
        )

    def test_policy_spine_is_preserved_bounded_and_base_stays_fixed(
        self,
    ) -> None:
        raw, effective = safe_action(valid_action())
        self.assertEqual(raw[:3], (0.1, -0.1, 0.2))
        self.assertEqual(effective[:3], (0.0, 0.0, 0.0))
        self.assertEqual(effective[19], raw[19])
        projected = valid_action()
        projected[17] = -0.1
        projected[18] = 1.0131
        projected[19] = 0.7
        raw, effective = safe_action(projected)
        self.assertEqual(raw[17:19], (-0.1, 1.0131))
        self.assertEqual(effective[17:19], (0.0, 1.0))
        self.assertEqual(effective[19], 0.6)
        projected[19] = -0.2
        _, effective = safe_action(projected)
        self.assertEqual(effective[19], 0.0)
        projected = valid_action()
        projected[13] = -3.08662
        raw, effective = safe_action(projected)
        self.assertEqual(raw[13], -3.08662)
        self.assertEqual(effective[13], -3.0770200167)
        projected[13] = -3.14735
        raw, effective = safe_action(projected)
        self.assertEqual(raw[13], -3.14735)
        self.assertEqual(effective[13], -3.0770200167)
        projected[17] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            safe_action(projected)
        invalid_arm = valid_action()
        invalid_arm[3] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            safe_action(invalid_arm)
        invalid_spine = valid_action()
        invalid_spine[19] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            safe_action(invalid_spine)

    def test_policy_publication_adds_only_spine_and_forbids_base(
        self,
    ) -> None:
        topics = {
            "bridge": {
                "joint_groups": {
                    "left_arm": {"command": "/left_arm"},
                    "right_arm": {"command": "/right_arm"},
                    "left_gripper": {"command": "/left_gripper"},
                    "right_gripper": {"command": "/right_gripper"},
                }
            },
            "teleop": {"spine_target": "/spine"},
        }
        command_topics = policy_command_topics(topics)
        self.assertEqual(
            set(command_topics),
            {
                "left_arm",
                "right_arm",
                "left_gripper",
                "right_gripper",
                "spine",
            },
        )
        self.assertEqual(command_topics["spine"], "/spine")
        self.assertNotIn("base", command_topics)

    def test_initial_low_spine_latches_then_policy_motion_is_allowed(
        self,
    ) -> None:
        gate = BaseReadinessGate(
            ReadinessConfig(1.0, 2.0, 0.0, settle_duration_s=1.0)
        )
        gate.reset()
        self.assertFalse(
            gate.update(0.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.02)
        )
        self.assertFalse(
            gate.update(0.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        )
        self.assertTrue(
            gate.update(1.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        )
        self.assertEqual(gate.phase, RunnerPhase.MANIPULATION_READY)
        gate.arm()
        self.assertTrue(
            gate.update(2.0, (1.01, 1.99, 0.01), (0.03, 0.0, 0.0), 0.486)
        )
        self.assertEqual(gate.phase, RunnerPhase.PI05_MANIPULATION)
        gate.reset()
        self.assertEqual(gate.phase, RunnerPhase.BASE_PREPOSITION)
        gate.update(3.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        gate.update(4.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        gate.arm()
        self.assertFalse(
            gate.update(5.0, (1.2, 2.0, 0.0), (0.0, 0.0, 0.0), 0.486)
        )
        self.assertEqual(gate.phase, RunnerPhase.STOPPED)
        gate.reset()
        gate.update(6.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        gate.update(7.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        gate.arm()
        gate.note_base_input()
        self.assertEqual(gate.phase, RunnerPhase.STOPPED)

    def test_dataset_staged_spine_uses_simulator_time_dwell(self) -> None:
        gate = BaseReadinessGate(
            ReadinessConfig(
                1.0,
                2.0,
                0.0,
                initial_spine_target_m=0.486,
                initial_spine_max_abs_m=0.02,
                settle_duration_s=1.0,
            )
        )
        gate.reset()
        self.assertFalse(
            gate.update(20.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.486)
        )
        self.assertFalse(
            gate.update(20.5, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.486)
        )
        self.assertTrue(
            gate.update(21.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.486)
        )


if __name__ == "__main__":
    unittest.main()
