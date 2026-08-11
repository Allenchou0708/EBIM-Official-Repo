# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light tests for the Task 2 PI0.5 live safety boundary."""

from __future__ import annotations

import math
import unittest
from collections import deque

import numpy as np

from task2_isaacsim.baselines.pi05.contract import STATE_NAMES
from task2_isaacsim.baselines.pi05.live.core import (
    BaseReadinessGate,
    FreshnessConfig,
    FreshnessError,
    ReadinessConfig,
    RunnerPhase,
    freshness_metrics,
    replace_action_queue,
    safe_action,
    validate_rgb_frame,
)
from task2_isaacsim.baselines.pi05.live.policy import LivePi05Policy
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
    def test_refill_replaces_residual_with_fresh_complete_chunk(self) -> None:
        queue = deque(
            ((float(index),), 10.0, 0) for index in range(24)
        )
        fresh_chunk = [(100.0 + index,) for index in range(50)]

        residual = replace_action_queue(
            queue,
            fresh_chunk,
            completed_at=12.5,
            event_index=1,
        )

        self.assertEqual(residual, 24)
        self.assertEqual(len(queue), 50)
        self.assertEqual(
            list(queue),
            [(action, 12.5, 1) for action in fresh_chunk],
        )

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
            state_time=9.98,
            last_camera_sequences={
                "head": 1,
                "wrist_left": 2,
                "wrist_right": 3,
            },
            config=FreshnessConfig(),
        )
        self.assertAlmostEqual(result["inter_camera_skew_s"], 0.02)
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

    def test_safety_adapter_projects_grippers_and_holds_mobile_axes(
        self,
    ) -> None:
        raw, effective = safe_action(valid_action(), spine_hold=0.44)
        self.assertEqual(raw[:3], (0.1, -0.1, 0.2))
        self.assertEqual(effective[:3], (0.0, 0.0, 0.0))
        self.assertEqual(effective[19], 0.44)
        projected = valid_action()
        projected[17] = -0.1
        projected[18] = 1.0131
        raw, effective = safe_action(projected, spine_hold=0.44)
        self.assertEqual(raw[17:19], (-0.1, 1.0131))
        self.assertEqual(effective[17:19], (0.0, 1.0))
        projected[17] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            safe_action(projected, spine_hold=0.44)
        invalid_arm = valid_action()
        invalid_arm[3] = 4.0
        with self.assertRaisesRegex(ValueError, "outside"):
            safe_action(invalid_arm, spine_hold=0.44)

    def test_spine_ready_latches_through_drift_and_revokes_safely(self) -> None:
        gate = BaseReadinessGate(
            ReadinessConfig(1.0, 2.0, 0.0, settle_duration_s=1.0)
        )
        gate.reset()
        self.assertFalse(
            gate.update(0.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        )
        self.assertFalse(
            gate.update(0.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.4857)
        )
        self.assertTrue(
            gate.update(1.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.4857)
        )
        self.assertEqual(gate.phase, RunnerPhase.MANIPULATION_READY)
        gate.arm()
        self.assertTrue(
            gate.update(2.0, (1.01, 1.99, 0.01), (0.03, 0.0, 0.0), 0.486)
        )
        self.assertEqual(gate.phase, RunnerPhase.PI05_MANIPULATION)
        gate.reset()
        self.assertEqual(gate.phase, RunnerPhase.BASE_PREPOSITION)
        gate.update(3.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.4857)
        gate.update(4.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.4857)
        gate.arm()
        self.assertFalse(
            gate.update(5.0, (1.2, 2.0, 0.0), (0.0, 0.0, 0.0), 0.486)
        )
        self.assertEqual(gate.phase, RunnerPhase.STOPPED)
        gate.reset()
        gate.update(6.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.4857)
        gate.update(7.0, (1.0, 2.0, 0.0), (0.0, 0.0, 0.0), 0.4857)
        gate.arm()
        gate.note_base_input()
        self.assertEqual(gate.phase, RunnerPhase.STOPPED)


if __name__ == "__main__":
    unittest.main()
