# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from task2_isaacsim.baselines.pi05.gripper_hold_gate import (
    LANDMARK_EXPECTED_OPEN,
    evaluate_gripper_records,
    landmark_frames,
)
from task2_isaacsim.baselines.pi05.phase_train import parse_action_loss_weights
from task2_isaacsim.baselines.pi05.v5_train import (
    ACTION_LOSS_WEIGHTS,
    RIGHT_GRIPPER_ACTION_INDEX,
    RIGHT_GRIPPER_LOSS_WEIGHT,
)


def action(gripper: float) -> list[float]:
    values = [0.0] * 20
    values[6] = -1.0
    values[13] = -1.0
    values[15] = 1.0
    values[17] = 1.0
    values[18] = gripper
    values[19] = 0.48
    return values


def passing_records() -> list[dict]:
    records = []
    for landmark, expected_open in LANDMARK_EXPECTED_OPEN.items():
        expected = 1.0 if expected_open else 0.0
        records.append(
            {
                "landmark": landmark,
                "predictions": [action(expected) for _ in range(5)],
                "reference_actions": [action(expected) for _ in range(5)],
                "effective_actions_safe": True,
            }
        )
    return records


class V5GripperTest(unittest.TestCase):
    def test_only_right_gripper_receives_extra_loss_weight(self) -> None:
        self.assertEqual(len(ACTION_LOSS_WEIGHTS), 20)
        self.assertEqual(
            ACTION_LOSS_WEIGHTS[RIGHT_GRIPPER_ACTION_INDEX],
            RIGHT_GRIPPER_LOSS_WEIGHT,
        )
        self.assertTrue(
            all(
                value == 1.0
                for index, value in enumerate(ACTION_LOSS_WEIGHTS)
                if index != RIGHT_GRIPPER_ACTION_INDEX
            )
        )
        self.assertEqual(
            parse_action_loss_weights(str(ACTION_LOSS_WEIGHTS).replace("'", '"')),
            ACTION_LOSS_WEIGHTS,
        )

    def test_action_loss_weights_reject_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "20 weights"):
            parse_action_loss_weights("[1, 2]")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_action_loss_weights("[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1]")
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_action_loss_weights(
                "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1e999,1]"
            )

    def test_landmarks_cover_close_hold_and_release(self) -> None:
        record = {
            "events": {
                "right_close": 400,
                "pad_move": 450,
                "target_arrival": 700,
                "right_release": 820,
            }
        }
        self.assertEqual(
            landmark_frames(record),
            {
                "approach_open": 395,
                "grasp_close": 400,
                "hold_early": 450,
                "hold_mid": 575,
                "hold_late": 810,
                "release_open": 820,
            },
        )

    def test_gate_requires_retention_not_only_initial_close(self) -> None:
        records = passing_records()
        self.assertTrue(evaluate_gripper_records(records)["go"])
        hold_mid = next(
            record for record in records if record["landmark"] == "hold_mid"
        )
        hold_mid["predictions"] = [action(1.0) for _ in range(5)]
        report = evaluate_gripper_records(records)
        self.assertTrue(report["checks"]["grasp_closes"])
        self.assertFalse(report["checks"]["grasp_is_retained"])
        self.assertFalse(report["go"])


if __name__ == "__main__":
    unittest.main()
