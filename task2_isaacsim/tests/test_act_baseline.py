# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

from task2_isaacsim.baselines.act.contract import (
    ACT_CONTRACT,
    deterministic_episode_split,
    validate_action_chunk,
)
from task2_isaacsim.baselines.act.official_metric import score_result


class ACTBaselineTest(unittest.TestCase):
    def test_split_is_deterministic_disjoint_180_20(self) -> None:
        train, validation = deterministic_episode_split()
        self.assertEqual(len(train), 180)
        self.assertEqual(len(validation), 20)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(sorted(train + validation), list(range(200)))
        self.assertEqual((train, validation), deterministic_episode_split())

    def test_action_contract_is_exactly_twenty_absolute_values(self) -> None:
        chunk = validate_action_chunk([[float(index) for index in range(20)]])
        self.assertEqual(len(chunk[0]), ACT_CONTRACT.action_dim)
        with self.assertRaises(ValueError):
            validate_action_chunk([[0.0] * 19])

    def test_orientation_failure_zeroes_official_score(self) -> None:
        score = score_result(
            {
                "target_bbox": {"x": 0},
                "pad_bbox": {"x": 0},
                "is_orientation_correct": False,
                "iou_thermalpad_vs_target_current": 0.8,
            }
        )
        self.assertEqual(score["valid_placement_iou"], 0.0)

    def test_orientation_success_keeps_iou(self) -> None:
        score = score_result(
            {
                "target_bbox": {"x": 0},
                "pad_bbox": {"x": 0},
                "is_orientation_correct": True,
                "iou_thermalpad_vs_target_current": 0.8,
            }
        )
        self.assertEqual(score["valid_placement_iou"], 0.8)


if __name__ == "__main__":
    unittest.main()
