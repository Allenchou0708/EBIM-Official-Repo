# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import unittest

from task2_isaacsim.baselines.pi05.phase_conditioned_dataset import PHASE_PROMPTS
from task2_isaacsim.baselines.pi05.phase_offline_gate import (
    evaluate_phase_records,
    phase_landmark_frames,
)
from task2_isaacsim.baselines.pi05.phase_train import (
    ordered_phase_prompts,
    resolve_phase_task,
)


def action(gripper: float = 1.0) -> list[float]:
    values = [0.0] * 20
    values[6] = -1.0
    values[13] = -1.0
    values[15] = 1.0
    values[17] = 1.0
    values[18] = gripper
    values[19] = 0.48
    return values


class V4PhaseOfflineGateTest(unittest.TestCase):
    def test_phase_task_index_is_resolved_before_training_collation(self) -> None:
        prompts = ["phase zero", "phase one"]
        sample = {"task": 1, "action": [0.0] * 20}
        resolved = resolve_phase_task(sample, prompts)
        self.assertEqual(resolved["task"], "phase one")
        self.assertEqual(sample["task"], 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_phase_task({"task": 2}, prompts)

    def test_prompt_order_does_not_follow_sorted_json_keys(self) -> None:
        serialized = json.dumps(
            {"phase_prompts": PHASE_PROMPTS}, sort_keys=True
        )
        manifest = json.loads(serialized)
        self.assertNotEqual(
            list(manifest["phase_prompts"].values()),
            list(PHASE_PROMPTS.values()),
        )
        self.assertEqual(
            ordered_phase_prompts(manifest), list(PHASE_PROMPTS.values())
        )

    def test_landmarks_cover_exact_phase_boundaries(self) -> None:
        record = {
            "events": {
                "spine_high": 100,
                "right_close": 300,
                "pad_move": 320,
                "target_arrival": 500,
                "right_release": 600,
            },
            "orientation_entry_frame": 200,
        }
        self.assertEqual(
            phase_landmark_frames(record),
            {
                "startup_rise": 95,
                "approach": 100,
                "orient_pregrasp": 200,
                "grasp_acquisition": 300,
                "lift_transfer": 320,
                "lower_place": 500,
                "release_retreat": 600,
            },
        )

    def test_identical_prompt_outputs_fail_discriminability(self) -> None:
        records = []
        for phase in PHASE_PROMPTS:
            reference = [action(0.0 if phase == "grasp_acquisition" else 1.0) for _ in range(5)]
            records.append(
                {
                    "phase": phase,
                    "state": [0.0] * 37,
                    "preclose_right_joints": [0.1] * 7,
                    "reference_actions": reference,
                    "predictions": {
                        prompt: copy.deepcopy(reference) for prompt in PHASE_PROMPTS
                    },
                }
            )
        report = evaluate_phase_records(records)
        self.assertFalse(report["checks"]["prompt_discriminability"])
        self.assertFalse(report["go"])

    def test_out_of_bounds_prediction_is_projected_and_reported(self) -> None:
        records = []
        for phase in PHASE_PROMPTS:
            reference = [action() for _ in range(5)]
            predictions = {
                prompt: copy.deepcopy(reference) for prompt in PHASE_PROMPTS
            }
            predictions[phase][0][13] = -4.0
            records.append(
                {
                    "phase": phase,
                    "state": [0.0] * 37,
                    "preclose_right_joints": [0.1] * 7,
                    "reference_actions": reference,
                    "predictions": predictions,
                }
            )
        report = evaluate_phase_records(records)
        self.assertTrue(report["checks"]["effective_action_safety"])
        self.assertLess(report["metrics"]["raw_arm_in_bounds_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
