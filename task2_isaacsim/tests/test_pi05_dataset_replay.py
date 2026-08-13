# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from task2_isaacsim.baselines.pi05.dataset_replay import (
    load_episode,
    summarize_trajectory,
)


class DatasetReplayLoaderTest(unittest.TestCase):
    def test_loads_contiguous_raw_20d_actions_and_summarizes_events(self) -> None:
        state = [0.0] * 37
        state[14:28] = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854] * 2
        state[29:31] = [1.0, 1.0]
        rows = []
        for frame in range(4):
            frame_state = state.copy()
            frame_state[28] = 0.1 * frame
            action = [0.0] * 20
            action[3:17] = state[14:28]
            action[17] = 1.0
            action[18] = 0.0 if frame >= 2 else 1.0
            action[19] = 0.1 * frame
            rows.append(
                {
                    "index": 100 + frame,
                    "episode_index": 7,
                    "frame_index": frame,
                    "timestamp": frame / 30.0,
                    "action": action,
                    "observation.state": frame_state,
                }
            )

        loaded = load_episode(Path("/unused"), 7, row_reader=lambda _: rows)
        summary = summarize_trajectory(loaded)

        self.assertEqual(len(loaded[0]["action"]), 20)
        self.assertEqual(len(loaded[0]["state"]), 37)
        self.assertEqual(summary["right_gripper_first_close_frame"], 2)
        self.assertEqual(summary["spine_first_0_10_frame"], 1)
        self.assertEqual(summary["spine_first_0_30_frame"], 3)
        self.assertEqual(summary["recorded_spine_first_0_10_frame"], 1)
        self.assertEqual(summary["recorded_spine_first_0_30_frame"], 3)
        self.assertAlmostEqual(summary["timestamp_step_s"], 1.0 / 30.0)
        self.assertEqual(
            summary["arm_velocity_limit_analysis"]["violation_count"], 0
        )
        self.assertTrue(summary["raw_actions"])
        self.assertFalse(summary["mapped_relative_actions"])

        changed_action = list(loaded[1]["action"])
        changed_action[10] = 0.1
        loaded[1]["action"] = tuple(changed_action)
        analysis = summarize_trajectory(loaded)["arm_velocity_limit_analysis"]
        self.assertEqual(analysis["first_violation"]["frame"], 1)
        self.assertEqual(analysis["first_violation"]["side"], "right")
        self.assertGreater(analysis["first_violation"]["required_rad_s"], 2.62)


if __name__ == "__main__":
    unittest.main()
