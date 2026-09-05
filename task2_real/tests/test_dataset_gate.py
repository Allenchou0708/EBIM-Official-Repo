import json
import math
import unittest
from pathlib import Path

from task2_real.dataset_gate import (
    continuous_gripper_lifecycle,
    image_references,
    policy_vectors,
    time_blocked_split,
    validity_eligibility,
)
from task2_real.video_alignment import nearest_timestamp_index


CONTRACT = json.loads(
    (Path(__file__).parents[1] / "contract.json").read_text(encoding="utf-8")
)


class DatasetGateTest(unittest.TestCase):
    def test_right_only_policy_vectors(self) -> None:
        state = [float(index) for index in range(42)]
        action = [float(index) for index in range(17)]
        selected_state, selected_action = policy_vectors(state, action, CONTRACT)
        self.assertEqual(selected_state, [float(index) for index in range(21, 29)])
        self.assertEqual(selected_action, [float(index) for index in range(8, 16)])
        self.assertNotIn(16.0, selected_action)

    def test_policy_vectors_reject_bad_raw_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "42 values"):
            policy_vectors([0.0] * 41, [0.0] * 17, CONTRACT)
        bad_action = [0.0] * 17
        bad_action[8] = math.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            policy_vectors([0.0] * 42, bad_action, CONTRACT)

    def test_split_is_time_blocked_and_immutable(self) -> None:
        split = time_blocked_split([8, 1, 2, 5, 3], held_out_fraction=0.2)
        self.assertEqual(split["train"], [1, 2, 3, 5])
        self.assertEqual(split["held_out"], [8])
        self.assertTrue(set(split["train"]).isdisjoint(split["held_out"]))

    def test_mixed_validity_episode_is_rejected_without_joining_runs(self) -> None:
        result = validity_eligibility([1, 1, 0, 1, 1, 1])
        self.assertFalse(result["eligible"])
        self.assertEqual(result["valid_frames"], 5)
        self.assertEqual(
            result["contiguous_valid_runs"],
            [
                {"start": 0, "end_exclusive": 2, "length": 2},
                {"start": 3, "end_exclusive": 6, "length": 3},
            ],
        )

    def test_image_view_excludes_left_wrist(self) -> None:
        images = image_references(1001, 1.25)
        self.assertEqual(
            set(images),
            {"observation.images.head", "observation.images.wrist_right"},
        )
        self.assertIn("chunk-001", images["observation.images.head"]["path"])

    def test_continuous_gripper_requires_one_contiguous_cycle(self) -> None:
        complete = continuous_gripper_lifecycle(
            [1.0, 0.8, *([0.22] * 10), 0.6, 0.95]
        )
        split_hold = continuous_gripper_lifecycle(
            [1.0, *([0.22] * 5), 0.5, *([0.22] * 5), 0.95]
        )
        starts_closed = continuous_gripper_lifecycle([*([0.22] * 12), 1.0])
        self.assertTrue(complete["complete"])
        self.assertFalse(split_hold["complete"])
        self.assertFalse(starts_closed["complete"])

    def test_nearest_video_timestamp_uses_sensor_timeline(self) -> None:
        self.assertEqual(nearest_timestamp_index([0.0, 0.1, 0.2], 0.16), 2)


if __name__ == "__main__":
    unittest.main()
