"""Tests for the Task 2 PI05 data and action boundary."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from task2_isaacsim.baselines.pi05.contract import (
    ACTION_NAMES,
    ACTION_SIZE,
    PI05_ACTION_SIZE,
    STATE_NAMES,
    apply_fixed_mobile_axes,
    pad_action,
    unpad_action,
    validate_dataset_root,
    validate_info,
)


def make_valid_info() -> dict:
    features = {
        "action": {
            "dtype": "float32",
            "shape": [ACTION_SIZE],
            "names": list(ACTION_NAMES),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": list(STATE_NAMES),
        },
    }
    for key, shape in {
        "observation.images.head": [720, 1280, 3],
        "observation.images.wrist_left": [480, 848, 3],
        "observation.images.wrist_right": [480, 848, 3],
        "observation.images.eval_camera": [720, 1280, 3],
    }.items():
        features[key] = {"dtype": "video", "shape": shape}
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "robot_type": "fr3duo_mobile_task2",
        "features": features,
        "total_episodes": 2,
        "total_frames": 10,
    }


class Pi05ContractTest(unittest.TestCase):
    def test_valid_metadata_passes(self) -> None:
        self.assertEqual(validate_info(make_valid_info()), [])

    def test_wrong_action_order_fails(self) -> None:
        info = make_valid_info()
        info["features"]["action"]["names"][0:2] = ["base.vy", "base.vx"]
        errors = validate_info(info)
        self.assertIn(
            "action.names do not match the official ordered contract",
            errors,
        )

    def test_missing_policy_camera_fails(self) -> None:
        info = make_valid_info()
        del info["features"]["observation.images.wrist_right"]
        self.assertIn(
            "missing feature: observation.images.wrist_right",
            validate_info(info),
        )

    def test_dataset_root_loads_info_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps(make_valid_info()),
                encoding="utf-8",
            )
            info, errors = validate_dataset_root(root)
        self.assertEqual(errors, [])
        self.assertEqual(info["total_episodes"], 2)

    def test_action_padding_round_trip(self) -> None:
        action = tuple(float(index) for index in range(ACTION_SIZE))
        padded = pad_action(action)
        self.assertEqual(len(padded), PI05_ACTION_SIZE)
        self.assertEqual(padded[ACTION_SIZE:], (0.0,) * 12)
        self.assertEqual(unpad_action(padded), action)

    def test_non_finite_action_is_rejected(self) -> None:
        action = [0.0] * ACTION_SIZE
        action[4] = math.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pad_action(action)

    def test_fixed_mobile_axes_preserve_arms_and_grippers(self) -> None:
        action = tuple(float(index) for index in range(ACTION_SIZE))
        safe = apply_fixed_mobile_axes(action, spine_height=0.42)
        self.assertEqual(safe[:3], (0.0, 0.0, 0.0))
        self.assertEqual(safe[3:19], action[3:19])
        self.assertEqual(safe[19], 0.42)


if __name__ == "__main__":
    unittest.main()
