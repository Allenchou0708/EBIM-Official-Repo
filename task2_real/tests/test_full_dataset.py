import json
import tempfile
import unittest
from pathlib import Path

from task2_real.full_dataset import (
    TAIL_REPEAT_CAP_S,
    load_frozen_split,
    timestamp_sample_policy,
)


class FrozenSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = {
            "dataset": {
                "repo_id": "ebim-benchmark/ebim_task2_realrobotdata",
                "revision": "495ebb7b56fb9e2f3952398a63d86f08cacb9531",
            }
        }

    def _write(self, payload: dict) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "split.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _payload(self) -> dict:
        return {
            "dataset_repo_id": self.contract["dataset"]["repo_id"],
            "dataset_revision": self.contract["dataset"]["revision"],
            "train": list(range(119)),
            "held_out": list(range(119, 149)),
            "train_frames": 71420,
            "held_out_frames": 15024,
        }

    def test_accepts_locked_counts_without_overlap(self) -> None:
        payload = self._payload()
        self.assertEqual(load_frozen_split(self._write(payload), self.contract), payload)

    def test_rejects_train_heldout_overlap(self) -> None:
        payload = self._payload()
        payload["held_out"][-1] = payload["train"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_frozen_split(self._write(payload), self.contract)

    def test_tail_repeat_requires_exhaustion_and_obeys_cap(self) -> None:
        method, gap = timestamp_sample_policy(
            query_s=10.25,
            selected_s=10.0,
            decoder_exhausted=True,
            camera_key="observation.images.head",
        )
        self.assertEqual(method, "tail_repeat")
        self.assertAlmostEqual(gap, 0.25)
        with self.assertRaisesRegex(ValueError, "tail-repeat gap"):
            timestamp_sample_policy(
                query_s=10.0 + TAIL_REPEAT_CAP_S + 0.001,
                selected_s=10.0,
                decoder_exhausted=True,
                camera_key="observation.images.head",
            )

    def test_large_non_tail_nearest_error_is_not_reclassified(self) -> None:
        with self.assertRaisesRegex(ValueError, "nearest timestamp error"):
            timestamp_sample_policy(
                query_s=10.25,
                selected_s=10.0,
                decoder_exhausted=False,
                camera_key="observation.images.head",
            )


if __name__ == "__main__":
    unittest.main()
