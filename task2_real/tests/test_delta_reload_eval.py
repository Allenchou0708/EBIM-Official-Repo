from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from task2_real.delta_reload_eval import _strict_load_delta, _summarize


class DeltaReloadEvalTest(unittest.TestCase):
    def test_strict_delta_load_records_witness(self) -> None:
        policy = torch.nn.Linear(2, 3)
        delta = {
            name: parameter.detach().clone() + 0.25
            for name, parameter in policy.named_parameters()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delta.safetensors"
            save_file(delta, path, metadata={"step": "1"})
            result = _strict_load_delta(policy, path, torch)
        self.assertTrue(result["strict_key_set_match"])
        self.assertTrue(result["witness_exact_after_load"])
        self.assertFalse(result["deployable"])
        for name, parameter in policy.named_parameters():
            self.assertTrue(torch.equal(parameter.detach(), delta[name]))

    def test_summary_keeps_empirical_envelope_warning(self) -> None:
        row = {
            "target_gripper_phase": "open",
            "predicted_gripper_phase": "open",
            "phase": "reopen",
            "joint_action_mae": 1.0,
            "gripper_action_mae": 2.0,
            "predicted_chunk_mean_abs_delta": 3.0,
            "empirical_train_action_envelope_violation_fraction": 0.25,
        }
        result = _summarize([row])
        self.assertEqual(result["gripper_phase_correct"], 1)
        self.assertEqual(result["per_landmark_phase"]["reopen"]["accuracy"], 1.0)
        self.assertIn("not a physical safety bound", result["envelope_metric_warning"])


if __name__ == "__main__":
    unittest.main()
