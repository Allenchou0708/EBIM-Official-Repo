import unittest
from types import SimpleNamespace

from task2_isaacsim.baselines.pi05.manipulation_only import (
    CHUNK_SIZE,
    PHASE_RATIOS,
    crop_record,
    phase_groups_for_record,
    source_records,
)
from task2_isaacsim.baselines.pi05.manipulation_gate import phase_sample_counts
from task2_isaacsim.baselines.pi05.manipulation_eval import gripper_phase
from task2_isaacsim.baselines.pi05.phase_train import install_draccus_encode_compat


class ManipulationOnlyTest(unittest.TestCase):
    def setUp(self):
        self.source = {
            "episode": 7,
            "length": 900,
            "events": {
                "start": 0,
                "spine_high": 150,
                "right_close": 420,
                "pad_move": 450,
                "target_arrival": 580,
                "right_release": 800,
                "end": 900,
            },
        }

    def test_crop_keeps_one_second_before_close_and_full_retreat(self):
        crop = crop_record(self.source)
        self.assertEqual(crop["source_start_frame"], 390)
        self.assertEqual(crop["source_end_exclusive"], 900)
        self.assertEqual(crop["frames"], 510)
        self.assertEqual(crop["derived_events"]["pre_close"], 0)
        self.assertEqual(crop["derived_events"]["grasp_acquisition"], 30)
        self.assertEqual(crop["derived_events"]["release_retreat"], 410)
        self.assertEqual(crop["derived_events"]["end"], 510)

    def test_groups_are_disjoint_and_only_full_horizon_starts(self):
        crop = crop_record(self.source)
        groups = phase_groups_for_record(crop, derived_global_start=1000)
        flattened = [value for name in PHASE_RATIOS for value in groups[name]]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(min(flattened), 1000)
        self.assertEqual(max(flattened), 1000 + crop["frames"] - CHUNK_SIZE)
        self.assertTrue(all(groups[name] for name in PHASE_RATIOS))

    def test_invalid_event_order_is_rejected(self):
        self.source["events"]["pad_move"] = 419
        with self.assertRaisesRegex(ValueError, "event order"):
            crop_record(self.source)

    def test_multi_episode_audit_rows_are_normalized(self):
        rows = []
        for episode in range(200):
            rows.append(
                {
                    "episode_index": episode,
                    "frames": 900,
                    "phase_frames": {
                        "grasp": 420,
                        "retained_lift": 450,
                        "transfer": 580,
                        "release": 800,
                    },
                }
            )
        records = source_records({"episode_records": rows})
        self.assertEqual(records[132]["events"]["right_close"], 420)
        self.assertEqual(records[132]["events"]["right_release"], 800)

    def test_balanced_epoch_uses_declared_ratios_exactly(self):
        manifest = {"train_sampling_groups": {name: [1] for name in PHASE_RATIOS}}
        counts = phase_sample_counts(manifest, epoch_size=1000)
        self.assertEqual(counts, {name: value * 10 for name, value in PHASE_RATIOS.items()})
        self.assertEqual(sum(counts.values()), 1000)

    def test_gripper_phase_uses_fixed_hysteresis_regions(self):
        self.assertEqual(gripper_phase(0.25), "closed")
        self.assertEqual(gripper_phase(0.50), "transition")
        self.assertEqual(gripper_phase(0.90), "open")

    def test_draccus_save_compat_ignores_legacy_schema_argument(self):
        module = SimpleNamespace(encode=lambda value: {"encoded": value})
        self.assertTrue(install_draccus_encode_compat(module))
        self.assertEqual(module.encode("value", object()), {"encoded": "value"})


if __name__ == "__main__":
    unittest.main()
