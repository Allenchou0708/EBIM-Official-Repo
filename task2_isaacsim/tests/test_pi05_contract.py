# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Task 2 PI05 data and action boundary."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from task2_isaacsim.baselines.pi05.contract import (
    ACTION_NAMES,
    ACTION_SIZE,
    PI05_ACTION_SIZE,
    RELATIVE_ACTION_STATE_INDICES,
    STATE_NAMES,
    apply_fixed_mobile_axes,
    pad_action,
    to_absolute_action,
    to_relative_action,
    unpad_action,
    validate_dataset_root,
    validate_info,
    validate_relative_action_state_indices,
)
from task2_isaacsim.baselines.pi05.loss_parity import loss_parity_report
from task2_isaacsim.baselines.pi05.portable import (
    SOURCE_ROOT,
    _require_external_output,
    _require_image_digest,
    deterministic_split,
    validate_profile,
)
from task2_isaacsim.baselines.pi05.relative_dataset import (
    compute_relative_action_stats,
    compute_vector_stats,
)
from task2_isaacsim.baselines.pi05.train_smoke import (
    HUB_PREFLIGHT_FILES,
    LEROBOT_SOURCE_COMMIT,
    PALIGEMMA_TOKENIZER_REPO,
    build_train_command,
    load_episode_labels,
    select_smoke_episodes,
    verify_checkpoint,
    verify_required_hub_access,
)
from task2_isaacsim.baselines.pi05.train_smoke import (
    main as train_smoke_main,
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


def make_valid_stats() -> dict:
    return {
        "action": {"q01": [0.0] * ACTION_SIZE, "q99": [1.0] * ACTION_SIZE},
        "observation.state": {
            "q01": [0.0] * len(STATE_NAMES),
            "q99": [1.0] * len(STATE_NAMES),
        },
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
            (root / "meta" / "stats.json").write_text(
                json.dumps(make_valid_stats()),
                encoding="utf-8",
            )
            info, errors = validate_dataset_root(root)
        self.assertEqual(errors, [])
        self.assertEqual(info["total_episodes"], 2)

    def test_missing_pi05_quantiles_fail_dataset_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps(make_valid_info()), encoding="utf-8"
            )
            (root / "meta" / "stats.json").write_text(
                json.dumps({"action": {}, "observation.state": {}}),
                encoding="utf-8",
            )
            _, errors = validate_dataset_root(root)
        self.assertIn("missing quantile statistics: action.q01", errors)
        self.assertIn(
            "missing quantile statistics: observation.state.q99", errors
        )

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

    def test_task2_relative_mapping_round_trip(self) -> None:
        action = tuple(100.0 + index for index in range(ACTION_SIZE))
        state = tuple(10.0 + index for index in range(len(STATE_NAMES)))
        relative = to_relative_action(action, state)

        self.assertEqual(relative[:3], action[:3])
        self.assertEqual(relative[3], action[3] - state[14])
        self.assertEqual(relative[9], action[9] - state[20])
        self.assertEqual(relative[10], action[10] - state[21])
        self.assertEqual(relative[16], action[16] - state[27])
        self.assertEqual(relative[17:19], action[17:19])
        self.assertEqual(relative[19], action[19] - state[28])
        self.assertEqual(to_absolute_action(relative, state), action)

    def test_relative_mapping_rejects_sequential_task2_shortcut(self) -> None:
        with self.assertRaisesRegex(ValueError, "20 entries"):
            validate_relative_action_state_indices([0, 1, 2])
        self.assertEqual(
            RELATIVE_ACTION_STATE_INDICES[:4], (None, None, None, 14)
        )

    def test_relative_stats_use_current_state_for_each_chunk(self) -> None:
        actions = [
            [float(frame + index) for index in range(ACTION_SIZE)]
            for frame in range(4)
        ]
        states = [
            [float(10 * frame + index) for index in range(len(STATE_NAMES))]
            for frame in range(4)
        ]
        stats, chunks = compute_relative_action_stats(
            actions,
            states,
            [0, 0, 0, 0],
            chunk_size=2,
            included_episodes=[0],
        )
        self.assertEqual(chunks, 3)
        self.assertEqual(stats["count"], [6])
        # Base stays raw. Left joint 1 uses state[14] at each chunk start.
        self.assertEqual(stats["min"][0], 0.0)
        self.assertEqual(stats["max"][0], 3.0)
        self.assertEqual(stats["min"][3], -29.0)
        self.assertEqual(stats["max"][3], -10.0)

    def test_vector_stats_include_pi05_quantiles(self) -> None:
        stats = compute_vector_stats([[0.0, 10.0], [2.0, 14.0]])
        self.assertEqual(stats["mean"], [1.0, 12.0])
        self.assertAlmostEqual(stats["q01"][0], 0.02)
        self.assertAlmostEqual(stats["q01"][1], 10.04)
        self.assertAlmostEqual(stats["q99"][0], 1.98)
        self.assertAlmostEqual(stats["q99"][1], 13.96)

    def test_loss_parity_keeps_native_20d_reduction(self) -> None:
        report = loss_parity_report()
        self.assertEqual(report["target_formula"], "noise - action")
        self.assertEqual(len(report["per_dimension_mse"]), PI05_ACTION_SIZE)
        self.assertEqual(report["decision"], "keep_lerobot_native_20d_loss")
        expected_32 = (
            report["lerobot_native_20d_loss"] * ACTION_SIZE
            + report["padded_12d_sum_contribution"]
        ) / PI05_ACTION_SIZE
        self.assertAlmostEqual(report["openpi_style_32d_loss"], expected_32)

    def test_ten_episode_split_is_deterministic_and_episode_level(
        self,
    ) -> None:
        train, held_out = deterministic_split(10, 2)
        self.assertEqual(train, list(range(8)))
        self.assertEqual(held_out, [8, 9])
        self.assertFalse(set(train) & set(held_out))

    def test_smoke_prefers_successful_episode_labels(self) -> None:
        labels = [
            {"episode_index": 0, "success": False},
            {"episode_index": 1, "success": True},
            {"episode_index": 2, "success": True},
            {"episode_index": 3, "success": True},
        ]
        episodes, uses_unsuccessful = select_smoke_episodes(
            labels, allow_unsuccessful=False, max_episodes=2
        )
        self.assertEqual(episodes, [1, 2])
        self.assertFalse(uses_unsuccessful)

    def test_failed_only_smoke_requires_explicit_opt_in(self) -> None:
        labels = [{"episode_index": 0, "success": False}]
        with self.assertRaisesRegex(ValueError, "allow-unsuccessful"):
            select_smoke_episodes(
                labels, allow_unsuccessful=False, max_episodes=2
            )
        episodes, uses_unsuccessful = select_smoke_episodes(
            labels, allow_unsuccessful=True, max_episodes=2
        )
        self.assertEqual(episodes, [0])
        self.assertTrue(uses_unsuccessful)

    def test_episode_label_loader_rejects_non_boolean_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extras = root / "task2_extras"
            extras.mkdir()
            (extras / "episodes_task2.jsonl").write_text(
                '{"episode_index":0,"success":1}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "boolean success"):
                load_episode_labels(root)

    def test_smoke_command_locks_local_only_pi05_contract(self) -> None:
        command = build_train_command(
            dataset_root=Path("dataset/example"),
            output_dir=Path("outputs/example"),
            dataset_repo_id="local/task2",
            episodes=[1, 2],
            steps=1,
            save_checkpoint=False,
        )
        self.assertIn("--policy.path=lerobot/pi05_base", command)
        self.assertTrue(
            any(
                item.startswith("--policy.pretrained_revision=")
                for item in command
            )
        )
        self.assertIn("--policy.max_state_dim=37", command)
        self.assertIn("--policy.max_action_dim=32", command)
        self.assertIn("--policy.train_expert_only=true", command)
        self.assertIn("--policy.freeze_vision_encoder=false", command)
        self.assertIn("--policy.use_relative_actions=true", command)
        mapping_arg = next(
            item
            for item in command
            if item.startswith("--policy.relative_action_state_indices=")
        )
        self.assertEqual(
            json.loads(mapping_arg.split("=", 1)[1]),
            list(RELATIVE_ACTION_STATE_INDICES),
        )
        self.assertIn("--dataset.episodes=[1,2]", command)
        self.assertIn("--policy.push_to_hub=false", command)
        self.assertIn("--save_checkpoint=false", command)
        rename_arg = next(
            item for item in command if item.startswith("--rename_map=")
        )
        self.assertNotIn("eval_camera", rename_arg)
        self.assertEqual(len(LEROBOT_SOURCE_COMMIT), 40)

    def test_checkpoint_verifier_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            missing = verify_checkpoint(output, 1)
        self.assertEqual(len(missing), 4)

    def test_hub_preflight_fetches_only_small_config_files(self) -> None:
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return f"/cache/{kwargs['repo_id']}/{kwargs['filename']}"

        dependencies = verify_required_hub_access(fake_download)

        self.assertEqual(len(dependencies), len(HUB_PREFLIGHT_FILES))
        self.assertEqual(len(calls), len(HUB_PREFLIGHT_FILES))
        self.assertTrue(
            all(call["filename"] == "config.json" for call in calls)
        )
        self.assertIn(
            {
                "repo_id": PALIGEMMA_TOKENIZER_REPO,
                "revision": "main",
                "filename": "config.json",
            },
            calls,
        )

    def test_hub_preflight_explains_paligemma_access_gate(self) -> None:
        def deny_paligemma(**kwargs):
            if kwargs["repo_id"] == PALIGEMMA_TOKENIZER_REPO:
                raise RuntimeError("401 Unauthorized")
            return "/cache/config.json"

        with self.assertRaisesRegex(
            RuntimeError,
            "accept the Google PaliGemma usage license.*hf auth login",
        ):
            verify_required_hub_access(deny_paligemma)

    def test_failed_episode_cli_dry_run_is_explicit_and_local_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            (root / "meta").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(
                json.dumps({**make_valid_info(), "total_episodes": 1}),
                encoding="utf-8",
            )
            (root / "meta" / "stats.json").write_text(
                json.dumps(make_valid_stats()), encoding="utf-8"
            )
            extras = root / "task2_extras"
            extras.mkdir()
            (extras / "episodes_task2.jsonl").write_text(
                '{"episode_index":0,"success":false}\n',
                encoding="utf-8",
            )
            stdout = StringIO()
            argv = [
                "train_smoke.py",
                "--dataset-root",
                str(root),
                "--output-dir",
                str(Path(directory) / "output"),
                "--allow-unsuccessful-smoke-data",
            ]
            with patch("sys.argv", argv), redirect_stdout(stdout):
                result = train_smoke_main()

        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn("mode=dry-run steps=1", output)
        self.assertIn("discard all resulting weights", output)
        self.assertIn("--policy.push_to_hub=false", output)

    def test_profiles_separate_disposable_expert_and_formal_full(self) -> None:
        profile_dir = (
            Path(__file__).resolve().parents[1]
            / "baselines"
            / "pi05"
            / "profiles"
        )
        smoke = (profile_dir / "smoke_expert.yaml").read_text(encoding="utf-8")
        full = (profile_dir / "full_finetune.yaml").read_text(encoding="utf-8")
        self.assertIn("train_expert_only: true", smoke)
        self.assertIn("train_expert_only: false", full)
        self.assertIn("freeze_vision_encoder: false", full)
        self.assertIn("use_relative_actions: true", smoke)
        self.assertIn("use_relative_actions: true", full)

    def test_profile_validation_requires_unfrozen_vision_encoder(self) -> None:
        profile = {
            "policy": {
                "dtype": "bfloat16",
                "gradient_checkpointing": True,
                "freeze_vision_encoder": True,
                "use_relative_actions": True,
                "max_state_dim": 37,
                "max_action_dim": 32,
                "relative_action_state_indices": list(
                    RELATIVE_ACTION_STATE_INDICES
                ),
                "push_to_hub": False,
                "train_expert_only": False,
            }
        }
        with patch(
            "task2_isaacsim.baselines.pi05.portable._load_yaml",
            return_value=profile,
        ):
            with self.assertRaisesRegex(ValueError, "freeze_vision_encoder"):
                validate_profile(Path("full_finetune.yaml"))

    def test_executable_run_requires_immutable_image_digest(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "immutable sha256 digest"):
                _require_image_digest()
        expected = "sha256:" + "a" * 64
        with patch.dict(
            os.environ, {"EBIM_PI05_IMAGE_DIGEST": expected}, clear=True
        ):
            self.assertEqual(_require_image_digest(), expected)

    def test_output_guard_rejects_repository_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            _require_external_output(SOURCE_ROOT / "outputs" / "run")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                _require_external_output(Path(directory)),
                Path(directory).resolve(),
            )

    def test_container_is_pinned_and_excludes_runtime_artifacts(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        dockerfile = (
            repository_root
            / "task2_isaacsim"
            / "baselines"
            / "pi05"
            / "docker"
            / "Dockerfile"
        ).read_text(encoding="utf-8")
        dockerignore = (repository_root / ".dockerignore").read_text(
            encoding="utf-8"
        )
        readme = (
            repository_root
            / "task2_isaacsim"
            / "baselines"
            / "pi05"
            / "README.md"
        ).read_text(encoding="utf-8")
        entrypoint = (
            repository_root
            / "task2_isaacsim"
            / "baselines"
            / "pi05"
            / "docker"
            / "entrypoint.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("30da8e687a6dfc617fcd94afc367ac7071c376ce", dockerfile)
        self.assertIn("lerobot-v0.6.0-task2-relative-map.patch", dockerfile)
        self.assertNotIn("HF_TOKEN=", dockerfile)
        self.assertIn("**/dataset/*", dockerignore)
        self.assertIn("**/*.safetensors", dockerignore)
        self.assertIn("$TASK2_PI05_ROOT/cache:/cache", readme)
        self.assertNotIn(
            "$TASK2_PI05_ROOT/cache/huggingface:/cache/huggingface", readme
        )
        self.assertIn(
            "Mount the writable host cache root at /cache", entrypoint
        )
        portable = (
            repository_root
            / "task2_isaacsim"
            / "baselines"
            / "pi05"
            / "portable.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"resume_manifest.json"', portable)
        self.assertIn('"resumed_checkpoint_hashes"', portable)


if __name__ == "__main__":
    unittest.main()
