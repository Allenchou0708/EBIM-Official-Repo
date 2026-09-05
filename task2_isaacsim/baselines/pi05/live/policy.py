# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed PI0.5 action-chunk inference for live observations."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from task2_isaacsim.baselines.pi05.contract import (
    ACTION_NAMES,
)


RIGHT_ONLY_STATE_INDICES = (*range(21, 28), 30)
RIGHT_ONLY_ACTION_NAMES = (*ACTION_NAMES[10:17], ACTION_NAMES[18])
LEGACY_PI05_CONFIG_FIELDS = frozenset(
    {
        "relative_action_state_indices",
        "use_visual_memory",
        "use_proprioceptive_memory",
        "memory_frames",
        "memory_stride",
        "memory_temporal_attention_every",
        "rtc_training_max_delay",
    }
)


def _compatible_config_payload(
    checkpoint: Path, config_class: type[Any]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Remove only known legacy metadata unsupported by this LeRobot build."""

    payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    supported = {field.name for field in fields(config_class)} | {"type"}
    unsupported = set(payload) - supported
    unknown = unsupported - LEGACY_PI05_CONFIG_FIELDS
    if unknown:
        raise ValueError(
            "checkpoint config contains unknown unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    return (
        {key: value for key, value in payload.items() if key not in unsupported},
        tuple(sorted(unsupported)),
    )


def _policy_action_size(config: Any) -> int:
    feature = config.output_features.get("action")
    shape = tuple(int(value) for value in feature.shape)
    if len(shape) != 1 or shape[0] not in (8, 20):
        raise ValueError(f"unsupported live policy action shape: {shape}")
    return shape[0]


def _right_only_state(state: tuple[float, ...]) -> tuple[float, ...]:
    if len(state) != 37:
        raise ValueError("live Task 2 state must contain 37 values")
    return tuple(float(state[index]) for index in RIGHT_ONLY_STATE_INDICES)


def _expand_right_only_action(
    action: list[float] | tuple[float, ...], state: tuple[float, ...]
) -> list[float]:
    """Embed an 8-D right-only checkpoint output in the official 20-D action."""

    values = tuple(float(value) for value in action)
    if len(values) != 8:
        raise ValueError("right-only policy action must contain 8 values")
    if len(state) != 37:
        raise ValueError("live Task 2 state must contain 37 values")
    return [
        0.0,
        0.0,
        0.0,
        *state[14:21],
        *values[:7],
        state[29],
        values[7],
        state[28],
    ]


def _saved_relative_action_state_indices(
    checkpoint: Path, action_size: int
) -> tuple[int | None, ...] | None:
    payload = json.loads(
        (checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8")
    )
    relative_steps = [
        step
        for step in payload["steps"]
        if step.get("registry_name") == "relative_actions_processor"
    ]
    if len(relative_steps) != 1:
        raise ValueError("checkpoint must declare one relative-actions processor")
    config = relative_steps[0]["config"]
    if config.get("enabled") is not True:
        return None
    indices = config.get("state_indices")
    if indices is None and action_size == 8:
        return (*range(7), None)
    if not isinstance(indices, list) or len(indices) != action_size:
        raise ValueError("invalid saved relative-action state mapping")
    return tuple(None if value is None else int(value) for value in indices)


def _write_compatible_processor_bundle(
    checkpoint: Path, destination: Path
) -> bool:
    """Disable an unsupported mapped inverse so the runner can apply it exactly."""

    pre_path = checkpoint / "policy_preprocessor.json"
    post_path = checkpoint / "policy_postprocessor.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    post = json.loads(post_path.read_text(encoding="utf-8"))
    mapped_steps = [
        step
        for step in pre["steps"]
        if step.get("registry_name") == "relative_actions_processor"
        and "state_indices" in step.get("config", {})
    ]
    if not mapped_steps:
        return False
    if len(mapped_steps) != 1:
        raise ValueError("checkpoint must declare exactly one mapped relative step")
    mapped_steps[0]["config"].pop("state_indices")
    mapped_steps[0]["config"]["enabled"] = False
    absolute_steps = [
        step
        for step in post["steps"]
        if step.get("registry_name") == "absolute_actions_processor"
    ]
    if len(absolute_steps) != 1 or absolute_steps[0]["config"].get("enabled") is not True:
        raise ValueError("mapped relative checkpoint requires one enabled absolute step")
    absolute_steps[0]["config"]["enabled"] = False
    destination.mkdir(parents=True, exist_ok=True)
    (destination / pre_path.name).write_text(json.dumps(pre), encoding="utf-8")
    (destination / post_path.name).write_text(json.dumps(post), encoding="utf-8")
    for state_file in checkpoint.glob("policy_*processor*.safetensors"):
        (destination / state_file.name).symlink_to(state_file)
    return True


class LivePi05Policy:
    """Load checkpoint processors once and preserve their relative inverse."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        instruction: str,
        seed: int,
        tokenizer_max_length: int | None = None,
    ):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.pi05 import PI05Config, PI05Policy

        self.torch = torch
        self.instruction = instruction
        self.seed = seed
        self.decision_index = 0
        config_payload, ignored_fields = _compatible_config_payload(
            checkpoint, PI05Config
        )
        self.ignored_legacy_config_fields = ignored_fields
        if ignored_fields:
            with tempfile.TemporaryDirectory(prefix="pi05-config-") as directory:
                Path(directory, "config.json").write_text(
                    json.dumps(config_payload), encoding="utf-8"
                )
                config = PreTrainedConfig.from_pretrained(
                    directory, local_files_only=True
                )
        else:
            config = PreTrainedConfig.from_pretrained(
                str(checkpoint), local_files_only=True
            )
        config.device = "cuda"
        self.action_size = _policy_action_size(config)
        self.relative_action_state_indices = _saved_relative_action_state_indices(
            checkpoint, self.action_size
        )
        if self.action_size == 8:
            if tuple(config.action_feature_names) != RIGHT_ONLY_ACTION_NAMES:
                raise ValueError(
                    "8-D checkpoint does not use the locked right-only action contract"
                )
        self.policy = PI05Policy.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            strict=True,
        )
        self.policy.eval()
        self.chunk_size = int(self.policy.config.chunk_size)
        self.n_action_steps = int(self.policy.config.n_action_steps)
        preprocessor_overrides: dict[str, dict[str, object]] = {
            "device_processor": {"device": "cuda"}
        }
        local_tokenizer = checkpoint / "tokenizer"
        tokenizer_overrides: dict[str, object] = {}
        if local_tokenizer.is_dir():
            # Older LeRobot processor loaders do not resolve a saved relative
            # tokenizer artifact against the checkpoint directory.
            tokenizer_overrides["tokenizer_name"] = str(local_tokenizer)
        if tokenizer_max_length is not None:
            if tokenizer_max_length <= 0:
                raise ValueError("tokenizer_max_length must be positive")
            tokenizer_overrides["max_length"] = tokenizer_max_length
        if tokenizer_overrides:
            preprocessor_overrides["tokenizer_processor"] = tokenizer_overrides
        with tempfile.TemporaryDirectory(prefix="pi05-processors-") as directory:
            compatibility_path = Path(directory)
            self.manual_relative_inverse = _write_compatible_processor_bundle(
                checkpoint, compatibility_path
            )
            processor_path = (
                compatibility_path if self.manual_relative_inverse else checkpoint
            )
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.policy.config,
                pretrained_path=str(processor_path),
                preprocessor_overrides=preprocessor_overrides,
            )

    def reset(self) -> None:
        self.policy.reset()
        self.decision_index = 0

    def predict_chunk(
        self,
        *,
        images: dict[str, Any],
        state: tuple[float, ...],
        instruction: str | None = None,
    ) -> tuple[list[list[float]], float]:
        """Return one complete postprocessed absolute action chunk."""

        torch = self.torch
        fixed_seed = self.seed + self.decision_index
        self.decision_index += 1
        torch.manual_seed(fixed_seed)
        torch.cuda.manual_seed_all(fixed_seed)
        policy_state = _right_only_state(state) if self.action_size == 8 else state
        observation: dict[str, Any] = {
            "observation.state": torch.tensor(
                policy_state, dtype=torch.float32
            ),
            "task": self.instruction if instruction is None else instruction,
        }
        for key, image in images.items():
            if self.action_size == 8 and key == "wrist_left":
                continue
            observation[f"observation.images.{key}"] = (
                torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            )
        started = time.monotonic()
        with torch.inference_mode():
            processed = self.preprocessor(observation)
            predicted_chunk = self.policy.predict_action_chunk(processed)
            absolute_chunk = self.postprocessor(predicted_chunk)
            if self.manual_relative_inverse:
                if self.relative_action_state_indices is None:
                    raise RuntimeError("mapped relative inverse is missing its state map")
                absolute_chunk = absolute_chunk.clone()
                for action_index, state_index in enumerate(
                    self.relative_action_state_indices
                ):
                    if state_index is not None:
                        absolute_chunk[..., action_index] += float(
                            state[state_index]
                        )
        torch.cuda.synchronize()
        latency = time.monotonic() - started
        values = (
            absolute_chunk.detach().cpu().reshape(-1, self.action_size).tolist()
        )
        if self.action_size == 8:
            values = [_expand_right_only_action(action, state) for action in values]
        return values, latency
