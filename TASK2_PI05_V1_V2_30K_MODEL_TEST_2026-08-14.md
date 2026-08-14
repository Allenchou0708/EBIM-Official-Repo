# Task 2 PI0.5 V1/V2 030000 model test result (2026-08-14)

## Decision

V1 step 030000 and V2 expert-only step 030000 both pass checkpoint integrity,
native held-out loss, deterministic offline replay, action-size, bounds, and
fixed-base projection gates. This is an offline GO for both models. It is not a
GUI publication GO: each model still needs its own executed staging manifest,
zero-publication shadow, fresh observation evidence, and right-wrist visual
confirmation.

No V3, V4, or 015000 checkpoint was evaluated in this comparison.

## Checkpoint and training provenance

Both checkpoints record step 30000, batch size 1, `train_expert_only=true`,
`freeze_vision_encoder=true`, `n_action_steps=5`, a complete 9.354 GB model,
and a complete 2.195 GB optimizer state.

The V2 run completed all 30,000 steps with return code zero in about three
hours. Its final logged training loss is 0.009 and the log reports about 12.87
GiB GPU memory. The checkpoint records batch size 1. The attempted outer YAML
change to batch size 2 did not change the LeRobot profile and therefore must
not be used to label this trained artifact.

Action mapping is checkpoint-derived:

- V1: arms relative; spine action 19 relative to state index 28.
- V2: arms relative; spine action 19 absolute (`state_indices[19] = null`).

## Offline results

The immutable held-out episodes were
10, 13, 17, 44, 46, 61, 76, 78, 91, 107, 118, 123, 149, 150, 156, 163,
173, 177, 180, and 193. Seed was 1000.

| Checkpoint | Native frames | Mean loss | Min | Max | Finite/replay/bounds | ROS publications |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| V1 030000 | 128 | 11.507817 | 6.460280 | 29.386192 | PASS | 0 |
| V2 030000 | 128 | 11.411030 | 7.044229 | 28.516628 | PASS | 0 |

V2 mean loss is about 0.84% below V1 on this fixed sample. This small offline
difference is not evidence of simulator task success; both must proceed through
the same shadow and GUI gates.

An additional three-samples-per-episode replay produced 60/60 finite 20-D
actions for each checkpoint. Both were reproducible, within post-projection
joint/gripper bounds, had zero effective base commands, and published no ROS
commands.

Bulk evidence:

- `/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_v1_v2_30k/model_verification.json`
- `/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_v1_v2_30k/v1_030000_offline.json`
- `/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_v1_v2_30k/v2_030000_offline.json`
- `/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_v1_v2_30k/v1_030000_heldout_gate.json`
- `/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_v1_v2_30k/v2_030000_heldout_gate.json`
- `/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_v1_v2_30k/source_control_clock_tests.log`
- `/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_v2_expert_30k/training_verification.json`

## Control and clock readiness

The live control path is implemented and source-tested. It resets the scene,
runs the verified base route, stages the measured robot through the
dataset-derived episode 176/frame 408 joint route, rejects timeout or tolerance
failure, discards all staging observations, then requires a newly settled three-
camera + joint + odom + two-EE tuple.

Base pulses, manipulation commands, stable dwell, hard5 action scheduling, and
hold publication are paced by `/isaac/clock`. Arm/gripper command headers use
the current simulator stamp. Host monotonic time is restricted to process and
transport watchdogs. Capture skew remains at or below 0.10 s; capture-to-ready
latency is recorded in simulator time. Hard5 uses only chunk indices 0--4 and
holds the last legal absolute target during inference. Fixed-base, action
bounds/projection, stream freshness, clock availability, and publisher
contention checks remain fail-closed.

The V1/V2-specific shadow verifier now validates the correct checkpoint
mapping via `--contract v1` or `--contract v2`; the previous verifier only
accepted V2 and would have falsely rejected a correct V1 shadow.

Focused control/contract tests pass 74/74 in the pinned training image after
the change. `bash -n`, Python bytecode compilation, and `git diff --check` also
pass. Live manifest schema 8 reports capture-to-ready simulator latency as the
formal `inference_latency_s`; host policy-compute and transport timing remain
available under explicitly named diagnostic fields.

Use
`task2_isaacsim/baselines/pi05/DEMO_AND_DATASET_REPLAY.md` for the exact,
model-isolated GUI commands. The operator owns GUI launch and visual pad-front
confirmation.
