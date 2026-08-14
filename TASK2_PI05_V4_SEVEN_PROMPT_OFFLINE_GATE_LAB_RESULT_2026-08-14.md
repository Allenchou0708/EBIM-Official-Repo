# Task 2 PI0.5 V4 Seven-Prompt Offline Gate Lab Result — 2026-08-14

## Outcome

**NO-GO for simulator/GUI.** A single 3000-step V4 continuation from the V2
6000-step checkpoint completed successfully, but the full 20-episode held-out
gate rejected it. V4 learned some prompt sensitivity and recovered useful
startup/open behavior, but grasp closing was not reliable, the correct phase
prompt was not reliably the best prompt for the demonstrated action, and the
orient-pregrasp prediction did not point strongly enough toward the demonstrated
vertical pre-grasp joint cluster.

No V4 shadow, Isaac Sim, GUI, or evaluator run was started. V1 and V2 remain
the GUI comparison baselines. V3 remains historical evidence only.

## Repository and input consistency

- Repository/branch/base:
  `/home/robot/2026_ebim_ssd/benchmark_task2_591def2`,
  `submit@71fb463e9ceb1eec9c2d35782375be0a87a24580`.
- The local worktree was clean before V4 work; `collab/submit` and a live
  `git ls-remote` both resolved to the same base commit.
- V2 initialization checkpoint:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v2_12k/training_12k/checkpoints/006000/pretrained_model`.
- Seven-prompt dataset:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v4_pregrasp/phase_conditioned_dataset`.
- The dataset reports 174719 frames, 200 episodes, and eight task entries:
  seven phase prompts plus the original fallback for excluded episode 132.
- The phase manifest contains 179 train and 20 held-out episodes; no held-out
  frame occurs in a training sampler group. Numeric actions/states, videos,
  normalization statistics, and representative `task2_extras` files compare
  identical to the V2 relative dataset view.
- The existing zero-publication live shadow, simulator-clock base-stage smoke,
  GUI UDPv4 log, and V2 checkpoint referenced by the preceding reports are all
  present. No previous evidence was overwritten.

## V4 implementation and experiment

V4 preserves relative arms, absolute spine, `n_action_steps=5`, expert-only
training, the frozen vision encoder, bfloat16, batch size 1, and the existing
action/state mapping. It initializes from V2, uses the exact seven task strings
in `phase_conditioned_manifest.json`, samples phases at
`20/15/20/20/10/10/5%`, runs at learning rate `5e-6`, and saves only the final
3000-step checkpoint.

The organizer schema exposes task indices. A pre-step gate found that LeRobot's
default collate path passed those indices as tensors to the PI0.5 text
processor. The V4-only phase collate now resolves each index through the
manifest before tokenization. It does not change numeric actions, observations,
camera data, or the live runner.

The first completed 3000-step attempt exposed a second precondition bug during
evidence review: JSON was serialized with sorted dictionary keys, while the
collate initially treated dictionary iteration order as task-index order. That
checkpoint is quarantined under the sibling directory ending in
`invalid_prompt_order` and is not used for the V4 conclusion. The formal run
uses the canonical phase order and records the complete task-index-to-prompt
mapping in `run_manifest.json`; training refuses a manifest whose strings drift
from that order.

Final run root:
`/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v4_from_v2_3k`.

- Training: 3000/3000 steps in `21:24`, including checkpoint serialization.
- Final loss/gradient norm: `0.025/0.774`.
- GPU memory: `12.87 GB`.
- Trainable/total parameters: `693422112/4143404816`.
- Final checkpoint:
  `training/checkpoints/003000/pretrained_model` (`model.safetensors` about
  9.35 GB).
- The only numbered checkpoint is `003000`; `last` is LeRobot's pointer to it.
- Run manifest return code: `0`.

Two shared-cache permission attempts stopped before dataset creation, and one
task-index attempt stopped at step 0 before the first update. They are retained
as diagnostic evidence under the sibling directories ending in
`failed_prestart_cache_lock`, `failed_prestart_cache_lock2`, and
`failed_prestep_task_tensor`. The formal run uses a run-local
`HF_DATASETS_CACHE` and is not mixed with those attempts.

## Offline GO/NO-GO gate

The gate uses all 20 immutable held-out episodes. For each episode it evaluates
seven observable event landmarks and runs all seven prompts on the same
observation with the same seed. Only hard5 indices 0--4 are scored. It performs
no ROS publication.

Primary report:
`/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v4_from_v2_3k/offline_gate/v4_offline_gate.json`.

The gate separates runtime-effective safety from raw projection diagnostics,
so harmless raw gripper values slightly above 1.0 are not confused with unsafe
effective commands.

| Gate | Metric | Threshold | Result |
|---|---:|---:|---|
| Effective action safety | 1.000 | 1.000 | PASS |
| Startup arm MAE | 0.0144 rad | <= 0.12 | PASS |
| Startup direction cosine | 0.802 | >= 0.40 | PASS |
| Approach stays open | 1.000 | >= 0.95 | PASS |
| Orient stays open | 1.000 | >= 0.95 | PASS |
| Orient right-arm MAE | 0.0357 rad | <= 0.12 | PASS |
| Orient-to-preclose cosine | 0.332 | >= 0.50 | **FAIL** |
| Orient joint-4 saturation | 0.000 | <= 0.02 | PASS |
| Grasp close recall | 0.650 | >= 0.80 | **FAIL** |
| Release open recall | 0.900 | >= 0.90 | PASS |
| Correct-prompt-best fraction | 0.121 | >= 0.45 | **FAIL** |
| Correct prompt margin, median | -0.00090 | >= 0 | **FAIL** |
| Prompt delta L2, median | 0.0226 | >= 0.02 | PASS |

Raw arm commands are within limits for `99.61%` of prompt actions, raw spine
commands for `100%`, and raw grippers for `62.55%`; most raw gripper failures
are small values above 1.0. The unchanged runtime clamp/projection contract
makes all effective actions safe. This diagnostic does not alter the NO-GO:
grasp closing, orientation direction, and correct prompt association fail.

## Observable phase manager design

`phase_manager.py` implements a forward-only seven-state prompt selector. It
uses simulator timestamps, stable-observation hysteresis, spine height,
right-EE position/quaternion, measured and commanded gripper state, and explicit
read-only camera evidence for pad centering, lift, target centering, and support.
The pre-grasp gate uses the audited quaternion cluster, local-Y vertical-axis
alignment, and position cluster. A simulator-time dwell timeout stops rather
than guessing progress.

The manager returns only the current task string. It has no base, spine, arm,
or gripper target interface and contains no IK. It is intentionally not wired
into `live/runner.py` because the offline model gate failed. A camera evidence
producer is also not implemented in this result; defining the phase-manager
contract before live integration was the requested boundary.

## Cost and rollback

The estimated cost for one formal training run was 21--30 minutes and 10--15 GB
VRAM; the valid run took 21:24 and 12.87 GB. Checkpoint load and the complete
offline gate added several minutes. The quarantined prompt-order attempt also
consumed one training run before the evidence review detected its invalid
mapping. No simulator time was consumed.

Rollback is immediate: ignore the new V4 run root and continue using the
already-pushed `71fb463` runner with the matching V1 or V2 checkpoint/dataset
pair. V1/V2/V3 checkpoints, the raw dataset, the V2 derived view, camera/state
capture, action publication, fixed-base staging, and simulator-clock timing
were not modified.

## Decision and next action

Do not run V4 in GUI and do not add an IK override. The next model iteration
should not simply add more steps to this checkpoint. The evidence instead
supports either stronger prompt/action separation at the orient landmark or
closed-loop recovery demonstrations that bring the right wrist back toward the
vertical pre-grasp manifold. Any V5 proposal should reuse this exact held-out
gate before simulator execution.
