# Task 2 PI0.5 preload and pre-grasp pose audit — 2026-08-14

## Outcome

Two distinct blockers were isolated and addressed without changing the base
controller, hard5 scheduler, camera contract, raw organizer dataset, or V1/V3
checkpoint semantics.

1. A standalone V3 checkpoint construction took `143.529 s`; imports took only
   `0.027 s`, while steady-state V3 inference was previously about `0.52 s`.
   The launcher had staged the base before this synchronous load. The runner
   now loads the policy first, invokes the existing fixed-base stager only after
   loading, discards staging-time observations, and requires a fresh settled
   tuple before warmup or inference.
2. A 199-episode state/pose audit confirms a distinct right-arm vertical
   pre-grasp transition that the six-phase V3 sampler did not identify. A
   non-destructive seven-prompt dataset view now separates `approach` from
   `orient_pregrasp` and preserves the original train/held-out split.

The source-level and metadata gates pass. The corrected load order also passed
a live zero-publication simulator shadow with aligned camera/state stamps. No
V4 training was started, so the arm-policy correction is data-ready but not yet
a task-success result.

## Repository and runtime

- Repository/branch/base: `submit@18cacbbb3e089f0283e7b68c989b1637515fcc86`.
- Runtime image used for load timing and dataset tools:
  `ebim-task2-pi05-submit:v3-hard5-20260814`.
- V3 checkpoint:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v3_from_v1_3k/training/checkpoints/003000/pretrained_model`.
- Checkpoint weights: one `model.safetensors` file of about `9.35 GB`.
- Focused contract/live tests after the clock/crash follow-up: `64` passed.
- Python compile, shell syntax, and `git diff --check`: passed.
- Ruff was not installed in the verified runtime image; no package was added
  solely for linting.

## Model-load correction

The previous order was:

```text
scene reset -> base stage -> spine stage -> camera preflight -> load PI0.5
```

The corrected order is:

```text
scene reset -> spine stage -> camera preflight -> load PI0.5
            -> fixed base stage -> discard staging observations
            -> fresh readiness gate -> warmup -> inference/publication
```

The VLA runner still creates no base publisher. Arm/spine publishers do not
publish before readiness, and a base command after the manipulation latch is
still a stop condition. Manifest schema 6 adds policy-load time, post-load
base-stage time/order, the base-stage report path, and a separate count for
base inputs after the manipulation latch.

Transformers parallel loading was not enabled blindly. The checkpoint is one
safetensors shard, while the documented parallel loader helps by loading
multiple weight files concurrently. The durable improvement here is sequencing
and keeping one loaded policy process for the attempt, not an unmeasured loader
flag or a new policy-server dependency.

## Dataset evidence for the right-arm transition

Audit:
`/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_pregrasp_20260814/pregrasp_pose_audit.json`.

- Auditable episodes: `199` (`179` train, `20` held out); episode `132` was
  already excluded by the physical-event manifest.
- Median spine rise to `spine_high`: `+0.44127 m`.
- Median left/right EE world-Z change over that rise:
  `+0.02207/+0.01958 m`. The demonstrations compensate almost the complete
  spine rise.
- Median right-EE rotation from `spine_high` to pre-close: `100.43 deg`.
- Median right-arm joint-space displacement over the same transition:
  `2.715 rad`.
- Median right-joint deltas J1--J7:
  `[+1.2175, -1.2868, -1.2542, +0.2907, -0.4215, +1.4015, -0.5247] rad`.
- The right EE enters the stable pre-close orientation a median `157` frames,
  or about `5.23 s`, before the close event.
- At pre-close, right-EE local Y is the vertical tool axis; its median absolute
  alignment with world Z is `0.99466`.
- Global pre-close quaternion cluster, xyzw:
  `[-0.02928, 0.73283, -0.67966, -0.01303]`; its q50/q90 angular dispersion is
  only `3.12/5.18 deg`.
- Median pre-close right-EE world position:
  `[1.75041, 2.14087, 0.87220] m`.

Episode 9 reproduces the same structure: `spine_high=184`, orientation entry
`302`, right close `464`, and pad motion `492`. Right-wrist frames show the
view changing from the table edge, through the RAM-board approach, to a close
view of the pad under a vertically oriented gripper. This supports the operator
observation with both proprioceptive and visual evidence.

## Seven-prompt derived dataset

Derived view:
`/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v4_pregrasp/phase_conditioned_dataset`.

The exact prompts are:

1. `startup_rise`: raise the spine while keeping both wrists at table height;
2. `approach`: move the open right gripper above the thermal pad;
3. `orient_pregrasp`: rotate the right gripper vertical and center the thermal
   pad in the right wrist view;
4. `grasp_acquisition`: close on the pad and lift it;
5. `lift_transfer`: carry the pad to the target RAM board;
6. `lower_place`: lower the pad onto the target;
7. `release_retreat`: open the right gripper and retreat safely.

The builder rewrites only data task indices and task/episode metadata. Videos,
numeric observations/actions, checkpoint normalization statistics, and the raw
dataset remain unchanged. LeRobot metadata loads `174719` frames, `200`
episodes, and eight task strings (seven phase prompts plus the original prompt
for excluded episode 132). No held-out episode appears in any training sampler
group. Episode 9 boundary checks change prompt exactly at frames
`184`, `302`, and `464` for approach, orientation, and close respectively.

## Recommended model experiment

In the next session, review and run one bounded V4 continuation from the V2
checkpoint using this derived view and its phase-conditioned manifest. Keep
relative arms, absolute spine, hard5, expert-only data, and one final
checkpoint. The offline GO gate should require:

- startup prompt retains demonstrated wrist-height compensation;
- approach keeps the right gripper open;
- `orient_pregrasp` produces the demonstrated approximately 100-degree pose
  transition and converges toward the tight pre-close quaternion/position
  cluster without saturating right joint 4;
- grasp prompt predicts a real right-gripper close on held-out landmarks;
- prompt-discriminability on the same observation is explicit;
- release prompt recovers an open gripper.

Only after that gate should a live phase manager use observable spine, right-EE
pose/quaternion, gripper state/action, and camera evidence to switch these exact
training prompts. If the model still leaves the wrist-height manifold, the next
experiment should collect recovery demonstrations or use DAgger-style
closed-loop corrections. Do not add guessed arm IK overrides: they would mask
the policy failure and conflict with the learned action contract.

## Known limitations

- The new load order passed a live zero-publication shadow at
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260814_160449/live_runner_manifest.json`:
  `valid=true`, `commands=0`, camera skew `0.033333336 s`, full camera/state
  capture skew `0.066666671 s`, and capture-to-ready simulator time
  `0.316666683 s`.
- The pre-grasp detector describes the demonstrated pose cluster; it does not
  prove that the current V1 or V2 checkpoint can recover to it closed-loop.
- V4 has not been trained, so the seven-prompt dataset is implementation
  evidence, not grasp-success evidence.
