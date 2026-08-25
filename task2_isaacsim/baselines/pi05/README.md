# Task 2 PI0.5 submission baseline

This directory contains the code-only Task 2 PI0.5 training and submission
runner. Datasets, caches, checkpoints, output, and evidence remain under the
host path configured by `TASK2_PI05_ROOT` and are never added to Git.

The separate, organizer-permitted Phase I ground-truth controller is described
in [GROUND_TRUTH_PHASE1.md](GROUND_TRUTH_PHASE1.md).  Its supporting policy
report and clean-evaluation commands are in
[PHASE1_POLICY_REPORT.md](PHASE1_POLICY_REPORT.md) and
[PHASE1_SUBMISSION_RUNBOOK.md](PHASE1_SUBMISSION_RUNBOOK.md).

## Setup and model inputs

Copy `.env.pi05.example` to `.env.pi05` and set the local paths and images.
The file is ignored by Git. Do not put a Hugging Face token in it:

```bash
TASK2_PI05_ROOT=/absolute/path/to/task2-pi05-runtime
PI05_TRAIN_IMAGE=ebim-task2-pi05:200-submit-20260812
PI05_LIVE_IMAGE=ebim-task2-pi05-submit:local
PI05_CHECKPOINT=/absolute/path/to/checkpoints/030000/pretrained_model
PI05_RELATIVE_DATASET=/absolute/path/to/relative_dataset
```

The original V1 formal config pins:

- dataset `hermanprawiro/task2_fixpos_200` at revision
  `46ab41f16fe836ee8ca791c7afaade44783eefe6`;
- base policy `lerobot/pi05_base` at revision
  `338b5c22c12dbdd0d2ab19046802de2eb7696a6b`;
- an episode-level seed-`20260812` split of 180 train and 20 held-out
  episodes;
- expert-only, frozen-vision, bfloat16 training for 30000 steps with
  checkpoints every 5000 steps.

The Hugging Face cache mounted under `${TASK2_PI05_ROOT}/cache` must already
have access to the PaliGemma-gated model files. Never put an access token in a
config, command, manifest, image, or Git file.

## Dataset and training

Run from this directory:

```bash
./run_pi05.sh doctor
./run_pi05.sh dataset --config configs/task2_fixpos_200_expert.yaml
./run_pi05.sh train --config configs/task2_fixpos_200_expert.yaml \
  --run task2_200_30k_v1
```

CLI path and run-name arguments override YAML defaults. Dataset QA checks the
LeRobot schema, 20-D action and 37-D state, all four readable camera streams,
episode/frame consistency, finite numeric data, action range and codec, plus
base/spine variance. Optional success, orientation, and drop metadata is
reported when present but is not required for technical eligibility.
`task2_extras` is QA-only and is never a policy input.

Relative action statistics use only the 180 training episodes. V1 keeps base
velocity and grippers absolute while arm joints and spine use the explicit
20-D action-to-37-D state mapping in `contract.py`.

The V3 calibration profile initializes from the existing V1 30k task
checkpoint, preserves its arm/gripper behavior with a low learning rate, and
changes only the learned spine representation to the V2 absolute target
contract. It runs 3000 phase-balanced steps and saves one final checkpoint:

```bash
PI05_V3_INIT_CHECKPOINT=/path/to/v1/checkpoints/030000/pretrained_model \
  ./run_pi05.sh train --config configs/task2_fixpos_200_v3.yaml \
  --run task2_pi05_v3_from_v1_3k
```

Phase-specific language is intentionally not used in this calibration run.
Adding it would change both the training labels and live phase-transition
contract, confounding the arm/spine representation test.

PI0.5 V2 keeps the arms relative but learns spine as the absolute command used
by the simulator. It also samples six physical event phases instead of uniform
frames and stores only checkpoints 6k and 12k:

```bash
./run_pi05.sh train --config configs/task2_fixpos_200_v2.yaml \
  --run task2_pi05_v2_12k
```

The V2 phase manifest is derived from recorded spine state, gripper commands,
and thermal-pad motion. Held-out episodes are excluded from every sampler
group. The selected relative-action mapping is serialized in the checkpoint;
the live and offline processors retain V1 decoding for older checkpoints.

V4 is a bounded continuation from the V2 6k checkpoint using the audited
seven-prompt dataset view. It keeps the V2 action contract and stores one final
3k checkpoint:

```bash
./run_pi05.sh train-v4 --run task2_pi05_v4_from_v2_3k
./run_pi05.sh gate-v4
```

`train-v4` validates the immutable split, seven prompt strings, V2 checkpoint
mapping, and dataset metadata before training. The V4-only collate resolves
dataset task indices to prompt strings before PI0.5 tokenization. `gate-v4`
runs all seven prompts at seven landmarks for every held-out episode, scores
only hard5 actions 0--4, writes no ROS command, and exits `3` for a measured
NO-GO. The 2026-08-14 V4 checkpoint failed orient-to-pregrasp direction and
correct-prompt discriminability, so it must not be run in GUI; see
`TASK2_PI05_V4_SEVEN_PROMPT_OFFLINE_GATE_LAB_RESULT_2026-08-14.md`.

Before a GUI run, one checkpoint shadow can exercise the real observation,
postprocessing, action-bound, base-isolation, spine, and time-alignment
contracts without publishing ROS commands:

```bash
docker run --rm --gpus all --ipc=host \
  --entrypoint python \
  -v "${PI05_CHECKPOINT}:/data/checkpoint:ro" \
  -v "${PI05_RELATIVE_DATASET}:/data/dataset:ro" \
  -v "${TASK2_PI05_ROOT}/evidence:/data/evidence" \
  "${PI05_LIVE_IMAGE}" \
  -m task2_isaacsim.baselines.pi05.live.policy_smoke \
  --checkpoint /data/checkpoint --dataset-root /data/dataset \
  --output /data/evidence/policy_shadow.json
```

## Simulator and evaluator terminals

The live path uses a dataset-derived ROS staging trajectory before every
policy rollout. It restores the demonstrated base, both arms, grippers, and
spine state, then verifies the right end effector relative to the thermal pad.
Restoring the complete recorded state is intentional: moving only the right
arm would still leave PI0.5 with an out-of-distribution left-arm/spine state.
No guessed inverse-kinematics target is used.

To inspect this initialization without running PI0.5, start the simulator and
run the standalone staging command in another terminal:

```bash
./run_pi05.sh sim-up --gui
./run_pi05.sh stage-init
```

Success is recorded as `success: true` with reason
`stable_dataset_camera_ready` in the generated `stage_init_manifest.json`.
Stop other teleoperation/policy publishers first; staging intentionally fails
when another process is publishing arm commands.

Use three terminals for a GUI run:

```bash
# Terminal 1
./run_pi05.sh sim-up --gui

# Terminal 2
./run_pi05.sh run-task \
  --hybrid-gt-pregrasp \
  --runtime-mode hard5 \
  --checkpoint /path/to/checkpoints/030000/pretrained_model \
  --dataset-root /path/to/relative_dataset \
  --confirm-right-wrist-pad-visible \
  --max-actions 600

# Terminal 3
./run_pi05.sh evaluate
```

The formal Phase II hybrid handoff is the complete recorded state from expert
episode 19, frame 399. Staging restores base, spine, both seven-joint arms, and
both open grippers. In particular, the left arm is not left at its home pose:
although the right arm performs the grasp, the left joint state and wrist image
are PI0.5 inputs and therefore affect its inferred task phase. The runner keeps
all policy publishers inactive while the checkpoint loads, reacquires a fresh
observation, and fails closed unless all 14 arm joints remain within `0.03 rad`
of the staging manifest and both grippers remain open. From the first accepted
decision onward, PI0.5 exclusively owns both arms, both grippers, and the spine;
the GT controller and pose/RMPflow paths are inactive.

State equivalence at the handoff is not trajectory equivalence. The staging
controller reaches frame 399 with a bounded base -> spine -> dual-arm sequence;
it does not replay every episode-19 action from frames 0--399. This preserves a
safe, auditable ownership boundary while matching every policy-observed robot
state at takeover. Use the full Phase-I controller, not this hybrid entry path,
when literal expert-trajectory replay is the experiment being measured.

The corrected 2026-08-25 V2 30k rollout is diagnostic evidence, not a passing
result. With the full dual-arm GT handoff, the policy initially matched the
expert transition (left gripper open, right gripper closed) and moved the right
arm forward, but began reopening the right gripper at decision 38 and kept it
open through most of the remaining rollout. The official evaluator reported
IoU `0.0000` with the pad still at the source. This isolates the remaining
failure to learned grasp/phase behavior rather than a partial-state handoff;
do not treat the V2 checkpoint as submission-ready without new training and
fresh multi-run evaluation.

`sim-up` mirrors Isaac/Kit output to the terminal and to a timestamped
`isaac_gui_*.log` below
`$TASK2_PI05_ROOT/evidence/task2_200_submit_20260812/launcher/`. The final log
line records the launcher exit code, so a bridge-start or Kit crash remains
diagnosable after the GUI closes. The Isaac ROS bridge uses UDPv4 like the
helper/live containers; disabling its ineffective cross-container Fast DDS
shared-memory transport prevents stale DDS files from filling the long-lived
container's `/dev/shm` and causing a Kit `Bus error`.

`run-task --hybrid-gt-pregrasp` requests a scene reset, runs the measured base
alignment, restores the demonstrated spine and full dual-arm frame-399 state,
and then starts the runner. The runner leaves its publishers inactive while
loading the checkpoint, discards staging-time observations, verifies that the
complete handoff state did not drift, and requires a fresh, settled
camera/state tuple before warmup or inference.
The route's command pulses, braking pauses, and settle duration use
`/isaac/clock`; its bounded process timeout remains on host monotonic time.
Short `0.05 s` correction pulses (`0.10 s` for forward/back) and `0.10 s`
braking pauses keep the base trajectory consistent when GUI real-time factor
changes without amplifying the previous wall-time pulse lengths.
After manipulation begins, effective base output is always zero. Action 19 is
clamped to the demonstrated `0.0–0.6 m` range and published to the existing
`/isaac/spine_target` bridge interface, so PI0.5 controls the spine together
with both arms and grippers. The runner creates no base publisher.

Frame and per-stream state age use the host-monotonic clock. Capture alignment
uses ROS simulator timestamps from all three images, full joint state, odom,
and both EE poses; an observation is rejected if any state stream is missing
or the combined capture skew exceeds `0.10 s`. Transport/callback delay is
therefore not misreported as capture-time misalignment. Arm and gripper
`JointState` commands carry the current `/isaac/clock` stamp. The default
`hard5` mode executes only chunk indices 0--4, matching checkpoint
`n_action_steps: 5`, then holds the last legal absolute target while the next
fresh observation is inferred.
All policy and hold publications are paced by `/isaac/clock`, so low GUI
real-time factor cannot consume the trajectory too quickly. The optional
`legacy` mode retains asynchronous full-chunk replacement for diagnostics.
The manifest records policy indices, hold publications, both capture and
arrival skew, capture-to-ready latency, and measured spine trajectory. Reset,
freshness, action bounds, command contention, and operator interrupt stop
publication safely.

Stop with `Ctrl-C` in the runner terminal or close all services with:

```bash
./run_pi05.sh down
```

Training output contains the relative dataset view, `train.log`, run manifest,
and checkpoints. Runtime image contents never include the training dataset;
`run-task` accepts a host checkpoint path as its single model entry point.
