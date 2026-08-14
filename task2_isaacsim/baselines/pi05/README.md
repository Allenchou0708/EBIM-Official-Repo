# Task 2 PI0.5 submission baseline

This directory contains the code-only Task 2 PI0.5 training and submission
runner. Datasets, caches, checkpoints, output, and evidence remain under the
host path configured by `TASK2_PI05_ROOT` and are never added to Git.

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

Use three terminals for a GUI run:

```bash
# Terminal 1
./run_pi05.sh sim-up --gui

# Terminal 2
./run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint /path/to/checkpoints/030000/pretrained_model \
  --max-actions 600

# Terminal 3
./run_pi05.sh evaluate
```

`run-task` requests a scene reset, follows the fixed base route to
`(2.10, 3.05, -1.571)`, returns the spine to the demonstrated near-zero
starting height, checks the evaluation camera, and then starts inference.
After manipulation begins, effective base output is always zero. Action 19 is
clamped to the demonstrated `0.0–0.6 m` range and published to the existing
`/isaac/spine_target` bridge interface, so PI0.5 controls the spine together
with both arms and grippers. The runner creates no base publisher.

Frame and state age use the host-monotonic clock. Cross-camera skew uses the
ROS image-header simulator timestamps, so transport/callback delay is not
misreported as capture-time misalignment. The default `hard5` mode executes
only chunk indices 0--4, matching checkpoint `n_action_steps: 5`, then holds
the last legal absolute target while the next fresh observation is inferred.
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
