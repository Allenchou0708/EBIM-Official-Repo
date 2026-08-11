# Task 2 PI0.5 action-expert baseline

This directory is the code-only boundary between EBiM Task 2 and LeRobot
PI0.5. Datasets, model weights, checkpoints, Hub tokens, caches, and full logs
stay outside the competition repository.

The submission baseline remains on LeRobot through the 2026-08-15 deadline.
Migration to Physical Intelligence `openpi` and full fine-tuning are separate,
post-baseline experiments.

## Reproducible source boundary

- official benchmark base: `0004645a4b8843f0e04a5ca531fce0598e058910`
- LeRobot: `v0.6.0@30da8e687a6dfc617fcd94afc367ac7071c376ce`
- base policy: `lerobot/pi05_base`
- base policy revision: `338b5c22c12dbdd0d2ab19046802de2eb7696a6b`
- organizer dataset: `hermanprawiro/task2_fixpos_v1`
- dataset revision: `1a7253a776b9a05d866da297789c456c2f0ed9f8`
- policy cameras: head, left wrist, and right wrist
- evaluation-only data: `eval_camera` and `task2_extras/**`
- official boundary: 37-D state and 20-D action; PI0.5 action boundary: 32-D

As verified on 2026-08-07, this pinned dataset revision contains no
`README.md`, its Hugging Face `card_data` is null, and its tags do not declare
a license. Do not treat the repository's expected `apache-2.0` value as
license evidence. The team confirmed that the dataset was provided by the
EBiM organizers for Task 2 competition use, so formal runs require a separate
checksum-backed organizer-use attestation. That attestation authorizes only
the recorded competition scope; it does not invent a dataset license.

The OCI image clones the pinned LeRobot source and applies
`patches/lerobot-v0.6.0-task2-relative-map.patch`. LeRobot's original
sequential relative-action behavior remains unchanged when a policy does not
provide an explicit mapping.

## Profiles

`profiles/smoke_expert.yaml` is disposable engineering evidence. It permits
failed episodes only with the explicit `--allow-unsuccessful-smoke-data` flag.

`profiles/expert_finetune.yaml` is the formal RTX 5090 baseline:

```yaml
policy:
  train_expert_only: true
  freeze_vision_encoder: true
  gradient_checkpointing: true
  dtype: bfloat16
  use_relative_actions: true
  chunk_size: 50
  n_action_steps: 5
steps: 6000
batch_size: 1
save_freq: 500
seed: 1000
```

The wrapper records this file as mode `expert_finetune` with `formal=true`;
those wrapper-only fields are deliberately not inserted into LeRobot's YAML
schema.

Formal action-expert training refuses failed episodes, held-out episodes, a
mutable dataset, or a dataset audit that does not authorize training.

`profiles/full_finetune.yaml` is retained but paused. It keeps both the VLM and
vision encoder trainable and fails below 70 GiB reported GPU memory. There is
no automatic LoRA or expert-only fallback.

## Relative-action and loss contract

Task 2 action and state vectors are not index-aligned:

| Action | State reference | Representation |
|---|---|---|
| base `vx/vy/wz`, 0-2 | none | absolute velocity command |
| left joints, 3-9 | state 14-20 | target minus current joint |
| right joints, 10-16 | state 21-27 | target minus current joint |
| grippers, 17-18 | none | absolute open fraction |
| spine, 19 | state 28 | target minus current height |

`relative_dataset.py` hard-links or copies the raw dataset to a derived view
and replaces only `meta/stats.json`. Relative statistics use the train split
only. The raw dataset is never edited, and the manifest records both source
and derived checksums.

`loss_parity.py` verifies the shared flow target `noise - action` and records
the difference between 20-D and padded 32-D reductions. This release keeps
LeRobot's native loss over the 20 real Task 2 dimensions.

## Build and immutable image

Build from the repository root:

```bash
docker build \
  -f task2_isaacsim/baselines/pi05/docker/Dockerfile \
  -t ghcr.io/allenchou0708/ebim-task2-pi05:v0.6.0-task2 \
  .
```

`.github/workflows/task2-pi05-image.yaml` publishes a Git-SHA tag and the
mutable convenience tag. Experiments must use the resulting immutable digest:

```bash
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export PI05_IMAGE=ghcr.io/allenchou0708/ebim-task2-pi05@sha256:REPLACE_ME

mkdir -p \
  "$TASK2_PI05_ROOT/cache" \
  "$TASK2_PI05_ROOT/datasets" \
  "$TASK2_PI05_ROOT/outputs" \
  "$TASK2_PI05_ROOT/evidence"
```

The image contains code and dependencies only. Use a Hugging Face read token
with the PaliGemma terms accepted, login against the mounted cache, and never
pass the token as a build argument or write it into a manifest.

## Organizer dataset download and audit

Download the exact revision to the HDD. The command creates a source manifest
beside the dataset without reading or recording the token:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/ebim-home \
  -e HF_HOME=/cache/huggingface \
  -v "$TASK2_PI05_ROOT/datasets:/data/datasets" \
  -v "$TASK2_PI05_ROOT/cache:/cache" \
  "$PI05_IMAGE" download-organizer \
    --destination /data/datasets/task2_fixpos_v1_1a7253a
```

Audit schema, license, checksums, codecs, frame counts, numeric ranges,
success metadata, stale/encoder drops, orientation, IoU, and mobile-axis
variance:

```bash
export ORGANIZER_USE_ATTESTATION="$TASK2_PI05_ROOT/evidence/organizer_use_attestation.json"
export ACKNOWLEDGED_UTC="$(date --iso-8601=seconds)"

tee "$ORGANIZER_USE_ATTESTATION" >/dev/null <<EOF
{
  "schema_version": 1,
  "authorization_kind": "organizer_provided_competition_data",
  "dataset_repo_id": "hermanprawiro/task2_fixpos_v1",
  "dataset_revision": "1a7253a776b9a05d866da297789c456c2f0ed9f8",
  "scope": "ebim_task2_training_and_evaluation",
  "acknowledged_by": "Allen_HZL",
  "acknowledged_utc": "$ACKNOWLEDGED_UTC",
  "statement": "The team confirms this dataset was provided by the EBiM organizers for Task 2 competition training and evaluation."
}
EOF

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/ebim-home \
  -e HF_HOME=/cache/huggingface \
  -v "$TASK2_PI05_ROOT/cache:/cache" \
  -v "$TASK2_PI05_ROOT/datasets:/data/datasets:ro" \
  -v "$TASK2_PI05_ROOT/evidence:/data/evidence" \
  "$PI05_IMAGE" audit-dataset \
    --dataset-root /data/datasets/task2_fixpos_v1_1a7253a \
    --organizer-use-attestation /data/evidence/organizer_use_attestation.json \
    --output /data/evidence/task2_fixpos_v1_audit.json

export AUDIT="$TASK2_PI05_ROOT/evidence/task2_fixpos_v1_audit.json"
export TRAIN_EPISODES="$(jq -r '.split.train | join(",")' "$AUDIT")"
export HELD_OUT_EPISODES="$(jq -r '.split.held_out | join(",")' "$AUDIT")"
jq '{technical_audit_pass,audit_pass,license_evidence,organizer_use_attestation,dataset_use_authorized,formal_training_allowed,split,rollout_constraints}' "$AUDIT"
```

If all 22 episodes pass, the pinned SHA-256 ranking creates 18 train and four
held-out episodes using seed `20260806`. Otherwise, held-out contains
`max(2, ceil(20% eligible))`. Fewer than ten train or two held-out episodes is
smoke-only. Held-out episodes never enter relative statistics or training.

`technical_audit_pass=true` alone is not permission to train. Formal profiles
still require both `audit_pass=true` and `formal_training_allowed=true`, backed
by either verifiable license evidence from the exact pinned revision or the
validated organizer-use attestation above.

## Gates and formal training

First verify the new image:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/ebim-home \
  -e HF_HOME=/cache/huggingface \
  -e EBIM_PI05_IMAGE_DIGEST="${PI05_IMAGE#*@}" \
  -v "$TASK2_PI05_ROOT/cache:/cache" \
  "$PI05_IMAGE" doctor --profile expert
```

Use the audit train split for one-step and two-episode overfit gates. Add
`--allow-train-subset`; formal training otherwise requires the complete split:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/ebim-home \
  -e HF_HOME=/cache/huggingface \
  -e EBIM_PI05_IMAGE_DIGEST="${PI05_IMAGE#*@}" \
  -v "$TASK2_PI05_ROOT/datasets/task2_fixpos_v1_1a7253a:/data/dataset:ro" \
  -v "$TASK2_PI05_ROOT/outputs:/data/output" \
  -v "$TASK2_PI05_ROOT/evidence:/data/evidence:ro" \
  -v "$TASK2_PI05_ROOT/cache:/cache" \
  "$PI05_IMAGE" train \
    --profile expert \
    --dataset-root /data/dataset \
    --audit-report /data/evidence/task2_fixpos_v1_audit.json \
    --output-dir /data/output/organizer_expert_6k \
    --episodes "$TRAIN_EPISODES" \
    --execute
```

For the one-step gate, add `--allow-train-subset`, pass two train episodes,
and add these repeated overrides:

```text
--override=--steps=1
--override=--save_checkpoint=true
--override=--save_freq=1
```

For the two-episode overfit gate, use the same two episodes with
`--allow-train-subset --require-loss-improvement` and override `--steps=1000`.
The wrapper compares the initial and final loss windows and rejects a run whose
final mean did not decrease.

Resume is verified separately:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e EBIM_PI05_IMAGE_DIGEST="${PI05_IMAGE#*@}" \
  -v "$TASK2_PI05_ROOT/outputs:/data/output" \
  -v "$TASK2_PI05_ROOT/cache:/cache" \
  "$PI05_IMAGE" resume \
    --checkpoint /data/output/GATE/training/checkpoints/000001 \
    --output-dir /data/output/GATE/training \
    --steps 2 \
    --execute
```

Every executable manifest records the image digest, profile mode, trainable
parameter count, organizer revision and split, relative stats checksum, GPU,
command, finite loss evidence, and checkpoint hashes.

## Held-out checkpoint sweep and shadow inference

The sweep evaluates every `*/pretrained_model` checkpoint using fixed
held-out frames and seeds. It also runs one deterministic shadow command per
held-out episode, checking finite 20-D output, relative-to-absolute recovery,
FR3 joint bounds, gripper `[0,1]`, and replay reproducibility.

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/ebim-home \
  -e HF_HOME=/cache/huggingface \
  -e EBIM_PI05_IMAGE_DIGEST="${PI05_IMAGE#*@}" \
  -v "$TASK2_PI05_ROOT/outputs:/data/output" \
  -v "$TASK2_PI05_ROOT/evidence:/data/evidence:ro" \
  -v "$TASK2_PI05_ROOT/cache:/cache" \
  "$PI05_IMAGE" checkpoint-sweep \
    --checkpoints-root /data/output/organizer_expert_6k/training/checkpoints \
    --dataset-root /data/output/organizer_expert_6k/relative_dataset \
    --audit-report /data/evidence/task2_fixpos_v1_audit.json \
    --output /data/output/organizer_expert_6k/heldout_sweep.json
```

The lowest finite held-out loss wins. If loss from step 5500 to 6000 improves
by more than 2%, the report recommends resuming to step 12000.

Shadow mode never imports ROS or publishes a command. If audit variance shows
fixed base velocity, its effective output clamps `vx/vy/wz` to zero. A fixed
spine is held at the dataset median. Closed-loop publication requires a
separate reviewed runner and a five-reset gate of at least 3/5 with correct
orientation and IoU greater than zero.

## Live simulator runner

The reviewed live path is in `live/`. It derives a ROS Jazzy image from the
immutable training digest, subscribes only to the three robot cameras and the
official 37-D state sources, and loads the checkpoint's saved pre/post
processors. The postprocessor performs the relative-to-absolute inverse once;
the runner does not repeat that transform.

The default is shadow mode and creates no command publishers. Simulator
publication requires `--arm-simulator --confirm-fixed-staging` and is limited
to the official left/right arm and gripper topics. Effective base velocity is
always zero and the spine holds its current measured position; the runner has
no base or spine publisher. Every reset clears image/action state and requires
the configurable pose/settle gate again. Finite raw gripper outputs remain
absolute and the simulator adapter projects them to `[0,1]`; arm limits remain
hard stops. Three warm-up decisions precede steady-state timing, and full
50-step chunks refill in a background thread while the ROS control loop keeps
executing its existing queue.

Build and run from the repository root:

```bash
docker build \
  -f task2_isaacsim/baselines/pi05/live/docker/Dockerfile \
  -t ebim-task2-pi05-live:local .

task2_isaacsim/baselines/pi05/live/run_live_runner.sh \
  --base-target 2.10 3.05 -1.571 \
  --base-coordinate-frame dataset_odom_world_verified_against_room_scene \
  --confirm-fixed-staging \
  --max-decisions 5
```

The example target is valid only when the checksum-backed organizer extraction
report identifies state indices 31:34 as reset-relative odometry and the room
scene is at its reproducible reset pose. Do not treat it as a table/world
transform. Add `--arm-simulator` only after a passing shadow run, operator
staging, teleop shutdown, and zero command-topic publisher counts.

Before the runner, `live/fixed_stage_base.py` uses the room-scene odometry to
execute `BACK -> stop -> right strafe -> stop -> correction -> STOP/settle` on
`/pedal/state`. It must run with Isaac Sim's bundled ROS environment and with
no other pedal publisher. The operator watches the GUI and retains emergency
stop authority, but does not steer the base. Spine height must be checked
against state index 28 in the train-only extraction report; the live runner
only holds the measured value and never publishes spine.

## VR dataset and Apptainer

Teammate VR data is audited independently. It may enter `mixed_v1` only if the
20-D/37-D contract, camera keys, 30 FPS, mapped relative actions, and episode
QA are compatible. It never changes the organizer held-out split.

The same immutable image digest can run through Apptainer:

```bash
apptainer pull task2-pi05.sif \
  docker://ghcr.io/allenchou0708/ebim-task2-pi05@sha256:REPLACE_ME
```
