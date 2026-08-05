# Task 2 PI0.5 baseline

This directory is the code-only boundary between the official EBiM Task 2
dataset contract and LeRobot PI0.5. Dataset frames, model weights, checkpoints,
Hub tokens, caches, and full logs are not part of the competition repository.

## Reproducible source boundary

- LeRobot version: `v0.6.0`
- LeRobot commit: `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- base policy: `lerobot/pi05_base`
- base policy revision: `338b5c22c12dbdd0d2ab19046802de2eb7696a6b`
- policy cameras: head, left wrist, right wrist
- evaluation-only inputs: `eval_camera` and `task2_extras/**`
- state/action contract: official 37-D state and 20-D action; PI0.5 boundary is 32-D
- default action representation: Task 2 mapped relative actions

The OCI image clones the pinned LeRobot source and applies
`patches/lerobot-v0.6.0-task2-relative-map.patch`. The patch is backward
compatible: ordinary LeRobot policies keep the sequential `action[i] - state[i]`
behavior when no mapping is provided.

## Training profiles

`profiles/smoke_expert.yaml` is the disposable RTX 5090 engineering profile. It
uses bfloat16, gradient checkpointing, and `train_expert_only=true`. It verifies
data loading, loss, backward, optimizer, checkpoint/resume, and inference, but
is not the formal baseline.

`profiles/full_finetune.yaml` is the formal single-GPU profile. It uses:

```yaml
policy:
  train_expert_only: false
  freeze_vision_encoder: false
  gradient_checkpointing: true
  dtype: bfloat16
```

The first supported release target is one A100 80 GB, H100 80 GB, or an
equivalent GPU. `doctor --profile full` stops below 70 GiB reported device
memory. It never silently switches to LoRA or expert-only training. Multi-GPU
launch arguments may be passed to LeRobot/Accelerate as a later experiment,
but multi-node training is not a release gate.

## Task 2 relative actions

Task 2 action and state indices are not aligned. The mapping is therefore
explicit and shared by preprocessing, postprocessing, statistics, profiles,
tests, and the run manifest.

| Action indices | State reference | Representation |
|---|---|---|
| base `vx/vy/wz`, 0–2 | none | original velocity command |
| left joints, 3–9 | state 14–20 | target minus current joint |
| right joints, 10–16 | state 21–27 | target minus current joint |
| grippers, 17–18 | none | absolute open fraction |
| spine, 19 | state 28 | target minus current height |

The raw dataset is never edited. `relative_dataset.py` creates a derived view,
hard-linking files when possible and copying otherwise, then replaces only the
view's `meta/stats.json`. Its manifest records the source checksum, relative
stats checksum, selected training episodes, mapping, chunk size, and whether a
file was linked or copied. Held-out episodes must not be included when these
statistics are calculated.

## Loss contract

`loss_parity.py` deterministically verifies that both implementations use the
flow target `noise - action`, records per-dimension MSE, and reports the
difference between a 20-D and padded 32-D reduction. The release keeps
LeRobot's native loss over the 20 real Task 2 dimensions. The padded 12-D mode
is evidence only and is not used for training.

```bash
python3 -m task2_isaacsim.baselines.pi05.loss_parity \
  --output /scratch1/2026_ebim/allen_task2_pi05/evidence/loss_parity.json
```

## Build and publish the image

Build from the repository root:

```bash
docker build \
  -f task2_isaacsim/baselines/pi05/docker/Dockerfile \
  -t ghcr.io/allenchou0708/ebim-task2-pi05:v0.6.0-task2 \
  .
```

The workflow `.github/workflows/task2-pi05-image.yaml` publishes both a Git-SHA
tag and `v0.6.0-task2` to GHCR. After its first successful run, make the package
public in the GitHub package settings. Always deploy by digest rather than a
mutable tag:

```bash
docker pull ghcr.io/allenchou0708/ebim-task2-pi05:v0.6.0-task2
docker image inspect \
  ghcr.io/allenchou0708/ebim-task2-pi05:v0.6.0-task2 \
  --format '{{index .RepoDigests 0}}'
```

The image contains code and dependencies only. `.dockerignore` excludes local
datasets, checkpoints, model weights, outputs, and caches.

## Docker run contract

Keep large data on the HDD and mount it at runtime:

```bash
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export PI05_IMAGE=ghcr.io/allenchou0708/ebim-task2-pi05@sha256:REPLACE_ME

mkdir -p \
  "$TASK2_PI05_ROOT/cache" \
  "$TASK2_PI05_ROOT/datasets" \
  "$TASK2_PI05_ROOT/outputs"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/ebim-home \
  -e HF_HOME=/cache/huggingface \
  -e EBIM_PI05_IMAGE_DIGEST="${PI05_IMAGE#*@}" \
  -v "$TASK2_PI05_ROOT/datasets:/data/dataset:ro" \
  -v "$TASK2_PI05_ROOT/outputs:/data/output" \
  -v "$TASK2_PI05_ROOT/cache:/cache" \
  "$PI05_IMAGE" doctor --profile smoke
```

Use a read token with the gated PaliGemma license already accepted. Run
`hf auth login` against the mounted `HF_HOME`; never pass a token as an image
build argument or commit it to Git.

Mount the entire writable cache root, not only `cache/huggingface`. PyTorch and
XDG caches are sibling directories under `/cache`; a host UID cannot create
them inside the root-owned image filesystem.

The entrypoint creates a temporary passwd/group view for arbitrary host UIDs,
so the earlier `getpwuid(): uid not found` failure is not reintroduced.

## Expert-only smoke and resume

For a failed-only engineering dataset, the opt-in is deliberately explicit:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e EBIM_PI05_IMAGE_DIGEST="${PI05_IMAGE#*@}" \
  -e HOME=/tmp/ebim-home \
  -e HF_HOME=/cache/huggingface \
  -v "$TASK2_PI05_ROOT/datasets/task2_smoke:/data/dataset:ro" \
  -v "$TASK2_PI05_ROOT/outputs:/data/output" \
  -v "$TASK2_PI05_ROOT/cache:/cache" \
  "$PI05_IMAGE" train \
    --profile smoke \
    --dataset-root /data/dataset \
    --output-dir /data/output/smoke_001 \
    --episodes 0,1 \
    --allow-unsuccessful-smoke-data \
    --execute
```

The wrapper refuses `success=false` data for the full profile. It also protects
the full/expert flag, vision freeze flag, relative mapping, model dimensions,
Hub upload flag, dataset path, and output path from CLI overrides.

Resume is a separate gate:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e EBIM_PI05_IMAGE_DIGEST="${PI05_IMAGE#*@}" \
  -v "$TASK2_PI05_ROOT/outputs:/data/output" \
  -v "$TASK2_PI05_ROOT/cache:/cache" \
  "$PI05_IMAGE" resume \
    --checkpoint /data/output/smoke_001/training/checkpoints/000001 \
    --output-dir /data/output/smoke_001/training \
    --steps 2 \
    --execute
```

Executable train, resume, offline, and shadow runs reject mutable or missing
image identifiers; `EBIM_PI05_IMAGE_DIGEST` must be `sha256:<64 hex>`. Outputs
inside the Git checkout are also rejected. Training records profile and patch
checksums, GPU, mapping, relative stats checksum, command, parameter counts,
finite loss evidence, and checkpoint hashes in `run_manifest.json`; resume
records source and resumed checkpoint hashes in `resume_manifest.json`.

## Full fine-tune gate

On the 80 GB server, first run:

```bash
docker run --rm --gpus all ... "$PI05_IMAGE" doctor --profile full
```

Then use `train --profile full` with successful training episodes. A successful
process is still rejected if the log does not report an approximately 4B total
model and at least 3.5B trainable parameters. Record peak memory and step time
from the full log before starting the long run.

## Ten-episode engineering gate

The deterministic split command returns `0..7` train and `8..9` held-out:

```bash
docker run --rm "$PI05_IMAGE" split --total-episodes 10 --held-out 2
```

All ten episodes may be `success=false` only for framework verification:

1. Overfit two episodes with the expert profile.
2. Train briefly on episodes `0..7` with the expert profile.
3. Keep episodes `8,9` out of relative stats and tuning.
4. Run offline inference on the held-out view.

```bash
docker run --rm --gpus all ... "$PI05_IMAGE" offline-inference \
  --checkpoint /data/output/smoke_8train/training/checkpoints/STEP/pretrained_model \
  --dataset-root /data/output/smoke_8train/relative_dataset \
  --episodes 8,9 \
  --output /data/output/smoke_8train/held_out_shadow.json
```

Offline/shadow inference validates finite 20-D output, mapped
relative-to-absolute conversion, gripper range, and reports non-zero base
commands. It never imports ROS or publishes a command. This gate proves only
that training and inference are wired correctly; it is not task-success or
baseline evidence.

## Apptainer

Use the same immutable GHCR digest:

```bash
apptainer pull task2-pi05.sif \
  docker://ghcr.io/allenchou0708/ebim-task2-pi05@sha256:REPLACE_ME

apptainer exec --nv \
  --bind "$TASK2_PI05_ROOT/datasets:/data/dataset:ro" \
  --bind "$TASK2_PI05_ROOT/outputs:/data/output" \
  --bind "$TASK2_PI05_ROOT/cache:/cache" \
  task2-pi05.sif python -m \
    task2_isaacsim.baselines.pi05.portable doctor --profile full
```

Different machines may run independent experiments with the same digest. The
first release does not claim synchronized multi-node training support.
