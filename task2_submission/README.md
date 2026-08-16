# EBiM 2026 Task 2 submission

This submission runs a PI0.5 policy for Task 2 (thermal-pad placement) in the
official Isaac Sim 5.1.0 environment. The submitted policy is the V1 checkpoint
trained for 30,000 steps. The runtime executes the checkpoint's five-action
receding horizon (`hard5`) and keeps the mobile base output disabled during
manipulation.

## Required files

The source repository must contain:

```text
task2_submission/Dockerfile
task2_submission/entrypoint.sh
task2_isaacsim/baselines/pi05/
```

Download the public model repository and place its complete contents at
`checkpoints/030000/pretrained_model`. Do not copy only `model.safetensors`;
the configuration and processor files stored beside it are also required.

The checkpoint used during development was:

```text
/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_200_30k_v1/training/checkpoints/030000/pretrained_model
```


model link on the hugging face : 



## Build

Run from the root of the official repository:

```bash
docker build \
  -f task2_submission/Dockerfile \
  -t ebim-task2-pi05:submission .
```

Verify the image and mounted checkpoint:

```bash
docker run --rm \
  -v "$PWD/checkpoints/030000/pretrained_model:/data/checkpoint:ro" \
  ebim-task2-pi05:submission health
```

## Run inference

The policy container communicates with the official simulator through ROS 2.
The checkpoint, matching relative dataset, output directory, and Hugging Face
cache are mounted from the host:

```bash
docker run --rm --gpus all --network host --ipc=host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  -v "$PWD/checkpoints/030000/pretrained_model:/data/checkpoint:ro" \
  -v "/absolute/path/to/relative_dataset:/data/dataset:ro" \
  -v "$PWD/output:/data/output" \
  -v "$HOME/.cache/huggingface:/cache/huggingface:ro" \
  ebim-task2-pi05:submission run-task \
  --checkpoint /data/checkpoint \
  --dataset-root /data/dataset \
  --dataset-repo-id hermanprawiro/task2_fixpos_200 \
  --output-dir /data/output \
  --runtime-mode hard5 \
  --arm-simulator \
  --confirm-fixed-base-staging \
  --max-decisions 120 \
  --max-publish-actions 600 \
  --max-duration-s 300
```

For the team's full scene-reset, camera-preflight, base-staging, and evaluator
workflow, use `task2_isaacsim/baselines/pi05/run_pi05.sh run-task` as documented
in `task2_isaacsim/baselines/pi05/README.md`.

## Model selection

The submitted V1 30k checkpoint is selected because the development history
shows both shadow and command-publishing inference runs with this exact model.
Experimental V3 and V4 checkpoints are not submission models: V3 did not
complete the grasp, and V4 failed its offline gate.
