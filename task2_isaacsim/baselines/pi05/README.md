# Task 2 PI05 baseline

This directory is the code-only boundary between the official EBiM Task 2
dataset contract and LeRobot PI05. Datasets, model weights, checkpoints, and
complete logs stay on the lab machine and are excluded from Git.

## Locked pilot contract

- LeRobot: `0.6.0`
- base policy: `lerobot/pi05_base`
- instruction: `Pick up the thermal pad and place it on the target RAM board.`
- state: official 37-D `observation.state`; set `max_state_dim=37`
- action: official absolute 20-D action; PI05 boundary is 32-D
- cameras used by the policy: head, left wrist, right wrist
- excluded inputs: `observation.images.eval_camera` and `task2_extras/**`
- initial rollout: fixed mobile base and fixed spine; model controls arms and grippers
- chunk size: 50; execute 5 actions before replanning
- memory settings: bfloat16, gradient checkpointing, batch size 1, expert-only training
- relative actions: disabled for the first pilot

The 37-D state is larger than PI05's default `max_state_dim=32`, so the override
is mandatory. The 20-D action is padded to 32 dimensions inside the policy
boundary and cropped back to 20 dimensions before publishing.

## Local metadata gate

This command uses only the Python standard library:

```bash
python3 task2_isaacsim/baselines/pi05/verify_dataset_contract.py \
  task2_isaacsim/dataset/task2_regression_591def2_v1
```

It validates the ordered state/action names and shapes, the four recorded
cameras, FPS, robot type, and LeRobot dataset version. A passing result proves
only that the metadata adapter is correct; the existing regression episode is
`success=false` and is not suitable for policy training.

## Lab-only PI05 smoke command

Install and pin LeRobot in a separate GPU environment. Do not modify the
CPU-only recorder image.

```bash
lerobot-train \
  --dataset.repo_id=LOCAL_OR_HF_DATASET_ID \
  --policy.path=lerobot/pi05_base \
  --output_dir=/home/robot/2026_ebim_ssd/ebim_outputs/pi05_task2_smoke \
  --job_name=pi05_task2_smoke \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.max_state_dim=37 \
  --policy.max_action_dim=32 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=5 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.use_relative_actions=false \
  --rename_map='{"observation.images.head":"observation.images.base_0_rgb","observation.images.wrist_left":"observation.images.left_wrist_0_rgb","observation.images.wrist_right":"observation.images.right_wrist_0_rgb"}' \
  --batch_size=1 \
  --steps=20 \
  --wandb.enable=false
```

Before running this command, verify its flags with `lerobot-train --help` in the
pinned LeRobot 0.6.0 environment. The smoke run requires successful episodes;
do not train on the current failed regression episode merely to obtain a loss.

## Safety boundary for the future policy runner

1. Reject wrong-sized or non-finite model actions.
2. Crop PI05 output from 32 to the official 20 dimensions.
3. Set base velocity indices 0..2 to zero.
4. Replace spine index 19 with the current/reset spine target.
5. Clamp grippers to `[0, 1]` and apply joint limits before ROS publication.
6. Stop publishing on stale observations, inference timeout, or emergency stop.

`contract.py` implements steps 1 through 4. ROS publication, joint-limit
clamping, and watchdog behavior belong in the later lab-validated runner.
