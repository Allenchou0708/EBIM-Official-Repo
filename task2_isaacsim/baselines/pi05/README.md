# Task 2 PI05 baseline

This directory is the code-only boundary between the official EBiM Task 2
dataset contract and LeRobot PI05. Datasets, model weights, checkpoints, and
complete logs stay on the lab machine and are excluded from Git.

## Locked pilot contract

- LeRobot: `0.6.0`
- base policy: `lerobot/pi05_base`
- base policy revision: `338b5c22c12dbdd0d2ab19046802de2eb7696a6b`
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
cameras, FPS, robot type, LeRobot dataset version, and the `q01`/`q99`
quantiles required by PI05 normalization. A passing result proves only that the
metadata adapter is correct; the existing regression episode is
`success=false` and is not suitable for policy training.

## Lab-only PI05 smoke gate

Install and pin LeRobot in a separate GPU environment. Do not modify the
CPU-only recorder image. The wrapper verifies the Task 2 metadata and success
labels, selects at most two successful episodes, pins LeRobot `v0.6.0` source
commit `30da8e687a6dfc617fcd94afc367ac7071c376ce`, checks CUDA/bfloat16, disables
Hub uploads, pins the model revision, and prints the exact command before
executing it.

The lab environment should use Python 3.12, LeRobot source `v0.6.0`, PyTorch
`2.11.0+cu128`, torchvision `0.26.0+cu128`, and the `training,pi` extras. Keep
that environment separate from Isaac Sim and the recorder container, and save
its `pip freeze` with the experiment evidence.

PI05 also loads the tokenizer from the gated
`google/paligemma-3b-pt-224` repository. Before `--execute`, sign in to Hugging
Face, review and accept Google's PaliGemma usage license, and run
`hf auth login` with the same `HF_HOME` used for training. Never put a Hub token
in Git, logs, shell history, or chat. The wrapper downloads only the pinned PI05
and PaliGemma `config.json` files during runtime preflight, so missing access
fails before a large model load.

```bash
python3 task2_isaacsim/baselines/pi05/train_smoke.py \
  --dataset-root task2_isaacsim/dataset/task2_regression_591def2_v1 \
  --output-dir /scratch1/2026_ebim/allen_task2_pi05/outputs/pi05_code_smoke \
  --lerobot-source-root /home/robot/2026_ebim_ssd/LeRobot_v0.6.0
```

That is a dry run. Add `--execute` only after Isaac Sim and its helper/evaluator
containers are stopped. The current regression dataset contains no successful
episode. It may still exercise one forward/backward/update step if
`--allow-unsuccessful-smoke-data` is explicitly supplied, but the output must
then be deleted and must never be called a baseline. Use `--save-checkpoint`
only for the separate checkpoint round-trip gate because a PI05 checkpoint is
large.

The generated preflight JSON and full training log belong under the dedicated
HDD workspace `/scratch1/2026_ebim/allen_task2_pi05`, never in Git.

## Safety boundary for the future policy runner

1. Reject wrong-sized or non-finite model actions.
2. Crop PI05 output from 32 to the official 20 dimensions.
3. Set base velocity indices 0..2 to zero.
4. Replace spine index 19 with the current/reset spine target.
5. Clamp grippers to `[0, 1]` and apply joint limits before ROS publication.
6. Stop publishing on stale observations, inference timeout, or emergency stop.

`contract.py` implements steps 1 through 4. ROS publication, joint-limit
clamping, and watchdog behavior belong in the later lab-validated runner.
