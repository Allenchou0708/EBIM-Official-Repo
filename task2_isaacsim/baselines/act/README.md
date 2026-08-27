# Task 2 ACT baseline

This directory trains and deploys an Action Chunking Transformer (ACT) for
Task 2. All commands below are for the lab Linux server, not the Windows/local
coordinator machine.

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2/task2_isaacsim/baselines/act
```

The implementation uses ACT from the same pinned LeRobot revision as the
existing PI0.5 image. It does not import `_pytorch`: the original repository
assumes the proprioceptive state and action have the same 14 dimensions,
whereas Task 2 has a 37-D state and a 20-D action. LeRobot's ACT retains the
ACT architecture and objective while supporting these different dimensions.

The policy consumes only the three robot cameras (`head`, `wrist_left`, and
`wrist_right`). The static evaluator camera is removed from the ACT dataset
view and can never become a policy input.

## 1. Conda environment

Run from this directory so the relative pip requirement resolves correctly:

```bash
conda env create -f environment.yml
conda activate ebim-act
```

`requirement.txt` is the requested compatibility entry point;
`requirements.txt` contains the actual pinned packages. CUDA is supplied by
the lab NVIDIA driver and PyTorch wheel selected by the pinned LeRobot
package. Verify it before training:

```bash
python -c 'import torch, lerobot; print(torch.__version__, torch.cuda.is_available())'
```

## 2. Prepare the 180/20 dataset view

The operation is zero-copy for the large parquet/video data. It creates
symlinks below the ACT HDD root, copies only the small metadata, removes the
evaluator camera from that metadata, writes the deterministic episode split,
and computes action/state normalization statistics from the 180 training
episodes only.

```bash
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export TASK2_ACT_ROOT=/scratch1/2026_ebim/allen_task2_act

./run_act.sh prepare
```

Inputs and outputs:

```text
source: /scratch1/2026_ebim/allen_task2_pi05/datasets/task2_fixpos_200_46ab41f
view:   /scratch1/2026_ebim/allen_task2_act/datasets/task2_fixpos_200_act
split:  .../task2_fixpos_200_act/act_split.json
stats:  .../task2_fixpos_200_act/act_train_stats.json
```

The split seed is `20260812`; the manifest always contains exactly 180 train
and 20 validation episodes with no overlap.

## 3. Train

Paper/repository defaults used here are batch size 8, learning rate `1e-5`,
chunk size 100, ResNet-18, hidden dimension 512, feed-forward dimension 3200,
KL weight 10, and 2000 epochs. As in original ACT, one epoch samples one
random timestep from every episode. With 180 episodes and batch size 8, 2000
epochs correspond to `2000 * ceil(180 / 8) = 46000` optimizer steps.
Original ACT saves every 100 epochs; for this split that maps to 2300 steps.

```bash
./train_act.sh \
  --steps 46000 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --checkpoint-every 2300 \
  --output-path /scratch1/2026_ebim/allen_task2_act/outputs/task2_act_paper_defaults
```

Or use the umbrella command:

```bash
./run_act.sh train \
  --steps 46000 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --checkpoint-every 2300
```

Training alternates one training epoch and one complete validation epoch.
After each pair it prints all accumulated lists and rewrites:

```text
/scratch1/2026_ebim/allen_task2_act/outputs/task2_act_paper_defaults/loss_history.json
```

The lists include total, L1, and KL losses for both train and validation.
Validation runs without gradients and with Dropout disabled, while retaining
ACT's VAE posterior path so its KL term is actually measured.
The optimized objective is exactly:

```text
total_loss = L1(action, predicted_action) + 10 * KL(q(z|state, action) || N(0, I))
```

Checkpoints are stored as:

```text
.../checkpoints/002300/pretrained_model/
.../checkpoints/004600/pretrained_model/
.../checkpoints/last -> final step
```

For a minimal GPU/data-path smoke, use one ACT epoch (23 steps at batch 8):

```bash
./run_act.sh train --steps 23 \
  --output-path /scratch1/2026_ebim/allen_task2_act/outputs/act_smoke_23
```

## 4. Model-only held-out test

```bash
export ACT_CHECKPOINT=/scratch1/2026_ebim/allen_task2_act/outputs/task2_act_paper_defaults/checkpoints/last/pretrained_model

./test_act.sh --batch-size 8 --checkpoint "${ACT_CHECKPOINT}"
```

The JSON result is written to:

```text
/scratch1/2026_ebim/allen_task2_act/evidence/act_test_metrics.json
```

It contains held-out total/L1/KL loss. It deliberately reports
`official_valid_placement_iou: null`: a demonstration-only forward pass does
not move the pad and therefore cannot produce an official placement score.

If Terminal 1 already has the GUI scene running, the test wrapper can perform
the held-out loss test, one simulator rollout, and the official local metric
in sequence:

```bash
./test_act.sh --batch-size 8 --checkpoint "${ACT_CHECKPOINT}" --sim
```

## 5. Isaac Sim inference and official local metric

The live adapter returns absolute `(chunk, 20)` Task 2 actions in exactly this
order:

```text
base(3), left arm(7), right arm(7), left/right gripper(2), spine(1)
```

The existing PI0.5 runner publishes the same arm `JointState`, gripper
`JointState`, and spine `Float64` messages for ACT. It also fixes the base
output to zero after staging and executes only five actions from each
100-action ACT chunk in `hard5` mode.

Use three terminals.

Terminal 1:

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2/task2_isaacsim/baselines/act
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export TASK2_ACT_ROOT=/scratch1/2026_ebim/allen_task2_act
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:v3-hard5-20260814
./run_act.sh sim-up --gui
```

Terminal 2:

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2/task2_isaacsim/baselines/act
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export TASK2_ACT_ROOT=/scratch1/2026_ebim/allen_task2_act
export ACT_CHECKPOINT=/scratch1/2026_ebim/allen_task2_act/outputs/task2_act_paper_defaults/checkpoints/last/pretrained_model
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:v3-hard5-20260814
./inference_act.sh --checkpoint "${ACT_CHECKPOINT}"
```

Terminal 3, after Terminal 2 finishes:

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2/task2_isaacsim/baselines/act
export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
./run_act.sh evaluate
```

The evaluator first prints its original Task 2 result, then prints:

```text
valid_placement_iou = pick_success * placement_orientation_success * placement_iou
```

The second JSON report is also saved to
`/scratch1/2026_ebim/allen_task2_act/evidence/act_official_metric.json`.

Thus an incorrect orientation always produces `valid_placement_iou = 0`.
The repository's local frame evaluator does not measure completion time, so
the official tie-breaker is reported as unavailable rather than fabricated.

Stop all services with:

```bash
./run_act.sh down
```

## Runtime validation boundary

The code in this directory can be syntax/unit tested without a GPU. The
following gates must still be run on the lab server before treating the model
as usable: environment import, `prepare`, one 23-step GPU smoke, checkpoint
load, one shadow/live inference, and one simulator evaluator capture.

ACT defaults and objective follow the official ACT implementation and the
LeRobot ACT port:

- <https://github.com/tonyzhaozh/act>
- <https://github.com/huggingface/lerobot/tree/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/policies/act>
