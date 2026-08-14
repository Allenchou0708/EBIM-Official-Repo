# Task 2 PI0.5：V1/V2 30k GUI 測試指令

## 變數與路徑

每個 terminal 都先執行：

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2

export TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
export PI05_TRAIN_IMAGE=ebim-task2-pi05:200-submit-20260812
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:v3-hard5-20260814
export ROS_DOMAIN_ID=62

export PI05_V1_30K_ROOT="$TASK2_PI05_ROOT/outputs/task2_200_30k_v1"
export PI05_V1_30K_CKPT="$PI05_V1_30K_ROOT/training/checkpoints/030000/pretrained_model"
export PI05_V1_DATASET="$PI05_V1_30K_ROOT/relative_dataset"

export PI05_V2_30K_CONFIG=configs/task2_fixpos_200_v2_expert_30k.yaml
export PI05_V2_30K_ROOT="$TASK2_PI05_ROOT/outputs/task2_pi05_v2_expert_30k"
export PI05_V2_30K_CKPT="$PI05_V2_30K_ROOT/training/checkpoints/030000/pretrained_model"
export PI05_V2_DATASET="$PI05_V2_30K_ROOT/relative_dataset"

export PI05_STAGING_AUDIT="$TASK2_PI05_ROOT/evidence/task2_pi05_v2_full_30k_preflight/startup_staging_audit.json"
export PI05_TEST_EVIDENCE="$TASK2_PI05_ROOT/evidence/task2_pi05_v1_v2_30k"
```

路徑說明：

- `PI05_V2_30K_CKPT` 才是本輪新完成的 V2 expert-only 30k model。
- `configs/task2_fixpos_200_v2.yaml` 與 output `task2_pi05_v2_12k` 是舊 12k V2，這輪不要使用。
- `PI05_LIVE_IMAGE` 名稱中的 `v3-hard5` 只是 runtime image tag，不是 V3 model。
- `PI05_STAGING_AUDIT` 是共用的 dataset-derived staging evidence，不是 model checkpoint。

## 30k checkpoint 與 offline gate

```bash
test "$(git branch --show-current)" = submit
test "$(git rev-parse HEAD)" = "$(git rev-parse collab/submit)"
git status --short --branch

test -d "$PI05_V1_30K_CKPT"
test -d "$PI05_V2_30K_CKPT"
test ! "$PI05_V2_30K_CKPT" -ef \
  "$TASK2_PI05_ROOT/outputs/task2_pi05_v2_12k/training_12k/checkpoints/006000/pretrained_model"

bash task2_isaacsim/baselines/pi05/run_pi05.sh parser-gate \
  --profile v2_expert_30k
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-models-30k
```

查看已完成的 V1/V2 30k offline 結果：

```bash
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d["results"][0]; print("V1", r["step"], r["mean_loss"], r["finite_20d_outputs"], r["joint_and_gripper_bounds_valid"])' \
  "$PI05_TEST_EVIDENCE/v1_030000_heldout_gate.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d["results"][0]; print("V2", r["step"], r["mean_loss"], r["finite_20d_outputs"], r["joint_and_gripper_bounds_valid"])' \
  "$PI05_TEST_EVIDENCE/v2_030000_heldout_gate.json"
```

## Terminal 1：GUI simulator

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh sim-up --gui
```

## Terminal 2：V1 30k

Shadow：

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V1_30K_CKPT" \
  --dataset-root "$PI05_V1_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label v1-030000-shadow \
  --shadow --max-actions 5 --max-duration-s 60

export PI05_V1_SHADOW="$(ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_v1-030000-shadow_* | head -1)"
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-shadow \
  --run-dir "$PI05_V1_SHADOW" --contract v1
```

確認 `$PI05_V1_SHADOW/settled_fresh_wrist_right.ppm` 看得到 pad 前方後，才執行：

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V1_30K_CKPT" \
  --dataset-root "$PI05_V1_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label v1-030000-hard5 \
  --confirm-right-wrist-pad-visible \
  --max-actions 600 --max-duration-s 300
```

停止 V1 runner 後再測 V2。

## Terminal 2：V2 expert-only 30k

Shadow：

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V2_30K_CKPT" \
  --dataset-root "$PI05_V2_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label v2-expert-030000-shadow \
  --shadow --max-actions 5 --max-duration-s 60

export PI05_V2_SHADOW="$(ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_v2-expert-030000-shadow_* | head -1)"
bash task2_isaacsim/baselines/pi05/run_pi05.sh verify-shadow \
  --run-dir "$PI05_V2_SHADOW" --contract v2
```

確認 `$PI05_V2_SHADOW/settled_fresh_wrist_right.ppm` 看得到 pad 前方後，才執行：

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --runtime-mode hard5 \
  --checkpoint "$PI05_V2_30K_CKPT" \
  --dataset-root "$PI05_V2_DATASET" \
  --staging-audit "$PI05_STAGING_AUDIT" \
  --run-label v2-expert-030000-hard5 \
  --confirm-right-wrist-pad-visible \
  --max-actions 600 --max-duration-s 300
```

## Monitor、evidence、停止

```bash
ros2 topic echo /isaac/clock --once
ros2 topic echo /isaac/joint_states_full --once
ros2 topic echo /isaac/odom --once
ros2 topic info /isaac/spine_target --verbose
```

```bash
ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_v1-030000-{shadow,hard5}_* 2>/dev/null
ls -dt "$TASK2_PI05_ROOT"/outputs/live_submit_v2-expert-030000-{shadow,hard5}_* 2>/dev/null
find "$PI05_TEST_EVIDENCE" -maxdepth 2 -type f -printf '%P\n' | sort
```

```bash
bash task2_isaacsim/baselines/pi05/run_pi05.sh down
```

## 注意事項

- 一律先 shadow；`verify-shadow` 通過且右腕相機看得到 pad 才能正式 publish。
- V1 使用 `--contract v1`；V2 30k 使用 `--contract v2`，不可交換。
- 不使用 V3/V4、舊 V2 12k 或任何 015000 checkpoint。
- staging timeout、clock reset/stall、skew/freshness、bounds 或 publisher contention 失敗即停止。
- 不降低 `0.10 s` skew 門檻，不增加猜測式 IK。
- runner 使用 `/isaac/clock`；host monotonic 只用於 watchdog/診斷。
