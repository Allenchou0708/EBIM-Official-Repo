# Task 2 PI05 200-Episode Submission — Codex Account Continuation Handoff (2026-08-12)

## 1. Read first

New Codex session must completely read:

1. `.agents/skills/ebim-local-lab-workflow/SKILL.md`
2. This handoff.
3. Original handoff: `/home/robot/2026_ebim_ssd/handoff/Task 2 PI05 200-Episode Retrain and Submission Runner Lab Handoff 2026-08-12.md`

Role remains **lab executor**. Follow the competition-speed policy: no routine hashes, full-tree checksums, repeated broad tests, or speculative rebuilds.

## 2. Continuation baseline before finalization

- Repository: `/home/robot/2026_ebim_ssd/benchmark_task2_591def2`
- Branch: `task2-pi05-200-submit-20260812`
- Local HEAD: `0ecf38133ced252597720546addc99dfb925c939`
- `collab/task2-pi05-200-submit-20260812`: same commit
- User has already authorized pushing code to that collab branch.
- Do **not** discard the current dirty worktree.

Dirty files:

```text
 M task2_isaacsim/baselines/pi05/run_pi05.sh
 M task2_isaacsim/scripts/run_isaacsim_teleop.sh
?? TASK2_PI05_200_SUBMIT_CODEX_ACCOUNT_HANDOFF_2026-08-12.md
```

Uncommitted code changes:

- `run_isaacsim_teleop.sh`: change `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` to `DEFAULT`.
- `run_pi05.sh`: final policy container uses `--network host --ipc=host` and `FASTDDS_BUILTIN_TRANSPORTS=DEFAULT`.
- `run_pi05.sh`: validate `--max-actions` and derive `max_decisions=max(40, ceil(max_actions/24)+2)`, so 1500/3000-action trials are not cut off by the old 40-decision limit.

These shell changes passed `bash -n`, focused hooks/tests, and diff review. The baseline above records the state at handoff creation; check `git log -1` and the remote ref for any later finalization commit.

## 3. Completed work that must not be repeated

- Phase A queue replacement gate: PASS at 600/600 actions, 15 valid decisions, 2400 ROS publications, 14 queue replacements, no invalid action or publish block.
- Formal dataset audit: PASS, 200 episodes / 174,719 frames, deterministic 180 train / 20 held out.
- Formal expert-only training: complete at 30,000 steps. Checkpoints 5k through 30k exist. Runtime 2:55:24, peak 12.87 GiB, initial mean loss 0.6126, final mean loss 0.02245, final logged loss 0.010.
- 30k offline held-out structural inference: PASS on all 20 held-out episodes.
- Final checkpoint:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_200_30k_v1/training/checkpoints/030000/pretrained_model`
- Relative dataset:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_200_30k_v1/relative_dataset`
- Runtime image: `ebim-task2-pi05-live:queue-replace-20260812`
- Training image: `ebim-task2-pi05:200-submit-20260812`

Do not rerun 30k training unless a new diagnosis explicitly requires a new dataset/config.

## 4. Live trials and evaluator results

All five original formal GUI trials ended with **Pick 0 / Orientation 1 / IoU 0**. A later focused cadence A/B also failed with the same evaluator result. The original 3/5 success gate is therefore failed; the current checkpoint is not submission-ready.

Trials 1–3 used 600 actions. Trial 3 became stable only after fixing ROS image transport with host IPC + Fast DDS default shared-memory transport:

- Trial 3 manifest:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_200_submit_20260812/gui_trial_03/live_runner_manifest.json`
- Result: 600/600, 15/15 decisions valid, 2400 publications, 14 replacements, no publish block.
- Evaluator:
  `/home/robot/docker/ebim-challenge/eval-task2/evaluate/eval_camera_iou_20260812_092333_747883.json`

User then explicitly requested longer Trial 4/5 rollouts.

### Trial 4 — 1500 actions

- Manifest:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260812_173005/live_runner_manifest.json`
- Complete: 1500/1500 actions, 37 valid decisions, no publish block.
- Queue replacements: 36; replaced residual actions: 322.
- Right gripper `<0.5`: 772 published actions.
- First actually published close action: 379; last: 1480; 11 closed runs.
- Arm-bound projection: 29 published actions.
- Evaluator: Pick 0 / Orientation 1 / IoU 0.
- Evaluator JSON:
  `/home/robot/docker/ebim-challenge/eval-task2/evaluate/eval_camera_iou_20260812_093639_820285.json`
- Annotated frame:
  `/home/robot/docker/ebim-challenge/eval-task2/evaluate/eval_camera_rgb_bbox2d_tight_20260812_093639_820285.jpg`
- Visual result: gripper closed after the right arm had passed/offset from the pad; pad remained on its liner.

### Trial 5 — 3000 actions

- Manifest:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260812_173705/live_runner_manifest.json`
- Complete: 3000/3000 actions, 72 valid decisions, no publish block.
- Queue replacements: 71; replaced residual actions: 566.
- Right gripper `<0.5`: 949 published actions.
- First actually published close action: 713; last: 2999; 28 closed runs.
- Arm-bound projection: 1057 actions, first at action 483 and continuing through action 2999.
- Transient freshness rejections: 166; none caused a stop. Inference p50 0.582 s, p95 0.613 s, max 0.642 s.
- Evaluator: Pick 0 / Orientation 1 / IoU 0.
- Evaluator JSON:
  `/home/robot/docker/ebim-challenge/eval-task2/evaluate/eval_camera_iou_20260812_094418_830664.json`
- Annotated frame:
  `/home/robot/docker/ebim-challenge/eval-task2/evaluate/eval_camera_rgb_bbox2d_tight_20260812_094418_830664.jpg`
- Visual result: right arm drifted into joint limits near the board; pad remained on its liner.

Conclusion: increasing the rollout length does not repair the task. Do not make 1500 or 3000 the submission default.

### Focused cadence A/B — 600 actions at 6 Hz, refill 49

- Manifest:
  `/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_200_submit_20260812/gui_cadence_ab_6hz_refill49/live_runner_manifest.json`
- Complete: 600/600 actions, 147 valid decisions, no publish block.
- Average executed actions per predicted chunk: 4.08, close to saved `n_action_steps=5`.
- Queue replacements: 146; replaced residual actions: 6704.
- Right gripper `<0.5`: 20 actions; first at action 409.
- Arm-bound projection: zero actions.
- Evaluator: Pick 0 / Orientation 1 / IoU 0.
- Evaluator JSON:
  `/home/robot/docker/ebim-challenge/eval-task2/evaluate/eval_camera_iou_20260812_100509_712506.json`
- Result: early replacement avoided joint-limit divergence but repeatedly replayed the chunk prefix; the arms accumulated only slow local deltas and never approached the demonstrated grasp pose. Do not adopt this cadence in the launcher.

## 5. Confirmed diagnostic facts

### Gripper mapping and checkpoint stats are not missing

- Task prompt exactly matches the dataset: `Pick up the thermal pad and place it on the target RAM board.`
- All 180 train episodes contain a right-gripper close.
- Demonstration first-close frame range: 272–595; median 398.5; no train episode first closes after frame 595.
- Train right-gripper action stats: min 0, max 1, mean 0.54408, std 0.49805; both checkpoint preprocessor and postprocessor contain the same stats.
- Trial 4/5 prove that live action 18 can reach 0 and the simulator closes the right gripper.

Therefore the failure is not a permanently-open gripper, wrong prompt, or missing gripper normalization.

### Strong staging distribution mismatch

This is the most important newly identified problem. It is measured, but causal impact still needs an A/B experiment.

- Across all 200 demonstrations, frame-0 spine state median is approximately `0.000014 m`; initial spine action median is `0.0`.
- The current launcher raises the spine before policy start and holds measured state near `0.485 m`.
- Only 161/200 demonstrations reach state spine `>=0.48 m`.
- They first reach it at frames 162–202 (median 171).
- At that point the two arm joint vectors have already moved far from reset: L2 difference from the frame-0 median is 2.18–4.51 (median 2.34).
- Right gripper is still open at that point.

Thus the live policy starts from a likely out-of-distribution combination: **high spine + reset arms**. In demonstrations, high spine occurs only after substantial arm motion. This is a stronger explanation than “600 actions were too short.”

### Queue cadence mismatch to inspect

- Saved PI05 config: `chunk_size=50`, `n_action_steps=5`.
- Live runner calls `predict_action_chunk()` directly and triggers a new async prediction when queue length is `<=24`.
- Due 0.5–0.65 s inference latency, Trial 4/5 execute roughly 40–42 actions from each 50-action chunk before replacement, not 5.
- This may amplify open-loop arm drift. Do not change it blindly: an inference takes about 17–20 action periods, so a literal 5-action queue cannot stay populated without a different scheduling strategy.
- A completed 6 Hz / refill-49 A/B executed 4.08 actions per chunk and still failed. It eliminated bound projection but trapped the stateless policy near the chunk-prefix behavior. Neither extreme is a viable runtime-only fix.

### Teacher-state inference versus live rollout

- Existing 30k inference on 20 held-out midpoint frames was compared directly with the ground-truth action at the same frame.
- Arm-target L2 error was only 0.017–0.046; none of 280 arm dimensions differed by more than 0.15.
- By contrast, Trial 4 first-close targets had 10/14 arm dimensions outside the 180-train first-close range; Trial 5 had 7/14 outside.
- The checkpoint is accurate under teacher-state observations but its stateless closed-loop rollout compounds distribution error. This rules out a simple serialization/normalization failure and explains why offline structural gates passed while GUI task success failed.

### PI05 state-shape observation

- Saved `config.json` says `observation.state.shape=[32]` while dataset state is 37-D and `max_state_dim=37`.
- Inspection of the installed PI05 implementation shows the model forward/sample path conditions on images + language and does not pass state separately; state is used by the relative-action processor/inverse.
- Treat this as a port limitation/contract smell, not the leading cause of this rollout failure.

## 6. Current live machine state

At handoff creation:

```text
eval_task2                       Up, image eval-task2:ebim2026
isaac-sim-5-1-0-workshop-shm    Up, image isaac-sim-5.1.0:ebim2026
isaac-sim-5-1-0-workshop        Up, original lab container; preserve it
```

- ROS domain used for fixed trials: `42`.
- The active GUI scene was launched in Codex PTY/session `78221` inside `isaac-sim-5-1-0-workshop-shm`. A new account may not inherit that PTY handle; verify the scene process and ROS topics rather than assuming it is gone.
- The original `isaac-sim-5-1-0-workshop` container uses private IPC and was intentionally left untouched/idle.
- `isaac-sim-5-1-0-workshop-shm` was created with host network and host IPC because the original container's private IPC cannot be shared after creation.
- Before any new rollout, verify `/isaac/head_camera/rgb` freshness and evaluator service visibility on domain 42.

Official Fast DDS rationale used for the transport fix:

- <https://fast-dds.docs.eprosima.com/en/3.4.x/docker/shm_docker.html>
- <https://fast-dds.docs.eprosima.com/en/3.x/fastdds/transport/shared_memory/shared_memory.html>

## 7. Recommended next sequence

1. Read the skill + both handoffs; inspect `git diff` and preserve the two uncommitted runtime fixes.
2. Verify the three containers/processes and ROS domain 42. Do not restart or rebuild images unless the existing processes are unavailable.
3. Move/copy Trial 4/5 evidence under the formal evidence directory if desired; do not delete the source artifacts.
4. Review the strong high-spine/reset-arm mismatch with the user before changing the launcher contract.
5. If more model work is authorized, change the training/runtime formulation rather than running another cadence sweep. The two leading directions are training on the actual fixed high-spine/reset-arm start distribution, or using a policy/runtime with task-progress memory/proprioceptive conditioning.
6. Run only focused tests/code review for any new formulation, then one GUI evaluator trial.
7. Report the failed GUI task-success gate honestly and preserve all existing evidence.

## 8. Useful commands

```bash
cd /home/robot/2026_ebim_ssd/benchmark_task2_591def2
git status --short --branch
git diff -- task2_isaacsim/baselines/pi05/run_pi05.sh task2_isaacsim/scripts/run_isaacsim_teleop.sh
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' | rg 'eval_task2|isaac-sim-5-1-0-workshop'
```

For a new formal rollout, keep the SHM container override:

```bash
ISAACSIM_CONTAINER=isaac-sim-5-1-0-workshop-shm ROS_DOMAIN_ID=42 \
  ./task2_isaacsim/baselines/pi05/run_pi05.sh run-task \
  --checkpoint /scratch1/2026_ebim/allen_task2_pi05/outputs/task2_200_30k_v1/training/checkpoints/030000/pretrained_model \
  --max-actions 600
```

Evaluator:

```bash
ROS_DOMAIN_ID=42 ./scripts/evaluation/task2/run.sh evaluate
```

Do not run another blind cadence or rollout-length sweep; the next experiment must change the demonstrated start distribution or the policy formulation.
