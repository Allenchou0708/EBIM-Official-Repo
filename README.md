# EBiM Benchmark

**Phase I Task 2 submission instructions are below.**  For the wider
benchmark repository's component status, see [STATUS.md](STATUS.md).

## Phase I Task 2 policy submission

This branch is the team's **Option A policy submission** for Task 2 thermal-pad
placement.  It uses simulator-provided ground-truth object poses and deformable
pad vertices.  This use is permitted for Phase I by the organizer's 2026-08-19
announcement and must be declared in the submission form.  It is not an
onboard-perception solution and is not eligible unchanged for Phase II.

Official scoring follows
[Autonomous Robot Benchmark Rulebook 1.0](https://ebim-benchmark.github.io/docs/Autonomous_Robot_Benchmark_Rulebook_1.0.pdf):
`Pick Success × Orientation Success × IoU`, with wrong orientation scoring
zero.  The organizer evaluates three runs and takes their mean.

### Requirements

- Linux, Docker Engine, NVIDIA Container Toolkit, and an NVIDIA GPU.
- The official Isaac Sim 5.1.0 environment in a container named
  `isaac-sim-5-1-0-workshop`.
- This repository cloned with submodules and mounted into that simulator at
  `/workspace/EBiM_Challenge` by the included launcher.
- X11 access when running the required GUI scene.
- The asset file Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd must be placed under task1_isaacsim/assets/.

No model checkpoint, training dataset, Hugging Face token, or private
credential is needed by the submitted GT policy.

### Build and self-test

From a clean checkout of the public `main` branch:

```bash
git clone --branch main --recurse-submodules \
  https://github.com/Allenchou0708/EBIM-Official-Repo.git ebim-task2-phase1
cd ebim-task2-phase1

docker build --pull -t ebim-task2-phase1-gt:latest .
docker run --rm ebim-task2-phase1-gt:latest health
docker run --rm ebim-task2-phase1-gt:latest unit-tests
```

Expected health output is `task2-phase1-gt health: PASS`.  The policy source,
ROS 2 runtime, base/spine stagers, and tests are baked into the image.  The
runner does not bind-mount source by default; `PI05_MOUNT_SOURCE=1` is reserved
for development before a final image build.

The image also exposes the individual commands `stage-base`, `stage-spine`,
and `ground-truth-controller`.  Normal evaluation should use the orchestrator
below so reset verification and the visible startup sequence cannot be
skipped.

### Run the nominal policy

Terminal 1 starts the unperturbed official scene.  It deliberately omits
`--randomize-objects`:

```bash
export PI05_LIVE_IMAGE=ebim-task2-phase1-gt:latest
export ROS_DOMAIN_ID=0
bash task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh \
  launch-nominal
```

Wait for `Isaac Sim fr3duo Task 2 room bridge started`.  In Terminal 2, save
three independent runs so one attempt cannot overwrite another. Before running the commands below, change EVIDENCE_ROOT to the absolute path where you want to save the evidence :

```bash
export PI05_LIVE_IMAGE=ebim-task2-phase1-gt:latest
export ROS_DOMAIN_ID=0
export EVIDENCE_ROOT=/absolute/path/to/task2-phase1-evidence

for run in 1 2 3; do
  TASK2_EVIDENCE_DIR="$EVIDENCE_ROOT/run_$run" \
    bash task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh \
      nominal 1 || exit $?
done
```

Every attempt verifies `randomized: false`, visibly drives the base from the
initial scene pose, settles the spine, converges to the grasp posture, picks
and lifts the pad, moves left over the memory, touches the pad to the table,
rotates the wrist inward/downward at protected EE height, releases, and
retracts.  A successful `controller_result.json` ends with
`stable_target_place_release_and_retract`.

The full clean-room procedure, JSON assertions, eval-camera capture, and
randomized action diagnostic are in
[`PHASE1_SUBMISSION_RUNBOOK.md`](task2_isaacsim/baselines/pi05/PHASE1_SUBMISSION_RUNBOOK.md).
Method and limitations are documented in
[`PHASE1_POLICY_REPORT.md`](task2_isaacsim/baselines/pi05/PHASE1_POLICY_REPORT.md).

### Verified three-run result

On 2026-08-21, the locked nominal configuration completed all three runs and
produced the correct visible liner orientation in every eval-camera capture:

| run | controller | orientation | IoU | final center error | mesh Z span |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | pass | correct (`liner_only`) | 0.28750 | 17.38 mm | 4.13 mm |
| 2 | pass | correct (`liner_only`) | 0.02429 | 20.46 mm | 3.82 mm |
| 3 | pass | correct (`liner_only`) | 0.43353 | 12.18 mm | 3.21 mm |
| mean | 3/3 | 3/3 | **0.24844** | **16.67 mm** | **3.72 mm** |

This reports the complete three-run sequence rather than selecting only the
best placement.  The large IoU spread is a known limitation of open-loop
deformable contact: small in-plane shifts strongly affect the narrow target
bbox even when GT centroid error is small and the pad is visibly flat.


## Archived PI0.5 learned-policy workflow (not the Phase I submission)

The following section documents the separate learned-policy research path.
It requires an out-of-band checkpoint, did not complete the task, and is not
the submitted Option A entry point.

For the Phase I simulator-ground-truth control route, current results and the
email-compliant submission procedure are documented in
[`GROUND_TRUTH_PHASE1.md`](task2_isaacsim/baselines/pi05/GROUND_TRUTH_PHASE1.md),
[`PHASE1_POLICY_REPORT.md`](task2_isaacsim/baselines/pi05/PHASE1_POLICY_REPORT.md),
and
[`PHASE1_SUBMISSION_RUNBOOK.md`](task2_isaacsim/baselines/pi05/PHASE1_SUBMISSION_RUNBOOK.md).
This route must be disclosed as using privileged Isaac Sim state and is not a
Phase II perception solution.

The `submit` branch contains the complete official benchmark tree plus the
team's code-only PI0.5 training and live runtime. It does not contain datasets,
caches, logs, tokens, or model weights.

### Requirements and image build

- Linux with Docker Engine, the NVIDIA Container Toolkit, and an NVIDIA GPU.
- A driver compatible with the pinned CUDA 12.8 training base image.
- Isaac Sim 5.1.0 with this repository mounted at
  `/workspace/EBiM_Challenge`, as described in
  [`task2_isaacsim/README.md`](task2_isaacsim/README.md).

From a clean clone of `submit`, including the official submodules:

```bash
git clone --branch submit --recurse-submodules \
  git@github.com:Allenchou0708/EBIM-Official-Repo.git ebim-task2-submit
cd ebim-task2-submit
docker build -t ebim-task2-pi05-submit:local .
docker run --rm ebim-task2-pi05-submit:local health
```

The root [`Dockerfile`](Dockerfile) now builds the Phase I GT submission shown
above.  In the archived learned-policy workflow, `run-task` was the PI0.5
evaluation command; `run_pi05.sh` performed reset, fixed-base and initial-spine
staging, and camera preflight before invoking it.  That command is retained
for research compatibility and is not the submitted Task 2 policy.

### Checkpoint and environment

Obtain the team's 30k checkpoint out of band and keep it outside this clone.
The checkpoint directory must contain the LeRobot PI0.5 pretrained-model files,
including `model.safetensors`, and the matching `relative_dataset` must be
available locally. There is currently **no public, no-login checkpoint URL**;
publishing a model ID is a submission blocker that requires the team's explicit
hosting decision. The repository never downloads or embeds a token.

```bash
cd task2_isaacsim/baselines/pi05
cp .env.pi05.example .env.pi05
```

Edit `.env.pi05` with absolute host paths:

```dotenv
TASK2_PI05_ROOT=/absolute/path/to/task2-pi05-runtime
PI05_LIVE_IMAGE=ebim-task2-pi05-submit:local
PI05_CHECKPOINT=/absolute/path/to/checkpoints/030000/pretrained_model
PI05_RELATIVE_DATASET=/absolute/path/to/relative_dataset
ROS_DOMAIN_ID=0
ISAACSIM_CONTAINER=isaac-sim-5-1-0-workshop
```

`TASK2_PI05_ROOT` is the configurable cache/output/evidence root. A future
public model may be recorded as `PI05_MODEL_ID`, but the present runner uses the
local `PI05_CHECKPOINT` mount.

### Three-terminal evaluation

Run all commands from `task2_isaacsim/baselines/pi05`:

```bash
# Terminal 1: launch the Task 2 GUI scene and ROS bridge
./run_pi05.sh sim-up --gui

# Terminal 2: reset, stage, validate health, and publish at most 600 actions
./run_pi05.sh run-task --runtime-mode hard5 --max-actions 600

# Terminal 3, after Terminal 2 completes: capture the official local metric
./run_pi05.sh evaluate
```

The runner keeps the base isolated, lets PI0.5 control both arms, grippers and
spine, and executes the checkpoint's five-action receding horizon on simulator
time. Camera age uses host monotonic time, while cross-camera synchronization
uses ROS header simulator timestamps. Use `Ctrl-C` in Terminal 2 for the operator stop. To inspect
the image contract without ROS control, run the documented checkpoint shadow
command in [`task2_isaacsim/baselines/pi05/README.md`](task2_isaacsim/baselines/pi05/README.md).
Container health can be checked at any time with:

```bash
docker run --rm ebim-task2-pi05-submit:local health
```

Stop and remove the evaluator/helper services with:

```bash
./run_pi05.sh down
```

### Training reproduction

With a pinned training image in `.env.pi05`, the code-only reproduction path is:

```bash
./run_pi05.sh doctor
./run_pi05.sh dataset --config configs/task2_fixpos_200_expert.yaml
./run_pi05.sh train --config configs/task2_fixpos_200_expert.yaml \
  --run task2_200_30k_v1
```

The config pins the dataset and base-policy revisions, 180/20 episode split,
expert-only frozen-vision profile, and 30,000 steps. Training data and outputs
remain below `TASK2_PI05_ROOT`, outside Git and the runtime image.

The phase-balanced absolute-spine V2 experiment uses the same audited split
and base policy, with only two checkpoints:

```bash
./run_pi05.sh train --config configs/task2_fixpos_200_v2.yaml \
  --run task2_pi05_v2_12k
```

### Validated status and known limits

The V3 hard5 GUI run completed 600/600 actions with 120/120 valid policy
decisions, 0 invalid actions, no queue replacement, and effective base output
zero. Every decision executed exactly indices 0--4. Accepted three-camera
capture skew was at most 0.0834 s. Spine reached the demonstrated range, but
the right gripper never closed and right joint 4 eventually reached its lower
bound; the pad was not grasped. This validates the runtime/input contract but
is not a task-success claim. The remaining leading cause is policy phase
progression and closed-loop robustness.


