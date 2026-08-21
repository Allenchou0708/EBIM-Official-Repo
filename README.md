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

No model checkpoint, training dataset, Hugging Face token, or private
credential is needed by the submitted GT policy.

### Build and self-test

From a clean checkout of the submission branch:

```bash
git clone --branch task2_gt_control_20260820 --recurse-submodules \
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
three independent runs so one attempt cannot overwrite another:

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

When filing the official
[Repository Submission issue](https://github.com/EBiM-Benchmark/submissions/issues/new?template=submission.yml),
select Task 2 and answer **“Yes — we use the simulator's ground-truth object
poses.”**  File only one Option A issue for this team/task; if replacing an
earlier issue, explicitly state that it supersedes the previous submission.

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

## Competition tasks

| Task | Engines | Where in this repo | Status |
|---|---|---|---|
| Task 1 — Cable Routing & Plugging | Isaac Sim, MuJoCo | [`task1_isaacsim/`](task1_isaacsim/), [`task1_mujoco/`](task1_mujoco/) | see [STATUS.md](STATUS.md) |
| Task 2 — Deformable Material Handling (Thermal Pad Placement) | Isaac Sim (Genesis committed) | [`task2_isaacsim/`](task2_isaacsim/), [`assets/task2_objects/`](assets/task2_objects/), [`scripts/evaluation/task2/`](scripts/evaluation/task2/) | see [STATUS.md](STATUS.md) |
| Task 3 — Assisted Living & Feeding | Isaac Sim, MuJoCo | [`task3_isaacsim/`](task3_isaacsim/), [`task3_mujoco/`](task3_mujoco/), [`scripts/evaluation/task3/`](scripts/evaluation/task3/) | see [STATUS.md](STATUS.md) |

Full rules and official scoring are on the competition page: https://ebim-benchmark.github.io/competition.html#tasks . The evaluation code in this repository is a development facilitator; official scoring follows the rules published there.

This repository provides a workshop-focused environment for an international competition. The active workflow uses `assets/robot_room.usd` as the base scene and launches the mobile dual-arm robot through Isaac Sim. Older tabletop scene generators are kept only for reference.

For the full developer workflow, see [`docs/developer_setup.md`](docs/developer_setup.md).

## Task 1 — Mobile FR3 Duo Teleoperation (Isaac Lab + Newton)

[`task1_isaacsim/`](task1_isaacsim/README.md) contains the Isaac Sim / Isaac Lab
implementation of Task 1: teleoperating the mobile dual-arm FR3 Duo on the
Newton / MJWarp backend, with an optional deformable-cable board-plugging world.
Keyboard is the default mobile-base input; GELLO leader arms + USB foot pedal is
the tested configuration. The teleoperation input devices come from the separate
[`EBiM-Benchmark/teleoperation`](https://github.com/EBiM-Benchmark/teleoperation)
repository. The MuJoCo variant lives in [`task1_mujoco/`](task1_mujoco/README.md)
(next section).

See [`task1_isaacsim/README.md`](task1_isaacsim/README.md) for full setup and run
instructions. Quick start (from the repo root, after the one-time setup):

```bash
EMBODIMENT=fr3duo_mobile bash task1_isaacsim/scripts/run_isaaclab_newton_teleop.sh \
  --usd-path assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd \
  --controller-mode position --with-keyboard-teleop
```

## Task 1 — Cable Management (MuJoCo)

[`task1_mujoco/`](task1_mujoco/README.md) contains the MuJoCo implementation of
Task 1 for the ManipulationNet **cable_management** benchmark: a mobile
dual-arm FR3 platform with Robotiq 2F-85 grippers routing a deformable cable
across a fixture board. Five input modes (keyboard / gamepad / VR / GELLO /
unified ROS 2 teleop), a single in-sim IK shared by all of them, and the
official ManipulationNet client vendored for end-to-end scored evaluation.
The directory is self-contained — native one-click launchers plus a Docker
evaluation stack:

```bash
cd task1_mujoco
./start.sh              # native teleoperation (Windows: double-click start.bat)
./eval.sh sim           # scored ManipulationNet evaluation (Docker), terminal 1
./eval.sh client        # terminal 2: official mnet client
```

See [`task1_mujoco/README.md`](task1_mujoco/README.md) for the full participant
guide (paths, input modes, controls, troubleshooting).

## Task 2 — Mobile FR3 Duo Teleoperation (Isaac Sim 5.1.0 / PhysX)

[`task2_isaacsim/`](task2_isaacsim/README.md) contains the Task 2 teleoperation
stack: driving the mobile FR3 Duo to place the deformable thermal pad, running
in plain **Isaac Sim 5.1.0 (PhysX)** because the pad asset needs PhysX GPU
deformables (Isaac Lab + Newton cannot run it). It reuses the Task 1 helper
containers (adapters, browser controller, republisher/position controller) and
the same ROS topic contract, and works with either the full robot room
(`--scene room`, which also publishes the `/isaac/eval_camera/*` topics for the
[Task 2 evaluation stack](scripts/evaluation/task2/)) or a barebone scene.
Input devices come from the same
[`EBiM-Benchmark/teleoperation`](https://github.com/EBiM-Benchmark/teleoperation)
repository.

Quick start (from the repo root, with the Isaac Sim 5.1.0 container running and
the robot USD downloaded — no special hardware, keyboard base + browser arms):

```bash
bash task2_isaacsim/scripts/run_isaacsim_teleop.sh \
  --scene barebone \
  --with-keyboard-teleop
```

See [`task2_isaacsim/README.md`](task2_isaacsim/README.md) for prerequisites,
the GELLO + foot-pedal configuration, and the architecture.

## Task 3 — Assisted Living & Feeding (Isaac Sim 5.1.0)

[`task3_isaacsim/`](task3_isaacsim/README.md) contains the runnable Task 3
preview with direct keyboard control and a ROS/browser/GELLO bridge. The ROS
launcher selects the complete robot/gripper profile with `--gripper`: Robotiq
is the competition default, while Panda preserves the current Franka-hand
asset.

```bash
bash task3_isaacsim/scripts/run_isaacsim_teleop.sh --gripper robotiq
```

See [`task3_isaacsim/README.md`](task3_isaacsim/README.md) for Docker setup,
no-hardware browser control, GELLO/pedal commands, and current limitations.

## Task 3 — Assisted Living & Feeding (MuJoCo)

[`task3_mujoco/`](task3_mujoco/README.md) contains the MuJoCo implementation of
Task 3: the mobile dual-FR3 with Robotiq 2F-85 grippers, a bowl of coffee beans,
a spoon, a plate, a cup, and an IKEA scale station, in 100- and 300-bean scene
variants. It runs natively — no Docker, no GPU container, no ROS — and is driven
entirely by [`task3_mujoco/config.json`](task3_mujoco/config.json): motion
frames, initial robot and object poses, contact-force gripper limits, the scale
sensor, and optional head/wrist camera windows.

Twenty-one visual meshes and textures exceed the repository's 2 MB per-file limit
and are hosted on OneDrive, the same flow Task 1 uses, so fetching them is a
required first step (see the task README for the manual fallback if OneDrive
refuses the direct download):

```bash
python -m pip install -r task3_mujoco/requirements.txt
task3_mujoco/scripts/download_large_assets.sh
cd task3_mujoco && ./run.sh
```

Keyboard control matches [`task1_mujoco/`](task1_mujoco/README.md) — same
`7`/`8`/`9` selection, arrow cluster, `R`, `G`, `V`/`Space`. See
[`task3_mujoco/README.md`](task3_mujoco/README.md) for the full control tables,
configuration reference, provenance, and verification status.

## Repository Layout

```text
benchmark/
├── task1_isaacsim/              # Task 1: mobile FR3 Duo teleoperation (Isaac Lab + Newton)
├── task1_mujoco/                # Task 1: cable-management teleoperation + eval (MuJoCo)
├── task2_isaacsim/              # Task 2: thermal-pad teleoperation (Isaac Sim 5.1.0 / PhysX)
├── task3_isaacsim/              # Task 3: assisted-living teleoperation (Isaac Sim 5.1.0)
├── task3_mujoco/                # Task 3: assisted-living bean scooping (MuJoCo, native)
├── assets/                      # USD assets and generated scene files
├── docker/                      # Docker Compose runtimes for Isaac Sim and Isaac Lab
├── docs/                        # Images and supporting documentation assets
├── newton/                      # Newton physics engine submodule
├── scripts/
│   ├── common/                  # Shared path and control helpers
│   ├── manual_tests/            # Small validation scenes for assets
│   ├── newton_examples/         # Standalone Newton quick-launch examples
│   ├── scenes/                  # Main workshop demos and scene scripts
│   └── tools/                   # USD composition and inspection utilities
├── third_party/
│   └── franka_description/      # Franka robot description submodule
├── .gitmodules                  # Submodule metadata
├── pyproject.toml               # Repository-wide lint/type-check configuration
└── README.md
```

## Cloning With Submodules

Clone this repository with all submodules initialized:

```bash
git clone --recurse-submodules <repository-url>
```

If the repository was already cloned without submodules, initialize them afterward:

```bash
git submodule update --init --recursive
```

To update submodules to the commits recorded by the current checkout:

```bash
git submodule update --init --recursive
```

The current submodules are:
- `newton/`
- `third_party/franka_description/`

## Git LFS Notes

Some large workshop assets may be tracked with Git LFS instead of regular Git blobs.

Before cloning or pulling LFS-tracked assets, install and enable Git LFS once on your machine:

```bash
git lfs install
```

After that, normal Git commands are usually enough:

```bash
git clone --recurse-submodules <repository-url>
git pull
```

If Git LFS is installed, the real large files are downloaded automatically during clone and pull. If Git LFS is not installed, Git will only check out small pointer files instead of the actual `.usd` or `.blend` assets. If that happens, run:

```bash
git lfs pull
```

To inspect which files are currently tracked through Git LFS:

```bash
git lfs ls-files
```

GitHub charges Git LFS storage and download bandwidth to the repository owner. If this repository is owned by an organization such as `HCIS-Lab`, pushes to its LFS-tracked files consume the organization's Git LFS quota, not the pusher's personal quota.

On a local checkout, Git LFS stores downloaded objects under `.git/lfs/objects`. On GitHub, the repository history stores pointer files, while the actual large-file content is stored in GitHub's managed Git LFS object storage for the repository.

## Supported Container Targets

The Docker stack is parameterized in `docker/.env.base` and `docker/docker-compose.yaml`.

### Isaac Sim 5.1.0
- Image: `nvcr.io/nvidia/isaac-sim:5.1.0`
- Local tag: `isaac-sim-5.1.0:ebim2026`
- Compose profile: `isaac-sim-5.1.0`
- Intended for GUI and simulation workflows with X11 support.

### Isaac Sim 6.0.0-dev2
- Image: `nvcr.io/nvidia/isaac-sim:6.0.0-dev2`
- Local tag: `isaac-sim-6.0.0-dev2:ebim2026`
- Compose profile: `isaac-sim-6.0.0`
- Uses the currently documented pre-GA container tag.

### Isaac Lab 2.3.2
- Image: `nvcr.io/nvidia/isaac-lab:2.3.2`
- Local tag: `isaac-lab-2.3.2:ebim2026`
- Compose profile: `isaac-lab-2.3.2`
- Documented as an alternative runtime. The primary workshop workflow remains the Isaac Sim images above.

## Prerequisites

1. Linux host with a supported NVIDIA GPU.
2. Docker Engine with Docker Compose v2.
3. NVIDIA Container Toolkit configured for Docker.
4. X11 available on the host for GUI workflows.
5. Permission to pull NVIDIA NGC images.

Before launching GUI containers, allow local X11 access on the host:

```bash
xhost +local:docker
export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
touch "$XAUTHORITY"
```

## Persistent Docker Storage

All container caches and runtime data are stored under:

```text
${HOME}/docker/ebim-challenge
```

Create the required directories before the first launch. A typical layout is:

```text
~/docker/ebim-challenge/
├── isaac-sim-5.1.0/
│   ├── cache/main/ov
│   ├── cache/main/warp
│   ├── cache/computecache
│   ├── config
│   ├── data/documents
│   ├── data/Kit
│   ├── logs
│   └── pkg
├── isaac-sim-6.0.0/
│   ├── cache/main/ov
│   ├── cache/main/warp
│   ├── cache/computecache
│   ├── config
│   ├── data/documents
│   ├── data/Kit
│   ├── logs
│   └── pkg
└── isaac-lab-2.3.2/
    ├── cache/kit
    ├── cache/ov
    ├── cache/pip
    ├── cache/glcache
    ├── cache/computecache
    ├── data
    ├── documents
    └── logs
```

For writable bind mounts from both the host and containers, the Isaac Sim
services run with `${HOST_UID}:${HOST_GID}` as their UID/GID and add
`${ISAAC_SIM_GID}` as a supplemental group so they can still access
`/isaac-sim`. Their `HOME` and XDG cache/data/config paths are pinned under
`/isaac-sim` so Omniverse does not try to write under `/`. `HOST_UID`/`HOST_GID`
must match the owner of this repository; the defaults in `docker/.env.base` are
set for this workspace. If your host user uses different IDs, export them before
building and running Compose:

```bash
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
```

Bootstrap the versioned cache layout with:

```bash
python3 scripts/tools/validate_docker_runtimes.py --prepare-dirs --skip-script-check
sudo chown -R "${HOST_UID:-$(id -u)}:${HOST_GID:-$(id -g)}" \
  "$HOME/docker/ebim-challenge/isaac-sim-5.1.0" \
  "$HOME/docker/ebim-challenge/isaac-sim-6.0.0"
sudo chmod -R g+rwX \
  "$HOME/docker/ebim-challenge/isaac-sim-5.1.0" \
  "$HOME/docker/ebim-challenge/isaac-sim-6.0.0"
```

The compose stack persists the main Kit cache, CUDA compute cache,
Omniverse data/config, Kit data, logs, and package data. It intentionally does
not bind-mount `/isaac-sim/extscache`, because those extension cache folders
also contain required bundled shader resources; an empty host directory there
would hide them and break RTX shader loading.

## Docker Quick Start

Run all commands from the repository root.

The compose file depends on values from `docker/.env.base`. Pass it explicitly:

```bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml config --profiles
```

### Build and validate all runtimes

```bash
python3 scripts/tools/validate_docker_runtimes.py \
  --prepare-dirs \
  --build \
  --up
```

This builds the three local images in parallel, starts the containers, and checks workspace mounts, cache mounts, X11, host networking, and script/USD smoke tests.

### Start Isaac Sim 5.1.0

Build the local Isaac Sim 5.1.0 runtime image:

```bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml \
  --profile isaac-sim-5.1.0 build isaac-sim-5-1-0
```

Start the container:

```bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml \
  --profile isaac-sim-5.1.0 up -d
```

Enter the container:

```bash
docker exec -it isaac-sim-5-1-0-workshop bash
```

Typical GUI launch inside the container:

```bash
./runapp.sh
```

### Launch Mobile FR3 In The Robot Room

After starting a runtime, use the participant launcher documented in the
corresponding task folder. The shared robot-room builder is an implementation
module, not the participant entry point. See the Task 1, Task 2, and Task 3
README links in the task overview above.

### Start Isaac Sim 6.0.0-dev2

```bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml \
  --profile isaac-sim-6.0.0 up -d
```

Enter the container:

```bash
docker exec -it isaac-sim-6-0-0-workshop bash
```

Typical GUI launch inside the container:

```bash
./runapp.sh
```

### Start Isaac Lab 2.3.2

```bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml \
  --profile isaac-lab-2.3.2 up -d
```

Enter the container:

```bash
docker exec -it isaac-lab-2-3-2-workshop bash
```

Stop all containers again with:

```bash
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml down
```

## Workspace Mounts

The full repository is mounted into each container at:

```text
/workspace/EBiM_Challenge
```

This makes live editing from the host available in all supported container targets.

## X11 Notes

The compose file mounts:
- `${DISPLAY}`
- `${XAUTHORITY}`
- `/tmp/.X11-unix`

If GUI applications fail to open:
1. confirm `xhost +local:docker` has been executed for the current graphical session,
2. verify `DISPLAY` is exported,
3. verify `XAUTHORITY` points to a valid file,
4. restart the container after changing those variables.

## Main Workshop Scripts

### Demo Scenes
- `scripts/scenes/scene_robot_room_keyboard.py` — shared robot-room stage builder used by task-specific launchers.

### Utilities
- `scripts/tools/inspect_usd.py` — print the prim hierarchy of a USD file.

<details>
<summary>Outdated scene generators and demos</summary>

- `scripts/deprecated/scene_robot_keyboard.py` — older tabletop scene with keyboard control.
- `scripts/deprecated/scene_robot_tables.py` — older tabletop scene with robot but without keyboard control.
- `scripts/deprecated/scene_11_tables.py` — older 11-table composition utility and preview.
- `scripts/deprecated/scene_with_table.py` — older single-table placement example.
- `scripts/deprecated/keyboard_control.py` — older reduced robot keyboard-control demo.
- `scripts/deprecated/launch_random_heads_scene.py` — older tabletop head randomization launcher.
- `scripts/deprecated/create_wall_room.py` — older wall-room USD generator. The current base room is `assets/robot_room.usd`.
- `scripts/deprecated/compose_scene_usd.py` — deprecated tabletop scene composer kept for reference. Active task scene composition is documented in each task folder.
- `assets/tabletop_task_scene_DEMO.usd` — older tabletop demo scene whose keyboard teleoperation is baked into the USD Action Graph. Only `scripts/deprecated/launch_random_heads_scene.py` still opens it; the supported keyboard teleoperation is documented in each task folder.

</details>

### Manual Validation Scenes
- `scripts/manual_tests/test_table_cutlery.py` — validate table plus cutlery placement.
- `scripts/manual_tests/test_table_letter.py` — validate table plus letter placement.

## Running Scripts

Task-specific launchers use the prebuilt `assets/robot_room.usd` base scene
through the shared `scripts/scenes/scene_robot_room_keyboard.py` builder. New
workshop task work should build on that room instead of generating new base
scenes.

Inspect the active robot-room USD hierarchy:

```bash
python scripts/tools/inspect_usd.py assets/robot_room.usd
```

<details>
<summary>Outdated scene generators</summary>

These scripts are kept for reference only. They do not define the current
competition base scene.

`scripts/deprecated/create_wall_room.py` creates a room USD asset.

- `--output PATH`: base output path. Default: `assets/plain_white_room.usd`. The script appends room dimensions, and `_partition` when enabled.
- `--length METERS`: inside room length along Y. Default: `30.0`.
- `--width METERS`: inside room width along X. Default: `20.0`.
- `--height METERS`: wall height. Default: `3.0`.
- `--wall-thickness METERS`: wall thickness. Default: `0.1`.
- `--material-preset NAME`: room material, one of `plain-white`, `matte-gray`, or `warm-white`. Default: `plain-white`.
- `--floor-only`: create only the floor, without walls.
- `--ceiling`: add a ceiling panel.
- `--light-density METERS`: target spacing between ceiling rect lights. Smaller values create more lights. Default: `1.8`.
- `--light-size NAME`: ceiling light panel shape, either `square` or `rectangle`. Default: `square`.
- `--partition`: add a 5m partition wall with a 1m x 2m door opening.

Official room generation example:

```bash
python scripts/deprecated/create_wall_room.py --length 30.0 --width 20.0 --height 3.0 --ceiling --partition
```

`scripts/deprecated/compose_scene_usd.py` composes the older tabletop task scene.
It is kept for reference and for inspecting the previous coffee bean setup, but
new robot-room task work should use the task-specific launcher and shared room
builder.

- `--output PATH`: USD file to write when `--save` is set. Default: `assets/tabletop_task_scene.usd`.
- `--save`: write the composed scene to `--output`.
- `--preview`: open the composed scene in Isaac Sim for visual checking.
- `--include-top-table`: add the top-center table. Do not combine this with `--with-robot`, because they occupy the same area.
- `--with-robot`: also reference the robot USD at `/World/Robot` for GUI validation.
- `--env PATH_OR_NONE`: optional environment USD. Use `none` or a USD path; relative paths resolve from the repository root. Default: `none`.
- `--randomize-cutlery-color`: apply random preview colors to cutlery assets.
- `--randomize-cutlery-placement`: randomize cutlery placement around the cutlery table.
- `--add-head`: add head payloads on the tables that have text labels.
- `--bean-count COUNT`: number of coffee bean rigid bodies to place in the bowl. Default: `150`.
- `--bean-color R G B`: coffee bean RGB color as three floats in `[0, 1]`. Default: `0.20 0.12 0.07`.
- `--bean-density VALUE`: coffee bean density for USD physics mass properties. Default: `850.0`.

Official scene composition example:

```bash
python scripts/deprecated/compose_scene_usd.py --env assets/plain_white_room_20_30_3_partition.usd --bean-count 300 --save
```

</details>

## Submodules

This repository uses Git submodules for external dependencies that should stay pinned to known commits:

```text
newton
third_party/franka_description
```

For fresh clones, use:

```bash
git clone --recurse-submodules <repository-url>
```

For existing clones, use:

```bash
git submodule update --init --recursive
```

## Asset and Path Handling

Workshop scripts use shared helpers from `scripts/common/path_utils.py` to resolve:
- repository root,
- `assets/` paths,
- `third_party/franka_description/urdfs/...` paths.

This removes the old assumption that runnable scripts must remain at the repository root.

## Physics and Control Notes

The mobile base follows a diagonal steer-drive layout. Shared helper logic in `scripts/common/tmr_base_control.py` provides:
- keyboard twist generation,
- wheel steering targets,
- wheel velocity targets,
- heading-hold compensation during translation.

This is still a simulation convenience layer, not a production-grade mobile
base controller. Physical-robot use still requires an external emergency stop
and watchdog.

### Simulation Performance

For scenes with many moving rigid bodies, such as hundreds of beans in a bowl,
enable PhysX Fabric in the Isaac Sim GUI:

1. Open `Window > Extensions`.
2. Search for `omni.physx.fabric`.
3. Enable the extension.
4. Open `Edit > Preferences > Physics > Fabric`.
5. Ensure Fabric is enabled.

Fabric improves performance by avoiding expensive per-frame USD transform
write-back for every moving rigid body. Without Fabric, PhysX updates are
written through USD transform attributes, USD notices, observer callbacks, and
Hydra render-transform synchronization. With Fabric, USD remains the authoring
format, but runtime body transforms are propagated through Fabric's simulation
data path to the renderer. This is much cheaper for dense dynamic scenes.

When Fabric is enabled, USD may not contain the latest live transforms(xform
transforms will be stale) duringsimulation. Use PhysX, Fabric-aware, or tensor
APIs for runtime state queriesinstead of reading moving body poses directly
from USD.

## Runtime Troubleshooting

If Isaac Sim reports permission errors for `/isaac-sim/kit/logs` or
`/isaac-sim/kit/data/Kit/.../user.config.json`, recreate the container after
updating the compose mounts and ensure the host cache directories are owned by
your container UID/GID:

```bash
python3 scripts/tools/validate_docker_runtimes.py --prepare-dirs --skip-script-check
sudo chown -R "${HOST_UID:-$(id -u)}:${HOST_GID:-$(id -g)}" \
  "$HOME/docker/ebim-challenge/isaac-sim-5.1.0" \
  "$HOME/docker/ebim-challenge/isaac-sim-6.0.0"
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml \
  --profile isaac-sim-5.1.0 up -d --force-recreate isaac-sim-5-1-0
```

If ROS2 bridge startup fails with missing `libament_index_cpp.so`, launch with
`--ros2-bridge fastdds` or `--ros2-bridge cyclonedds` so the bundled ROS2
library path is configured before Isaac Sim starts. The launcher re-execs
itself once in ROS mode so `LD_LIBRARY_PATH` is visible to the dynamic loader
from process startup, and stores ROS logs under `/isaac-sim/kit/logs/ros`.

## Contributing

Linting and formatting run through [pre-commit](https://pre-commit.com/). Install the hooks once after cloning:

```bash
pip install pre-commit
pre-commit install
```

Run them across the repository before pushing:

```bash
pre-commit run --all-files
```

CI runs the same command on every pull request (`.github/workflows/pre-commit.yaml`), and `pre-commit` is the required status check on `main`, so a pull request cannot merge while it is red. The hooks cover Ruff (lint and format), codespell, license headers, and a set of file checks. Their configuration lives in `.pre-commit-config.yaml`, with Ruff's rules in `pyproject.toml`; several directories are excluded, listed under `exclude` at the end of `.pre-commit-config.yaml`. Not all of those exclusions are cosmetic. Three of them — `.vscode/`, `scripts/newton_examples/`, and `task1_isaacsim/` — hold files carrying third-party copyright headers (Isaac Lab's in `.vscode/tools/setup_vscode.py`, the Newton Developers' in the other two), and the exclusion is the only thing stopping the license-header hook from stamping an EBiM copyright on top of someone else's. To narrow the list, add a per-hook `exclude:` to `insert-license` covering those paths first; [LICENSES/README.md](LICENSES/README.md) covers the Isaac Lab case in detail. `task3_mujoco/` is a narrower case and needs no such protection: it is ported from an upstream repository, but the code was contributed under Apache-2.0 (see [NOTICE](NOTICE)), so the license-header hook stamps it normally. It is excluded from Ruff alone, via `extend-exclude` in `pyproject.toml`, because reformatting its tuned control code to line-length 79 would churn it and make upstream syncs conflict-prone. Its `assets/` subtree is in the `exclude` list for the same reason as the root `assets/` — mesh and texture data, not source. Every other hook, including the 2 MB `check-added-large-files` guard, applies to it.

`pyproject.toml` here holds tool configuration only — this repository is not a pip-installable package, so there is no `pip install -e .` step.

For runtime and environment setup — host requirements, Docker targets, and the simulation stack — see [docs/developer_setup.md](docs/developer_setup.md).

## Validation Checklist

After changes, verify the following:

1. `docker compose` resolves all configured profiles.
2. The repository appears inside each container at `/workspace/EBiM_Challenge`.
3. Isaac Sim GUI launches correctly through X11.
4. Each task-specific participant launcher starts and resolves its required USD assets.
5. `third_party/franka_description/urdfs/mobile_fr3_duo_v0_2_franka_hand.usd` is available.
6. No tools or docs still reference the removed `source/robot_lab` tree.

## Known Follow-Up Items

- Keep submodule URLs and pinned commits in `.gitmodules` up to date.
- Clean any generated URDF files in `third_party/franka_description/urdfs/` that still contain absolute paths from previous machines.
- Optionally add helper shell scripts for directory bootstrap of the Docker cache layout.

## References

- Isaac Sim 5.1.0 container documentation: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html>
- Isaac Sim 6.0.0 container documentation: <https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html>
- Isaac Lab Docker guide: <https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html>
