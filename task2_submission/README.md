# EBiM 2026 Phase I Task 2 submission

This directory packages the team's **Option A ground-truth policy** for Task 2
thermal-pad placement.  It supersedes the earlier PI0.5-checkpoint draft that
previously occupied `task2_submission/`.  The submitted controller requires no
checkpoint, dataset, Hugging Face token, or private credential.

The policy uses simulator-provided `thermalpad` and `board_target` poses and
deformable-pad vertices.  This privileged state is permitted for Phase I by
the organizer's 2026-08-19 announcement and must be declared in the submission
form.  It is not an onboard-perception solution and cannot be used unchanged
for Phase II.

## Build and verify

Run from the repository root after cloning the public `main` branch:

```bash
docker build --pull \
  -f task2_submission/Dockerfile \
  -t ebim-task2-phase1-gt:submission .

docker run --rm ebim-task2-phase1-gt:submission health
docker run --rm ebim-task2-phase1-gt:submission unit-tests
```

Expected health output:

```text
task2-phase1-gt health: PASS
```

This Dockerfile and the root `Dockerfile` expose the same audited commands.
`task2_submission/entrypoint.sh` delegates to the single baked implementation
at `task2_isaacsim/baselines/pi05/submission_entrypoint.sh`, preventing the two
image layouts from drifting.

## Run the nominal policy

Requirements are Linux, Docker Engine, NVIDIA Container Toolkit, an NVIDIA
GPU, X11, and the official Isaac Sim 5.1.0 environment in a container named
`isaac-sim-5-1-0-workshop`.  The simulator launcher runs on the host; the ROS 2
base/spine/controller processes run from the image without a source bind mount.

Terminal 1:

```bash
export PI05_LIVE_IMAGE=ebim-task2-phase1-gt:submission
export ROS_DOMAIN_ID=0
bash task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh \
  launch-nominal
```

Wait until the Task 2 room bridge is ready.  Terminal 2 runs one attempt and
writes its base, spine, and controller evidence outside the repository:

```bash
export PI05_LIVE_IMAGE=ebim-task2-phase1-gt:submission
export ROS_DOMAIN_ID=0
export TASK2_EVIDENCE_DIR=/absolute/path/to/evidence/run_1

bash task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh \
  nominal 1
```

The visible startup sequence is initial pose, physical base drive, spine
settling, bounded pre-grasp convergence, grasp/lift, target alignment,
pad-first table contact, inward/downward wrist rotation, release, and vertical
retract.  The launcher rejects a randomized reset in nominal mode and exits
nonzero when a physical or safety gate fails.

For the required three independent runs, eval-camera capture, JSON assertions,
and unselected mean calculation, follow
[`PHASE1_SUBMISSION_RUNBOOK.md`](../task2_isaacsim/baselines/pi05/PHASE1_SUBMISSION_RUNBOOK.md).
The implementation, safety decisions, results, limitations, and Phase II plan
are documented in
[`PHASE1_POLICY_REPORT.md`](../task2_isaacsim/baselines/pi05/PHASE1_POLICY_REPORT.md).

## Verified local result

The locked nominal configuration completed all three local runs with correct
`liner_only` orientation.  The unselected mean IoU was `0.24844`; mean final
target-center error was `16.67 mm`, and mean released-pad mesh Z span was
`3.72 mm`.  The organizer's independent three container runs remain
authoritative.

## Submission declaration

File only one Task 2 issue for the team.  Select Option A and state clearly:

> Task 2 uses Isaac Sim ground-truth object poses and deformable-pad vertices
> for Phase I control and success gating. It does not claim an onboard
> perception solution and will require RGB/depth perception for Phase II.

Do not submit the supporting policy report as a second Option B entry for the
same task.
