# EBiM Phase I Task 2 submission runbook

## 1. Submission decision and deadline

Use one Task 2 submission option only.  This runbook targets **Option A:
ground-truth policy**.  In the submission form, explicitly declare simulator
ground-truth object poses and deformable-pad vertices.  The deadline from the
organizer email is Aug 22, 2026 AoE (Aug 23 11:59 UTC / 19:59 Taipei).

Do not submit `PHASE1_POLICY_REPORT.md` as a separate Option B entry if Option A
is selected.  It is the method/evidence attachment for reviewers.

## 2. Clean checkout and image

Requirements: Linux, Docker Engine, NVIDIA Container Toolkit, NVIDIA GPU, and
an Isaac Sim 5.1.0 container named `isaac-sim-5-1-0-workshop` with this clean
checkout mounted at `/workspace/EBiM_Challenge`.

```bash
git clone --branch task2_gt_control_20260820 --recurse-submodules \
  git@github.com:Allenchou0708/EBIM-Official-Repo.git ebim-task2-phase1
cd ebim-task2-phase1

docker build -t ebim-task2-pi05-submit:phase1 .
docker run --rm ebim-task2-pi05-submit:phase1 health
```

Set the shared runtime values in every terminal:

```bash
cd /absolute/path/to/ebim-task2-phase1
export PI05_LIVE_IMAGE=ebim-task2-pi05-submit:phase1
export ROS_DOMAIN_ID=0
export TASK2_EVIDENCE_DIR=/absolute/path/to/evidence/task2-phase1-nominal
mkdir -p "$TASK2_EVIDENCE_DIR"
```

Evidence must be outside the Git checkout and writable by the current user.

## 3. Official nominal scene

Terminal 1 launches an unperturbed GUI scene.  The command intentionally omits
`--randomize-objects`:

```bash
PI05_LIVE_IMAGE="$PI05_LIVE_IMAGE" \
  bash task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh \
  launch-nominal
```

Wait until the scene reports that the robot room and ROS bridge are ready.
Leave this terminal running.

Terminal 2 runs exactly three attempts.  Every attempt requests a fresh reset
and rejects the run unless the reset event contains `randomized: false`:

```bash
PI05_LIVE_IMAGE="$PI05_LIVE_IMAGE" \
TASK2_EVIDENCE_DIR="$TASK2_EVIDENCE_DIR" \
  bash task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh \
  nominal 3 | tee "$TASK2_EVIDENCE_DIR/nominal_three_runs.log"
```

The command exits 0 only when all three attempts pass.  `controller_result.json`
contains the last attempt; the terminal log contains all three summaries.  If
per-attempt JSON files are needed, use a different `TASK2_EVIDENCE_DIR` for
each `nominal 1` invocation.

For each JSON, require:

```bash
python3 -c '
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d["success"] is True
assert d["placement_contract"] == "nominal"
assert d["reason"] == "stable_target_place_release_and_retract"
assert d["release_started_sim"] is not None
assert d["retract_completed_sim"] is not None
assert d["final_pad_target_xy_error_m"] <= 0.055
assert d["final_pad_target_z_error_m"] <= 0.012
assert d["final_pad_z_span_m"] <= 0.020
print("PASS", d["final_pad_target_xy_error_m"], d["final_pad_z_span_m"])
' "$TASK2_EVIDENCE_DIR/controller_result.json"
```

## 4. Randomized action diagnostic

This is not the nominal target-overlap score.  Stop Terminal 1 with `Ctrl-C`,
then relaunch with perturbations:

```bash
bash task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh \
  launch-random
```

In Terminal 2:

```bash
export TASK2_EVIDENCE_DIR=/absolute/path/to/evidence/task2-phase1-random
mkdir -p "$TASK2_EVIDENCE_DIR"
bash task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh \
  trials 3
```

Each reset must report `randomized: true`.  The `randomized-flat` contract does
not require target XY accuracy, but still requires lift, contact, downward
wrist rotation, release, table height, mesh Z span at most 20 mm, and retract.
A grasp/transport failure remains a failed attempt and must not be reported as
a placement success.

## 5. Tests and source checks

```bash
python3 -m py_compile \
  task2_isaacsim/baselines/pi05/live/ground_truth_joint_lift.py
bash -n \
  task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh
git diff --check

docker run --rm --entrypoint /bin/bash \
  -e PYTHONPATH=/workspace/EBiM_Challenge:/opt/ros/jazzy/lib/python3.12/site-packages \
  -v "$PWD:/workspace/EBiM_Challenge:ro" \
  "$PI05_LIVE_IMAGE" -lc \
  'source /opt/ros/jazzy/setup.bash && /opt/lerobot/.venv/bin/python \
   -m unittest task2_isaacsim.tests.test_ground_truth_joint_lift'
```

## 6. Submission checklist

- Public repository URL resolves without credentials.
- Root `Dockerfile` builds and its `health` command exits 0.
- README/runbook state the exact launch and evaluation commands.
- Ground-truth use is declared in the form and report.
- Three nominal attempts are attached or linked; report their mean rather
  than selecting only the best run.
- JSON logs and eval-camera evidence are outside Git but accessible to the
  reviewers through the submitted issue/link.
- The issue identifies Task 2 and Option A and supersedes any earlier Task 2
  issue if the organizer requests one issue per team/task.
- No access token, private dataset, cache, or local absolute path is committed.
- Do not claim this policy is eligible for Phase II physical evaluation.

If the Docker build or formal entry point cannot be made public before the
deadline, stop and deliberately choose Option B Technical Report instead; do
not submit both options.
