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
git clone --branch main --recurse-submodules \
  https://github.com/Allenchou0708/EBIM-Official-Repo.git ebim-task2-phase1
cd ebim-task2-phase1

docker build --pull -t ebim-task2-phase1-gt:latest .
docker run --rm ebim-task2-phase1-gt:latest health
docker run --rm ebim-task2-phase1-gt:latest unit-tests
```

Set the shared runtime values in every terminal:

```bash
cd /absolute/path/to/ebim-task2-phase1
export PI05_LIVE_IMAGE=ebim-task2-phase1-gt:latest
export ROS_DOMAIN_ID=0
export EVIDENCE_ROOT=/absolute/path/to/evidence/task2-phase1-nominal
mkdir -p "$EVIDENCE_ROOT"
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

Each attempt then performs the visible startup sequence required for the GT
demonstration:

1. reset to the scene's initial robot pose;
2. physically drive the base with `/pedal/state` to the live-pad grasp pose;
3. command and verify the spine height;
4. apply only a bounded millimetre-scale base trim, then converge the arm to
   dataset frame 399;
5. grasp, carry, lower only until the pad edge reaches the table, rotate the
   wrist downward without further Z drop, release, and retract.

For the nominal scene, the launcher uses a calibrated 50 mm Y sweep rather
than the randomized diagnostic's 10 mm sweep.  This starts contact 40 mm
farther toward negative Y so the positive-Y motion during wrist rotation puts
the pad center near the memory centerline.  `CONTACT_SWEEP_Y_M` is an explicit
engineering override; leave it unset for formal scoring.

The launcher does not pass the legacy `--preposition-at-live-grasp-base`
switch.  A direct root-pose jump from the initial scene to the grasp pose is
not part of the submission path.

Before the three runs, start the repository eval-camera service in Terminal 3.
It is the local development implementation of the Rulebook IoU/orientation
metric; the organizer's own evaluator remains authoritative:

```bash
export ROS_DOMAIN_ID=0
export ISAAC_DOCKER_ROOT="$EVIDENCE_ROOT/eval_runtime"
bash scripts/evaluation/task2/setup.sh
bash scripts/evaluation/task2/run.sh up
```

Terminal 2 runs exactly three attempts into independent directories and
captures the eval-camera result before the next reset.  Every attempt rejects
the run unless the reset event contains `randomized: false`:

```bash
export EVIDENCE_ROOT=/absolute/path/to/evidence/task2-phase1-nominal
export ISAAC_DOCKER_ROOT="$EVIDENCE_ROOT/eval_runtime"
for run in 1 2 3; do
  PI05_LIVE_IMAGE="$PI05_LIVE_IMAGE" \
  TASK2_EVIDENCE_DIR="$EVIDENCE_ROOT/run_$run" \
    bash task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh \
      nominal 1 || exit $?
  bash scripts/evaluation/task2/run.sh evaluate || exit $?
done
```

Each policy invocation exits 0 only when that attempt passes.  Every run
directory contains its own `base_stage.json`, `spine_stage.json`, and
`controller_result.json`; the evaluator directory contains one timestamped
capture per run.

After all three, require three correct-orientation captures and compute the
unselected mean IoU:

```bash
python3 -c '
import json, pathlib, statistics, sys
files = sorted(pathlib.Path(sys.argv[1]).glob("eval_camera_iou_*.json"))
assert len(files) == 3, files
rows = [json.loads(path.read_text()) for path in files]
assert all(row["is_orientation_correct"] for row in rows)
ious = [row["iou_thermalpad_vs_target_current"] for row in rows]
print("IOU", ious, "MEAN", statistics.fmean(ious))
' "$ISAAC_DOCKER_ROOT/eval-task2/evaluate"
```

First verify that startup staging was physical and settled:

```bash
for run in 1 2 3; do
RUN_DIR="$EVIDENCE_ROOT/run_$run"
python3 -c '
import json, math, pathlib, sys
base = json.loads(pathlib.Path(sys.argv[1]).read_text())
spine = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert base["success"] is True
assert base["target_source"] == "live_thermalpad_gt"
assert [p["phase"] for p in base["route"]][:3] == [
    "BACK", "STOP_AFTER_BACK", "STRAFE_RIGHT"
]
assert math.dist(base["final_pose"][:2], base["target"][:2]) <= 0.006
assert spine["success"] is True and spine["reason"] == "stable"
print("STAGING PASS", base["final_pose"], spine["final_position_m"])
' "$RUN_DIR/base_stage.json" "$RUN_DIR/spine_stage.json"
done
```

For each JSON, require:

```bash
for run in 1 2 3; do
RUN_DIR="$EVIDENCE_ROOT/run_$run"
python3 -c '
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d["success"] is True
assert d["placement_contract"] == "nominal"
assert d["reason"] == "stable_target_place_release_and_retract"
assert d["base_preposition_mode"] == "measured_odom_bounded_alignment"
assert d["base_preposition_completed_sim"] is not None
assert d["release_started_sim"] is not None
assert d["retract_completed_sim"] is not None
assert d["minimum_contact_ee_z_m"] == 0.903
assert d["contact_ee_tracking_margin_m"] == 0.019
assert d["contact_ee_clearance_tolerance_m"] == 0.001
assert d["contact_ee_z_m"] >= d["minimum_contact_ee_z_m"]
assert (d["minimum_observed_contact_ee_z_m"]
        + d["contact_ee_clearance_tolerance_m"]
        >= d["minimum_contact_ee_z_m"])
assert d["release_ee_z_m"] >= 0.900
assert d["final_pad_target_xy_error_m"] <= 0.060
assert d["final_pad_target_z_error_m"] <= 0.012
assert d["final_pad_z_span_m"] <= 0.020
print("PASS", d["final_pad_target_xy_error_m"], d["final_pad_z_span_m"])
' "$RUN_DIR/controller_result.json"
done
```

The EE-Z assertions are the nominal-scene clearance audit, not a generic link
collision sensor.  Preserve an eval-camera video/contact sheet for each formal
run and visually verify that the pad reaches the table first, the fingertips
retain a gap, and the gripper retracts.  A run reporting
`unsafe_gripper_table_clearance` is a failed attempt and must not be retried by
lowering the floor.

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
  task2_isaacsim/baselines/pi05/live/ground_truth_joint_lift.py \
  task2_isaacsim/baselines/pi05/live/fixed_stage_base.py
bash -n \
  task2_isaacsim/baselines/pi05/live/run_ground_truth_random_gui.sh
git diff --check

docker run --rm "$PI05_LIVE_IMAGE" health
docker run --rm "$PI05_LIVE_IMAGE" unit-tests
```

Do not set `PI05_MOUNT_SOURCE=1` for this audit.  Both commands must exercise
the source baked into the exact image that will be submitted.

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
