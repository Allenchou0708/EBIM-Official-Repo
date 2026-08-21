# EBiM Phase I Task 2 policy report

## Executive summary

This work implements a disclosed simulator-ground-truth policy for Task 2,
thermal-pad placement.  The selected Phase I route is organizer Option A
(policy); this report is supporting documentation, not a simultaneous Option B
submission.  A fixed-scene run has been verified end to end: the robot grasps
and lifts the pad, moves it above the memory, makes light table contact,
rotates the wrist inward/downward, opens the gripper, and retracts.  The
released pad overlaps the target RAM and remains on the table.

## Rule compliance and disclosure

The organizer's 2026-08-19 email explicitly permits simulator ground-truth
positions/states in Phase I if declared.  This policy reads:

- `/isaac/task2/object_poses` for `thermalpad` and `board_target` poses;
- `/isaac/task2/pad_points` for the deformable pad centroid and Z span;
- measured joint and right-end-effector feedback.

Suggested form disclosure:

> Task 2 uses Isaac Sim ground-truth object poses and deformable-pad vertices
> for Phase I control and success gating. It does not claim an onboard
> perception solution and will require RGB/depth perception for Phase II.

The repository contains a root Dockerfile and reproducible commands.  No
dataset, model credential, log, or checkpoint is committed.  Evaluation must
use three runs and report the mean, as required by the email.

## Method

The controller combines a demonstration-derived grasp/transport prior with a
feedback-gated Cartesian placement state machine.  Episode 19 of the audited
Task 2 dataset provides safe joint landmarks and two wrist orientations:
pre-contact and downward placement.  Live ground truth anchors the base and
target yaw.

The key placement change is contact-first control.  XY is aligned while the
pad is safely elevated.  The gripper then descends only to first table contact;
it does not chase a precise Z or keep descending to repair XY.  Contact causes
an immediate bounded quaternion interpolation toward the demonstrated inward,
downward wrist pose with a 5 mm wrist drop.  Only after orientation and table
height gates pass does the gripper open.  A vertical retract prevents dragging
the released pad.

Nominal and randomized trials use different acceptance contracts.  Nominal
requires target overlap.  Randomized trials measure whether the manipulation
sequence remains physically correct and the released pad lies flat, without
pretending that an open-loop controller can perfectly cancel arbitrary object
perturbations.

## Implementation and safety

Primary files:

- `live/ground_truth_joint_lift.py`: controller, gates, JSON evidence;
- `live/run_ground_truth_random_gui.sh`: nominal/random simulator launcher,
  verified reset, spine staging, exact-attempt runner, evidence mount;
- `tests/test_ground_truth_joint_lift.py`: transform, bounded-step,
  quaternion, contact-offset, and release-contract tests.

Safety behavior includes command-subscriber ownership checks, joint
preposition tolerance, grasp dwell, lift and frame-520 transport checkpoints,
bounded Cartesian displacement/speed, simulator-time pacing, a table-height
release gate, an open-gripper requirement, and measured retract completion.
The process exits nonzero on failed physical gates and writes the reason to
JSON.

## Experiments and results

Episode 19 was inspected at frames 530, 600, 700, 800, 834, 835, and 900.
The source data confirmed that the pad first approaches the table while held
upright, the wrist rotates downward before frame 835, the gripper then opens,
and the pad settles after retraction.  Source mesh Z span falls from about
120 mm while held upright to about 2.7 mm after settling.

The final nominal run used a fresh unperturbed scene reset and passed every
controller gate:

| metric | value |
| --- | ---: |
| maximum pad lift | 167.84 mm |
| contact pad centroid | (2.1490, 1.9431, 0.8606) m |
| release/final target XY error | 28.38 mm |
| final target Z error | 2.10 mm |
| final mesh Z span | 13.84 mm |
| release wrist orientation error | 0.53 deg |
| retract completion (sim time) | 23.383 s |
| result | `stable_target_place_release_and_retract` |

The saved top-down image confirms target overlap and gripper clearance.  A
particularly well-centered preceding calibration run reached 3.01 mm XY error
after release, but exceeded the initial over-strict 10 mm mesh-span threshold;
it was correctly not counted as a pass.  Failed runs also exposed two useful
failure modes: grasp loss under large perturbation and continued descent when
XY drift blocked the old contact transition.  The latter was fixed by making
physical contact take precedence over XY refinement.

The randomized diagnostic was intentionally not scored on target XY.  Of
three fresh attempts, one stopped during spine staging before manipulation,
one stopped after the pad slipped during XY alignment, and one completed the
full contact/rotate/release/retract sequence.  The successful randomized run
reported 58.03 mm target XY error, 0.73 mm Z error, 12.32 mm mesh Z span, and
1.33 deg release orientation error.  This is evidence for the requested
motion/flatness behavior, not a claim of robust randomized target alignment.

Evidence root:

```text
/scratch1/2026_ebim/allen_task2_pi05/evidence/phase1_gt_contact_place_20260821
```

The root Dockerfile was rebuilt from this working tree as
`ebim-task2-pi05-submit:phase1`; the image build completed and its `health`
entry point returned `task2-pi05-submit health: PASS`.  The ROS unit suite
contains 11 tests and passes in the built runtime environment.

## Limitations and Phase II plan

The policy is intentionally a Phase I control baseline.  It is sensitive to
grasp pose, contact friction, and deformable-pad slip.  A randomized object
pose may fail before placement, and final in-plane yaw/centering is not
actively estimated after release.  These are reported as limitations rather
than hidden by a wide success metric.

For Phase II, privileged object poses and pad vertices must be removed.  The
next system should estimate pad/target pose and deformation from the head and
wrist RGB/depth cameras, use tactile/current or visual contact estimation,
and close the loop on target overlap before release.  The present contact,
orientation, release, watchdog, and retract gates can remain as the low-level
safety controller.

## Reproduction

Follow [`PHASE1_SUBMISSION_RUNBOOK.md`](PHASE1_SUBMISSION_RUNBOOK.md).  It
covers a clean Docker build, simulator launch, verified nominal/random reset,
three-run evidence collection, result inspection, and the required submission
declaration.
