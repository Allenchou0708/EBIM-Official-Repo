# EBiM Phase I Task 2 policy report

## Executive summary

This work implements a disclosed simulator-ground-truth policy for Task 2,
thermal-pad placement.  The selected Phase I route is organizer Option A
(policy); this report is supporting documentation, not a simultaneous Option B
submission.  A fixed-scene run has been verified end to end: the robot grasps
and lifts the pad, moves it above the memory, makes light table contact,
rotates the wrist inward/downward, opens the gripper, and retracts.  The
released pad overlaps the target RAM and remains on the table.  The submitted
path starts at the scene's initial robot pose and visibly drives the mobile
base before spine and arm staging; it does not teleport directly to the grasp
pose.

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

## Technical-report rubric coverage

This document is the technical report supporting the Option A policy; it must
not be filed as a second Option B submission for the same task.  Nevertheless,
it records the four areas named in the organizer's Technical Report rubric:

- **Method and system completeness (25%)**: the Method and Implementation
  sections describe the implemented end-to-end controller and its safety
  gates, rather than only a proposed design.
- **Simulation and task completion (30%)**: Experiments and Results reports
  all three nominal attempts, their mean, and the associated machine-readable
  and visual evidence.
- **Policy and code readiness (25%)**: the root Dockerfile, README, health
  command, 15 tests, and runbook define a reproducible execution path.
- **Real-world deployment readiness (20%)**: the Phase II section below
  distinguishes reusable control components from work that is still missing.

The last category is currently a plan, not a claim of real-robot readiness.

## Method

The controller combines a demonstration-derived grasp/transport prior with a
feedback-gated Cartesian placement state machine.  Episode 19 of the audited
Task 2 dataset provides safe joint landmarks and two wrist orientations:
pre-contact and downward placement.  Live ground truth anchors the base and
target yaw.

Joint-space transport now hands off at demonstrated frame 544, the last
landmark where the pad was consistently fully supported.  A later frame-560
handoff intermittently unloaded the deformable grasp before Cartesian
alignment.  The remaining target motion is bounded at 0.10 m/s.

Startup is staged in the same order a reviewer sees in the GUI.  The robot
first drives from approximately `(4.4, 2.6)` to the live GT-derived grasp base
through BACK, right-strafe, braking, and odometry correction phases.  It then
raises and settles the spine.  The main controller latches measured odometry,
performs only a bounded residual base trim at 0.10 m/s and 0.30 rad/s, and
only then moves the arm to dataset frame 399.  The former direct live-grasp
root preposition remains a legacy CLI option but is disabled by the launcher.

The key placement change is contact-first control.  XY is aligned while the
pad is safely elevated.  The gripper then descends only until the pad edge is
near first table contact; it does not chase a precise Z or keep descending to
repair XY.  Contact causes an immediate bounded quaternion interpolation
toward the demonstrated inward, downward wrist pose at constant commanded Z.
There is no post-contact wrist drop.  The official launcher also rejects a
contact transition below the scene-audited `0.903 m` EE-Z clearance floor,
allows only 1 mm of feedback noise, and commands a 19 mm tracking margin
above that floor.  The nominal pre-contact point is additionally shifted
40 mm toward negative Y: the inward wrist rotation moves the released pad
toward positive Y, so this compensation places the pad center, rather than
its upper edge, near the memory centerline.  Only after orientation and
table-height gates pass does the gripper open.  A vertical retract prevents
dragging the released pad.

Nominal and randomized trials use different acceptance contracts.  Nominal
requires target overlap.  Randomized trials measure whether the manipulation
sequence remains physically correct and the released pad lies flat, without
pretending that an open-loop controller can perfectly cancel arbitrary object
perturbations.

## Implementation and safety

Primary files:

- `live/ground_truth_joint_lift.py`: controller, gates, JSON evidence;
- `live/fixed_stage_base.py`: live-GT base target, physical pedal route,
  odometry correction, and settle evidence;
- `live/run_ground_truth_random_gui.sh`: nominal/random simulator launcher,
  verified reset, base/spine staging, exact-attempt runner, evidence mount;
- `tests/test_ground_truth_joint_lift.py`: transform, bounded-step,
  quaternion, contact-offset, and release-contract tests.

Safety behavior includes command-subscriber ownership checks, joint
preposition tolerance, grasp dwell, lift and frame-520 transport checkpoints,
bounded Cartesian displacement/speed, simulator-time pacing, an EE-height
clearance gate, a table-height release gate, an open-gripper requirement, and
measured retract completion.  The process exits nonzero on failed physical
gates and writes the reason to JSON.

## Experiments and results

Episode 19 was inspected at frames 530, 600, 700, 800, 834, 835, and 900.
The source data confirmed that the pad first approaches the table while held
upright, the wrist rotates downward before frame 835, the gripper then opens,
and the pad settles after retraction.  Source mesh Z span falls from about
120 mm while held upright to about 2.7 mm after settling.

Three fresh unperturbed-scene validations completed the full policy and were
captured by the repository eval-camera service.  Runs 1 and 2 used the earlier
stricter 55 mm internal release gate; their 17.38 and 20.46 mm errors follow
the identical path under the final 60 mm gate.  Run 3 used the final gate.
The organizer will independently run the submitted container three times and
take the mean.

| run | controller | orientation | IoU | final XY | final Z span | minimum contact-rotation EE Z |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | pass | correct (`liner_only`) | 0.28750 | 17.38 mm | 4.13 mm | 0.91648 m |
| 2 | pass | correct (`liner_only`) | 0.02429 | 20.46 mm | 3.82 mm | 0.91671 m |
| 3 | pass | correct (`liner_only`) | 0.43353 | 12.18 mm | 3.21 mm | 0.91672 m |
| mean | 3/3 | 3/3 | **0.24844** | **16.67 mm** | **3.72 mm** | **0.91664 m** |

The saved video and 0.5-second contact sheet confirm that the pad reaches the
table while a visible gap remains below the fingertips, then the gripper opens
and retracts.  A deliberately audited lower run reached contact at EE Z
`0.8952 m`, squeezed the pad 142 mm off target, and visually brought the
fingertips to table height; it was rejected.  Raising the transition, removing
the 5 mm post-contact drop, retaining the slower 0.10 m/s Cartesian alignment,
and applying the nominal-only Y compensation produced the passing controller.
The eval-camera results also expose the remaining limitation: small in-plane
deformation or narrow-axis shift causes a large IoU change even when the GT
centroid error remains 12--21 mm and the mesh is flat.

The randomized diagnostic is intentionally not scored on target XY.  A
post-base-rework perturbed run completed the full
contact/rotate/release/retract sequence with 189.71 mm ungated target XY
error, 2.02 mm target Z error, 14.10 mm mesh Z span, and 3.28 deg release
orientation error.  That run preceded the final nominal clearance refinement,
so it is diagnostic evidence rather than a substitute for fresh formal
current-configuration trials.

The formal three-run scoring evidence is not a video.  It contains the
base/spine/controller JSON records and, for each run, eval-camera RGB, IoU,
depth, segmentation, and bounding-box artifacts:

```text
/scratch1/2026_ebim/allen_task2_pi05/evidence/phase1_gt_formal_20260821/formal_current
```

A separate representative video shows the centered, clearance-safe placement
sequence, including contact, inward wrist rotation, release, and retract:

```text
/scratch1/2026_ebim/allen_task2_pi05/evidence/phase1_gt_physical_base_20260821/
  nominal_y_centered_safe/place_stage.mp4
  nominal_y_centered_safe/contact_sheet.png
  nominal_y_centered_safe/contact_detail.png
```

The video is 22 MB and its matching controller JSON reports success, 20.42 mm
final target-center error, 4.49 mm mesh Z span, and 0.91602 m minimum observed
EE Z during contact rotation.  It is representative visual safety evidence;
the three `formal_current/run_*` directories remain the unselected numerical
result set.  Before filing the issue, upload the video and the compact formal
JSON/RGB artifacts to a public attachment or link because `/scratch1` is a
lab-local path.

The ROS unit suite contains 15 tests.  On 2026-08-21 the root Dockerfile was
rebuilt from this final working tree as `ebim-task2-phase1-gt:latest`; its
source-independent `health` check and all 15 baked-source tests passed.  A
fresh public clone should repeat those same checks after the final commit is
pushed.

## Limitations and Phase II plan

The policy is intentionally a Phase I control baseline.  It is sensitive to
grasp pose, contact friction, and deformable-pad slip.  A randomized object
pose may fail before placement, and final in-plane yaw/centering is not
actively estimated after release.  These are reported as limitations rather
than hidden by a wide success metric.

Phase II is still a substantial integration effort; the current result should
not be read as nearly deployable on the physical robot.  The intended system
is a hybrid pipeline rather than asking one model to solve navigation,
localization, manipulation, and safety simultaneously:

1. **Perception and localization.**  Replace simulator object GT with
   calibrated head/wrist RGB-D.  A detector or segmenter such as YOLO can
   locate the thermal pad, memory target, hands, and relevant work surface.
   Robot/table localization must be supplied by the physical platform's
   odometry and, where global or drift-resistant localization is needed, a
   SLAM or fiducial-assisted estimate.  Depth and camera-to-base extrinsics
   then convert detections into uncertainty-aware robot-frame goals.
2. **Deterministic approach and staging.**  Reuse the current visible control
   structure to drive the base from its initial pose, settle the spine, and
   servo the arms to a verified collision-safe pre-grasp posture.  The Phase I
   GT anchor is replaced by perception/localization feedback; the present
   bounded motion, settle, timeout, and clearance checks remain useful.
3. **VLA manipulation handoff.**  At the verified pre-grasp state, hand control
   to a physical-robot-adapted PI0.5 VLA for grasp, deformable-pad transport,
   contact-aware rotation, placement, and recovery.  Integrate
   [Real-Time Chunking (RTC)](https://www.pi.website/research/real_time_chunking)
   so the next flow-policy action chunk can be inferred while the committed
   part of the current chunk executes, reducing pauses and discontinuities at
   chunk boundaries without retraining the base policy.
4. **Independent safety supervisor.**  Keep collision bounds, joint and EE
   limits, stale-sensor watchdogs, gripper/contact gates, operator stop, and
   controlled retract outside the VLA.  The learned policy may request motion,
   but it must not be able to bypass these constraints.

The principal missing work is real sensor calibration and object
segmentation, non-GT base/target localization, physical action-space and robot
interface validation, PI0.5 checkpoint adaptation with real demonstrations,
RTC integration and latency measurement, and contact/safety validation on the
bench.  A staged Phase II validation should therefore first prove perception
and deterministic pre-grasp repeatability, then shadow-test PI0.5+RTC without
actuation, then enable grasp/placement under the safety supervisor, and only
afterward measure repeated end-to-end success.  The current GT controller can
serve as a demonstration generator and a behavior reference, but cannot be
the Phase II perception solution.

## Reproduction

Follow [`PHASE1_SUBMISSION_RUNBOOK.md`](PHASE1_SUBMISSION_RUNBOOK.md).  It
covers a clean Docker build, simulator launch, verified nominal/random reset,
three-run evidence collection, result inspection, and the required submission
declaration.
