# Task 2 Phase I ground-truth controller

## Submission status

The organizer email received on 2026-08-19 permits simulator ground-truth
positions and states in Phase I when their use is declared truthfully.  The
reopened deadline is Aug 22, 2026 AoE (Aug 23 11:59 UTC / 19:59 Taipei).
Ground truth is not permitted as a replacement for onboard perception in the
Phase II physical evaluation.

Recommended submission choice: **Option A, policy with disclosed simulator
ground truth**.  Do not also submit the supporting report as an Option B entry;
the email allows one option per team and task.  The extension weight stated in
the email is 0.90 for a ground-truth policy, 0.95 for a policy using the team's
own perception, and 0.65 for a Technical Report.

## Current controller

[`live/ground_truth_joint_lift.py`](live/ground_truth_joint_lift.py) uses the
simulator's `thermalpad`, `board_target`, deformed pad vertices, joint state,
and right end-effector pose.  Episode 19 supplies the grasp and transport
landmarks.  Placement then follows a contact-first sequence:

1. carry the pad above the target without over-constraining Z;
2. move to a calibrated pre-contact XY point while preserving the grasp;
3. descend until the pad edge lightly contacts the table, subject to the
   nominal-scene EE-Z clearance floor;
4. immediately rotate the wrist inward at constant commanded Z to the
   demonstrated downward pose;
5. release, wait for the deformable pad to settle, and retract vertically.

The controller does not continue descending to improve XY after contact and
does not apply a post-contact wrist drop.  It also does not claim success
merely because a command was sent: lift, contact, wrist orientation, release,
retract, target height, and deformed-mesh flatness are measured from simulator
feedback.

Two explicit evaluation contracts are implemented:

- `nominal`: the unperturbed scene must overlap the target RAM, release, lie
  on the table, and survive gripper retraction.  The bounded center-distance
  gate is 60 mm.  A nominal-only 40 mm negative-Y pre-contact compensation
  accounts for the positive-Y shift during inward wrist rotation.  Cartesian
  control begins at dataset frame 544 to avoid unloading the pad at frame 560.
- `randomized-flat`: object perturbations are treated as a controller
  generalization diagnostic.  Target XY is not a success gate, but contact,
  inward wrist rotation, release, table height, mesh flatness, and retraction
  remain mandatory.

The pad is accepted as flat when its deformed vertices span no more than 20 mm
in world Z after release.  This threshold distinguishes the upright/contact
shape (roughly 100 mm Z span) from a released pad while allowing the visible
elastic curl in Isaac Sim.  The eval-camera image remains the human audit.

## Verified nominal results

On 2026-08-21 three fresh resets reported `randomized: false`; all three
trials ended with `stable_target_place_release_and_retract` and the eval-camera
reported the correct `liner_only` orientation:

| run | IoU | final target XY error | final mesh Z span | minimum contact-rotation EE Z |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.28750 | 17.38 mm | 4.13 mm | 0.91648 m |
| 2 | 0.02429 | 20.46 mm | 3.82 mm | 0.91671 m |
| 3 | 0.43353 | 12.18 mm | 3.21 mm | 0.91672 m |
| mean | **0.24844** | **16.67 mm** | **3.72 mm** | **0.91664 m** |

Runs 1 and 2 used the preceding stricter 55 mm release gate and are
behavior-compatible with the final 60 mm gate because both were already below
21 mm.  Run 3 used the final gate.  Official results remain whatever the
organizer obtains from its own three container runs.

Evidence is intentionally outside Git:

```text
/scratch1/2026_ebim/allen_task2_pi05/evidence/phase1_gt_formal_20260821/formal_current/
  formal_summary.json
  run_1/{base_stage.json,spine_stage.json,controller_result.json,eval/}
  run_2/{base_stage.json,spine_stage.json,controller_result.json,eval/}
  run_3/{base_stage.json,spine_stage.json,controller_result.json,eval/}
```

The eval-camera images show the released pad on the memory with the correct
liner face visible.  The measured EE stayed at or above 0.91648 m throughout
contact rotation.  A lower diagnostic
run at 0.8952 m contact EE Z brought the fingertips to table height and pushed
the pad 142 mm off target; it is rejected evidence.  Earlier randomized runs
and calibration failures are diagnostic only.

A fresh post-base-rework perturbed scene passed `randomized-flat` before the
final nominal clearance refinement: 189.71 mm reported (but ungated) target
XY error, 2.02 mm target Z error, 14.10 mm mesh Z span, and 3.28 deg wrist
orientation error.  Its JSON and image are under `randomized/` beside the
nominal evidence.  It demonstrates the requested action/flatness behavior but
does not replace fresh current-configuration formal trials.

## Known limitations

- Open-loop joint landmarks and pre-contact compensation have poor
  generalization.  A perturbed grasp can slip before placement.
- Contact friction makes center placement vary by centimeters even in the
  nominal scene.  The verified result is overlapping and task-successful, not
  millimeter-accurate pose estimation.
- The method consumes privileged simulator state and cannot be used unchanged
  in Phase II.  Replace object poses and pad vertices with onboard RGB/depth
  perception and retain the measured contact/release safety state machine.

See [`PHASE1_POLICY_REPORT.md`](PHASE1_POLICY_REPORT.md) for the supporting
report and [`PHASE1_SUBMISSION_RUNBOOK.md`](PHASE1_SUBMISSION_RUNBOOK.md) for
the clean-build and evaluation procedure.
