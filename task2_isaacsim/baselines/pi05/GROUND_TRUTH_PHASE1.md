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
3. descend until the pad edge lightly contacts the table;
4. immediately rotate the wrist inward to the demonstrated downward pose;
5. release, wait for the deformable pad to settle, and retract vertically.

The controller does not continue descending to improve XY after contact.  It
also does not claim success merely because a command was sent: lift, contact,
wrist orientation, release, retract, target height, and deformed-mesh flatness
are measured from simulator feedback.

Two explicit evaluation contracts are implemented:

- `nominal`: the unperturbed scene must overlap the target RAM, release, lie
  on the table, and survive gripper retraction.  The bounded center-distance
  gate is 55 mm; the final verified run achieved 28.38 mm.
- `randomized-flat`: object perturbations are treated as a controller
  generalization diagnostic.  Target XY is not a success gate, but contact,
  inward wrist rotation, release, table height, mesh flatness, and retraction
  remain mandatory.

The pad is accepted as flat when its deformed vertices span no more than 20 mm
in world Z after release.  This threshold distinguishes the upright/contact
shape (roughly 100 mm Z span) from a released pad while allowing the visible
elastic curl in Isaac Sim.  The eval-camera image remains the human audit.

## Verified nominal result

On 2026-08-21 a fresh reset reported `randomized: false`, and the nominal
trial ended with `stable_target_place_release_and_retract`:

| measurement | result | gate |
| --- | ---: | ---: |
| maximum lift above target | 167.84 mm | at least 130 mm |
| release/final target XY error | 28.38 mm | at most 55 mm |
| final target Z error | 2.10 mm | at most 12 mm |
| final pad mesh Z span | 13.84 mm | at most 20 mm |
| wrist orientation error at release | 0.53 deg | at most 3 deg |
| release and retract | completed | required |

Evidence is intentionally outside Git:

```text
/scratch1/2026_ebim/allen_task2_pi05/evidence/phase1_gt_contact_place_20260821/nominal/
  controller_result.json
  final_eval_camera.png
```

The image shows the released pad overlapping the red-outlined target RAM and
the gripper retracted.  Earlier randomized runs and calibration failures are
diagnostic only.  In particular, the three older runs documented before this
revision passed loose XY/Z checks but used the wrong pad orientation; they are
not valid evidence for the contact-first controller.

A fresh perturbed scene also passed `randomized-flat` with
`stable_flat_place_release_and_retract`: 167.88 mm maximum lift, 58.03 mm
reported (but ungated) target XY error, 0.73 mm target Z error, 12.32 mm mesh Z
span, and 1.33 deg wrist orientation error.  Its JSON and image are under
`randomized_attempt3/` beside the nominal evidence.  Two preceding diagnostics
were not counted: one spine-staging timeout before manipulation and one
`pad_lost_during_xy_alignment` stop.

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
