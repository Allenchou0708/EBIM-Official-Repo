# EBiM Task 2 Phase II generalization strategy (2026-08-25)

## Decision

The primary Phase II route is now a **perception-retargeted skill controller**,
not an end-to-end PI0.5 rollout. PI0.5 remains a secondary research candidate
and may later provide bounded local residual proposals, but it must not own
phase transitions or either gripper.

The controller should use an organizer trajectory as a nominal motion prior,
retarget it from policy-facing head/right-wrist cameras and robot state, and
execute it through an observable finite-state machine with explicit gripper
latches. Simulator ground truth and the evaluation camera may be used only for
training labels, diagnostics, and external scoring, never as runtime policy
inputs.

This replaces the previous milestone in which Phase I ground truth produced a
pre-grasp and PI0.5 owned the rest. That milestone was useful for isolating the
learned policy, but it is not the submission architecture we should optimize.

## Phase II constraints and schedule

The team participates remotely. The organizer email states:

- organizer Task 2 trajectories are released on **2026-08-28**;
- the remote submission deadline is **2026-09-02 AoE**;
- remote participation does not change scoring;
- two of three testbeds are selected randomly, three rounds are run per site,
  the best round from each site is retained, and the two site results are
  averaged;
- Phase II is scored fresh and does not carry over the Phase I score.

The practical objective is therefore cross-site repeatability, not tuning one
nominal simulator scene. A retry may use only policy-legal observations. It
must not inspect the evaluator result, evaluation-camera annotations, object
ground truth, or deformable vertices during runtime.

## Current evidence

The strongest PI0.5 checkpoint is still not submission-ready:

| candidate | approach open | grasp close | early/mid/late hold | release open | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| V2 expert 30k | 0.70 | 0.95 | 1.00 / 1.00 / 0.80 | 0.95 | NO-GO |
| V5 gripper-weighted step 100 | 0.75 | 0.90 | 1.00 / 1.00 / 0.80 | 0.95 | NO-GO |
| V5 gripper-weighted step 3000 | 0.95 | 0.70 | 1.00 / 1.00 / 0.45 | 0.95 | NO-GO |

The V2 model predicts a closed gripper under most teacher-forced grasp/hold
observations, yet its closed-loop rollout drifts and reopens. Uniform 8x
right-gripper weighting merely trades opening accuracy against acquisition and
late retention. This is evidence for covariate shift and ambiguous temporal
phase, rather than a single weak output dimension. See
[`TASK2_PI05_V5_WEIGHTED_GRIPPER_LAB_RESULT_2026-08-25.md`](baselines/pi05/TASK2_PI05_V5_WEIGHTED_GRIPPER_LAB_RESULT_2026-08-25.md).

The Phase I deterministic controller is a better motion prior: it already has
audited grasp, lift, transport, place, release, RMPflow assets, and a
right-arm reference trajectory. Its runtime object anchors are privileged and
must be replaced by vision estimates for Phase II. The existing left-arm
behavior is intentionally simple: reach the audited safe pose, remain open,
and hold that pose throughout right-arm manipulation. The new controller must
preserve this ownership instead of allowing PI0.5 to move the left arm.

## Public Task 2 approaches reviewed

These are public, self-reported repository designs, not official Phase II
leaderboard results.

| team | public method | public evidence | lesson / caveat |
| --- | --- | --- | --- |
| HKUST-RockAI | default deterministic phase controller; resolves target and retries; optional PI0.5 route | scripted evidence reports IoU 0.6003; checked optional VLA evidence files were zero | explicit phases and gripper ownership are useful; evaluator-aware target/retry logic must not be copied into a policy |
| EDL | odometry, head-camera pad/board estimate, one of two expert motions, base-drift correction, wrist alignment, RMPFlow retarget | one documented regression reports IoU 0.7948 | strongest public geometric result, but it is one seed and its published randomization is narrower than cross-site Phase II |
| Camelo | demonstration-derived reference trajectory plus online right-arm IK correction | no score published | CPU-friendly and close to the intended primary route; implementation is hidden in the image |
| StarCore SJTU | vision state machine: detect, descend/insert, latch, proximal-joint lift, target detection, IK retarget, release | no multi-seed score published | good contact-aware phase decomposition; part of runtime depends on a private expert trajectory |
| Hajimi | PI0.5 base plus a small frozen DINOv2 residual head trained with DAgger/HIL; bounded right-arm/gripper residual | no score published | best learned-recovery idea; use residual learning only after deterministic phase/gripper control works |
| ROBOT DREAMS | full PI0.5 fine-tune, 50-step chunks with asynchronous prefetch | no task-success evidence published | confirms another pure-VLA route exists, but supplies no reason to prefer it over our failed closed-loop evidence |

Sources:

- submission index: <https://github.com/EBiM-Benchmark/submissions/issues>
- HKUST-RockAI: <https://github.com/cgboy520/ebim-task2-autonomy>
- EDL: <https://github.com/Speidel0402/ebim-task2-submission>
- Camelo: <https://github.com/ostjul/camelo-ebim-task2-submission>
- StarCore SJTU: <https://github.com/StarCoreRobotics/ebim-challenge>
- Hajimi: <https://github.com/sevleete/ebim-task2-submission>
- ROBOT DREAMS: <https://github.com/Jjshi2000/ebim-task2-pi05-submission>

## Primary architecture: Phase2 Hybrid V1

### Runtime information boundary

Allowed inputs after reset:

- head RGB or RGB-D and camera calibration;
- right-wrist RGB or RGB-D and camera calibration;
- left-wrist camera only for safety/visibility if it proves useful;
- joint state, end-effector pose/FK, gripper state, spine state, and odometry;
- elapsed simulator/robot time and the controller's own finite-state history.

Forbidden runtime inputs:

- `/isaac/eval_camera/*` images, depth, semantic labels, or bounding boxes;
- evaluator service result or score;
- thermal-pad/board world pose publishers or deformable vertices;
- GT action after reset, including GT-generated pre-grasp;
- a hidden retry decision selected by an external scorer.

GT remains permitted offline to label perception error and score development
runs. Every runtime manifest must list subscribed topics so this boundary is
auditable.

### Ownership and phases

| phase | right arm | right gripper | left arm / gripper | transition evidence |
| --- | --- | --- | --- | --- |
| S0 stage | odometry/spine and expert safe pose | open | audited expert safe pose, open | odometry, spine, joint/FK tolerance |
| S1 perceive | hold | open | hold | pad lip and target confidence stable for N frames |
| S2 approach/insert | RMPFlow/IK visual servo to retargeted grasp | forced open | hold | wrist alignment and EE pose tolerance |
| S3 acquire | hold/short inward motion | close once, then latch closed | hold | gripper/contact proxy stable; timeout otherwise |
| S4 peel/lift | contact-aware reference waypoints | forced closed | hold | camera lift/retention evidence |
| S5 transfer/place | reference trajectory retargeted to detected target | forced closed | hold | target alignment, support/contact proxy |
| S6 release/retreat | hold, then safe retract | open once support is confirmed | hold | separation and supported-pad evidence |

Transitions are forward-only, hysteretic, and time-bounded. No model output is
allowed to reopen the gripper in S3-S5. A failed acquisition returns to S1 or
stops safely; it must not oscillate between adjacent actions.

### Motion and perception

1. Extract the organizer trajectory into named contact landmarks rather than
   replaying every joint sample blindly.
2. Estimate the pad grasp lip and target center/yaw from head RGB-D. Start with
   color/geometry segmentation because objects are visually distinctive and
   data time is short; retain a learned segmenter as a fallback only if the
   cross-site images invalidate the simple detector.
3. Use right-wrist imagery for the last 30-50 mm of grasp alignment and for
   placement residual correction.
4. Retarget translation and yaw relative to the detected pad/target. Keep the
   demonstrated wrist orientation and contact sequence unless collision or
   reachability checks reject them.
5. Send targets through the existing left/right Lula RMPflow interface with
   rate, workspace, contact-clearance, and joint-limit bounds.
6. Hold the left arm at the audited expert pose. Give it no PI0.5 command and
   no mirrored right-arm command.

### Learned component, only after Hybrid V1 works

Collect DAgger/HIL recovery samples from policy-induced states around approach,
edge insertion, early lift, and final alignment. Train a small bounded residual
head on wrist image + proprioception + nominal action. Initially permit it to
change only the right-arm Cartesian/joint correction; keep phase, gripper,
spine, base, and left arm under the deterministic controller.

PI0.5 may later be compared as an action proposal source under the same
finite-state safety envelope. Do not run broad PI0.5 retraining or online RL
before the geometric controller establishes a task-success baseline and a
validated reward. If RL is later used, train only the residual offline or in
simulation with penalties for loss of grasp, unsafe contact, action jerk, and
incorrect release.

## Execution gates

### Gate A — organizer trajectory audit (after 2026-08-28)

- identify grasp, close, retained lift, transport, supported placement,
  release, and retreat frames;
- compare left/right arm ownership with the Phase I reference;
- verify camera availability and transform conventions;
- save named landmarks and a compact audit, not bulk trajectory data in Git.

### Gate B — perception replay

- run on organizer data and randomized simulator captures without actuation;
- audit using GT only outside the policy;
- require stable pad/target detection on at least 90% of eligible frames;
- require median translation error at most 15 mm and yaw error at most 8
  degrees before live grasp retargeting.

### Gate C — zero-publication shadow

- execute S0-S6 logic with no ROS command publications;
- require legal topics only, fresh synchronized observations, reachable
  targets, forward-only phases, and exactly one close/open latch;
- confirm the left arm remains at its expert safe pose.

### Gate D — simulator progression

Run one attempt at a time and stop at the first failed stage:

1. fixed-seed grasp and retained lift;
2. fixed-seed placement with correct orientation and IoU greater than zero;
3. ten randomized simulator seeds, reporting all runs rather than best-only;
4. go/no-go threshold: at least 8/10 retained grasps, at least 7/10 correct
   orientation with nonzero IoU, and no safety or forbidden-input violation.

### Gate E — learned residual comparison

Only if Gate D identifies repeatable local alignment failures, compare Hybrid
V1 against Hybrid V1 + residual on the same held-out seeds. Adopt the residual
only if it improves task success without increasing safety stops or gripper
phase errors.

## Immediate implementation handoff

Objective: implement and validate **Phase2 Hybrid V1** on the lab simulator,
using the organizer trajectory when it becomes available.

Reuse:

- `baselines/pi05/live/ground_truth_task.py` only as an offline source of
  phase names, safe left-arm ownership, contact order, and reference
  landmarks; remove all runtime object-world-pose and mesh-centroid reads;
- `baselines/pi05/phase_manager.py` as a starting pattern for a forward-only
  observable phase machine, changing it from prompt selection into explicit
  skill ownership;
- `assets/lula/mobile_fr3_duo/*_arm_rmpflow_config.yaml` and existing command,
  freshness, bounds, and contention checks;
- robot head/right-wrist camera topics, never the evaluation camera.

Recommended new module boundary:

```text
task2_isaacsim/baselines/hybrid_v1/
  perception.py       # robot-camera-only pad/target estimates
  reference.py        # organizer landmark loader and retargeting
  phase_controller.py # S0-S6 ownership, hysteresis, latches, timeouts
  live_runner.py      # ROS I/O and evidence manifests
```

Definition of done for the first lab cycle:

- Gate A audit and Gate B replay report;
- focused unit tests for transforms, forward-only transitions, gripper latch,
  forbidden topics, and left-arm hold;
- Gate C zero-publication shadow manifest;
- at least fixed-seed retained lift, or an exact first failed gate with compact
  evidence;
- final commit pushed to the collaboration branch, with no dataset,
  checkpoint, image, video, cache, or full log added to Git.

Stop and request a rules decision before allowing any evaluator, eval-camera,
GT object-state, or post-reset GT action into the runtime policy.
