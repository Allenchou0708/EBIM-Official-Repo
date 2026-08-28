# EBiM Phase II Task 2 — report_hybrid technical report

Date: 2026-08-28
Status: simulator implementation under gated validation; no real-robot or
generalization claim

## Executive decision

The primary policy is a deterministic hybrid, not an end-to-end PI0.5
rollout. It composes four independently auditable elements:

1. a policy-legal S0 sequence (`base -> spine -> left safe hold -> right
   observation pose`);
2. head and right-wrist RGB-D perception projected through camera FK;
3. online translation/yaw retargeting of demonstration-derived contact
   landmarks through the official Lula/RMPFlow pose-target interface; and
4. a forward-only contact-aware state machine with explicit right-gripper
   ownership.

This decision follows the most repeated pattern in the public Task 2
submissions and directly addresses our measured PI0.5 failure: successful
teacher-forced close/hold predictions did not survive rollout covariate shift,
and increasing gripper loss only traded grasp accuracy against late hold. A
learned residual remains a later ablation after the deterministic baseline has
demonstrated retained grasp and nonzero placement IoU.

## Official task and submission contract

The current official competition page defines Task 2 as peeling, transporting,
aligning, and attaching a deformable thermal pad. Its primary score is Pick
Success × Orientation Success × placement IoU; wrong orientation is zero. It
also states Full Autonomy and force-threshold safety gates. As of this report,
the current page lists the Phase II hands-on window as 25 August–3 September,
organizer-run evaluation as 4–12 September, and the deadline as 12 September
2026 AoE. The benchmark Task 2 README identifies Isaac Sim 5.1.0, PhysX GPU
deformables, the three robot cameras, `config/topics.yaml`, and the development
evaluator. The competition page, not the development evaluator, is
authoritative.

Primary sources:

- official rules and current schedule: <https://ebim-benchmark.github.io/competition.html>
- official benchmark: <https://github.com/EBiM-Benchmark/benchmark>
- official Task 2 runtime: <https://github.com/EBiM-Benchmark/benchmark/tree/main/task2_isaacsim>
- submissions index: <https://github.com/EBiM-Benchmark/submissions/issues>

The policy does not consume evaluation-camera streams or the development
evaluator. The simulator-only external audit may read object truth after a
zero-publication shadow, but its process is isolated and its output never
feeds the policy.

## Public technical-report review

The table records what the public authors claim; it is not an official
leaderboard comparison. Repositories were reviewed at the listed commits on
2026-08-28.

| Team / primary source | Public method | Self-reported evidence | Limitation and decision |
| --- | --- | --- | --- |
| [EDL issue #13](https://github.com/EBiM-Benchmark/submissions/issues/13), [repo](https://github.com/Speidel0402/ebim-task2-submission) (`287c098`) | Camera estimates, two fixed expert motion assets, base-drift compensation, wrist alignment, Cartesian targets through a delivered Lula/RMPFlow bridge | One seed: IoU `0.794776`, correct orientation, coverage `0.95946`, precision `0.82239` | The report explicitly says one regression, not multi-seed performance; packaged randomization changes board XY only, not pad/yaw. Adopt camera + reference + RMPFlow pattern, not its score as a generalization claim. |
| [Camelo issue #12](https://github.com/EBiM-Benchmark/submissions/issues/12), [repo](https://github.com/ostjul/camelo-ebim-task2-submission) (`6cd3c46`) | Demonstration-derived reference trajectory with online right-arm IK correction; CPU container; complete episode runner | Provides batch scoring/evidence plumbing, but the public README does not publish a numeric successful aggregate | Adopt reference + online IK pattern. Its policy source is private in a pinned image, so implementation details are not copied. |
| [StarCore issue #14](https://github.com/EBiM-Benchmark/submissions/issues/14), [repo](https://github.com/StarCoreRobotics/ebim-challenge) (`082c99b`) | Explicit S1–S6 FSM, head RGB-D/HSV lip detection, preserved expert wrist/contact order, Lula IK transport retarget, gripper latch | Design targets ±2 cm/±10° and documents phase-specific gates; public runtime packages a private expert replay for S4–S6 | Adopt explicit phases, preserved contact orientation, and IK retarget. Treat its target robustness as design intent because no public multi-seed success result is supplied. |
| [HKUST-RockAI issue #16](https://github.com/EBiM-Benchmark/submissions/issues/16), [repo](https://github.com/cgboy520/ebim-task2-autonomy) (`20025ab`) | Default route is a scripted pick/carry/place/sweep policy; optional PI0.5 route; repeated attempts and accept ladder | Public evaluator artifacts include multiple correct-orientation positive IoUs; packaged evidence includes `0.6003`. Optional VLA evidence includes zero-IoU cases | Do not adopt evaluator loose-bbox input, evaluator-in-loop retries, or success cherry-picking for a Phase II policy. Adopt only the deterministic phase/gripper ownership lesson. |
| [Hajimi issue #11](https://github.com/EBiM-Benchmark/submissions/issues/11), [repo](https://github.com/sevleete/ebim-task2-submission) (`732283c`) | PI0.5 base plus a frozen, bounded DINOv2 residual trained with DAgger/HIL; deterministic navigate/pregrasp and residual smoothing/fallback | Public README states full autonomous operation and ≥16 GB GPU, but publishes no numeric task-success result | The bounded residual is the best learning follow-up, but it remains secondary until this baseline succeeds. The residual may correct right-arm pose only; never phase, gripper, left arm, base, or spine. |
| [ROBOT DREAMS issue #17](https://github.com/EBiM-Benchmark/submissions/issues/17), [repo](https://github.com/Jjshi2000/ebim-task2-pi05-submission) (`8c169d6`) | Full PI0.5 fine-tune at 20k, 37-D state, 20-D action, 50-step chunks executed at 30 Hz with async prefetch | Reproducible public checkpoint/runtime; no numeric successful Task 2 result in the README | Pure long-horizon VLA does not supply stronger public evidence than the repeated hybrid pattern. Also, the listed policy subscribes to `pad_points`, which is outside this report's Phase II policy boundary. |

No source code from these submissions is copied into this repository. Their
reports informed only the architecture decision and audit questions.

## Dataset-derived reference without episode-19 ownership

The training-data audit read 200 unique successful fixed-position scenarios
(174,719 frames) plus 22 exact legacy duplicates. The immutable split is 180
development / 20 held-out by `source_episode_index % 10 == 7`; duplicates
inherit their original scenario split. The reference library is fit only from
the 180 development scenarios, with support 20/25/90/45 across collections
v1/v2/v3/v4. Episode 19 is one development sample and owns no constant.

Across the 20 held-out scenarios, all phase landmarks and correct right
gripper close/hold/open were recovered. Right-EE reference error was 10.7 mm
p50, 23.2 mm p90, and 49.8 mm worst. The right wrist retained a pad signal in
60/60 audited phase images. Head target visibility was only 60% overall but
20/20 at pregrasp, demonstrating that an odometry-only S0 is incomplete and
that spine/arm observation staging is necessary.

All training scenarios are fixed-position and `randomized=false`, and the
dataset lacks synchronized depth/intrinsics/extrinsics. These facts support a
safe nominal/contact prior; they do not establish position generalization or
the 3-D perception gate.

## Method

### S0 staging

The base target is a transform in the latched initial odometry frame, not an
object/world lookup. In the official room reset frame (yaw −90°), the robust
body-frame delta is approximately `[-0.4632, -2.2999, +0.00031]`. The target delta and all Cartesian poses are robust
development-set summaries. The stages are strictly ordered:

1. Base: bounded discrete commands from initial-relative odometry until
   translation, yaw, and velocity settle.
2. Spine: command 0.50 m and verify the demonstrated measured height near
   0.4857 m from joint state.
3. Left: move through RMPFlow to the robust safe hold; keep the left gripper
   open thereafter.
4. Right: move through RMPFlow to the robust high, non-contact observation
   pose.

S0 then fails closed unless fresh synchronized head/right-wrist RGB-D,
intrinsics, optical-frame camera FK, head target signal, and wrist pad signal
are all available. There is no base or spine command interface in the
manipulation process.

### Perception and retargeting

The current simulator V1 uses conservative appearance priors: a red target
mask in the head image and a neutral-dark pad mask in a bounded right-wrist
ROI. Valid masked depth is deprojected with camera intrinsics and transformed
through camera optical-frame FK. Robust center and in-plane principal axis
yield pad/target XYZ and yaw.

The observed pad offset retargets approach, insert, acquire, and peel-lift.
The target offset retargets place, release, and retreat; transfer blends the
two. Translation is bounded to 80 mm and yaw to ±10°. The demonstrated wrist
orientation and contact order are preserved, with only bounded world-Z yaw
correction. Every target must remain inside the audited RMPFlow workspace.

The color masks are intentionally a minimal simulator experiment, not the
final real-image detector. Once the geometric controller succeeds, real bench
data should replace these masks with a calibrated detector/segmenter while
leaving the control and audit interfaces unchanged.

### Forward-only manipulation

The state machine is:

`observe -> approach -> insert -> acquire -> peel_lift -> transfer_place -> release -> retreat -> done`

`acquire` closes the right gripper exactly once. The close is latched through
peel-lift and transfer/place. A right-wrist retained-pad signal is required
after lift. At supported placement the FSM opens exactly once, then retreats.
No phase can transition backward. Failure stops at the current safe target;
it does not alternate between approach/retract or open/close. Left pose and
open gripper are held constant throughout.

## Safety and policy boundary

- shadow before manipulation publication;
- no `/isaac/eval_camera/*`, evaluator, object pose, pad vertices, deformable
  state, or GT action subscription in the policy;
- camera pose is robot kinematic state, produced from calibrated TF on hardware
  and from robot Camera prim FK in simulation;
- GT/evaluator access exists only in a separate external diagnostic process;
- initial-relative odometry only for base staging;
- command ownership/contention checks before publication;
- bounded base, spine, arm, gripper, retarget translation, and retarget yaw;
- base/spine unavailable after acquire latch;
- exactly one close and one release;
- SIGINT/SIGTERM stop and base `NONE` teardown.

## Validation gates and claims

| Gate | Requirement | Claim rule |
| --- | --- | --- |
| A | This design, primary-source matrix, explicit limitations | Documentation only |
| B | Focused tests: forbidden topics, odom-relative S0, 180-reference support, RGB-D projection, bounded retarget, forward-only FSM, one close/open, constant left hold/workspace | Code-pipeline evidence only |
| C | Live fresh/synchronized RGB-D/FK; target and pad visible; legal topics; reachable targets; zero policy command publication; external median ≤15 mm / ≤8° | Perception/shadow evidence only |
| D | One fixed-seed actuation after B/C: retained grasp, then release/retreat and external correct orientation with IoU >0 | Simulator task evidence |

A passing B or C is not simulator task success. One fixed-seed D success is
not a generalization result. Generalization requires fresh randomized seeds
and later real-hardware evaluation. Results are recorded separately in the
dated lab-result report so failed experiments cannot rewrite this method.

The first implementation result is documented in
[`LAB_RESULT_2026-08-28.md`](LAB_RESULT_2026-08-28.md). It is NO-GO at the
policy-legal base stage; Gate C and task actuation were not reached.

## Learning follow-up

After a deterministic retained-grasp/placement baseline succeeds, collect
randomized and recovery trajectories with DAgger/HIL. Compare a small bounded
right-arm residual against the unchanged deterministic baseline. The residual
must have a stale-output zero fallback and cannot own phase, either gripper,
left arm, base, or spine. Reinforcement learning and PI0.5 knowledge-insulation
changes are not first-line work because neither creates missing recovery-state
coverage by itself.
