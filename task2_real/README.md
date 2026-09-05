# Phase II Task 2 real-robot branch

Status: dataset/training gates and a dependency-light runtime safety gate are
implemented; no hardware actuation has been validated

This branch retires the simulator-only pure-control manipulation experiment.
It keeps the part that transferred cleanly across our simulator runs and
assigns each actuator to one owner:

`base navigation -> spine -> left safe hold -> right observation pose -> PI0.5`

PI0.5 owns only the right arm and right gripper. Base, spine, the left arm,
and the left gripper remain deterministic and cannot be emitted by the model.

## Organizer handoff facts

The released Munich dataset is
[`ebim-benchmark/ebim_task2_realrobotdata`](https://huggingface.co/datasets/ebim-benchmark/ebim_task2_realrobotdata)
at revision `495ebb7b56fb9e2f3952398a63d86f08cacb9531`:

- 238 episodes, 144,965 frames, 20 Hz, and about 2.01 hours;
- head RGB at 1280x720 and two wrist RGB streams at 640x480;
- 42-D raw state and 17-D raw action;
- recordings begin at the table, so they contain no route from the site start;
- only Munich hardware is represented, while evaluation selects two of three
  sites and scores the best of three runs at each site.

The raw metadata is not internally self-consistent. `info.json` declares
42-D state and 17-D action, while `modality.json` state slices end at 56 and
action slices end at 16. The extra raw action is a spine target; a sampled
episode stores it as the constant `434.0`, while no spine observation is in
the 42-D state. The release also contains failed conversions and per-frame
`annotation.human.validity`; invalid frames/episodes must be excluded before
any split or normalization statistics are computed.

Run the metadata audit after downloading the two small metadata files:

```bash
python3 -m task2_real.contract \
  --info /data/task2_munich/meta/info.json \
  --modality /data/task2_munich/meta/modality.json
```

The default exits `2` because the raw release is ambiguous. Add
`--acknowledge-documented-conflicts` only when using the exact right-arm-only
adapter locked in [`contract.json`](contract.json). Any future dataset revision
must be audited again.

## Runtime ownership and gates

[`runtime_core.py`](runtime_core.py) defines the handoff and right-action
safety boundary without importing ROS. [`runtime_shadow.py`](runtime_shadow.py)
evaluates a captured state tuple and always reports zero command publications;
it is intended to fail closed before an on-site ROS publisher is enabled.

Before filling any site calibration value, run the subscriber-only interface
capture on the evaluation robot:

```bash
python3 -m task2_real.ros_preflight_capture \
  --duration-s 10 \
  --output /data/evidence/ros_preflight.json
```

It records actual JointState names/order/fields and the raw spine/gripper
values needed to determine their scales, camera shape/encoding, lidar
availability, external wrench fields, discovered
publisher endpoint types, and command publisher counts. It never creates a
publisher and does not authorize handoff; the returned evidence must be
reviewed before updating a site profile.

### 1. Base navigation

The top-right Task 2 room has a right-side 1.2 m doorway and a narrow table
approach. The organizer trajectories do not contain this segment. Reuse the
Phase-I route semantics—clear the start footprint, execute an ordered lateral
or forward corridor, brake, then perform a short odometry correction—but do
not reuse simulator world coordinates.

Each site needs a calibration profile containing an initial-relative final
pose and waypoint corridor. `/mobile_base/pose`, swerve odometry, and both
lidars close the loop. A timeout, stale lidar, unexpected obstacle, command
publisher contention, or failure to settle publishes zero twist and blocks
all later stages.

### 2. Spine and left safe hold

The real data indicates a constant raw spine target of `434.0`, but the topic
reference does not state its unit. Before commanding it, compare a bounded
test command with `/spine/joint_states` and record the unit and limits. Then
fit the final spine and collision-safe left-arm pose from all valid real
episodes, not from a simulator pose or one episode. Hold the left gripper open.

The sampled real trajectories contain non-trivial left-arm motion even though
the left gripper remains fixed. We intentionally do not imitate that motion:
the deterministic hold must be checked for collision clearance at every site,
and PI0.5 is denied the left command topics.

### 3. Right observation pose and handoff

Fit a robust pre-grasp right-joint pose from valid real trajectories and
approach it under joint velocity, torque, and wrench limits. Handoff requires:

- base zero and settled;
- spine stable at the calibrated height;
- left arm at the safe hold and left gripper open;
- right pose within tolerance;
- fresh head and right-wrist RGB plus fresh measured right joints/gripper;
- exactly one publisher for each command topic.

Discard every observation captured during staging. PI0.5 starts from the first
fresh post-settle tuple.

### 4. Right-arm-only PI0.5

The policy view deliberately uses two cameras, 8-D right proprioception, and
8-D right commands. Raw state indices 21-28 are right joints plus gripper; raw
action indices 8-15 are the matching GELLO targets plus gripper percentage.
External torque/wrench signals stay outside the model and act as independent
safety stops. The raw left and spine actions are dropped.

This removes the left-control discrepancy seen in the simulator PI0.5 runs
and prevents covariate errors in base/spine from accumulating inside a long
VLA action chunk. The runtime gate now rejects stale output, out-of-range or
over-large joint steps, publisher contention, and force/torque limit events;
it exposes at most five actions from one decoded chunk.

### Site calibration and zero-publication shadow gate

[`site_profile_munich.template.json`](site_profile_munich.template.json)
contains train-derived staging candidates, not verified commands. It is
deliberately unarmable until on-site staff record the initial-relative base
corridor, resolve the spine unit, enter controller joint/force limits, and
collision-check both staging poses. Simulator coordinates are not accepted as
real-site calibration.

Capture a post-settle state tuple using
[`handoff_snapshot.template.json`](handoff_snapshot.template.json), then run:

```bash
python3 -m task2_real.runtime_shadow \
  --profile /data/site_profile.json \
  --snapshot /data/handoff_snapshot.json \
  --output /data/evidence/runtime_shadow.json \
  --now <same-monotonic-clock-seconds>
```

The gate returns ready only when staging errors are within tolerance, all four
policy inputs are fresh, the cameras are synchronized, command-topic publisher
counts match the ownership contract, and external torque/wrench signals are
below calibrated limits. It does not publish a ROS command.

## Training decision

The $6,000 Google Cloud credit is enough for both controlled ablations and a
full run; compute is not the limiting factor. The limiting factors are the
single-site two-hour dataset, invalid samples, schema adaptation, and the
five-day remote deadline.

Use this order:

1. **Expert-only baseline (primary):** keep the VLM frozen and train the action
   expert/projections using only valid frames in the right-only view. This is
   already supported by our pinned LeRobot stack and is the lowest-risk route.
2. **LoRA ablation:** use a separate current-LeRobot image only after a
   one-batch load/train/save/reload smoke test. PI0.5 has no policy-specific
   default target list, so explicitly target the action expert attention and
   action projections. Do not silently treat this as equivalent to the pinned
   expert-only implementation.
3. **Full fine-tune:** run only if the expert-only and LoRA held-out gates show
   inadequate visual adaptation. Use one 80 GB-class GPU, bfloat16, gradient
   checkpointing, gradient clipping, a low backbone learning rate, and early
   checkpoint evaluation. Full fine-tuning a single-site dataset first would
   be the highest-risk choice for cross-device generalization.

Freeze an episode-level, time-blocked held-out set before training. Do not
randomly split frames from the same demonstrations. Compare every checkpoint
on the same held-out episodes and inspect phase landmarks, right-gripper
close/hold/open behavior, action bounds, and prompt sensitivity before any
real publication.

Gate A has now audited all 238 released parquet episodes. It retained 149
fully-valid episodes (86,444 frames), split as 119 train and 30 later-index
held-out episodes; the 89 excluded episodes are structurally readable but
visually show failed or poor placement outcomes. A 200-step expert-only smoke
run reduced fixed held-out diffusion loss from 0.22244 to 0.10482. However,
fresh-reload eight-seed landmark evaluation reached only 9/72 exact gripper
phases and 0% on the closed landmark, so its delta is evidence of learnability,
not a deployable checkpoint.

The proposed stop-gradient plus auxiliary discretized-action objective is a
research architecture change, not a switch in the current training code.
`train_expert_only=true` already blocks action-loss gradients from the VLM,
but it also prevents real-image adaptation. An auxiliary loss could preserve
representation quality during full fine-tuning, yet it cannot repair missing
cross-site data or a wrong state/action adapter. It is therefore deferred
until the baseline contract and held-out evaluation pass.

## Current non-claims

- No real command has been published by this branch.
- The simulator base route is not a calibrated real-site route.
- The 200-step real-data expert delta is not deployable and has not passed a
  hardware shadow, closed-loop replay, or task-success test.
- Dataset success, cross-site generalization, and real task success are not
  established by metadata or offline loss alone.
