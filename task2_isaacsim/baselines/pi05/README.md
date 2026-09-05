# Task 2 PI0.5 simulator controller

`run_pi05.sh` is the only supported operator entry point.  It selects one of
three checkpoint contracts while keeping reset, base navigation, spine
staging, RMPflow pre-grasp staging, evidence paths, and shutdown consistent.
Checkpoints, datasets, images, logs, and run evidence remain under
`/scratch1/2026_ebim/allen_task2_pi05/` and are not stored in Git.

## Start the simulator

From this directory:

```bash
./run_pi05.sh models
ROS_DOMAIN_ID=30 ./run_pi05.sh sim-up --gui
```

`models` must report all three profiles as `ready`.  Keep the GUI terminal
open, then run exactly one policy command in a second terminal.

## Three model commands

Our manipulation-only 20k checkpoint uses the head and right-wrist cameras,
8-D right-arm/right-gripper state and action, relative actions, and a 15-action
execution horizon:

```bash
ROS_DOMAIN_ID=30 ./run_pi05.sh run \
  --model ours-20k \
  --execution-horizon 15 \
  --seed 1001 \
  --run-label ours-20k-seed1001-h15
```

To diagnose whether richer language conditioning helps that same checkpoint,
add a rolling, per-frame numeric GT trajectory prompt:

```bash
ROS_DOMAIN_ID=30 ./run_pi05.sh run \
  --model ours-20k \
  --execution-horizon 15 \
  --with_language_gt \
  --seed 1001 \
  --run-label ours-20k-language-gt-seed1001-h15
```

This option changes only the PI0.5 `task` string. It does not replay actions or
subscribe to simulator ground truth. At every policy decision, the prompt lists
the next 15 exact episode-19 outputs as
`[delta_joint1_rad,...,delta_joint7_rad,gripper_open_fraction_absolute]`.
Every joint target in a window is relative to the live measured right-joint
state captured at the start of that inference, matching the checkpoint's
training transform; the gripper remains absolute. The first window is source
frames 370-384, the second is 385-399, and so on through frame 949. These are
the 580 rows in the manipulation-only training crop; the prompt window advances
by the number of actions actually published. The flag raises only this run's
tokenizer limit from the trained value of 200 to 1024 tokens and caps the run
at the 580 available GT actions. Each event in `live_runner_manifest.json`
records `language_gt_frames`, `language_gt_reference_right_joints`, and
`language_prompt`.
Treat this as an out-of-distribution language-conditioning ablation, not action
replay or evidence of generalization: the checkpoint was not trained to parse
numeric trajectory tables.

The submitted 30k checkpoint loads its saved whole-body 20-D transform.  The
normal live mode deliberately publishes only its right-arm/right-gripper
slice; base and spine stay fixed and the left arm stays in its staged hold:

```bash
ROS_DOMAIN_ID=30 ./run_pi05.sh run \
  --model submitted-30k \
  --execution-horizon 15 \
  --seed 1001 \
  --run-label submitted-30k-seed1001-h15
```

The Robot Dreams checkpoint must be tested with its native training contract:
head, left-wrist, and right-wrist cameras; 37-D state; absolute 20-D action;
and asynchronous 50-action chunks.  Its base output is forced to zero after
staging, but both arms, grippers, and spine remain model-owned:

```bash
ROS_DOMAIN_ID=30 ./run_pi05.sh run \
  --model robot-dreams-20k \
  --robot-dreams-native \
  --seed 1001 \
  --run-label robot-dreams-native-seed1001-h50
```

Do not add `--robot-dreams-native` to the other two profiles.  Do not combine
it with `--hybrid-transport`.

## Shared control sequence

All three commands execute this measured sequence before PI0.5:

1. reset the seeded room scene;
2. run the Phase-I `BACK -> STRAFE_RIGHT -> odometry correction` base route;
3. raise and settle the spine;
4. recheck base drift after spine motion;
5. move the left arm to safe hold and the right arm to the demonstrated
   observation pose using collision-aware RMPflow;
6. confirm the right-wrist pad signal;
7. start the selected PI0.5 ownership mode.

The reference pose is a robust development-only summary of 180 unique
training episodes, not a replay of episode 19.  The policy process does not
subscribe to evaluator, object-pose, deformable-object, or other ground-truth
topics.

Use shadow mode to validate checkpoint loading and one inference without any
VLA action publication.  The explicitly requested deterministic staging still
runs:

```bash
ROS_DOMAIN_ID=30 ./run_pi05.sh run \
  --model submitted-30k \
  --shadow \
  --seed 1001 \
  --run-label submitted-30k-contract-shadow
```

## Experimental hybrid transport

The current scheme-1 prototype lets `ours-20k` approach and close on the pad,
waits for measured gripper closure, then transfers command ownership to a
right-only RMPflow retain/lift/transfer/place/release state machine:

```bash
ROS_DOMAIN_ID=30 ./run_pi05.sh run \
  --model ours-20k \
  --hybrid-transport \
  --seed 1001 \
  --run-label ours-20k-hybrid-rmpflow-seed1001
```

This remains experimental and is not the submission default.  The formal
2026-09-03 run passed VLA grasp detection, measured close confirmation,
publisher handoff, and the RMPflow `retain` stage.  It then timed out in
`peel_lift` after 180 wall seconds with the right gripper still closed
(open-fraction 0.112).  Therefore this run is a pipeline milestone, not task
success; the lift pose/orientation and collision path need redesign before
another rollout.

## Measured results (2026-09-03)

| Profile | Verified interface | Simulator observation | Status |
|---|---|---|---|
| `ours-20k`, horizon 15 | two cameras, state/action 8-D, relative-action inverse | grasped the pad, but did not transport it left consistently; stopped at the demonstrated workspace gate after 199 actions | manipulation NO-GO |
| `submitted-30k` | saved 20-D whole-body transform, right 8-D publication slice; live shadow pipeline | one-inference shadow validation publishes no VLA actions | contract-only |
| `robot-dreams-20k --robot-dreams-native` | three cameras, state 37-D, absolute action 20-D | 600 valid actions across 24 decisions with zero invalid actions; pad still did not move left reliably | manipulation NO-GO |
| `ours-20k --hybrid-transport` | VLA grasp followed by measured RMPflow handoff | grasp/retain passed; `peel_lift` did not converge | experimental NO-GO |

These results distinguish a valid runtime contract from task success.  None of
the three PI0.5 checkpoints currently demonstrates reliable end-to-end task
completion in the simulator.

## Research direction

- [Code as Policies](https://code-as-policies.github.io/) supports the useful
  part of scheme 1: a small reactive program can compose perception checks,
  feedback loops, and waypoint primitives.  Here the generated-code idea is
  replaced by a reviewed, bounded FSM; it must still use perception-relative
  lift/place targets instead of the current fixed landmark.
- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
  suggests the stronger learning follow-up: collect states from the deployed
  student's own failed rollouts and provide dense teacher targets there,
  rather than adding more off-policy demonstrations from states the policy
  never visits.
- [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) is an LLM
  token-distillation method, not a drop-in robotics or diffusion-policy loss.
  Its on-policy sampling and clipped dense-divergence ideas are relevant only
  after defining a continuous-action PI0.5 teacher objective and simulator
  safety envelope.

The practical next experiment is therefore perception-relative RMPflow
lifting, followed by an on-policy dataset of PI0.5 failure states labelled by
the simulator controller.  Blindly increasing the same demonstration-only
training steps is lower priority.

## Shutdown and overrides

```bash
./run_pi05.sh down
```

Optional lab-layout overrides are:

```bash
TASK2_PI05_ROOT=/scratch1/2026_ebim/allen_task2_pi05
PI05_LIVE_IMAGE=ebim-task2-pi05-submit:local
TASK2_PI05_DATASET=/absolute/path/to/task2_fixpos_200
TASK2_PI05_STAGING_AUDIT=/absolute/path/to/startup_staging_audit.json
TASK2_PI05_LEROBOT_SRC=/absolute/path/to/lerobot/src
ROS_DOMAIN_ID=30
```

`Ctrl-C` stops the current policy runner.  The Phase-I submission container
entrypoint remains separate because its Dockerfiles depend on it; it is not a
Phase-II operator interface.
