# Task 2 PI0.5 V2-full 30k pre-training gate — 2026-08-14

> Superseded: the corrected one-step retry reached `optimizer.step()` and
> confirmed CUDA OOM. See
> `TASK2_PI05_V2_FULL_OOM_AND_EXPERT_30K_FALLBACK_2026-08-14.md`; do not start
> the full 30k command from this historical gate.

## Outcome

**READY to retry the operator-run one-step VRAM smoke; 30k remains NO-GO until
that smoke passes.** The first operator attempt was interrupted during relative
dataset view creation and never reached model construction or an optimizer
step. No checkpoint continuation, Isaac Sim, shadow, GUI, V3, or V4 run was
started.

The new run preserves the V2 data/action contract and phase-balanced sampler,
starts from `lerobot/pi05_base`, changes `train_expert_only` to `false`, keeps
the vision encoder frozen, and saves at steps 15000 and 30000 only. Parser,
dataset, staging-audit, source, and focused unit gates pass. The lab GPU is an
RTX 5090 with only 31.35 GiB, so a manifest-backed one-step smoke is mandatory
before the launcher permits the same-contract 30k command.

## Repository and inputs

- Repository/branch/base:
  `/home/robot/2026_ebim_ssd/benchmark_task2_591def2`,
  `submit@cf6c9ea132b17daa330acee645031dbf352b1552`.
- Bulk root: `/scratch1/2026_ebim/allen_task2_pi05`.
- Raw dataset:
  `datasets/task2_fixpos_200_46ab41f`, 200 episodes / 174719 frames.
- Base model revision: `338b5c22c12dbdd0d2ab19046802de2eb7696a6b`.
- Training image: `ebim-task2-pi05:200-submit-20260812`.
- New run name: `task2_pi05_v2_full_30k`.
- V4 remains offline NO-GO and is not an initialization source.

## Training contract and cost gate

The Draccus/LeRobot parser reports:

- `train_expert_only=false`;
- `freeze_vision_encoder=true`;
- `use_relative_actions=true`;
- arm action indices 3--16 mapped to state 14--27;
- grippers and spine absolute, with mapping index 19 `null`;
- `chunk_size=50`, `n_action_steps=5`, bfloat16, gradient checkpointing;
- V2 learning rate/schedule retained (`2.5e-5`, 500-step warmup,
  12000-step decay), while required total steps/save frequency become
  30000/15000.

The preceding expert-only run measured 693,422,112 trainable parameters out of
4,143,404,816, 12.87 GiB VRAM, and roughly 2.4--2.5 steps/s. Full-mode
trainable count, first-step VRAM, and step time must come from the new smoke
manifest; the gate requires at least 3.5B trainable parameters, finite loss,
and a reported memory value. No automatic LoRA, expert-only fallback, smaller
dataset, or vision unfreeze is allowed.

Current estimates, pending smoke measurement:

- VRAM: 31.35 GiB is high OOM risk; an 80GB-class device is the conservative
  preflight threshold.
- Time: approximately 10--18 hours for 30k, versus a 3.5-hour expert-only
  linear lower bound.
- Disk: approximately 50--60 GB for one relative dataset view, two 9.35 GB
  model files, larger full-mode optimizer states, log, and manifests.
- `/scratch1` has about 1.9 TB free, so disk is not the blocker.

The formal launcher accepts the low-VRAM device only when the named smoke
`run_manifest.json` proves the identical mode, `steps=1`, no checkpoint save,
full parameter count, finite loss, memory metric, and zero return code.

## Interrupted smoke diagnosis and correction

The operator attempt `task2_pi05_v2_full_30k_smoke_1step` stopped after 3.7 GB
of a relative dataset partial view had been created. Its log shows `os.link`
first failing across separate container bind mounts, followed by `copy2` of a
video, then an operator `Ctrl-C`. There is no `run_manifest.json`, model load,
parameter count, loss, memory metric, or optimizer step, so this attempt says
nothing about full-mode VRAM viability. The later `verify-smoke` failure is the
expected missing-manifest consequence, not a second training failure.

The launcher now exposes the dataset, audit, output, and smoke manifest through
one `/data/task2_pi05` mount. Dataset files are owned by `nobody`, so Linux
protected-hardlink rules can still reject a hardlink from the non-root training
user. The relative-view builder therefore falls back to relative symlinks for
immutable videos and copies only small data/metadata files; it never modifies
the raw dataset. The retry uses the new name
`task2_pi05_v2_full_30k_smoke_retry1`, leaving the interrupted evidence intact.

A no-training mechanical probe is preserved at
`evidence/task2_pi05_v2_full_30k_preflight/relative_view_symlink_probe`. It
completed in about 9 seconds with 40 video symlinks, 207 copied non-video
files, zero hardlinks, and `raw_dataset_modified=false`; the resulting view is
about 532 MB rather than another video copy. TorchCodec decoded its first
symlinked frame successfully (`5637` frames, shape `3x720x1280`). This probe
uses episode 176 and exists only to test the filesystem/view mechanism; it is
not a training dataset or a substitute for the V2 parser/smoke contract.

## Startup/pre-grasp dataset audit

Evidence directory:
`/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_v2_full_30k_preflight`.

The audit covers 199 physically auditable episodes (179 train / 20 held out).
It analyzes 31,061 startup frames and 31,888 audited pre-grasp frames. Key
action quantiles (q01/q50/q99) are:

- startup spine: `0.00 / 0.50 / 0.55 m`;
- pre-grasp spine: `0.47 / 0.50 / 0.54 m`;
- both startup and pre-grasp grippers: `1.0 / 1.0 / 1.0` open fraction;
- pre-grasp right J1: `0.163 / 0.703 / 0.884 rad`;
- pre-grasp right J2: `-1.788 / -1.673 / -1.375 rad`;
- pre-grasp right J3: `-2.404 / -1.905 / -1.778 rad`;
- pre-grasp right J6: `3.505 / 3.795 / 4.206 rad`.

The final pre-close episode manifold contains clear robust outliers (largest
scores include episodes 54, 133, and 130). Selection is therefore not a raw
coordinate-wise median or episode 9: the audit chooses the train-split sample
nearest the robust multivariate median across both arms, spine, grippers, left
EE height, right EE position, and the global audited orientation cluster.

Selected provenance is episode 176, frame 408, time 13.6000004 s. Its robust
distance is 0.2100 and right orientation is only 1.0476 degrees from the global
pre-close quaternion center. The command target is:

```text
left arm:
[-0.2692660093, -0.3863082528, 1.3468056917, -2.5714480877,
  0.5722454190,  2.3957438469, 1.4037436247]

right arm:
[ 0.8058103323, -1.6702280045, -1.8443320990, -2.4219164848,
 -0.7698581219,  3.7833797932, -0.4604721665]

left/right gripper open fraction: [1.0, 1.0]
spine command: 0.5 m
```

The measured reference is spine 0.4857426 m, left EE Z 0.9111526 m, and right
EE `[1.7504573, 2.1418896, 0.8728775] m` with quaternion
`[-0.0265317, 0.7274954, -0.6855346, -0.0094230]` (xyzw). The audited dominant
vertical tool axis is local Y. Both arms are within official FR3 position
bounds and both grippers are still open immediately before close.

Across all startup/pre-grasp segments, raw 30 Hz commands contain 100 arm
velocity-limit exceedances in 29 episode/segment records; raw spine target
steps reach roughly 3 m/s. The chosen episode has no raw arm-limit violation
in its startup/pre-grasp segments (max 0.417/1.951 rad/s), but still has the
large discrete spine target steps. Runtime staging therefore follows the raw
episode-176 joint-space waypoints while stretching intervals to 50% of each
official arm velocity limit, 0.05 m/s spine, and 1.0 open-fraction/s gripper.
Its scheduled duration is 23.2678 simulator seconds. It does not smooth or
invent a pose and uses no IK.

## Deterministic staging and live contract

After scene reset and the verified fixed-base route, the new stager publishes
the selected dataset trajectory using `/isaac/clock`. Before dataset frame 0,
it captures the measured arm/spine/gripper state and performs a simulator-time
linear entry transition to that first dataset target under the same velocity
limits. Every entry and route command is finite and position-bounded; no IK or
invented terminal pose is used. Arm/gripper JointState headers use the current
simulator stamp. It holds the final legal command and requires measured
feedback for:

- left/right arm maximum error <= 0.04 rad;
- spine error <= 0.02 m;
- each gripper error <= 0.05 open fraction;
- left EE Z error <= 0.04 m;
- right EE position error <= 0.04 m;
- right EE orientation error <= 12 degrees;
- all checks stable for 1.0 simulator second.

Host monotonic time is used only for the process watchdog. A timeout or another
publisher is a stop. The live runner creates policy publishers only after the
stager exits, discards every staging observation, then requires a newly settled
three-camera + joints + odom + both-EE tuple. It saves the fresh RGB frames as
PPM evidence. Formal publication additionally requires explicit operator
confirmation that the shadow right-wrist frame sees the pad front.

Existing fixed-base pulses, correction waits, readiness dwell, action pacing,
hard5 hold, action queue age, and capture-to-ready latency remain on simulator
time. Camera/state freshness ages and complete capture alignment use simulator
capture stamps; host arrival ages are diagnostic only, and only the process
watchdog gates on host monotonic time. Complete capture skew stays at or below
0.10 s; no threshold was relaxed.

## Validation completed

- Branch/HEAD/collab baseline and clean starting tree: PASS.
- Parser integration for `v2_full_30k`: PASS.
- Absolute spine mapping and frozen vision/full expert flag: PASS.
- Dataset-derived staging audit and legal final target: PASS.
- One-step command dry-run through the phase entry point: PASS; no dataset
  view, optimizer step, or checkpoint created.
- Focused contract/live/staging tests in the pinned image: 71/71 PASS.
- Python compile, shell syntax, and `git diff --check`: PASS.
- GPU doctor: expected NO-GO at 31.35 GiB pending one-step smoke.

No simulator staging execution has run yet. Unit/audit PASS is not a claim that
the live robot reached the target or that the wrist camera sees the pad.

## Operator next action

Follow `task2_isaacsim/baselines/pi05/DEMO_AND_DATASET_REPLAY.md`. The immediate
command is the documented one-step smoke run. If and only if it passes, launch
the documented 30k command with `--vram-smoke-run`. After training, run
`verify-training` and the two-checkpoint `offline-gate`; then test 015000 and
030000 separately through labeled shadow and hard5 outputs.

Rollback is immediate: ignore/remove only the new smoke/run/evidence roots and
continue from the preceding pushed `submit` commit. Existing V1/V2/V4 datasets,
checkpoints, output directories, and raw organizer data were not modified.
