# Task 2 PI0.5 V2-full OOM and expert-only 30k fallback — 2026-08-14

## Outcome

V2-full is **NO-GO** on the lab RTX 5090. The same-contract one-step retry
loaded the complete model and reached the first `optimizer.step()`, then failed
with a CUDA out-of-memory error. Per the operator decision, the active
experiment is now a fresh V2 expert-only 30k run named
`task2_pi05_v2_expert_30k`. No expert-only optimizer step or 30k training was
started during this implementation pass.

## Full-mode failure evidence

Evidence root:
`/scratch1/2026_ebim/allen_task2_pi05/outputs/task2_pi05_v2_full_30k_smoke_retry1`.

- return code: `1`;
- trainable parameters: `3,730,962,464`;
- total parameters: `4,143,404,816`;
- failure location: first Adam `optimizer.step()`;
- GPU capacity: `31.35 GiB`;
- free at failure: `119.75 MiB`;
- PyTorch allocated: `29.93 GiB`;
- attempted allocation: `20 MiB`;
- completed optimizer steps: `0`;
- loss/memory training metrics: absent because step 1 never completed.

This is a capacity failure, not the earlier interrupted dataset copy and not a
valid checkpoint. The failed full output is preserved as evidence and is not a
continuation source.

## Authorized fallback contract

The fallback changes only the parameter-training mode relative to the V2-full
plan:

- `train_expert_only=true`;
- `freeze_vision_encoder=true`;
- fresh `lerobot/pi05_base` revision
  `338b5c22c12dbdd0d2ab19046802de2eb7696a6b`;
- V2 phase-balanced sampling;
- relative arm actions with absolute grippers and absolute spine;
- relative mapping index 19 remains `null`;
- `n_action_steps=5`, live execution `hard5` indices 0--4;
- 30,000 steps, `save_freq=15000`, numbered checkpoints only at 015000 and
  030000;
- unchanged deterministic dataset-derived staging and GUI GO/NO-GO gates.

New source interfaces:

- config: `configs/task2_fixpos_200_v2_expert_30k.yaml`;
- profile: `profiles/v2_expert_30k.yaml`;
- run: `task2_pi05_v2_expert_30k`.

The preceding V2 expert-only measurement was 693,422,112 trainable parameters,
12.87 GiB VRAM, and about 2.4--2.5 steps/s. The expected 30k duration is about
3.4--4.0 hours, with approximately 25--35 GB for the relative view, two model
and optimizer checkpoints, log, and manifests.

## Validation and next boundary

Completed source gates:

- parser integration: PASS with expert-only `true`, frozen vision `true`, and
  V2 mapping index 19 `null`;
- GPU doctor: PASS on the 31.35 GiB RTX 5090 for mode `v2_expert_30k`;
- exact 30k dry-run: PASS with the complete 180-episode train split, phase
  entry point, new output root, 30,000 steps, and 15,000 save cadence from the
  profile; no output or optimizer step was created;
- focused contract/live tests in the pinned image: 73/73 PASS;
- Python compile, shell syntax, and `git diff --check`: PASS.

The operator starts the foreground training only after the source commit is
pushed. After completion, `verify-training --mode v2_expert_30k` must confirm
the two checkpoints, expert-only parameter count, config mapping, finite loss,
and final step before any offline/shadow/GUI gate.
