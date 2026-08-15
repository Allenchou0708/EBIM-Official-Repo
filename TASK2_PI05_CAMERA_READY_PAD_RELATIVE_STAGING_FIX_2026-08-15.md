# Task 2 PI0.5 camera-ready pad-relative staging fix (2026-08-15)

## Status

- Source/unit gate: PASS before the final entry-order patch (75/75); final patch also passes compile, shell syntax, `git diff --check`, and a focused entry-feedback smoke.
- Dataset target audit: PASS.
- Final shadow gate: PENDING; do not run formal hard5 publication yet.

## Root causes

1. The old target was train episode 176 frame 408, the last pre-close frame. It was too close to the pad.
2. The old feedback compared live and dataset right-EE world coordinates. The first completed shadow showed 0 policy publications and only this group failed (`0.118391 m`); arms, spine, grippers, left height, and right orientation passed.
3. On repeated shadows, the bridge can hold the previous arm target after scene reset. Entry calibration must therefore occur only after measured feedback confirms the robot has returned to dataset route frame 0.

## New contract

- Audit: `/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_camera_ready_pad_relative_20260815/startup_staging_audit.json`
- Selection: train episode 19 frame 310, 30 frames after audited orientation entry.
- Provenance video: `videos/observation.images.wrist_right/chunk-000/file-000.mp4`, timestamp `610.366666 s`.
- Right EE relative to thermalpad: `[0.02251935, 0.31717277, 0.03549623] m` (norm about `0.3199 m`).
- Old frame 408 example was about `0.1932 m` from the pad.
- Command remains the raw dataset joint-space target. No IK is introduced.
- Live feedback subscribes to `/isaac/task2/object_poses`, requires object-pose-to-clock skew `<= 0.10 s`, returns to route frame 0 with measured feedback and sim-time dwell, calibrates the entry EE-to-pad offset, then executes the dataset route.
- Pad-relative tolerance remains `0.04 m`; orientation remains `12 deg`.

## Evidence

- `live_submit_v1-030000-shadow_20260815_000713`: old target, completed watchdog diagnostic, policy publications `0`, world-position-only failure.
- `live_submit_v1-030000-camera-ready-shadow_20260815_003412`: new target diagnostic, manually stopped after exposing the repeated-run entry calibration ordering issue; no final manifest, never use as GO evidence.
- Raw dataset wrist image at episode 19 frame 310 visibly contains the blue thermal pad while the gripper remains farther away than pre-close.

## Required next gate

Run the V1 shadow command in `task2_isaacsim/baselines/pi05/DEMO_AND_DATASET_REPLAY.md`. Require `verify-shadow` PASS and visually inspect `settled_fresh_wrist_right.ppm` before any formal run. Then repeat for V2 30k.
