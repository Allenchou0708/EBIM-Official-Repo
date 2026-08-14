# Task 2 PI0.5 simulator-clock and GUI crash fix — 2026-08-14

## Outcome

PASS. The submit runtime now keeps base-stage timing, camera/state capture
alignment, and action header stamps on Isaac simulator time. PI0.5 is loaded
before moving the base, so the robot no longer waits at the staged pose during
the roughly three-minute checkpoint load. GUI output is persisted, and the
reproduced Kit crash was traced to Fast DDS exhausting the Isaac container's
64 MiB `/dev/shm`; the launcher now uses UDPv4 and no longer adds Fast DDS SHM
files.

The operator-selected comparison is V1 versus V2. V3 is retained as historical
evidence only because its stage behavior differed and its load time did not
improve. V4 training and the seven-prompt phase manager are explicitly deferred
to a new session.

## Implemented runtime changes

- `fixed_stage_base.py` subscribes to `/isaac/clock`. Motion pulses, braking,
  correction intervals, and settle duration use simulator time; only the
  maximum-duration safety watchdog remains on host monotonic time.
- Base staging runs after policy construction. Staging-time camera/state samples
  are discarded, followed by a fresh readiness gate before inference or any
  arm/spine publication.
- Camera and all four required state streams (`joints`, `odom`, `ee_left`,
  `ee_right`) are aligned using ROS message-header simulator stamps. The gate
  rejects a full observation capture skew over `0.10 s`.
- Published action `JointState` headers use the latest `/isaac/clock` stamp.
- `sim-up --gui` saves complete terminal output and the launcher exit code under
  `/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_200_submit_20260812/launcher`.
- The Isaac ROS bridge uses `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`. ROS peers are in
  separate containers, so shared-memory transport supplied no local benefit.
- The GUI runbook now gives matching V1 and V2 commands and historical model
  references; it no longer recommends V3 for the next experiment.

## Clock-alignment evidence

Live zero-publication shadow manifest:

`/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_20260814_160449/live_runner_manifest.json`

- manifest valid: `true`;
- published commands: `0`;
- camera stamps: head `46.183335741`, left/right wrist `46.216669077`;
- all state stamps: `46.150002406`;
- inter-camera skew: `0.033333336 s`;
- complete camera/state capture skew: `0.066666671 s`;
- capture-to-ready simulator time: `0.316666683 s`.

Final base-stage gate:

`/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_pi05_stage_clock_20260814/fixed_base_sim_clock_smoke_retry.json`

- success: `true`;
- clock: `simulator`, topic `/isaac/clock`;
- elapsed: `58.2658 s` wall, `10.7500 s` simulator;
- final position error: `0.0270746 m` (limit `0.03 m`);
- final yaw error: `0.0376966 rad` (limit `0.04 rad`);
- final speed: `0.0064788` (limit `0.02`);
- route reached `COMPLETE` after back, right strafe, braking, odometry
  correction, and settle phases.

An earlier mechanically converted smoke used correction pulses that were too
long in simulator time and oscillated until the wall watchdog. It is preserved
at `fixed_base_sim_clock_smoke.json`. The verified defaults are `0.05 s` for a
correction pulse, `0.10 s` for a straight correction pulse, and `0.10 s` for a
stop interval, all in simulator time.

## GUI crash diagnosis and verification

Reproduction log:

`/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_200_submit_20260812/launcher/isaac_gui_20260814_162820.log`

Kit exited with `Bus error` before the bridge became ready. At that point the
long-running `isaac-sim-5-1-0-workshop` container had a private 64 MiB
`/dev/shm`, 100% usage, and hundreds of stale `fastrtps_*` entries accumulated
over several days. Restarting the inactive simulator container cleared SHM and
preserved the repository mount.

Successful post-cleanup log:

`/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_200_submit_20260812/launcher/isaac_gui_20260814_162959.log`

Successful final UDPv4 log:

`/scratch1/2026_ebim/allen_task2_pi05/evidence/task2_200_submit_20260812/launcher/isaac_gui_20260814_163538.log`

Both successful logs reached bridge ready and exited cleanly with code `0` on
Ctrl-C. During the final UDPv4 run the Fast DDS file count stayed unchanged at
48; no new SHM transport files were created.

## Validation

- Focused PI0.5 contract/live tests: `64/64` passed.
- Shell syntax for both launchers: passed.
- Python compile for the sim-clock stager: passed.
- Runtime-image ROS import smoke: passed.
- `git diff --check`: passed.

## Next-session boundary

Do not start V4 from this result automatically. The next session should first
review this report and the pre-grasp audit, then decide the V2-based V4
continuation and observable seven-prompt phase manager. No checkpoint training,
GUI V4 run, or task-success claim belongs to this session.
