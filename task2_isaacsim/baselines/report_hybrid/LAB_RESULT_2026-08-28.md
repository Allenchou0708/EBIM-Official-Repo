# report_hybrid simulator result — 2026-08-28

## Outcome

**NO-GO.** The new source/report pipeline and focused controller contract pass,
and live Isaac Sim proved that the robot RGB-D plus optical camera-FK interface
works with zero policy publications. The required policy-legal S0 did not
finish: the initial-relative base controller stalled before reaching the
observation pose. Therefore post-S0 Gate C, the external 15 mm / 8° audit, and
the retained-grasp/placement attempt were not run.

This is an honest pipeline/blocker result. It is not simulator task success,
position generalization, or real-robot evidence.

## Reproducible setup

- repository base: `24c16076027165ec94e1cc16f511ccb211d7baa8`
- branch: `task2_phase2_report_hybrid_sim_20260828`
- simulator: official room scene, Isaac Sim 5.1.0, RTX 5090
- scene flags: `--record --robot-camera-depth --arm-pose-command-control
  --disable-browser-command-topics --headless`
- isolated ROS domain: `28`; Fast DDS UDPv4
- training reference: 180 development scenarios from
  `task2_fixpos_200_46ab41f`, with support 20/25/90/45 across v1/v2/v3/v4
- evidence root:
  `/scratch1/2026_ebim/allen_task2_pi05/evidence/phase2_report_hybrid_sim_20260828/`

No retired Hybrid V1 source was revived. Public submission source was reviewed
but not copied.

## Gate results

### Gate A — technical design and source matrix: PASS

[`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) records the official task and
submission contract, primary public-source matrix, method, legal information
boundary, claims, and staged gates.

### Gate B focused tests: PASS

The existing `ebim-task2-pi05-submit:local` image ran:

```text
python3 -m unittest task2_isaacsim.tests.test_report_hybrid
Ran 8 tests in 0.083s — OK (final Docker rerun)
```

Coverage includes:

- 180-development / four-collection support with no episode-19 ownership;
- initial-relative odometry composition;
- fail-closed forbidden-topic contract;
- RGB mask and RGB-D projection geometry;
- translation/yaw/workspace-bounded landmark retargeting;
- forward-only FSM with exactly one close/open latch;
- constant left safe hold/open gripper;
- policy/external-audit process separation.

This is code-pipeline evidence, not live perception accuracy.

### Robot camera-FK interface and pre-S0 shadow: PASS/PARTIAL

The bridge published these live robot-state topics, one publisher each:

```text
/isaac/head_camera/pose
/isaac/left_wrist_camera/pose
/isaac/right_wrist_camera/pose
```

The optical poses are derived only from robot camera prim FK; their hardware
equivalent is calibrated TF. The zero-publication policy node received fresh
head/right-wrist RGB-D, intrinsics, camera poses, odometry, joints, and EE
poses. It created no publishers, found zero command publishers, and made zero
publications.

At the reset pose the expected fail-closed result was:

```text
head_target_not_visible: pixels=0
```

Evidence: `pre_s0_camera_callback_shadow.json`. This confirms the callbacks
and legal boundary while repeating the known fact that reset is not a
camera-ready observation pose.

### Policy-legal S0: FAIL at base

S0 is ordered `base -> spine -> left safe hold -> right observation pose`.
Only the base stage actuated. The spine and arms never received commands.

The first run used a wall-clock timeout. It reached approximately
`(3.1217, 2.6268, -1.4924)` and timed out. One permitted evidence-driven
correction was made: wait for the sole base subscriber before timing, then use
45 seconds of simulator time with a 300-second wall safeguard.

The fresh retry reproduced the same physical stall:

| measure | value |
| --- | --- |
| initial odometry | `(4.40008, 2.62044, -1.57051)` |
| initial-relative target | `(2.10000, 3.08299, -1.57019)` |
| final odometry | `(3.12137, 2.62602, -1.49175)` |
| elapsed | `45.0000 sim s / 243.42 wall s` |
| result | `base_stage_timeout` |

During the stall the policy was publishing the right-strafe token `B`, and
the simulator reported applied body twist `(0.0, -0.5, 0.0)`, but odometry no
longer progressed. This establishes the exact blocker: the greedy direct
odometry route cannot traverse the room from reset to the demonstrated base
pose. The stationary odometry despite a nonzero applied twist is consistent
with the base being physically obstructed; no object/evaluator GT was needed
for that diagnosis.

Evidence: `gate_c_stage.json` and `gate_c_stage_retry.json`.

### Gate C / external audit / Gate D: NOT REACHED

Because the same base timeout repeated after the one minimal correction:

- no post-S0 zero-publication shadow was labeled PASS;
- no camera anchor was compared with external object GT at the 15 mm / 8°
  thresholds;
- no right-arm approach, gripper close, retained lift, release, or placement
  command was published;
- no official orientation or IoU score was requested;
- `task_success=false`.

No threshold was weakened and no route/parameter sweep followed the repeated
failure.

## Safety and teardown

- Domain 0 exposed a stale `/keyboard_to_base` publisher; the stager rejected
  it before actuation. The experiment was restarted in isolated domain 28,
  where the base topic had zero publishers and one simulator subscriber.
- The policy stage process exited after the bounded failure.
- Isaac Sim was stopped with SIGINT and the compose container was stopped.
- No policy, simulator, helper, or actuator publisher was left running.
- Bulk evidence remained under scratch and was not added to Git.

## Decision

The hybrid architecture remains plausible, but this run has not validated its
manipulation feasibility. The next experiment must replace the direct greedy
base correction with the Phase-I-proven, obstacle-safe ordered route
(`BACK -> stop -> right strafe -> brake -> odometry correction`) expressed
only in the latched initial odometry frame. It must start from a fresh reset and
repeat this branch's S0/shadow/audit gates before any grasp actuation.
