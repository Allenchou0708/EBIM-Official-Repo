# Task 2 PI0.5 shadow staging startup fix (2026-08-15)

## Outcome

The V1 030000 shadow reached successful fixed-base staging but stopped before
manipulation staging. No manipulation or policy command was published.

Failed evidence:

`/scratch1/2026_ebim/allen_task2_pi05/outputs/live_submit_v1-030000-shadow_20260814_235643`

The manifest reports `dataset_pregrasp_staging_failed`, zero decisions, and
zero command publications. Fixed-base staging completed within tolerance.

## Root cause and fix

`ManipulationStager`, a subclass of `rclpy.node.Node`, assigned a dictionary to
`self.publishers`. In the installed ROS 2 Jazzy `rclpy`, `publishers` is a
read-only Node property, so construction failed immediately with:

`AttributeError: property 'publishers' of 'ManipulationStager' object has no setter`

The internal dictionary is now named `self.command_publishers`. No topic,
target, tolerance, staging route, clock, or publication behavior changed.

## Verification

- The same live image/module initialization probe now constructs the ROS node
  and exits through the expected zero-duration watchdog path.
- Probe command publications: 0.
- Dataset staging audit validation: PASS, episode 176/frame 408, 409 frames.
- Focused PI0.5 control/contract tests: 74/74 PASS.

The operator must rerun the V1 shadow. GUI publication remains NO-GO until the
new shadow completes staging, forms one valid decision with zero policy command
publications, passes `verify-shadow --contract v1`, and the right-wrist image
shows the pad front.
