# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from task2_isaacsim.baselines.pi05.phase_manager import (
    CameraEvidence,
    ObservablePhaseManager,
    PhaseManagerConfig,
    PhaseObservation,
    V4Phase,
)


PREGRASP = (
    1.7504082918167114,
    2.140868902206421,
    0.8722001910209656,
    -0.02928313829834448,
    0.7328308972762488,
    -0.6796555833030944,
    -0.013025432569340717,
)


def observation(
    sim_time_s: float,
    *,
    spine_m: float = 0.0,
    pose: tuple[float, ...] = PREGRASP,
    gripper: float = 1.0,
    command: float | None = None,
    camera: CameraEvidence | None = None,
) -> PhaseObservation:
    return PhaseObservation(
        sim_time_s=sim_time_s,
        spine_m=spine_m,
        right_ee_xyzw=pose,  # type: ignore[arg-type]
        right_gripper_open_fraction=gripper,
        right_gripper_command=command,
        camera=camera or CameraEvidence(),
    )


class ObservablePhaseManagerTest(unittest.TestCase):
    def test_manager_advances_only_after_stable_observable_evidence(self) -> None:
        manager = ObservablePhaseManager(
            PhaseManagerConfig(stable_observations=2)
        )
        manager.update(observation(0.0, spine_m=0.45))
        self.assertEqual(manager.phase, V4Phase.STARTUP_RISE)
        manager.update(observation(0.1, spine_m=0.45))
        self.assertEqual(manager.phase, V4Phase.APPROACH)

        approach_pose = (1.801, 2.287, 0.880, *PREGRASP[3:])
        centered = CameraEvidence(
            pad_visible=True, pad_centered_right_wrist=True
        )
        manager.update(observation(0.2, spine_m=0.45, pose=approach_pose))
        manager.update(
            observation(
                0.3, spine_m=0.45, pose=approach_pose, camera=centered
            )
        )
        self.assertEqual(manager.phase, V4Phase.APPROACH)
        manager.update(
            observation(
                0.4, spine_m=0.45, pose=approach_pose, camera=centered
            )
        )
        self.assertEqual(manager.phase, V4Phase.ORIENT_PREGRASP)

        manager.update(
            observation(0.5, spine_m=0.45, camera=centered)
        )
        manager.update(
            observation(0.6, spine_m=0.45, camera=centered)
        )
        self.assertEqual(manager.phase, V4Phase.GRASP_ACQUISITION)

    def test_manager_never_converts_evidence_to_actuator_targets(self) -> None:
        manager = ObservablePhaseManager()
        self.assertFalse(hasattr(manager, "arm_target"))
        self.assertFalse(hasattr(manager, "spine_target"))
        self.assertIn("spine", manager.prompt.lower())

    def test_full_progression_requires_gripper_and_camera_evidence(self) -> None:
        manager = ObservablePhaseManager(
            PhaseManagerConfig(stable_observations=1)
        )
        centered = CameraEvidence(
            pad_visible=True, pad_centered_right_wrist=True
        )
        manager.update(observation(0, spine_m=0.45))
        approach_pose = (1.801, 2.287, 0.880, *PREGRASP[3:])
        manager.update(
            observation(1, spine_m=0.45, pose=approach_pose, camera=centered)
        )
        manager.update(observation(2, spine_m=0.45, camera=centered))
        self.assertEqual(manager.phase, V4Phase.GRASP_ACQUISITION)
        manager.update(
            observation(
                3,
                spine_m=0.45,
                gripper=0.1,
                command=0.0,
                camera=CameraEvidence(pad_lifted=True),
            )
        )
        self.assertEqual(manager.phase, V4Phase.LIFT_TRANSFER)
        manager.update(
            observation(
                4,
                spine_m=0.45,
                gripper=0.1,
                command=0.0,
                camera=CameraEvidence(
                    pad_lifted=True,
                    target_visible=True,
                    target_centered_right_wrist=True,
                ),
            )
        )
        manager.update(
            observation(
                5,
                spine_m=0.45,
                gripper=0.1,
                command=0.0,
                camera=CameraEvidence(pad_supported_on_target=True),
            )
        )
        manager.update(
            observation(
                6,
                spine_m=0.45,
                gripper=0.9,
                command=1.0,
                camera=CameraEvidence(pad_supported_on_target=True),
            )
        )
        self.assertTrue(manager.complete)
        self.assertEqual(manager.phase, V4Phase.RELEASE_RETREAT)

    def test_timeout_uses_simulator_time_and_stops_progression(self) -> None:
        manager = ObservablePhaseManager(
            PhaseManagerConfig(maximum_phase_sim_s=1.0)
        )
        manager.update(observation(10.0))
        manager.update(observation(11.1))
        self.assertEqual(manager.stop_reason, "phase_timeout:startup_rise")
        self.assertEqual(manager.phase, V4Phase.STARTUP_RISE)

    def test_simulator_time_regression_is_rejected(self) -> None:
        manager = ObservablePhaseManager()
        manager.update(observation(2.0))
        with self.assertRaisesRegex(ValueError, "regressed"):
            manager.update(observation(1.0))


if __name__ == "__main__":
    unittest.main()
