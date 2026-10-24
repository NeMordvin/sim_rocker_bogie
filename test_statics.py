import math
import unittest
from dataclasses import replace

from geometry import (
    ChassisDesign,
    DitchTerrain,
    SolverSettings,
    StepTerrain,
    calculate_frame,
    neutral_pose,
    simulate_traversal,
)
from statics import (
    PhysicalParameters,
    analyze_frame,
    required_friction_coefficient,
    validate_physical,
)


class StaticModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.design = ChassisDesign()
        self.physical = PhysicalParameters()

    def test_default_physical_parameters_are_valid(self) -> None:
        self.assertEqual(validate_physical(self.physical), ())
        invalid = replace(self.physical, mass_kg=0.0, friction_coefficient=0.0)
        self.assertGreaterEqual(len(validate_physical(invalid)), 2)

    def test_neutral_equilibrium_and_symmetric_reactions(self) -> None:
        result = analyze_frame(
            self.design,
            self.physical,
            neutral_pose(self.design),
            StepTerrain(0.0),
            1,
            "full_width",
        )
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(len(result.contacts), 6)
        expected_weight = (
            self.physical.mass_kg
            * self.physical.gravity_m_s2
            * self.physical.dynamic_factor
        )
        self.assertAlmostEqual(result.design_weight_n, expected_weight, places=8)
        self.assertAlmostEqual(
            sum(load.normal_force_n for load in result.contacts),
            expected_weight,
            places=7,
        )
        for wheel in ("front", "middle", "rear"):
            left = next(
                load
                for load in result.contacts
                if load.side == "левая" and load.wheel == wheel
            )
            right = next(
                load
                for load in result.contacts
                if load.side == "правая" and load.wheel == wheel
            )
            self.assertAlmostEqual(left.normal_force_n, right.normal_force_n, places=7)
            self.assertAlmostEqual(left.motor_torque_nm, right.motor_torque_nm, places=7)
        self.assertLess(result.force_residual_n, 1e-7)
        self.assertLess(result.moment_residual_nm, 1e-7)

    def _front_on_edge_pose(self, height_mm: float):
        traversal = simulate_traversal(
            self.design, StepTerrain(height_mm), 1, SolverSettings.fine()
        )
        self.assertTrue(traversal.success, traversal.failure_reason)
        # Первый кадр, где одиночное колесо оторвалось от нижней площадки.
        return next(
            pose
            for pose in traversal.poses
            if calculate_frame(self.design, pose, 1).front_wheel.y
            > self.design.wheel_radius_mm + 5.0
        )

    def test_torque_and_friction_limits_are_enforced(self) -> None:
        pose = neutral_pose(self.design)
        low_torque = analyze_frame(
            self.design,
            replace(self.physical, wheel_torque_nm=0.1),
            pose,
            StepTerrain(0.0),
            1,
            "full_width",
        )
        self.assertFalse(low_torque.feasible)
        self.assertIn("момент", low_torque.reason)

        # На ровной площадке суммарная сила от грунта равна нулю, и сцепление
        # не требуется. Подъём на кромку без сцепления невозможен.
        flat = analyze_frame(
            self.design,
            replace(self.physical, friction_coefficient=0.01),
            pose,
            StepTerrain(0.0),
            1,
            "full_width",
        )
        self.assertTrue(flat.feasible, flat.reason)

        low_friction = analyze_frame(
            self.design,
            replace(self.physical, friction_coefficient=0.05, wheel_torque_nm=1e4),
            self._front_on_edge_pose(60.0),
            StepTerrain(60.0),
            1,
            "full_width",
        )
        self.assertFalse(low_friction.feasible)
        self.assertIn("сцеплен", low_friction.reason)

    def test_required_friction_drops_when_front_wheel_is_unloaded(self) -> None:
        # Центр масс ближе к bogie разгружает колесо на кромке: задние колёса
        # прижимают его с меньшей силой, и требуемое сцепление снижается.
        pose = self._front_on_edge_pose(80.0)
        requirements = [
            required_friction_coefficient(
                replace(self.design, com_x_from_main_pivot_mm=offset),
                self.physical,
                pose,
                StepTerrain(80.0),
                1,
                "full_width",
            )
            for offset in (100.0, 0.0, -150.0)
        ]
        self.assertTrue(all(math.isfinite(value) for value in requirements))
        self.assertGreater(requirements[0], requirements[1])
        self.assertGreater(requirements[1], requirements[2])

    def test_flat_ground_needs_no_friction(self) -> None:
        mu = required_friction_coefficient(
            self.design,
            self.physical,
            neutral_pose(self.design),
            StepTerrain(0.0),
            1,
            "full_width",
        )
        self.assertLess(mu, 0.01)

    def test_lateral_stability_uses_track_and_com_offset(self) -> None:
        shifted = replace(self.physical, com_lateral_mm=295.0)
        result = analyze_frame(
            self.design,
            shifted,
            neutral_pose(self.design),
            StepTerrain(0.0),
            1,
            "full_width",
        )
        self.assertFalse(result.feasible)
        self.assertLess(result.lateral_margin_mm, shifted.minimum_support_margin_mm)

    def test_one_side_obstacle_creates_roll_and_lateral_reactions(self) -> None:
        traversal = simulate_traversal(
            self.design,
            StepTerrain(20.0),
            1,
            SolverSettings.coarse(),
        )
        frames = [
            analyze_frame(
                self.design,
                self.physical,
                pose,
                StepTerrain(20.0),
                1,
                "one_side",
                index,
            )
            for index, pose in enumerate(traversal.poses)
        ]
        feasible = [frame for frame in frames if frame.feasible]
        self.assertTrue(feasible)
        self.assertGreater(max(abs(frame.body_roll_deg) for frame in feasible), 0.1)
        self.assertTrue(
            any(
                abs(load.lateral_force_n) > 1e-6
                for frame in feasible
                for load in frame.contacts
            )
        )

    def test_ditch_wheel_loss_is_accepted_when_equilibrium_is_preserved(self) -> None:
        traversal = simulate_traversal(
            self.design,
            DitchTerrain(100.0),
            1,
            SolverSettings.coarse(),
        )
        candidate_indices = [
            index
            for index, pose in enumerate(traversal.poses)
            if len(pose.contacts) == 2
        ]
        feasible = []
        for index in candidate_indices:
            result = analyze_frame(
                self.design,
                self.physical,
                traversal.poses[index],
                DitchTerrain(100.0),
                1,
                "full_width",
                index,
            )
            if result.feasible:
                feasible.append(result)
        self.assertTrue(feasible)
        self.assertTrue(any(len(result.contacts) == 4 for result in feasible))
        self.assertTrue(
            all(math.isfinite(result.support_polygon_margin_mm) for result in feasible)
        )


if __name__ == "__main__":
    unittest.main()
