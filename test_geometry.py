import unittest
from dataclasses import replace

from geometry import (
    ChassisDesign,
    DitchTerrain,
    SolverSettings,
    StepTerrain,
    calculate_frame,
    neutral_dimensions,
    neutral_pose,
    simulate_traversal,
    validate_design,
)


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.design = ChassisDesign()
        self.settings = SolverSettings.coarse()

    def test_default_design_is_valid(self) -> None:
        self.assertEqual(validate_design(self.design), ())
        self.assertLessEqual(
            neutral_dimensions(self.design)["total_length_mm"],
            self.design.max_total_length_mm,
        )

    def test_neutral_wheels_have_equal_height(self) -> None:
        frame = calculate_frame(self.design, neutral_pose(self.design))
        wheel_heights = [wheel.y for wheel in frame.wheels]
        self.assertLess(max(wheel_heights) - min(wheel_heights), 1e-8)

    def test_step_higher_than_radius_is_crossable_in_both_directions(self) -> None:
        # У исходного варианта bogie проходит слишком близко к левому углу
        # корпуса. Смещение корпуса на 10 мм устраняет самопересечение и
        # позволяет отдельно проверить новую контактную кинематику.
        contact_safe_design = replace(self.design, attachment_x_mm=300.0)
        for direction in (1, -1):
            result = simulate_traversal(
                contact_safe_design,
                StepTerrain(120.0),
                direction,
                self.settings,
            )
            self.assertTrue(result.success, result.failure_reason)
            self.assertGreater(len(result.poses), 2)
            self.assertTrue(any(pose.wall_contacts for pose in result.poses))
            for pose in result.poses:
                self.assertEqual(
                    set(pose.contacts) | set(pose.wall_contacts),
                    {"front", "middle", "rear"},
                )
            self.assertEqual(
                set(result.poses[-1].contacts),
                {"front", "middle", "rear"},
            )

    def test_main_pivot_may_move_back_while_wheel_climbs_wall(self) -> None:
        design = replace(self.design, attachment_x_mm=300.0)
        result = simulate_traversal(
            design,
            StepTerrain(120.0),
            1,
            self.settings,
        )
        self.assertTrue(result.success, result.failure_reason)
        wall_poses = [pose for pose in result.poses if "front" in pose.wall_contacts]
        self.assertGreater(len(wall_poses), 1)
        self.assertLess(
            wall_poses[-1].main_pivot_x_mm,
            wall_poses[0].main_pivot_x_mm,
        )

    def test_small_ditch_is_crossable_in_both_directions(self) -> None:
        for direction in (1, -1):
            result = simulate_traversal(
                self.design,
                DitchTerrain(100.0),
                direction,
                self.settings,
            )
            self.assertTrue(result.success, result.failure_reason)

    def test_length_limit_is_enforced(self) -> None:
        invalid = replace(self.design, max_total_length_mm=900.0)
        errors = validate_design(invalid)
        self.assertTrue(any("превышает ограничение" in error for error in errors))

    def test_center_of_mass_changes_directional_result(self) -> None:
        shifted = replace(self.design, com_x_from_body_left_mm=180.0)
        forward = simulate_traversal(
            shifted,
            DitchTerrain(240.0),
            1,
            self.settings,
        )
        reverse = simulate_traversal(
            shifted,
            DitchTerrain(240.0),
            -1,
            self.settings,
        )
        self.assertNotEqual(
            (forward.success, forward.minimum_support_margin_mm),
            (reverse.success, reverse.minimum_support_margin_mm),
        )


if __name__ == "__main__":
    unittest.main()
