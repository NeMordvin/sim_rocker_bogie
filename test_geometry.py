import unittest
from dataclasses import fields, replace

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

    def test_default_design_is_valid_and_within_length_limit(self) -> None:
        self.assertEqual(validate_design(self.design), ())
        self.assertLessEqual(
            neutral_dimensions(self.design)["total_length_mm"],
            self.design.max_total_length_mm,
        )

    def test_body_dimensions_are_absent(self) -> None:
        field_names = {field.name for field in fields(ChassisDesign)}
        forbidden = {
            "body_length_mm",
            "body_height_mm",
            "attachment_x_mm",
            "rocker_thickness_mm",
            "bogie_thickness_mm",
        }
        self.assertTrue(forbidden.isdisjoint(field_names))
        frame = calculate_frame(self.design, neutral_pose(self.design))
        self.assertEqual(frame.body_corners, ())

    def test_neutral_wheels_have_equal_height(self) -> None:
        frame = calculate_frame(self.design, neutral_pose(self.design))
        wheel_heights = [wheel.y for wheel in frame.wheels]
        self.assertLess(max(wheel_heights) - min(wheel_heights), 1e-8)

    def test_link_lengths_and_angles_are_derived(self) -> None:
        dimensions = neutral_dimensions(self.design)
        for prefix in (
            "rocker_front",
            "rocker_bogie",
            "bogie_middle",
            "bogie_rear",
        ):
            self.assertGreater(dimensions[f"{prefix}_length_mm"], 0.0)
            self.assertIn(f"{prefix}_angle_deg", dimensions)

    def test_geometric_step_path_exists_in_both_directions(self) -> None:
        for direction in (1, -1):
            result = simulate_traversal(
                self.design,
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

    def test_ditch_path_allows_temporary_wheel_contact_loss(self) -> None:
        result = simulate_traversal(
            self.design,
            DitchTerrain(100.0),
            1,
            self.settings,
        )
        self.assertTrue(result.success, result.failure_reason)
        self.assertTrue(any(len(pose.contacts) == 2 for pose in result.poses))

    def test_length_and_angle_limits_are_enforced(self) -> None:
        too_long = replace(self.design, max_total_length_mm=900.0)
        self.assertTrue(
            any("превышает ограничение" in error for error in validate_design(too_long))
        )
        invalid_angles = replace(
            self.design,
            rocker_min_angle_deg=20.0,
            rocker_max_angle_deg=-20.0,
        )
        self.assertTrue(
            any("rocker" in error.lower() for error in validate_design(invalid_angles))
        )


if __name__ == "__main__":
    unittest.main()
