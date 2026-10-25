import unittest
from dataclasses import replace

import mujoco

from model import Design, Platform, build_mjcf, design_from_dict, design_to_dict, link_dimensions, neutral_points, overall_length_mm, validate


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.design = Design()
        self.platform = Platform()

    def test_default_design_is_valid(self):
        self.assertEqual(validate(self.design, self.platform), [])

    def test_length_limit(self):
        long = replace(self.design, front_wheel_x_mm=1100.0, main_pivot_x_mm=700.0)
        self.assertTrue(any("габарит" in error for error in validate(long, self.platform)))
        self.assertLessEqual(overall_length_mm(self.design, self.platform), self.platform.max_length_mm)

    def test_track_from_overall_width(self):
        self.assertAlmostEqual(self.platform.track_mm, 600.0 - self.platform.wheel_width_mm)

    def test_outer_struts_tilt_mirrored_toward_center(self):
        tilted = replace(self.design, outer_strut_angle_deg=20.0)
        points = neutral_points(tilted, self.platform)
        self.assertLess(points.strut_tops["front"][0], points.wheels["front"][0])
        self.assertGreater(points.strut_tops["rear"][0], points.wheels["rear"][0])
        self.assertAlmostEqual(points.strut_tops["front"][1], points.strut_tops["rear"][1])

    def test_strut_height_and_angle(self):
        points = neutral_points(replace(self.design, middle_strut_angle_deg=0.0, middle_strut_height_mm=200.0),
                                self.platform)
        self.assertAlmostEqual(points.strut_tops["middle"][1] - points.wheels["middle"][1], 200.0)

    def test_link_dimensions_rows(self):
        rows = link_dimensions(self.design, self.platform)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["Длина, мм"] > 0 for row in rows))

    def test_mjcf_compiles_with_differential(self):
        model = mujoco.MjModel.from_xml_string(build_mjcf(self.design, self.platform, []))
        self.assertEqual(model.nu, 6)
        self.assertEqual(model.neq, 1)
        torque_model = mujoco.MjModel.from_xml_string(
            build_mjcf(self.design, replace(self.platform, control="torque"), []))
        self.assertAlmostEqual(torque_model.actuator_ctrlrange[0][1], self.platform.wheel_torque_nm)

    def test_dict_roundtrip(self):
        self.assertEqual(design_from_dict(design_to_dict(self.design)), self.design)


if __name__ == "__main__":
    unittest.main()
