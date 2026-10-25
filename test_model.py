import unittest
from dataclasses import replace
from math import hypot

import mujoco

from model import (
    COORDINATE_FIELDS,
    Design,
    Platform,
    build_mjcf,
    coordinates,
    design_from_coordinates,
    design_from_dict,
    design_to_dict,
    link_dimensions,
    neutral_points,
    overall_length_mm,
    validate,
    violation_amount,
)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.design = Design()
        self.platform = Platform()

    def test_default_design_is_valid(self):
        self.assertEqual(validate(self.design, self.platform), [])

    def test_length_limit(self):
        long = replace(self.design, rocker_front_arm_mm=700.0)
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

    def test_arm_lengths_match_points(self):
        points = neutral_points(self.design, self.platform)

        def distance(a, b):
            return hypot(a[0] - b[0], a[1] - b[1])

        self.assertAlmostEqual(distance(points.bogie_pivot, points.strut_tops["rear"]), self.design.bogie_rear_arm_mm)
        self.assertAlmostEqual(distance(points.bogie_pivot, points.strut_tops["middle"]),
                               self.design.bogie_front_arm_mm)
        self.assertAlmostEqual(distance(points.main_pivot, points.bogie_pivot), self.design.rocker_rear_arm_mm)
        self.assertAlmostEqual(distance(points.main_pivot, points.strut_tops["front"]),
                               self.design.rocker_front_arm_mm)
        # Все колёса на грунте.
        self.assertTrue(all(abs(z - self.platform.wheel_radius_mm) < 1e-6 for _, z in points.wheels.values()))

    def test_angle_below_180_is_lambda_above_is_v(self):
        def above_chord(angle):
            # Высота шарнира bogie над прямой, соединяющей концы плеч.
            points = neutral_points(replace(self.design, bogie_angle_deg=angle), self.platform)
            (rx, rz), (fx, fz) = points.strut_tops["rear"], points.strut_tops["middle"]
            px, pz = points.bogie_pivot
            return pz - (rz + (fz - rz) * (px - rx) / (fx - rx))

        self.assertGreater(above_chord(150.0), 0.0)
        self.assertAlmostEqual(above_chord(180.0), 0.0, places=6)
        self.assertLess(above_chord(210.0), 0.0)

    def test_unreachable_arm_is_invalid_with_growing_penalty(self):
        # Плечи bogie короче перепада высот верхов стоек заднего и среднего колёс.
        short = replace(self.design, bogie_rear_arm_mm=40.0, bogie_front_arm_mm=40.0)
        shorter = replace(self.design, bogie_rear_arm_mm=30.0, bogie_front_arm_mm=30.0)
        self.assertTrue(validate(short, self.platform))
        self.assertGreater(violation_amount(shorter, self.platform), violation_amount(short, self.platform))
        self.assertEqual(violation_amount(self.design, self.platform), 0.0)

    def test_coordinates_roundtrip(self):
        rows = {row["Величина"]: row["мм"] for row in coordinates(self.design, self.platform)}
        old = dict(zip(COORDINATE_FIELDS, (
            rows["X среднего колеса (от заднего)"], rows["X переднего колеса (от заднего)"],
            rows["Главный шарнир: X"], rows["Главный шарнир: высота над землёй"],
            rows["Шарнир bogie: X"], rows["Шарнир bogie: высота над землёй"],
        )))
        rebuilt = design_from_coordinates(self.design, self.platform, **old)
        for name in ("bogie_rear_arm_mm", "bogie_front_arm_mm", "bogie_angle_deg",
                     "rocker_rear_arm_mm", "rocker_front_arm_mm", "rocker_angle_deg"):
            self.assertAlmostEqual(getattr(rebuilt, name), getattr(self.design, name), delta=0.6, msg=name)

    def test_old_coordinate_format_is_converted(self):
        old = {
            "middle_wheel_x_mm": 380.0, "front_wheel_x_mm": 940.0, "main_pivot_x_mm": 600.0,
            "main_pivot_z_mm": 470.0, "bogie_pivot_x_mm": 190.0, "bogie_pivot_z_mm": 390.0,
            "outer_strut_height_mm": 250.0, "outer_strut_angle_deg": 0.0,
            "middle_strut_height_mm": 150.0, "middle_strut_angle_deg": 0.0,
        }
        design = design_from_dict(old, self.platform)
        points = neutral_points(design, self.platform)
        self.assertAlmostEqual(points.wheels["front"][0], 940.0, delta=1.0)
        self.assertAlmostEqual(points.wheels["middle"][0], 380.0, delta=1.0)
        self.assertAlmostEqual(points.main_pivot[1], 470.0, delta=1.0)
        self.assertLess(design.bogie_angle_deg, 180.0)


if __name__ == "__main__":
    unittest.main()
