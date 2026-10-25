import unittest
from dataclasses import replace
from math import sqrt

import mujoco
import numpy as np

from model import Design, Platform, neutral_points
from simulate import Obstacle, _boundary, build_terrain, find_limit, required_torque, run

DESIGN = Design()
PLATFORM = Platform()


def _single_wheel_lifts(push_n: float, weight_n: float, torque_nm: float, step_m: float, radius_m: float) -> bool:
    """Одно колесо с мотором прижато к кромке горизонтальной силой."""
    mass = weight_n / 9.81
    xml = f"""
    <mujoco>
      <option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="10"/>
      <worldbody>
        <geom type="plane" size="0 0 1" friction="1.5 0.005 0.0001" solref="0.004 1"/>
        <geom type="box" pos="{radius_m + 0.0005 + 5} 0 {step_m / 2}" size="5 0.5 {step_m / 2}"
              friction="1.5 0.005 0.0001" solref="0.004 1"/>
        <body name="hub" pos="0 0 {radius_m}">
          <joint name="x" type="slide" axis="1 0 0"/>
          <joint name="z" type="slide" axis="0 0 1"/>
          <inertial pos="0 0 0" mass="{mass}" diaginertia="0.1 0.1 0.1"/>
          <body name="wheel">
            <joint name="spin" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="{radius_m} 0.025" zaxis="0 1 0" mass="0.5"
                  friction="1.5 0.005 0.0001" solref="0.004 1"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor joint="spin" ctrlrange="-100 100"/></actuator>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    hub = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hub")
    # Квазистатика: момент нарастает за 0,2 с, толкающая сила — за 2 с.
    highest = 0.0
    while data.time < 3.0:
        data.ctrl[0] = torque_nm * min(1.0, data.time / 0.2)
        data.xfrc_applied[hub, 0] = push_n * min(1.0, data.time / 2.0)
        mujoco.mj_step(model, data)
        highest = max(highest, data.xpos[hub][2] - radius_m)
    return highest > 0.5 * step_m


class SingleWheelCalibration(unittest.TestCase):
    """MuJoCo должен воспроизводить статический порог подъёма колеса на кромку."""

    def test_push_threshold_matches_equilibrium(self):
        # Колесо на кромке без опоры на пол: N·cosφ + f·sinφ = F, P = N·sinφ − f·cosφ,
        # трение на кромке ограничено моментом мотора f = τ/R.
        radius = 0.108
        for weight, torque, step in ((300.0, 8.0, 0.04), (300.0, 4.0, 0.06)):
            with self.subTest(weight=weight, torque=torque, step=step):
                cos_phi = (radius - step) / radius
                sin_phi = sqrt(1.0 - cos_phi**2)
                friction = torque / radius
                total = weight + 0.5 * 9.81  # плюс масса самого колеса
                threshold = (total - friction * sin_phi) * sin_phi / cos_phi - friction * cos_phi
                self.assertTrue(_single_wheel_lifts(1.08 * threshold, weight, torque, step, radius))
                self.assertFalse(_single_wheel_lifts(0.92 * threshold, weight, torque, step, radius))


class Traversal(unittest.TestCase):
    def test_flat_ground_needs_little_torque(self):
        result = run(DESIGN, PLATFORM, Obstacle("step_up", 0.0, 1), record=True)
        self.assertTrue(result.success, result.reason)
        self.assertLess(result.max_torque_ratio, 0.4)
        self.assertLess(result.max_pitch_deg, 1.0)
        self.assertGreater(len(result.frames), 20)

    def test_both_control_modes_climb_small_step(self):
        for control in ("speed", "torque"):
            with self.subTest(control=control):
                result = run(DESIGN, replace(PLATFORM, control=control), Obstacle("step_up", 20.0, 1), record=False)
                self.assertTrue(result.success, result.reason)

    def test_high_step_fails_with_reason(self):
        result = run(DESIGN, PLATFORM, Obstacle("step_up", 300.0, -1), record=False)
        self.assertFalse(result.success)
        self.assertTrue(result.reason)

    def test_more_torque_does_not_lower_limit(self):
        weak = find_limit(DESIGN, PLATFORM, "step_up", 1, upper_mm=200.0, tolerance_mm=15.0)
        strong = find_limit(DESIGN, replace(PLATFORM, wheel_torque_nm=40.0, friction=1.2), "step_up", 1,
                            upper_mm=200.0, tolerance_mm=15.0)
        self.assertGreaterEqual(strong.value_mm, weak.value_mm)
        self.assertGreater(strong.value_mm, 0.0)

    def test_step_down_moderate_height(self):
        result = run(DESIGN, PLATFORM, Obstacle("step_down", 150.0, 1), record=False)
        self.assertTrue(result.success, result.reason)

    def test_unreachable_one_side_drop_is_not_success(self):
        # Перепад больше колеи под одним бортом невозможно пройти, оставаясь на всех колёсах.
        result = run(DESIGN, PLATFORM, Obstacle("step_down", 700.0, 1, one_side=True), record=False)
        self.assertFalse(result.success)

    def test_terrain_mirrors_with_direction(self):
        forward, _, _, _ = build_terrain(DESIGN, PLATFORM, Obstacle("step_up", 100.0, 1))
        backward, _, _, _ = build_terrain(DESIGN, PLATFORM, Obstacle("step_up", 100.0, -1))
        self.assertGreater(forward[0].x0, neutral_points(DESIGN, PLATFORM).wheels["front"][0])
        self.assertLess(backward[0].x1, 0.0)

    def test_one_side_obstacle_only_under_left(self):
        boxes, _, _, _ = build_terrain(DESIGN, PLATFORM, Obstacle("step_up", 100.0, 1, one_side=True))
        self.assertEqual(boxes[0].y0, 0.0)

    def test_normal_reactions_balance_weight_on_flat(self):
        result = run(DESIGN, PLATFORM, Obstacle("step_up", 0.0, 1), record=True)
        frame = result.frames[len(result.frames) // 2]
        weight = PLATFORM.total_mass_kg * 9.81
        self.assertTrue(np.isclose(sum(frame.normal_n.values()), weight, rtol=0.05))

    def test_frames_contain_forces_balancing_weight(self):
        result = run(DESIGN, PLATFORM, Obstacle("step_up", 0.0, 1), record=True)
        frame = result.frames[len(result.frames) // 2]
        vertical = sum(force[2] for _, _, force in frame.forces)
        weight = PLATFORM.total_mass_kg * 9.81
        self.assertTrue(np.isclose(vertical, weight, rtol=0.05), (vertical, weight))


class _Fake:
    def __init__(self, success: bool):
        self.success = success
        self.reason = "" if success else "не пройдено"


class BoundarySearch(unittest.TestCase):
    def test_guess_finds_same_boundary_with_fewer_runs(self):
        calls = []

        def attempt(value):
            calls.append(value)
            return _Fake(value <= 137.0)

        plain = _boundary(attempt, 0.0, 300.0, 5.0)
        plain_runs = len(calls)
        calls.clear()
        guided = _boundary(attempt, 0.0, 300.0, 5.0, guess=135.0)
        self.assertLessEqual(abs(plain[0] - 137.0), 5.0)
        self.assertLessEqual(abs(guided[0] - 137.0), 5.0)
        self.assertLess(len(calls), plain_runs)

    def test_required_torque_on_flat_ground_is_small(self):
        result = required_torque(DESIGN, PLATFORM, Obstacle("step_up", 0.0, 1), upper_nm=20.0, tolerance_nm=2.0)
        self.assertTrue(result.passable)
        self.assertLess(result.torque_nm, 5.0)


if __name__ == "__main__":
    unittest.main()
