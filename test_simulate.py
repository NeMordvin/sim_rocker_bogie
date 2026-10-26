import unittest
from dataclasses import replace
from math import sqrt

import mujoco
import numpy as np

from model import Design, Platform, neutral_points
from simulate import (SETTLE_S, STUCK_WINDOW_S, Obstacle, _boundary, _Ids, _Motor, _verify, build_terrain,
                      find_limit, make_model, required_torque, run)

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

    def test_wheel_climbing_wall_is_not_stuck(self):
        # Колесо долго ползёт вверх по стенке, а корпус в это время стоит или откатывается —
        # это продвижение, а не застревание.
        platform = replace(PLATFORM, body_mass_kg=36.0, friction=1.8)
        result = run(DESIGN, platform, Obstacle("step_up", 365.0, -1), record=False)
        self.assertTrue(result.success, result.reason)

    def test_tire_deflection_matches_stiffness(self):
        model, *_ = make_model(DESIGN, PLATFORM, Obstacle("step_up", 0.0, 1))
        data = mujoco.MjData(model)
        while data.time < 1.0:
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        stiffness = PLATFORM.tire_stiffness_n_mm
        for wheel in ("front", "middle", "rear"):
            with self.subTest(wheel=wheel):
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"tire_z_{wheel}_L")
                deflection_mm = data.qpos[model.jnt_qposadr[joint]] * 1000.0
                load = 0.0
                wheel_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{wheel}_L")
                force = np.zeros(6)
                for index in range(data.ncon):
                    contact = data.contact[index]
                    if wheel_body in (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]):
                        mujoco.mj_contactForce(model, data, index, force)
                        load += force[0]
                # Обод поднимается относительно оси на прогиб = нагрузка / жёсткость
                # (без веса самого колеса, который пружина не несёт).
                own = PLATFORM.wheel_module_mass_kg * 9.81
                self.assertAlmostEqual(deflection_mm, (load - own) / stiffness, delta=0.15)

    def test_zero_stiffness_is_rigid_wheel(self):
        model, *_ = make_model(DESIGN, replace(PLATFORM, tire_stiffness_n_mm=0.0), Obstacle("step_up", 0.0, 1))
        self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tire_z_front_L"), -1)

    def test_obvious_stall_ends_early(self):
        result = run(DESIGN, PLATFORM, Obstacle("step_up", 300.0, 1), record=False)
        self.assertFalse(result.success)
        self.assertIn("не хватает момента", result.reason)
        self.assertLess(result.duration_s, SETTLE_S + STUCK_WINDOW_S)

    def test_climbing_edges_are_rounded(self):
        for obstacle, rounded in ((Obstacle("step_up", 100.0, 1), ("x0",)), (Obstacle("step_up", 100.0, -1), ("x1",)),
                                  (Obstacle("ditch", 300.0, 1), ("x1", "x0"))):
            with self.subTest(obstacle=obstacle.label):
                boxes, *_ = build_terrain(DESIGN, PLATFORM, obstacle)
                for box, edge in zip(boxes, rounded):
                    self.assertEqual(getattr(box, f"round_{edge}"), PLATFORM.tire_edge_wrap_mm)
        low, *_ = build_terrain(DESIGN, PLATFORM, Obstacle("step_up", 4.0, 1))
        self.assertEqual(low[0].round_x0, 4.0)  # не больше высоты стенки
        model, *_ = make_model(DESIGN, PLATFORM, Obstacle("step_up", 100.0, 1))
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground0_edge_x0"), 0)

    def test_rounded_outline_follows_arc(self):
        boxes, *_ = build_terrain(DESIGN, PLATFORM, Obstacle("step_up", 100.0, 1))
        box = boxes[0]
        r = box.round_x0
        for x, z in box.outline(-500.0, box.x0 - 100.0, box.x0 + 300.0)[1:-3]:
            if x < box.x0 + r - 1e-6:
                self.assertAlmostEqual(np.hypot(x - box.x0 - r, z - box.z_top + r), r, places=6)

    def test_motor_drive_torque_falls_with_speed(self):
        model, *_ = make_model(DESIGN, PLATFORM, Obstacle("step_up", 0.0, 1))
        data = mujoco.MjData(model)
        ids = _Ids(model)
        motor = _Motor(ids, PLATFORM)
        half = (PLATFORM.motor_knee_speed_m_s + PLATFORM.motor_no_load_speed_m_s) / 2.0
        data.qvel[motor.dofs[0]] = half / (PLATFORM.wheel_radius_mm / 1000.0)
        lower, upper = motor.limits(data)
        peak = PLATFORM.wheel_torque_nm
        self.assertAlmostEqual(upper[0], 0.5 * peak, places=6)  # разгон — по характеристике
        self.assertAlmostEqual(lower[0], -peak, places=6)       # торможение — полный момент
        self.assertAlmostEqual(upper[1], peak, places=6)        # колесо стоит

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

    def test_verify_rejects_lucky_boundary(self):
        # Проходит до 50 и случайно в 98–104: деление пополам могло найти 100.
        def attempt(value):
            return _Fake(value <= 50.0 or 98.0 <= value <= 104.0)

        good, _, _, _, unstable = _verify(attempt, 100.0, 105.0, 0.0, 5.0)
        self.assertTrue(unstable)
        self.assertLessEqual(abs(good - 50.0), 5.0)

        def torque(value):
            return _Fake(10.0 <= value <= 10.5 or value >= 30.0)

        good, _, _, _, unstable = _verify(torque, 10.0, 9.5, 40.0, 0.5)
        self.assertTrue(unstable)
        self.assertLessEqual(abs(good - 30.0), 0.5)

    def test_verify_keeps_clean_boundary(self):
        good, _, _, runs, unstable = _verify(lambda value: _Fake(value <= 137.0), 135.0, 140.0, 0.0, 5.0)
        self.assertFalse(unstable)
        self.assertEqual(good, 135.0)
        self.assertEqual(runs, 2)

    def test_required_torque_on_flat_ground_is_small(self):
        result = required_torque(DESIGN, PLATFORM, Obstacle("step_up", 0.0, 1), upper_nm=20.0, tolerance_nm=2.0)
        self.assertTrue(result.passable)
        self.assertLess(result.torque_nm, 5.0)


if __name__ == "__main__":
    unittest.main()
