import unittest

from streamlit.testing.v1 import AppTest

from model import Design, Platform
from optimize import (FAILED_TORQUE_FACTOR, INVALID_COST, Criterion, Goal, criterion_label, evaluate, jobs_for,
                      optimize, summarize)
from simulate import LimitResult, Obstacle, TorqueResult

QUICK_GOAL = Goal(criteria=(Criterion("step_up", 1.0, 80.0),), tolerance_mm=20.0)


class OptimizeTests(unittest.TestCase):
    def test_jobs_cover_both_directions_and_sides(self):
        goal = Goal(one_side=True)
        self.assertEqual(len(jobs_for(goal)), 5 * 2 * 2)
        self.assertEqual(len(jobs_for(Goal())), 5 * 2)

    def test_bumps_of_different_width_are_scored_separately(self):
        goal = Goal(criteria=(Criterion("bump", 1.0, 300.0, 0.0, 120.0), Criterion("bump", 0.5, 300.0, 0.0, 250.0)))
        self.assertEqual({job.width_mm for job in jobs_for(goal)}, {120.0, 250.0})
        results = [LimitResult(Obstacle("bump", value, direction, bump_width_mm=width), value, "", 300.0, 1)
                   for width, direction, value in ((120.0, 1, 90.0), (120.0, -1, 70.0),
                                                   (250.0, 1, 60.0), (250.0, -1, 80.0))]
        evaluation = summarize(Design(), goal, results)
        self.assertEqual(evaluation.values, {"bump_120": 70.0, "bump_250": 60.0})
        self.assertAlmostEqual(evaluation.score, 70.0 + 0.5 * 60.0)
        self.assertEqual(criterion_label("bump_250"), "кочка 250 мм")

    def test_score_uses_worst_direction(self):
        evaluation = evaluate(Design(), Platform(), QUICK_GOAL)
        self.assertTrue(evaluation.valid)
        worst = min(limit.value_mm for limit in evaluation.results)
        self.assertAlmostEqual(evaluation.values["step_up"], worst)
        self.assertAlmostEqual(evaluation.score, worst)
        self.assertAlmostEqual(evaluation.cost, -worst)

    def test_invalid_design_is_not_simulated(self):
        evaluation = evaluate(Design(rocker_front_arm_mm=1500.0), Platform(), QUICK_GOAL)
        self.assertFalse(evaluation.valid)
        self.assertEqual(evaluation.results, [])
        self.assertGreater(evaluation.cost, INVALID_COST)

    def test_torque_cost_penalizes_impassable(self):
        goal = Goal(criteria=(Criterion("step_up", 1.0, 300.0, 80.0),), mode="torque", max_torque_nm=40.0)
        obstacle = Obstacle("step_up", 80.0, 1)
        passable = summarize(Design(), goal, [TorqueResult(obstacle, 20.0, "", 40.0, 1),
                                              TorqueResult(obstacle, 10.0, "", 40.0, 1)])
        self.assertAlmostEqual(passable.score, 20.0)
        self.assertAlmostEqual(passable.cost, 20.0 + 0.1 * 15.0)
        failed = summarize(Design(), goal, [TorqueResult(obstacle, float("inf"), "задевает", 40.0, 1),
                                            TorqueResult(obstacle, 10.0, "", 40.0, 1)])
        self.assertEqual(failed.score, float("inf"))
        self.assertGreater(failed.cost, FAILED_TORQUE_FACTOR * 40.0)

    def test_optimization_keeps_start_and_never_gets_worse(self):
        base = Design()
        result = optimize(base, Platform(), QUICK_GOAL, {"bogie_angle_deg": (180.0, 240.0)},
                          population=5, generations=1, seed=3, refine=0)
        start = next(item for item in result.evaluations if item.design == base)
        self.assertIsNotNone(result.best)
        self.assertGreaterEqual(result.best.score, start.score)
        self.assertEqual(len(result.history), 1)


class AppTests(unittest.TestCase):
    def test_app_renders_and_simulates(self):
        app = AppTest.from_file("app.py", default_timeout=120)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 5)
        simulate = next(button for button in app.button if button.label == "Смоделировать")
        simulate.click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success or app.error)


if __name__ == "__main__":
    unittest.main()
