import unittest

from streamlit.testing.v1 import AppTest

from model import Design, Platform
from optimize import Criterion, Goal, evaluate, jobs_for, optimize

QUICK_GOAL = Goal(criteria=(Criterion("step_up", 1.0, 80.0),), tolerance_mm=20.0)


class OptimizeTests(unittest.TestCase):
    def test_jobs_cover_both_directions_and_sides(self):
        goal = Goal(one_side=True)
        self.assertEqual(len(jobs_for(goal)), 3 * 2 * 2)
        self.assertEqual(len(jobs_for(Goal())), 3 * 2)

    def test_score_uses_worst_direction(self):
        evaluation = evaluate(Design(), Platform(), QUICK_GOAL)
        self.assertTrue(evaluation.valid)
        worst = min(limit.value_mm for limit in evaluation.limits)
        self.assertAlmostEqual(evaluation.values["step_up"], worst)
        self.assertAlmostEqual(evaluation.score, worst)

    def test_invalid_design_is_not_simulated(self):
        evaluation = evaluate(Design(front_wheel_x_mm=1500.0), Platform(), QUICK_GOAL)
        self.assertFalse(evaluation.valid)
        self.assertEqual(evaluation.limits, [])

    def test_optimization_keeps_start_and_never_gets_worse(self):
        base = Design()
        result = optimize(base, Platform(), QUICK_GOAL, {"bogie_pivot_z_mm": (220.0, 400.0)},
                          population=5, generations=1, seed=3)
        start = next(item for item in result.evaluations if item.design == base)
        self.assertIsNotNone(result.best)
        self.assertGreaterEqual(result.best.score, start.score)


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
