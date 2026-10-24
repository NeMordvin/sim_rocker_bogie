import unittest

from analysis import (
    check_obstacle,
    find_limits,
    find_physical_limit,
    optimize_step_then_calculate_ditch,
)
from geometry import ChassisDesign, SolverSettings
from statics import PhysicalParameters


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.design = ChassisDesign()
        self.physical = PhysicalParameters()

    def test_obstacle_check_covers_two_scenarios(self) -> None:
        checked = check_obstacle(
            self.design,
            self.physical,
            "step",
            20.0,
            1,
            SolverSettings.coarse(),
            static_sample_stride=1,
        )
        self.assertTrue(checked.success, checked.failure_reason)
        self.assertEqual(
            {item.scenario for item in checked.statics},
            {"full_width", "one_side"},
        )

    def test_binary_search_returns_a_verified_limit(self) -> None:
        result = find_physical_limit(
            self.design,
            self.physical,
            "ditch",
            1,
            120.0,
            SolverSettings.coarse(),
            iterations=3,
            scenarios=("full_width",),
            static_sample_stride=3,
        )
        self.assertGreaterEqual(result.value_mm, 0.0)
        self.assertLessEqual(result.value_mm, 120.0)
        self.assertIsNotNone(result.checked)
        self.assertTrue(result.checked.success)

    def test_physical_limit_never_exceeds_geometric(self) -> None:
        result = find_physical_limit(
            self.design,
            self.physical,
            "step",
            1,
            400.0,
            SolverSettings.coarse(),
            iterations=4,
            scenarios=("full_width",),
            static_sample_stride=3,
        )
        self.assertIsNotNone(result.geometric)
        self.assertLessEqual(result.value_mm, result.geometric_mm)
        self.assertGreater(result.geometric_mm, result.value_mm)
        self.assertTrue(result.failure_reason)

    def test_parallel_and_serial_limits_agree(self) -> None:
        jobs = {"a": ("ditch", 1, 60.0), "b": ("ditch", -1, 60.0)}
        arguments = (self.design, self.physical, jobs, SolverSettings.coarse(), 3)
        serial = find_limits(*arguments, parallel=False)
        parallel = find_limits(*arguments, parallel=True)
        for key in jobs:
            self.assertEqual(serial[key].value_mm, parallel[key].value_mm)
            self.assertEqual(serial[key].geometric_mm, parallel[key].geometric_mm)

    def test_sequential_optimization_reports_ditch_only_for_step_winner(self) -> None:
        result = optimize_step_then_calculate_ditch(
            base=self.design,
            physical=self.physical,
            free_ranges={"middle_wheel_x_mm": (350.0, 370.0)},
            count=2,
            seed=42,
            maximum_step_mm=5.0,
            maximum_ditch_mm=10.0,
            local_search_rounds=0,
        )
        self.assertEqual(len(result.ranked_step_designs), 2)
        self.assertIsNotNone(result.best_step_design)
        self.assertEqual(
            result.best_step_design,
            max(result.ranked_step_designs, key=lambda item: item.tie_break_key),
        )
        self.assertIsNotNone(result.ditch_single_first)
        self.assertIsNotNone(result.ditch_bogie_first)
        self.assertEqual(result.bidirectional_ditch_mm, 10.0)


if __name__ == "__main__":
    unittest.main()
