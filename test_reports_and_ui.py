import unittest

from analysis import check_obstacle
from drawing import create_design_figure, create_key_frame_figure
from geometry import ChassisDesign, SolverSettings
from reports import select_key_frames
from sources import SOURCES
from statics import PhysicalParameters


class ReportsAndDrawingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = ChassisDesign()
        cls.checked = check_obstacle(
            cls.design,
            PhysicalParameters(),
            "step",
            20.0,
            1,
            SolverSettings.coarse(),
            static_sample_stride=2,
        )

    def test_key_frames_have_full_calculation_and_sources(self) -> None:
        self.assertTrue(self.checked.success, self.checked.failure_reason)
        frames = select_key_frames(self.checked, "one_side", maximum_frames=12)
        self.assertGreaterEqual(len(frames), 2)
        self.assertLessEqual(len(frames), 12)
        for frame in frames:
            self.assertGreaterEqual(len(frame.static.equations), 10)
            self.assertTrue(frame.static.source_ids)
            self.assertTrue(
                all(source_id in SOURCES for source_id in frame.static.source_ids)
            )

    def test_figures_are_static(self) -> None:
        design_figure = create_design_figure(self.design)
        self.assertEqual(len(design_figure.frames), 0)
        key_frame = select_key_frames(self.checked, "one_side")[0]
        figure = create_key_frame_figure(
            self.design,
            self.checked.traversal,
            key_frame.pose,
            key_frame.static,
            key_frame.title,
        )
        self.assertEqual(len(figure.frames), 0)
        self.assertGreater(len(figure.data), 0)


class StreamlitSmokeTest(unittest.TestCase):
    def test_application_opens_without_exceptions(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("app.py").run(timeout=30)
        self.assertEqual([item.value for item in app.exception], [])
        self.assertEqual(
            [tab.label for tab in app.tabs],
            [
                "1. Геометрия",
                "2. Расчёт варианта",
                "3. Оптимизация",
                "4. Ключевые кадры",
                "5. Методика и статьи",
            ],
        )


if __name__ == "__main__":
    unittest.main()
