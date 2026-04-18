"""SimulationRenderer 单元测试"""
import pytest
import numpy as np
from unittest.mock import patch


@pytest.fixture
def renderer():
    from plugins.indoor_fence.standalone.simulation_renderer import SimulationRenderer
    return SimulationRenderer(
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        resolution=(640, 480),
        yellow_line_y=3.0,
        cabinets=[
            (2.0, 4.75, 2.0, 1.5, "Cabinet A"),
        ],
        scenario_name="test_scenario",
    )


class TestWorldToPixel:
    def test_origin_maps_to_margin(self, renderer):
        px, py = renderer.world_to_pixel(0.0, 0.0)
        assert px >= 0
        assert py >= 0

    def test_max_bounds_within_resolution(self, renderer):
        px, py = renderer.world_to_pixel(10.0, 6.0)
        assert px <= 640
        assert py <= 480

    def test_monotonic_x(self, renderer):
        px1, _ = renderer.world_to_pixel(2.0, 3.0)
        px2, _ = renderer.world_to_pixel(5.0, 3.0)
        assert px2 > px1

    def test_monotonic_y(self, renderer):
        _, py1 = renderer.world_to_pixel(5.0, 1.0)
        _, py2 = renderer.world_to_pixel(5.0, 4.0)
        assert py2 > py1


class TestRender:
    def test_output_shape(self, renderer):
        frame = renderer.render([], 0)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_empty_persons_no_crash(self, renderer):
        frame = renderer.render([], 100)
        assert frame is not None

    def test_single_person(self, renderer):
        persons = [{"id": 0, "x": 5.0, "y": 3.0, "z": 0.0}]
        frame = renderer.render(persons, 1)
        assert frame.shape == (480, 640, 3)
        assert frame.max() > 30  # not all dark gray

    def test_multiple_persons(self, renderer):
        persons = [
            {"id": 0, "x": 2.0, "y": 1.0},
            {"id": 1, "x": 5.0, "y": 4.0},
            {"id": 2, "x": 8.0, "y": 2.0},
        ]
        frame = renderer.render(persons, 10)
        assert frame.shape == (480, 640, 3)

    def test_trail_accumulates(self, renderer):
        persons = [{"id": 0, "x": 5.0, "y": 3.0}]
        renderer.render(persons, 1)
        renderer.render(persons, 2)
        renderer.render(persons, 3)
        assert len(renderer._trails[0]) == 3

    def test_reset_trails(self, renderer):
        persons = [{"id": 0, "x": 5.0, "y": 3.0}]
        renderer.render(persons, 1)
        renderer.render(persons, 2)
        renderer.reset_trails()
        assert len(renderer._trails) == 0

    def test_person_at_boundary(self, renderer):
        persons = [{"id": 0, "x": 0.0, "y": 0.0}]
        frame = renderer.render(persons, 1)
        assert frame.shape == (480, 640, 3)

    def test_different_resolutions(self):
        from plugins.indoor_fence.standalone.simulation_renderer import SimulationRenderer
        for res in [(320, 240), (800, 600), (1280, 720)]:
            r = SimulationRenderer(
                scene_bounds=(0.0, 0.0, 10.0, 6.0),
                resolution=res,
            )
            frame = r.render([{"id": 0, "x": 5.0, "y": 3.0}], 1)
            assert frame.shape == (res[1], res[0], 3)


class TestNoCV2Fallback:
    def test_render_without_cv2(self):
        from plugins.indoor_fence.standalone import simulation_renderer as mod
        original = mod.CV2_AVAILABLE
        mod.CV2_AVAILABLE = False
        try:
            from plugins.indoor_fence.standalone.simulation_renderer import SimulationRenderer
            r = SimulationRenderer(
                scene_bounds=(0.0, 0.0, 10.0, 6.0),
                resolution=(640, 480),
            )
            frame = r.render([{"id": 0, "x": 5.0, "y": 3.0}], 1)
            assert frame.shape == (480, 640, 3)
        finally:
            mod.CV2_AVAILABLE = original
