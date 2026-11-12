"""표시하는 로프 길이와 양 끝점의 일치를 검증한다."""

import numpy as np
import pytest

from sim.rope_visual import rope_curve


@pytest.mark.parametrize("gap", [0., 0.01, 0.5, 1.5, 1.999, 2., 3.])
@pytest.mark.parametrize("direction", [[1, 0, 0], [0, 0, 1], [1, 2, 3]])
def test_curve_preserves_endpoints_and_does_not_shorten_rope(gap, direction):
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    start = np.array([3., -2., 1.])
    end = start + gap * direction
    points = rope_curve(start, end, 2.)
    np.testing.assert_array_equal(points[0], start)
    np.testing.assert_array_equal(points[-1], end)
    length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    assert length == pytest.approx(max(2., gap), abs=1e-9)


def test_slack_line_bends_to_both_sides():
    points = rope_curve([0, 0, 1], [1.5, 0, 1], 2.)
    assert points[:, 1].max() > 0.1
    assert points[:, 1].min() < -0.1


def test_taut_line_is_straight():
    np.testing.assert_array_equal(rope_curve([0, 0, 0], [3, 0, 0], 2.),
                                  [[0, 0, 0], [3, 0, 0]])


@pytest.mark.parametrize("length", [-1, 0, float("nan")])
def test_invalid_rest_lengths_are_rejected(length):
    with pytest.raises(ValueError):
        rope_curve([0, 0, 0], [1, 0, 0], length)


def test_rendered_slack_segments_keep_two_meters_and_leave_physics_unchanged():
    mujoco = pytest.importorskip("mujoco")
    from sim.view_tether import PullComparison

    demo = PullComparison()
    scene = mujoco.MjvScene(demo.world.model, maxgeom=128)
    original_spring = demo.world.model.tendon_lengthspring.copy()
    original_gap = demo.world.data.ten_length.copy()
    demo.decorate(scene)
    assert scene.ngeom == 65  # 로프 64조각과 바닥의 시작 위치 표시
    # 표시용 MjvGeom 캡슐의 반길이는 size[2] (물리 geom의 size와 배열 규약이 다름).
    displayed_length = sum(2 * scene.geoms[i].size[2] for i in range(64))
    assert displayed_length == pytest.approx(2.0)
    assert demo.readings()["length"] == pytest.approx(1.5)
    np.testing.assert_array_equal(demo.world.model.tendon_lengthspring, original_spring)
    np.testing.assert_array_equal(demo.world.data.ten_length, original_gap)
    assert demo.rope.tension(demo.world.model, demo.world.data) == 0
