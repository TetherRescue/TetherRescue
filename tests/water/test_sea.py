"""반사 바다(Sea)가 기존 사인파와 맞고, 벽·그늘·배율이 따로 동작하는지."""
import numpy as np
import pytest

from sim import config
from sim.water.sea import Sea
from sim.water.water import wave_height


def test_still_sea_is_flat_and_team_matches_wave_height():
    still = Sea('team').scaled(0)
    assert float(still.eta(3.0, -1.0, 1.5)) == 0.0
    sea = Sea('team')
    x = np.linspace(-4.0, 4.0, 5)
    y = np.linspace(-2.0, 2.0, 4)
    got = sea.eta(x[:, None], y[None, :], 1.7)
    assert got.shape == (5, 4)
    for i, px in enumerate(x):
        for j, py in enumerate(y):
            assert got[i, j] == pytest.approx(wave_height(px, py, 1.7, config.WaterParams.WAVES_DEFAULT))
    assert wave_height(1.2, -0.4, 0.8, sea) == pytest.approx(float(sea.eta(1.2, -0.4, 0.8)))


def test_wall_raises_the_crest_and_shadow_lowers_the_lee():
    open_sea = Sea('regular')
    walled = Sea('regular')
    walled.add_wall((0.0, 0.0), (-1.0, 0.0), 0.8)
    times = np.linspace(0.0, 6.0, 120)
    open_crest = max(abs(float(open_sea.eta(-0.05, 0.0, t))) for t in times)
    wall_crest = max(abs(float(walled.eta(-0.05, 0.0, t))) for t in times)
    assert wall_crest > open_crest

    shaded = Sea('regular')
    shaded.add_shadow((0.0, 0.0), 0.0, 0.5, 0.5)
    ahead = max(abs(float(shaded.eta(-3.0, 0.0, t))) for t in times)
    behind = max(abs(float(shaded.eta(2.0, 0.0, t))) for t in times)
    assert behind < ahead


def test_scale_replaces_instead_of_stacking():
    sea = Sea('three')
    sea.scaled(2)
    sea.scaled(2)
    assert float(np.max(np.abs(sea.groups[0][1]))) == pytest.approx(0.4)
    sea.scaled(1)
    assert float(np.max(np.abs(sea.groups[0][1]))) == pytest.approx(0.2)


def test_wave_height_ignores_objects_that_only_look_like_a_sea():
    class _Fake:
        def eta(self, x, y, t):
            raise AssertionError('Sea 가 아니면 eta 를 쓰면 안 됩니다.')

        def __iter__(self):
            yield (0.1, 8.0, 0.0, 0.0)

    assert wave_height(0.0, 0.0, 0.0, _Fake()) == pytest.approx(
        wave_height(0.0, 0.0, 0.0, ((0.1, 8.0, 0.0, 0.0),)))
