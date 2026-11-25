"""최종 실행이 맵 둘과 파도·물때·날씨 설정을 따로 여는지 확인한다."""
import argparse

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim.run_sim import MAPS, SceneSetup, open_session


def _args():
    return argparse.Namespace(launch_delay=.05, victim_range=30.)


def test_two_maps_and_wolmido_settings_are_separate():
    assert MAPS == ('hanagae', 'wolmido')
    hanagae = open_session(SceneSetup('hanagae', waves=False), _args())
    assert hanagae.step() is False
    assert np.isfinite(hanagae.world.data.qpos).all()
    wolmido = open_session(SceneSetup('wolmido', waves=True, tide='low', weather='rain'), _args())
    assert wolmido.world.model.geom('wolmido_wall').id >= 0
    assert wolmido.sky.name == 'rain'
    assert wolmido.step() is False
    assert np.isfinite(wolmido.world.data.qpos).all()