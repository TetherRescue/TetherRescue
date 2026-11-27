"""발견 → 배 찾기 → 물가에 띄우기. 박자가 서로 다르게 보여야 한다."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.world.coast_map import SHORE_X
from sim.world.map_wolmido import WolmidoMap
from sim.world.shore_call import ShoreCall, hanagae_shore_call, wolmido_shore_call
from sim.world.view_wolmido import beach_start


def _beat_start(call, phase):
    elapsed = 0.0
    for beat in call.route.beats:
        if beat['name'] == phase:
            return elapsed
        elapsed += beat['duration']
    raise AssertionError(phase)


def test_four_beats_use_different_poses_on_hanagae():
    call = hanagae_shore_call((SHORE_X - 40.0, 0.0))
    names = []
    previous = None
    boats = []
    t = 0.0
    while t <= call.duration:
        sample = call.sample(t)
        if sample['phase'] != previous:
            names.append(sample['phase'])
            previous = sample['phase']
        if sample['phase'] == 'push':
            boats.append(sample['boat'][0])
        t += 0.25
    assert names == ['walk', 'spot', 'seek', 'push', 'aside']
    assert boats[0] > SHORE_X > boats[-1]
    assert boats[-1] < boats[0] - 5.0

    # 한 주기의 1/4 지점이면 팔·다리 흔들림이 가장 크다.
    walk = ShoreCall.pose('walk', 0.25 / 1.45)
    seek = ShoreCall.pose('seek', 0.25 / 3.6)
    spot = ShoreCall.pose('spot', 1.0)
    push = ShoreCall.pose('push', 0.4)
    assert spot['arm_r'] > 2.2
    assert abs(walk['arm_r']) < 0.8
    assert abs(seek['leg_l']) > 0.9 > abs(walk['leg_l'])
    assert push['torso'] > 0.5 > walk['torso']
    assert push['pelvis'] < walk['pelvis'] - 0.2
    assert abs(call.sample(_beat_start(call, 'seek') + 2.0)['feet'][1]) < abs(call.sample(0.4)['feet'][1])


def test_wolmido_pushes_the_boat_from_dry_sand_to_the_water_start():
    for drop in (0.0, config.Wolmido.LOW_TIDE_DROP):
        world_map = WolmidoMap(tide_drop=drop)
        start, yaw, _, _ = beach_start(world_map, (-38.0, 4.0))
        call = wolmido_shore_call(world_map, start, yaw, (-38.0, 4.0))
        assert call.route.boat_stow[0] > start[0] + 1.0
        assert call.sample(0.0)['phase'] == 'walk'
        assert call.sample(0.0)['feet'][2] > config.Wolmido.WALL_TOP_Z + drop
        end = call.sample(call.duration)['boat']
        np.testing.assert_allclose(end[:2], start, atol=1e-6)
        assert call.sample(call.duration)['phase'] == 'aside'


def test_skip_launch_starts_with_the_boat_afloat():
    from sim.world.coast_map import SHORE_X
    from sim.world.sortie import RescueSortie, SortieSettings, SortieStage

    sortie = RescueSortie(settings=SortieSettings(launch_delay=0.0, launch_hold=10.0, victim_range=30.0),
                          skip_intro=True)
    assert sortie.stage == SortieStage.LAUNCHING
    assert sortie.shore.done
    assert sortie.world.data.xpos[sortie.robot.body_id, 0] < SHORE_X
    assert sortie.world.model.geom('shore_caller_head').type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
    assert sortie.world.model.geom('hanagae_bather_0_torso').type == mujoco.mjtGeom.mjGEOM_MESH


def test_prelude_steps_stay_finite_on_both_maps():
    from sim.run_sim import SceneSetup, open_session
    import argparse

    args = argparse.Namespace(launch_delay=0.0, victim_range=30.0)
    hanagae = open_session(SceneSetup('hanagae', waves=False), args)
    wolmido = open_session(SceneSetup('wolmido', waves=False, tide='high', weather='clear'), args)
    for session in (hanagae, wolmido):
        for _ in range(30):
            session.step()
        data = session.world.data
        assert np.isfinite(data.qpos).all()
        assert not data.warning.number.any()
        assert session.shore.phase_name() == 'walk'
