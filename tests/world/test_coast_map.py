"""하나개 해안 맵 조립과 배·익수자·물 결합을 검사한다."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.world.coast_map import (
    CoastalMap, LAND_EAST, SHORE_X, VICTIM_RANGE, build_hanagae_world,
)
from sim.world.world import advance_world, check_geom_table


def test_coast_map_land_is_on_east_edge_and_victim_is_offshore():
    world, robot, water, human = build_hanagae_world()
    check_geom_table(world.model)
    names = {world.model.geom(i).name for i in range(world.model.ngeom)}
    for required in ('hanagae_seabed', 'hanagae_tidal_flat', 'hanagae_beach',
                     'hanagae_land', 'hanagae_road', 'hanagae_water_surface',
                     config.Names.HULL_GEOM):
        assert required in names
    assert config.Names.WATER_SURFACE_GEOM not in names  # Water 기본 50m 평면은 끔

    land = world.model.geom('hanagae_land').id
    beach = world.model.geom('hanagae_beach').id
    # 육지는 해안보다 동쪽, 맵 동쪽 끝까지 닿는다
    assert world.model.geom_pos[land, 0] > world.model.geom_pos[beach, 0] > 0
    land_east = world.model.geom_pos[land, 0] + world.model.geom_size[land, 0]
    assert land_east == pytest.approx(LAND_EAST)
    # 지면 바닥이 수면 아래라 떠 보이지 않는다
    land_bottom = world.model.geom_pos[land, 2] - world.model.geom_size[land, 2]
    assert land_bottom < config.WaterParams.SURFACE_Z - 1.0

    boat_x = float(world.data.xpos[robot.body_id, 0])
    victim_x = float(world.data.xpos[human.body_id, 0])
    assert boat_x < SHORE_X
    assert victim_x == pytest.approx(SHORE_X - VICTIM_RANGE, abs=.05)
    assert abs(victim_x - SHORE_X) == pytest.approx(VICTIM_RANGE, abs=.05)
    assert water.body_id == robot.body_id


def test_boat_and_victim_stay_finite_on_hanagae_water():
    world, robot, _, human = build_hanagae_world()
    for _ in range(500):
        advance_world(world)
    assert np.isfinite(world.data.qpos).all()
    assert np.isfinite(world.data.qvel).all()
    assert not world.data.warning.number.any()
    assert world.data.xpos[robot.body_id, 2] > config.WaterParams.SURFACE_Z - 1.0
    assert world.data.xpos[human.body_id, 2] > config.WaterParams.SURFACE_Z - 2.0


def test_unknown_coast_scene_is_rejected():
    with pytest.raises(ValueError, match='hanagae'):
        CoastalMap(scene='other')
