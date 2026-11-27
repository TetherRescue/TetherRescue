"""하나개 해안 맵 조립과 배·익수자·물 결합을 검사한다."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.world.coast_map import (
    REEF_CENTER, REEF_LENGTH, CoastalMap, LAND_EAST, SHORE_X, VICTIM_RANGE, build_hanagae_world,
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
    bathers = [world.model.body(i).name for i in range(world.model.nbody)
               if world.model.body(i).name.startswith('hanagae_bather_')
               and 'arm' not in world.model.body(i).name]
    assert len(bathers) == 10
    for name in bathers:
        assert world.model.body(name).pos[0] < SHORE_X
    reefs = [world.model.geom(i) for i in range(world.model.ngeom)
             if world.model.geom(i).name.startswith('hanagae_reef_')]
    assert len(reefs) >= 5
    assert all(int(np.asarray(rock.type).item()) == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID) for rock in reefs)
    surface = config.WaterParams.SURFACE_Z
    assert min(rock.pos[2] - rock.size[2] for rock in reefs) < surface
    assert max(rock.pos[2] + rock.size[2] for rock in reefs) > surface + .5
    span = max(rock.pos[0] + rock.size[0] for rock in reefs) - min(rock.pos[0] - rock.size[0] for rock in reefs)
    assert span == pytest.approx(REEF_LENGTH, abs=1.5)
    assert all((rock.contype, rock.conaffinity) == (config.Contacts.ENV, config.Contacts.ENV_AFFINITY)
               for rock in reefs)


def test_rising_wave_splashes_instead_of_passing_through_land():
    from sim.world.wave_impact import WaveImpact

    class _Rising:
        def eta(self, x, y, t):
            return np.full(np.broadcast(np.asarray(x, float), np.asarray(y, float)).shape, 0.2)

        def rise(self, x, y, t):
            return np.full(np.broadcast(np.asarray(x, float), np.asarray(y, float)).shape, 1.0)

    impact = WaveImpact(_Rising())
    impact.update(1.0)
    assert len(impact.drops) >= 4
    assert len(impact.bumps) >= 1
    surface = config.WaterParams.SURFACE_Z
    verts = np.array([
        [SHORE_X + 6.0, 2.0, surface + 0.25],
        [REEF_CENTER[0], REEF_CENTER[1], surface + 0.3],
        [-30.0, 8.0, surface + 0.1],
    ])
    out = impact.dress(verts, 1.0)
    assert out[0, 0] < SHORE_X
    assert out[0, 2] <= surface + 0.4
    rx, ry = REEF_LENGTH / 2, 2.0
    nx = (out[1, 0] - REEF_CENTER[0]) / rx
    ny = (out[1, 1] - REEF_CENTER[1]) / ry
    assert nx * nx + ny * ny == pytest.approx(1.0, abs=0.08)
    assert out[2, 0] == pytest.approx(-30.0)
    assert out[2, 2] == pytest.approx(surface + 0.1)


def test_splash_sea_three_changes_height_near_the_reef():
    from sim.water.water import wave_height
    world, *_ = build_hanagae_world(splash_sea=True)
    water = next(part for part in world.parts if hasattr(part, 'waves') and hasattr(part.waves, 'eta'))
    assert water.impact is not None
    plain = config.WaterParams.WAVES_THREE
    x, y = REEF_CENTER[0] + 3.0, 0.0
    assert wave_height(x, y, 1.2, water.waves) != pytest.approx(wave_height(x, y, 1.2, plain))


def test_beachgoers_move_while_the_simulation_runs():
    world, *_ = build_hanagae_world()
    coast = next(part for part in world.parts if isinstance(part, CoastalMap))
    before = world.data.mocap_pos[coast._bathers[0]['mocap']].copy()
    arm = coast._bathers[0]['joints'][0][0]
    for _ in range(120):
        advance_world(world)
    assert not np.allclose(world.data.mocap_pos[coast._bathers[0]['mocap']], before)
    assert abs(world.data.qpos[arm]) > .2


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
