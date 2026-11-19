"""월미도 해안 맵: 형상 위치·간조 이동·충돌 비트·실제 배 충돌, 환경별 구조 시나리오."""

import math

import numpy as np
import pytest

from sim import config
from sim.world.map_wolmido import WolmidoMap
from sim.world.world import build_world

mujoco = pytest.importorskip("mujoco")


def _riprap_top_surface(model, data):
    geom = model.geom(config.Names.WOLMIDO_RIPRAP_GEOM).id
    rotation, center, size = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom], model.geom_size[geom]
    return [(center + rotation @ [sign * size[0], 0, size[2]])[[0, 2]] for sign in (1, -1)]


@pytest.mark.parametrize("drop", [0.0, config.Wolmido.LOW_TIDE_DROP])
def test_terrain_matches_config_and_low_tide_raises_it(drop):
    cfg = config.Wolmido
    world_map = WolmidoMap(tide_drop=drop)
    world = build_world(parts=[world_map])  # 조립 시 Groups·Contacts 표 검사도 통과해야 한다
    model, data = world.model, world.data
    top, toe = _riprap_top_surface(model, data)
    np.testing.assert_allclose(top, [cfg.WALL_FACE_X + cfg.RIPRAP_TOP[0], cfg.RIPRAP_TOP[1] + drop], atol=1e-9)
    np.testing.assert_allclose(toe, [cfg.WALL_FACE_X + cfg.RIPRAP_TOE[0], cfg.RIPRAP_TOE[1] + drop], atol=1e-9)
    wall = model.geom(config.Names.WOLMIDO_WALL_GEOM).id
    assert data.geom_xpos[wall][0] - model.geom_size[wall][0] == pytest.approx(cfg.WALL_FACE_X)
    assert data.geom_xpos[wall][2] + model.geom_size[wall][2] == pytest.approx(cfg.WALL_TOP_Z + drop)
    assert model.nv == 0 and model.nbody == 1  # 고정 지형: 질량·자유도 없음
    kinds = {name.split('_')[1] for name in world_map.geom_ids}
    assert kinds == {'wall', 'riprap', 'boulder', 'tetrapod', 'step', 'pontoon', 'pile', 'gangway', 'moored', 'buoy',
                     'beach'}
    assert sum('tetrapod' in name for name in world_map.geom_ids) == 4 * len(cfg.TETRAPODS)
    # 물에 뜬 폰툰·부표는 간조에도 수면 높이에 남고, 바닥에 박힌 말뚝은 지형과 함께 올라간다.
    surface = config.WaterParams.SURFACE_Z
    pontoon = model.geom(f"{config.Names.WOLMIDO_PREFIX}_pontoon").id
    assert data.geom_xpos[pontoon][2] + model.geom_size[pontoon][2] == pytest.approx(surface + cfg.PONTOON_FREEBOARD)
    pile = model.geom(f"{config.Names.WOLMIDO_PREFIX}_pile_0").id
    assert data.geom_xpos[pile][2] + model.geom_size[pile][1] == pytest.approx(cfg.PILE_TOP_Z + drop)
    boulder_tops = [data.geom_xpos[i][2] + model.geom_size[i][2] for name, i in world_map.geom_ids.items()
                    if 'boulder' in name]
    # 만조에는 바위가 모두 물속, 간조에는 수면 위로 드러난다.
    assert all((top > surface) if drop else (top < surface - 1.5) for top in boulder_tops)


def test_terrain_collides_with_boat_person_and_ring_but_not_itself():
    contacts = config.Contacts
    for kind in (contacts.BOAT, contacts.HUMAN, contacts.LIFEBUOY):
        assert contacts.ENV_AFFINITY & kind
    assert not contacts.ENV_AFFINITY & contacts.ENV
    assert (contacts.ENV, contacts.ENV_AFFINITY) in contacts.ALLOWED_PHYSICS


def test_boat_pushed_into_seawall_does_not_pass_through():
    from sim.robot.robot import Robot
    from sim.water.water import Water

    draft = config.Hull.MASS / (config.Physics.RHO_WATER * config.Hull.LENGTH * config.Hull.BEAM)
    boat = Robot(position=(config.Wolmido.WALL_FACE_X - config.Hull.LENGTH / 2 - .3, -3.,  # 해변이 아닌 방파제 앞
                           config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - draft))
    world_map = WolmidoMap()
    world = build_world(parts=[Water(waves=()), boat, world_map])
    model, data = world.model, world.data
    boat.set_command(1.0, 0.0)  # 최대 전진으로 방파제(+x)를 계속 민다
    hull = [g for g in range(model.ngeom) if model.geom_bodyid[g] == boat.body_id and model.geom_contype[g]]
    wall = world_map.geom_ids[config.Names.WOLMIDO_WALL_GEOM]
    deepest, touched = 0., False
    from sim.world.world import advance_world
    for step in range(2500):
        advance_world(world)
        if step % 10 == 0:
            gap = min(mujoco.mj_geomDistance(model, data, g, wall, 1., None) for g in hull)
            deepest, touched = max(deepest, -gap), touched or gap < .005
    assert touched
    assert deepest < .005
    assert np.isfinite(data.qpos).all() and not data.warning.number.any()


@pytest.mark.parametrize("drop", [0.0, config.Wolmido.LOW_TIDE_DROP])
def test_boat_starts_at_beach_waterline(drop):
    world_map = WolmidoMap(tide_drop=drop)
    model = build_world(parts=[world_map]).model
    x = world_map.beach_x_at_depth(config.Wolmido.BEACH_START_DEPTH)
    # 그 지점 바로 위에서 아래로 쏜 광선이 해변 경사에 닿는 높이 = 수면 - 출발 깊이
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array([x, config.Wolmido.BEACH_START_Y, 5.]), np.array([0., 0., -1.]),
                             None, True, -1, hit)
    assert model.geom(hit[0]).name == f"{config.Names.WOLMIDO_PREFIX}_beach"
    assert 5. - distance == pytest.approx(config.WaterParams.SURFACE_Z - config.Wolmido.BEACH_START_DEPTH, abs=1e-6)


def test_environments_are_named_and_unknown_is_rejected():
    assert set(config.Wolmido.ENVIRONMENTS) == {"calm", "waves", "low_tide"}
    for name, setting in config.Wolmido.ENVIRONMENTS.items():
        assert hasattr(config.WaterParams, setting["waves"]), name
        # 사람은 바다 한가운데, 해변 출발 지점에서 15m 이상 떨어져 있다.
        from sim.world.view_wolmido import beach_start
        start, _, _, tow = beach_start(WolmidoMap(tide_drop=setting["tide_drop"]), setting["victim"])
        assert np.linalg.norm(np.asarray(setting["victim"]) - start) > 15
        assert 2 < tow <= config.Wolmido.BEACH_TOW_MAX
    with pytest.raises(ValueError):
        WolmidoMap.environment("storm")
    with pytest.raises(ValueError):
        WolmidoMap(tide_drop=-1)


def test_lifebuoy_starts_on_the_robot_not_with_the_person():
    from sim.world.view_wolmido import build_wolmido

    mission, _ = build_wolmido("calm")
    data = mission.world.data
    stern = data.site_xpos[mission.stern_site]
    ring = data.xpos[mission.ring_body]
    assert np.linalg.norm(ring[:2] - stern[:2]) < 1.5          # 배 뒤 짧은 줄 끝
    assert mission.reach.gap(data) > 20                          # 사람 손과는 멀리
    assert not mission.grasp.is_held(data)
    assert mission.robot.delivery_state == "STOWED"
    assert mission.robot.rope_length(mission.world.model) == pytest.approx(config.Winch.STOW_LENGTH)
    # 물에 눕힌 자세: 구멍 축(로컬 x)이 위를 본다.
    axis = data.xmat[mission.ring_body].reshape(3, 3)[:, 0]
    assert axis[2] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("env", ["calm", "low_tide"])
def test_robot_delivers_lifebuoy_then_tows_without_touching_terrain(env):
    from sim.world.view_wolmido import boat_terrain_clearance, build_wolmido

    mission, world_map = build_wolmido(env)
    model = mission.world.model
    def geoms(body):
        return [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g]]

    human_geoms, ring_geoms = geoms(mission.human_body), geoms(mission.ring_body)
    terrain = list(world_map.geom_ids.values())
    boat_min = human_min = ring_min = math.inf
    seen, steps = [], 0
    while mission.world.data.time < 300 and mission.stage != "COMPLETE":
        assert mission.step(), mission.failure
        steps += 1
        if not seen or seen[-1] != mission.stage:
            seen.append(mission.stage)
        if steps % 50 == 0:
            data = mission.world.data
            boat_min = min(boat_min, boat_terrain_clearance(mission, world_map))
            human_min = min(human_min, min(mujoco.mj_geomDistance(model, data, a, b, 10., None)
                                           for a in human_geoms for b in terrain))
            ring_min = min(ring_min, min(mujoco.mj_geomDistance(model, data, a, b, 10., None)
                                         for a in ring_geoms for b in terrain))
    assert seen == ["APPROACH", "DELIVER", "GRAB", "TOW", "COMPLETE"]
    assert mission.grasp.is_held(mission.world.data)
    assert mission.readings()["tow_progress"] > 2
    # 배·구명환은 해변 물가(물 깊이 0.6m)에서 출발하므로 바닥과 가깝다. 닿지만 않으면 된다. 사람은 지형에서 멀다.
    assert boat_min > .2 and ring_min > 0 and human_min > 1.0
