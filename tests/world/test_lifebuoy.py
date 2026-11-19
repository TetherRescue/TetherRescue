"""구명환의 실제 구멍·총질량·손에 쥔 배치·연결점 좌표 검증. 구명환은 몸에 끼우지 않는다."""

import numpy as np
import pytest

from sim import config
from sim.world.human import Human
from sim.world.lifebuoy import Lifebuoy
from sim.world.world import build_world

mujoco = pytest.importorskip("mujoco")


def test_default_ring_is_an_independent_free_body_with_separate_mass():
    human, ring = Human(), Lifebuoy()
    world = build_world(parts=[human, ring])
    model = world.model
    assert model.ngeom == len(human.geom_ids) + config.Lifebuoy.SEGMENTS
    assert model.nq == 14 and model.nv == 12
    assert model.body_parentid[ring.body_id] == 0
    assert model.body_mass[human.body_id] == pytest.approx(70)
    assert model.body_mass[ring.body_id] == pytest.approx(config.Lifebuoy.MASS)
    assert model.body_subtreemass[human.body_id] == pytest.approx(70)
    assert model.site_bodyid[ring.site_id] == ring.body_id


def test_center_is_open_but_rim_intercepts_ray():
    world = build_world(parts=[Lifebuoy(position=(0, 0, 1))])
    hit_id = np.array([-1], dtype=np.int32)
    direction = np.array([1., 0., 0.])
    # x 방향에서 yz 평면의 중심을 본다. 구멍을 통과하는 광선은 geom에 닿지 않는다.
    center_distance = mujoco.mj_ray(world.model, world.data,
                                   np.array([-1., 0., 1.]), direction, None, True, -1, hit_id)
    assert center_distance == -1
    rim_distance = mujoco.mj_ray(world.model, world.data,
                                np.array([-1., 0.325, 1.]), direction, None, True, -1, hit_id)
    assert rim_distance == pytest.approx(0.925)
    assert hit_id[0] >= 0


def _held(**human_options):
    from sim.world.grasp import HumanGrasp
    human = Human(grip_sites=True, **human_options)
    ring, grasp = Lifebuoy(held_by=human), HumanGrasp(held=True)
    return human, ring, grasp


def test_held_ring_starts_in_hand_and_follows_the_person():
    human, ring, grasp = _held()
    world = build_world(parts=[human, ring, grasp], gravity=0)
    model, data = world.model, world.data
    assert grasp.is_held(data)
    assert grasp.metrics(model, data)['distance'] == pytest.approx(.02, abs=1e-9)
    rotation = data.xmat[human.body_id].reshape(3, 3)
    tow_b = rotation.T @ (data.site_xpos[ring.site_id] - data.xpos[human.body_id])
    np.testing.assert_allclose(tow_b, Lifebuoy.held_tow_point_b(), atol=1e-9)
    start = rotation.T @ (data.xpos[ring.body_id] - data.xpos[human.body_id])
    for _ in range(1000):
        world.acc.reset()
        world.acc.add(ring.body_id, [5, 0, 0], point_w=data.site_xpos[ring.site_id])
        world.acc.apply(model, data)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    rotation = data.xmat[human.body_id].reshape(3, 3)
    relative = rotation.T @ (data.xpos[ring.body_id] - data.xpos[human.body_id])
    np.testing.assert_allclose(relative, start, atol=.002)
    assert data.qvel[0] > 0  # 환을 당긴 힘이 손을 통해 사람에게 전달된다
    assert not data.warning.number.any()


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_held_ring_does_not_touch_or_surround_the_body(size):
    human, ring, grasp = _held(size=size, yaw=.7)
    world = build_world(parts=[human, ring, grasp], gravity=0)
    model, data = world.model, world.data
    ring_ids = np.flatnonzero(model.geom_bodyid == ring.body_id)
    for human_id in human.geom_ids.values():
        if model.geom_contype[human_id]:
            for ring_id in ring_ids:
                assert mujoco.mj_geomDistance(model, data, human_id, ring_id, 1.0, None) > .005
    # 몸통 중심이 환 구멍 안에 있지 않다 (환 중심이 손 쪽 바깥에 있다).
    rotation = data.xmat[human.body_id].reshape(3, 3)
    center_b = rotation.T @ (data.xpos[ring.body_id] - data.xpos[human.body_id])
    assert center_b[1] > config.Lifebuoy.CENTERLINE_RADIUS + config.Lifebuoy.TUBE_RADIUS


def test_default_world_includes_lifebuoy():
    world = build_world()
    assert world.model.body(config.Names.LIFEBUOY_BODY).id > 0
    assert world.model.site(config.Names.LIFEBUOY_TOW_SITE).id >= 0


def test_wearing_is_removed_and_held_ring_needs_human_first():
    with pytest.raises(ValueError, match="삭제"):
        Lifebuoy(attached=True)
    with pytest.raises(ValueError, match="Human"):
        build_world(parts=[Lifebuoy(held_by=Human(grip_sites=True))])
    with pytest.raises(ValueError, match="Human"):
        build_world(parts=[Human(), Lifebuoy(held_by=Human())])  # 잡기 site 없는 사람


@pytest.mark.parametrize("name,value", [("SEGMENTS", 2), ("TUBE_RADIUS", 0.5), ("MASS", -1)])
def test_invalid_geometry_or_mass_is_rejected(monkeypatch, name, value):
    monkeypatch.setattr(config.Lifebuoy, name, value)
    with pytest.raises(ValueError):
        build_world(parts=[Lifebuoy(position=(0, 0, 1))])
