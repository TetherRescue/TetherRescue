"""구명환의 실제 구멍·총질량·고정 연결·연결점 좌표 검증."""

import numpy as np
import pytest

from sim import config
from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.world import build_world

mujoco = pytest.importorskip("mujoco")


def test_mass_and_fixed_attachment_do_not_duplicate_human_mass():
    human, ring = Human(), Lifebuoy()
    world = build_world(parts=[human, ring])
    model = world.model
    assert model.ngeom == 18
    assert model.nq == 7 and model.nv == 6  # 환에 별도 자유 관절을 추가하지 않는다.
    assert model.body_parentid[ring.body_id] == human.body_id
    assert model.body_mass[human.body_id] == pytest.approx(70)
    assert model.body_mass[ring.body_id] == pytest.approx(2)
    assert model.body_subtreemass[human.body_id] == pytest.approx(72)
    assert model.site_bodyid[ring.site_id] == ring.body_id


def test_center_is_open_but_rim_intercepts_ray():
    world = build_world(parts=[Lifebuoy(attached=False)])
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


def test_tow_site_and_fixed_offset_follow_rotating_human_for_thousand_steps():
    human, ring = Human(), Lifebuoy()
    world = build_world(parts=[human, ring])
    # 사람을 z축으로 90도 회전시킨 뒤 환에 편심 힘을 가해 함께 회전시킨다.
    world.data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    for _ in range(1000):
        mujoco.mj_forward(world.model, world.data)
        rotation = world.data.xmat[human.body_id].reshape(3, 3)
        expected_ring = world.data.xpos[human.body_id] + rotation @ config.Lifebuoy.POSITION_IN_HUMAN_B
        np.testing.assert_allclose(world.data.xpos[ring.body_id], expected_ring, atol=1e-10)
        expected_site = expected_ring + rotation @ config.Lifebuoy.TOW_POINT_B
        np.testing.assert_allclose(world.data.site_xpos[ring.site_id], expected_site, atol=1e-10)
        world.acc.reset()
        world.acc.add(ring.body_id, [1, 0, 0], point_w=world.data.site_xpos[ring.site_id])
        world.acc.apply(world.model, world.data)
        mujoco.mj_step(world.model, world.data)
        assert np.isfinite(world.data.qpos).all()
        assert np.isfinite(world.data.qvel).all()
    assert not world.data.warning.number.any()


def test_default_world_includes_lifebuoy():
    world = build_world()
    assert world.model.body(config.Names.LIFEBUOY_BODY).id > 0
    assert world.model.site(config.Names.LIFEBUOY_TOW_SITE).id >= 0


def test_humanoid_fits_hole_and_arms_extend_outside_without_penetration():
    human, ring = Human(), Lifebuoy()
    world = build_world(parts=[human, ring])
    model, data = world.model, world.data
    ring_ids = np.flatnonzero(model.geom_bodyid == ring.body_id)
    for human_id in human.geom_ids.values():
        for ring_id in ring_ids:
            assert mujoco.mj_geomDistance(model, data, human_id, ring_id, 1.0, None) > 0
    # 상체 중심은 환 구멍 안에 있고, 손 쪽 끝은 환의 바깥 반지름을 넘는다.
    opening_radius = (config.Lifebuoy.CENTERLINE_RADIUS * np.cos(np.pi / config.Lifebuoy.SEGMENTS)
                      - config.Lifebuoy.TUBE_RADIUS)
    assert config.Human.TORSO_RADIUS < opening_radius
    outer_radius = config.Lifebuoy.CENTERLINE_RADIUS + config.Lifebuoy.TUBE_RADIUS
    assert config.Human.ARM_L_FROMTO_B[1][1] > outer_radius
    assert config.Human.ARM_R_FROMTO_B[1][1] < -outer_radius


def test_attached_ring_needs_human_first():
    with pytest.raises(ValueError, match="Human.build"):
        build_world(parts=[Lifebuoy()])


@pytest.mark.parametrize("name,value", [("SEGMENTS", 2), ("TUBE_RADIUS", 0.5), ("MASS", -1)])
def test_invalid_geometry_or_mass_is_rejected(monkeypatch, name, value):
    monkeypatch.setattr(config.Lifebuoy, name, value)
    with pytest.raises(ValueError):
        build_world(parts=[Lifebuoy(attached=False)])
