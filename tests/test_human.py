"""사람 강체의 질량·관성·이동 자유도와 실제 자유낙하를 검증한다."""

import numpy as np
import pytest

from sim import config
from sim.human import Human
from sim.world import build_world

mujoco = pytest.importorskip("mujoco")


def test_default_world_includes_human():
    world = build_world()
    assert world.model.body(config.Names.HUMAN_BODY).mass[0] == pytest.approx(config.Human.MASS)
    assert any(isinstance(part, Human) and part.body_id is not None for part in world.parts)


def test_build_and_bind_preserve_mass_shape_and_free_joint():
    human = Human(position=(2, 3, 4), shape="box")
    world = build_world(parts=[human])
    model = world.model
    assert model.nbody == 2  # 월드와 사람
    assert model.nq == 7 and model.nv == 6  # 위치+quaternion, 선속도+각속도
    assert model.jnt_type[human.joint_id] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body_mass[human.body_id] == pytest.approx(70)
    assert model.body_subtreemass[human.body_id] == pytest.approx(70)
    np.testing.assert_allclose(model.geom_size[human.geom_id], [0.85, 0.2, 0.1])
    np.testing.assert_allclose(model.body_inertia[human.body_id],
                               [1.1666666666666667, 17.091666666666665, 17.791666666666668])
    np.testing.assert_allclose(world.data.xipos[human.body_id], [2, 3, 4])
    assert model.geom_contype[human.geom_id] == model.geom_conaffinity[human.geom_id] == 0
    assert model.nsite == 0 and model.ntendon == 0


def test_human_falls_under_gravity_for_thousand_steps():
    human = Human()
    world = build_world(parts=[human])
    for _ in range(1000):
        mujoco.mj_forward(world.model, world.data)
        world.acc.reset()
        human.add_forces(world.model, world.data, world.acc)
        world.acc.apply(world.model, world.data)
        mujoco.mj_step(world.model, world.data)
        for state in (world.data.qpos, world.data.qvel, world.data.qacc):
            assert np.isfinite(state).all()
    elapsed = 1000 * config.Sim.TIMESTEP
    np.testing.assert_allclose(world.data.qvel[:3], [0, 0, -config.Physics.GRAVITY * elapsed], atol=1e-10)
    np.testing.assert_allclose(world.data.qvel[3:], 0, atol=1e-10)
    assert world.data.qpos[2] < 1.0
    assert not world.data.warning.number.any()


def test_com_is_read_from_config(monkeypatch):
    monkeypatch.setattr(config.Human, "COM", (0.1, 0, 0))
    human = Human(position=(2, 3, 4), shape="box")
    world = build_world(parts=[human])
    np.testing.assert_allclose(world.data.xipos[human.body_id], [2.1, 3, 4])


def test_bind_fails_if_human_is_missing():
    with pytest.raises(ValueError, match="build.*compile"):
        Human().bind(mujoco.MjSpec().compile())


@pytest.mark.parametrize("name,value", [("MASS", -1), ("LENGTH", 0), ("HEIGHT", float("nan"))])
def test_invalid_physical_parameters_fail_before_compile(monkeypatch, name, value):
    monkeypatch.setattr(config.Human, name, value)
    with pytest.raises(ValueError):
        build_world(parts=[Human(shape="box")])


def test_humanoid_has_detailed_fixed_parts_and_total_mass_seventy():
    human = Human()
    world = build_world(parts=[human])
    model = world.model
    assert len(human.geom_ids) == model.ngeom
    assert model.geom_type[human.geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    for part in ("forearm_l", "forearm_r", "shin_l", "shin_r", "hand_l", "hand_r", "foot_l", "foot_r", "nose"):
        assert f"{config.Names.HUMAN_DETAIL_PREFIX}_{part}" in human.geom_ids
    assert model.nbody == 2 and model.nv == 6  # 팔다리 개별 운동 관절은 아직 없음
    assert model.body_mass[human.body_id] == pytest.approx(70)
    assert model.body_subtreemass[human.body_id] == pytest.approx(70)
    assert np.all(model.body_inertia[human.body_id] > 0)
    # 좌우 대칭과 몸통 내부 COM, 질량 중복·잘못된 오프셋을 확인한다.
    com = model.body_ipos[human.body_id]
    assert abs(com[1]) < 1e-10
    assert -.1 < com[0] < .15 and abs(com[2]) < .03
    for geom_id in human.geom_ids.values():
        assert model.geom_bodyid[geom_id] == human.body_id


def test_humanoid_uses_total_mass_from_config(monkeypatch):
    monkeypatch.setattr(config.Human, "MASS", 80)
    world = build_world(parts=[Human()])
    assert world.model.body(config.Names.HUMAN_BODY).mass[0] == pytest.approx(80)


@pytest.mark.parametrize("fractions", [(1, 1, 1, 1, 1, 1), (0.5, 0.5), (0, 0.2, 0.2, 0.2, 0.2, 0.2)])
def test_humanoid_rejects_invalid_mass_distribution(monkeypatch, fractions):
    monkeypatch.setattr(config.Human, "MASS_FRACTIONS", fractions)
    with pytest.raises(ValueError, match="MASS_FRACTIONS"):
        build_world(parts=[Human()])
