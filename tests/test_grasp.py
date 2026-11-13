"""배 없이 사람↔구명환↔로프의 받기 전·잡기·놓기 힘 전달을 검증한다."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.frames import body_velocity_w
from sim.grasp import HumanGrasp, GraspSettings
from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.scenarios import GraspDemo
from sim.world import build_world


def physics_step(world):
    mujoco.mj_forward(world.model, world.data)
    world.acc.reset()
    for part in world.parts:
        part.add_forces(world.model, world.data, world.acc)
    world.acc.apply(world.model, world.data)
    mujoco.mj_step(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)


def acceleration(world):
    mujoco.mj_forward(world.model, world.data)
    world.acc.reset()
    for part in world.parts:
        part.add_forces(world.model, world.data, world.acc)
    world.acc.apply(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)


def test_initially_independent_not_worn_and_rope_already_connected():
    demo = GraspDemo()
    model, data = demo.world.model, demo.world.data
    assert model.body_parentid[demo.ring.body_id] == 0
    assert model.body_parentid[demo.human.body_id] == 0
    assert model.body_mass[demo.human.body_id] == pytest.approx(70)
    assert model.body_mass[demo.ring.body_id] == pytest.approx(2)
    assert model.nv == 18  # 시험 질점·사람·구명환 각각 자유 관절
    assert not demo.grasp.is_held(data)
    assert demo.rope.attached
    assert demo.rope.body_ids[1] == demo.ring.body_id
    assert config.Names.BOAT_BODY not in [model.body(i).name for i in range(model.nbody)]
    # 환은 몸통을 둘러싸지 않고 손 옆에 있다.
    assert data.xpos[demo.ring.body_id, 1] > .9
    ring_geoms = np.flatnonzero(model.geom_bodyid == demo.ring.body_id)
    assert min(mujoco.mj_geomDistance(model, data, human, ring, 1., None)
               for human in demo.human.geom_ids.values() for ring in ring_geoms) > 0


def test_rope_pulls_only_ring_before_receipt():
    demo = GraspDemo()
    world = demo.world
    world.data.qpos[0] += 1.5  # 시험 초기 배치: 줄을 기준 길이보다 1m 늘린다.
    acceleration(world)
    dof = world.model.jnt_dofadr[demo.human.joint_id]
    np.testing.assert_allclose(world.data.qacc[dof:dof + 6], 0, atol=1e-10)
    ring_joint = world.model.joint(config.Names.LIFEBUOY_FREE).id
    ring_dof = world.model.jnt_dofadr[ring_joint]
    assert world.data.qacc[ring_dof] == pytest.approx(50.)
    assert demo.rope.tension(world.model, world.data) == pytest.approx(100.)


@pytest.mark.parametrize('condition', ['no_signal', 'far', 'fast', 'spinning', 'nonfinite'])
def test_grab_rejects_invalid_delivery_conditions(condition):
    demo = GraspDemo()
    world = demo.world
    ring_joint = world.model.joint(config.Names.LIFEBUOY_FREE).id
    qadr, dof = world.model.jnt_qposadr[ring_joint], world.model.jnt_dofadr[ring_joint]
    if condition == 'far':
        world.data.qpos[qadr] += .5
    elif condition == 'fast':
        world.data.qvel[dof] = .3
    elif condition == 'spinning':
        # 잡기 site를 통과하는 회전축: site 선속도는 0에 가깝지만 회전은 빠르다.
        offset = world.model.site_pos[demo.grasp.ring_site]
        omega = 2 * offset / np.linalg.norm(offset)
        world.data.qvel[dof + 3:dof + 6] = omega
    mujoco.mj_forward(world.model, world.data)
    if condition == 'nonfinite':
        world.data.qvel[dof] = float('nan')
    assert not demo.grasp.try_grab(world.model, world.data, delivery_ready=condition != 'no_signal')
    assert not demo.grasp.is_held(world.data)


def test_grab_captures_current_pose_without_teleport_or_velocity_reset():
    demo = GraspDemo()
    world = demo.world
    # 전체를 90도 돌려 최초 배치에 끌려가는 잘못된 고정을 검출한다.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for name in (config.Names.HUMAN_FREE, config.Names.LIFEBUOY_FREE):
        joint = world.model.joint(name).id
        address = world.model.jnt_qposadr[joint]
        world.data.qpos[address:address + 3] = rotation @ world.data.qpos[address:address + 3]
        world.data.qpos[address + 3:address + 7] = [2**-.5, 0, 0, 2**-.5]
    ring_joint = world.model.joint(config.Names.LIFEBUOY_FREE).id
    ring_address = world.model.jnt_qposadr[ring_joint]
    tilted = np.empty(4)
    mujoco.mju_mulQuat(tilted, world.data.qpos[ring_address + 3:ring_address + 7],
                      np.array([np.cos(.2), np.sin(.2), 0., 0.]))
    world.data.qpos[ring_address + 3:ring_address + 7] = tilted
    mujoco.mj_forward(world.model, world.data)
    world.data.qpos[ring_address:ring_address + 3] += (
        world.data.site_xpos[demo.grasp.hand_site] - world.data.site_xpos[demo.grasp.ring_site] + [.02, 0, 0])
    mujoco.mj_forward(world.model, world.data)
    world.data.qvel[:] = .001
    mujoco.mj_forward(world.model, world.data)
    qpos, qvel, time = world.data.qpos.copy(), world.data.qvel.copy(), world.data.time
    assert demo.grasp.try_grab(world.model, world.data, delivery_ready=True)
    np.testing.assert_array_equal(world.data.qpos, qpos)
    np.testing.assert_array_equal(world.data.qvel, qvel)
    assert world.data.time == time
    mujoco.mj_forward(world.model, world.data)
    np.testing.assert_allclose(world.data.efc_pos[:6], 0, atol=1e-10)
    # 중복 전달 신호로 제약이나 외력이 중복 생성되지 않는다.
    assert demo.grasp.try_grab(world.model, world.data, delivery_ready=True)
    assert world.model.neq == 1


def test_human_receives_rope_force_only_after_grabbing_ring():
    demo = GraspDemo()
    world = demo.world
    world.data.qpos[0] += 1.5
    mujoco.mj_forward(world.model, world.data)
    assert demo.grasp.try_grab(world.model, world.data, delivery_ready=True)
    acceleration(world)
    dof = world.model.jnt_dofadr[demo.human.joint_id]
    assert world.data.qacc[dof] > 1.
    initial = world.data.xpos[demo.human.body_id].copy()
    for _ in range(500):
        physics_step(world)
    assert world.data.xpos[demo.human.body_id, 0] > initial[0] + .05
    assert demo.grasp.is_held(world.data)
    assert not world.data.warning.number.any()


def test_releasing_preserves_momentum_and_removes_further_direct_rope_force():
    demo = GraspDemo()
    for _ in range(3500):
        demo.step()  # 7초: 받은 뒤 견인되는 도중.
    world = demo.world
    assert demo.grasp.is_held(world.data)
    qpos, qvel = world.data.qpos.copy(), world.data.qvel.copy()
    assert demo.grasp.release(world.model, world.data)
    np.testing.assert_array_equal(world.data.qpos, qpos)
    np.testing.assert_array_equal(world.data.qvel, qvel)
    velocity, _ = body_velocity_w(world.model, world.data, demo.human.body_id)
    assert np.linalg.norm(velocity) > .05
    start = world.data.xipos[demo.human.body_id].copy()
    for _ in range(500):
        demo.fixture.target[0] += .001
        physics_step(world)
    new_velocity, _ = body_velocity_w(world.model, world.data, demo.human.body_id)
    np.testing.assert_allclose(new_velocity, velocity, atol=1e-8)
    assert np.linalg.norm(world.data.xipos[demo.human.body_id] - start) > .05
    assert world.data.nefc == 0
    assert not demo.grasp.is_held(world.data)


def test_right_hand_can_receive_the_opposite_rim():
    human = Human(grip_sites=True)
    hand = np.asarray(config.Human.GRIP_R_POINT_B) + human.position
    ring = Lifebuoy(attached=False, grip_sites=True,
                    position=hand - np.asarray(config.Lifebuoy.GRIP_R_POINT_B) + [.02, 0, 0])
    grasp = HumanGrasp(hand='right')
    world = build_world(parts=[human, ring, grasp], gravity=0)
    assert grasp.try_grab(world.model, world.data, delivery_ready=True)
    assert grasp.is_held(world.data)


def test_reset_releases_constraint_and_restores_its_reference():
    demo = GraspDemo()
    world = demo.world
    reference = world.model.eq_data.copy()
    assert demo.grasp.try_grab(world.model, world.data, delivery_ready=True)
    mujoco.mj_resetData(world.model, world.data)
    demo.grasp.reset(world.model, world.data)
    assert not demo.grasp.is_held(world.data)
    np.testing.assert_array_equal(world.model.eq_data, reference)


def test_gravity_buoyancy_and_drag_with_independent_person_and_ring():
    demo = GraspDemo(water=True)
    assert demo.world.model.opt.gravity[2] < 0
    assert not demo.grasp.is_held(demo.world.data)
    water = demo.world.parts[0]
    assert {target['body_id'] for target in water.targets} == {demo.human.body_id, demo.ring.body_id}
    for _ in range(2000):
        demo.step()
    assert demo.grasp.is_held(demo.world.data)
    assert demo.world.data.time == pytest.approx(4.)
    assert np.max(np.abs(demo.world.data.efc_pos)) < .03
    assert demo.rope.tension(demo.world.model, demo.world.data) > 1.
    assert not demo.world.data.warning.number.any()


@pytest.mark.parametrize('options', [dict(max_distance=0), dict(max_relative_speed=-1),
                                     dict(max_relative_angular_speed=float('nan'))])
def test_invalid_grasp_settings_fail(options):
    with pytest.raises(ValueError):
        GraspSettings(**options)


def test_grasp_requires_independent_bodies_and_explicit_sites():
    with pytest.raises(ValueError, match='site'):
        build_world(parts=[Human(), Lifebuoy(attached=False), HumanGrasp()])
    world = build_world(parts=[Human(grip_sites=True), Lifebuoy(grip_sites=True), HumanGrasp()])
    assert world.model.body_parentid[world.model.body(config.Names.LIFEBUOY_BODY).id] == 0


def test_moving_fixture_without_delivery_never_tows_the_person():
    demo = GraspDemo(receive=False)
    start = demo.world.data.xpos[demo.human.body_id].copy()
    ring_start = demo.world.data.xpos[demo.ring.body_id].copy()
    maximum_tension = 0.
    for _ in range(2500):
        demo.step()
        maximum_tension = max(maximum_tension, demo.rope.tension(demo.world.model, demo.world.data))
    # 사람–구명환 접촉이 켜져 있어 끌려가는 환이 왼쪽 아래팔을 잠깐 스친다 (약 1.7 mm 밀림).
    # 줄이 사람을 끄는 것과는 다르므로 cm 단위 이동만 견인으로 본다.
    np.testing.assert_allclose(demo.world.data.xpos[demo.human.body_id], start, atol=5e-3)
    assert np.linalg.norm(demo.world.data.xpos[demo.ring.body_id] - ring_start) > .05
    assert maximum_tension > 1
    assert not demo.grasp.is_held(demo.world.data)
