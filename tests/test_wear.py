"""양팔이 굽혀지지 않고 동시에 환을 통과하는 실제 물리 경로를 검증한다."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.scenarios import WearDemo


@pytest.fixture(scope='module')
def motion():
    demo = WearDemo()
    world, wear = demo.world, demo.wear
    result = dict(demo=demo, stages=[], depth=0., head_gap=10., elbow_angle=0.,
                  arm_bend=0., joint_limit_error=0., control_jump=0., crossings={}, symmetry_error=0.)
    original_update = wear.update

    def checked_update(data):
        qpos, qvel, controls = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        original_update(data)
        # 제어기는 실제 상태를 직접 덮어쓰지 않는다.
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        result['control_jump'] = max(result['control_jump'], np.max(np.abs(data.ctrl - controls)))

    wear.update = checked_update
    head = world.model.geom(config.Names.HUMAN_HEAD_GEOM).id
    ring_geoms = np.flatnonzero(world.model.geom_bodyid == demo.ring.body_id)
    for index in range(10000):
        demo.step()
        data, model = world.data, world.model
        if not result['stages'] or result['stages'][-1] != wear.phase:
            result['stages'].append(wear.phase)
        for side in ('l', 'r'):
            joints = demo.human.arm_joints[side]
            angles = data.qpos[wear.qadr[side]]
            limits = model.jnt_range[joints]
            result['joint_limit_error'] = max(result['joint_limit_error'], np.max(limits[:, 0] - angles),
                                              np.max(angles - limits[:, 1]))
            result['elbow_angle'] = max(result['elbow_angle'], abs(angles[3]))
            elbow = model.body(f'{config.Names.HUMAN_ARM_BODY_PREFIX}_{side}_forearm').id
            hand = model.body(f'{config.Names.HUMAN_ARM_BODY_PREFIX}_{side}_hand').id
            shoulder = model.body(f'{config.Names.HUMAN_ARM_BODY_PREFIX}_{side}_upper').id
            upper = data.xpos[elbow] - data.xpos[shoulder]
            forearm = data.xpos[hand] - data.xpos[elbow]
            bend = np.arccos(np.clip(upper @ forearm / (np.linalg.norm(upper) * np.linalg.norm(forearm)), -1., 1.))
            result['arm_bend'] = max(result['arm_bend'], bend)
            normal = data.xmat[wear.ring].reshape(3, 3)[:, 0]
            if (data.site_xpos[wear.sites[side]] - data.xpos[wear.ring]) @ normal > 0:
                result['crossings'].setdefault(side, float(data.time))
        left, right = (wear._local_hand(data, s) for s in ('l', 'r'))
        result['symmetry_error'] = max(result['symmetry_error'], np.linalg.norm(left - right * [1, -1, 1]))
        assert not demo.grasp.is_held(data)
        if wear.phase != 'WORN':
            assert not data.eq_active[wear.eq]
        result['depth'] = min(result['depth'], min((c.dist for c in data.contact), default=0.))
        if index % 20 == 0:
            result['head_gap'] = min(result['head_gap'], min(
                mujoco.mj_geomDistance(model, data, head, ring, 10., None) for ring in ring_geoms))
        if wear.phase == 'WORN':
            break
    assert wear.phase == 'WORN'
    result['completion_time'] = world.data.time
    result['final_state'] = wear.measurements(world.data)
    result['final_angles'] = {s: world.data.qpos[wear.qadr[s]].copy() for s in ('l', 'r')}
    result['final_hands'] = {s: wear._local_hand(world.data, s) for s in ('l', 'r')}
    start = world.data.xpos[demo.human.body_id].copy()
    demo.fixture.target[0] += 2.
    maximum_tension = 0.
    for _ in range(1500):
        demo.step()
        maximum_tension = max(maximum_tension, demo.rope.tension(world.model, world.data))
    result['tow_movement'] = world.data.xpos[demo.human.body_id] - start
    result['maximum_tension'] = maximum_tension
    return result


def test_articulated_arms_keep_mass_and_bounded_internal_motors():
    demo = WearDemo()
    model = demo.world.model
    assert model.body_subtreemass[demo.human.body_id] == pytest.approx(70.)
    assert model.body_mass[demo.ring.body_id] == pytest.approx(2.)
    assert (model.nu, model.nv) == (14, 32)
    np.testing.assert_array_equal(model.actuator_forcerange[:, 1], [40.] * 4 + [22.] * 3 + [40.] * 4 + [22.] * 3)
    np.testing.assert_array_equal(model.actuator_forcerange[:, 0], -model.actuator_forcerange[:, 1])
    assert not demo.world.data.eq_active[demo.wear.eq]


def test_entire_sequence_uses_one_simultaneous_insertion(motion):
    assert motion['stages'] == ['WAITING', 'ALIGN_ARMS', 'INSERT_BOTH', 'WORN']
    assert 13 < motion['completion_time'] < 16
    assert set(motion['crossings']) == {'l', 'r'}
    assert abs(motion['crossings']['l'] - motion['crossings']['r']) <= .01
    assert motion['symmetry_error'] < .002


def test_both_arms_remain_physically_straight_throughout(motion):
    # 실제 위팔·아래팔 벡터도 검사해 0도 관절의 형상 자체가 굽은 경우를 잡는다.
    assert motion['elbow_angle'] < np.deg2rad(1.)
    assert motion['arm_bend'] < np.deg2rad(1.)
    for angles in motion['final_angles'].values():
        np.testing.assert_allclose(angles[3:], 0., atol=.01)


def test_ring_does_not_pass_through_head_or_limbs(motion):
    assert motion['head_gap'] > .03
    assert motion['depth'] > -.003


def test_worn_ring_is_centered_and_arms_stay_extended(motion):
    state = motion['final_state']
    assert state['worn'] and state['ready_to_tow']
    assert state['position_error'] < .015
    assert state['axis_error'] < .03
    for side, hand in motion['final_hands'].items():
        assert hand[0] > 1.05
        assert abs(hand[1]) < .15
        np.testing.assert_allclose(motion['final_angles'][side], motion['demo'].wear.extended[side], atol=.025)


def test_rope_load_reaches_person_and_approach_support_is_off_after_wearing(motion):
    demo = motion['demo']
    assert not demo.grasp.is_held(demo.world.data)
    assert demo.rope.attached
    assert motion['maximum_tension'] > 1.
    assert motion['tow_movement'][0] > .05
    demo.world.acc.reset()
    demo.approach_fixture.add_forces(demo.world.model, demo.world.data, demo.world.acc)
    assert not demo.world.acc._loads
    assert not demo.world.data.warning.number.any()


def test_no_delivery_signal_leaves_person_and_ring_independent():
    demo = WearDemo()
    qpos, qvel = demo.world.data.qpos.copy(), demo.world.data.qvel.copy()
    assert not demo.wear.start(demo.world.data, delivery_ready=False)
    demo.wear.update(demo.world.data)
    assert demo.wear.phase == 'WAITING'
    assert demo.wear.progress == 0.
    assert not demo.grasp.is_held(demo.world.data)
    np.testing.assert_array_equal(demo.world.data.qpos, qpos)
    np.testing.assert_array_equal(demo.world.data.qvel, qvel)


def test_reset_clears_arm_commands_progress_and_torso_support(motion):
    demo = WearDemo()
    for _ in range(4500):
        demo.step()
    assert demo.wear.progress > 0
    demo.reset()
    assert demo.wear.phase == 'WAITING'
    assert demo.wear.progress == 0.
    assert not demo.world.data.eq_active[demo.wear.eq]
    np.testing.assert_array_equal(demo.world.data.ctrl, 0)
    np.testing.assert_array_equal(demo.world.data.qpos, demo.world.model.qpos0)
    np.testing.assert_array_equal(demo.world.data.qvel, 0)
    assert demo.rope.attached


def test_motion_keeps_joint_ranges_and_limits_command_speed(motion):
    assert motion['joint_limit_error'] < .01
    assert motion['control_jump'] <= .025 + 1e-12


def test_approach_fixture_applies_forces_without_rewriting_pose():
    demo = WearDemo()
    data, model = demo.world.data, demo.world.model
    demo.wear.progress = .5
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    demo.world.acc.reset()
    demo.approach_fixture.add_forces(model, data, demo.world.acc)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    assert len(demo.world.acc._loads) == 2
    assert any(np.linalg.norm(load[1]) > 0 for load in demo.world.acc._loads)
