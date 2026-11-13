"""실제 물리 엔진에서 목표 접근·제동·방향 정렬·정지 유지를 확인한다."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim.phase_manager import ApproachSettings, Phase, PhaseManager
from sim.view_approach import build_approach_demo


@pytest.mark.parametrize('start,yaw,target,heading,velocity', [
    ((0, 0), 0., (4, 2), 0., (0, 0)),
    ((6, 3), math.pi, (0, 0), math.pi / 2, (0, 0)),
    ((0, 0), 0., (-3, 0), 0., (0, 0)),
    ((0, 0), 0., (1, 0), 0., (1., 0)),
])
def test_arrives_from_different_positions_and_holds_for_five_seconds(start, yaw, target, heading, velocity):
    manager = build_approach_demo(start=start, target=target, yaw=yaw, target_heading=heading)
    world = manager.world
    dof = world.model.jnt_dofadr[manager.robot.joint_id]
    world.data.qvel[dof:dof + 2] = velocity
    first_arrival = None
    last_time = 0.
    used_reverse = False
    for _ in range(30000):
        manager.step()
        assert world.data.time == pytest.approx(last_time + world.model.opt.timestep)
        last_time = world.data.time
        used_reverse |= np.any(manager.robot._commands < 0)
        if manager.arrived:
            if first_arrival is None:
                first_arrival = world.data.time
        if first_arrival is not None:
            reading = manager.readings()
            assert manager.arrived
            assert reading['distance'] <= manager.settings.position_tolerance
            assert reading['speed'] <= manager.settings.speed_tolerance
            assert abs(reading['heading_error']) <= manager.settings.heading_tolerance
            assert abs(reading['yaw_speed']) <= manager.settings.yaw_speed_tolerance
            if world.data.time >= first_arrival + 5:
                break
    assert first_arrival is not None, manager.readings()
    assert world.data.time >= first_arrival + 5
    assert not world.data.warning.number.any()
    assert np.isfinite(world.data.qpos).all() and np.isfinite(world.data.qvel).all()
    if velocity == (1., 0):
        assert used_reverse  # 목표를 통과하기 전에 역추력으로 제동해야 한다.


def test_arrival_requires_dwell_not_just_crossing_goal():
    manager = build_approach_demo(target=(0, 0))
    for _ in range(250):
        manager.step()
    assert manager.phase == Phase.SETTLE
    assert not manager.arrived
    for _ in range(300):
        manager.step()
    assert manager.arrived

    moving = build_approach_demo(target=(0, 0))
    moving.world.data.qvel[0] = 1.
    moving.step()
    assert not moving.arrived
    assert moving.forward < 0  # 목표 지점에 있어도 속도가 남으면 먼저 제동한다.


def test_turning_at_goal_does_not_translate_due_to_asymmetric_thrusters():
    manager = build_approach_demo(target=(0, 0), yaw=math.pi)
    for _ in range(6500):
        manager.step()
    assert manager.arrived
    # 새 선체의 치우친 COM·수력에 따른 cm 단위 이동을 허용한다.
    # 순수 회전의 추력 합력 0은 별도의 실제 힘 검사로 검증한다.
    assert manager.readings()['distance'] < .05
    assert abs(manager.readings()['heading_error']) < manager.settings.heading_tolerance


def test_world_reset_clears_arrival_state_and_restarts_control():
    manager = build_approach_demo(target=(0, 0))
    for _ in range(600):
        manager.step()
    assert manager.arrived
    mujoco.mj_resetData(manager.world.model, manager.world.data)
    manager.step()
    assert not manager.arrived
    assert manager.arrival_seconds == 0
    assert manager.world.data.time == pytest.approx(manager.world.model.opt.timestep)


def test_displacement_after_arrival_triggers_reapproach():
    manager = build_approach_demo(target=(0, 0))
    for _ in range(600):
        manager.step()
    assert manager.arrived
    manager.world.data.qpos[0] += 1
    for _ in range(11):
        manager.step()
    assert manager.phase == Phase.APPROACH
    assert manager.arrival_seconds == 0


def test_commands_follow_control_rate_and_physics_keeps_running():
    manager = build_approach_demo(target=(4, 0))
    manager.step()
    first = manager.robot._commands.copy()
    # 다음 제어 시각 전에는 물리 스텝만 진행하고 명령은 유지한다.
    for _ in range(9):
        manager.step()
        np.testing.assert_array_equal(manager.robot._commands, first)
    assert manager.world.data.time == pytest.approx(.02)
    manager.step()
    assert not np.array_equal(manager.robot._commands, first)


@pytest.mark.parametrize('target,heading', [((float('nan'), 0), 0), ((1,), 0), ((0, 0), float('inf'))])
def test_invalid_goal_is_rejected(target, heading):
    manager = build_approach_demo()
    with pytest.raises(ValueError, match='목표'):
        PhaseManager(manager.world, manager.robot, target, target_heading=heading)


@pytest.mark.parametrize('options', [dict(max_speed=0), dict(speed_gain=float('nan')),
                                     dict(capture_radius=.4), dict(heading_tolerance=4)])
def test_invalid_settings_are_rejected(options):
    with pytest.raises(ValueError):
        ApproachSettings(**options)


@pytest.mark.parametrize('force,torque', [(0., 6.), (100., 6.), (-100., -6.)])
def test_thrust_saturation_preserves_requested_yaw_torque(force, torque):
    from sim import config

    manager = build_approach_demo(target=(0, 0))
    manager._set_force(force, torque)
    left, right = manager.robot.thrusts()
    assert -config.Thruster.MAX_REVERSE <= left <= config.Thruster.MAX_FORWARD
    assert -config.Thruster.MAX_REVERSE <= right <= config.Thruster.MAX_FORWARD
    world = manager.world
    world.acc.reset()
    manager.robot.add_forces(world.model, world.data, world.acc)
    world.acc.apply(world.model, world.data)
    load = world.data.xfrc_applied[manager.robot.body_id]
    assert load[5] == pytest.approx(torque)
    if force == 0:
        assert load[0] == pytest.approx(0.)
