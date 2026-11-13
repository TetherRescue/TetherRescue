"""부유하는 구명환에 선미를 맞추고 정지하는 실제 엔진 시험."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.rescue_approach import RescueApproachSettings, RescueStage
from sim.view_rescue import build_rescue_demo
from sim.view_towing import advance


@pytest.mark.parametrize('start,yaw,heading', [
    ((6, 3), 0., 0.),
    ((-6, 0), 0., 0.),
    ((5, 6), math.pi, math.pi / 2),
])
def test_reaches_connection_position_without_intersecting_person(start, yaw, heading):
    manager = build_rescue_demo(start=start, yaw=yaw, tow_heading=heading)
    world = manager.world
    assert world.model.ntendon == 0
    stages = set()
    ready_since = None
    for index in range(90000):
        manager.step()
        stages.add(manager.stage)
        if index % 20 == 0:
            # 제어기의 판정과 독립적으로 실제 형상 사이 거리를 측정한다.
            assert manager.clearance() > .1
            assert not world.data.warning.number.any()
            assert np.isfinite(world.data.qpos).all() and np.isfinite(world.data.qvel).all()
        if manager.arrived and ready_since is None:
            ready_since = float(world.data.time)
        if ready_since is not None:
            assert manager.arrived
            state = manager.readings()
            assert abs(state['endpoint_gap'] - manager.settings.connection_gap) <= manager.settings.gap_tolerance
            assert state['relative_speed'] <= manager.settings.relative_speed_tolerance
            assert abs(state['heading_error']) <= manager.navigation.heading_tolerance
            if world.data.time >= ready_since + 5:
                break
    assert ready_since is not None, manager.readings()
    assert world.data.time >= ready_since + 5
    assert stages == set(RescueStage)
    assert manager.minimum_observed_clearance > manager.settings.minimum_clearance
    # 실제 선미 연결점이 구명환에서 지정한 견인 방향에 놓이는지 확인한다.
    offset = world.data.site_xpos[manager.stern_site, :2] - world.data.site_xpos[manager.ring_site, :2]
    assert offset @ manager.direction > .5
    assert np.linalg.norm(offset - (offset @ manager.direction) * manager.direction) < .15


def test_person_motion_is_unaffected_by_disconnected_robot_approach():
    manager = build_rescue_demo()
    baseline = build_rescue_demo()
    world = manager.world
    joint = world.model.joint(config.Names.HUMAN_FREE).id
    address = world.model.jnt_qposadr[joint]
    for _ in range(500):
        manager.step()
        advance(baseline.world)
    np.testing.assert_allclose(world.data.qpos[address:address + 7],
                               baseline.world.data.qpos[address:address + 7], atol=1e-12)
    assert world.model.ntendon == 0


def test_too_close_initial_position_is_rejected():
    with pytest.raises(ValueError, match='떨어뜨려'):
        build_rescue_demo(start=(1, 0))


def test_clearance_guard_stops_commands_before_another_physics_step():
    manager = build_rescue_demo()
    world = manager.world
    human = world.model.body(config.Names.HUMAN_BODY).id
    world.data.qpos[:2] = world.data.xpos[human, :2]
    manager.robot.set_command(1, 0)
    with pytest.raises(RuntimeError, match='간격'):
        manager.step()
    np.testing.assert_array_equal(manager.robot._commands, 0)
    assert world.data.time == 0


def test_world_reset_restarts_route_and_clears_ready_dwell():
    manager = build_rescue_demo(yaw=math.pi / 2)
    for _ in range(20):
        manager.step()
    manager.stage = RescueStage.READY
    manager._ready_since = float(manager.world.data.time)
    mujoco.mj_resetData(manager.world.model, manager.world.data)
    np.testing.assert_allclose(manager.world.data.qpos[3:7], [2**-.5, 0, 0, 2**-.5])
    manager.step()
    assert manager.stage == RescueStage.TRANSIT
    assert manager._ready_since is None
    assert manager.world.data.time == pytest.approx(manager.world.model.opt.timestep)


@pytest.mark.parametrize('options', [dict(connection_gap=-1), dict(docking_speed=float('nan')),
                                     dict(connection_gap=.1), dict(staging_gap=.3)])
def test_invalid_settings_are_rejected(options):
    with pytest.raises(ValueError):
        RescueApproachSettings(**options)
