"""시뮬레이션 한계·수치 오류·전복 감시와 실패 후 물리 동결을 검사한다."""
from dataclasses import replace
import math

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.rescue_mission import MissionStage
from sim.robot import Robot
from sim.view_rescue import build_rescue_mission, apply_failure_demo, run_headless, decorate_rescue
from sim.watchdog import FailureCode, MissionWatchdog, WatchdogSettings
from sim.world import build_world


@pytest.mark.parametrize('stage,field', [('APPROACH', 'approach_timeout'),
                                         ('ATTACH', 'attach_timeout'), ('TOW', 'tow_timeout')])
def test_each_stage_has_its_own_deadline(stage, field):
    robot = Robot()
    world = build_world(parts=[robot])
    watchdog = MissionWatchdog(replace(WatchdogSettings(), **{field: .008}))
    assert watchdog.check(world.model, world.data, robot.body_id, stage) is None
    world.data.time = .006
    assert watchdog.check(world.model, world.data, robot.body_id, stage) is None
    world.data.time = .008
    failure = watchdog.check(world.model, world.data, robot.body_id, stage)
    assert failure.code == FailureCode.STAGE_TIMEOUT and failure.stage == stage
    assert failure.time == pytest.approx(.008)


def test_changing_stages_does_not_restart_overall_deadline():
    robot = Robot()
    world = build_world(parts=[robot])
    watchdog = MissionWatchdog(WatchdogSettings(mission_timeout=1.))
    assert watchdog.check(world.model, world.data, robot.body_id, 'APPROACH') is None
    world.data.time = .8
    assert watchdog.check(world.model, world.data, robot.body_id, 'ATTACH') is None
    world.data.time = 1.
    failure = watchdog.check(world.model, world.data, robot.body_id, 'TOW')
    assert failure.code == FailureCode.MISSION_TIMEOUT


@pytest.mark.parametrize('roll,pitch,yaw,expected', [
    (61, 0, 0, FailureCode.CAPSIZE), (0, 61, 0, FailureCode.CAPSIZE),
    (180, 0, 0, FailureCode.CAPSIZE), (0, 0, 180, None), (20, 20, 90, None),
])
def test_orientation_is_checked_from_qpos_before_forward(roll, pitch, yaw, expected):
    mission = build_rescue_mission()
    angles = np.deg2rad([roll, pitch, yaw])
    quaternion = np.empty(4)
    mujoco.mju_euler2Quat(quaternion, angles, 'XYZ')
    address = mission.world.model.jnt_qposadr[mission.robot.joint_id]
    mission.world.data.qpos[address + 3:address + 7] = quaternion
    # mj_forward를 호출하지 않아 파생 행렬은 아직 초기 자세다.
    failure = mission.watchdog.check(mission.world.model, mission.world.data, mission.robot.body_id, 'APPROACH')
    assert (None if failure is None else failure.code) == expected


@pytest.mark.parametrize('kind,code', [('timeout', FailureCode.STAGE_TIMEOUT),
                                      ('capsize', FailureCode.CAPSIZE),
                                      ('nonfinite', FailureCode.INVALID_STATE),
                                      ('warning', FailureCode.ENGINE_WARNING)])
def test_injected_failure_latches_zero_commands_and_freezes_physics(kind, code):
    mission = build_rescue_mission()
    apply_failure_demo(mission, kind)
    mission.robot.set_command(1, .2)
    for _ in range(600):
        mission.step()
        if mission.failed:
            break
    assert mission.failed and mission.stage == MissionStage.FAILED
    assert mission.failure.code == code and mission.failure.stage == 'APPROACH'
    assert not mission.arrived
    np.testing.assert_array_equal(mission.robot._commands, 0)
    np.testing.assert_array_equal(mission.world.data.xfrc_applied, 0)
    np.testing.assert_array_equal(mission.world.data.qfrc_applied, 0)
    frozen_time = mission.world.data.time
    frozen_qpos = mission.world.data.qpos.copy()
    frozen_qvel = mission.world.data.qvel.copy()
    original_failure = mission.failure
    for _ in range(20):
        assert mission.step() is False
    assert mission.world.data.time == frozen_time
    np.testing.assert_array_equal(mission.world.data.qpos, frozen_qpos)
    np.testing.assert_array_equal(mission.world.data.qvel, frozen_qvel)
    assert mission.failure is original_failure
    assert np.isfinite(mission.display_data.qpos).all()
    assert np.isfinite(mission.display_data.xmat).all()
    state = mission.readings()
    assert state['phase'] == 'FAILED' and state['failure_code'] == code.value
    scene = mujoco.MjvScene(mission.world.model, maxgeom=100)
    decorate_rescue(mission, scene)
    assert all(np.isfinite(geom.pos).all() for geom in scene.geoms[:scene.ngeom])
    if kind == 'nonfinite':
        # MuJoCo가 qpos를 자동 복원하기 전에 중단해야 한다.
        assert np.isnan(mission.world.data.qpos[0])
        assert mission.world.data.warning.number.sum() == 0


@pytest.mark.parametrize('field', ['qvel', 'qacc', 'xfrc_applied', 'qfrc_applied'])
def test_nonfinite_state_arrays_are_rejected(field):
    mission = build_rescue_mission()
    values = getattr(mission.world.data, field)
    values.flat[0] = float('inf')
    mission.step()
    assert mission.failed and mission.failure.code == FailureCode.INVALID_STATE
    assert mission.world.data.time == 0


def test_zero_quaternion_is_rejected_before_mujoco_normalizes_it():
    mission = build_rescue_mission()
    mission.world.data.qpos[3:7] = 0
    mission.step()
    assert mission.failure.code == FailureCode.INVALID_STATE
    assert mission.world.data.time == 0


def test_post_step_warning_stops_before_any_further_physics(monkeypatch):
    mission = build_rescue_mission()
    original_step = mission.controller.step

    def warning_step():
        original_step()
        mission.world.data.warning[mujoco.mjtWarning.mjWARN_BADQVEL].number += 1

    monkeypatch.setattr(mission.controller, 'step', warning_step)
    assert mission.step() is False
    assert mission.failure.code == FailureCode.ENGINE_WARNING
    assert mission.world.data.time == pytest.approx(mission.world.model.opt.timestep)
    for _ in range(10):
        mission.step()
    assert mission.world.data.time == pytest.approx(mission.world.model.opt.timestep)


def test_existing_clearance_error_uses_the_same_failed_state():
    mission = build_rescue_mission()
    human = mission.world.model.body(config.Names.HUMAN_BODY).id
    mission.world.data.qpos[:2] = mission.world.data.xpos[human, :2]
    assert mission.step() is False
    assert mission.failure.code == FailureCode.CONTROL_ERROR
    assert mission.world.data.time == 0


def test_failure_requires_explicit_reset_and_restores_detached_rope():
    mission = build_rescue_mission()
    mission.rope.attach(mission.world.model)
    apply_failure_demo(mission, 'nonfinite')
    mission.step()
    assert mission.failed
    with pytest.raises(ValueError, match='mj_resetData'):
        mission.reset()
    mujoco.mj_resetData(mission.world.model, mission.world.data)
    assert mission.step() is False  # 실패 후에는 월드 리셋만으로 재출발하지 않는다.
    mission.reset()
    assert not mission.failed and mission.failure is None
    assert mission.stage == MissionStage.APPROACH and not mission.rope.attached
    assert mission.step() is True


def test_headless_reports_a_nonzero_failure_instead_of_hanging():
    mission = build_rescue_mission()
    with pytest.raises(RuntimeError, match='MISSION_TIMEOUT'):
        run_headless(mission, .01)
    assert mission.failed
    assert mission.world.data.time == pytest.approx(.01)


def test_completed_mission_keeps_monitoring_physics_without_success_timeout():
    mission = build_rescue_mission()
    watchdog = MissionWatchdog(WatchdogSettings(mission_timeout=1.))
    mission.world.data.time = 10.
    assert watchdog.check(mission.world.model, mission.world.data, mission.robot.body_id, 'COMPLETE') is None
    apply_failure_demo(mission, 'capsize')
    failure = watchdog.check(mission.world.model, mission.world.data, mission.robot.body_id, 'COMPLETE')
    assert failure.code == FailureCode.CAPSIZE


@pytest.mark.parametrize('options', [dict(approach_timeout=0), dict(attach_timeout=-1),
                                     dict(tow_timeout=float('nan')), dict(roll_limit=2.)])
def test_invalid_watchdog_settings_are_rejected(options):
    with pytest.raises(ValueError):
        WatchdogSettings(**options)


def test_nonfinite_time_is_reported_with_last_finite_failure_timestamp():
    mission = build_rescue_mission()
    mission.world.data.time = float('nan')
    assert mission.step() is False
    assert mission.failure.code == FailureCode.INVALID_STATE
    assert math.isfinite(mission.failure.time)
    assert math.isfinite(mission.display_data.time)
    assert np.isnan(mission.world.data.time)


@pytest.mark.parametrize('kind,code', [('timeout', 'STAGE_TIMEOUT'), ('capsize', 'CAPSIZE'),
                                      ('nonfinite', 'INVALID_STATE'), ('warning', 'ENGINE_WARNING')])
def test_headless_cli_reports_failure_and_exits_nonzero(kind, code):
    import subprocess
    import sys

    result = subprocess.run([sys.executable, '-m', 'sim.view_rescue', '--headless', '--failure-demo', kind],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert code in result.stderr
    assert 'Traceback' not in result.stderr
