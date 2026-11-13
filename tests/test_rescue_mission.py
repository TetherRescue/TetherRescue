"""미연결 출발부터 실제 연결·견인·정지까지 검증한다."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim.phase_manager import Phase
from sim.rescue_mission import MissionStage, RescueMissionSettings
from sim.view_rescue import build_rescue_mission, build_rescue_demo, decorate_rescue


@pytest.mark.parametrize('heading,start', [(0., (6, 3)), (math.pi / 2, (5, 6))])
def test_continuous_approach_attach_tow_and_stop(heading, start):
    mission = build_rescue_mission(start=start, tow_heading=heading)
    world, rope = mission.world, mission.rope
    assert world.model.ntendon == 1
    assert not rope.attached
    phases = set()
    previous_time = 0.
    completed_at = None
    maximum_tension = 0.
    attachment_checks = 0
    for index in range(120000):
        previous_phase = mission.stage
        previous_qpos = world.data.qpos.copy()
        mission.step()
        phases.add(mission.stage)
        assert world.data.time == pytest.approx(previous_time + world.model.opt.timestep)
        previous_time = float(world.data.time)
        if mission.attachment_time is None:
            assert not rope.attached
            assert rope.tension(world.model, world.data) == 0
            assert world.model.tendon_stiffness[rope.tendon_id] == 0
        if previous_phase == MissionStage.ATTACH and mission.stage == MissionStage.TOW:
            attachment_checks += 1
            # 물리 한 스텝의 이동만 허용한다. 연결 시 재배치·리셋으로 점프하지 않는다.
            assert np.max(np.abs(world.data.qpos - previous_qpos)) < .005
            assert rope.attached
            assert world.data.ten_length[rope.tendon_id] < world.model.tendon_lengthspring[rope.tendon_id, 1]
            assert rope.tension(world.model, world.data) == 0
        maximum_tension = max(maximum_tension, rope.tension(world.model, world.data))
        if index % 25 == 0:
            assert mission.approach.clearance() > .1
            assert -.3 < world.data.xpos[mission.robot.body_id, 2] < .4
            assert -.6 < world.data.xpos[mission.human_body, 2] < .4
            assert not world.data.warning.number.any()
            assert np.isfinite(world.data.qpos).all() and np.isfinite(world.data.qvel).all()
        if mission.arrived and completed_at is None:
            completed_at = float(world.data.time)
        if completed_at is not None:
            assert mission.arrived
            state = mission.readings()
            assert state['tow_progress'] >= mission.settings.minimum_tow_progress
            assert state['human_speed'] <= mission.settings.human_speed_tolerance
            assert state['distance'] <= mission.controller.settings.position_tolerance
            assert state['tension'] <= mission.settings.residual_tension
            if world.data.time >= completed_at + 5:
                break
    assert completed_at is not None, mission.readings()
    assert world.data.time >= completed_at + 5
    assert phases == set(MissionStage) - {MissionStage.FAILED}
    assert attachment_checks == 1
    assert maximum_tension > 1


def test_stale_arrival_flag_does_not_allow_remote_attachment():
    mission = build_rescue_mission()
    mission.stage = MissionStage.ATTACH
    mission.controller.phase = Phase.ARRIVED
    mission.step()
    assert mission.stage == MissionStage.APPROACH
    assert not mission.rope.attached
    assert mission.attachment_time is None
    assert mission.world.data.time == pytest.approx(mission.world.model.opt.timestep)


def test_world_reset_restores_detached_model_and_mission_state():
    mission = build_rescue_mission()
    for _ in range(20):
        mission.step()
    mission.rope.attach(mission.world.model)
    mission.stage = MissionStage.TOW
    mission.attachment_time = float(mission.world.data.time)
    mujoco.mj_resetData(mission.world.model, mission.world.data)
    mission.step()
    assert mission.stage == MissionStage.APPROACH
    assert mission.attachment_time is None
    assert not mission.rope.attached
    assert mission.world.model.tendon_stiffness[mission.rope.tendon_id] == 0
    assert mission.readings()['tension'] == 0


def test_approach_only_factory_still_has_no_rope():
    manager = build_rescue_demo()
    assert manager.world.model.ntendon == 0


def test_disconnected_rope_is_not_drawn():
    mission = build_rescue_mission()
    scene = mujoco.MjvScene(mission.world.model, maxgeom=100)
    decorate_rescue(mission, scene)
    assert scene.ngeom == 1  # 목표 방향 화살표만 표시
    assert not mission.rope.attached


@pytest.mark.parametrize('options', [dict(tow_speed=0), dict(tow_distance=float('nan')),
                                     dict(minimum_tow_progress=-1)])
def test_invalid_mission_settings_are_rejected(options):
    with pytest.raises(ValueError):
        RescueMissionSettings(**options)


def test_tow_destination_must_allow_real_person_progress():
    with pytest.raises(ValueError, match='견인 거리'):
        build_rescue_mission(tow_distance=3.)
