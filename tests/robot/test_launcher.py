"""구명환 발사대 (담당: 동경). 고정·겨누기·발사 반동·윈치 자유 풀림."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.frames import body_velocity_w
from sim.robot.launcher import FLYING, LOADED, PUSHING, ballistic_speed
from sim.robot.phase_manager import Phase
from sim.robot.robot import PAID_OUT
from sim.world.launch_mission import build_launch_mission
from sim.world.view_launch import COMPUTE_PER_FRAME, Playback
from sim.world.world import advance_world


def _mount_error(mission):
    """발사대 자리와 실제 환 중심 사이 거리 [m]."""
    data, launcher = mission.world.data, mission.launcher
    boat = launcher.boat_body
    mount = data.xpos[boat] + data.xmat[boat].reshape(3, 3) @ np.asarray(config.Launcher.MOUNT_B)
    return float(np.linalg.norm(data.xpos[launcher.ring_body] - mount))


def test_ballistic_speed_by_hand():
    # 같은 높이로 45° 면 R = v²/g → v = √(g·R). 2 m 면 √(9.81·2) ≈ 4.43 m/s
    assert ballistic_speed(2.0, 0.0, math.radians(45)) == pytest.approx(math.sqrt(config.Physics.GRAVITY * 2.0))
    # 닿을 수 없는 높이면 에러 (35° 로 1 m 앞, 1 m 위)
    with pytest.raises(ValueError):
        ballistic_speed(1.0, 1.0, math.radians(35))


def test_ring_starts_on_mount_and_rides_with_boat():
    mission = build_launch_mission()
    assert mission.launcher.state == LOADED
    assert _mount_error(mission) < 1e-6
    for _ in range(round(8.0 / mission.world.model.opt.timestep)):   # 접근하며 돌고 달리는 동안
        mission.step()
    assert mission.stage.value == 'APPROACH'
    assert _mount_error(mission) < .02
    # 실린 환은 간격 검사 대상이 아님 (배에 붙어 있어 0 이 됨)
    assert mission.minimum_observed_clearance > .5


def test_fire_reaches_aim_velocity_with_recoil_and_frees_winch():
    mission = build_launch_mission()
    world, launcher, robot = mission.world, mission.launcher, mission.robot
    mujoco.mj_forward(world.model, world.data)
    hand = world.data.site_xpos[mission.grasp.hand_site].copy()
    velocity, distance, _ = launcher.aim(world.data, hand)
    boat_before, _ = body_velocity_w(world.model, world.data, robot.body_id)
    launcher.fire(world.model, world.data, velocity)
    robot.release_winch(distance + config.Launcher.ROPE_SLACK)
    assert robot.delivery_state == PAID_OUT
    assert robot.rope_length(world.model) == pytest.approx(distance + config.Launcher.ROPE_SLACK)
    assert launcher.state == PUSHING
    with pytest.raises(RuntimeError):
        launcher.fire(world.model, world.data, velocity)   # 두 번 쏠 수 없음
    for _ in range(round(config.Launcher.PUSH_SECONDS / world.model.opt.timestep) + 1):
        advance_world(world)
    assert launcher.state == FLYING
    ring_v, _ = body_velocity_w(world.model, world.data, launcher.ring_body)
    # 미는 동안의 중력은 힘에서 보상했으므로 미는 끝 속도 = 겨눈 속도 (물·로프 힘이 작아 2 %)
    np.testing.assert_allclose(ring_v, velocity, rtol=.02, atol=.05)
    # 반동: 배는 환과 반대쪽 수평 속도를 얻는다
    boat_after, _ = body_velocity_w(world.model, world.data, robot.body_id)
    assert (boat_after - boat_before)[:2] @ velocity[:2] < 0


def test_release_winch_never_reels_in():
    mission = build_launch_mission()
    with pytest.raises(ValueError):
        mission.robot.release_winch(config.Winch.STOW_LENGTH - .1)


def test_launch_mission_lands_at_hand_grabs_and_tows():
    # ★ 발사 구조 완료 기준: 발사 → 잡는 점이 손 근처에 떨어짐 → 사람이 잡음 → 견인 완료, 사람과 안 부딪힘
    mission = build_launch_mission()
    data = mission.world.data
    while data.time < 150 and not mission.arrived:
        assert mission.step() is not False, mission.readings()
    state = mission.readings()
    assert mission.arrived, state
    assert state['shot']['time'] < 40
    assert mission.landing['grip_to_hand'] < .15
    assert mission.grab_time - mission.landing['time'] < 5
    assert state['held'] and state['tow_progress'] >= mission.settings.minimum_tow_progress
    assert mission.minimum_observed_clearance > 1.0   # 배는 사람에게 다가가지 않음
    assert data.time - mission.grab_time < 35         # 견인 약 28 초 (속도 이득 60 일 때 46 초)


def test_tow_uses_its_own_gain_and_reapproaches_outside_tolerance():
    mission = build_launch_mission()
    model, data, controller = mission.world.model, mission.world.data, mission.controller
    mujoco.mj_forward(model, data)
    mission._start_tow()
    navigation = mission.approach.navigation
    assert controller.settings.speed_gain == mission.launch.tow_speed_gain > navigation.speed_gain
    assert controller.settings.max_force == navigation.max_force   # 힘 상한은 그대로
    # 허용 오차 밖에서 옆으로 벗어나 멈춘 배는 (옆으로 못 가므로) 정착에 갇히지 않고 다시 다가가야 한다
    side = data.xmat[mission.robot.body_id].reshape(3, 3)[:2, 1]
    for offset, phase in ((navigation.position_tolerance - .05, Phase.SETTLE),
                          (navigation.position_tolerance + .2, Phase.APPROACH)):
        controller.target_xy[:] = data.xipos[mission.robot.body_id, :2] + offset * side
        controller.phase = Phase.SETTLE
        controller._update_control(0.)
        assert controller.phase == phase


def test_reset_reloads_launcher():
    mission = build_launch_mission()
    data = mission.world.data
    while mission.launcher.state == LOADED and data.time < 60:
        mission.step()
    assert mission.launcher.state != LOADED
    mujoco.mj_resetData(mission.world.model, data)
    mission.reset()
    assert mission.launcher.state == LOADED and mission.stage.value == 'APPROACH'
    assert mission.robot.rope_length(mission.world.model) == pytest.approx(config.Winch.STOW_LENGTH)
    assert _mount_error(mission) < 1e-6


def test_playback_follows_real_time_not_frame_count():
    for frames in (60, 10):   # 화면이 느려도 (10장/초) 같은 시간만큼 진행
        now = [0.]
        steps = []
        playback = Playback(2., .005, clock=lambda: now[0])
        for _ in range(frames):
            now[0] += 1 / frames
            playback.advance(lambda: steps.append(1))
        assert len(steps) * .005 == pytest.approx(2., abs=.01)   # 1초 × 배속 2


def test_playback_drops_lag_when_compute_is_too_slow():
    now = [0.]
    steps = []

    def slow_step():
        now[0] += .01   # 5 ms 를 계산하는 데 10 ms
        steps.append(1)

    playback = Playback(1., .005, clock=lambda: now[0])
    now[0] += 10.       # 크게 밀려도
    playback.advance(slow_step)
    assert len(steps) == math.floor(COMPUTE_PER_FRAME / .01) + 1 and playback.budget == 0
    playback.advance(lambda: False)   # 실패한 step 은 바로 멈춤
    assert playback.budget == 0
