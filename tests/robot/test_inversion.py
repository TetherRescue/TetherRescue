"""상하 뒤바뀜 (담당: 동경). 전복 판정 대신 뒤집힘을 알아채고 계속 추진, 카메라는 다시 위로 (2026-11-19)."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.frames import body_velocity_w
from sim.robot.robot import Robot
from sim.water.water import Water
from sim.world.world import advance_world, build_world


def _world(roll_deg=0.0):
    """잔잔한 물 + 노즐 수면 판정을 켠 배. roll_deg 만큼 기울여 띄운다 (뒤집으면 원점이 거울 자리에 오도록)."""
    water = Water(waves=())
    robot = Robot(surface_height=water.wave_height)
    world = build_world(parts=[water, robot])
    adr = robot.qpos_adr
    half = math.radians(roll_deg) / 2
    world.data.qpos[adr + 3:adr + 7] = (math.cos(half), math.sin(half), 0, 0)
    if abs(roll_deg) > 90:   # 뒤집히면 바닥이 위로 → 원점 높이를 수면 기준으로 거울 자리에
        world.data.qpos[adr + 2] = 2 * config.WaterParams.SURFACE_Z - world.data.qpos[adr + 2]
    mujoco.mj_forward(world.model, world.data)
    return world, robot


def _run(world, seconds):
    for _ in range(round(seconds / world.model.opt.timestep)):
        advance_world(world)


def test_upright_boat_is_not_inverted_and_inverted_boat_is():
    world, robot = _world(0)
    _run(world, .1)
    assert not robot.inverted
    world, robot = _world(180)
    _run(world, .1)
    assert robot.inverted


def test_inversion_has_hysteresis_around_ninety_degrees():
    world, robot = _world(0)
    robot.inverted = True                      # 뒤집힌 판정 상태에서
    adr = robot.qpos_adr
    for roll, expected in ((95, True), (85, True), (70, False)):   # 78° 보다 바로 서야 해제
        half = math.radians(roll) / 2
        world.data.qpos[adr + 3:adr + 7] = (math.cos(half), math.sin(half), 0, 0)
        mujoco.mj_forward(world.model, world.data)
        robot.update_inversion(world.model, world.data)
        assert robot.inverted == expected, roll


@pytest.mark.parametrize('roll', [0, 180])
def test_forward_and_left_turn_work_both_ways_up(roll):
    # 뒤집혀도 노즐이 물속이라 추력이 나오고, 전진은 세상 앞(+x), 조향 + 는 세상 왼쪽(요 각속도 +)
    world, robot = _world(roll)
    robot.set_command(forward=1.)
    _run(world, 2.)
    velocity, _ = body_velocity_w(world.model, world.data, robot.body_id)
    assert velocity[0] > .5 and robot.nozzles_submerged(world.data) == (True, True)

    world, robot = _world(roll)
    robot.set_command(steering=1.)
    _run(world, 1.)
    _, omega = body_velocity_w(world.model, world.data, robot.body_id)
    assert omega[2] > .1


def _total_com_b(world, robot):
    """배 + 카메라 팔 전체 무게중심 [몸체 좌표]."""
    data, b = world.data, robot.body_id
    return data.xmat[b].reshape(3, 3).T @ (data.subtree_com[b] - data.xpos[b])


def test_camera_turns_back_on_top_when_inverted():
    world, robot = _world(180)
    _run(world, config.Inversion.CAMERA_FLIP_SECONDS + .5)
    assert abs(robot.camera_roll(world.data)) == pytest.approx(math.pi, abs=.05)
    cam = robot.camera_id
    # 카메라는 배 중심보다 위, 화면 위쪽(카메라 +y)이 세상 위쪽
    assert world.data.cam_xpos[cam, 2] > world.data.xpos[robot.body_id, 2]
    assert world.data.cam_xmat[cam].reshape(3, 3)[2, 1] > .9
    # 카메라 무게도 팔과 같이 돌아 전체 무게중심이 몸체 −z 쪽으로 0.6 kg × 0.34 m ÷ 19.6 kg ≈ 1.04 cm 움직인다
    mass, (_, _, cam_z) = config.Hull.PARTS["camera"]
    shift = _total_com_b(world, robot) - np.asarray(config.Hull.COM)
    assert shift[2] == pytest.approx(-2 * cam_z * mass / config.Hull.MASS, abs=1e-3)


def test_camera_stays_up_while_upright_and_returns_after_reset():
    world, robot = _world(0)
    _run(world, 2.)
    assert abs(robot.camera_roll(world.data)) < .02
    world, robot = _world(180)
    _run(world, 2.)
    mujoco.mj_resetData(world.model, world.data)
    robot.reset_inversion(world.model)
    mujoco.mj_forward(world.model, world.data)
    assert robot.camera_roll(world.data) == 0 and not robot.inverted
    np.testing.assert_allclose(_total_com_b(world, robot), config.Hull.COM, atol=1e-12)


def test_camera_arm_motor_keeps_total_momentum_in_zero_gravity():
    # 팔을 돌리는 힘은 배와 팔 사이 안쪽 힘 → 무중력·물 없음이면 전체 운동량이 0 그대로
    robot = Robot()
    world = build_world(parts=[robot], gravity=0)
    # 빠른 회전의 적분 오차를 줄이려고 RK4 (tests/test_integration.py 의 운동량 시험과 같은 이유)
    world.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    adr = robot.qpos_adr
    world.data.qpos[adr + 3:adr + 7] = (0, 1, 0, 0)   # 뒤집힌 자세 → 팔이 반 바퀴 돈다
    mujoco.mj_forward(world.model, world.data)
    _run(world, 1.5)
    mujoco.mj_subtreeVel(world.model, world.data)
    momentum = world.model.body_subtreemass[0] * world.data.subtree_linvel[0]
    assert abs(robot.camera_roll(world.data)) > 2.5
    np.testing.assert_allclose(momentum, 0, atol=1e-6)
