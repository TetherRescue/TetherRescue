"""로봇 단위 테스트 (담당: 동경).

물·world.py 없이 로봇만 MjSpec 에 올려 확인합니다.
"""
import mujoco
import numpy as np
import pytest

from sim import config, frames
from sim.config import Hull, Names, Sensors, Thruster
from sim.forces import ForceAccumulator
from sim.robot import Robot
from sim.world import build_world, advance_world as advance
from sim.frames import body_velocity_w


def _compile():
    spec = mujoco.MjSpec()
    Robot().build(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _id(model, obj, name):
    i = mujoco.mj_name2id(model, obj, name)
    assert i >= 0, f"{name} 없음"
    return i


def test_build_names_exist():
    model, _ = _compile()
    _id(model, mujoco.mjtObj.mjOBJ_BODY, Names.BOAT_BODY)
    j = _id(model, mujoco.mjtObj.mjOBJ_JOINT, Names.BOAT_FREE)
    assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    _id(model, mujoco.mjtObj.mjOBJ_GEOM, Names.HULL_GEOM)
    for site in (Names.THRUSTER_L_SITE, Names.THRUSTER_R_SITE, Names.TOW_POINT_SITE, Names.SONAR_SITE):
        _id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    _id(model, mujoco.mjtObj.mjOBJ_CAMERA, Names.CAMERA_BOW)


def test_build_mass_and_inertia_match_config():
    # 물은 배 무게를 MuJoCo 모델에서 읽음 → 모델 값이 config 와 다르면 흘수 정답값과 조용히 어긋남
    model, data = _compile()
    b = _id(model, mujoco.mjtObj.mjOBJ_BODY, Names.BOAT_BODY)
    assert model.body_mass[b] == Hull.MASS
    np.testing.assert_allclose(model.body_ipos[b], Hull.COM)
    np.testing.assert_allclose(model.body_inertia[b], Hull.INERTIA)
    np.testing.assert_allclose(model.body_iquat[b], (1, 0, 0, 0))   # 관성 축 = 몸체 축 (축이 돌아가 있지 않음)


def test_build_hull_size_matches_config():
    model, _ = _compile()
    g = _id(model, mujoco.mjtObj.mjOBJ_GEOM, Names.HULL_GEOM)
    assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(model.geom_size[g], (Hull.LENGTH / 2, Hull.BEAM / 2, Hull.HEIGHT / 2))
    np.testing.assert_allclose(model.geom_pos[g], 0)   # 상자 중심 = body 원점 (Frames.BODY_ORIGIN)


def test_build_site_positions_match_config():
    model, _ = _compile()
    for name, pos in ((Names.THRUSTER_L_SITE, Thruster.POS_L_B),
                      (Names.THRUSTER_R_SITE, Thruster.POS_R_B),
                      (Names.TOW_POINT_SITE, Hull.TOW_POINT_B),
                      (Names.SONAR_SITE, Sensors.SONAR_POS_B)):
        np.testing.assert_allclose(model.site_pos[_id(model, mujoco.mjtObj.mjOBJ_SITE, name)], pos)


def test_camera_looks_forward_from_bow():
    # 카메라는 선수 위(Sensors.CAMERA_POS_B)에서 배 앞(+x_b)을 봄. MuJoCo 카메라는 자기 −z 로 봄
    model, data = _compile()
    c = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, Names.CAMERA_BOW)
    np.testing.assert_allclose(model.cam_pos[c], Sensors.CAMERA_POS_B)
    view = -data.cam_xmat[c].reshape(3, 3)[:, 2]    # 보는 방향 (시작 자세는 몸체 = 월드 방향)
    up = data.cam_xmat[c].reshape(3, 3)[:, 1]       # 화면 위
    pitch = Sensors.CAMERA_PITCH_DOWN or 0.0
    np.testing.assert_allclose(view, (np.cos(pitch), 0, -np.sin(pitch)), atol=1e-9)
    assert up[2] > 0.9   # 화면이 뒤집히지 않음
    assert Sensors.CAMERA_POS_B[2] > Hull.HEIGHT / 2   # 선체 윗면 위 (물에 안 잠김)


def test_build_starts_at_draft():
    # 시작 자세: 선체 바닥이 수면 아래 정확히 흘수만큼
    model, data = _compile()
    b = _id(model, mujoco.mjtObj.mjOBJ_BODY, Names.BOAT_BODY)
    bottom_z = data.xpos[b][2] - Hull.HEIGHT / 2
    np.testing.assert_allclose(bottom_z, config.WaterParams.SURFACE_Z - config.Expect.DRAFT, atol=1e-12)


def test_bind_caches_ids():
    model, _ = _compile()
    robot = Robot()
    robot.bind(model)
    assert robot.body_id == _id(model, mujoco.mjtObj.mjOBJ_BODY, Names.BOAT_BODY)
    assert robot.thr_l_site_id == _id(model, mujoco.mjtObj.mjOBJ_SITE, Names.THRUSTER_L_SITE)
    assert robot.thr_r_site_id == _id(model, mujoco.mjtObj.mjOBJ_SITE, Names.THRUSTER_R_SITE)
    assert robot.tow_site_id == _id(model, mujoco.mjtObj.mjOBJ_SITE, Names.TOW_POINT_SITE)
    assert robot.qpos_adr == 0 and robot.qvel_adr == 0   # 로봇만 있는 모델에선 자유 관절이 맨 앞


def test_bind_fails_loudly_without_robot():
    # build 없이 만든 모델 → 이름이 없으니 조용히 −1 이 아니라 바로 멈춰야 함
    model = mujoco.MjSpec().compile()
    with pytest.raises(AssertionError, match=Names.BOAT_BODY):
        Robot().bind(model)


def test_bind_fails_when_mass_changed():
    spec = mujoco.MjSpec()
    Robot().build(spec)
    spec.body(Names.BOAT_BODY).mass = Hull.MASS + 10
    with pytest.raises(AssertionError, match="무게"):
        Robot().bind(spec.compile())


# ── add_forces ──────────────────────────────────────────────
def _robot_world(yaw_deg=0.0):
    """물 없이 로봇만, 중력 끔 (물이 없으면 배가 떨어지니까 수평 움직임만 봄)."""
    spec = mujoco.MjSpec()
    spec.option.gravity = (0, 0, 0)
    spec.option.timestep = config.Sim.TIMESTEP
    robot = Robot()
    robot.build(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    robot.bind(model)
    half = np.radians(yaw_deg) / 2
    data.qpos[robot.qpos_adr + 3:robot.qpos_adr + 7] = (np.cos(half), 0, 0, np.sin(half))
    mujoco.mj_forward(model, data)
    return model, data, robot


def _run(model, data, robot, seconds):
    acc = ForceAccumulator()
    for _ in range(round(seconds / model.opt.timestep)):
        acc.reset()
        robot.add_forces(model, data, acc)
        acc.apply(model, data)
        mujoco.mj_step(model, data)


def test_thrusts_mapping():
    robot = Robot()
    robot.set_command(1.0)
    assert robot.thrusts() == (Thruster.MAX_FORWARD, Thruster.MAX_FORWARD)
    robot.set_command(-1.0)
    assert robot.thrusts() == (-Thruster.MAX_REVERSE, -Thruster.MAX_REVERSE)   # 후진은 MAX_REVERSE
    robot.set_command(0.0, 0.5)
    assert robot.thrusts() == (-0.5 * Thruster.MAX_REVERSE, 0.5 * Thruster.MAX_FORWARD)   # 왼쪽으로 돎 (후진이 약해 앞으로 조금 감)
    robot.set_command(1.0, 1.0)
    assert robot.thrusts() == (0.0, Thruster.MAX_FORWARD)   # 2 → 1 로 자름
    robot.set_command(5.0, -5.0)
    assert (robot.throttle, robot.steer) == Thruster.CMD_RANGE[::-1]   # 범위 밖 명령도 자름


def test_forward_at_yaw90_accelerates_world_y():
    # 1주차 완료 기준: 90° 돌려 놓고 "앞으로" → 월드 +y 로 가속
    model, data, robot = _robot_world(yaw_deg=90)
    robot.set_command(1.0)
    _run(model, data, robot, 0.5)
    v = data.qvel[robot.qvel_adr:robot.qvel_adr + 3]   # 자유 관절 선속도는 월드 좌표
    expected = 2 * Thruster.MAX_FORWARD / Hull.MASS * 0.5   # 물 없음 → v = F/m · t = 1.2 m/s
    assert v[1] == pytest.approx(expected, rel=0.05)
    assert abs(v[0]) < 0.01 * expected   # 옆(월드 x)으로는 안 감


def test_steer_positive_turns_left():
    # 조향 + → 오른쪽 추진기가 셈 → 왼쪽(반시계, 위에서 볼 때)으로 돎 = 요 각속도 +
    model, data, robot = _robot_world()
    robot.set_command(0.0, 0.5)
    _run(model, data, robot, 0.5)
    _, ang_w = frames.body_velocity_w(model, data, robot.body_id)
    assert ang_w[2] > 0.1


# ── 노즐이 물 밖이면 추력 0 (Robot(surface_height=...)) ─────────
def test_nozzles_submerged_in_calm_water():
    # 잔잔한 물에 흘수만큼 띄운 시작 자세 → 바닥의 노즐 둘 다 물속
    from sim import water
    model, data, robot = _robot_world()
    robot.surface_height = lambda x, y, t: water.wave_height(x, y, t, config.WaterParams.WAVES_CALM)
    assert robot.nozzles_submerged(data) == (True, True)


def test_dry_nozzle_gives_no_thrust():
    # 왼쪽(+y) 노즐만 수면 위 → 오른쪽 추력만 걸림 → 전진 명령에도 왼쪽으로 돎
    model, data, robot = _robot_world()
    robot.surface_height = lambda x, y, t: np.where(np.asarray(y) > 0, -10.0, 10.0)
    assert robot.nozzles_submerged(data) == (False, True)
    robot.set_command(1.0)
    _run(model, data, robot, 0.5)
    _, ang_w = frames.body_velocity_w(model, data, robot.body_id)
    assert ang_w[2] > 0.1


def test_no_surface_height_keeps_week1_assumption():
    # surface_height 없으면 1주차 가정 그대로 "항상 물속" (WaterAPI.USE_IN_WEEK1 = False)
    model, data, robot = _robot_world()
    data.qpos[robot.qpos_adr + 2] = 100.0   # 물 위 높이 올려도
    mujoco.mj_forward(model, data)
    assert robot.nozzles_submerged(data) == (True, True)


# 기존 세계 조립·접근 제어 API와의 호환 검증.
def make_robot():
    robot = Robot()
    return build_world(parts=[robot], gravity=0), robot


def test_hull_mass_geometry_and_optional_tow_site():
    world, robot = make_robot()
    assert world.model.body_subtreemass[robot.body_id] == pytest.approx(config.Hull.MASS)
    np.testing.assert_allclose(world.model.geom_size[robot.geom_id],
                               np.array([config.Hull.LENGTH, config.Hull.BEAM, config.Hull.HEIGHT]) / 2)
    np.testing.assert_allclose(world.model.body_ipos[robot.body_id], config.Hull.COM)
    assert world.model.nsite == 4  # 추진기 2 + main의 합의된 선미 연결점 + 소나 자리


@pytest.mark.parametrize('command,force', [(1, 2 * config.Thruster.MAX_FORWARD),
                                            (-1, -2 * config.Thruster.MAX_REVERSE), (0, 0)])
def test_actual_forward_reverse_acceleration(command, force):
    world, robot = make_robot()
    robot.set_command(command)
    advance(world)
    assert body_velocity_w(world.model, world.data, robot.body_id)[0][0] / world.model.opt.timestep == pytest.approx(force / config.Hull.MASS)


def test_yaw_ninety_forward_accelerates_world_y():
    world, robot = make_robot()
    address = world.model.jnt_qposadr[robot.joint_id]
    world.data.qpos[address + 3:address + 7] = [2**-.5, 0, 0, 2**-.5]
    robot.set_command(1)
    advance(world)
    dof = world.model.jnt_dofadr[robot.joint_id]
    assert world.data.qvel[dof] == pytest.approx(0, abs=1e-12)
    assert body_velocity_w(world.model, world.data, robot.body_id)[0][1] / world.model.opt.timestep == pytest.approx(2 * config.Thruster.MAX_FORWARD / config.Hull.MASS)


def test_positive_steering_turns_left_and_mixing_saturates():
    world, robot = make_robot()
    robot.set_command(steering=1)
    advance(world)
    dof = world.model.jnt_dofadr[robot.joint_id]
    assert world.data.qvel[dof + 5] > 0
    world, robot = make_robot()
    robot.set_command(forward=1, steering=1)
    advance(world)
    assert body_velocity_w(world.model, world.data, robot.body_id)[0][0] / world.model.opt.timestep == pytest.approx(config.Thruster.MAX_FORWARD / config.Hull.MASS)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_commands_rejected(value):
    with pytest.raises(ValueError):
        Robot().set_command(value)


@pytest.mark.parametrize('forward,steering', [(.3, -.2), (2., -2.), (-1., .4)])
def test_navigation_command_aliases_match_main_throttle_contract(forward, steering):
    original, compatible = Robot(), Robot()
    original.set_command(forward, steering)
    compatible.set_command(forward=forward, steering=steering)
    assert compatible.thrusts() == original.thrusts()
    np.testing.assert_array_equal(compatible._commands, original._commands)


def test_explicit_trial_placement_and_tow_site_do_not_change_main_default():
    position = (3., 2., .7)
    point = (-Hull.LENGTH / 2, 0., .05)
    robot = Robot(position=position, tow_point_b=point)
    world = build_world(parts=[robot], gravity=0)
    np.testing.assert_allclose(world.data.xpos[robot.body_id], position)
    np.testing.assert_allclose(world.model.site_pos[robot.tow_site_id], point)
