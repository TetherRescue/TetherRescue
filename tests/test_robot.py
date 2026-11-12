"""로봇 단위 테스트 (담당: 동경).

물·world.py 없이 로봇만 MjSpec 에 올려 확인합니다.
"""
import mujoco
import numpy as np
import pytest

from sim import config, frames
from sim.config import Hull, Names, Thruster
from sim.robot import Robot


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
    for site in (Names.THRUSTER_L_SITE, Names.THRUSTER_R_SITE, Names.TOW_POINT_SITE):
        _id(model, mujoco.mjtObj.mjOBJ_SITE, site)


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
                      (Names.TOW_POINT_SITE, Hull.TOW_POINT_B)):
        np.testing.assert_allclose(model.site_pos[_id(model, mujoco.mjtObj.mjOBJ_SITE, name)], pos)


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
class _TempAcc:
    """테스트용 임시 ForceAccumulator (가영 님 forces.py 가 생기면 그걸로 교체).

    add 로 모은 힘을 apply 에서 mj_applyFT 로 qfrc_applied 에 씀. 점에 건 힘의 돌림힘은 mj_applyFT 가 계산.
    """
    def __init__(self):
        self.items = []

    def reset(self):
        self.items = []

    def add(self, body_id, force_w, torque_w=None, point_w=None):
        self.items.append((body_id, np.asarray(force_w, float), torque_w, point_w))

    def apply(self, model, data):
        data.qfrc_applied[:] = 0
        for body_id, force_w, torque_w, point_w in self.items:
            torque = np.zeros(3) if torque_w is None else np.asarray(torque_w, float)
            point = data.xipos[body_id] if point_w is None else np.asarray(point_w, float)
            mujoco.mj_applyFT(model, data, force_w, torque, point, body_id, data.qfrc_applied)


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
    acc = _TempAcc()
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
