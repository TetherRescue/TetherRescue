"""로봇 단위 테스트 (담당: 동경).

물·world.py 없이 로봇만 MjSpec 에 올려 확인합니다.
"""
import mujoco
import numpy as np
import pytest

from sim import config, frames
from sim.config import Hull, Names, Sensors, Thruster
from sim.forces import ForceAccumulator
from sim.robot.robot import Robot
from sim.world.world import build_world, advance_world as advance
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
    # 카메라는 회전 팔 몸체에 따로 있으므로 배 + 팔(subtree)을 합쳐 config 와 비교한다 (2026-11-19)
    model, data = _compile()
    b = _id(model, mujoco.mjtObj.mjOBJ_BODY, Names.BOAT_BODY)
    arm = _id(model, mujoco.mjtObj.mjOBJ_BODY, Names.CAMERA_ARM_BODY)
    assert model.body_subtreemass[b] == pytest.approx(Hull.MASS)
    assert model.body_mass[arm] == Hull.PARTS["camera"][0]
    rotation = data.xmat[b].reshape(3, 3)
    np.testing.assert_allclose(rotation.T @ (data.subtree_com[b] - data.xpos[b]), Hull.COM, atol=1e-12)
    np.testing.assert_allclose(model.body_iquat[b], (1, 0, 0, 0))   # 관성 축 = 몸체 축 (축이 돌아가 있지 않음)
    # 배 몸체 관성 + 카메라 점 질량(평행축) = Hull.INERTIA (카메라 상자 자체의 작은 관성 1e-4 만 차이)
    from sim.robot.robot import FRAME_COM, FRAME_MASS
    com = np.asarray(Hull.COM)
    total = np.array(model.body_inertia[b], dtype=float)
    for mass, point in ((FRAME_MASS, FRAME_COM), Hull.PARTS["camera"]):
        d = np.asarray(point) - com
        total += mass * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    np.testing.assert_allclose(total, Hull.INERTIA, atol=2e-4)


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


# ── 흡입구가 물 밖이면 추력 0 (Robot(surface_height=...)) ─────────
def test_intakes_submerged_in_calm_water():
    # 잔잔한 물에 흘수만큼 띄운 시작 자세 → 바닥 아래 흡입구 둘 다 물속
    from sim.water import water
    model, data, robot = _robot_world()
    robot.surface_height = lambda x, y, t: water.wave_height(x, y, t, config.WaterParams.WAVES_CALM)
    assert robot.intakes_submerged(data) == (True, True)


@pytest.mark.parametrize('inverted', [False, True])
def test_intakes_are_six_cm_outside_the_hull_on_the_wet_side(inverted):
    # 바로 섰으면 바닥 아래 6 cm, 뒤집히면 갑판 쪽 6 cm. 좌우는 각 다리 안 (손 계산: 0.10 + 0.06)
    model, data, robot = _robot_world()
    robot.inverted = inverted
    points_b = data.xmat[robot.body_id].reshape(3, 3).T @ (robot.intake_points_w(data) - data.xpos[robot.body_id]).T
    z = 0.16 if inverted else -0.16
    np.testing.assert_allclose(points_b.T, [(-0.30, 0.525, z), (-0.30, -0.525, z)], atol=1e-12)
    assert all(config.hull_contains(np.array([[x, y, 0.0]]))[0] for x, y, _ in points_b.T)


def test_thrust_continues_when_only_the_nozzle_is_out_of_water():
    # 수면이 노즐(바닥)과 흡입구(바닥 아래 6 cm) 사이: 노즐은 물 밖이어도 흡입구가 잠겨 추력이 그대로,
    # 힘은 흡입구가 아니라 노즐 자리에 걸린다
    model, data, robot = _robot_world()
    nozzle_z = data.site_xpos[robot.thr_l_site_id][2]
    robot.surface_height = lambda x, y, t: np.full(np.shape(x), nozzle_z - 0.03)
    assert robot.intakes_submerged(data) == (True, True)
    robot.set_command(1.0)
    acc = ForceAccumulator()
    calls = []
    original = acc.add
    acc.add = lambda body, force, **kw: (calls.append((body, np.array(force), kw.get('point_w'))), original(body, force, **kw))
    robot.add_forces(model, data, acc)
    thrust = [(f, p) for b, f, p in calls if b == robot.body_id and p is not None]
    assert len(thrust) == 2
    for (force, point), site in zip(thrust, robot.site_ids):
        np.testing.assert_allclose(np.linalg.norm(force), Thruster.MAX_FORWARD)
        np.testing.assert_allclose(point, data.site_xpos[site])
    # 수면이 흡입구보다 낮아지면 둘 다 끊긴다
    robot.surface_height = lambda x, y, t: np.full(np.shape(x), nozzle_z - 0.07)
    assert robot.intakes_submerged(data) == (False, False)


def test_dry_intake_gives_no_thrust():
    # 왼쪽(+y) 흡입구만 수면 위 → 오른쪽 추력만 걸림 → 전진 명령에도 왼쪽으로 돎
    model, data, robot = _robot_world()
    robot.surface_height = lambda x, y, t: np.where(np.asarray(y) > 0, -10.0, 10.0)
    assert robot.intakes_submerged(data) == (False, True)
    robot.set_command(1.0)
    _run(model, data, robot, 0.5)
    _, ang_w = frames.body_velocity_w(model, data, robot.body_id)
    assert ang_w[2] > 0.1


def test_no_surface_height_keeps_week1_assumption():
    # 수면 함수가 연결되지 않으면 "항상 물속"
    model, data, robot = _robot_world()
    data.qpos[robot.qpos_adr + 2] = 100.0   # 물 위 높이 올려도
    mujoco.mj_forward(model, data)
    assert robot.intakes_submerged(data) == (True, True)


def _com_velocity(world, robot):
    """배 + 카메라 팔 전체 무게중심의 선속도 [월드]. 힘 ÷ 전체 무게 = 이 속도의 변화율."""
    mujoco.mj_subtreeVel(world.model, world.data)
    return world.data.subtree_linvel[robot.body_id].copy()


# 기존 세계 조립·접근 제어 API와의 호환 검증.
def make_robot():
    robot = Robot()
    return build_world(parts=[robot], gravity=0), robot


def test_hull_mass_geometry_and_optional_tow_site():
    world, robot = make_robot()
    assert world.model.body_subtreemass[robot.body_id] == pytest.approx(config.Hull.MASS)
    np.testing.assert_allclose(world.model.geom_size[robot.geom_id],
                               np.array([config.Hull.LENGTH, config.Hull.BEAM, config.Hull.HEIGHT]) / 2)
    # 배 + 카메라 팔 전체 무게중심 = Hull.COM (카메라는 팔 몸체에 따로, 2026-11-19)
    data, b = world.data, robot.body_id
    np.testing.assert_allclose(data.xmat[b].reshape(3, 3).T @ (data.subtree_com[b] - data.xpos[b]), config.Hull.COM,
                               atol=1e-12)
    assert world.model.nsite == 4  # 추진기 2 + main의 합의된 선미 연결점 + 소나 자리


@pytest.mark.parametrize('command,force', [(1, 2 * config.Thruster.MAX_FORWARD),
                                            (-1, -2 * config.Thruster.MAX_REVERSE), (0, 0)])
def test_actual_forward_reverse_acceleration(command, force):
    world, robot = make_robot()
    robot.set_command(command)
    advance(world)
    assert _com_velocity(world, robot)[0] / world.model.opt.timestep == pytest.approx(force / config.Hull.MASS)


def test_yaw_ninety_forward_accelerates_world_y():
    world, robot = make_robot()
    address = world.model.jnt_qposadr[robot.joint_id]
    world.data.qpos[address + 3:address + 7] = [2**-.5, 0, 0, 2**-.5]
    robot.set_command(1)
    advance(world)
    dof = world.model.jnt_dofadr[robot.joint_id]
    assert world.data.qvel[dof] == pytest.approx(0, abs=1e-12)
    assert _com_velocity(world, robot)[1] / world.model.opt.timestep == pytest.approx(2 * config.Thruster.MAX_FORWARD / config.Hull.MASS)


def test_positive_steering_turns_left_and_mixing_saturates():
    world, robot = make_robot()
    robot.set_command(steering=1)
    advance(world)
    dof = world.model.jnt_dofadr[robot.joint_id]
    assert world.data.qvel[dof + 5] > 0
    world, robot = make_robot()
    robot.set_command(forward=1, steering=1)
    advance(world)
    # 한쪽만 밀어 첫 스텝부터 돌기 시작 → 암묵 적분의 회전 결합으로 1e-5 수준 차이 → 1e-4
    assert _com_velocity(world, robot)[0] / world.model.opt.timestep == pytest.approx(
        config.Thruster.MAX_FORWARD / config.Hull.MASS, rel=1e-4)


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


# ── 구명환 윈치 (짧게 끌고 가다가 줄을 풀어 건네줌) ─────────────────

def _winch_world():
    """물·중력 없이 배 + 독립 구명환 + 처음부터 연결된 로프. 환은 선미 뒤 0.5 m."""
    from sim.world.lifebuoy import Lifebuoy
    from sim.world.tether import Tether
    robot = Robot(winch=True)
    stern = np.asarray(Hull.TOW_POINT_B)
    ring = Lifebuoy(attached=False, position=stern - (0.5 + config.Lifebuoy.TOW_POINT_B[0], config.Lifebuoy.TOW_POINT_B[1], 0))
    rope = Tether(enabled=True, attached=True)
    world = build_world(parts=[robot, ring, rope], gravity=0)
    return world, robot, ring, rope


def test_winch_starts_stowed_short():
    world, robot, _, rope = _winch_world()
    assert robot.delivery_state == 'STOWED' and not robot.delivery_ready
    assert robot.rope_length(world.model) == pytest.approx(config.Winch.STOW_LENGTH)
    assert world.model.tendon_lengthspring[rope.tendon_id, 1] == pytest.approx(config.Winch.STOW_LENGTH)


def test_pay_out_lengthens_at_speed_then_signals_ready():
    world, robot, _, _ = _winch_world()
    model = world.model
    assert robot.pay_out() is True
    assert robot.pay_out() is False   # 두 번 시작하지 않음
    for _ in range(round(1.0 / model.opt.timestep)):
        advance(world)
        assert not robot.delivery_ready or robot.rope_length(model) == pytest.approx(config.Winch.PAYOUT_LENGTH)
    expected = config.Winch.STOW_LENGTH + config.Winch.PAYOUT_SPEED * 1.0
    assert robot.rope_length(model) == pytest.approx(expected, abs=2 * config.Winch.PAYOUT_SPEED * model.opt.timestep)
    seconds = (config.Winch.PAYOUT_LENGTH - config.Winch.STOW_LENGTH) / config.Winch.PAYOUT_SPEED
    for _ in range(round(seconds / model.opt.timestep)):
        advance(world)
    assert robot.delivery_state == 'PAID_OUT' and robot.delivery_ready
    assert robot.rope_length(model) == pytest.approx(config.Winch.PAYOUT_LENGTH)


def test_short_rope_pulls_ring_and_payout_releases_tension():
    """감긴 상태에서 배가 앞으로 가면 환이 끌려오고, 줄을 다 풀면 같은 간격에서 장력이 없다."""
    world, robot, ring, rope = _winch_world()
    model, data = world.model, world.data
    robot.set_command(1.0)
    for _ in range(round(1.5 / model.opt.timestep)):
        advance(world)
    assert rope.tension(model, data) > 0
    assert data.qvel[model.jnt_dofadr[model.body_jntadr[ring.body_id]]] > 0.05   # 환이 +x 로 끌려옴
    robot.set_command(0.0)
    data.qvel[:] = 0
    robot.pay_out()
    for _ in range(round(5.0 / model.opt.timestep)):
        advance(world)
    assert robot.delivery_ready
    mujoco.mj_forward(model, data)
    assert rope.tension(model, data) == 0.0   # 간격(약 1 m) < 푼 길이 3 m


def test_reset_winch_restores_stowed_length():
    world, robot, _, _ = _winch_world()
    robot.pay_out()
    for _ in range(round(5.0 / world.model.opt.timestep)):
        advance(world)
    robot.reset_winch(world.model)
    assert robot.delivery_state == 'STOWED'
    assert robot.rope_length(world.model) == pytest.approx(config.Winch.STOW_LENGTH)


def test_winch_requires_rope_and_flag():
    with pytest.raises(ValueError):
        Robot().pay_out()
    with pytest.raises(AssertionError):   # 로프(tendon) 없이 winch=True 면 bind 에서 멈춤
        build_world(parts=[Robot(winch=True)], gravity=0)


# ─────────────────────────────────────────────────────────────
# A형 선체 (2026-11-19)
# ─────────────────────────────────────────────────────────────
from sim.robot.robot import A_FRAME_GEOMS  # noqa: E402


def test_a_frame_collision_geoms_replace_box():
    # 감싸는 상자는 부력 기준일 뿐 부딪히지 않고, 다리·연결부·뱃머리가 실제로 부딪힌다
    if Hull.SHAPE != "a_frame":
        pytest.skip("A형 선체 시험")
    model, _ = _compile()
    box = _id(model, mujoco.mjtObj.mjOBJ_GEOM, Names.HULL_GEOM)
    assert (model.geom_contype[box], model.geom_conaffinity[box]) == (0, 0)
    assert model.geom_group[box] == config.Groups.VISUAL
    for name in A_FRAME_GEOMS:
        g = _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_group[g] == config.Groups.PHYSICS
        assert (model.geom_contype[g], model.geom_conaffinity[g]) == (config.Contacts.BOAT, config.Contacts.BOAT_AFFINITY)


def test_a_frame_collision_geoms_cover_hull_shape():
    # 충돌 형상(위에서 본 모양)과 물이 쓰는 모양(hull_contains)이 같아야 함: 격자 점마다 비교
    if Hull.SHAPE != "a_frame":
        pytest.skip("A형 선체 시험")
    model, data = _compile()
    xs = np.linspace(-Hull.LENGTH / 2 + .01, Hull.LENGTH / 2 - .01, 60)
    ys = np.linspace(-Hull.BEAM / 2 + .01, Hull.BEAM / 2 - .01, 50)
    points = np.array([(x, y, 0.) for x in xs for y in ys])
    geoms = [_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in A_FRAME_GEOMS]
    # 몸체 원점이 월드 원점·회전 없음이 되도록 옮긴 뒤 점-형상 거리로 안/밖 판정 (작은 구를 점으로 사용)
    data.qpos[:3], data.qpos[3:7] = 0, (1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    inside_geoms = []
    for p in points:
        local = False
        for g in geoms:
            size, pos = model.geom_size[g], data.geom_xpos[g]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                local |= bool(np.all(np.abs(p - pos) <= size))
            else:   # 뱃머리 사다리꼴: x 구간 안에서 반폭이 선형으로 줄어듦
                t = (p[0] - Hull.BOW_START_X) / (Hull.LENGTH / 2 - Hull.BOW_START_X)
                half = Hull.BEAM / 2 - t * (Hull.BEAM / 2 - Hull.BOW_TIP_HALF_BEAM)
                local |= bool(0 <= t <= 1 and abs(p[1]) <= half)
        inside_geoms.append(local)
    np.testing.assert_array_equal(np.array(inside_geoms), config.hull_contains(points))


def test_slot_fits_lifebuoy_and_tow_point_is_at_slot_front():
    # 홈 폭 > 구명환 바깥지름, 줄 묶는 점 = 홈 앞끝 가운데 (구명환이 홈 가운데로 끌려 나감)
    if Hull.SHAPE != "a_frame":
        pytest.skip("A형 선체 시험")
    outer = 2 * (config.Lifebuoy.CENTERLINE_RADIUS + config.Lifebuoy.TUBE_RADIUS)
    assert Hull.SLOT_WIDTH > outer
    slot_front = -Hull.LENGTH / 2 + Hull.SLOT_DEPTH
    assert Hull.TOW_POINT_B[:2] == pytest.approx((slot_front, 0.))
    assert not config.hull_contains(np.array([slot_front - .01, 0., 0.]))   # 바로 뒤는 비어 있음 (홈)
    assert config.hull_contains(np.array([slot_front + .01, 0., 0.]))       # 바로 앞은 연결부


def test_center_of_mass_sits_over_center_of_buoyancy():
    # 무게중심이 수면 넓이 중심(똑바로 뜬 배의 부력 중심) 위에 있어야 앞뒤로 기울지 않고 수평으로 뜬다
    assert Hull.COM[0] == pytest.approx(Hull.WATERPLANE_CENTER_X, abs=0.01)
    assert Hull.COM[1] == pytest.approx(0., abs=1e-12)


def test_camera_sits_on_bridge_center():
    # 카메라는 연결부 가운데 (뱃머리보다 덜 흔들림, 2026-11-19 실험)
    if Hull.SHAPE != "a_frame":
        pytest.skip("A형 선체 시험")
    x, y, z = Sensors.CAMERA_POS_B
    assert -Hull.LENGTH / 2 + Hull.SLOT_DEPTH < x < Hull.BOW_START_X and y == 0
    assert z > Hull.HEIGHT / 2
