"""물 단위 테스트 (담당: 미현). 1~3 은 MuJoCo 없이, 4(복셀 부력)는 상자 하나짜리 MuJoCo 모델로 확인합니다."""
import math

import mujoco
import numpy as np
import pytest

from sim import config, water
from sim.config import WaterParams


# ─────────────────────────────────────────────────────────────
# 1. REQUIRES 는 config 한 곳에만
# ─────────────────────────────────────────────────────────────
def test_requires_comes_from_config():
    # 같은 객체여야 함 → config 만 고치면 water.py 도 따라감
    assert water.Water.REQUIRES is WaterParams.REQUIRES


def test_requires_matches_timestep():
    # 노션 06 문제 5: 시뮬레이션 한 칸(Sim.TIMESTEP)이 물이 허용하는 최대값보다 크면 안 됨
    assert config.Sim.TIMESTEP <= WaterParams.REQUIRES["max_dt"]


# ─────────────────────────────────────────────────────────────
# 2. wave_height: 숫자·배열 모두 받기
# ─────────────────────────────────────────────────────────────
def test_calm_water_is_flat():
    x = np.linspace(-10, 10, 50)
    z = water.wave_height(x, np.zeros_like(x), t=3.0, waves=WaterParams.WAVES_CALM)
    assert np.all(z == WaterParams.SURFACE_Z)


def test_scalar_in_scalar_out():
    z = water.wave_height(1.0, 2.0, 0.5)
    assert isinstance(z, float)


def test_array_matches_scalar():
    # 배열로 한 번에 물어본 결과 = 점마다 하나씩 물어본 결과
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(-5, 5, 20), rng.uniform(-5, 5, 20)
    t = 1.7
    batch = water.wave_height(xs, ys, t)
    one_by_one = [water.wave_height(x, y, t) for x, y in zip(xs, ys)]
    np.testing.assert_allclose(batch, one_by_one)


def test_output_shape_follows_input():
    # 복셀 격자처럼 2차원 배열을 넣어도 같은 모양으로 나와야 함
    x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-0.5, 0.5, 4))
    assert water.wave_height(x, y, 0.0).shape == (4, 8)


def test_single_component_by_hand():
    # 성분 하나를 손으로 계산한 값과 비교
    amp, length, direction, phase = 0.1, 8.0, 0.0, 0.3
    k = 2 * math.pi / length
    omega = math.sqrt(config.Physics.GRAVITY * k)
    x, t = 1.5, 2.0
    expected = WaterParams.SURFACE_Z + amp * math.cos(k * x - omega * t + phase)
    assert water.wave_height(x, 0.0, t, waves=((amp, length, direction, phase),)) == pytest.approx(expected)


def test_wave_travels_in_its_direction():
    # 진행 방향 90°(+y) 파도는 x 를 바꿔도 높이가 같고, y 를 바꾸면 달라짐
    waves = ((0.1, 8.0, math.pi / 2, 0.0),)
    assert water.wave_height(0.0, 1.0, 0.0, waves) == pytest.approx(water.wave_height(3.0, 1.0, 0.0, waves))
    assert water.wave_height(0.0, 1.0, 0.0, waves) != pytest.approx(water.wave_height(0.0, 3.0, 0.0, waves))


def test_water_object_uses_its_waves():
    assert water.Water(waves=WaterParams.WAVES_CALM).wave_height(1.0, 1.0, 1.0) == WaterParams.SURFACE_Z


# ─────────────────────────────────────────────────────────────
# 3. 회전 감쇠: 기울였다 놓으면 흔들림이 멎는지
# ─────────────────────────────────────────────────────────────
MASS = config.Hull.MASS
INERTIA = config.Hull.INERTIA


def _free_rock(stiffness, inertia, damping, tilt0=math.radians(15), seconds=3.0, dt=config.Sim.TIMESTEP):
    """배를 tilt0 만큼 기울였다가 놓았을 때 각도 기록 (회전 1방향만, 직선 저항만).
    I·θ'' + c·θ' + k·θ = 0 을 Sim.TIMESTEP 간격으로 계산."""
    theta, omega = tilt0, 0.0
    history = []
    for _ in range(int(seconds / dt)):
        omega += (-damping * omega - stiffness * theta) / inertia * dt
        theta += omega * dt
        history.append(theta)
    return np.array(history)


def test_restoring_stiffness_by_hand():
    # 손 계산 값: 흘수 0.1016 m, KG(바닥→무게중심) 0.15 m
    #   roll  GM = 0.0508 + 1.0²/(12·0.1016) − 0.15 ≈ 0.721 m → k ≈ 1768
    #   pitch GM = 0.0508 + 2.4²/(12·0.1016) − 0.15 ≈ 4.625 m → k ≈ 11342
    k_roll, k_pitch = water.restoring_stiffness(MASS)
    assert k_roll == pytest.approx(1768, rel=0.01)
    assert k_pitch == pytest.approx(11342, rel=0.01)


def test_rot_drag_gives_target_damping_ratio():
    c = water.rot_drag_linear(MASS, INERTIA)
    for axis, (zeta, k, inertia) in enumerate(zip(WaterParams.ROT_DAMPING_RATIO, water.restoring_stiffness(MASS), INERTIA)):
        assert c[axis] / (2 * math.sqrt(k * inertia)) == pytest.approx(zeta)
    assert c[2] == WaterParams.ROT_DRAG_LINEAR[2]  # yaw 는 숫자 그대로


@pytest.mark.parametrize("axis, name", [(0, "roll"), (1, "pitch")])
def test_tilt_settles(axis, name):
    # 15° 기울였다 놓으면 3 초 뒤 마지막 0.5 초 동안 흔들림이 처음의 2% 미만이어야 함
    k = water.restoring_stiffness(MASS)[axis]
    c = water.rot_drag_linear(MASS, INERTIA)[axis]
    history = _free_rock(k, INERTIA[axis], c)
    tail = history[-int(0.5 / config.Sim.TIMESTEP):]
    assert np.max(np.abs(tail)) < 0.02 * math.radians(15), f"{name} 흔들림이 멎지 않음"


@pytest.mark.parametrize("axis", [0, 1])
def test_old_small_coefficient_keeps_rocking(axis):
    # 재발 방지: 예전 제안값 8.0 이면 3 초 뒤에도 처음의 절반 넘게 흔들림 (동경 님 지적 재현)
    k = water.restoring_stiffness(MASS)[axis]
    history = _free_rock(k, INERTIA[axis], damping=8.0)
    tail = history[-int(0.5 / config.Sim.TIMESTEP):]
    assert np.max(np.abs(tail)) > 0.5 * math.radians(15)


def test_unstable_hull_raises(monkeypatch):
    # 무게중심이 너무 높으면 (GM ≤ 0) 바로 에러
    monkeypatch.setattr(config.Hull, "COM", (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="뒤집힘"):
        water.restoring_stiffness(MASS)


# ─────────────────────────────────────────────────────────────
# 4. 복셀 부력 — MuJoCo 상자 하나로 확인
# ─────────────────────────────────────────────────────────────
# voxel_buoyancy 는 좌표 변환을 frames.py 로 하므로 MuJoCo data 가 필요하다.
# world.py(가영) 없이도 되도록, 동경 님 tests/test_frames.py 처럼 상자 하나짜리 모델을 직접 만든다.
CALM = WaterParams.WAVES_CALM
RHO_G = config.Physics.RHO_WATER * config.Physics.GRAVITY
HULL_VOLUME = config.Hull.LENGTH * config.Hull.BEAM * config.Hull.HEIGHT
# 흘수만큼 잠긴 위치: 원점(상자 중심)이 수면 위 (높이/2 − 흘수) 에 있어야 선체 바닥이 수면 아래 흘수 깊이에 온다
#   0.4/2 − 0.1016 = +0.0984 m
FLOAT_Z = config.Hull.HEIGHT / 2 - config.Expect.DRAFT


def _boat(z=FLOAT_Z, roll_deg=0.0, pitch_deg=0.0):
    """선체 상자 하나짜리 MuJoCo 모델을 만들어 원하는 높이·기울기에 놓는다.

    물리 시뮬레이션은 돌리지 않고(mj_forward 만) "이 자세일 때 부력이 얼마인가"만 본다.
    돌려주는 값: (model, data, body_id)
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=config.Names.BOAT_BODY)
    body.add_freejoint()
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(config.Hull.LENGTH / 2, config.Hull.BEAM / 2, config.Hull.HEIGHT / 2),  # MuJoCo box 는 절반 크기
    )
    body.explicitinertial = True
    body.mass = config.Hull.MASS
    body.ipos = config.Hull.COM
    body.inertia = config.Hull.INERTIA
    model = spec.compile()
    data = mujoco.MjData(model)

    # 자세 = 쿼터니언 (w, x, y, z). 롤은 x 축(앞), 피치는 y 축(왼쪽) 둘레 회전.
    # 두 회전을 곱해서 한 번에 (롤 → 피치 순서는 여기선 한쪽만 쓰므로 상관없음)
    roll_q = np.array([math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2), 0, 0])
    pitch_q = np.array([math.cos(math.radians(pitch_deg) / 2), 0, math.sin(math.radians(pitch_deg) / 2), 0])
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, pitch_q, roll_q)

    data.qpos[0:3] = (0.0, 0.0, z)   # freejoint 위치
    data.qpos[3:7] = quat            # freejoint 자세
    mujoco.mj_forward(model, data)   # 위치·자세로부터 xpos, xmat, xipos(무게중심) 계산
    return model, data, model.body(config.Names.BOAT_BODY).id


def test_voxels_fill_the_hull():
    centers_b, cell_size, cell_volume = water.make_voxels()
    assert len(centers_b) == np.prod(WaterParams.VOXEL_RES)  # 8×4×4 = 128 칸
    assert cell_volume * len(centers_b) == pytest.approx(HULL_VOLUME)  # 칸 부피 합 = 선체 부피
    np.testing.assert_allclose(centers_b.mean(axis=0), 0.0, atol=1e-12)  # 칸들의 가운데 = 원점(상자 중심)


def test_out_of_water_no_buoyancy():
    _, data, bid = _boat(z=5.0)
    forces, _, volume = water.voxel_buoyancy(data, bid, waves=CALM)
    assert volume == 0.0
    assert np.all(forces == 0.0)


def test_fully_submerged_buoyancy():
    # 완전히 잠기면 부력 = 선체 부피 × ρg
    _, data, bid = _boat(z=-5.0)
    forces, _, volume = water.voxel_buoyancy(data, bid, waves=CALM)
    assert volume == pytest.approx(HULL_VOLUME)
    assert forces[:, 2].sum() == pytest.approx(HULL_VOLUME * RHO_G)


def test_buoyancy_points_up_only():
    _, data, bid = _boat(roll_deg=17)
    forces, _, _ = water.voxel_buoyancy(data, bid, waves=WaterParams.WAVES_DEFAULT)
    assert np.all(forces[:, :2] == 0.0)  # 옆으로 미는 성분 없음
    assert np.all(forces[:, 2] >= 0.0)   # 아래로 당기는 부력 없음


def test_floats_at_hand_draft():
    # 완료 기준 미리 확인: 흘수 0.1016 m 위치에서 부력 합 = 배 무게
    model, data, bid = _boat()
    forces, _, _ = water.voxel_buoyancy(data, bid, waves=CALM)
    weight = model.body_subtreemass[bid] * config.Physics.GRAVITY  # 배 무게도 MuJoCo 모델에서 읽음
    assert forces[:, 2].sum() == pytest.approx(weight)


def test_uses_simulation_time():
    # t 를 안 주면 data.time 을 쓴다 → 파도가 있으면 시간에 따라 부력이 달라짐
    _, data, bid = _boat()
    early = water.voxel_buoyancy(data, bid)[2]
    data.time = 1.3
    late = water.voxel_buoyancy(data, bid)[2]
    assert early != pytest.approx(late)


def _tilt_torque(**tilt):
    """흘수 위치에서 배를 기울였을 때, 무게중심 기준 부력 토크 (월드 좌표).
    토크 = Σ (작용점 − 무게중심) × 힘. 무게중심 위치는 MuJoCo 가 계산한 data.xipos 를 쓴다."""
    _, data, bid = _boat(**tilt)
    forces, points, _ = water.voxel_buoyancy(data, bid, waves=CALM)
    return np.cross(points - data.xipos[bid], forces).sum(axis=0)


@pytest.mark.parametrize("deg", [5, 15, -15])
def test_roll_restores(deg):
    # PoC 재발 방지: 기울이면 반대 방향으로 되돌리는 토크가 생겨야 함 (한 점 부력이면 0 이 됨)
    torque = _tilt_torque(roll_deg=deg)
    assert np.sign(torque[0]) == -np.sign(deg)
    # 크기는 직육면체 손 계산(k·θ)과 30% 안. 폭 방향 4칸뿐이라 생기는 근사 오차
    k_roll = water.restoring_stiffness(config.Hull.MASS)[0]
    assert abs(torque[0]) == pytest.approx(k_roll * math.radians(abs(deg)), rel=0.3)


@pytest.mark.parametrize("deg", [3, -3])
def test_pitch_restores(deg):
    torque = _tilt_torque(pitch_deg=deg)
    assert np.sign(torque[1]) == -np.sign(deg)


# ─────────────────────────────────────────────────────────────
# 5. 물 저항
# ─────────────────────────────────────────────────────────────
COEFFS = water.drag_coefficients(config.Hull.MASS, config.Hull.INERTIA)


def _moving_boat(yaw_deg=0.0, vel_w=(0.0, 0.0, 0.0), omega_b=(0.0, 0.0, 0.0)):
    """흘수 위치의 상자를 yaw_deg 만큼 돌려 놓고, 속도를 준 상태로 만든다.

    MuJoCo freejoint 속도(qvel) 규칙: 앞 3개 = 이동 속도(월드 좌표), 뒤 3개 = 회전 속도(몸체 좌표)
    """
    model, data, bid = _boat()
    half = math.radians(yaw_deg) / 2
    data.qpos[3:7] = (math.cos(half), 0, 0, math.sin(half))  # z 축(위) 둘레 회전 = 방향 돌리기
    data.qvel[0:3] = vel_w
    data.qvel[3:6] = omega_b
    mujoco.mj_forward(model, data)
    return model, data, bid


# ── 계수 ──
def test_heave_damping_ratio():
    # 계산한 위아래 계수를 거꾸로 감쇠비로 바꾸면 config 값(0.3)이 나와야 함
    k = config.Physics.RHO_WATER * config.Physics.GRAVITY * config.Hull.LENGTH * config.Hull.BEAM
    zeta = COEFFS["lin"][2] / (2 * math.sqrt(config.Hull.MASS * k))
    assert zeta == pytest.approx(WaterParams.HEAVE_DAMPING_RATIO)


def test_surge_drag_by_hand():
    # 앞으로 1 m/s: F = −(선형 5 × 1 + ½ × 1025 × Cd 0.6 × 앞면 넓이(폭 1.0 × 흘수 0.1016) × 1²) ≈ −36.24 N
    force_b, _ = water.drag_b([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], COEFFS)
    expected = -(5.0 + 0.5 * 1025 * 0.6 * (1.0 * config.Expect.DRAFT))
    assert force_b[0] == pytest.approx(expected)
    assert force_b[1] == 0.0 and force_b[2] == 0.0  # 앞으로만 움직이면 옆·위아래 저항 없음


def test_sideways_harder_than_forward():
    # 같은 속도라도 옆으로 미는 게 훨씬 힘들어야 함 (옆면이 앞면보다 넓고 Cd 도 큼)
    forward, _ = water.drag_b([1.0, 0, 0], [0, 0, 0], COEFFS)
    sideways, _ = water.drag_b([0, 1.0, 0], [0, 0, 0], COEFFS)
    assert abs(sideways[1]) > 3 * abs(forward[0])


# ── 방향: 저항은 언제나 움직임의 반대 ──
def test_no_motion_no_drag():
    force_b, torque_b = water.drag_b([0, 0, 0], [0, 0, 0], COEFFS)
    assert np.all(force_b == 0.0) and np.all(torque_b == 0.0)


def test_drag_opposes_motion():
    # 아무렇게나 움직여도, 축마다 저항 부호가 속도 부호의 반대
    rng = np.random.default_rng(1)
    for _ in range(20):
        v, w = rng.uniform(-3, 3, 3), rng.uniform(-2, 2, 3)
        force_b, torque_b = water.drag_b(v, w, COEFFS)
        assert np.all(force_b * v < 0)
        assert np.all(torque_b * w < 0)


def test_drag_grows_faster_than_speed():
    # 2차 항 때문에 속도 2배 → 힘은 2배보다 커야 함
    slow, _ = water.drag_b([1.0, 0, 0], [0, 0, 0], COEFFS)
    fast, _ = water.drag_b([2.0, 0, 0], [0, 0, 0], COEFFS)
    assert abs(fast[0]) > 2 * abs(slow[0])


# ── MuJoCo + frames 로 실제 상태에서 ──
def test_drag_uses_body_axes():
    # 좌표 축 검사 (노션 06 문제 2): 배를 90° 돌려 놓으면 "앞" = 월드 +y.
    # 월드 +y 로 1 m/s 움직이면, 배 입장에선 "앞으로" 가는 것이므로 앞뒤 저항(약 36 N)이 −y 로 나와야 함.
    # 축이 섞이면 좌우 저항(훨씬 큼)이 나와서 이 테스트가 깨진다.
    model, data, bid = _moving_boat(yaw_deg=90, vel_w=(0.0, 1.0, 0.0))
    force_w, _ = water.water_drag(model, data, bid, COEFFS)
    surge_drag, _ = water.drag_b([1.0, 0, 0], [0, 0, 0], COEFFS)
    np.testing.assert_allclose(force_w, (0.0, surge_drag[0], 0.0), atol=1e-9)


def test_current_pushes_still_boat(monkeypatch):
    # 해류: 배는 가만히 있고 물이 동쪽(+x)으로 1 m/s 흐르면, 배는 동쪽으로 밀려야 함
    monkeypatch.setattr(WaterParams, "WATER_VELOCITY_W", (1.0, 0.0, 0.0))
    model, data, bid = _moving_boat()
    force_w, _ = water.water_drag(model, data, bid, COEFFS)
    assert force_w[0] > 0
    np.testing.assert_allclose(force_w[1:], 0.0, atol=1e-9)


def test_drifting_with_current_no_drag(monkeypatch):
    # 물과 같은 속도로 떠내려가면 물에 대해선 멈춰 있는 것 → 저항 0
    monkeypatch.setattr(WaterParams, "WATER_VELOCITY_W", (0.5, -0.3, 0.0))
    model, data, bid = _moving_boat(yaw_deg=30, vel_w=(0.5, -0.3, 0.0))
    force_w, _ = water.water_drag(model, data, bid, COEFFS)
    np.testing.assert_allclose(force_w, 0.0, atol=1e-9)


def test_roll_rate_resisted():
    # 좌우로 기우는 중(roll 회전 속도 +)이면 반대 방향(−) 토크로 막아야 함
    model, data, bid = _moving_boat(omega_b=(0.5, 0.0, 0.0))
    _, torque_w = water.water_drag(model, data, bid, COEFFS)
    assert torque_w[0] < 0


# ─────────────────────────────────────────────────────────────
# 6. 실제 시뮬레이션: 물 + 로봇을 조립해서 시간을 흘려 본다 (1주차 완료 기준)
# ─────────────────────────────────────────────────────────────
# 지금까지(1~5)는 "이 자세라면 힘이 얼마인가"를 본 정지 사진 테스트.
# 여기서는 MuJoCo 가 0.002 초씩 배를 실제로 움직이게 하고, 몇 초 뒤 어디에 멈췄는지 본다.
# 부품이 각각 맞아도 조립(부호·중복·좌표·g 불일치 등)이 틀리면 여기서 드러난다.
from sim.robot import Robot  # noqa: E402  (이 섹션에서만 씀)
from sim.water import Water  # noqa: E402
from sim.world import build_world  # noqa: E402

# 흘수 허용 오차: config.Expect.DRAFT_TOL 이 채연 님과 합의 전이라 None → 그때까지 임시로 1 mm
DRAFT_TOL = config.Expect.DRAFT_TOL if config.Expect.DRAFT_TOL is not None else 0.001


def _sim_world(lift=0.0, roll_deg=0.0):
    """가영 님 build_world 로 물(잔잔한 물) + 동경 님 로봇(배)만 조립.

    로봇 build 가 배를 처음부터 흘수 위치에 놓는다. lift 만큼 더 올리고 roll_deg 만큼 기울여서 시작.
    """
    world = build_world(parts=(Water(waves=CALM), Robot()))
    water_part, robot = world.parts
    adr = robot.qpos_adr
    world.data.qpos[adr + 2] += lift
    half = math.radians(roll_deg) / 2
    world.data.qpos[adr + 3:adr + 7] = (math.cos(half), math.sin(half), 0, 0)  # x 축(앞) 둘레 = 롤
    mujoco.mj_forward(world.model, world.data)
    return world, water_part, robot


def _run(world, seconds):
    """seconds 초만큼 시뮬레이션. 한 칸(0.002 초)마다 팀 약속 순서대로:
    acc 비우기 → 물 → 로봇 → acc 를 MuJoCo 에 쓰기 → 한 칸 전진 (phase_manager.py 의 순서)

    돌려주는 값: 칸마다의 흘수 기록 [m] (흘수 = 높이/2 − 배 원점 높이)
    """
    water_part, robot = world.parts
    adr = robot.qpos_adr
    drafts = []
    for _ in range(round(seconds / world.model.opt.timestep)):
        world.acc.reset()
        water_part.add_forces(world.model, world.data, world.acc)
        robot.add_forces(world.model, world.data, world.acc)  # 명령 0 → 추력 0
        world.acc.apply(world.model, world.data)
        mujoco.mj_step(world.model, world.data)
        drafts.append(config.Hull.HEIGHT / 2 - world.data.qpos[adr + 2])
    return np.array(drafts)


def _roll_deg(world):
    q = world.data.qpos[world.parts[1].qpos_adr + 3:][:4]  # 쿼터니언 (w, x, y, z)
    return math.degrees(math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))


def test_bind_fails_loudly_without_boat():
    # 로봇 없이 물만 조립하면 배를 못 찾음 → 조용히 넘어가지 말고 바로 멈춰야 함
    with pytest.raises(AssertionError, match="boat"):
        build_world(parts=(Water(),))


def test_bind_reads_model():
    world, water_part, _ = _sim_world()
    assert water_part.body_id == world.model.body(config.Names.BOAT_BODY).id
    assert len(water_part.voxels[0]) == np.prod(WaterParams.VOXEL_RES)


def test_drop_settles_at_hand_draft():
    # ★ 1주차 완료 기준: 물 위 5 cm 에서 떨어뜨리면 출렁이다가 손 계산 흘수(0.1016 m)에 멈춘다
    world, _, _ = _sim_world(lift=0.05)
    drafts = _run(world, seconds=5.0)
    assert np.all(np.isfinite(drafts)), "계산이 폭발함 (NaN)"
    assert drafts.max() > config.Expect.DRAFT + 0.005, "정답보다 더 깊이 내려갔다 올라오는 출렁임이 있어야 함"
    assert drafts[-1] == pytest.approx(config.Expect.DRAFT, abs=DRAFT_TOL)
    # 마지막 1 초 동안 흔들림이 거의 없어야 "멈췄다"고 봄
    last_second = drafts[-round(1.0 / config.Sim.TIMESTEP):]
    assert last_second.max() - last_second.min() < DRAFT_TOL


def test_starts_still_at_draft():
    # 로봇이 처음부터 흘수 위치에 놓으므로, 그대로 두면 거의 움직이지 않아야 함
    world, _, _ = _sim_world()
    drafts = _run(world, seconds=1.0)
    assert np.abs(drafts - config.Expect.DRAFT).max() < DRAFT_TOL


def test_roll_release_settles():
    # 동경 님 요청: 15° 기울였다 놓으면 흔들리다 멈추는지 (실제 MuJoCo + 복셀 부력 + 회전 감쇠)
    world, _, _ = _sim_world(roll_deg=15)
    _run(world, seconds=3.0)
    assert abs(_roll_deg(world)) < 0.5
