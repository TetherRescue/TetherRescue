"""물 단위 테스트 (담당: 미현). 1~3 은 MuJoCo 없이, 4(복셀 부력)는 상자 하나짜리 MuJoCo 모델로 확인합니다."""
import math

import mujoco
import numpy as np
import pytest

from sim import config
from sim.water import water
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


def test_waterplane_of_a_frame_by_hand():
    # A형 (2026-11-19, 1.5 × 1.2 m, 홈 0.9 × 0.9, 다리 폭 0.15, 뱃머리 x 0.45~0.75 에서 반폭 0.6 → 0.2)
    #   넓이 = 1.5·1.2 − 홈 0.9·0.9 − 뱃머리 양옆 삼각형 2·(0.3·0.4/2) = 1.8 − 0.81 − 0.12 = 0.87 m²
    #   roll 2차 모멘트 ∫y² dA (x 축 둘레):
    #     다리 2개  2 · 1.2 · (0.6³ − 0.45³)/3 = 0.09990
    #     연결부    0.3 · 2 · 0.45³/3          = 0.01823
    #     뱃머리    ∫ 2h³/3 dx, h: 0.6→0.2     = (2/3) · 0.3 · (0.6⁴ − 0.2⁴)/(4 · 0.4) = 0.01600
    #     합 ≈ 0.1341 m⁴
    if config.Hull.SHAPE != "a_frame":
        pytest.skip("A형 선체 손 계산")
    assert config.Hull.WATERPLANE_AREA == pytest.approx(0.87, rel=1e-3)
    assert config.Hull.WATERPLANE_INERTIA[0] == pytest.approx(0.1341, rel=0.01)


def test_box_waterplane_matches_box_formula(monkeypatch):
    # 상자 모양이면 예전 직육면체 공식과 같아야 함: 넓이 L·B, roll L·B³/12, pitch B·L³/12
    monkeypatch.setattr(config.Hull, "SHAPE", "box")
    L, B = config.Hull.LENGTH, config.Hull.BEAM
    area, center_x, (i_roll, i_pitch) = config._waterplane()
    assert area == pytest.approx(L * B)
    assert center_x == pytest.approx(0.0, abs=1e-12)
    assert i_roll == pytest.approx(L * B ** 3 / 12, rel=1e-3)
    assert i_pitch == pytest.approx(B * L ** 3 / 12, rel=1e-3)


def test_restoring_stiffness_by_hand():
    # 손 계산 (벽이 곧은 선체): 흘수 T = m/(ρA), KB = T/2, BM = I/V (V = m/ρ), KG = H/2 + COM_z
    # A형 19.6 kg: T = 19.6/(1025·0.87) ≈ 0.02198 m, V ≈ 0.019122 m³, KG ≈ 0.0771 m
    #   roll  GM = 0.01099 + 0.13412/0.019122 − 0.0771 ≈ 6.948 m → k = m·g·GM ≈ 1336
    #   pitch GM = 0.01099 + 0.12733/0.019122 − 0.0771 ≈ 6.593 m → k ≈ 1268
    rho, g = config.Physics.RHO_WATER, config.Physics.GRAVITY
    volume = MASS / rho
    draft = volume / config.Hull.WATERPLANE_AREA
    kg = config.Hull.HEIGHT / 2 + config.Hull.COM[2]
    expected = [MASS * g * (draft / 2 + i / volume - kg) for i in config.Hull.WATERPLANE_INERTIA]
    np.testing.assert_allclose(water.restoring_stiffness(MASS), expected, rtol=1e-12)
    if config.Hull.SHAPE == "a_frame":
        assert water.restoring_stiffness(MASS) == pytest.approx((1336, 1268), rel=0.01)


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
    # 재발 방지: 예전 제안값 8.0 은 250 kg 배 기준 감쇠비 약 0.02 → 3 초 뒤에도 처음의 절반 넘게 흔들림
    # (동경 님 지적 재현). 선체가 바뀌어도 같은 뜻이 되게 숫자 대신 감쇠비 0.02 로 계수를 만들고,
    # 시간도 초 대신 흔들림 주기로 잰다 (가벼운 배는 주기가 짧아 3 초면 훨씬 많이 흔들림).
    k = water.restoring_stiffness(MASS)[axis]
    damping = 2 * 0.02 * math.sqrt(k * INERTIA[axis])
    period = 2 * math.pi * math.sqrt(INERTIA[axis] / k)
    history = _free_rock(k, INERTIA[axis], damping=damping, seconds=4 * period)
    tail = history[-int(period / config.Sim.TIMESTEP):]   # 마지막 한 주기
    assert np.max(np.abs(tail)) > 0.5 * math.radians(15)


def test_unstable_hull_raises(monkeypatch):
    # 무게중심이 너무 높으면 (GM ≤ 0) 바로 에러. A형은 넓어서 BM 이 약 7 m 라 10 m 위로 올린다
    monkeypatch.setattr(config.Hull, "COM", (0.0, 0.0, 10.0))
    with pytest.raises(ValueError, match="뒤집힘"):
        water.restoring_stiffness(MASS)


# ─────────────────────────────────────────────────────────────
# 4. 복셀 부력 — MuJoCo 상자 하나로 확인
# ─────────────────────────────────────────────────────────────
# voxel_buoyancy 는 좌표 변환을 frames.py 로 하므로 MuJoCo data 가 필요하다.
# world.py(가영) 없이도 되도록, 동경 님 tests/test_frames.py 처럼 상자 하나짜리 모델을 직접 만든다.
CALM = WaterParams.WAVES_CALM
RHO_G = config.Physics.RHO_WATER * config.Physics.GRAVITY
HULL_VOLUME = config.Hull.WATERPLANE_AREA * config.Hull.HEIGHT   # 벽이 곧은 선체: 위에서 본 넓이 × 높이
# 흘수만큼 잠긴 위치: 원점(상자 중심)이 수면 위 (높이/2 − 흘수) 에 있어야 선체 바닥이 수면 아래 흘수 깊이에 온다
#   0.2/2 − 0.022 ≈ +0.078 m   (Hull.HEIGHT 0.2 m, Expect.DRAFT ≈ 2.2 cm)
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
    assert len(centers_b) == np.prod(WaterParams.VOXEL_RES)  # 감싸는 상자 전체 30×8×4 = 960 칸
    np.testing.assert_allclose(centers_b.mean(axis=0), 0.0, atol=1e-12)  # 칸들의 가운데 = 원점(상자 중심)
    # 선체 모양 밖(홈·뱃머리 양옆)을 버린 칸: 부피 합 = 선체 부피, 가운데 x = 넓이 중심
    hull_b, _, _ = water.make_hull_voxels()
    assert cell_volume * len(hull_b) == pytest.approx(HULL_VOLUME)
    # 뱃머리 비스듬한 면은 칸 계단 근사라 넓이 중심이 1 mm 안쪽으로 어긋날 수 있음
    assert hull_b[:, 0].mean() == pytest.approx(config.Hull.WATERPLANE_CENTER_X, abs=1e-3)
    assert config.hull_contains(hull_b).all()


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
    # 완료 기준 미리 확인: 흘수(Expect.DRAFT ≈ 0.022 m) 위치에서 부력 합 = 배 무게
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


# 바닥 모서리가 물 밖으로 나오는 각도 = atan(2·흘수 / 폭). A형(흘수 2.2 cm, 폭 1.2 m)은 약 2.1°.
# 이 각도를 넘으면 복원 토크가 거의 늘지 않아서 작은 각도 공식 k·θ 가 맞지 않는다.
EDGE_EMERGE_DEG = math.degrees(math.atan(2 * config.Expect.DRAFT / config.Hull.BEAM))


@pytest.mark.parametrize("deg", [3, -3, 15, -15])
def test_roll_restores(deg):
    # PoC 재발 방지: 기울이면 반대 방향으로 되돌리는 토크가 생겨야 함 (한 점 부력이면 0 이 됨)
    torque = _tilt_torque(roll_deg=deg)
    assert np.sign(torque[0]) == -np.sign(deg)


@pytest.mark.parametrize("deg", [1.5, -1.5])
def test_roll_small_angle_matches_hand(deg):
    # 크기는 모서리가 잠겨 있는 작은 각도에서만 직육면체 손 계산(k·θ)과 비교한다.
    # 30% 허용: 폭 방향 칸 수가 적어서 생기는 복셀 근사 오차
    assert abs(deg) < EDGE_EMERGE_DEG, "작은 각도 공식은 모서리가 잠겨 있을 때만 성립"
    torque = _tilt_torque(roll_deg=deg)
    k_roll = water.restoring_stiffness(config.Hull.MASS)[0]
    assert abs(torque[0]) == pytest.approx(k_roll * math.radians(abs(deg)), rel=0.3)


@pytest.mark.parametrize("deg", [3, -3])
def test_pitch_restores(deg):
    torque = _tilt_torque(pitch_deg=deg)
    assert np.sign(torque[1]) == -np.sign(deg)


# ─────────────────────────────────────────────────────────────
# 5. 물 저항
# ─────────────────────────────────────────────────────────────
# Water.add_forces 와 같은 경로: 완전히 잠긴 기준 계수 → 흘수만큼 잠긴 상태로 줄이기
# (똑바로 흘수에 떠 있으면 wet_ratio = 흘수 / 높이, damping_scale = 1)
BASE_COEFFS = water.boat_drag_coefficients(config.Hull.MASS, config.Hull.INERTIA)
COEFFS = water.wetted_drag(BASE_COEFFS, config.Expect.DRAFT / config.Hull.HEIGHT, 1.0)


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
    # 계산한 위아래 계수를 거꾸로 감쇠비로 바꾸면 config 값(0.3)이 나와야 함 (용수철 세기 = ρ·g·수면 넓이)
    k = config.Physics.RHO_WATER * config.Physics.GRAVITY * config.Hull.WATERPLANE_AREA
    zeta = COEFFS["lin"][2] / (2 * math.sqrt(config.Hull.MASS * k))
    assert zeta == pytest.approx(WaterParams.HEAVE_DAMPING_RATIO)


def test_surge_drag_by_hand():
    # 앞으로 1 m/s: F = −(선형 계수 × 1 + ½ × 물 밀도 × Cd × 앞면 넓이(폭 × 흘수) × 1²)
    # 숫자를 직접 적지 않고 config 에서 읽는다 → 선체 크기가 바뀌어도 테스트를 고칠 필요 없음
    force_b, _ = water.drag_b([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], COEFFS)
    expected = -(WaterParams.DRAG_LINEAR[0]
                 + 0.5 * config.Physics.RHO_WATER * WaterParams.DRAG_CD[0]
                 * (config.Hull.BEAM * config.Expect.DRAFT))
    assert force_b[0] == pytest.approx(expected)
    assert force_b[1] == 0.0 and force_b[2] == 0.0  # 앞으로만 움직이면 옆·위아래 저항 없음


def test_sideways_harder_than_forward():
    # 같은 속도라도 옆으로 미는 게 훨씬 힘들어야 함 (옆면이 앞면보다 넓고 Cd 도 큼)
    # A형은 길이 1.5 · 폭 1.2 로 옆면/앞면 넓이 비가 1.25 라 직육면체(2.2)보다 차이가 작음 → 2배로 확인
    forward, _ = water.drag_b([1.0, 0, 0], [0, 0, 0], COEFFS)
    sideways, _ = water.drag_b([0, 1.0, 0], [0, 0, 0], COEFFS)
    assert abs(sideways[1]) > 2 * abs(forward[0])


def test_heave_quad_uses_full_bottom_area():
    # 위아래로 물을 미는 면은 바닥(길이 × 폭)이라 잠긴 깊이와 상관없이 넓이가 같다.
    #   → 앞뒤·옆 2차 계수만 잠긴 비율만큼 줄고, heave 2차 계수는 damping_scale 만 곱해진다.
    wet = config.Expect.DRAFT / config.Hull.HEIGHT
    full = water.wetted_drag(BASE_COEFFS, wet, 1.0)
    np.testing.assert_allclose(full["quad"][:2], BASE_COEFFS["quad"][:2] * wet)
    assert full["quad"][2] == pytest.approx(BASE_COEFFS["quad"][2])
    # 물 밖으로 나오면(damping_scale 0) heave 2차 저항도 사라진다
    dry = water.wetted_drag(BASE_COEFFS, 0.0, 0.0)
    assert dry["quad"][2] == 0.0


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
from sim.robot.robot import Robot  # noqa: E402  (이 섹션에서만 씀)
from sim.water.water import Water  # noqa: E402
from sim.world.world import build_world  # noqa: E402

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
    assert len(water_part.voxels[0]) == len(water.make_hull_voxels()[0])   # 모양 밖 칸은 버림


def test_drop_settles_at_hand_draft():
    # ★ 1주차 완료 기준: 물 위 5 cm 에서 떨어뜨리면 출렁이다가 손 계산 흘수(Expect.DRAFT ≈ 0.022 m)에 멈춘다
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


# ─────────────────────────────────────────────────────────────
# 7. 부가질량 (surge·sway 만. heave·yaw·결합항은 아직 없음)
# ─────────────────────────────────────────────────────────────
# 속도가 바뀔 때만 생기는 힘: F_b = −m_a · a_b (m_a = 계수 × 배 질량, 축마다 따로).
# 가속도는 직전 스텝과의 속도 차이로 구한다 (한 스텝 늦음 → 계수 < 1 이어야 안정).
from sim import frames  # noqa: E402

# 잔잔한 물 표류 기준: config.Expect.DRIFT_* 가 채연 님과 합의 전이라 None → 노션 초안 60 s / 0.1 m
DRIFT_SECONDS = config.Expect.DRIFT_SECONDS if config.Expect.DRIFT_SECONDS is not None else 60.0
DRIFT_MAX = config.Expect.DRIFT_MAX if config.Expect.DRIFT_MAX is not None else 0.1


@pytest.fixture
def added_mass_on(monkeypatch):
    monkeypatch.setattr(WaterParams, "ADDED_MASS_ENABLED", True)


@pytest.fixture
def no_surge_sway_drag(monkeypatch):
    # 손 계산을 깔끔하게: 앞뒤·좌우 저항을 0 으로 (위아래는 그대로 둬서 배가 떠 있게)
    monkeypatch.setattr(WaterParams, "DRAG_LINEAR", (0.0, 0.0, None))
    monkeypatch.setattr(WaterParams, "DRAG_CD", (0.0, 0.0, WaterParams.DRAG_CD[2]))


def _push(world, force_b, seconds):
    """배에 몸체 기준 일정한 힘 force_b [N] 을 무게중심에 걸고 seconds 초 진행.
    돌려주는 값: 마지막 몸체 기준 속도 (앞뒤, 좌우, 위아래) [m/s]"""
    water_part, robot = world.parts
    bid = water_part.body_id
    for _ in range(round(seconds / world.model.opt.timestep)):
        world.acc.reset()
        water_part.add_forces(world.model, world.data, world.acc)
        robot.add_forces(world.model, world.data, world.acc)
        world.acc.add(bid, frames.body_to_world(world.data, bid, force_b))
        world.acc.apply(world.model, world.data)
        mujoco.mj_step(world.model, world.data)
    vel_w, _ = frames.body_velocity_w(world.model, world.data, bid)
    return frames.world_to_body(world.data, bid, vel_w)


# ── 계수 ──
def test_added_mass_from_config():
    # 배 질량 대비 비율 (떠 있는 배는 배 질량 = 밀어낸 물 질량). heave 는 이번 주 미지원 → 0
    m_a = water.added_mass(MASS)
    assert m_a[0] == pytest.approx(WaterParams.ADDED_MASS_COEF[0] * MASS)
    assert m_a[1] == pytest.approx(WaterParams.ADDED_MASS_COEF[1] * MASS)
    assert m_a[2] == 0.0


def test_added_mass_coefficients_below_one():
    # 직전 스텝 가속도를 쓰는 방식은 계수 ≥ 1 이면 발산할 수 있음 → 설정 단계에서 막는다
    assert all(0.0 <= c < 1.0 for c in WaterParams.ADDED_MASS_COEF[:2])


def test_heave_added_mass_rejected(monkeypatch, added_mass_on):
    # heave 값을 넣으면 조용히 무시하지 말고 바로 멈춘다
    monkeypatch.setattr(WaterParams, "ADDED_MASS_COEF", (*WaterParams.ADDED_MASS_COEF[:2], 1.0))
    with pytest.raises(ValueError, match="heave"):
        build_world(parts=(Water(waves=CALM), Robot()))


# ── 방향 (노션 06 문제 2): 몸체 축 기준, 가속 반대 방향 ──
def test_added_mass_opposes_accel_in_body_axes():
    # 배를 90° 돌려 놓으면 몸체 앞(x) = 월드 +y, 몸체 왼쪽(y) = 월드 −x
    _, data, bid = _moving_boat(yaw_deg=90)
    m_a = water.added_mass(MASS)
    surge = water.added_mass_force_w(data, bid, (0.0, 1.0, 0.0), m_a)   # 앞으로 1 m/s²
    np.testing.assert_allclose(surge, (0.0, -m_a[0], 0.0), atol=1e-9)
    sway = water.added_mass_force_w(data, bid, (-1.0, 0.0, 0.0), m_a)   # 왼쪽으로 1 m/s²
    np.testing.assert_allclose(sway, (m_a[1], 0.0, 0.0), atol=1e-9)


# ── 실제 시뮬레이션 ──
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("enabled", [True, False])
def test_push_matches_hand(monkeypatch, no_surge_sway_drag, axis, enabled):
    # 저항 없이 일정한 힘 F 를 t 초 → 속도 = F · t / (m + m_a)   (부가질량 끄면 m_a = 0)
    monkeypatch.setattr(WaterParams, "ADDED_MASS_ENABLED", enabled)
    world, water_part, _ = _sim_world()
    mass = world.model.body_subtreemass[water_part.body_id]
    m_a = water.added_mass(mass)[axis] if enabled else 0.0
    force, seconds = 10.0, 0.5
    force_b = np.zeros(3)
    force_b[axis] = force
    vel_b = _push(world, force_b, seconds)
    assert vel_b[axis] == pytest.approx(force * seconds / (mass + m_a), rel=0.02)


def test_terminal_speed_unchanged(monkeypatch):
    # 부가질량은 속도가 바뀔 때만 작용 → 오래 밀어 일정 속도가 되면 켜든 끄든 같아야 함
    final = {}
    for enabled in (False, True):
        monkeypatch.setattr(WaterParams, "ADDED_MASS_ENABLED", enabled)
        world, _, _ = _sim_world()
        vel_b = _push(world, (0.0, 20.0, 0.0), seconds=8.0)
        assert np.all(np.isfinite(vel_b)), "계산이 폭발함 (NaN)"
        final[enabled] = vel_b[1]
    assert final[True] == pytest.approx(final[False], rel=0.01)


def _first_load(world, water_part):
    """배에 속도를 준 직후, 물이 처음 거는 외력 (xfrc_applied 사본)."""
    dof = world.model.body_dofadr[water_part.body_id]   # 배 freejoint 의 속도 시작 위치
    world.data.qvel[dof:dof + 3] = (1.0, 0.5, 0.0)      # 월드 기준 이동 속도
    mujoco.mj_forward(world.model, world.data)
    world.acc.reset()
    water_part.add_forces(world.model, world.data, world.acc)
    world.acc.apply(world.model, world.data)
    return world.data.xfrc_applied.copy()


def test_no_kick_on_first_step_or_reset(monkeypatch):
    # 직전 속도가 없을 때(첫 스텝, 리셋 직후)는 속도가 있어도 부가질량 힘 0
    # → 같은 상태에서 부가질량을 끈 물과 외력이 똑같아야 함
    loads = {}
    for enabled in (True, False):
        monkeypatch.setattr(WaterParams, "ADDED_MASS_ENABLED", enabled)
        world, water_part, _ = _sim_world()
        first = _first_load(world, water_part)
        _push(world, (0.0, 20.0, 0.0), seconds=0.1)       # 몇 스텝 돌려 직전 속도를 남김
        mujoco.mj_resetData(world.model, world.data)       # 시간이 0 으로 돌아감 = 리셋
        mujoco.mj_forward(world.model, world.data)
        loads[enabled] = (first, _first_load(world, water_part))
    for on, off in zip(loads[True], loads[False]):
        np.testing.assert_allclose(on, off, atol=1e-9)


def test_calm_water_no_drift(added_mass_on):
    # 노션 06 통합 테스트: 파도·추력 0 이면 오래 둬도 수평으로 거의 안 움직여야 함
    world, water_part, _ = _sim_world()
    bid = water_part.body_id
    start = world.data.xipos[bid, :2].copy()
    vel_b = _push(world, (0.0, 0.0, 0.0), seconds=DRIFT_SECONDS)
    assert np.all(np.isfinite(vel_b)), "계산이 폭발함 (NaN)"
    assert np.linalg.norm(world.data.xipos[bid, :2] - start) < DRIFT_MAX


# ─────────────────────────────────────────────────────────────
# 8. 큰 각도 복원력: 복셀 부력이 "정답"을 잘 따라가는지 (재발 방지)
# ─────────────────────────────────────────────────────────────
# 왜 필요한가
#   복원력 소실각(넘으면 스스로 못 돌아옴)은 이 복셀 부력으로 잰 복원팔 GZ 에서 나온다 (전복 판정은 2026-11-19 없앰).
#   부력 코드나 VOXEL_RES 가 바뀌어 큰 각도 계산이 틀어지면, 전복 기준각 근거도 조용히 무너진다.
#   → 정답과 비교하는 테스트를 두어, 틀어지면 바로 실패하게 한다.
#
# "정답"은 어떻게 구하나
#   상자였을 때는 기울인 단면(직사각형)을 수면 아래만 잘라 정확히 계산했다 (2026-11-16).
#   A형(2026-11-19)은 길이 방향으로 단면이 달라(홈·뱃머리) 단면 하나로 못 구한다.
#   → 같은 모양을 각 방향 2~4배 촘촘한 칸(REFERENCE_RES)으로 잘라 잰 값을 정답으로 쓴다.
#     칸을 늘리면 값이 정답에 수렴한다는 건 상자 때 확인함 (16×8×8, 32×16×16).
#
# GZ (복원팔) 란
#   배를 기울였을 때 부력 중심이 무게중심에서 옆으로 떨어진 거리 [m].
#   양수 = 다시 일어섬, 음수 = 더 넘어감. GZ 가 0 이 되는 각도 = 복원력 소실각 (넘으면 못 돌아옴).
#   두 방법 모두 "부력 = 배 무게" 가 되는 높이에 띄운 뒤 GZ 를 잰다.
#
# 2026-11-19 측정 (A형, VOXEL_RES 30×8×4 vs 정답 60×32×16)
#   roll : 10~80° 에서 2~5% 작음 (폭 방향 8칸). pitch: 2% 이내. 90° GZ 는 둘 다 +0.023 m (소실각 90° 넘음)
LARGE_ANGLE_GZ_TOL = 0.15        # GZ 허용 오차 (측정 최대 약 5% + 상자 때와 같은 여유)
VANISHING_ANGLE_TOL_DEG = 2.0    # 소실각 허용 오차 [°]
REFERENCE_RES = (60, 32, 16)     # 정답용 촘촘한 칸 (칸 약 1.5만 개)


@pytest.fixture(scope="module")
def tilt_boat():
    """선체 상자 하나짜리 모델 (4번 섹션 _boat). 각도만 바꿔 가며 재사용."""
    return _boat()


def _voxel_gz(tilt_boat, deg, axis, voxels=None):
    """복셀 부력(water.voxel_buoyancy)으로 잰 GZ [m]. voxels 를 안 주면 실제 시뮬레이션 칸(VOXEL_RES)."""
    model, data, bid = tilt_boat
    half = math.radians(deg) / 2
    quat = [math.cos(half), 0.0, 0.0, 0.0]
    quat[1 + axis] = math.sin(half)   # roll = x 축, pitch = y 축 둘레 회전
    weight = model.body_subtreemass[bid] * config.Physics.GRAVITY

    def buoyancy_at(z):
        data.qpos[0:3] = (0.0, 0.0, z)
        data.qpos[3:7] = quat
        mujoco.mj_forward(model, data)
        return water.voxel_buoyancy(data, bid, waves=CALM, voxels=voxels)

    lo, hi = -1.0, 1.0
    for _ in range(60):  # 반씩 줄이며 "부력 = 무게" 인 높이 찾기
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if buoyancy_at(mid)[0][:, 2].sum() > weight else (lo, mid)
    forces, points, _ = buoyancy_at((lo + hi) / 2)
    torque = np.cross(points - data.xipos[bid], forces).sum(axis=0)   # 무게중심 기준 회전력
    return -torque[axis] / weight   # GZ = 회전력 ÷ 무게, 일어서는 쪽을 양수로


@pytest.fixture(scope="module")
def exact_gz(tilt_boat):
    """정답 GZ 함수 (deg, axis) → GZ [m]. 같은 선체 모양을 REFERENCE_RES 칸으로 잘라 잰다."""
    reference = water.make_hull_voxels(res=REFERENCE_RES)
    return lambda deg, axis: _voxel_gz(tilt_boat, deg, axis, voxels=reference)


def _vanishing_angle(gz_of, start=80.0, stop=110.0, step=0.5):
    """GZ 가 양수 → 음수로 바뀌는 각도 [°] (사이는 직선으로 보간)."""
    angles = np.arange(start, stop + step / 2, step)
    values = [gz_of(a) for a in angles]
    for a0, a1, g0, g1 in zip(angles, angles[1:], values, values[1:]):
        if g0 > 0 >= g1:
            return a0 + (a1 - a0) * g0 / (g0 - g1)
    pytest.fail(f"{start}~{stop}° 안에서 GZ 가 0 이 되지 않음")


@pytest.mark.parametrize("axis", [0, 1], ids=["roll", "pitch"])
@pytest.mark.parametrize("deg", [10, 30, 50, 70, 80])
def test_large_angle_gz_matches_exact(tilt_boat, exact_gz, axis, deg):
    # 큰 각도에서도 복셀 GZ 가 정답과 ±15% 안
    assert _voxel_gz(tilt_boat, deg, axis) == pytest.approx(exact_gz(deg, axis), rel=LARGE_ANGLE_GZ_TOL)


@pytest.mark.parametrize("axis", [0, 1], ids=["roll", "pitch"])
def test_vanishing_angle_matches_exact(tilt_boat, exact_gz, axis):
    # 복원력 소실각이 정답과 ±2° 안 (복셀 부력이 큰 각도에서도 맞는지)
    exact = _vanishing_angle(lambda a: exact_gz(a, axis))
    voxel = _vanishing_angle(lambda a: _voxel_gz(tilt_boat, a, axis))
    assert voxel == pytest.approx(exact, abs=VANISHING_ANGLE_TOL_DEG)


# ─────────────────────────────────────────────────────────────
# 고속 가짜 양력 회귀 시험 (2026-11-20, 동경 님 제기: 빨라지면 떠올라 저항이 0 이 됨)
# ─────────────────────────────────────────────────────────────
def test_forward_speed_does_not_lift_pitched_boat():
    # 선수를 5° 든 배가 5 m/s 로 앞으로만 달릴 때, heave 저항이 배를 위로 밀면 안 된다.
    #   예전: 앞 속도 일부가 배 기준 z 속도로 섞여 heave 저항(바닥 넓이 · Cd 1.5)이 위로 수백 N 을 냈다.
    #   지금: heave 저항은 월드 위아래 속도로만 계산 → 위쪽 힘이 없어야 한다 (앞뒤 저항의 기울어진 성분만 아래로).
    world, water_part, robot = _sim_world()
    adr, dof = robot.qpos_adr, world.model.body_dofadr[robot.body_id]
    half = math.radians(-5) / 2                       # y 축(왼쪽) 둘레 음수 회전 = 선수가 들림
    world.data.qpos[adr + 3:adr + 7] = (math.cos(half), 0, math.sin(half), 0)
    world.data.qvel[dof:dof + 6] = 0.0
    world.data.qvel[dof] = 5.0                        # 월드 +x 로 5 m/s
    mujoco.mj_forward(world.model, world.data)
    force_w, _ = water.water_drag(world.model, world.data, robot.body_id, water_part.targets[0]['coeffs'])
    assert force_w[0] < 0                             # 앞으로 가면 뒤로 미는 저항은 그대로
    assert force_w[2] <= 1e-9                         # 위로 띄우는 힘은 없음


def test_drag_acts_at_center_of_buoyancy():
    # 저항은 잠긴 부분 가운데(부력 중심)에 걸려야 한다. 무게중심에 걸면 추진기의 선수 들림 돌림힘을 못 막음.
    world, water_part, robot = _sim_world()
    world.acc.reset()
    water_part.add_forces(world.model, world.data, world.acc)
    loads = [load for load in world.acc._loads if load[0] == robot.body_id]
    buoyancy, drag = loads[0], loads[1]
    assert buoyancy[3] is not None
    np.testing.assert_allclose(drag[3], buoyancy[3])
