"""물 단위 테스트 (담당: 미현). MuJoCo 없이 돌아갑니다."""
import math

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
