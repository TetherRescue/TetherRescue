"""1 물: 파도·복셀 부력·저항·부가질량 (담당: 미현).

설정값은 전부 config.WaterParams(물 설정)·config.Physics(밀도·중력)·config.Hull(선체)에서 읽는다.
이 파일에는 숫자를 직접 적지 않는다.
"""

import math

import numpy as np

from sim import config


# ─────────────────────────────────────────────────────────────
# 파도: 수면 높이
# ─────────────────────────────────────────────────────────────
def wave_height(x, y, t, waves=config.WaterParams.WAVES_DEFAULT):
    """(x, y) 위치, t 초에서의 수면 높이 z [m] (월드 좌표, ENU).

    x, y 는 숫자 하나도, numpy 배열도 된다 (WaterAPI.WAVE_HEIGHT_VECTORIZED = True).
      - 숫자를 넣으면 숫자(float) 하나가 나온다.
      - 배열을 넣으면 같은 모양의 배열이 나온다. 예: 복셀 128개 위치를 한 번에 물어볼 수 있다.
    t 는 숫자 하나 (지금 시각).

    파도 = 사인파 성분들의 합. 성분 하나 = (진폭, 파장, 진행 방향, 위상)
      높이 = 진폭 · cos(파수 · (진행 방향으로 잰 거리) − 각진동수 · t + 위상)
      파수 = 2π / 파장,  각진동수 = sqrt(g · 파수)  ("깊은 물" 공식)
    waves=WaterParams.WAVES_CALM(빈 튜플)이면 어디서나 수면 높이 = SURFACE_Z.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.full(np.broadcast(x, y).shape, config.WaterParams.SURFACE_Z, dtype=float)

    g = config.Physics.GRAVITY
    for amp, length, direction, phase in waves:
        k = 2.0 * math.pi / length       # 파수: 1 m 당 물결이 몇 rad 만큼 바뀌나
        omega = math.sqrt(g * k)         # 각진동수: 1 초 당 몇 rad 만큼 오르내리나
        along = x * math.cos(direction) + y * math.sin(direction)  # 진행 방향으로 잰 거리
        z = z + amp * np.cos(k * along - omega * t + phase)

    return float(z) if z.ndim == 0 else z


# ─────────────────────────────────────────────────────────────
# 회전 감쇠: 감쇠비 → 계수
# ─────────────────────────────────────────────────────────────
def restoring_stiffness(mass):
    """배를 살짝 기울였을 때 되돌아오려는 힘의 세기 k [N·m/rad] = (roll, pitch).

    직육면체 선체(config.Hull) 공식:
      흘수 T = m / (rho · L · B)            # 물에 잠기는 깊이
      KB = T / 2                            # 바닥 → 부력 중심 (잠긴 부분의 가운데)
      BM = 폭² / (12 · T)                   # 부력 중심 → 메타센터 (roll 은 폭 B, pitch 는 길이 L)
      KG = HEIGHT / 2 + COM_z               # 바닥 → 무게중심 (원점이 상자 중심, COM_z 는 원점 기준)
      GM = KB + BM − KG                     # 메타센터 높이: 클수록 오뚝이처럼 잘 일어남
      k  = m · g · GM

    mass 는 MuJoCo 모델에서 읽은 배 무게(body_subtreemass)를 넣는다.
    GM ≤ 0 이면 배가 스스로 일어나지 못해 뒤집히는 설계라 바로 에러.
    """
    rho = config.Physics.RHO_WATER
    g = config.Physics.GRAVITY
    L, B, H = config.Hull.LENGTH, config.Hull.BEAM, config.Hull.HEIGHT

    draft = mass / (rho * L * B)
    kb = draft / 2.0
    kg = H / 2.0 + config.Hull.COM[2]

    stiffness = []
    for name, width in (("roll", B), ("pitch", L)):
        gm = kb + width ** 2 / (12.0 * draft) - kg
        if gm <= 0:
            raise ValueError(
                f"{name} 메타센터 높이 GM = {gm:.3f} m ≤ 0 → 기울면 되돌아오지 못하고 뒤집힘. "
                "무게중심(Hull.COM)을 낮추거나 선체 크기를 확인할 것."
            )
        stiffness.append(mass * g * gm)
    return tuple(stiffness)


def rot_drag_linear(mass, inertia):
    """회전 속도에 비례하는 저항 계수 [N·m·s/rad] = (roll, pitch, yaw).

    roll·pitch: 감쇠비 WaterParams.ROT_DAMPING_RATIO 에서 c = 2 · zeta · sqrt(k · I) 로 계산.
    yaw: 되돌아오려는 힘이 없어 감쇠비를 쓸 수 없으므로 WaterParams.ROT_DRAG_LINEAR[2] 를 그대로 쓴다.

    mass    = 배 무게 [kg]           (MuJoCo model.body_subtreemass)
    inertia = (roll, pitch, yaw) 관성 [kg·m²]  (MuJoCo model.body_inertia)
    """
    coeffs = []
    for zeta, k, inertia_axis in zip(config.WaterParams.ROT_DAMPING_RATIO, restoring_stiffness(mass), inertia):
        coeffs.append(2.0 * zeta * math.sqrt(k * inertia_axis))
    coeffs.append(config.WaterParams.ROT_DRAG_LINEAR[2])
    return tuple(coeffs)


# ─────────────────────────────────────────────────────────────
# 물 모듈 본체
# ─────────────────────────────────────────────────────────────
class Water:
    # 요구사항은 config 한 곳에만 적고 여기서는 가져다 쓴다 (값을 두 곳에 두지 않기).
    REQUIRES = config.WaterParams.REQUIRES

    def __init__(self, waves=config.WaterParams.WAVES_DEFAULT):
        self.waves = waves

    def wave_height(self, x, y, t):
        """이 물 객체의 파도로 계산한 수면 높이. 설명은 모듈 함수 wave_height 참고."""
        return wave_height(x, y, t, self.waves)

    def build(self, spec):
        """세계 조립 때 자기 몸(body/geom 등)을 spec 에 추가한다."""
        # TODO(미현)

    def bind(self, model):
        """이름 → id 를 한 번만 찾아 캐시한다. 없으면 assert 로 멈춘다."""
        # TODO(미현)

    def add_forces(self, model, data, acc):
        """스텝마다 힘을 acc(ForceAccumulator)에 더한다. 월드 좌표."""
        # TODO(미현)
