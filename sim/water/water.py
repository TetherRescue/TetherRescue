"""1 물: 파도·복셀 부력·저항·부가질량 (담당: 미현).

설정값은 전부 config.WaterParams(물 설정)·config.Physics(밀도·중력)·config.Hull(선체)에서 읽는다.
물성 계수는 config에서 읽고, 복셀 적분의 수치 연산은 이 모듈에서 수행한다.
"""

import math

import numpy as np

from sim import config, frames


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

    벽이 곧은 선체(config.Hull, 위에서 본 모양 hull_contains) 공식:
      흘수 T = m / (rho · A)                # 물에 잠기는 깊이. A = Hull.WATERPLANE_AREA
      KB = T / 2                            # 바닥 → 부력 중심 (잠긴 부분의 가운데)
      BM = I / V                            # 부력 중심 → 메타센터. I = Hull.WATERPLANE_INERTIA (roll, pitch),
                                            #   V = m / rho. 상자면 I = L·B³/12 → BM = B² / (12 · T)
      KG = HEIGHT / 2 + COM_z               # 바닥 → 무게중심 (원점이 상자 중심, COM_z 는 원점 기준)
      GM = KB + BM − KG                     # 메타센터 높이: 클수록 오뚝이처럼 잘 일어남
      k  = m · g · GM

    mass 는 MuJoCo 모델에서 읽은 배 무게(body_subtreemass)를 넣는다.
    GM ≤ 0 이면 배가 스스로 일어나지 못해 뒤집히는 설계라 바로 에러.
    """
    rho = config.Physics.RHO_WATER
    g = config.Physics.GRAVITY
    H = config.Hull.HEIGHT
    volume = mass / rho

    draft = volume / config.Hull.WATERPLANE_AREA
    kb = draft / 2.0
    kg = H / 2.0 + config.Hull.COM[2]

    stiffness = []
    for name, inertia in zip(("roll", "pitch"), config.Hull.WATERPLANE_INERTIA):
        gm = kb + inertia / volume - kg
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
# 복셀 부력: 선체를 작은 상자로 잘라 칸마다 잠긴 만큼 부력을 건다
# ─────────────────────────────────────────────────────────────
def make_voxels(size=None, res=None):
    """선체 상자를 res = (nx, ny, nz) 칸으로 자른다.

    돌려주는 값
      centers_b : (칸 수, 3) 배열. 칸 중심 위치 (몸체 좌표, 원점 = 선체 상자 중심)
      cell_size : (dx, dy, dz) 칸 하나의 길이·폭·높이 [m]
      cell_volume : 칸 하나의 부피 [m³]
    size 를 안 주면 config.Hull (길이, 폭, 높이), res 를 안 주면 WaterParams.VOXEL_RES.
    """
    if size is None:
        size = (config.Hull.LENGTH, config.Hull.BEAM, config.Hull.HEIGHT)
    if res is None:
        res = config.WaterParams.VOXEL_RES

    cell_size = np.array(size, dtype=float) / np.array(res)
    # 축마다 칸 중심 좌표: 예) 길이 2.4 m 를 8칸 → −1.05, −0.75, …, +1.05
    axes = [(np.arange(n) + 0.5) * d - s / 2 for n, d, s in zip(res, cell_size, size)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    centers_b = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return centers_b, cell_size, float(np.prod(cell_size))


def make_hull_voxels(size=None, res=None):
    """make_voxels 와 같지만 선체 모양(config.hull_contains) 밖의 칸을 뺀다. 칸 중심이 안이면 그 칸 전체를 씀.

    VOXEL_RES 는 홈·다리 경계가 칸 경계와 맞게 골라 둠 → 상자 부분은 정확, 뱃머리 비스듬한 면만 계단 근사.
    """
    centers_b, cell_size, cell_volume = make_voxels(size=size, res=res)
    return centers_b[config.hull_contains(centers_b)], cell_size, cell_volume


def voxel_buoyancy(data, body_id, t=None, waves=config.WaterParams.WAVES_DEFAULT, voxels=None,
                   gravity=config.Physics.GRAVITY):
    """지금 MuJoCo 상태(data)에서 몸체의 칸마다 부력과 작용점 (월드 좌표).

    data    : MuJoCo MjData. 몸체의 현재 위치·기울기가 들어 있다
    body_id : body 번호. bind 때 한 번 찾아 둔 값
    t       : 시각 [s] (파도 계산용). 안 주면 data.time (시뮬레이션 현재 시각)
    voxels  : (칸 중심(몸체 좌표), 칸 크기, 칸 부피). 안 주면 선체 make_hull_voxels().
              배 모양은 안 변하니 bind 때 한 번 만들어 넘기면 됨
    gravity : 중력 크기 [m/s²]. Water 는 MuJoCo model.opt.gravity 크기를 넘긴다

    돌려주는 값
      forces_w : (칸 수, 3) 칸마다 부력. 위쪽(+z)만 있음
      points_w : (칸 수, 3) 칸마다 부력을 거는 점 = 그 칸에서 잠긴 부분의 가운데
      submerged_volume : 잠긴 부피 합 [m³]

    좌표 변환은 직접 계산하지 않고 전부 sim/frames.py 함수로 한다 (노션 06 문제 2, 동경 요청).
      각자 회전행렬을 곱하면 축이 어긋나도 오류 없이 조용히 틀리기 때문.
    """
    centers_b, cell_size, cell_volume = voxels if voxels is not None else make_hull_voxels()
    if t is None:
        t = data.time

    # 1) 칸 중심을 월드 좌표로 옮기기
    #    centers_b = "배 기준" 위치 (예: 배 중심에서 앞으로 1 m, 왼쪽 0.3 m)
    #    centers_w = "세상 기준" 위치 (예: 동쪽 5 m, 북쪽 2 m, 높이 −0.05 m)
    #    point_body_to_world 가 배의 기울기만큼 돌리고 배 위치만큼 옮겨 준다. 128 칸을 (128, 3) 배열로 한 번에.
    centers_w = frames.point_body_to_world(data, body_id, centers_b)

    # 2) 칸마다 그 자리의 수면 높이 (128 칸을 한 번에)
    surface = wave_height(centers_w[:, 0], centers_w[:, 1], t, waves)

    # 3) 칸의 "위아래 폭" 구하기
    #    똑바로 떠 있으면 칸 높이 dz(0.1 m) 그대로지만, 배가 기울면 칸도 비스듬해져서
    #    맨 아래 꼭짓점 ~ 맨 위 꼭짓점의 높이 차이가 커진다.
    #    칸의 세 모서리(앞뒤 dx, 좌우 dy, 위아래 dz)를 각각 세상 방향으로 돌려 보고,
    #    그 모서리들이 위아래로 얼마나 서 있는지(z 성분 크기)를 더하면 된다.
    #      body_to_world = 방향만 돌리는 함수 (위치 이동 없음) → 모서리 "방향·길이" 변환에 맞음
    edges_b = np.diag(cell_size)   # 행 0 = (dx,0,0) 앞뒤 모서리, 행 1 = (0,dy,0) 좌우, 행 2 = (0,0,dz) 위아래
    height = sum(abs(frames.body_to_world(data, body_id, edge_b)[2]) for edge_b in edges_b)

    # 4) 잠긴 비율 = (수면 − 칸 바닥) ÷ 칸 위아래 폭, 0 ~ 1 로 자름
    #    0 = 물 밖, 1 = 완전히 잠김, 0.3 = 아래쪽 30% 만 잠김
    bottom = centers_w[:, 2] - height / 2
    fraction = np.clip((surface - bottom) / height, 0.0, 1.0)

    # 5) 부력 = 잠긴 부피 × 물 밀도 × g, 방향은 위 (+z). 중력은 크기만 양수로 적기로 함
    #    부력은 원래 "세상 기준 위쪽"이라 월드 좌표에서 바로 만든다 (변환 필요 없음)
    volume = fraction * cell_volume
    forces_w = np.zeros_like(centers_w)
    forces_w[:, 2] = config.Physics.RHO_WATER * gravity * volume

    # 6) 작용점: 칸 중심 대신 "잠긴 부분의 가운데" 높이 (반쯤 잠긴 칸은 아래쪽에 걸림)
    #    → 기울었을 때 되돌리는 힘(토크)이 더 정확해진다
    points_w = centers_w.copy()
    points_w[:, 2] = bottom + fraction * height / 2

    return forces_w, points_w, float(volume.sum())


# ─────────────────────────────────────────────────────────────
# 물 저항: 배가 움직이면 물이 반대 방향으로 브레이크를 건다
# ─────────────────────────────────────────────────────────────
# 저항은 세 가지를 더한 것이다.
#   ① 선형 저항   F = −c · v                  느릴 때 주로 작용 (배를 완전히 멈추게 해 줌)
#   ② 2차 저항    F = −½·ρ·Cd·A · |v| · v     빠를 때 주로 작용 (속도 2배 → 힘 4배)
#   ③ 감쇠        c = 2·ζ·√(m·k)              위아래 출렁임·좌우앞뒤 기울기를 멈추게 함 (①의 계수를 감쇠비로 정한 것)
# 축마다 따로 계산한다. 배는 앞으로는 잘 나가고 옆으로는 잘 안 밀리듯, 방향마다 저항이 다르기 때문.
#   이동 축 순서: (surge 앞뒤, sway 좌우, heave 위아래)   — 배 기준(몸체 좌표, FLU)
#   회전 축 순서: (roll 좌우 기울기, pitch 앞뒤 기울기, yaw 방향 돌기)


def heave_damping(mass, area=None, gravity=config.Physics.GRAVITY):
    """위아래 출렁임을 멈추게 하는 선형 계수 c [N·s/m] (감쇠비 HEAVE_DAMPING_RATIO 에서 계산).

    배를 살짝 눌렀다 놓으면 용수철처럼 위아래로 출렁인다.
      용수철 세기 k = ρ · g · A   (1 m 더 잠기면 부력이 이만큼 늘어남 = 수면에 닿은 면적 A × ρg)
      무게       m = 배 무게
    감쇠비 ζ 로 "얼마나 빨리 멎을지"를 정하면 필요한 계수는 c = 2 · ζ · √(m · k).
    → 배 무게·크기가 바뀌어도 출렁임이 멎는 빠르기는 그대로 유지된다.
    area 를 안 주면 Hull.WATERPLANE_AREA. Water 는 MuJoCo 모델의 중력을 넘긴다.
    """
    if area is None:
        area = config.Hull.WATERPLANE_AREA
    k = config.Physics.RHO_WATER * gravity * area
    return 2.0 * config.WaterParams.HEAVE_DAMPING_RATIO * math.sqrt(mass * k)


def drag_coefficients(linear, angular, cd, angular_quad, dimensions):
    """완전히 잠겼을 때 기준 저항 계수. 몸 모양은 안 변하니 bind 때 한 번만 부르면 됨.

    linear       : 이동 선형 계수 (surge, sway, heave)       [N·s/m]
    angular      : 회전 선형 계수 (roll, pitch, yaw)         [N·m·s/rad]
    cd           : 이동 항력 계수 Cd (surge, sway, heave)
    angular_quad : 회전 2차 계수 (roll, pitch, yaw)          [N·m·s²/rad²]
    dimensions   : 몸체 상자 크기 (길이, 폭, 높이) [m]

    돌려주는 dict (모두 길이 3 배열, 축 순서는 위 설명 참고)
      lin      : 이동 선형 계수 c        [N·s/m]       → F = −lin · v
      quad     : 이동 2차 계수 ½·ρ·Cd·A  [N·s²/m²]     → F = −quad · |v| · v   (A = 완전히 잠긴 면적)
      rot_lin  : 회전 선형 계수          [N·m·s/rad]   → τ = −rot_lin · ω
      rot_quad : 회전 2차 계수           [N·m·s²/rad²] → τ = −rot_quad · |ω| · ω
    잠긴 정도에 맞춘 값은 wetted_drag() 로 매 스텝 만든다.
    """
    coeffs = np.asarray([linear, angular, cd, angular_quad], dtype=float)
    if coeffs.shape != (4, 3) or not np.isfinite(coeffs).all() or np.any(coeffs < 0):
        raise ValueError("물 저항 계수는 유한한 0 이상 값이어야 합니다.")
    L, B, H = dimensions
    # 각 방향에서 "물을 정면으로 미는 면적" A [m²]: 앞뒤 = 폭 × 높이, 좌우 = 길이 × 높이, 위아래 = 길이 × 폭
    area = np.array([B * H, L * H, L * B])
    return {
        "lin": coeffs[0],
        "quad": 0.5 * config.Physics.RHO_WATER * coeffs[2] * area,  # ½ · ρ · Cd · A (속도만 빼고 미리 곱함)
        "rot_lin": coeffs[1],
        "rot_quad": coeffs[3],
    }


def boat_drag_coefficients(mass, inertia, size=None, gravity=config.Physics.GRAVITY):
    """선체 기준 저항 계수 (drag_coefficients 에 선체 값을 채운 것).

    선형 계수: 앞뒤·좌우는 config 숫자, 위아래는 감쇠비로 계산 (config 에서 None 으로 비워 둔 칸).
    회전 선형: roll·pitch 는 감쇠비로 계산, yaw 는 config 숫자 (rot_drag_linear).
    size 를 안 주면 config.Hull (길이, 폭, 높이).
    """
    if size is None:
        size = (config.Hull.LENGTH, config.Hull.BEAM, config.Hull.HEIGHT)
    surge_lin, sway_lin, _ = config.WaterParams.DRAG_LINEAR
    linear = (surge_lin, sway_lin, heave_damping(mass, gravity=gravity))
    return drag_coefficients(linear, rot_drag_linear(mass, inertia), config.WaterParams.DRAG_CD,
                             config.WaterParams.ROT_DRAG_QUAD, size)


def wetted_drag(coeffs, wet_ratio, damping_scale):
    """잠긴 정도에 맞춰 저항 계수를 줄인다. 물 밖으로 나오면 저항이 사라진다.

    wet_ratio     : 잠긴 부피 ÷ 전체 부피 (0 ~ 1). 2차 항 중 앞뒤·옆(surge·sway) 면적을 이만큼 줄인다
                    (똑바로 떠 있으면 높이 × wet_ratio = 흘수 → 앞면 넓이 = 폭 × 흘수)
    damping_scale : min(1, 잠긴 부피 × ρ ÷ 질량). 평형 잠김량 이상이면 1 → 감쇠비 계약값 그대로.
                    선형 항, 회전 항, heave 2차 항에 곱한다

    heave(위아래) 2차 항은 wet_ratio 로 줄이지 않는다.
      위아래로 물을 미는 면은 바닥(길이 × 폭)이라, 얼마나 잠겼든 넓이가 같기 때문.
      (wet_ratio 를 곱하면 흘수 상태에서 바닥 넓이가 약 1/8 로 줄어든 것처럼 계산됨)
      물 밖으로 나오면 사라지도록 damping_scale 만 곱한다.
    """
    quad_scale = np.array([wet_ratio, wet_ratio, damping_scale])   # (surge, sway, heave)
    return {
        "lin": coeffs["lin"] * damping_scale,
        "quad": coeffs["quad"] * quad_scale,
        "rot_lin": coeffs["rot_lin"] * damping_scale,
        "rot_quad": coeffs["rot_quad"] * damping_scale,
    }


def drag_b(vel_b, omega_b, coeffs):
    """배 기준(몸체 좌표) 속도를 넣으면 배 기준 저항 힘·토크를 돌려준다. 좌표 변환 없는 순수 계산.

    vel_b   : 물에 대한 배의 이동 속도 (앞뒤, 좌우, 위아래) [m/s]
    omega_b : 배의 회전 속도 (roll, pitch, yaw) [rad/s]
    coeffs  : drag_coefficients() 결과

    공식 (축마다 따로)
      힘  F = −lin · v − quad · |v| · v
      토크 τ = −rot_lin · ω − rot_quad · |ω| · ω
    부호가 항상 "−" 라서 저항은 언제나 움직이는 반대 방향이다.
    |v| · v 로 쓰는 이유: v² 는 항상 양수라 방향을 잃는데, |v|·v 는 크기는 v² 이면서 부호는 v 를 따라감.
    """
    v = np.asarray(vel_b, dtype=float)
    w = np.asarray(omega_b, dtype=float)
    force_b = -coeffs["lin"] * v - coeffs["quad"] * np.abs(v) * v
    torque_b = -coeffs["rot_lin"] * w - coeffs["rot_quad"] * np.abs(w) * w
    return force_b, torque_b


def water_drag(model, data, body_id, coeffs, heave_world_only=False):
    """지금 MuJoCo 상태에서 배가 받는 물 저항 (월드 좌표). 힘은 무게중심에 작용한다.

    heave_world_only: True 면 몸체 z 축(heave) 저항을 세상 위아래 속도로만 계산한다 (아래 3-1).
      몸체 z 축이 위를 향한다고 가정하는 보정이라 배에만 쓴다. 사람·구명환은 몸체 축이 어디로든 누울 수 있어
      (예: 착수한 환은 몸체 z 축이 옆을 향함) 켜면 그 축의 저항이 사라진다 → 기본값 False.

    순서
      1) 배 속도를 세상 기준으로 받는다 (무게중심의 이동 속도, 회전 속도)
      2) 해류를 뺀다 → "물에 대한" 속도. 물과 같이 떠내려가면 저항이 없어야 하므로
      3) 배 기준으로 돌린다 → 앞뒤·좌우·위아래로 나눠 계산하기 위해
      4) 축마다 저항 계산 (drag_b)
      5) 다시 세상 기준으로 돌린다 → ForceAccumulator 에는 월드 좌표만 넣는 규칙
    좌표 변환은 전부 sim/frames.py 함수로 한다 (직접 회전행렬 곱하지 않기).

    돌려주는 값: (force_w, torque_w). acc.add(body_id, force_w, torque_w) 로 넣으면 됨
      (point_w 를 안 주면 ForceAccumulator 가 무게중심에 건다 → body_velocity_w 도 무게중심 기준이라 맞음)

    1주차 단순화: 파도 때문에 물이 위아래로 움직이는 것은 빼고, 해류(WATER_VELOCITY_W)만 반영한다.
    """
    # 1) 세상 기준 속도. ⚠️ mj_objectVelocity(..., 1) 을 직접 쓰지 않는다 (PoC 발산 원인) → frames 함수만
    vel_w, omega_w = frames.body_velocity_w(model, data, body_id)

    # 2) 물에 대한 속도 = 배 속도 − 물(해류) 속도. 회전은 물이 돌지 않으므로 그대로
    rel_vel_w = vel_w - np.asarray(config.WaterParams.WATER_VELOCITY_W, dtype=float)

    # 3) 배 기준으로 돌리기 (방향만 돌림)
    rel_vel_b = frames.world_to_body(data, body_id, rel_vel_w)
    omega_b = frames.world_to_body(data, body_id, omega_w)

    # 3-1) heave(위아래) 저항은 "실제로 위아래로 움직이는 속도"로만 계산한다 (2026-11-20, 동경 님 고속 문제 제기).
    #   배가 기울면 앞으로 가는 속도 일부가 배 기준 z 축에 섞인다. 이걸 그대로 쓰면 큰 heave 계수
    #   (바닥 전체 넓이 · Cd 1.5, 감쇠비 선형 항)가 앞으로 달리는 배를 위로 밀어 올리는 가짜 양력이 되어,
    #   고속에서 배가 뜨고 → 잠긴 부피가 줄어 저항이 줄고 → 더 빨라지는 되먹임으로 물 밖까지 날아갔다.
    #   → 월드 위아래 속도만 배 기준으로 돌려 heave 칸에 쓴다. 앞으로 달릴 때 생기는 활주 양력은 따로 넣는다.
    #   배에만 적용 (heave_world_only). 사람·구명환에 쓰면 몸체 z 축이 옆을 향할 때 그 방향 저항이 0 이 되어,
    #   착수한 환이 멈추지 않고 미끄러졌다 (2026-11-20, 발사 미션 ATTACH 시간 초과의 원인).
    if heave_world_only:
        rel_vel_b[2] = frames.world_to_body(data, body_id, np.array([0.0, 0.0, rel_vel_w[2]]))[2]

    # 4) 축마다 저항
    force_b, torque_b = drag_b(rel_vel_b, omega_b, coeffs)

    # 5) 세상 기준으로 되돌리기
    force_w = frames.body_to_world(data, body_id, force_b)
    torque_w = frames.body_to_world(data, body_id, torque_b)
    return force_w, torque_w


# ─────────────────────────────────────────────────────────────
# 고속 활주 양력: 빨리 달리는 배가 물을 아래로 밀어내며 떠오르는 힘 (2026-11-21, 동경 님 요청 1)
# ─────────────────────────────────────────────────────────────
def planing_lift(data, body_id, voxels, forces_w, gravity, vel_b):
    """활주 양력 (월드 좌표 힘, 작용점). 힘이 없으면 (None, None).

    voxels   : 이 몸체의 (칸 중심(몸체 좌표), 칸 크기, 칸 부피). Water.targets 의 값
    forces_w : voxel_buoyancy 가 돌려준 칸별 부력 → 칸마다 얼마나 잠겼는지 거꾸로 구하는 데 씀
    gravity  : 중력 크기 [m/s²]
    vel_b    : 물에 대한 무게중심 속도 (몸체 좌표) [m/s]

    순서 (식과 출처는 config.WaterParams.PLANING_* 주석)
      1) 앞으로 가는 속도 V, 뱃머리 들린 각도 τ. 뒤로 가거나 뱃머리가 숙여져 있으면 양력 없음
      2) 바닥 칸 중 젖은 칸으로 젖은 바닥의 길이·넓이 → 평균 폭 b, 길이비 λ
      3) L = ½ρV² · b² · C · λ^0.5 · τ^1.1  (τ 는 도 단위, Savitsky 의 속도 항만)
      4) 힘은 바닥에 수직(몸체 +z). 수직 성분이 L 이 되도록 L / cos τ 로 건다
         → 남는 수평 성분이 뒤를 향해 유도 저항이 된다 (L · tan τ)
      5) 작용점 = 젖은 바닥 뒤끝에서 앞으로 lp (Savitsky 압력 중심 식), 높이는 바닥면
    """
    if not config.WaterParams.PLANING_ENABLED or gravity <= 0:
        return None, None
    speed = float(vel_b[0])
    if speed <= 0:
        return None, None
    # 1) 트림 τ = 몸체 앞축(x_b)이 수평에서 위로 들린 각도
    fwd_w = frames.body_to_world(data, body_id, (1.0, 0.0, 0.0))
    trim = math.atan2(fwd_w[2], math.hypot(fwd_w[0], fwd_w[1]))
    if trim <= 0:
        return None, None
    trim_deg = min(math.degrees(trim), config.WaterParams.PLANING_MAX_TRIM_DEG)

    # 2) 바닥 칸(가장 아래 층)이 얼마나 젖었나. 칸별 부력 ÷ (ρ g 칸부피) = 잠긴 비율
    centers_b, cell_size, cell_volume = voxels
    z_low = centers_b[:, 2].min()
    bottom = np.isclose(centers_b[:, 2], z_low)
    wet_depth = forces_w[bottom, 2] / (config.Physics.RHO_WATER * gravity * cell_volume) * cell_size[2]
    weight = np.clip(wet_depth / config.WaterParams.PLANING_WET_DEPTH, 0.0, 1.0)
    if weight.sum() <= 0:
        return None, None
    xs = centers_b[bottom, 0][weight > 0]
    wet_length = float(xs.max() - xs.min() + cell_size[0])
    wet_area = float(weight.sum() * cell_size[0] * cell_size[1])
    beam = wet_area / wet_length
    lam = wet_length / beam

    # 3) 양력 크기 (수직 성분)
    lift = (0.5 * config.Physics.RHO_WATER * speed ** 2 * beam ** 2
            * config.WaterParams.PLANING_LIFT_COEF * lam ** 0.5 * trim_deg ** 1.1)

    # 4) 바닥에 수직인 힘 → 월드
    force_w = frames.body_to_world(data, body_id, (0.0, 0.0, lift / math.cos(math.radians(trim_deg))))

    # 5) 압력 중심: lp / (λ b) = 0.75 − 1 / (5.21 · Cv² / λ² + 2.39),  Cv = V / √(g b)
    cv = speed / math.sqrt(gravity * beam)
    lp = lam * beam * (0.75 - 1.0 / (5.21 * cv ** 2 / lam ** 2 + 2.39))
    y_mid = float(np.average(centers_b[bottom, 1], weights=weight))
    point_b = (float(xs.min() - cell_size[0] / 2 + lp), y_mid, float(z_low - cell_size[2] / 2))
    return force_w, frames.point_body_to_world(data, body_id, point_b)


# ─────────────────────────────────────────────────────────────
# 부가질량: 배가 속도를 바꾸면 주변 물도 같이 밀려서 배가 더 무겁게 움직인다
# ─────────────────────────────────────────────────────────────
# 기본값은 꺼져 있다 (config.WaterParams.ADDED_MASS_ENABLED 주석 참고).
# 이번 구현 범위: surge·sway 만. heave·yaw, 축끼리 섞이는 결합항(Munk 모멘트 등)은 아직 없음.
def added_mass(mass):
    """축별 부가질량 [kg] = (surge, sway, heave). heave 는 미지원이라 0.

    mass = 배 무게 [kg] (MuJoCo model.body_subtreemass)
    """
    surge, sway, heave = config.WaterParams.ADDED_MASS_COEF
    if heave is not None:
        raise ValueError("heave 부가질량은 아직 지원하지 않습니다. ADDED_MASS_COEF 의 heave 는 None 으로 두세요.")
    for name, coef in (("surge", surge), ("sway", sway)):
        # 직전 스텝 가속도로 힘을 거는 방식이라 계수 ≥ 1 이면 계산이 발산할 수 있다
        if not 0.0 <= coef < 1.0:
            raise ValueError(f"{name} 부가질량 계수는 0 이상 1 미만이어야 합니다 (지금 {coef}).")
    return np.array([surge * mass, sway * mass, 0.0])


def added_mass_force_w(data, body_id, accel_w, added_mass_b):
    """가속도 반대 방향의 부가질량 힘 (월드 좌표). F_b = −m_a · a_b (몸체 축마다 따로).

    accel_w      : 물에 대한 배 무게중심의 가속도 (월드 좌표) [m/s²]
    added_mass_b : added_mass() 결과 [kg]
    축마다 부가질량이 다르므로(옆이 앞보다 큼) 몸체 좌표로 돌려 계산하고 다시 월드로 되돌린다.
    """
    accel_b = frames.world_to_body(data, body_id, np.asarray(accel_w, dtype=float))
    return frames.body_to_world(data, body_id, -np.asarray(added_mass_b) * accel_b)


# ─────────────────────────────────────────────────────────────
# 물 모듈 본체
# ─────────────────────────────────────────────────────────────
class Water:
    # 요구사항은 config 한 곳에만 적고 여기서는 가져다 쓴다 (값을 두 곳에 두지 않기).
    REQUIRES = config.WaterParams.REQUIRES

    def __init__(self, waves=config.WaterParams.WAVES_DEFAULT, *, floating_bodies=(), surface_visual=False, include_boat=True):
        self.waves = waves
        self.surface_visual = surface_visual
        self.include_boat = include_boat
        # 사람·구명환의 실험 부력은 명시적으로 선택한다. 선체 부력은 기본 적용.
        self.floating_bodies = tuple(floating_bodies)
        self.targets = []

    def wave_height(self, x, y, t):
        """이 물 객체의 파도로 계산한 수면 높이. 설명은 모듈 함수 wave_height 참고."""
        return wave_height(x, y, t, self.waves)

    def build(self, spec):
        """선택한 수면 표시는 접촉·질량이 없는 평면이다 (파도 변형은 미표시)."""
        if self.surface_visual:
            import mujoco
            spec.worldbody.add_geom(name=config.Names.WATER_SURFACE_GEOM,
                                    type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    pos=[0, 0, config.WaterParams.SURFACE_Z], size=[50, 50, .1],
                                    rgba=[.06, .32, .45, .4], contype=0, conaffinity=0,
                                    group=0 if config.Groups.VISUAL is None else config.Groups.VISUAL)
            spec.worldbody.add_light(pos=[0, 0, 8], dir=[0, 0, -1], diffuse=[.7, .7, .7])

    def bind(self, model):
        """선체 복셀과 명시한 고정 연결 몸체의 합집합 복셀을 준비한다."""
        from sim.water.buoyancy import body_voxels
        import mujoco

        self.targets = []
        # 부가질량 (선체만). 꺼져 있으면 None. 켜져 있으면 bind 때 축별 부가질량을 한 번 계산
        self.added_mass_b = None
        self._prev_vel = None   # (시각, 월드 속도): 가속도 = 속도 차이 ÷ 시간 간격 을 구하려고 기억
        seen = set()
        # 로봇 담당 PR 머지 전에도 쓰던 값: 미현 테스트·뷰어가 읽는 배 번호·복셀·잠긴 부피
        self.body_id = None
        self.voxels = None
        self.submerged_volume = 0.0
        if self.include_boat:
            def find(obj, name):
                # 이름이 틀리면 −1 이 조용히 들어가지 않도록 assert 로 바로 멈춘다
                i = mujoco.mj_name2id(model, obj, name)
                assert i >= 0, f"모델에 {name!r} 없음 — 로봇 build() 를 했는지, config.Names 와 같은지 확인"
                return i

            body_id = find(mujoco.mjtObj.mjOBJ_BODY, config.Names.BOAT_BODY)
            geom_id = find(mujoco.mjtObj.mjOBJ_GEOM, config.Names.HULL_GEOM)
            if model.geom_bodyid[geom_id] != body_id or int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError("선체 부력에는 boat body에 속한 box hull이 필요합니다.")
            counts = np.asarray(config.WaterParams.VOXEL_RES, dtype=int)
            if counts.shape != (3,) or np.any(counts < 1):
                raise ValueError("VOXEL_RES는 양의 정수 3개여야 합니다.")
            # hull geom의 회전은 지원하지 않는다 (위치는 body 좌표로 반영).
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
            if not np.allclose(rotation.reshape(3, 3), np.eye(3)):
                raise ValueError("현재 선체 부력은 body 축과 정렬된 box hull만 지원합니다.")
            size = 2 * model.geom_size[geom_id]
            # 선체 상자를 VOXEL_RES 칸으로 자른 뒤 (make_voxels: 중심, 칸 크기, 칸 부피)
            # 선체 모양(config.hull_contains: A형 홈·뱃머리 양옆) 밖의 칸을 버린다
            self.body_id = body_id
            self.voxels = make_hull_voxels(size=size)
            centers_b, cell_size, cell_volume = self.voxels
            voxels = (centers_b + model.geom_pos[geom_id], cell_size, cell_volume)
            mass = model.body_subtreemass[body_id]
            coeffs = boat_drag_coefficients(mass, model.body_inertia[body_id], size,
                                            float(np.linalg.norm(model.opt.gravity)))
            self.targets.append(dict(body_id=body_id, voxels=voxels, coeffs=coeffs))
            seen.add(body_id)
            if config.WaterParams.ADDED_MASS_ENABLED:
                self.added_mass_b = added_mass(mass)   # heave 값이 있거나 계수 ≥ 1 이면 여기서 에러
        for name in self.floating_bodies:
            extra_id = model.body(name).id
            if extra_id in seen or extra_id == 0:
                raise ValueError("부력 대상 몸체가 중복되거나 world입니다.")
            # 부모와 자식을 동시에 지정하면 같은 부피에 힘을 두 번 넣게 된다.
            for other in seen:
                for child, ancestor in ((extra_id, other), (other, extra_id)):
                    while child and child != ancestor:
                        child = int(model.body_parentid[child])
                    if child == ancestor:
                        raise ValueError("부력 대상은 서로 독립된 몸체여야 합니다.")
            seen.add(extra_id)
            points, half, dimensions = body_voxels(model, extra_id, config.RescueWater.VOXEL_SIZE)
            voxels = (points, 2 * half, float(np.prod(2 * half)))
            coeffs = drag_coefficients(config.RescueWater.DRAG_LINEAR, config.RescueWater.ROT_DRAG_LINEAR,
                                       config.RescueWater.DRAG_CD, config.RescueWater.ROT_DRAG_QUAD,
                                       dimensions)
            self.targets.append(dict(body_id=extra_id, voxels=voxels, coeffs=coeffs))

    def add_forces(self, model, data, acc):
        """부력은 잠긴 복셀 중심에, 저항은 현재 물 상대 속도에 적용한다.

        계산은 모듈 함수(voxel_buoyancy, wetted_drag, water_drag, added_mass_force_w)로 한다
        → 단위 테스트가 실제 경로를 검증.
        파도는 수면 높이만 바꾸며 물 입자의 파도 속도는 아직 계산하지 않는다.
        """
        accel_w = self._boat_accel_w(model, data)
        if model.opt.gravity[0] != 0 or model.opt.gravity[1] != 0 or model.opt.gravity[2] > 0:
            raise ValueError("물 모듈은 월드 -z 방향 중력만 지원합니다.")
        gravity = float(np.linalg.norm(model.opt.gravity))
        current = np.asarray(config.WaterParams.WATER_VELOCITY_W, dtype=float)
        if current.shape != (3,) or not np.isfinite(current).all():
            raise ValueError("물 속도는 유한한 월드 벡터여야 합니다.")
        self.submerged_volume = 0.0
        for target in self.targets:
            body_id = target['body_id']
            forces_w, points_w, volume = voxel_buoyancy(data, body_id, waves=self.waves,
                                                        voxels=target['voxels'], gravity=gravity)
            if volume == 0:
                continue
            if body_id == self.body_id:
                self.submerged_volume = volume  # 뷰어·디버깅용 (배의 잠긴 부피)
            # 칸별 부력을 합력 하나로: 크기 = 합, 작용점 = 부력 가중 평균 (토크가 같음)
            #   중력 0 이면 부력도 0 → 작용점을 구할 수 없으니 건너뛰고 저항만 건다
            lift = forces_w[:, 2]
            center_w = None
            if lift.sum() > 0:
                center_w = (points_w * lift[:, None]).sum(axis=0) / lift.sum()
                acc.add(body_id, [0.0, 0.0, lift.sum()], point_w=center_w)

            # 평형 잠김량에서는 감쇠비 계약값이 그대로 적용되고, 물 밖에서는 사라진다.
            wet_ratio = volume / (len(points_w) * target['voxels'][2])
            damping_scale = min(1.0, volume * config.Physics.RHO_WATER / model.body_subtreemass[body_id])
            coeffs = wetted_drag(target['coeffs'], wet_ratio, damping_scale)
            force_w, torque_w = water_drag(model, data, body_id, coeffs,
                                           heave_world_only=(body_id == self.body_id))
            # 저항은 물에 잠긴 부분이 받으므로 잠긴 부분의 가운데(= 부력 중심)에 건다 (2026-11-20).
            #   무게중심에 걸면, 무게중심보다 아래(선체 바닥)를 미는 추진기가 선수를 드는 돌림힘을
            #   상쇄하는 힘이 없어 고속에서 선수가 계속 들렸다. 중력 0 등으로 부력 중심이 없으면 무게중심.
            acc.add(body_id, force_w, torque_w=torque_w, point_w=center_w)

            # 활주 양력은 배에만 (Savitsky 식은 배 바닥 같은 판 모양 기준. 사람·구명환에는 맞지 않음)
            if body_id == self.body_id:
                vel_w, _ = frames.body_velocity_w(model, data, body_id)
                vel_b = frames.world_to_body(data, body_id, vel_w - current)
                lift_w, lift_point_w = planing_lift(data, body_id, target['voxels'], forces_w, gravity, vel_b)
                if lift_w is not None:
                    acc.add(body_id, lift_w, point_w=lift_point_w)

            # 부가질량도 감쇠처럼 물 밖으로 나오면 줄어든다 (같이 밀 물이 없으므로)
            if body_id == self.body_id and accel_w is not None:
                acc.add(body_id, damping_scale * added_mass_force_w(data, body_id, accel_w, self.added_mass_b))

    def _boat_accel_w(self, model, data):
        """배 무게중심의 가속도 (월드 좌표) = (지금 속도 − 직전 스텝 속도) ÷ 시간 간격.

        부가질량이 꺼져 있으면 None.
        직전 기록이 없거나(첫 스텝) 시간이 한 칸만큼 흐르지 않았으면(리셋 등) None
        → 가짜로 큰 가속도가 생겨 배가 튕기는 것을 막는다.
        해류는 일정하므로 물에 대한 가속도 = 배 가속도.
        """
        if self.added_mass_b is None:
            return None
        vel_w, _ = frames.body_velocity_w(model, data, self.body_id)
        prev, self._prev_vel = self._prev_vel, (data.time, vel_w.copy())
        if prev is None or not np.isclose(data.time - prev[0], model.opt.timestep):
            return None
        return (vel_w - prev[1]) / model.opt.timestep
