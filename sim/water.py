"""1 물: 파도·복셀 부력·저항·부가질량 (담당: 미현).

설정값은 전부 config.WaterParams(물 설정)·config.Physics(밀도·중력)·config.Hull(선체)에서 읽는다.
이 파일에는 숫자를 직접 적지 않는다.
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


def voxel_buoyancy(data, body_id, t=None, waves=config.WaterParams.WAVES_DEFAULT, voxels=None):
    """지금 MuJoCo 상태(data)에서 배의 칸마다 부력과 작용점 (월드 좌표).

    data    : MuJoCo MjData. 배의 현재 위치·기울기가 들어 있다
    body_id : 배 body 번호. bind 때 Names.BOAT_BODY 로 한 번 찾아 둔 값
    t       : 시각 [s] (파도 계산용). 안 주면 data.time (시뮬레이션 현재 시각)
    voxels  : make_voxels() 결과. 배 모양은 안 변하니 bind 때 한 번 만들어 넘기면 됨

    돌려주는 값
      forces_w : (칸 수, 3) 칸마다 부력. 위쪽(+z)만 있음
      points_w : (칸 수, 3) 칸마다 부력을 거는 점 = 그 칸에서 잠긴 부분의 가운데
      submerged_volume : 잠긴 부피 합 [m³]

    좌표 변환은 직접 계산하지 않고 전부 sim/frames.py 함수로 한다 (노션 06 문제 2, 동경 요청).
      각자 회전행렬을 곱하면 축이 어긋나도 오류 없이 조용히 틀리기 때문.
    """
    centers_b, cell_size, cell_volume = voxels if voxels is not None else make_voxels()
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
    forces_w[:, 2] = config.Physics.RHO_WATER * config.Physics.GRAVITY * volume

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


def heave_damping(mass):
    """위아래 출렁임을 멈추게 하는 선형 계수 c [N·s/m] (감쇠비 HEAVE_DAMPING_RATIO 에서 계산).

    배를 살짝 눌렀다 놓으면 용수철처럼 위아래로 출렁인다.
      용수철 세기 k = ρ · g · L · B   (1 m 더 잠기면 부력이 이만큼 늘어남 = 수면에 닿은 면적 × ρg)
      무게       m = 배 무게
    감쇠비 ζ 로 "얼마나 빨리 멎을지"를 정하면 필요한 계수는 c = 2 · ζ · √(m · k).
    → 배 무게·크기가 바뀌어도 출렁임이 멎는 빠르기는 그대로 유지된다.
    """
    k = config.Physics.RHO_WATER * config.Physics.GRAVITY * config.Hull.LENGTH * config.Hull.BEAM
    return 2.0 * config.WaterParams.HEAVE_DAMPING_RATIO * math.sqrt(mass * k)


def drag_coefficients(mass, inertia):
    """저항 계산에 쓰는 계수들을 한 번에 준비한다. 배 무게·모양은 안 변하니 bind 때 한 번만 부르면 됨.

    mass    = 배 무게 [kg]                (MuJoCo model.body_subtreemass)
    inertia = (roll, pitch, yaw) 관성 [kg·m²] (MuJoCo model.body_inertia)

    돌려주는 dict (모두 길이 3 배열, 축 순서는 위 설명 참고)
      lin      : 이동 선형 계수 c        [N·s/m]       → F = −lin · v
      quad     : 이동 2차 계수 ½·ρ·Cd·A  [N·s²/m²]     → F = −quad · |v| · v
      rot_lin  : 회전 선형 계수          [N·m·s/rad]   → τ = −rot_lin · ω
      rot_quad : 회전 2차 계수           [N·m·s²/rad²] → τ = −rot_quad · |ω| · ω
    """
    rho = config.Physics.RHO_WATER
    L, B = config.Hull.LENGTH, config.Hull.BEAM
    # 물에 잠긴 깊이: 가만히 떠 있을 때의 흘수를 쓴다 (출렁여도 단면적은 크게 안 변한다고 보는 단순화)
    draft = mass / (rho * L * B)

    # 각 방향에서 "물을 정면으로 미는 면적" A [m²]
    #   앞뒤로 갈 때: 배 앞면 중 물에 잠긴 부분 = 폭 × 흘수
    #   좌우로 갈 때: 배 옆면 중 물에 잠긴 부분 = 길이 × 흘수
    #   위아래로 갈 때: 배 바닥 = 길이 × 폭
    area = np.array([B * draft, L * draft, L * B])

    # 선형 계수: 앞뒤·좌우는 config 숫자, 위아래는 감쇠비로 계산 (config 에서 None 으로 비워 둔 칸)
    surge_lin, sway_lin, _ = config.WaterParams.DRAG_LINEAR
    lin = np.array([surge_lin, sway_lin, heave_damping(mass)])

    # 2차 계수: ½ · ρ · Cd · A  (공식에서 속도만 빼고 미리 곱해 둔 값)
    quad = 0.5 * rho * np.array(config.WaterParams.DRAG_CD) * area

    return {
        "lin": lin,
        "quad": quad,
        "rot_lin": np.array(rot_drag_linear(mass, inertia)),  # roll·pitch 는 감쇠비로 계산, yaw 는 숫자
        "rot_quad": np.array(config.WaterParams.ROT_DRAG_QUAD, dtype=float),
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


def water_drag(model, data, body_id, coeffs):
    """지금 MuJoCo 상태에서 배가 받는 물 저항 (월드 좌표). 힘은 무게중심에 작용한다.

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

    # 4) 축마다 저항
    force_b, torque_b = drag_b(rel_vel_b, omega_b, coeffs)

    # 5) 세상 기준으로 되돌리기
    force_w = frames.body_to_world(data, body_id, force_b)
    torque_w = frames.body_to_world(data, body_id, torque_b)
    return force_w, torque_w


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
        """세계 조립 때 자기 몸(body/geom 등)을 spec 에 추가한다.

        물은 MuJoCo 안에 "물체"로 존재하지 않는다 (수면은 wave_height 함수일 뿐).
        배(boat)와 선체 상자(hull)는 로봇(동경 robot.py)이 만든다. 그래서 여기서는 추가할 것이 없다.
        """

    def bind(self, model):
        """시작할 때 한 번: 배를 찾고, 매 스텝 쓸 재료(복셀·저항 계수)를 미리 만들어 둔다.

        이름 → 번호(id) 를 한 번만 찾는 이유: 매 스텝 이름으로 찾으면 느리고,
        이름이 틀리면 −1 이 조용히 들어가 엉뚱한 곳에 힘이 걸리기 때문 → 없으면 assert 로 바로 멈춘다.
        """
        import mujoco  # MuJoCo 없는 환경에서도 이 파일 import(스모크 테스트)는 되도록 여기서 가져옴

        def find(obj, name):
            i = mujoco.mj_name2id(model, obj, name)
            assert i >= 0, f"모델에 {name!r} 없음 — 로봇 build() 를 했는지, config.Names 와 같은지 확인"
            return i

        # 1) 배 몸체와 선체 상자의 번호
        self.body_id = find(mujoco.mjtObj.mjOBJ_BODY, config.Names.BOAT_BODY)
        hull_id = find(mujoco.mjtObj.mjOBJ_GEOM, config.Names.HULL_GEOM)

        # 2) 선체 상자 확인 — 복셀은 "몸체 원점 = 상자 중심"을 전제로 자른다 (Frames.BODY_ORIGIN)
        #    상자가 원점에서 벗어나 있거나 돌아가 있으면 부력이 엉뚱한 자리에 걸리므로 바로 멈춤
        assert model.geom_type[hull_id] == mujoco.mjtGeom.mjGEOM_BOX, "hull 이 상자(box)가 아님"
        assert np.allclose(model.geom_pos[hull_id], 0.0), "hull 상자 중심이 배 원점과 다름"
        assert np.allclose(model.geom_quat[hull_id], (1, 0, 0, 0)), "hull 상자가 배 기준으로 돌아가 있음"

        # 3) 선체 크기는 MuJoCo 모델에서 읽는다 (MuJoCo box 크기는 절반 길이 → ×2).
        #    저항 단면적은 config.Hull 값을 쓰므로 둘이 같은지도 확인 (다르면 조용히 틀림)
        size = 2.0 * model.geom_size[hull_id]
        hull_cfg = (config.Hull.LENGTH, config.Hull.BEAM, config.Hull.HEIGHT)
        assert np.allclose(size, hull_cfg), f"모델 선체 크기 {size} ≠ config.Hull {hull_cfg}"

        # 4) 배 무게·관성도 모델에서 읽는다 (물은 질량을 정하지 않음, 소유자 = 로봇)
        #    body_subtreemass = 이 몸체 + 그 아래 붙은 부품까지 합친 무게
        mass = float(model.body_subtreemass[self.body_id])
        inertia = model.body_inertia[self.body_id].copy()  # (roll, pitch, yaw) 축 관성

        # 5) 매 스텝 쓸 재료를 미리 만들기 (배 모양·무게는 안 변하니 한 번이면 충분)
        self.voxels = make_voxels(size=size)               # 128 칸의 위치·크기·부피
        self.drag_coeffs = drag_coefficients(mass, inertia)  # 저항 계수들

        # 디버깅·로봇 연동용으로 마지막 스텝 값을 남겨 둠 (add_forces 가 채움)
        self.submerged_volume = 0.0

    def add_forces(self, model, data, acc):
        """매 스텝(0.002 초마다): 부력과 물 저항을 acc(ForceAccumulator)에 넣는다. 전부 월드 좌표.

        넣기만 하고 MuJoCo 에 직접 쓰지 않는다 (xfrc_applied 는 forces.py 만 쓰는 규칙).
        """
        bid = self.body_id

        # ── 1) 부력 ──
        forces_w, points_w, self.submerged_volume = voxel_buoyancy(
            data, bid, waves=self.waves, voxels=self.voxels
        )
        total = forces_w[:, 2].sum()  # 128 칸 부력의 합 [N] (모두 위쪽)
        if total > 0.0:
            # 128 칸 힘을 하나씩 넣는 대신 "부력 중심" 한 점에 합친 힘 하나를 넣는다.
            #   부력 중심 = 칸마다의 작용점을 부력 크기로 가중 평균한 점
            # 모든 칸의 부력이 같은 방향(위)이라 이렇게 합쳐도 힘·토크가 정확히 같다 (평행한 힘의 합성).
            #   → 결과는 같고 acc 에 넣는 횟수만 128 번 → 1 번으로 줄어 빠름.
            # ⚠️ 1단계 설명의 "한 점 부력"(PoC 실패)과 다름: 그건 중심을 고정해서 기울어도 안 움직였고,
            #    이건 매 스텝 칸마다 계산한 결과라 배가 기울면 부력 중심이 잠긴 쪽으로 옮겨 가 배를 세운다.
            center_w = (forces_w[:, 2] @ points_w) / total
            acc.add(bid, (0.0, 0.0, total), point_w=center_w)

        # ── 2) 물 저항 (이동 + 회전) ──
        # 힘은 무게중심에 걸고(point_w 생략), 회전 저항은 순수 토크(torque_w)로 넣는다
        drag_force_w, drag_torque_w = water_drag(model, data, bid, self.drag_coeffs)
        acc.add(bid, drag_force_w, torque_w=drag_torque_w)
