"""🔒 계약: 이름·좌표계·단위·dt·geom 그룹 상수.

담당: 섹션마다 그 값에 걸린 담당자가 정하고 씁니다 (물·로봇 섹션: 동경·미현).
      가영은 통합 충돌 확인 — 이름 겹침, 공통 값(timestep·그룹)이 모든 모듈과 맞는지.
모든 모듈은 이름·번호를 여기 상수로만 씁니다. 문자열을 코드에 직접 쓰지 않습니다.
수정 PR에는 가영 + 그 섹션 담당자를 태그합니다. 규칙은 노션 06 문서 참고.

현재 상태: 물(미현) ↔ 로봇(동경) 합의 항목 초안.
값은 전부 None 이고, 합의가 끝나면 채웁니다. 주석의 "PoC" 값은 tether_rescue_test 저장소에서
쓰던 값으로, 출발점 제안일 뿐 이 저장소에서 검증된 값이 아닙니다.
단위는 SI 고정 (m, kg, s, N, rad).
"""

import math


# ─────────────────────────────────────────────────────────────
# 1. 이름
# ─────────────────────────────────────────────────────────────
class Names:
    # 선체 body. 물은 이 body 에 부력·저항을 건다.
    # 후보: "boat"(PoC) / "robot"(노션 초안) — 하나로 통일
    BOAT_BODY = None
    # 선체 freejoint. PoC: "boat_free"
    BOAT_FREE = None
    # 부력 계산에 쓰는 선체 geom. 물이 이 모양을 복셀로 잘라 잠긴 부피를 계산한다.
    # PoC: "hull" (겉모양 geom 은 따로 있고 질량·충돌 없음)
    HULL_GEOM = None
    # 추진기 위치 site (좌·우). 추력 작용점이자, 추진기가 물에 잠겼는지 보는 점.
    # PoC: "thr_l", "thr_r"
    THRUSTER_L_SITE = None
    THRUSTER_R_SITE = None
    # 선외기 body — 선외기 질량을 따로 둘 때만 필요 (Hull.OUTBOARD_SEPARATE 참고)
    OUTBOARD_BODY = None
    # 줄 고정점 site (로봇 ↔ 가영 tether.py). PoC: "tow_point" (선미)
    TOW_POINT_SITE = None


# ─────────────────────────────────────────────────────────────
# 2. 좌표계와 기준점
# ─────────────────────────────────────────────────────────────
class Frames:
    # 월드 축. 노션 초안: "ENU" (x 동, y 북, z 위), 수면 z=0
    WORLD = None
    # 몸체 축. 노션 초안: "FLU" (x 앞=선수, y 왼쪽, z 위)
    BODY = None
    # 몸체 원점 위치. 물은 "선체 바닥 = 원점 − 높이/2" 처럼 이 기준으로 계산하므로 꼭 맞춰야 함.
    # PoC: 선체 박스 중심
    BODY_ORIGIN = None
    # 힘을 거는 기준점: 몸체 원점 or 무게중심. PoC: 무게중심(data.xipos)
    FORCE_REFERENCE = None
    # 속도는 공통 헬퍼 하나로만 구한다 (이름·위치 합의 필요).
    # ⚠️ mj_objectVelocity(..., flg_local=1) 결과는 몸체 좌표가 아님 (PoC 발산 원인).
    #    월드 속도를 받아 회전행렬로 직접 몸체 좌표로 돌린다.
    VELOCITY_HELPER = None


# ─────────────────────────────────────────────────────────────
# 3. 선체 (값은 로봇이 정해서 물에 넘김)
# ─────────────────────────────────────────────────────────────
class Hull:
    # 크기 [m]: 길이(x)·폭(y)·높이(z). xml geom 과 부력 계산이 둘 다 이 값만 읽는다
    # (PoC 는 두 곳에 따로 적고 시작 시 비교했음 → 여기 한 곳으로).
    LENGTH = None
    BEAM = None
    HEIGHT = None
    # 부력용 모양. PoC: 직육면체 근사 (뾰족한 선수·공기 튜브는 그림일 뿐 부력에 안 들어감)
    SHAPE = None
    # 질량 [kg] — 소유자: 로봇. 물은 질량을 건드리지 않는다.
    MASS = None
    # 무게중심 (몸체 좌표) [m]. PoC: 중심보다 약간 아래 (장비 때문)
    COM = None
    # 관성 대각 성분 [kg·m²]
    INERTIA = None
    # 선외기 질량을 별도 body 로 둘지 (True/False).
    # True 면 MASS 에서 선외기 질량을 빼야 함 — 두 번 세기 가장 쉬운 지점.
    # PoC: 별도 body 없음 (MASS 에 포함된 셈)
    OUTBOARD_SEPARATE = None
    OUTBOARD_MASS = None


# ─────────────────────────────────────────────────────────────
# 4. 추진기 (로봇)
# ─────────────────────────────────────────────────────────────
class Thruster:
    # 추력을 넣는 방법: "acc" (ForceAccumulator.add, point_w=추진기 위치)
    #                  or "actuator" (MuJoCo motor → data.ctrl, PoC 방식)
    # 노션 계약은 "모든 힘은 ForceAccumulator" 라서 actuator 를 쓰면 예외로 적어 둬야 함.
    METHOD = None
    # 명령 범위와 단위. PoC: [-1, 1] 정규화 → 최대 전진/후진 추력 [N] 로 변환 (태수 PhaseManager 와 합의)
    CMD_RANGE = None
    MAX_FORWARD = None
    MAX_REVERSE = None
    # 조향 방식. PoC: 좌우 추력 차이 (differential thrust)
    STEERING = None
    # 추진기가 이만큼 [m] 잠겨야 추력 100%. 파도로 물 밖에 나오면 추력 감소.
    # 물의 수면 높이 함수가 필요함 (WaterAPI 참고). 1주차에 넣을지 합의.
    FULL_THRUST_DEPTH = None


# ─────────────────────────────────────────────────────────────
# 5. 물리 상수 (둘이 같은 값을 써야 함)
# ─────────────────────────────────────────────────────────────
class Physics:
    # 물 밀도 [kg/m³]. PoC: 바닷물
    RHO_WATER = None
    # 중력 [m/s²]. MuJoCo option gravity 와 물 코드의 g 가 반드시 같아야 함 → 이 값 하나만 사용
    GRAVITY = None


# ─────────────────────────────────────────────────────────────
# 5-1. 물 모듈 내부 설정 (미현) — 합의 안건 아님, 제안값을 채워 둠
# ─────────────────────────────────────────────────────────────
# 물 밀도와 중력은 여기서 따로 만들지 않고 Physics.RHO_WATER, Physics.GRAVITY 를 쓴다.
# (같은 값을 두 곳에 적으면 한쪽만 바뀌는 사고가 나기 쉬움)
class Water:
    # 파도가 없을 때 수면의 높이 [m]. 월드 z=0 이 수면.                     [미현 / 제안]
    SURFACE_Z = 0.0

    # 파도 = 여러 개의 사인파(물결)를 겹친 것.
    # 성분 하나 = (진폭 m, 파장 m, 진행 방향 rad, 위상 rad)
    #   진폭: 물결이 수면에서 위아래로 움직이는 높이
    #   파장: 물결 마루에서 다음 마루까지의 거리
    #   진행 방향: 물결이 나아가는 방향 (0 = 월드 +x 쪽)
    #   위상: 물결의 시작 위치를 살짝 어긋나게 해서 성분끼리 마루가 겹치지 않게 함
    # 물결이 얼마나 빨리 오르내리는지(각진동수)는 따로 적지 않는다.
    # water.py 가 "깊은 물" 공식 omega = sqrt(g · 2π / 파장) 으로 계산한다.
    WAVES_CALM = ()                                                         # [미현 / 제안] 잔잔한 물 (테스트용)
    WAVES_DEFAULT = (                                                       # [미현 / 제안] 기본 파도: 사인파 3개 합
        (0.10, 8.0, 0.0, 0.0),
        (0.05, 4.0, math.radians(30), 1.3),
        (0.02, 2.0, math.radians(-45), 2.1),
    )

    # 부력 계산용 복셀(작은 상자) 개수: 선체를 몸체 x·y·z 방향으로 각각 몇 칸으로 자를지.
    # 각 상자가 물에 얼마나 잠겼는지 더해서 부력을 구한다. 칸이 많을수록 정확하지만 느려진다.
    # water.py 가 시작(bind) 때 Names.HULL_GEOM 의 크기를 읽어 상자를 직접 만든다.  [미현 / 제안]
    VOXEL_RES = (8, 4, 4)
    # 선체를 속이 꽉 찬 덩어리로 본다 (갑판 위로 물이 들어차는 침수는 계산하지 않음).  [미현 / 제안]
    HULL_IS_SOLID = True

    # 물 자체가 흐르는 속도 (월드 좌표) [m/s].
    # 1주차는 해류 없음. 해류를 누가 넣을지는 Owner.CURRENT 합의 후 결정.          [미현 / 제안]
    WATER_VELOCITY_W = (0.0, 0.0, 0.0)

    # ── 물 저항 (배가 물을 헤치고 움직일 때 받는 브레이크 힘) ──
    # 순서: (surge 앞뒤, sway 좌우, heave 위아래)
    # 속도 제곱에 비례하는 저항 계수: F = -0.5 · rho · Cd · A · |v| · v
    #   (A = 그 방향에서 본 선체 단면적, v = 물에 대한 상대속도)           [미현 / 제안]
    DRAG_CD = (0.6, 1.2, 1.5)
    # 속도에 그냥 비례하는 저항 계수 [N·s/m]: 느리게 움직일 때 배를 멈추게 해 줌.
    # heave(위아래)는 아래 HEAVE_DAMPING_RATIO 로 자동 계산하므로 None.  [미현 / 제안]
    DRAG_LINEAR = (5.0, 15.0, None)
    # 위아래 출렁임을 얼마나 빨리 가라앉힐지 (감쇠비, 0~1 사이).
    #   0 에 가까우면 오래 출렁이고, 1 에 가까우면 바로 멈춘다.
    # water.py 계산식: c = 2 · zeta · sqrt(m · k),  k = rho · g · L · B
    #   (m 은 MuJoCo 의 model.body_subtreemass 에서 읽음 → 질량이 바뀌어도 자동 반영)
    # PoC 에서는 이 값을 작게 잡았다가 파도와 박자가 맞아 출렁임이 계속 커졌음(공진). [미현 / 제안]
    HEAVE_DAMPING_RATIO = 0.3
    # 회전 저항: 배가 기울거나 도는 것을 막는 힘. 순서: (roll 좌우 기울기, pitch 앞뒤 기울기, yaw 방향 돌기)
    ROT_DRAG_LINEAR = (8.0, 8.0, 6.0)                                       # [미현 / 제안] 회전 속도에 비례
    ROT_DRAG_QUAD = (2.0, 2.0, 4.0)                                         # [미현 / 제안] 회전 속도 제곱에 비례

    # 부가질량: 배가 가속할 때 주변 물도 같이 밀어야 해서 실제보다 무겁게 느껴지는 효과.
    # 1주차는 끔, 2주차에 켤 수 있음. 계수 순서는 (surge, sway, heave).     [미현 / 제안]
    ADDED_MASS_ENABLED = False
    ADDED_MASS_COEF = (0.05, 0.5, 1.0)

    # 물 모듈이 제대로 돌기 위한 조건. max_dt = 허용하는 가장 긴 시뮬레이션 한 칸 시간 [s].
    # Sim.TIMESTEP 합의 결과에 맞춰 바꿀 것.                                 [미현 / 제안]
    REQUIRES = {"max_dt": 0.004}


# ─────────────────────────────────────────────────────────────
# 6. 소유권 — 같은 힘을 두 번 넣지 않기 위한 약속
# ─────────────────────────────────────────────────────────────
# 확정 (노션 06):
#   선체 질량·관성 → 로봇      부력 → 물        선체 물 저항 → 물
#   추력           → 로봇      부가질량 → 물    (로봇 질량에 미리 더하지 않음)
#   로봇은 추진 모델에 저항 항을 넣지 않고, 물은 프로펠러 효과를 넣지 않는다.
class Owner:
    # 노션 표에 없는 회색 지대: "water" or "robot"
    # 바람 힘 (물 위로 나온 선체가 받는 힘). PoC: 로봇(boat.py)이 넣었음
    WIND = None
    # 해류 (항력 계산의 상대속도에 반영). PoC: 물(hull.py)
    CURRENT = None
    # 저항 계수 값을 정하는 사람 (계산은 물, 계수는 선체 모양에 따라 달라짐)
    DRAG_COEFFS = None


# ─────────────────────────────────────────────────────────────
# 7. 물 ↔ 로봇 주고받는 함수 (형태만 합의, 구현은 water.py)
# ─────────────────────────────────────────────────────────────
class WaterAPI:
    # 수면 높이 함수 형태. 후보: wave_height(x, y, t) -> z
    # 여러 점을 한 번에 받는 배열 입력으로 할지 (PoC: 호출 횟수가 점 개수보다 비쌌음)
    WAVE_HEIGHT_VECTORIZED = None
    # 스텝 순서가 water → robot 이므로, 물이 계산한 값(잠긴 비율, 추진기 위치 수면 높이)을
    # 로봇이 같은 스텝에서 어디서 읽을지. 후보: Water 객체 속성 / 공유 dict
    SHARED_STATE = None
    # 1주차 범위: False 면 1주차엔 "추진기는 항상 물속" 으로 가정
    USE_IN_WEEK1 = None


# ─────────────────────────────────────────────────────────────
# 8. 시뮬레이션 설정 (world.py 에서만 적용)
# ─────────────────────────────────────────────────────────────
class Sim:
    # ⚠️ PoC 는 500 Hz 로 돌렸고 노션 초안은 250 Hz — 다르다. 각 모듈 REQUIRES["max_dt"] 와 맞출 것
    TIMESTEP = None
    # PoC·노션 초안: "implicitfast"
    INTEGRATOR = None
    # 제어·기록 주기 [Hz]
    CONTROL_HZ = None
    # MuJoCo 내장 유체 모델(option density/viscosity)은 반드시 끔 → 켜면 저항이 두 번 들어감.
    # world.py 시작 시 assert
    BUILTIN_FLUID_OFF = True


# ─────────────────────────────────────────────────────────────
# 9. Geom 그룹·충돌 (가영 표와 맞춤)
# ─────────────────────────────────────────────────────────────
class Groups:
    VISUAL = None    # 시각용, 충돌 없음 (노션 초안: 0)
    PHYSICS = None   # 물리 충돌용 — 선체 geom 은 여기 (사람·해저와 부딪힘) (노션 초안: 3)
    SENSOR = None    # 센서 전용 (노션 초안: 4)


# ─────────────────────────────────────────────────────────────
# 10. 테스트 정답값 (물·로봇이 둘 다 있어야 의미 있는 테스트)
# ─────────────────────────────────────────────────────────────
def _divide_or_none(numerator, *denominators):
    """나눗셈 도우미: 값 중 하나라도 아직 None(합의 전)이면 계산하지 않고 None 을 돌려준다.
    그냥 나누면 None 때문에 import 하는 순간 에러가 나기 때문."""
    if numerator is None or any(d is None for d in denominators):
        return None
    result = numerator
    for d in denominators:
        result /= d
    return result


class Expect:
    # 정수 흘수 [m] = MASS / (RHO_WATER · LENGTH · BEAM)  (직육면체 손 계산)
    # 흘수 = 가만히 떠 있을 때 배가 물에 잠기는 깊이.
    # 숫자를 직접 적지 않고 계산식으로 둔다 → 선체 질량·크기가 바뀌어도 정답값이 자동으로 따라감.
    # 필요한 값이 하나라도 아직 None 이면 DRAFT 도 None.
    DRAFT = _divide_or_none(Hull.MASS, Physics.RHO_WATER, Hull.LENGTH, Hull.BEAM)
    DRAFT_TOL = None          # 흘수 허용 오차
    # 평형 자세 (롤·피치) 와 허용 오차
    EQUILIBRIUM_TILT_TOL = None
    # 잔잔한 물 표류: DRIFT_SECONDS 동안 수평 이동 DRIFT_MAX 미만 (노션 초안 있음)
    DRIFT_SECONDS = None
    DRIFT_MAX = None
    # 총질량 테스트: model.body_subtreemass 가 문서 값과 이 비율 안에서 일치 (노션 초안 있음)
    MASS_TOL = None
    # 방향 테스트: 몸체를 90° 돌리고 "앞으로" → 월드 +y 가속 (로봇 1주차 완료 기준, 값 없음)


# ─────────────────────────────────────────────────────────────
# 11. 로봇 ↔ 다른 담당 (미현 님과는 아니지만 로봇이 걸친 것)
# ─────────────────────────────────────────────────────────────
class Capsize:
    # 전복 판정 기준각 (롤·피치) [rad] — 로봇이 판정, 이후 처리는 태수 PhaseManager
    ROLL_LIMIT = None
    PITCH_LIMIT = None
    # 전복 시 처리: 에피소드 종료 / 안전 모드 등 (태수와 합의)
    ON_CAPSIZE = None


# ─────────────────────────────────────────────────────────────
# 12. 자동 검사 — import 할 때 숫자가 말이 되는지 바로 확인
# ─────────────────────────────────────────────────────────────
# 잘못된 숫자로 시뮬레이션이 "조용히 틀리게" 도는 대신, 시작하자마자 에러를 낸다.
# 아직 None(합의 전)인 값은 그 검사만 건너뛴다 → 합의 전에도 import 는 된다.
def _check_water_robot():
    # 흘수 vs 선체 높이
    if Expect.DRAFT is not None and Hull.HEIGHT is not None:
        if Expect.DRAFT >= Hull.HEIGHT:
            raise ValueError(
                f"흘수({Expect.DRAFT:.3f} m)가 선체 높이({Hull.HEIGHT} m) 이상 → 배가 가라앉음. "
                "Hull.MASS 를 줄이거나 선체 크기를 키울 것."
            )
        if Expect.DRAFT <= 0.1 * Hull.HEIGHT:
            raise ValueError(
                f"흘수({Expect.DRAFT:.3f} m)가 선체 높이의 10% 이하 → 너무 얕게 떠서 불안정. "
                "Hull.MASS·크기를 확인할 것."
            )

    # 저항 계수는 (앞뒤, 좌우, 위아래) 또는 (roll, pitch, yaw) 세 개씩
    for name in ("DRAG_CD", "DRAG_LINEAR", "ROT_DRAG_LINEAR", "ROT_DRAG_QUAD"):
        value = getattr(Water, name)
        if len(value) != 3:
            raise ValueError(f"Water.{name} 은 값이 3개여야 함 (지금 {len(value)}개).")

    # 복셀은 각 방향 최소 1칸
    if any(n < 1 for n in Water.VOXEL_RES):
        raise ValueError(f"Water.VOXEL_RES 의 각 값은 1 이상이어야 함 (지금 {Water.VOXEL_RES}).")

    # 감쇠비는 0 과 1 사이
    if not 0 < Water.HEAVE_DAMPING_RATIO < 1:
        raise ValueError(
            f"Water.HEAVE_DAMPING_RATIO 는 0 과 1 사이여야 함 (지금 {Water.HEAVE_DAMPING_RATIO})."
        )

    # 파도 성분: 진폭·파장은 양수, 너무 가파르면(진폭/파장 ≥ 1/7) 파도가 부서짐(쇄파)
    # → 매끈한 사인파로 계산하는 이 모델이 맞지 않는 범위
    for waves_name in ("WAVES_CALM", "WAVES_DEFAULT"):
        for i, (amp, length, _direction, _phase) in enumerate(getattr(Water, waves_name)):
            where = f"Water.{waves_name}[{i}]"
            if amp <= 0:
                raise ValueError(f"{where}: 진폭은 0 보다 커야 함 (지금 {amp}).")
            if length <= 0:
                raise ValueError(f"{where}: 파장은 0 보다 커야 함 (지금 {length}).")
            if amp / length >= 1 / 7:
                raise ValueError(
                    f"{where}: 진폭/파장 = {amp / length:.3f} ≥ 1/7 → 파도가 부서지는 범위라 "
                    "사인파 모델로 계산할 수 없음."
                )

    # 중력은 채워졌다면 양수 (부호는 아래 방향으로 water.py 가 처리)
    if Physics.GRAVITY is not None and Physics.GRAVITY <= 0:
        raise ValueError(f"Physics.GRAVITY 는 0 보다 커야 함 (지금 {Physics.GRAVITY}).")


_check_water_robot()
