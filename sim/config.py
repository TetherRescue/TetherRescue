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
class Expect:
    # 정수 흘수 [m] = MASS / (RHO_WATER · LENGTH · BEAM)  (직육면체 손 계산)
    DRAFT = None
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
