"""🔒 계약: 이름·좌표계·단위·dt·geom 그룹 상수.

담당: 섹션마다 그 값에 걸린 담당자가 정하고 씁니다 (물·로봇 섹션: 동경·미현).
      가영은 통합 충돌 확인 — 이름 겹침, 공통 값(timestep·그룹)이 모든 모듈과 맞는지.
모든 모듈은 이름·번호를 여기 상수로만 씁니다. 문자열을 코드에 직접 쓰지 않습니다.
수정 PR에는 가영 + 그 섹션 담당자를 태그합니다. 규칙은 노션 06 문서 참고.

현재 상태: 물(미현) ↔ 로봇(동경) 계약 회의(2026-11-12) 진행 중.
값이 None 인 칸은 아직 합의 전입니다. 주석의 "PoC" 값은 tether_rescue_test 저장소에서
쓰던 값으로, 이 저장소에서 검증된 값이 아닙니다.
단위는 SI 고정 (m, kg, s, N, rad).

주석 읽는 법
  무엇: 이 상수가 뜻하는 것
  사용: 이 값을 읽는 코드 (누구 파일)
  이유: 이 값을 고른 근거. 실험으로 확인한 값인지, 관례·PoC 에서 가져온 값인지 구분
"""

import math


# ─────────────────────────────────────────────────────────────
# 1. 이름 — MuJoCo 모델 안 부품의 이름표
# ─────────────────────────────────────────────────────────────
# 합의: 2026-11-12 물↔로봇 계약 회의 (동경·미현). 이름은 PoC 와 같게 함 (검증 대상 아님, 이름일 뿐).
# 코드에서는 mj_name2id(model, ..., Names.XXX) 로 한 번만 id 를 찾아 두고 쓴다 (bind 단계).
class Names:
    # 무엇: 배(선체) 부품의 이름표.
    # 사용: 물(부력·저항을 이 배에 걸고, 배 무게 body_subtreemass 를 읽음), 로봇(추력을 이 배에 걸음)
    BOAT_BODY = "boat"
    # 무엇: 배에 붙은 자유 관절(freejoint)의 이름. free = 이동 3방향 + 회전 3방향이 모두 자유롭다는 뜻.
    #       이 관절이 없으면 배가 세상에 고정돼 움직이지 않는다.
    # 사용: 배를 특정 위치·방향에 놓거나 리셋할 때 (로봇 방향 테스트, 태수 reset, 채연 표류 테스트)
    BOAT_FREE = "boat_free"
    # 무엇: 부력 계산에 쓰는 선체 상자 모양의 이름. 화면에 보이는 겉모양 geom 과는 별개.
    # 사용: 물 — 이 상자 크기를 읽어 복셀(WaterParams.VOXEL_RES 칸)로 자르고 잠긴 부피를 더한다
    HULL_GEOM = "hull"
    # 무엇: 좌·우 추진기(프로펠러) 위치를 표시한 점(site).
    # 사용: 로봇 — 추력을 거는 작용점, 프로펠러가 물에 잠겼는지 보는 점
    THRUSTER_L_SITE = "thr_l"
    THRUSTER_R_SITE = "thr_r"
    # 무엇: 선외기(배 뒤 바깥에 달린 엔진+프로펠러) 부품의 이름.
    # 사용: Hull.OUTBOARD_SEPARATE = True 일 때만 (선외기를 별도 부품으로 둘 때).
    #       2026-11-12 회의에서 False 로 정해서 지금은 쓰지 않음 (이름 자리만 남겨 둠)
    OUTBOARD_BODY = "outboard"
    # 무엇: 줄(tether)을 배에 묶는 점. PoC 에서는 선미(배 뒤쪽).
    # 사용: 가영 tether.py — 줄의 한쪽 끝을 이 점에 연결
    TOW_POINT_SITE = "tow_point"

    # 가영 담당 이름 초안 — 기존 이름과 겹치지 않게 정함. 팀에 공유할 이름이며,
    # 아래 상수를 추가하는 것만으로 실제 몸체·관절·줄이 생성되지는 않는다.
    HUMAN_BODY = "human"                 # 사람의 움직임·질량을 갖는 body
    HUMAN_FREE = "human_free"            # 사람이 자유롭게 이동·회전하는 관절
    HUMAN_GEOM = "human_geom"            # 사람 몸의 모양
    HUMAN_HEAD_GEOM = "human_head"
    HUMAN_ARM_L_GEOM = "human_arm_l"
    HUMAN_ARM_R_GEOM = "human_arm_r"
    HUMAN_LEG_L_GEOM = "human_leg_l"
    HUMAN_LEG_R_GEOM = "human_leg_r"
    HUMAN_TOW_SITE = "human_tow_point"    # 이전 초안의 예약 이름. 실제 로프는 구명환에 연결
    TETHER_TENDON = "tow_tendon"          # 배와 구명환을 연결할 tendon
    LIFEBUOY_BODY = "lifebuoy"
    LIFEBUOY_FREE = "lifebuoy_free"       # 구명환 단독 시험 때만 사용하는 자유 관절
    LIFEBUOY_GEOM_PREFIX = "lifebuoy_segment"
    LIFEBUOY_TOW_SITE = "lifebuoy_tow_point"


# ─────────────────────────────────────────────────────────────
# 2. 좌표계와 기준점 — "방향"과 "기준점"을 모두 같은 뜻으로 쓰기 위한 약속
# ─────────────────────────────────────────────────────────────
# 합의: 2026-11-12 물↔로봇 계약 회의 (동경·미현).
class Frames:
    # 무엇: 세상(월드)의 축 방향 = 지도의 동서남북. x 동, y 북, z 위(하늘). 수면은 z=0.
    # 사용: 전원 (물: 수면 높이·파도 진행 방향 기준, 로봇: 배 위치)
    # 이유: MuJoCo 와 ROS 모두 z 가 위라 변환이 필요 없음 (관례)
    WORLD = "ENU"
    # 무엇: 배 기준 축 방향 = 배에 탄 사람 기준 앞·왼쪽·위. x 앞(선수), y 왼쪽, z 위.
    # 사용: 로봇("앞으로" 추력 = 몸체 +x), 물(저항 계수 순서 앞뒤·좌우·위아래가 이 축 기준)
    # 이유: ROS 표준(REP-103)과 같음 (관례)
    BODY = "FLU"
    # 무엇: 배 위치를 말할 때 찍는 기준점 = 선체 상자의 정가운데.
    # 사용: 물 — "선체 바닥 = 원점 − Hull.HEIGHT/2" 로 복셀을 배치. 무게중심은 Hull.COM 으로 따로 둔다
    # 이유: PoC 와 같음. MuJoCo box 모양도 중심 기준이라 추가 계산이 없음
    BODY_ORIGIN = "hull_center"
    # 무엇: 힘을 거는 위치 = 실제로 미는 점. acc.add(body_id, force_w, point_w=작용점_w)
    #       같은 힘도 미는 위치에 따라 배가 기우는 정도가 달라진다 (뒤쪽 아래를 밀면 앞이 들림).
    # 사용: 물(복셀마다 그 상자 중심에 부력), 로봇(추진기 site 에 추력). 토크 변환은 ForceAccumulator(가영)
    # 이유: PoC 도 추진기는 site 에, 부력은 기둥마다 작용점 기준 토크를 계산했음 (효과 같음)
    FORCE_REFERENCE = "point"
    # 무엇: 배 속도를 구하는 공용 함수 = 둘이 같은 속도계를 쓰기.
    # 사용: 물(저항 계산용 "물에 대한 배 속도"), 로봇(상태 확인). 구현 담당은 회의 ③에서 정함
    # 이유: 각자 만들면 한쪽만 틀려도 서로 다른 속도를 봄.
    # ⚠️ mj_objectVelocity(..., flg_local=1) 결과는 몸체 좌표가 아님 (PoC 발산 원인).
    #    월드 속도를 받아 회전행렬로 직접 몸체 좌표로 돌린다.
    VELOCITY_HELPER = "frames.body_velocity_w"


# ─────────────────────────────────────────────────────────────
# 3. 선체 — 값은 로봇이 정해서 물에 넘김
# ─────────────────────────────────────────────────────────────
# 합의: 2026-11-12 물↔로봇 계약 회의 (동경·미현). 값은 PoC 그대로 — 이 저장소에서는 아직 미검증,
# 물의 흘수 테스트(Expect.DRAFT ≈ 0.102 m)가 첫 검증.
class Hull:
    # 무엇: 배 상자의 길이(x, 앞뒤)·폭(y, 좌우)·높이(z, 위아래) [m].
    # 사용: 로봇(xml 선체 geom 크기), 물(복셀 크기, 흘수 계산). 두 곳 모두 이 값만 읽는다
    #       (PoC 는 두 곳에 따로 적고 시작 시 비교했음 → 여기 한 곳으로).
    # 이유: PoC 에서 띄워 본 소형 구조정(RIB) 크기
    LENGTH = 2.4
    BEAM = 1.0
    HEIGHT = 0.4
    # 무엇: 부력 계산에 쓰는 선체 모양. "box" = 직육면체로 근사.
    # 사용: 물(복셀이 상자 기준), 테스트(흘수를 곱셈·나눗셈만으로 손 계산 가능)
    # 이유: 뾰족한 선수·공기 튜브는 그림일 뿐 부력에 안 들어감. 단순하고 정답을 손으로 구할 수 있음
    SHAPE = "box"
    # 무엇: 배 무게 [kg]. 소유자 = 로봇. 물은 질량을 건드리지 않는다.
    # 사용: 로봇(xml inertial), 물(흘수·위아래 감쇠 — 단 물은 MuJoCo 모델에서 직접 읽음)
    # 이유: PoC 값. 흘수가 선체 높이의 약 1/4 이 되는 무게
    MASS = 250.0
    # 무게중심 (몸체 좌표) [m] = 원점(상자 중심)에서 얼마나 떨어져 있나. 지금은 5 cm 아래.
    # 사용: 로봇(xml inertial pos). 무게중심이 낮을수록 잘 안 뒤집힌다
    # 이유: PoC 값. 장비가 바닥에 있다고 가정 (실측 아님)
    COM = (0.0, 0.0, -0.05)
    # 무엇: 관성 = 돌리기 어려운 정도 [kg·m²]. 순서 (롤 좌우 기울기, 피치 앞뒤 기울기, 요 방향 돌기).
    # 사용: 로봇(xml inertial). 물은 쓰지 않음
    # 이유: MASS 짜리 균일 상자 공식 m/12·(b²+c²) 로 계산한 값 (검산함). COM 을 내린 효과는 반영 안 함 (차이 작음)
    INERTIA = (24.2, 123.0, 140.8)
    # 무엇: 선외기 무게를 별도 부품으로 둘지 (True / False).
    #   False = 선외기 무게를 MASS 에 포함, 선외기는 그림으로만 (PoC 방식)
    #   True  = 선외기 부품(Names.OUTBOARD_BODY)에 OUTBOARD_MASS 를 주고, 그만큼 MASS 에서 뺀다
    # 사용: 로봇(xml), 물(Expect.DRAFT 계산식 — True 면 선외기 무게도 더해야 함)
    # 합의: 2026-11-12 회의 — False (선외기를 별도 부품으로 두지 않음).
    # 이유: 1주차 조향은 좌우 추력 차이라 선외기가 따로 움직일 필요가 없고, 무게를 두 번 셀 위험이 없음.
    #       엔진 각도 조향 등이 필요해지면 그때 True 로 바꾸고 MASS·Expect.DRAFT 를 같이 고칠 것.
    OUTBOARD_SEPARATE = False
    # 무엇: 선외기 1대(또는 전체) 무게 [kg]. OUTBOARD_SEPARATE = True 일 때만 사용 (지금은 안 씀)
    OUTBOARD_MASS = None


# ─────────────────────────────────────────────────────────────
# 4. 추진기 — 로봇
# ─────────────────────────────────────────────────────────────
# 결정: 2026-11-12 동경 (로봇 내부 값이라 물과 합의 불필요. 명령 형식 CMD_RANGE 만 태수와 맞춤)
class Thruster:
    # 선택지 (METHOD)
    ACC = "acc"             # ForceAccumulator.add(..., point_w=추진기 위치)
    ACTUATOR = "actuator"   # MuJoCo motor → data.ctrl (PoC 방식). 쓰면 계약의 예외로 적어 둬야 함
    # 무엇: 추력을 시뮬레이터에 넣는 방법.
    # 사용: 로봇
    # 이유: 노션 06 계약 "모든 힘은 ForceAccumulator" 와 같고, 힘 장부에 추력도 출처별로 기록돼 디버깅이 쉬움
    METHOD = ACC
    # 무엇: 추진 명령의 범위 = 리모컨 스틱. (−1 최대 후진, 0 정지, +1 최대 전진)
    # 사용: 로봇(명령 → 힘 변환), 태수 PhaseManager(명령을 보내는 쪽)
    # 이유: PoC 방식. 강화학습 라이브러리(SB3)도 −1~+1 을 기본으로 씀. 보내는 쪽은 힘 단위(N)를 몰라도 됨
    CMD_RANGE = (-1.0, 1.0)
    # 무엇: 추진기 1대의 최대 전진 / 최대 후진 추력 [N]. 명령 +1 / −1 일 때의 힘 (2대 합쳐 전진 600 N)
    # 사용: 로봇
    # 이유: PoC 값. 후진이 절반인 건 실제 프로펠러가 역회전 때 효율이 떨어지기 때문.
    #       최고 속도 어림(계산만, 검증 아님): 미현 저항값 기준 약 4.3 m/s, PoC 저항값 기준 약 2.9 m/s
    MAX_FORWARD = 300.0
    MAX_REVERSE = 150.0
    # 무엇: 방향을 바꾸는 방식. "differential" = 좌우 추력 차이 (탱크처럼, 별도 키 없음)
    #   왼쪽 = 전진 − 조향, 오른쪽 = 전진 + 조향  (조향 > 0 → 왼쪽으로 돎)
    # 사용: 로봇
    # 이유: PoC 방식. 러더·선외기 회전이 필요 없어 단순 (그래서 Hull.OUTBOARD_SEPARATE = False)
    STEERING = "differential"
    # 무엇: 좌·우 추진기 위치 [m], 몸체 좌표(FLU, 선체 중심 기준). Names.THRUSTER_L/R_SITE 가 이 자리에 놓인다.
    #   x −1.15 = 선미(배 끝 −1.2)에서 5 cm 안쪽, y ±0.35 = 좌(+)·우(−) 대칭, z −0.2 = 선체 바닥
    #   바닥에 두는 이유: 흘수 0.10 m 라 수면은 중심 기준 약 −0.10 m. 바닥이면 잔잔한 물에서 항상 잠김
    # 사용: 로봇 build(site 위치), add_forces(추력 작용점)
    # 이유: PoC boat_scene.xml 값. 좌우 간격(0.7 m)이 넓을수록 같은 추력 차이로 더 잘 돈다
    POS_L_B = (-1.15, 0.35, -0.2)
    POS_R_B = (-1.15, -0.35, -0.2)
    # 무엇: 프로펠러가 이 깊이 [m] 이상 잠겨야 추력 100%. 파도로 물 밖에 나오면 추력이 줄어든다.
    # 사용: 로봇. 물의 수면 높이 함수가 필요함 (WaterAPI 참고)
    # 결정: 비워 둠. WaterAPI.USE_IN_WEEK1 = False 라 1주차엔 안 씀. 연동할 때 정함 (PoC: 0.08 m)
    FULL_THRUST_DEPTH = None


# ─────────────────────────────────────────────────────────────
# 5. 물리 상수 — 물과 로봇이 같은 값을 써야 함
# ─────────────────────────────────────────────────────────────
class Physics:
    # 무엇: 물 밀도 [kg/m³] = 물 1 m³ 의 무게. 바닷물은 민물(1000)보다 조금 무겁다.
    # 사용: 물(부력 = 밀도 × 잠긴 부피 × g, 저항), 테스트(Expect.DRAFT)
    # 합의: 2026-11-12 회의. 이유: 바닷물 표준값 (구조 현장이 바다). 민물이면 1000
    RHO_WATER = 1025.0
    # 무엇: 중력 가속도의 크기 [m/s²].
    # 사용: 물(부력), world.py(MuJoCo option gravity). 둘이 반드시 같아야 함 → 이 값 하나만 사용
    # ⚠️ 크기만 양수로 적는다. 아래 방향(−z) 부호는 코드에서 처리 (MuJoCo 에는 (0, 0, −GRAVITY)).
    #    아래 자동 검사도 양수를 전제로 함.
    # 합의: 2026-11-12 회의. 이유: 지구 표준 중력값
    GRAVITY = 9.81


# ─────────────────────────────────────────────────────────────
# 5-1. 물 모듈 내부 설정 (미현) — 합의 안건 아님, 제안값을 채워 둠
# ─────────────────────────────────────────────────────────────
# 물 밀도와 중력은 여기서 따로 만들지 않고 Physics.RHO_WATER, Physics.GRAVITY 를 쓴다.
# (같은 값을 두 곳에 적으면 한쪽만 바뀌는 사고가 나기 쉬움)
# 이름이 WaterParams 인 이유: sim/water.py 의 Water(물 모듈 본체)와 헷갈리지 않게.
#   config.WaterParams = 물의 "설정값 묶음", water.Water = 그 값을 읽어 실제로 힘을 계산하는 모듈.
class WaterParams:
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
    # 좌우·앞뒤 기울기(roll·pitch)는 위아래 출렁임처럼 "감쇠비"로 정하고, 계수는 water.py 가 계산한다.
    #   계산식: c = 2 · zeta · sqrt(k · I)
    #     I = 돌리기 어려운 정도(관성), k = 기울였을 때 되돌아오려는 힘의 세기(복원 강성)
    #     k = m · g · GM   (GM = 메타센터 높이: 클수록 오뚝이처럼 잘 일어남. 직육면체 공식으로 계산)
    #   → 배 무게·크기·무게중심이 바뀌어도 "얼마나 빨리 흔들림이 멎는지"는 그대로 유지된다.
    # 이유: 처음 제안한 숫자(8, 8)는 이 배 기준 감쇠비 약 0.02·0.003 이라 기울면 거의 안 멈췄음
    #       (동경 님 지적, PoC 롤 300·피치 600). PoC heave 공진과 같은 종류의 문제.  [미현 / 제안]
    ROT_DAMPING_RATIO = (0.3, 0.3)                                          # [미현 / 제안] (roll, pitch) 감쇠비
    # 회전 속도에 비례하는 계수 [N·m·s/rad]. roll·pitch 는 위 감쇠비로 자동 계산하므로 None.
    # yaw(방향 돌기)는 되돌아오려는 힘이 없어 감쇠비를 정의할 수 없음 → 숫자로 둠.  [미현 / 제안]
    ROT_DRAG_LINEAR = (None, None, 6.0)
    ROT_DRAG_QUAD = (2.0, 2.0, 4.0)                                         # [미현 / 제안] 회전 속도 제곱에 비례

    # 부가질량: 배가 가속할 때 주변 물도 같이 밀어야 해서 실제보다 무겁게 느껴지는 효과.
    # 1주차는 끔, 2주차에 켤 수 있음. 계수 순서는 (surge, sway, heave).     [미현 / 제안]
    ADDED_MASS_ENABLED = False
    ADDED_MASS_COEF = (0.05, 0.5, 1.0)

    # 물 모듈이 제대로 돌기 위한 조건. max_dt = 허용하는 가장 긴 시뮬레이션 한 칸 시간 [s].
    # Sim.TIMESTEP 합의 결과에 맞춰 바꿀 것.                                 [미현 / 제안]
    # 추천: 0.002 s로 시작하고, 채연 님 속도 측정(bench_speed.py) 결과를 보고 0.004 s로 올려도 안정적인지 나중에 확인. 
    REQUIRES = {"max_dt": 0.002}


# ─────────────────────────────────────────────────────────────
# 6. 소유권 — 같은 힘을 두 번 넣지 않기 위한 약속
# ─────────────────────────────────────────────────────────────
# 확정 (노션 06):
#   선체 질량·관성 → 로봇      부력 → 물        선체 물 저항 → 물
#   추력           → 로봇      부가질량 → 물    (로봇 질량에 미리 더하지 않음)
#   로봇은 추진 모델에 저항 항을 넣지 않고, 물은 프로펠러 효과를 넣지 않는다.
# 아래는 노션 표에 없는 회색 지대. 값은 Owner.WATER 또는 Owner.ROBOT 만 쓴다 (문자열 직접 쓰지 않음).
# 코드에서는 "if config.Owner.WIND == config.Owner.WATER: 바람을 넣는다" 처럼 담당일 때만 넣는다.
# 합의: 2026-11-12 회의 — 원칙 "배가 스스로 내는 힘은 로봇, 바깥 환경이 배에 주는 힘은 물".
class Owner:
    # 선택지 (담당 모듈 이름표)
    WATER = "water"   # 미현 water.py
    ROBOT = "robot"   # 동경 robot.py
    # 무엇: 바람 힘(물 위로 나온 선체·콘솔이 받는 힘)을 누가 넣나. PoC 는 로봇(boat.py)이 넣었음
    # 이유: 파도·해류와 같은 환경 힘이라 한 파일에 모음. 바람 받는 면적은 Hull 값에서 읽음. 1주차는 바람 0
    WIND = WATER
    # 무엇: 해류(물 자체의 흐름)를 누가 넣나. 미현 님 WaterParams.WATER_VELOCITY_W 가 이 자리 (1주차는 0)
    # 이유: 저항 계산("물에 대한 배 속도") 안에 들어가는 값. PoC 도 물(hull.py)이 처리
    CURRENT = WATER
    # 무엇: 저항 계수 값(WaterParams.DRAG_* 등)을 누가 정하나
    # 이유: 계산식이 물 코드에 있으니 계수도 같은 곳에서 관리.
    # ⚠️ 회전 저항(ROT_DRAG_*)이 PoC 보다 수십 배 작음 (PoC 롤 300·피치 600 vs 8·8) — 흔들림 테스트로 확인할 것
    #    → 반영: roll·pitch 를 감쇠비(WaterParams.ROT_DAMPING_RATIO = 0.3)로 바꿈. 계산값 롤 약 124·피치 약 708.
    #      흔들림이 멎는지는 tests/test_water.py 에서 확인 (미현)
    DRAG_COEFFS = WATER


# ─────────────────────────────────────────────────────────────
# 7. 물 ↔ 로봇 주고받는 함수 — 형태만 합의, 구현은 water.py
# ─────────────────────────────────────────────────────────────
class WaterAPI:
    # 무엇: 수면 높이 함수 wave_height(x, y, t) -> z 가 여러 점을 한 번에(배열로) 받을지 (True / False).
    # 사용: 로봇(프로펠러 위치의 수면 높이), 물(복셀 위치의 수면 높이)
    # 이유: PoC 에서 점 개수보다 함수 호출 횟수가 더 비쌌음 → 한 번에 받는 쪽이 빠름.
    #       물 복셀(8×4×4 = 128칸)도 한 번에 물어야 함.
    # 합의: 2026-11-12 회의 — True. 미현은 1주차에 wave_height 의 x, y 를 numpy 배열로도 받게 만들어 둔다
    #       (연동은 안 해도 함수 모양만 맞춰 두면 나중에 로봇이 바로 씀).
    WAVE_HEIGHT_VECTORIZED = True
    # 무엇: 물이 계산한 값(잠긴 비율, 프로펠러 위치 수면 높이)을 로봇이 같은 스텝에서 어디서 읽나.
    #       스텝 순서가 water → robot 이라 가능. 후보: Water 객체 속성 / 공유 dict
    # 합의: 2026-11-12 회의 — 아직 정하지 않음. 실제로 연동할 때 정한다.
    SHARED_STATE = None
    # 무엇: 위 연동을 1주차에 넣을지 (True / False). False 면 1주차엔 "프로펠러는 항상 물속" 으로 가정
    # 합의: 2026-11-12 회의 — False. 1주차 완료 기준(잔잔한 물에서 뜨기·앞으로 가기)에 파도가 필요 없음.
    #       언제 넣을지는 11/23 회의(2주차 목표 정할 때)에서 결정 (2주차 후보).
    USE_IN_WEEK1 = False


# ─────────────────────────────────────────────────────────────
# 8. 시뮬레이션 설정 — world.py 에서만 적용
# ─────────────────────────────────────────────────────────────
class Sim:
    # 무엇: 시뮬레이션 한 칸의 시간 [s]. 작을수록 정확하지만 느리다. (0.002 s = 1초에 500칸 = 500 Hz)
    # 사용: world.py(MuJoCo option timestep). world.py 는 시작 시 모든 모듈의
    #       REQUIRES["max_dt"] ≥ TIMESTEP 인지 검사하고 아니면 멈춘다 (가영, 노션 06 문제 5)
    # 합의: 2026-11-12 회의 — 0.002 s (500 Hz).
    # 이유: PoC 에서 실제로 돌려 안정적이었던 값. 노션 초안 0.004 s(250 Hz)는 이 배 모델로 돌려 본 적 없음.
    #       채연 속도 측정(bench_speed.py) 후 0.004 s 로 올려도 안정적인지 확인하면 그때 올린다
    #       (올릴 때는 WaterParams.REQUIRES["max_dt"] 도 같이 올려야 world.py 검사를 통과).
    # 결정은 가영(world.py)·태수(단계 전환)에게 공유할 것.
    TIMESTEP = 0.002
    # 무엇: 시뮬레이터가 다음 칸을 계산하는 방법.
    # 이유: PoC·노션 초안과 같음. 저항·감쇠 힘이 큰 모델에서 빠르고 안정적인 편 (MuJoCo 권장 관례)
    INTEGRATOR = "implicitfast"
    # 무엇: 제어 명령을 바꾸고 기록을 남기는 주기 [Hz]. 50 Hz = 1초에 50번 = 물리 10칸(0.002 s × 10)마다 한 번.
    # 이유: 노션 초안 값. 사람이 조종하는 배의 반응 속도로 충분 (실험으로 정한 값 아님)
    CONTROL_HZ = 50
    # 무엇: MuJoCo 내장 유체 모델(option density/viscosity)을 끈다 (True 고정).
    # 이유: 미현 님 물 모델과 같이 켜면 저항이 두 번 들어감. world.py 시작 시 assert (노션 06 확정 규칙)
    BUILTIN_FLUID_OFF = True


# ─────────────────────────────────────────────────────────────
# 9. Geom 그룹·충돌 — 가영 표와 맞춤
# ─────────────────────────────────────────────────────────────
# geom = MuJoCo 의 모양 하나. 그룹 번호로 "보이기만 하는 것 / 부딪히는 것 / 센서용"을 나눈다.
class Groups:
    VISUAL = None    # 무엇: 화면에만 보이고 부딪히지 않는 모양 (겉모양 선체 등). 노션 초안: 0
    PHYSICS = None   # 무엇: 실제로 부딪히는 모양 — 선체 hull 은 여기 (사람·해저와 충돌). 노션 초안: 3
    SENSOR = None    # 무엇: 센서(소나 광선 등)만 맞는 모양. 노션 초안: 4


# ─────────────────────────────────────────────────────────────
# 9-1. 사람 모델 내부 설정 — 가영의 독립 구현용 시험 초안
# ─────────────────────────────────────────────────────────────
# 아래 수치는 실측·논문·팀 회의에서 확정한 값이 아니다. 단순 모델 구현을 시작하기 위해
# 가영 작업 브랜치에 둔 가정값이며, 실제 사람을 정확히 재현한다는 뜻이 아니다.
# human.py의 단순 강체 모델이 이 값을 읽는다. 사람 부력·줄 연결은 아직 미구현이다.
class Human:
    # humanoid: 몸통·머리·양팔·양다리를 한 body에 고정한 형상 시험용 모델.
    # 관절 운동·기도·접촉 지지는 후속 작업. 기존 box도 Human(shape='box')로 시험 가능.
    SHAPE = "humanoid"
    # 아래 세 값은 기존 상자 시험 모드의 전체 길이·폭·두께 [m].
    # MuJoCo box의 size는 반길이이므로
    # 구현할 때 (LENGTH/2, WIDTH/2, HEIGHT/2)를 전달해야 한다.
    # 사람 로컬 축 제안: +x 머리 쪽, +y 왼쪽, +z 몸의 앞면 쪽.
    # 월드에서 눕거나 서 있는 자세는 배치 시 회전으로 정한다 (아직 미정).
    LENGTH = 1.7
    WIDTH = 0.4
    HEIGHT = 0.2
    # 무엇: 사람 모델 전체 질량 [kg]. 단순 구현용 가정값이며 검증된 대표값이 아니다.
    # 질량은 human.py에서만 설정하고, 물 모듈이 이를 다시 추가하지 않는다.
    MASS = 70.0
    # 기존 box 모드에서만 지정하는 무게중심 [m].
    # humanoid 모드의 무게중심·관성은 아래 부위별 형상·질량에서 MuJoCo가 계산한다.
    COM = (0.0, 0.0, 0.0)

    # 아래 형상·질량 비율도 모두 검증 전 가정값이다. +x 머리, +y 왼쪽, +z 몸 앞면.
    # 캡슐 좌표는 양 끝 구 중심 두 점이다 (튜브 반지름만큼 더 뻗어 나옴).
    TORSO_FROMTO_B = ((-0.12, 0.0, 0.0), (0.40, 0.0, 0.0))
    TORSO_RADIUS = 0.14
    HEAD_POS_B = (0.63, 0.0, 0.0)
    HEAD_RADIUS = 0.12
    # 팔은 환보다 머리 쪽(+x)에 두고 양옆으로 뻗어 환의 바깥으로 나온다.
    ARM_L_FROMTO_B = ((0.50, 0.10, 0.0), (0.50, 0.66, 0.0))
    ARM_R_FROMTO_B = ((0.50, -0.10, 0.0), (0.50, -0.66, 0.0))
    ARM_RADIUS = 0.055
    LEG_L_FROMTO_B = ((-0.20, 0.08, 0.0), (-0.85, 0.11, 0.0))
    LEG_R_FROMTO_B = ((-0.20, -0.08, 0.0), (-0.85, -0.11, 0.0))
    LEG_RADIUS = 0.065
    # 순서: 몸통·머리·왼팔·오른팔·왼다리·오른다리. 합은 1, 총질량은 MASS 그대로.
    MASS_FRACTIONS = (0.50, 0.08, 0.06, 0.06, 0.15, 0.15)
    # 이전 직접 연결 초안의 예약 항목. 실제 로프는 구명환에 연결하므로 사용하지 않는다.
    TOW_POINT_B = None


# 구명환 — 가영의 형상·연결 시험용 가정값. 실제 제품 사양·부력 검증값이 아니다.
class Lifebuoy:
    # 원형 중심선 반지름과 튜브 반지름 [m]. 바깥 지름 약 0.8m, 안쪽 약 0.5m.
    # 실제 형상은 정다각형의 변을 캡슐로 둘러 근사하므로 안쪽 폭은 조금 작아진다.
    CENTERLINE_RADIUS = 0.325
    TUBE_RADIUS = 0.075
    SEGMENTS = 12
    # 전체 질량 [kg]. 캡슐 12개에 나눠 주며 12배로 증가시키지 않는다.
    MASS = 2.0
    # 고정 연결 시험: 사람 좌표에서 머리 쪽(+x)으로 0.35m인 상체 부근.
    # 환은 사람의 x축에 수직인 yz 평면에 놓여 상자 몸통을 둘러싼다.
    POSITION_IN_HUMAN_B = (0.35, 0.0, 0.0)
    # 환 로컬 좌표의 로프 연결점 [m]. 현재는 환 표면의 시험 위치이며 정확한 묶음 위치는 조정 대상.
    # 배를 +x 방향에 둔 시험을 가정: +x 쪽 튜브 표면, y가 양수인 테두리.
    TOW_POINT_B = (TUBE_RADIUS, CENTERLINE_RADIUS, 0.0)


# ─────────────────────────────────────────────────────────────
# 9-2. 줄 모델 내부 설정 — 가영의 독립 구현용 시험 초안
# ─────────────────────────────────────────────────────────────
# 아래 값은 tendon 스프링·인장 감쇠 실험의 출발점일 뿐,
# 합의·검증된 로프 사양이나 안정성을 보장하는 값이 아니다.
class Tether:
    # 무엇: 줄이 늘어나지 않은 기준 길이 [m].
    # 로봇과 사람의 처음 배치 거리와는 별개이며, 배치할 때 함께 확인해야 한다.
    REST_LENGTH = 2.0
    # 무엇: 늘어남에 대한 스프링 강성 [N/m].
    # 인장 스프링 예시: 0.1m 늘어나면 이 항의 크기는 100×0.1=10N.
    # tendon의 springlength=(0, REST_LENGTH)로 짧을 때 스프링 압축력을 없앤다.
    STIFFNESS = 100.0
    # 무엇: 줄 길이가 변하는 속도를 억제하는 감쇠 계수 [N·s/m].
    # 스프링의 출렁임을 줄이는 실험용 값. 감쇠는 강성과 다른 항이다.
    DAMPING = 10.0
    # 미정: 처음부터 연결할지, 접촉 단계 후 연결할지 (동경·태수와 합의).
    ATTACH_ON_START = None
    # 미정: 끊김을 모델링할지와 끊김 기준 장력 [N].
    # None은 '절대 끊어지지 않는다'는 확정이 아니라 아직 정책을 정하지 않았다는 뜻.
    BREAK_FORCE = None


# ─────────────────────────────────────────────────────────────
# 10. 테스트 정답값 — 물·로봇이 둘 다 있어야 의미 있는 테스트
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
    # 무엇: 시뮬레이션 흘수가 DRAFT 와 이만큼 [m] 차이 나도 "맞다"고 본다. 채연과 합의
    DRAFT_TOL = None
    # 무엇: 가만히 떠 있을 때 롤·피치(기울기)가 이 각도 [rad] 안이면 "평형"으로 본다
    EQUILIBRIUM_TILT_TOL = None
    # 무엇: 잔잔한 물 표류 테스트 — 파도·추력 0 으로 DRIFT_SECONDS [s] 동안 두었을 때
    #       수평으로 DRIFT_MAX [m] 미만만 움직여야 통과. 노션 초안: 60 s / 0.1 m
    DRIFT_SECONDS = None
    DRIFT_MAX = None
    # 무엇: 총질량 테스트 — MuJoCo 가 계산한 배 무게(body_subtreemass)가 Hull.MASS 와
    #       이 비율 안에서 같아야 함 (무게를 두 번 셌는지 잡음). 노션 초안: 1% (0.01)
    MASS_TOL = None
    # 방향 테스트: 몸체를 90° 돌리고 "앞으로" → 월드 +y 가속 (로봇 1주차 완료 기준, 상수 없음)


# ─────────────────────────────────────────────────────────────
# 11. 로봇 ↔ 다른 담당 — 미현 님과는 아니지만 로봇이 걸친 것
# ─────────────────────────────────────────────────────────────
class Capsize:
    # 무엇: 배가 이 각도 [rad] 이상 기울면 "전복"으로 판정. 롤 = 좌우, 피치 = 앞뒤.
    # 사용: 로봇(판정), 태수 PhaseManager(판정 후 처리)
    ROLL_LIMIT = None
    PITCH_LIMIT = None
    # 무엇: 전복 판정 후 무엇을 할지. 후보: 에피소드 종료 / 안전 모드 (태수와 합의)
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
        value = getattr(WaterParams, name)
        if len(value) != 3:
            raise ValueError(f"WaterParams.{name} 은 값이 3개여야 함 (지금 {len(value)}개).")

    # 복셀은 각 방향 최소 1칸
    if any(n < 1 for n in WaterParams.VOXEL_RES):
        raise ValueError(f"WaterParams.VOXEL_RES 의 각 값은 1 이상이어야 함 (지금 {WaterParams.VOXEL_RES}).")

    # 감쇠비는 0 과 1 사이
    if not 0 < WaterParams.HEAVE_DAMPING_RATIO < 1:
        raise ValueError(
            f"WaterParams.HEAVE_DAMPING_RATIO 는 0 과 1 사이여야 함 (지금 {WaterParams.HEAVE_DAMPING_RATIO})."
        )
    if len(WaterParams.ROT_DAMPING_RATIO) != 2 or not all(0 < z < 1 for z in WaterParams.ROT_DAMPING_RATIO):
        raise ValueError(
            f"WaterParams.ROT_DAMPING_RATIO 는 (roll, pitch) 2개, 각각 0 과 1 사이여야 함 "
            f"(지금 {WaterParams.ROT_DAMPING_RATIO})."
        )

    # 파도 성분: 진폭·파장은 양수, 너무 가파르면(진폭/파장 ≥ 1/7) 파도가 부서짐(쇄파)
    # → 매끈한 사인파로 계산하는 이 모델이 맞지 않는 범위
    for waves_name in ("WAVES_CALM", "WAVES_DEFAULT"):
        for i, (amp, length, _direction, _phase) in enumerate(getattr(WaterParams, waves_name)):
            where = f"WaterParams.{waves_name}[{i}]"
            if amp <= 0:
                raise ValueError(f"{where}: 진폭은 0 보다 커야 함 (지금 {amp}).")
            if length <= 0:
                raise ValueError(f"{where}: 파장은 0 보다 커야 함 (지금 {length}).")
            if amp / length >= 1 / 7:
                raise ValueError(
                    f"{where}: 진폭/파장 = {amp / length:.3f} ≥ 1/7 → 파도가 부서지는 범위라 "
                    "사인파 모델로 계산할 수 없음."
                )

    # 담당자 이름표는 정해진 선택지만 (오타가 나면 아무도 그 힘을 안 넣는 사고가 남)
    for name in ("WIND", "CURRENT", "DRAG_COEFFS"):
        owner = getattr(Owner, name)
        if owner is not None and owner not in (Owner.WATER, Owner.ROBOT):
            raise ValueError(f"Owner.{name} 은 Owner.WATER 또는 Owner.ROBOT 이어야 함 (지금 {owner!r}).")

    # 추진기: 넣는 방법은 정해진 선택지만, 명령 범위는 (작은 값, 큰 값), 최대 추력은 양수
    if Thruster.METHOD is not None and Thruster.METHOD not in (Thruster.ACC, Thruster.ACTUATOR):
        raise ValueError(
            f"Thruster.METHOD 는 Thruster.ACC 또는 Thruster.ACTUATOR 여야 함 (지금 {Thruster.METHOD!r})."
        )
    if Thruster.CMD_RANGE is not None and not Thruster.CMD_RANGE[0] < Thruster.CMD_RANGE[1]:
        raise ValueError(f"Thruster.CMD_RANGE 는 (작은 값, 큰 값) 이어야 함 (지금 {Thruster.CMD_RANGE}).")
    for name in ("MAX_FORWARD", "MAX_REVERSE"):
        value = getattr(Thruster, name)
        if value is not None and value <= 0:
            raise ValueError(f"Thruster.{name} 은 0 보다 커야 함 (지금 {value}). 후진도 크기만 양수로 적음.")
    # 추진기 위치: 선체 상자 안(표면 포함), 왼쪽은 y>0 · 오른쪽은 y<0 (FLU 에서 왼쪽이 +y)
    half = None if None in (Hull.LENGTH, Hull.BEAM, Hull.HEIGHT) else (Hull.LENGTH / 2, Hull.BEAM / 2, Hull.HEIGHT / 2)
    for name in ("POS_L_B", "POS_R_B"):
        pos = getattr(Thruster, name)
        if pos is None:
            continue
        if len(pos) != 3:
            raise ValueError(f"Thruster.{name} 은 (x, y, z) 3개여야 함 (지금 {pos}).")
        if half is not None and any(abs(p) > h for p, h in zip(pos, half)):
            raise ValueError(f"Thruster.{name} {pos} 가 선체 상자(±{half}) 밖에 있음.")
    if Thruster.POS_L_B is not None and Thruster.POS_L_B[1] <= 0:
        raise ValueError(f"왼쪽 추진기 y 는 양수여야 함 (FLU, 지금 {Thruster.POS_L_B[1]}).")
    if Thruster.POS_R_B is not None and Thruster.POS_R_B[1] >= 0:
        raise ValueError(f"오른쪽 추진기 y 는 음수여야 함 (FLU, 지금 {Thruster.POS_R_B[1]}).")

    # 중력은 채워졌다면 양수 (부호는 아래 방향으로 water.py 가 처리)
    if Physics.GRAVITY is not None and Physics.GRAVITY <= 0:
        raise ValueError(f"Physics.GRAVITY 는 0 보다 커야 함 (지금 {Physics.GRAVITY}).")


_check_water_robot()
