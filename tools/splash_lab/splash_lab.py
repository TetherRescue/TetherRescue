"""조사 1 · 파도가 지형에 부딪혀 철썩 — 작은 시험 맵.

바다 + 비스듬한 모래사장 + 바위 하나. 파도(사인파)가 모래사장 쪽(+x)으로 밀려온다.
파도가 지형에 부딪히는 곳에서 물이 튀는 효과를 여러 방식으로 비교한다.

  off : 효과 없음
  a   : 작은 구체(물방울)를 물리로 던짐 (freejoint 구체, 지형과만 충돌)
  b   : flex 변형체 덩어리를 물리로 던짐
  c   : 화면 효과 ① 물보라 기둥 + ② 거품 띠   (= fx 의 plume+foam)
  fx  : 화면 효과를 골라 조합 (--fx 로 지정, + 로 연결)
          plume   ① 반투명 흰 타원체가 솟았다 흐려짐
          foam    ② 거품 띠 (납작한 흰 타원체)
          streak  ③ 속도 방향으로 늘어난 물줄기 (캡슐)
          sprite  ④ 물보라 그림(투명 PNG)을 카메라를 향한 판에 붙임
          foamtex ④' 거품 그림을 수면에 눕혀 붙임 (foam 대신)
          bulge   ⑤ 부딪힌 곳의 수면 메시를 잠깐 솟게 함
          spray   ④+ 물보라 조각 그림 수십 장을 입자처럼 뿌림 (sprite 대신, 추천)
화면 효과(c, fx)는 물리에 영향이 없다 — 뷰어 장면에 그림만 덧그린다.
하나개 크기 맵은 hanagae_lab.py (이 파일의 파도·효과·로봇 연결을 가져다 씀).

실행 (저장소 루트에서, 가상환경 켠 상태):
  python tools/splash_lab/splash_lab.py --mode fx --fx spray+foamtex+bulge
  python tools/splash_lab/splash_lab.py --mode fx --fx spray+foamtex+bulge --robot --throttle 0.6
  python tools/splash_lab/splash_lab.py --mode a
  python tools/splash_lab/splash_lab.py --mode fx --fx spray+streak+foamtex+bulge [--sea three|team|regular|irregular] [--no-reflect] [--no-shadow] [--no-shore]
콘솔에 2초마다 물리 스텝 시간·화면 sync 시간·효과 그리기 시간을 출력한다.

화면 없이 비교 그림만 뽑기 (행 = 모드, 열 = 시각):
  python tools/splash_lab/splash_lab.py --snapshot out.png --snapshot-modes off,c,fx:streak,fx:sprite+foamtex+bulge
"""
import argparse
import math
import os
import time

import mujoco
import numpy as np

from sprites import NAMES as SPRITE_NAMES, SPRAY_KINDS, load_sprites

# 팀 코드(sim/) 위치: 이 파일이 tools/splash_lab/ 에 있으므로 두 단계 위가 저장소 루트.
# 다른 곳으로 옮겨 쓸 때는 환경 변수 TETHER_ROOT 로 지정
REPO_ROOT = os.environ.get("TETHER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 맵 / 파도 설정 ─────────────────────────────────────────────
HALF_X, HALF_Y = 10.0, 5.0          # 맵 크기 20 m x 10 m
GRID = 0.2                          # 부딪힘 감지·거품 칸 [m]
MESH_GRID = 0.1                     # 화면용 수면 메시 칸 [m] (작을수록 부드럽고 무거움, --mesh-grid)
                                    #   0.2 보다 작으면 0.2 격자 높이를 3차 보간으로 늘려 씀
WAVE_A = 0.30                       # 진폭 [m]
WAVE_L = 6.0                        # 파장 [m]
WAVE_T = 3.0                        # 주기 [s]
BEACH_X0, BEACH_SLOPE = 3.0, 0.18   # 모래사장: x=3 에서 수심 0, 기울기
SEABED_Z = -2.0      # 바닥 높이. 모래사장·바위·물 모두 여기까지 채움 (옆에서 봐도 빈 틈 없음)
ROCK = dict(pos=(0.0, 2.2, 0.0), half=(0.5, 0.9, 1.0))   # 바위 (박스)

# ── 부딪힘 감지 ────────────────────────────────────────────────
FX_EVERY = 4          # 화면 효과(감지·갱신)를 몇 스텝마다 할지 (4 x 2 ms = 8 ms)
RISE_TH = 0.25        # 수면 상승 속도가 이 이상일 때만 "철썩" [m/s]
COOLDOWN = 0.6        # 같은 칸에서 다시 튀기까지 [s]
STEEP = 0.25          # 앞 칸 지형이 수면보다 이만큼 높으면 "벽"(바위)

# ── A / B: 물리 ────────────────────────────────────────────────
LIFE = 2.5            # 물방울·덩어리 수명 [s]
N_DROPS = 500         # A 물방울 풀 크기
DROPS_PER_HIT = 4     # A 한 칸이 부딪힐 때 던지는 물방울 수
N_BLOBS = 8           # B 덩어리 풀 크기
BLOB_CHANCE = 0.30    # B 부딪힌 칸 중 덩어리를 던지는 비율
PARK = np.array([200.0, 0.0, 5.0])   # 안 쓰는 것을 치워 두는 곳 (맵 옆 멀리, 바닥과 안 닿게)

# ── 로봇(배) ───────────────────────────────────────────────────
ROBOT_START_DIST = 5.0    # 바위 앞면에서 이만큼 바다 쪽에서 바위를 보고 출발 [m]
ROBOT_THROTTLE = 0.3      # 처음 전진 명령 (−1 ~ 1). 뷰어에서 방향키로 바꿈
AP_KP, AP_KD = 1.5, 0.4   # 방향 유지 자동조종 이득 (조향 = KP x 방향 오차[rad] − KD x 회전 속도[rad/s])

# ── 화면 효과 ──────────────────────────────────────────────────
WHITE = np.array([0.96, 0.98, 1.0])
# ① plume
PLUME_LIFE = 1.3
PUFFS_PER_HIT = 6
BEACH_PLUME_CHANCE = 0.08
MAX_PUFFS = 400
PUFF_G = 7.0
# ② foam / foamtex
FOAM_TAU = 0.45       # 물 위 거품(흰 물결)이 옅어지는 시간 상수 [s]
FOAM_TAU_DRY = 0.5    # 물이 빠진 모래 위 거품 (빨리 스며듦)
FOAM_MIN = 0.04
# ③ streak
STREAKS_WALL = 28     # 철썩 이벤트 하나에서 나가는 물줄기 수
STREAK_LIFE = 1.4
STREAK_R = 0.012      # 물줄기 굵기 (반지름) [m]
STREAK_LEN = 0.04     # 길이 = 속력 x 이 값 [s]
MAX_STREAKS = 900
# ④ sprite
SPRITE_LIFE_PUFF = 1.5
SPRITE_LIFE_BURST = 1.1
CRASH_LEAN = 0.3      # 솟는 물보라를 바다 쪽으로 기울이는 각 [rad]
MAX_SPRITES = 120
# ④+ spray: 작은 그림 조각 수십 개를 입자처럼 뿌림 (철썩 이벤트 하나당)
SPRAY_N = dict(sheet=10, drop=22, mist=6)   # 종류별 조각 수
SPRAY_EMIT = 0.35     # 조각이 차례로 나오는 시간 [s] (한꺼번에 나타나지 않게)
MAX_SPRAY = 700
# 철썩 물보라 조절 (③ streak + ④+ spray 공통). 1.0 = 처음 버전
SPLASH_UPWARD = True   # False = 위로 튀는 조각(물막·물방울·물줄기) 없음. 솟는 물·거품·낮은 물안개만
SPLASH_UP = 0.75      # 위로 솟는 속도 배율 (작을수록 덜 쏘아 올림. 너무 낮으면 파도 윤곽 위로 안 올라와 '물속'처럼 보임)
SPLASH_OUT = 0.8      # 바다 쪽·옆으로 퍼지는 속도 배율 (크면 물 위에 낮게 깔려 물속 거품처럼 보임)
SPLASH_DRAG = 1.5     # 공기 저항 배율 (클수록 금방 감속해 흩날림)
SPLASH_FADE_IN = 0.15  # 나타나는 데 걸리는 시간 [s] (한 점에서 툭 튀어나오지 않게)
SPLASH_Z_SPREAD = 1   # 1 = 보이는 수면 바로 위에서 높이를 조금씩 흩어 나옴, 0 = 한 높이에서
SPLASH_ALPHA = 1.3    # 공중에 뜬 물보라 진하기 배율
SPLASH_LAND_FADE = 0.3  # 떨어질 때 수면 위 이 높이[m]부터 옅어져 물에 닿으면 사라짐
SPLASH_FALL_FADE = 1.5  # 떨어지는 속도가 이 값[m/s]이 되면 완전히 사라짐 (꼭대기에서 흩어짐 → 물에 '다이빙'하지 않게)
SPLASH_FALL_GROW = 0.8  # 떨어지는 동안 흩어지며 커지는 정도
MIST_DELAY = (0.0, 0.1)   # 물안개가 물보라보다 늦게 나오는 시간 [s] (처음 0.12~0.4)
MIST_LIFE = (0.5, 0.8)    # 물안개 수명 [s] (처음 1.2~2.0)
MIST_FADE_IN = 0.06       # 물안개가 나타나는 데 걸리는 시간 [s]
SPLASH_LIFT = 0.8     # 그림 판 아래 가장자리를 수면 위로 (판 반지름 x 이 값만큼 띄움)
# ⑤ bulge
# 물 투명도. 1 미만(반투명)이면 MuJoCo 가 물을 화면 효과보다 나중에 그려서
# 물보라·거품·물안개가 물 '아래'에 깔린 것처럼 보임 → 불투명(1.0)으로 둠
WATER_ALPHA = 1.0
BULGE_MAX = 1.5       # 수면 메시가 솟을 수 있는 최대 높이 [m] (hfield 범위)
BULGE_T = 0.6         # 솟았다 가라앉는 시간 [s] (꼭대기는 절반 시점)
BULGE_SX, BULGE_SY = 0.55, 0.9    # 솟는 모양 크기 (가우스 σ) [m] — 벽 쪽으로 얇고 벽 따라 넓게
BULGE_H = (0.35, 0.35)  # 솟는 높이 = 앞 + 뒤 x 세기 [m]
# 철썩 이벤트 (벽)
IMPACT_SPACING = 1.0      # 벽을 따라 이 간격마다 이벤트 하나 [m]
IMPACT_REFRACTORY = 1.5   # 같은 구간은 이 시간 안에 다시 터지지 않음 (= 파도 하나에 한 번) [s]
IMPACT_DELAY = 0.2        # 솟기 시작 → 꼭대기에서 터지기까지 [s]
# 모래사장 흰 물결 (쇄파)
BREAK_DEPTH = -0.4        # 이보다 얕은 곳(지형 높이)에서 파도가 부서짐 [m]
WHITEWATER_ETA = 0.4      # 마루 앞면(η > 이 비율 x 진폭, 빠르게 오르는 중)을 흰 물결로
WHITEWATER_RISE = 0.6     # 앞면 판정: 수면 상승 속도 > 이 비율 x 최대 상승 속도
BREAK_SPRAY_CHANCE = 0.25 # 부서지는 선에서 낮은 물안개가 이는 비율


# 파도 성분 = (진폭 [m], 파장 [m], 진행 방향 [rad, +x 기준], 위상 [rad])  — 본 시뮬 WaterParams.WAVES_* 와 같은 형식
#   높이 = 진폭 · cos(파수 · (진행 방향 거리) − 각진동수 · t + 위상),  파수 = 2π/파장,  각진동수 = √(g · 파수)
WAVES_THREE = (
    (0.20, 12.0, math.radians(0), 0.0),      # 너울: 멀리서 온 길고 큰 파도 (주기 약 2.8 s)
    (0.10, 5.0, math.radians(25), 1.3),      # 풍랑: 바람이 만든 중간 파도 (약 1.8 s), 비스듬히
    (0.04, 2.0, math.radians(-40), 2.1),     # 잔물결: 짧은 물결 (약 1.1 s), 반대쪽으로 비스듬히
)
WAVES_TEAM = (                               # 본 시뮬 config.WaterParams.WAVES_DEFAULT 그대로 (2026-11-25 기준)
    (0.10, 8.0, 0.0, 0.0),
    (0.05, 4.0, math.radians(30), 1.3),
    (0.02, 2.0, math.radians(-45), 2.1),
)
# 반사 (벽)
R_ROCK = 0.8          # 바위(수직 벽) 반사율
R_BEACH = 0.3         # 모래사장(경사면) 반사율
ROCK_SPREAD_DEG = 35  # 바위 반사파가 퍼지는 각 (작은 벽이라 부채꼴로 퍼짐)
ROCK_DECAY = 8.0      # 바위 반사파가 약해지는 거리 [m]
# 바위 뒤 그늘 (차폐 + 회절)
SHADOW_ON = True
# 해안: 파도는 수심 d 에서 높이(마루~골)가 SHORE_GAMMA x (d + SHORE_SWASH) 를 넘지 못함 (부서짐 한계).
#   → 해안에 가까울수록 작아지고, 모래 높이가 SHORE_SWASH 를 넘는 곳부터는 파도가 없음 (모래 속 통과 X)
SHORE_ON = True
SHORE_GAMMA = 0.8      # 부서짐 한계 (파고 / 수심). 실제 바다에서 흔히 0.78
SHORE_SWASH = 0.35     # 물이 모래를 타고 올라가는 여유 높이 [m] (클수록 해안선이 더 멀리 오르내림)
SHADOW_EDGE = 0.15    # 바위 바로 뒤 그늘 경계의 번짐 [m]
SHADOW_BLUR = 0.35    # 멀어질수록 경계가 번지는 정도 (√(파장 x 거리) 배)
SHADOW_KEEP = 2.0     # 그늘이 유지되는 거리 = 이 값 x 바위 반폭 + 4 w0²/λ


class Sea:
    """수면 = 사인파 성분들의 합 + 벽에서 되돌아오는 반사파.

    regular  : 성분 1개 (진폭 WAVE_A, 파장 WAVE_L, 주기 WAVE_T, +x 방향) — 처음 쓰던 파도
    three    : WAVES_THREE (너울·풍랑·잔물결)
    team     : 본 시뮬 WAVES_DEFAULT
    irregular: Pierson–Moskowitz 스펙트럼을 N 개 주파수로 나누고 방향을 퍼뜨린 불규칙 파도.
               Hs(유의파고) = 4√(Σaᵢ²/2), Tp(첨두 주기). 시드가 같으면 항상 같은 바다.

    반사파 (add_wall): 벽 면에 거울처럼 비친 성분. 벽 위에서 입사파와 위상이 같게 맞춘다.
      → 벽 앞에서 입사파 + 반사파 = 제자리에서 위아래로 출렁이는 정상파(clapotis).
      벽이 짧으면(half_len) 반사파가 부채꼴로 퍼지며 약해진다: 폭 w(s) = half_len + s·tan(spread),
      세기 ∝ √(half_len / w(s)) · exp(−s / decay)   (s = 벽에서 바다 쪽으로 잰 거리)
    """

    def __init__(self, kind="regular", hs=2 * WAVE_A, tp=WAVE_T, spread_deg=30.0, n=16, seed=0):
        self.kind = kind
        if kind in ("regular", "three", "team"):
            comps = {"regular": ((WAVE_A, WAVE_L, 0.0, 0.0),), "three": WAVES_THREE, "team": WAVES_TEAM}[kind]
            a = np.array([c[0] for c in comps])
            k = 2 * math.pi / np.array([c[1] for c in comps])
            th = np.array([c[2] for c in comps])
            # regular 은 처음 쓰던 주기(WAVE_T)를 그대로, 나머지는 깊은 물 공식 ω = √(g k)
            w = np.array([2 * math.pi / WAVE_T]) if kind == "regular" else np.sqrt(9.81 * k)
            ph = np.array([c[3] for c in comps]) + math.pi / 2          # cos → sin 으로 바꿔 씀
            a_ref = float(np.sqrt(np.sum(a ** 2)))                      # 판정 기준 진폭
            rise_ref = float(np.sqrt(np.sum((a * w) ** 2)))
        else:
            rng = np.random.default_rng(seed)
            fp = 1 / tp
            edges = np.linspace(0.65 * fp, 2.6 * fp, n + 1)
            f = 0.5 * (edges[:-1] + edges[1:]) + rng.uniform(-.3, .3, n) * np.diff(edges)   # 주파수 살짝 흔들기
            df = np.diff(edges)
            S = (5 / 16) * hs ** 2 * fp ** 4 * f ** -5 * np.exp(-1.25 * (fp / f) ** 4)       # PM 스펙트럼
            a = np.sqrt(2 * S * df)
            a *= (hs / 4) / math.sqrt(0.5 * np.sum(a ** 2))                                # Hs 맞춤
            w = 2 * math.pi * f
            k = w ** 2 / 9.81                                                              # 깊은 물 분산 관계
            th = np.radians(spread_deg) * np.clip(rng.normal(0, 0.5, n), -1.5, 1.5)        # 방향 퍼짐 (+x 기준)
            ph = rng.uniform(0, 2 * math.pi, n)
            a_ref = hs / 2
            rise_ref = a_ref * 2 * math.pi / tp
        self.a_ref, self.rise_ref = a_ref, rise_ref        # "보통 파도" 진폭·상승 속도 (흰 물결 판정 기준)
        # 그룹 = (창 함수 또는 None, 진폭, kx, ky, ω, 위상, 바위 그늘 적용 여부).  첫 그룹 = 입사파
        self.groups = [(None, a, k * np.cos(th), k * np.sin(th), w, ph, True)]
        self.walls = []
        self.shadows = []          # 파도를 가리는 물체 (바위)
        self.shores = []           # (기준점, 바다 쪽 방향, 기울기) — 얕아지는 경사 해안

    def add_shore(self, point, seaward, slope):
        """경사 해안: point 에서 수심 0, seaward 방향으로 갈수록 slope 로 깊어짐."""
        n = np.asarray(seaward, float)
        self.shores.append((np.asarray(point, float), n / np.linalg.norm(n), float(slope)))
        self.__dict__.pop("_cache", None)

    def _shore_factor(self, x, y):
        """해안 감쇠 (0~1): 파고 ≤ γ(d + 여유) 가 되도록 진폭을 줄임. 깊은 곳은 1."""
        f = np.ones(np.broadcast(x, y).shape)
        a_sig = 1.4 * self.a_ref                           # 보통 파도의 마루 높이
        for p, n, slope in self.shores:
            d = slope * ((x - p[0]) * n[0] + (y - p[1]) * n[1])  # 수심 (모래 위는 음수)
            f = np.minimum(f, np.clip(SHORE_GAMMA * (d + SHORE_SWASH) / (2 * a_sig), 0.0, 1.0))
        return f

    def add_shadow(self, center, yaw, half_x, half_y):
        """파도를 가리는 박스(바위) 추가. 성분마다 그 진행 방향으로 뒤쪽에 그늘이 생긴다.

        성분 진행 방향 d, 바위 기준 좌표 along(= d 방향 거리), lat(= 옆 방향 거리)
          바위가 막는 반폭 w0 = hx|sin(θ−yaw)| + hy|cos(θ−yaw)|,  앞뒤 반길이 hd = hx|cos| + hy|sin|
          b = 바위 뒷면에서 잰 거리 (along − hd, 0 이상)
          그늘 깊이 = 앞면 이후 · 옆 범위 안 · exp(−b / L_rec)
            옆 경계 번짐 σ = SHADOW_EDGE + SHADOW_BLUR · √(λ b)     (회절: 멀수록·파장이 길수록 번짐)
            회복 거리 L_rec = SHADOW_KEEP · w0 + 4 w0² / λ             (짧은 파도일수록 그늘이 오래감)
          → 그 성분의 진폭 x (1 − 그늘 깊이)
        """
        self.shadows.append((np.asarray(center, float), yaw, half_x, half_y))

    def _shadow_factor(self, kx, ky, x, y):
        """(성분 수, ...) 모양: 성분마다 그 위치에서 남는 진폭 비율 (1 = 그늘 없음)."""
        sh = (-1,) + (1,) * x.ndim
        k = np.hypot(kx, ky)
        th = np.arctan2(ky, kx).reshape(sh)
        lam = (2 * math.pi / k).reshape(sh)
        F = np.ones((len(kx),) + x.shape)
        for c, yaw, hx, hy in self.shadows:
            rel = th - yaw
            w0 = hx * np.abs(np.sin(rel)) + hy * np.abs(np.cos(rel))
            hd = hx * np.abs(np.cos(rel)) + hy * np.abs(np.sin(rel))
            dx, dy = x - c[0], y - c[1]
            along = np.cos(th) * dx + np.sin(th) * dy
            lat = -np.sin(th) * dx + np.cos(th) * dy
            b = np.maximum(along - hd, 0.0)
            sigma = SHADOW_EDGE + SHADOW_BLUR * np.sqrt(lam * b)
            lateral = 0.5 * (1 - np.tanh((np.abs(lat) - w0) / sigma))
            started = 0.5 * (1 + np.tanh((along + hd) / 0.1))          # 바위 앞면부터 그늘
            recover = np.exp(-b / (SHADOW_KEEP * w0 + 4 * w0 ** 2 / lam))
            F *= 1 - started * lateral * recover
        return F

    @property
    def amax(self):
        """이론상 최대 높이 (메시 범위용): 입사 진폭 합 x (1 + 반사율 합)."""
        return float(np.sum(np.abs(self.groups[0][1]))) * (1 + sum(R for _, R in self.walls))

    def add_wall(self, center, normal, R, half_len=None, spread_deg=ROCK_SPREAD_DEG, decay=ROCK_DECAY, name="wall",
                 shadowed=True):
        """벽 하나 추가. center: 벽 면 가운데 (x, y), normal: 벽에서 바다 쪽을 향하는 방향, R: 반사율.
        half_len=None 이면 끝없이 긴 벽 (모래사장 해안선).
        shadowed: 이 반사파도 바위 그늘에 가려지는지 (바위 자신이 만든 반사파는 False)."""
        _, a, kx, ky, w, ph, _ = self.groups[0]
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        c = np.asarray(center, float)
        kn = kx * n[0] + ky * n[1]
        hit = kn < 0                                   # 벽 쪽으로 오는 성분만 반사
        if not hit.any():
            return
        kx_r, ky_r = kx - 2 * kn * n[0], ky - 2 * kn * n[1]
        ph_r = ph + 2 * kn * (n @ c)                   # 벽 위에서 입사파와 같은 위상 → 벽에서 높이 (1+R) 배
        if half_len is None:
            window, gain = None, R
        else:
            t_dir = np.array([-n[1], n[0]])
            tan_s = math.tan(math.radians(spread_deg))

            def window(x, y):
                dx, dy = x - c[0], y - c[1]
                s = dx * n[0] + dy * n[1]              # 벽에서 바다 쪽 거리
                u = dx * t_dir[0] + dy * t_dir[1]      # 벽 따라 거리
                sp = np.maximum(s, 0.0)
                width = half_len + sp * tan_s
                lateral = 0.5 * (1 - np.tanh((np.abs(u) - width) / 0.4))
                front = 0.5 * (1 + np.tanh((s + 0.3) / 0.1))   # 벽 면(s=0)에서는 1, 벽 안쪽 깊이 들어가면 0
                return R * front * lateral * np.sqrt(half_len / width) * np.exp(-sp / decay)
            gain = 1.0
        self.groups.append((window, a[hit] * gain, kx_r[hit], ky_r[hit], w[hit], ph_r[hit], shadowed))
        self.walls.append((name, R))

    def _static(self, gi, x, y):
        """그룹 gi 의 시간과 무관한 부분 (성분별 진폭 x 그늘 x 반사 범위). 큰 격자는 한 번만 계산해 둔다."""
        window, a, kx, ky, _, _, shadowed = self.groups[gi]
        key = None
        if x.size > 1000:                                  # 수면 격자 (점 몇 개짜리 질의는 그냥 계산)
            key = (gi, x.shape, float(x.flat[0]), float(x.flat[-1]), float(y.flat[0]), float(y.flat[-1]))
            hit = getattr(self, "_cache", {}).get(key)
            if hit is not None:
                return hit
        sh = (-1,) + (1,) * x.ndim
        A = np.broadcast_to(a.reshape(sh), (len(a),) + x.shape)
        if shadowed and self.shadows:
            A = A * self._shadow_factor(kx, ky, x, y)
        if window is not None:
            A = A * window(x, y)
        if self.shores:
            A = A * self._shore_factor(x, y)
        if key is not None:
            self.__dict__.setdefault("_cache", {})[key] = A
        return A

    def _sum(self, x, t, y, deriv):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(0.0 if y is None else y, float))
        sh = (-1,) + (1,) * x.ndim
        out = np.zeros(x.shape)
        for gi, (_, a, kx, ky, w, ph, _) in enumerate(self.groups):
            phase = kx.reshape(sh) * x + ky.reshape(sh) * y - w.reshape(sh) * t + ph.reshape(sh)
            A = self._static(gi, x, y)                     # (성분, ...) 진폭 — 그늘·반사 범위 포함
            if deriv:
                out += np.einsum("i...,i...->...", A * (-w).reshape(sh), np.cos(phase))
            else:
                out += np.einsum("i...,i...->...", A, np.sin(phase))
        return out

    def orbital(self, x, y, z, t):
        """(x, y, z) 의 물 입자 속도 (u, v, w) [m/s] — 깊은 물 선형 파 이론.

        성분 η = a sin θ 마다:  수평 = a ω sin θ · (진행 방향) · e^{k(z−η)},  위아래 = −a ω cos θ · e^{k(z−η)}
        (수면에서 위아래 속도 = ∂η/∂t). 반사파·바위 그늘도 같은 진폭 계수로 반영.
        z 가 수면보다 위면 수면 값을 쓴다 (e^{k(z−η)} ≤ 1).
        """
        x, y, z = (np.asarray(v, float) for v in (x, y, z))
        depth = np.minimum(z - self.eta(x, t, y), 0.0)
        sh = (-1,) + (1,) * x.ndim
        u = np.zeros((3,) + x.shape)
        for gi, (_, a, kx, ky, w, ph, _) in enumerate(self.groups):
            k = np.hypot(kx, ky)
            theta = kx.reshape(sh) * x + ky.reshape(sh) * y - w.reshape(sh) * t + ph.reshape(sh)
            A = self._static(gi, x, y) * w.reshape(sh) * np.exp(k.reshape(sh) * depth)
            horiz = A * np.sin(theta)
            u[0] += np.sum(horiz * (kx / k).reshape(sh), axis=0)
            u[1] += np.sum(horiz * (ky / k).reshape(sh), axis=0)
            u[2] += np.sum(-A * np.cos(theta), axis=0)
        return u

    def eta(self, x, t, y=None):
        return self._sum(x, t, y, False)

    def rise(self, x, t, y=None):
        return self._sum(x, t, y, True)


SEA = Sea()


def wave_eta(x, t, y=None):
    return SEA.eta(x, t, y)


def wave_grid(gx, gy, t):
    """격자 전체의 (수면 높이, 상승 속도) — [row(y), col(x)]."""
    X, Y = np.meshgrid(gx, gy)
    return SEA.eta(X, t, Y), SEA.rise(X, t, Y)


def parse_mode(spec):
    """'c' → ('c', {plume, foam}),  'fx:sprite+bulge' → ('fx', {sprite, bulge})."""
    if spec == "c":
        return "c", {"plume", "foam"}
    if spec.startswith("fx"):
        _, _, fx = spec.partition(":")
        return "fx", set(filter(None, fx.split("+")))
    return spec, set()


def build_xml(mode, flex_dof, sea):
    nx, ny = int(round(2 * HALF_X / MESH_GRID)) + 1, int(round(2 * HALF_Y / MESH_GRID)) + 1
    ang = math.atan(BEACH_SLOPE)
    # 모래사장 박스: 윗면이 x=BEACH_X0 에서 z=0 을 지나고 BEACH_SLOPE 로 올라감.
    # 물속 깊은 곳(x=-4, z≈-1.3)부터 시작해야 앞 끝 단면이 수면 근처에 "벽"으로 드러나지 않는다.
    x0, x1 = -4.0, HALF_X + 0.5
    beach_len = (x1 - x0) / math.cos(ang)
    xc = 0.5 * (x0 + x1)
    bx = xc + 0.5 * math.sin(ang)
    bz = BEACH_SLOPE * (xc - BEACH_X0) - 0.5 * math.cos(ang)
    rp, rh = ROCK["pos"], ROCK["half"]
    # 모래사장 쐐기: 옆에서 본 단면 = 바닥(SEABED_Z) ~ 경사 윗면, 맵 가장자리보다 1 cm 크게(물 옆면을 덮음)
    wx0, wx1, wy = -4.0, HALF_X + 0.01, HALF_Y + 0.01
    top0, top1 = BEACH_SLOPE * (wx0 - BEACH_X0), BEACH_SLOPE * (wx1 - BEACH_X0)
    wedge = " ".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, z in ((wx0, SEABED_Z), (wx0, top0), (wx1, top1), (wx1, SEABED_Z))
                     for y in (-wy, wy))
    rock_top = rp[2] + rh[2]                       # 바위도 바닥까지 (윗면 높이는 그대로)

    drops = "\n".join(
        f'<body name="drop{i}" pos="{PARK[0] + 0.1 * i} {PARK[1]} {PARK[2]}"><freejoint/>'
        f'<geom type="sphere" size=".04" mass=".002" rgba=".88 .95 1 .9" group="2" '
        f'contype="2" conaffinity="0" condim="1"/></body>'
        for i in range(N_DROPS if mode == "a" else 0))
    blobs = "\n".join(
        f'<flexcomp name="blob{i}" type="ellipsoid" count="4 4 4" spacing=".07 .07 .05" '
        f'pos="{PARK[0] + i} {PARK[1]} {PARK[2]}" radius=".015" dim="3" mass=".4" dof="{flex_dof}" '
        f'rgba=".85 .94 1 .85">'
        f'<elasticity young="2e3" poisson="0.25" damping="0.003"/>'
        f'<contact contype="2" conaffinity="0" selfcollide="none" condim="3"/></flexcomp>'
        for i in range(N_BLOBS if mode == "b" else 0))
    sprites = "\n".join(
        f'<texture name="spr_{n}" type="2d" file="spr_{n}.png" nchannel="4"/>'
        f'<material name="spr_{n}" texture="spr_{n}" rgba="1 1 1 1"/>' for n in SPRITE_NAMES)

    return f"""
<mujoco model="splash_lab">
  <option timestep="0.002" integrator="discrete"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>
  <asset>
    <hfield name="wave" nrow="{ny}" ncol="{nx}" size="{HALF_X} {HALF_Y} {2 * sea.amax + BULGE_MAX} {SEABED_Z * -1 - sea.amax:.4f}"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".6 .75 .9" rgb2=".95 .97 1" width="64" height="64"/>
    <material name="sand" rgba=".86 .78 .6 1"/>
    <mesh name="beach_wedge" vertex="{wedge}"/>
    <material name="rock" rgba=".45 .43 .42 1"/>
    <material name="sea" rgba=".15 .42 .62 {WATER_ALPHA}" specular=".6" shininess=".8"/>
    {sprites}
  </asset>
  <worldbody>
    <light pos="0 0 15" dir="0 0 -1" directional="true" diffuse=".8 .8 .8"/>
    <geom name="seabed" type="box" size="{HALF_X} {HALF_Y} .1" pos="0 0 {SEABED_Z - 0.1}" rgba=".5 .5 .45 1" group="1" contype="8" conaffinity="7"/>
    <geom name="beach" type="mesh" mesh="beach_wedge" material="sand" group="1" contype="8" conaffinity="7"/>
    <geom name="rock" type="box" size="{rh[0]} {rh[1]} {(rock_top - SEABED_Z) / 2}" pos="{rp[0]} {rp[1]} {(rock_top + SEABED_Z) / 2}" euler="0 0 20" material="rock" group="1" contype="8" conaffinity="7"/>
    <geom name="water" type="hfield" hfield="wave" pos="0 0 {-sea.amax}" material="sea" group="0" contype="0" conaffinity="0"/>
    {drops}
    {blobs}
  </worldbody>
</mujoco>"""


def _catmull_rom_matrix(src, dst):
    """균등 격자 src 의 값을 dst 위치로 3차 보간하는 행렬 (len(dst) x len(src)). 끝은 가장자리 값으로 채움."""
    n = len(src)
    u = (np.asarray(dst) - src[0]) / (src[1] - src[0])
    i = np.clip(np.floor(u).astype(int), 0, n - 2)
    t = u - i
    w = np.stack([(-t ** 3 + 2 * t ** 2 - t) / 2, (3 * t ** 3 - 5 * t ** 2 + 2) / 2,
                  (-3 * t ** 3 + 4 * t ** 2 + t) / 2, (t ** 3 - t ** 2) / 2], axis=1)
    M = np.zeros((len(dst), n))
    for j, off in enumerate((-1, 0, 1, 2)):
        np.add.at(M, (np.arange(len(dst)), np.clip(i + off, 0, n - 1)), w[:, j])
    return M


def _decor(geom, emission=0.7):
    """화면 효과 geom 공통 설정: 그림자 안 만듦(DECOR), 스스로 밝게, 반사 없음."""
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    geom.emission = emission
    geom.specular = 0.0
    geom.shininess = 0.0
    geom.reflectance = 0.0


def cam_position(cam):
    """MjvCamera(자유 카메라)의 눈 위치."""
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.array(cam.lookat) - cam.distance * fwd


def billboard(pos, cam_pos, axial=False, roll=0.0):
    """로컬 z 가 카메라를 향하는 회전행렬(행 우선 9개). axial=True 면 위쪽을 세계 z 로 고정."""
    z = cam_pos - pos
    if axial:
        z[2] = 0.0
    z /= max(np.linalg.norm(z), 1e-9)
    x = np.cross([0.0, 0.0, 1.0], z)
    nx = np.linalg.norm(x)
    x = x / nx if nx > 1e-6 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        x, y = c * x + s * y, -s * x + c * y
    return np.column_stack([x, y, z]).ravel()


class SplashLab:
    def __init__(self, mode="a", flex_dof="full", seed=0, fx=None, sea=None, reflect=True, shadow=True,
                 robot=False, throttle=None, drag_fix=True, autopilot=True):
        global SEA
        SEA = self.sea = sea or Sea()
        if not self.sea.walls and not self.sea.shadows:
            add_map_walls(self.sea, reflect, shadow)
        self.mode = mode
        self.fx = set(fx or ())
        xml, assets = build_xml(mode, flex_dof, self.sea), load_sprites()
        self.robot = None
        if robot:
            self._build_with_robot(xml, assets, throttle, drag_fix, autopilot)
        else:
            self.m = mujoco.MjModel.from_xml_string(xml, assets)
            self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        m = self.m
        self.hfield_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, "wave")
        self.vrow, self.vcol = m.hfield_nrow[self.hfield_id], m.hfield_ncol[self.hfield_id]   # 화면 메시 격자
        self.vx = np.linspace(-HALF_X, HALF_X, self.vcol)
        self.vy = np.linspace(-HALF_Y, HALF_Y, self.vrow)
        self.ncol, self.nrow = int(2 * HALF_X / GRID) + 1, int(2 * HALF_Y / GRID) + 1        # 감지·효과 격자
        self.gx = np.linspace(-HALF_X, HALF_X, self.ncol)
        self.gy = np.linspace(-HALF_Y, HALF_Y, self.nrow)
        self.terrain = self._terrain_heights()            # [row(y), col(x)]
        self.last_hit = np.full(self.terrain.shape, -1e9)
        self.hits = 0
        self._wet = None

        # A 물방울 — freejoint 가 qpos/qvel 앞쪽에 연속으로 있음
        n = N_DROPS if mode == "a" else 0
        self.dq = self.d.qpos[:7 * n].reshape(n, 7)       # 보기(view): 쓰면 d.qpos 가 바뀜
        self.dv = self.d.qvel[:6 * n].reshape(n, 6)
        self.drop_park = self.dq.copy()
        self.drop_born = np.full(n, -1.0)                 # <0 = 치워 둠

        # B flex 덩어리 — 각 덩어리 정점(또는 노드) 몸체에 달린 slide 관절들
        self.blob_joints = []
        for i in range(N_BLOBS if mode == "b" else 0):
            fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, f"blob{i}")
            va, vn = m.flex_vertadr[fid], m.flex_vertnum[fid]
            na, nn = m.flex_nodeadr[fid], m.flex_nodenum[fid]
            bodies = set(m.flex_vertbodyid[va:va + vn]) | set(m.flex_nodebodyid[na:na + nn])
            bodies.discard(0)
            self.blob_joints.append([j for j in range(m.njnt) if m.jnt_bodyid[j] in bodies])
        self.blob_born = np.full(len(self.blob_joints), -1.0)

        # 화면 효과 상태
        self.puffs = np.zeros((0, 9))      # ① x0 y0 z0 vx vy vz born r0 s
        self.streaks = np.zeros((0, 7))    # ③ x0 y0 z0 vx vy vz born
        self.sprites = np.zeros((0, 12))   # ④ x0 y0 z0 vx vy vz born size0 grow life kind roll
        self.bumps = np.zeros((0, 4))      # ⑤ x y born H
        # ④+ spray 조각: x0 y0 z0 vx vy vz born life size0 grow matid spin roll0 drag align g alpha
        self.spray = np.zeros((0, 17))
        self.impacts = []                  # 최근 철썩 이벤트 (x, y, t)
        self.pending = []                  # 예약된 터짐 (t_fire, x, y, s, 솟은 높이)
        self.last_break = np.full(self.terrain.shape, -1e9)
        self.foam = np.zeros(self.terrain.shape)
        self.foam_jitter = self.rng.uniform(0.75, 1.35, self.terrain.shape)
        self.foam_offset = self.rng.uniform(-0.45, 0.45, self.terrain.shape + (2,)) * GRID
        yaw = self.rng.uniform(0, 2 * math.pi, self.terrain.shape)
        c_, s_ = np.cos(yaw), np.sin(yaw)
        z_, o_ = np.zeros_like(yaw), np.ones_like(yaw)
        self.foam_mat = np.stack([c_, -s_, z_, s_, c_, z_, z_, z_, o_], axis=-1)
        self.prev_wet = None
        self.spr_mat = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"spr_{n}") for n in SPRITE_NAMES}
        self.draw_ms = 0.0
        self.update_water_mesh()

    # ── 지형 높이: 위에서 아래로 광선 (group 1 = 지형) ────────────
    def _terrain_heights(self):
        mujoco.mj_forward(self.m, self.d)
        grp = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        gid = np.zeros(1, dtype=np.int32)
        h = np.empty((self.nrow, self.ncol))
        for r, y in enumerate(self.gy):
            for c, x in enumerate(self.gx):
                dist = mujoco.mj_ray(self.m, self.d, np.array([x, y, 10.0]), np.array([0, 0, -1.0]),
                                     grp, 1, -1, gid)
                h[r, c] = 10.0 - dist if dist >= 0 else -2.0
        return h

    def update_water_mesh(self):
        """화면용 수면 메시(hfield) 갱신. 메시 격자(MESH_GRID)는 감지 격자(GRID)와 따로다."""
        t = self.d.time
        if self.vcol == self.ncol:
            eta, _ = wave_grid(self.vx, self.vy, t)
        else:
            # 감지 격자(0.2 m)에서 계산한 높이를 3차(Catmull-Rom) 보간으로 촘촘한 메시에 늘림.
            # 점마다 사인을 다 계산하는 것보다 훨씬 싸고, 모양 차이는 1 cm 안팎.
            if not hasattr(self, "_up"):
                self._up = (_catmull_rom_matrix(self.gy, self.vy), _catmull_rom_matrix(self.gx, self.vx))
            coarse, _ = wave_grid(self.gx, self.gy, t)
            eta = self._up[0] @ coarse @ self._up[1].T
        if len(self.bumps):                                   # ⑤ 부딪힌 곳 수면 솟음
            lift = np.zeros_like(eta)
            for bx, by, born, H in self.bumps:
                tau = t - born
                if 0 <= tau < BULGE_T:
                    env = math.sin(math.pi * tau / BULGE_T) ** 1.5
                    # 겹친 솟음은 더하지 않고 큰 쪽만 (여러 칸이 한꺼번에 부딪혀도 벽처럼 쌓이지 않게)
                    np.maximum(lift, H * env * np.exp(-((self.vx[None, :] - bx) / BULGE_SX) ** 2
                                                      - ((self.vy[:, None] - by) / BULGE_SY) ** 2), out=lift)
            eta += lift
        adr = self.m.hfield_adr[self.hfield_id]
        data = np.clip((eta + self.sea.amax) / (2 * self.sea.amax + BULGE_MAX), 0, 1)
        self.m.hfield_data[adr:adr + self.vrow * self.vcol] = data.ravel()
        return self.hfield_id

    def surface_z(self, x, y, t):
        """화면에 보이는 수면 높이 = 파도 + ⑤ 솟은 물(bulge). 물보라가 물 '속'에서 나오지 않게 쓰임."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = wave_eta(x, t, y)
        lift = np.zeros_like(z)
        for bx, by, born, H in self.bumps:
            tau = t - born
            if 0 <= tau < BULGE_T:
                env = math.sin(math.pi * tau / BULGE_T) ** 1.5
                np.maximum(lift, H * env * np.exp(-((x - bx) / BULGE_SX) ** 2 - ((y - by) / BULGE_SY) ** 2), out=lift)
        return z + lift

    # ── 파도가 지형에 부딪힌 칸 찾기 ─────────────────────────────
    def detect_hits(self):
        t = self.d.time
        eta, rise = wave_grid(self.gx, self.gy, t)
        self._grid = (t, eta, rise)
        wet = eta > self.terrain
        dry_ahead = np.zeros_like(wet)
        dry_ahead[:, :-1] = ~wet[:, 1:]                 # 파도 진행 방향(+x) 바로 앞이 마른 땅
        hit = wet & dry_ahead & (rise > RISE_TH) & (t - self.last_hit > COOLDOWN)
        rows, cols = np.nonzero(hit)
        self.last_hit[rows, cols] = t
        self._wet = wet
        return [(self.gx[c], self.gy[r], eta[r, c], rise[r, c], r, c) for r, c in zip(rows, cols)]

    # ── A / B: 물리 물방울·덩어리 ────────────────────────────────
    def _splash_velocity(self, rise):
        up = 1.5 + 4.0 * rise * self.rng.uniform(0.6, 1.2)
        return np.array([self.rng.uniform(-1.8, 0.6), self.rng.uniform(-1.0, 1.0), up])

    def spawn(self, hits):
        t = self.d.time
        for x, y, eta, rise, _r, _c in hits:
            self.hits += 1
            if self.mode == "a":
                free = np.nonzero(self.drop_born < 0)[0][:DROPS_PER_HIT]
                k = len(free)
                if k == 0:
                    continue
                self.dq[free, 0] = x + self.rng.uniform(-.1, .1, k)
                self.dq[free, 1] = y + self.rng.uniform(-.1, .1, k)
                self.dq[free, 2] = eta + .05
                self.dq[free, 3:] = (1, 0, 0, 0)
                self.dv[free] = 0
                self.dv[free, :3] = [self._splash_velocity(rise) for _ in range(k)]
                self.drop_born[free] = t
            elif self.mode == "b" and self.rng.random() < BLOB_CHANCE:
                free = np.nonzero(self.blob_born < 0)[0]
                if len(free) == 0:
                    continue
                i = free[0]
                rest = np.array([PARK[0] + i, PARK[1], PARK[2]])
                self._place_blob(i, np.array([x, y, eta + .15]) - rest, self._splash_velocity(rise))
                self.blob_born[i] = t

    def _place_blob(self, i, shift, vel):
        for j in self.blob_joints[i]:
            axis = self.m.jnt_axis[j]
            self.d.qpos[self.m.jnt_qposadr[j]] = float(axis @ shift)
            self.d.qvel[self.m.jnt_dofadr[j]] = float(axis @ vel)

    def recycle(self):
        t = self.d.time
        if len(self.drop_born):
            live = self.drop_born >= 0
            under = (self.dq[:, 2] < wave_eta(self.dq[:, 0], t, self.dq[:, 1]) - .05) & (self.dv[:, 2] < 0)
            self.drop_born[live & ((t - self.drop_born > LIFE) | under)] = -1
            parked = self.drop_born < 0                  # 치워 둔 것은 떨어지지 않게 고정
            self.dq[parked] = self.drop_park[parked]
            self.dv[parked] = 0
        for i in range(len(self.blob_born)):
            if self.blob_born[i] >= 0 and t - self.blob_born[i] > LIFE:
                self.blob_born[i] = -1
            if self.blob_born[i] < 0:
                self._place_blob(i, np.zeros(3), np.zeros(3))

    def active(self):
        return int((self.drop_born >= 0).sum() + (self.blob_born >= 0).sum())

    # ── 화면 효과: 상태 갱신 (물리와 무관) ───────────────────────
    def update_effects(self, hits):
        t = self.d.time
        dt = self.m.opt.timestep * FX_EVERY
        U = self.rng.uniform
        new_puffs, new_streaks, new_sprites, new_bumps = [], [], [], []
        for x, y, eta, rise, r, c in hits:
            self.hits += 1
            ahead = self.terrain[r, min(c + 1, self.ncol - 1)]
            wall = ahead - eta > STEEP
            s = min(rise / 0.6, 1.5) * (1.0 if wall else 0.45)        # 세기
            self.foam[r, c] = 1.0
            if "plume" in self.fx and (wall or self.rng.random() < BEACH_PLUME_CHANCE):
                for _ in range(PUFFS_PER_HIT if wall else 2):
                    new_puffs.append((x + U(-.1, .1), y + U(-.15, .15), eta,
                                      U(-1.4, -0.3) * s, U(-.5, .5) * s, U(2.0, 3.6) * s, t,
                                      U(.10, .18) * (0.6 + 0.4 * s), s))
            # ── 벽(바위) 철썩: 파도 하나·벽 구간 하나당 이벤트 한 번 ──
            #   ⑤ 수면이 벽을 타고 솟음 → IMPACT_DELAY 뒤 꼭대기에서 ④ 물보라 + ③ 물줄기가 터짐
            if wall and not any(abs(y - ey) < IMPACT_SPACING and abs(x - ex) < 1.0 and t - et < IMPACT_REFRACTORY
                                for ex, ey, et in self.impacts):
                self.impacts.append((x, y, t))
                H = BULGE_H[0] + BULGE_H[1] * s
                if "bulge" in self.fx:
                    new_bumps.append((x - 0.1, y, t, H))
                self.pending.append((t + IMPACT_DELAY, x, y, s, H if "bulge" in self.fx else 0.1))
        self.impacts = [e for e in self.impacts if t - e[2] < IMPACT_REFRACTORY]
        # 예약된 "터짐" 실행: 솟은 물의 꼭대기 높이에서 물보라·물줄기·물안개
        fire = [p for p in self.pending if p[0] <= t]
        self.pending = [p for p in self.pending if p[0] > t]
        for _, x, y, s, top in fire:
            if "sprite" in self.fx:
                new_sprites.append((x - 0.15, y + U(-.1, .1), top * 0.5, 0, 0, 0, t, U(1.3, 1.8) * s, 0.0,
                                    SPRITE_LIFE_BURST, 1, U(-.12, .12)))
                for _ in range(3):                                         # 남는 물안개
                    new_sprites.append((x - 0.2 + U(-.2, .2), y + U(-.5, .5), wave_eta(x, t, y) + top * U(.6, 1.2),
                                        U(-0.8, -0.2), U(-.3, .3), U(0.6, 1.4) * s, t,
                                        U(0.4, 0.6) * s, U(0.6, 1.0), SPRITE_LIFE_PUFF, 0, U(0, 2 * math.pi)))
            if "spray" in self.fx:
                self._emit_spray(x, y, s, top, t)
            if "streak" in self.fx and SPLASH_UPWARD:
                for _ in range(STREAKS_WALL):
                    sx, sy, born = x - 0.1 + U(-.1, .1), y + U(-.45, .45), t + U(0, SPRAY_EMIT)
                    z0 = float(self.surface_z(sx, sy, born)) + (U(0.02, 0.25) if SPLASH_Z_SPREAD else 0.05)
                    new_streaks.append((sx, sy, z0,
                                        U(-2.8, -0.3) * s * SPLASH_OUT, U(-1.4, 1.4) * s * SPLASH_OUT,
                                        U(2.5, 6.0) * s * SPLASH_UP, born))   # 바다 쪽으로만

        # ── 모래사장: 얕은 곳에서 파도가 부서져 흰 물결이 해안으로 밀려옴 ──
        _, eta_g, rise_g = self._grid                                 # detect_hits 에서 이미 계산
        shallow = (self.terrain > BREAK_DEPTH) & (self.terrain < 0.6)
        crest = shallow & (eta_g > WHITEWATER_ETA * self.sea.a_ref) & (rise_g > WHITEWATER_RISE * self.sea.rise_ref)
        self.foam[crest] = 1.0                                       # 마루를 따라 흰 물결 (②/④' 로 그려짐)
        if "sprite" in self.fx:                                      # 부서지는 선: 낮은 물안개
            brk = (np.abs(self.terrain - BREAK_DEPTH) < 0.08) & (eta_g > 0.85 * self.sea.a_ref) \
                & (t - self.last_break > COOLDOWN)
            for r, c in zip(*np.nonzero(brk)):
                self.last_break[r, c] = t
                if self.rng.random() < BREAK_SPRAY_CHANCE:
                    new_sprites.append((self.gx[c] + U(-.1, .1), self.gy[r] + U(-.1, .1), eta_g[r, c] + 0.15,
                                        U(0.2, 0.8), U(-.2, .2), U(0.2, 0.5), t,
                                        U(0.3, 0.45), U(0.8, 1.3), SPRITE_LIFE_PUFF * 0.8, 0, U(0, 2 * math.pi)))

        lists = (("puffs", new_puffs, MAX_PUFFS, PLUME_LIFE, 6), ("streaks", new_streaks, MAX_STREAKS, STREAK_LIFE, 6),
                 ("sprites", new_sprites, MAX_SPRITES, None, 6), ("bumps", new_bumps, 200, BULGE_T, 2))
        if len(self.spray):                                         # 수명 다한 spray 조각 정리
            self.spray = self.spray[t - self.spray[:, 6] < self.spray[:, 7]]
        for name, new, cap, life, born_col in lists:
            arr = getattr(self, name)
            if new:
                arr = np.vstack([arr, np.array(new, float)])[-cap:]
            if len(arr):
                lives = arr[:, 9] if life is None else life
                arr = arr[t - arr[:, born_col] < lives]
            setattr(self, name, arr)
        if len(self.streaks):                                       # 물에 떨어진 물줄기는 없앰
            tau = t - self.streaks[:, 6]
            z = self.streaks[:, 2] + self.streaks[:, 5] * tau - 4.9 * tau ** 2
            xx = self.streaks[:, 0] + self.streaks[:, 3] * tau
            yy = self.streaks[:, 1] + self.streaks[:, 4] * tau
            falling = self.streaks[:, 5] - 9.81 * tau < 0
            self.streaks = self.streaks[~(falling & (z < wave_eta(xx, t, yy)))]
        # 거품: 새로 젖은 칸에 생기고, 물 위는 천천히 / 마른 모래 위는 빨리 옅어짐
        wet = self._wet
        if self.prev_wet is not None:
            self.foam[wet & ~self.prev_wet & (self.terrain > -0.6)] = 1.0
        self.prev_wet = wet
        self.foam *= np.where(wet, math.exp(-dt / FOAM_TAU), math.exp(-dt / FOAM_TAU_DRY))

    def _emit_spray(self, x, y, s, top, t):
        """철썩 이벤트 하나 → 물막 조각·물방울 무리·물안개 조각 수십 개 (차례로 나옴)."""
        U = self.rng.uniform
        rows = []
        f = 0.6 + 0.6 * s                      # 세기 → 속도 배율 (약한 파도도 어느 정도는 솟게)
        mats = {k: [self.spr_mat[f"{k}{i}"] for i in range(c)] for k, c in SPRAY_KINDS.items()}
        for kind, count in SPRAY_N.items():
            if not SPLASH_UPWARD and kind != "mist":
                continue
            for _ in range(count):
                born = t + U(0, SPRAY_EMIT) + (U(*MIST_DELAY) if kind == "mist" else 0.0)
                px, py = x - 0.15 + U(-.1, .1), y + U(-.5, .5)
                bz = float(self.surface_z(px, py, born)) + (U(0.02, 0.3) if SPLASH_Z_SPREAD else 0.05)
                mat = float(self.rng.choice(mats[kind]))
                up, out, dr = SPLASH_UP, SPLASH_OUT, SPLASH_DRAG
                if kind == "sheet":     # 물막: 빠르게 솟고 날아가는 방향으로 늘어나 보임
                    v = (U(-2.4, -0.6) * f * out, U(-1.2, 1.2) * f * out, U(4.0, 7.0) * f * up)
                    rows.append((px, py, bz, *v, born, U(0.7, 1.0), U(0.3, 0.45), 1.0, mat,
                                 0.0, 0.0, 0.6 * dr, 1, 9.81, 0.95))
                elif kind == "drop":    # 물방울 무리: 넓게 퍼지며 돌고 떨어짐
                    v = (U(-3.0, -0.4) * f * out, U(-1.8, 1.8) * f * out, U(3.0, 6.5) * f * up)
                    rows.append((px, py, bz, *v, born, U(0.9, 1.4), U(0.22, 0.38), 0.7, mat,
                                 U(-3, 3), U(0, 2 * math.pi), 0.4 * dr, 0, 9.81, 0.9))
                else:                   # 물안개: 늦게 나와 천천히 떠오르며 퍼짐
                    v = (U(-0.6, -0.1), U(-0.3, 0.3), U(0.6, 1.4) * f if SPLASH_UPWARD else U(0.05, 0.25))
                    rows.append((px - 0.2, py, bz + (top * U(0.2, 0.7) if SPLASH_UPWARD else U(0.0, 0.15)), *v, born, U(*MIST_LIFE),
                                 U(0.4, 0.65) * f, 1.4, mat, U(-.6, .6), U(0, 2 * math.pi), 2.0, 0, 0.4, 0.4))
        self.spray = np.vstack([self.spray, np.array(rows)])[-MAX_SPRAY:]

    def _draw_spray(self, add, t, cam_pos):
        if not len(self.spray):
            return
        P = self.spray
        tau = t - P[:, 6]
        P, tau = P[tau >= 0], tau[tau >= 0]                       # 아직 안 나온 조각은 건너뜀
        c, g = P[:, 13], P[:, 15]
        e = np.exp(-c * tau)
        k = (1 - e) / c
        pos = np.stack([P[:, 0] + P[:, 3] * k, P[:, 1] + P[:, 4] * k,
                        P[:, 2] + (P[:, 5] + g / c) * k - g * tau / c], axis=1)
        vel = np.stack([P[:, 3] * e, P[:, 4] * e, (P[:, 5] + g / c) * e - g / c], axis=1)
        u = np.clip(tau / P[:, 7], 0, 1)
        # 공중에선 진하게 유지하다 끝에 빨리 옅어짐 (흐릿하게 오래 남으면 파란 물 위에서 '물속 거품'처럼 보임)
        mist = P[:, 15] < 1.0                                       # 물안개: 빨리 나타나 곧바로 옅어지기 시작
        alpha = (np.minimum(P[:, 16] * SPLASH_ALPHA, 1.0)
                 * np.clip(tau / np.where(mist, MIST_FADE_IN, SPLASH_FADE_IN), 0, 1)
                 * np.where(mist, (1 - u) ** 1.5, 1 - u ** 2))
        sea = self.surface_z(pos[:, 0], pos[:, 1], t)
        falling = vel[:, 2] < 0                                     # 떨어지며 수면에 가까워지면 물에 섞이듯 사라짐
        alpha[falling] *= np.clip((pos[falling, 2] - sea[falling]) / SPLASH_LAND_FADE, 0, 1)
        fall = np.clip(-vel[:, 2] / SPLASH_FALL_FADE, 0, 1)        # 0 = 오르는 중, 1 = 충분히 떨어지는 중
        is_mist = P[:, 15] < 1.0                                    # 물안개(중력 약함)는 그대로 둠
        alpha[~is_mist] *= (1 - fall[~is_mist]) ** 1.5              # 꼭대기를 지나면 흩어지며 사라짐
        alpha[pos[:, 2] < sea - 0.02] = 0                           # 보이는 수면(솟은 물 포함) 아래는 안 그림
        size = P[:, 8] * (1 + P[:, 9] * tau) * (1 + SPLASH_FALL_GROW * fall * ~is_mist)
        # 그림 판이 커서(반지름 0.2~0.6 m) 중심이 수면 바로 위면 아래 절반이 파도 면에 잘려 '물속'처럼 보임
        # → 그릴 때만 판 아래 가장자리가 수면 위에 오게 올림 (움직임 계산은 그대로)
        pos[:, 2] = np.maximum(pos[:, 2], sea + SPLASH_LIFT * size)
        for i in np.nonzero(alpha > 0.01)[0]:
            if P[i, 14] > 0.5:          # 날아가는 방향으로 세움 (화면에서 진행 방향 = 그림 위쪽)
                zc = cam_pos - pos[i]
                zc /= np.linalg.norm(zc)
                yv = vel[i] - (vel[i] @ zc) * zc
                ny = np.linalg.norm(yv)
                if ny < 1e-6:
                    continue
                yv /= ny
                xv = np.cross(yv, zc)
                mat = np.column_stack([xv, yv, zc]).ravel()
                stretch = 1 + 0.12 * max(vel[i][2], 0.0)              # 오를 때만 늘어남 (떨어질 때 화살처럼 보이지 않게)
                half = (size[i] * 0.6, size[i] * stretch, 0.0005)
            else:
                mat = billboard(pos[i], cam_pos, roll=P[i, 12] + P[i, 11] * tau[i])
                half = (size[i], size[i], 0.0005)
            add(mujoco.mjtGeom.mjGEOM_BOX, half, pos[i], mat, (1, 1, 1, min(0.99, alpha[i])),
                emission=0.85, matid=int(P[i, 10]))

    # ── 화면 효과: 그리기 ────────────────────────────────────────
    def draw_effects(self, scn, cam_pos):
        """scn(MjvScene)에 화면 효과를 덧그린다. 물리에는 영향 없음."""
        t0 = time.perf_counter()
        t = self.d.time
        eye = np.eye(3).ravel()
        self._g = scn.ngeom
        self._scn = scn

        def add(typ, size, pos, mat, rgba, emission=0.7, matid=-1):
            if self._g >= scn.maxgeom:
                return None
            g = scn.geoms[self._g]
            mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), mat,
                                np.asarray(rgba, np.float32))
            _decor(g, emission)
            if matid >= 0:
                g.matid = matid
            self._g += 1
            return g

        eta_g, _ = wave_grid(self.gx, self.gy, t)
        rows, cols = np.nonzero(self.foam > FOAM_MIN)
        if "foam" in self.fx:                                         # ② 납작한 흰 타원체
            for r, c in zip(rows, cols):
                f, j = self.foam[r, c], self.foam_jitter[r, c]
                ox, oy = self.foam_offset[r, c]
                add(mujoco.mjtGeom.mjGEOM_ELLIPSOID, (GRID * 0.55 * j, GRID * 0.4 * (2.1 - j), 0.004),
                    (self.gx[c] + ox, self.gy[r] + oy, max(eta_g[r, c], self.terrain[r, c]) + 0.015),
                    self.foam_mat[r, c], np.append(WHITE, 0.75 * f ** 0.8))
        if "foamtex" in self.fx:                                      # ④' 거품 그림을 눕혀 붙임
            for r, c in zip(rows, cols):
                if (r + c) % 2:                                       # 칸 절반만 (그림이 커서 충분)
                    continue
                f, j = self.foam[r, c], self.foam_jitter[r, c]
                ox, oy = self.foam_offset[r, c]
                add(mujoco.mjtGeom.mjGEOM_BOX, (GRID * 0.85 * j, GRID * 0.85 * j, 0.0005),
                    (self.gx[c] + ox, self.gy[r] + oy, max(eta_g[r, c], self.terrain[r, c]) + 0.02),
                    self.foam_mat[r, c], (1, 1, 1, min(0.99, 1.1 * f)),
                    emission=0.85, matid=self.spr_mat["foam"])
        if "plume" in self.fx:                                        # ① 물보라 퍼프
            for x0, y0, z0, vx, vy, vz, born, r0, s in self.puffs:
                tau = t - born
                life = min(tau / PLUME_LIFE, 1.0)
                pos = np.array([x0 + vx * tau, y0 + vy * tau, z0 + vz * tau - 0.5 * PUFF_G * tau * tau])
                pos[2] = max(pos[2], wave_eta(pos[0], t, pos[1]))
                rad = r0 + 0.22 * tau * (0.6 + 0.4 * s)
                stretch = 1.0 + 1.2 * max(0.0, 1.0 - 2.5 * life)
                add(mujoco.mjtGeom.mjGEOM_ELLIPSOID, (rad, rad, rad * stretch), pos, eye,
                    np.append(WHITE, 0.55 * (1.0 - life) ** 1.8), emission=0.35)
        if "streak" in self.fx:                                       # ③ 속도 방향 물줄기
            for x0, y0, z0, vx, vy, vz, born in self.streaks:
                tau = t - born
                if tau < 0:                                           # 차례로 나옴: 아직 안 나온 물줄기
                    continue
                p = np.array([x0 + vx * tau, y0 + vy * tau, z0 + vz * tau - 4.9 * tau * tau])
                if p[2] < float(self.surface_z(p[0], p[1], t)) - 0.02:  # 물 속이면 안 그림
                    continue
                v = np.array([vx, vy, vz - 9.81 * tau])
                tail = p - v * STREAK_LEN
                life = min(tau / STREAK_LIFE, 1.0)
                fade = (1 - min(max(-v[2] / SPLASH_FALL_FADE, 0.0), 1.0)) ** 1.5   # 떨어지기 시작하면 사라짐
                if fade < 0.02:
                    continue
                g = add(mujoco.mjtGeom.mjGEOM_CAPSULE, (STREAK_R, 0.01, 0), p, eye,
                        np.append(WHITE, 0.8 * (1 - life) ** 0.7 * fade), emission=0.6)
                if g is not None:
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, STREAK_R * (1 - 0.5 * life), tail, p)
        if "sprite" in self.fx:                                       # ④ 물보라 그림
            for x0, y0, z0, vx, vy, vz, born, size0, grow, life, kind, roll in self.sprites:
                tau = t - born
                u = min(tau / life, 1.0)
                if kind == 1:            # 솟는 물보라: 아래가 부딪힌 점에 붙고, 위로 빠르게 자람
                    h = size0 * (0.35 + 0.65 * min(1.0, tau / 0.22))
                    w = h * 1.3                    # 부채꼴로 넓게
                    base = np.array([x0, y0, wave_eta(x0, t, y0) + z0 - 0.05])     # z0 = 수면 위 높이
                    mat0 = billboard(base, cam_pos, axial=True)
                    lean = CRASH_LEAN * np.sign(mat0[0])     # 화면에서 위쪽이 바다(-x) 쪽으로 기울게
                    mat = billboard(base, cam_pos, axial=True, roll=roll + lean)
                    center = base + mat.reshape(3, 3)[:, 1] * h
                    alpha = min(0.99, 1.0 - max(0.0, (u - 0.35) / 0.65) ** 1.2)
                    add(mujoco.mjtGeom.mjGEOM_BOX, (w, h, 0.0005), center, mat, (1, 1, 1, alpha),
                        emission=0.9, matid=self.spr_mat["burst"])
                else:                    # 물안개: 떠오르며 커지고 흐려짐
                    pos = np.array([x0 + vx * tau, y0 + vy * tau, z0 + vz * tau - 1.5 * tau * tau])
                    size = size0 * (1 + grow * tau)
                    alpha = min(0.99, 0.85 * (1 - u) ** 1.5)
                    add(mujoco.mjtGeom.mjGEOM_BOX, (size, size, 0.0005), pos,
                        billboard(pos, cam_pos, roll=roll + 0.4 * tau), (1, 1, 1, alpha),
                        emission=0.8, matid=self.spr_mat["puff"])
        if "spray" in self.fx:
            self._draw_spray(add, t, cam_pos)
        scn.ngeom = self._g
        self.draw_ms = 1000 * (time.perf_counter() - t0)

    # ── 로봇(배): 팀 Robot·Water 모듈을 그대로 붙임. 수면만 이 시험 맵 바다로 바꿔 끼움 ──
    def _build_with_robot(self, xml, assets, throttle, drag_fix=True, autopilot=True):
        import os
        import sys
        root = REPO_ROOT
        if root not in sys.path:
            sys.path.insert(0, root)
        if self.mode == "b":
            raise ValueError("B(flex) 모드는 적분기가 discrete 라 로봇과 같이 못 씀")
        from sim import config
        from sim.forces import ForceAccumulator
        from sim.robot.robot import Robot
        import sim.water.water as W

        # 물 모듈의 부력·노즐 판정이 묻는 수면 = 이 시험 맵 바다 (파도 3개 + 반사 + 그늘). 이 프로세스 안에서만 바뀜
        W.wave_height = lambda x, y, t, waves=None: SEA.eta(x, t, y)

        # [고친 버전] 물 저항을 "정지한 물"이 아니라 "그 자리 물 입자 속도"에 대해 계산한다.
        #   물 모듈 water_drag 는 config.WaterParams.WATER_VELOCITY_W(해류)를 빼고 계산하므로,
        #   부르는 순간에만 해류 + 파도 궤도 속도로 바꿔 넘긴다 → 기존 계산 경로(heave 보정 등)는 그대로.
        if not hasattr(W, "_orig_water_drag"):
            W._orig_water_drag = W.water_drag
        if drag_fix:
            orig = W._orig_water_drag

            def water_drag_fixed(model, data, body_id, coeffs, heave_world_only=False):
                p = data.xipos[body_id]
                saved = config.WaterParams.WATER_VELOCITY_W
                config.WaterParams.WATER_VELOCITY_W = tuple(
                    np.asarray(saved, float) + SEA.orbital(p[0], p[1], p[2], data.time))
                try:
                    return orig(model, data, body_id, coeffs, heave_world_only)
                finally:
                    config.WaterParams.WATER_VELOCITY_W = saved
            W.water_drag = water_drag_fixed
        else:
            W.water_drag = W._orig_water_drag

        spec = mujoco.MjSpec.from_string(xml, assets=assets)
        spec.option.timestep = config.Sim.TIMESTEP
        spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{config.Sim.INTEGRATOR.upper()}")
        # 바위 앞면을 향해 ROBOT_START_DIST 떨어진 곳에서 바위 쪽을 보고 출발
        yaw = math.radians(20)
        ex = np.array([math.cos(yaw), math.sin(yaw)])
        face = np.array(ROCK["pos"][:2]) - ROCK["half"][0] * ex
        start = face - ROBOT_START_DIST * ex
        start_z = config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - (config.Expect.DRAFT or 0.0)
        self.robot = Robot(position=(start[0], start[1], start_z), yaw=yaw)
        self.water = W.Water(waves=(), surface_visual=False)
        self.robot.build(spec)
        self.water.build(spec)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.robot.bind(self.m)
        self.water.bind(self.m)
        self.robot.surface_height = self.water.wave_height
        self.acc = ForceAccumulator()
        self.robot.set_command(ROBOT_THROTTLE if throttle is None else throttle, 0.0)
        self.autopilot = autopilot             # 바위 앞면을 향하게 조향 자동 보정 (←→ 누르면 꺼짐)
        self.target_xy = face
        self.boat_id = self.water.body_id
        self.rock_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "rock")
        self.beach_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "beach")
        self.touching = {}            # 지형 이름 → 지금 닿아 있나
        self.hit_log = []             # (시각, 지형, 닿기 직전 속도 [m/s], 최대 접촉력 [N])
        self._peak = {}
        mujoco.mj_forward(self.m, self.d)

    def _steer_to_target(self):
        """방향 유지: 목표(바위 앞면) 쪽 방위와 배 방향의 차이로 조향 (P-D). 파도에 배가 돌아가도 다시 바위를 향함."""
        b = self.boat_id
        R = self.d.xmat[b].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        dx, dy = self.target_xy - self.d.xpos[b][:2]
        err = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = self.d.qvel[self.m.body_dofadr[b] + 5]
        self.robot.set_command(self.robot.throttle, float(np.clip(AP_KP * err - AP_KD * yaw_rate, -1, 1)))

    def _robot_forces(self):
        if self.autopilot and self._n_step % 10 == 0:          # 50 Hz (본 시뮬 제어 주기와 같음)
            self._steer_to_target()
        mujoco.mj_forward(self.m, self.d)
        self.acc.reset()
        self.robot.add_forces(self.m, self.d, self.acc)
        self.water.add_forces(self.m, self.d, self.acc)
        self.acc.apply(self.m, self.d)

    def _check_contacts(self):
        """배가 바위·모래사장에 닿기 시작한 순간과 최대 접촉력을 기록."""
        now = {}
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            if self.boat_id not in (b1, b2):
                continue
            other = con.geom2 if b1 == self.boat_id else con.geom1
            name = {self.rock_gid: "rock", self.beach_gid: "beach"}.get(other)
            if name is None:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f6)
            now[name] = max(now.get(name, 0.0), abs(f6[0]))
        for name, force in now.items():
            if not self.touching.get(name):
                v = self.d.qvel[self.m.body_dofadr[self.boat_id]:self.m.body_dofadr[self.boat_id] + 3]
                self.hit_log.append([round(self.d.time, 2), name, float(np.linalg.norm(v)), force])
                self._peak[name] = len(self.hit_log) - 1
            else:
                k = self._peak[name]
                self.hit_log[k][3] = max(self.hit_log[k][3], force)
        self.touching = {n: (n in now) for n in ("rock", "beach")}

    def step(self):
        self._n_step = getattr(self, "_n_step", 0) + 1
        if self.fx and self._n_step % FX_EVERY == 0:      # 효과는 FX_EVERY 스텝마다 (화면 60fps 보다 촘촘)
            self.update_effects(self.detect_hits())
        elif self.mode in ("a", "b"):
            self.spawn(self.detect_hits())
        self.recycle()
        if self.robot is not None:
            self._robot_forces()
        mujoco.mj_step(self.m, self.d)
        if self.robot is not None:
            self._check_contacts()


def run_viewer(lab):
    import mujoco.viewer

    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE = 265, 264, 263, 262, 32

    def on_key(key):
        """배 조종: ↑↓ 전진 명령 ±0.1, ←→ 조향 ±0.2 (자동조종 꺼짐), 스페이스 = 멈춤, A = 자동조종 다시 켬."""
        r = lab.robot
        if r is None:
            return
        th, st = r.throttle, r.steer
        if key == KEY_UP:
            th += 0.1
        elif key == KEY_DOWN:
            th -= 0.1
        elif key in (KEY_LEFT, KEY_RIGHT):
            lab.autopilot = False
            st += 0.2 if key == KEY_LEFT else -0.2
        elif key == KEY_SPACE:
            th = st = 0.0
        elif key == ord("A"):
            lab.autopilot = True
        else:
            return
        r.set_command(round(th, 2), round(st, 2))
        print(f"[배] 전진 {r.throttle:+.1f} · 조향 {r.steer:+.1f} · 자동조종 {'켬' if lab.autopilot else '끔'}")

    n_hits = 0
    with mujoco.viewer.launch_passive(lab.m, lab.d, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = (2.0, 1.0, 0.3)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 10, 35, -18
        if lab.robot is not None:
            viewer.opt.geomgroup[3] = 1            # 팀 로봇 A형 선체는 geom 그룹 3 (PHYSICS) → 켜야 보임
            viewer.cam.lookat[:] = (-2.2, 1.6, 0.2)
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7, 75, -18
            print("[배] 방향키 ↑↓ 전진·후진, ←→ 조향(자동조종 꺼짐), A 자동조종 켬, 스페이스 멈춤")
        steps_per_frame = int(round((1 / 60) / lab.m.opt.timestep))
        t_phys = t_sync = t_draw = 0.0
        n_frames = 0
        t_report = time.perf_counter()
        while viewer.is_running():
            wall0 = time.perf_counter()
            with viewer.lock():
                for _ in range(steps_per_frame):
                    lab.step()
                hid = lab.update_water_mesh()
                if lab.fx:
                    viewer.user_scn.ngeom = 0
                    lab.draw_effects(viewer.user_scn, cam_position(viewer.cam))
                    t_draw += lab.draw_ms
                if lab.robot is not None and lab.hit_log:
                    still = lab.touching.get(lab.hit_log[-1][1])        # 아직 닿는 중이면 최대 접촉력이 확정 전
                    for t_hit, name, v, f in (lab.hit_log[n_hits:-1] if still else lab.hit_log[n_hits:]):
                        print(f"[충돌] {t_hit:6.2f}s {'바위' if name == 'rock' else '모래사장'} · "
                              f"닿을 때 속도 {v:.2f} m/s · 최대 접촉력 {f:.0f} N")
                        n_hits += 1
            t1 = time.perf_counter()
            viewer.update_hfield(hid)
            viewer.sync()
            t2 = time.perf_counter()
            t_phys += t1 - wall0
            t_sync += t2 - t1
            n_frames += 1
            if t2 - t_report > 2.0:
                if lab.fx:
                    extra = (f" · 효과 그리기 {t_draw / n_frames:.2f} ms/프레임 · 효과 geom {viewer.user_scn.ngeom}")
                else:
                    extra = f" · 활성 {lab.active()}"
                print(f"[{lab.mode}{':' + '+'.join(sorted(lab.fx)) if lab.mode == 'fx' else ''}] "
                      f"물리 {1000 * (t_phys - t_draw / 1000) / (n_frames * steps_per_frame):.3f} ms/스텝 · "
                      f"화면 sync {1000 * t_sync / n_frames:.2f} ms/프레임{extra}")
                t_phys = t_sync = t_draw = 0.0
                n_frames = 0
                t_report = t2
            spare = 1 / 60 - (time.perf_counter() - wall0)
            if spare > 0:
                time.sleep(spare)


def save_png(path, img):
    import struct
    import zlib

    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def snapshot(path, flex_dof, times=(5.0, 5.3, 5.6), specs=("off", "a", "b", "c"),
             view=((2.0, 1.0, 0.3), 9, 35, -16)):
    """모드들을 같은 시각들에 렌더 — 행 = 모드, 열 = 시각."""
    rows = []
    for spec in specs:
        spec, _, sea_spec = spec.partition("@")
        sea_kind, _, opt = (sea_spec or "three").partition("/")
        mode, fx = parse_mode(spec)
        opts = set(opt.split("/")) if opt else set()
        lab = SplashLab(mode, flex_dof, fx=fx, sea=make_sea(sea_kind),
                        reflect="norefl" not in opts, shadow="noshadow" not in opts, robot="robot" in opts,
                        drag_fix="nodragfix" not in opts, autopilot="noauto" not in opts)
        r = mujoco.Renderer(lab.m, 360, 640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = view[0]
        cam.distance, cam.azimuth, cam.elevation = view[1:]
        row = []
        for target in times:
            while lab.d.time < target:
                lab.step()
            lab.update_water_mesh()
            r._gl_context.make_current()
            mujoco.mjr_uploadHField(lab.m, r._mjr_context, lab.hfield_id)
            opt = mujoco.MjvOption()
            opt.geomgroup[3] = 1                   # 팀 로봇 선체 (그룹 3)
            r.update_scene(lab.d, cam, scene_option=opt)
            if lab.fx:
                lab.draw_effects(r.scene, cam_position(cam))
            row.append(r.render().copy())
        r.close()
        rows.append(np.hstack(row))
    save_png(path, np.vstack(rows))
    print("saved", path)


SEA_ARGS = dict(hs=2 * WAVE_A, tp=WAVE_T, spread_deg=30.0, seed=0)


def add_map_walls(sea, reflect=True, shadow=SHADOW_ON):
    """시험 맵의 반사 벽: 바위 앞면(짧은 수직 벽) + 모래사장 해안선(긴 경사면), 그리고 바위 뒤 그늘."""
    yaw = math.radians(20)                                      # 바위 박스 euler z
    ex = np.array([math.cos(yaw), math.sin(yaw)])
    face = np.array(ROCK["pos"][:2]) - ROCK["half"][0] * ex     # 바다 쪽 면 가운데
    if reflect:
        sea.add_wall(face, -ex, R_ROCK, half_len=ROCK["half"][1], name="rock", shadowed=False)
        sea.add_wall((BEACH_X0, 0.0), (-1.0, 0.0), R_BEACH, name="beach")
    if shadow:
        sea.add_shadow(ROCK["pos"][:2], yaw, ROCK["half"][0], ROCK["half"][1])
    if SHORE_ON:
        sea.add_shore((BEACH_X0, 0.0), (-1.0, 0.0), BEACH_SLOPE)


def make_sea(kind):
    return Sea(kind) if kind != "irregular" else Sea("irregular", **SEA_ARGS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "a", "b", "c", "fx"], default="fx")
    ap.add_argument("--fx", default="sprite+foamtex+bulge", help="fx 모드에서 켤 효과 (+ 로 연결)")
    ap.add_argument("--flex-dof", choices=["full", "trilinear"], default="full")
    ap.add_argument("--sea", choices=["three", "team", "regular", "irregular"], default="three",
                    help="파도: three(너울·풍랑·잔물결) / team(본 시뮬 WAVES_DEFAULT) / regular(사인파 1개) / irregular")
    ap.add_argument("--no-reflect", action="store_true", help="반사파 끄기")
    ap.add_argument("--no-shadow", action="store_true", help="바위 뒤 그늘 끄기 (파도가 바위를 통과)")
    ap.add_argument("--mesh-grid", type=float, default=MESH_GRID, help="화면 수면 메시 칸 [m] (0.2 / 0.1 / 0.05)")
    ap.add_argument("--robot", action="store_true", help="팀 로봇(배)을 띄움 (TetherRescue 폴더에서 실행)")
    ap.add_argument("--throttle", type=float, default=ROBOT_THROTTLE, help="배 처음 전진 명령 (−1 ~ 1)")
    ap.add_argument("--no-drag-fix", action="store_true", help="물 저항을 지금 물 모듈 그대로 (정지한 물 기준)")
    ap.add_argument("--no-autopilot", action="store_true", help="방향 유지 자동조종 끄고 시작")
    ap.add_argument("--no-nozzle-cutoff", action="store_true",
                    help="노즐이 물 밖으로 나와도 추력 유지 (비교용. 로봇 모듈의 노즐 노출 판정을 끔)")
    ap.add_argument("--hs", type=float, default=2 * WAVE_A, help="irregular: 유의파고 Hs [m]")
    ap.add_argument("--tp", type=float, default=WAVE_T, help="irregular: 첨두 주기 Tp [s]")
    ap.add_argument("--spread", type=float, default=30.0, help="irregular: 방향 퍼짐 [도]")
    ap.add_argument("--sea-seed", type=int, default=0, help="irregular: 시드 (같으면 같은 바다)")
    ap.add_argument("--snapshot", help="화면 없이 비교 PNG 저장")
    ap.add_argument("--snapshot-modes", default="off,a,b,c",
                    help="snapshot 행들 (쉼표 구분, @irregular 붙이면 불규칙 파도). 예: c,fx:sprite+bulge@irregular")
    ap.add_argument("--times", default="5.0,5.3,5.6", help="snapshot 시각들 [s] (쉼표 구분)")
    ap.add_argument("--view", default="2,1,0.3,9,35,-16", help="snapshot 카메라: lookat x,y,z,거리,방위각,고도각")
    ap.add_argument("--no-shore", action="store_true", help="해안 감쇠 끄기 (파도가 모래 속을 그대로 통과 — 이전)")
    args = ap.parse_args()
    if args.no_shore:
        SHORE_ON = False
    SEA_ARGS.update(hs=args.hs, tp=args.tp, spread_deg=args.spread, seed=args.sea_seed)
    MESH_GRID = args.mesh_grid
    if args.snapshot:
        v = [float(x) for x in args.view.split(",")]
        snapshot(args.snapshot, args.flex_dof, tuple(float(x) for x in args.times.split(",")),
                 tuple(args.snapshot_modes.split(",")), (tuple(v[:3]), *v[3:]))
    else:
        mode, fx = parse_mode(args.mode if args.mode != "fx" else "fx:" + args.fx)
        lab = (SplashLab(mode, args.flex_dof, fx=fx, sea=make_sea(args.sea),
                             reflect=not args.no_reflect, shadow=not args.no_shadow,
                             robot=args.robot, throttle=args.throttle,
                             drag_fix=not args.no_drag_fix, autopilot=not args.no_autopilot))
        if lab.robot is not None and args.no_nozzle_cutoff:
            lab.robot.surface_height = None
        run_viewer(lab)
