"""하나개 크기 시험 맵 — 철썩 효과를 실제 맵 크기에서 돌려 보고 렉을 잰다.

지형: sim/world/coast_map.py 의 하나개 맵 치수를 따름 (바다 x −260~0, y ±80 / 갯벌 25 m / 해안선 수직 턱 0.4 m
      모래사장 12 m / 육지 x 80 까지) + 철썩 시험용 바위 3개.
      다른 점: 바다 바닥을 본 맵(−0.4 m)보다 낮춤. 바위 앞 반사파 골(최대 −0.6 m)이 바닥 위로 드러나면
      "마른 땅"으로 판정돼 바다 한가운데서 가짜 철썩이 생기기 때문. 맵 지형은 모두 충돌체(배가 부딪힘).
물:   본 시뮬처럼 배(또는 카메라) 주변 패치만 촘촘히 계산 (기본 50 m). 패치는 PATCH_SNAP 단위로 끊어 따라가고
      (물결이 미끄러져 보이지 않게), 가장자리 PATCH_FADE 에서 파도를 0 으로 줄여 바깥 평평한 바다 상자와 잇는다.
      --full 이면 패치 = 바다 전체 (렉 비교용).
효과·파도·로봇은 splash_lab.py 것을 그대로 씀 (물보라, 거품, 솟는 물, 반사파, 바위 그늘, 물 저항 수정).

실행 (저장소 루트에서, 가상환경 켠 상태):
  python tools/splash_lab/hanagae_lab.py --robot --throttle 0.6
  python tools/splash_lab/hanagae_lab.py --robot --throttle 0.6 --mode off   # 효과 끔
  python tools/splash_lab/hanagae_lab.py --full --mesh-grid 0.5 --grid 0.5  # 바다 전체 계산
"""
import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np

import splash_lab as L

# ── 하나개 맵 치수 (sim/world/coast_map.py 와 같게) ──────────────────
SHORE_X = 0.0
SEA_WEST = -260.0
SEA_HALF_Y = 80.0
LAND_EAST = 80.0
BEACH_WIDTH = 12.0
FLAT_WIDTH = 25.0
BEACH_TOP = 0.4
LAND_TOP = 1.5
GROUND_BOTTOM = -8.0          # 모래사장·육지 바닥 (본 맵 LAND_DEPTH)
# 바다 바닥: 본 맵은 −0.4 m 이지만, 반사파 골보다 낮아야 해서 내림 (위 설명)
SEABED_TOP = -1.0             # x < −25 (먼 바다)
FLAT_TOP = -0.7               # −25 ≤ x < 0 (갯벌)
SEABED_BOTTOM = -4.0

R_SEAWALL = 0.6               # 해안선 수직 턱 반사율
# 철썩 시험용 바위 (본 맵에는 없음). 첫 번째가 로봇 목표 (yaw 20° 는 splash_lab 로봇 출발 계산과 맞춤)
ROCKS = [
    dict(name="rock", pos=(-12.0, -20.0), half=(1.5, 3.0), top=1.2, yaw=20.0),
    dict(name="rock2", pos=(-35.0, 12.0), half=(1.0, 1.6), top=0.9, yaw=-10.0),
    dict(name="rock3", pos=(-6.0, 34.0), half=(2.2, 1.2), top=1.5, yaw=35.0),
]

# ── 물 패치 ──────────────────────────────────────────────────────
PATCH_HALF = (25.0, 25.0)     # 패치 반폭 (x, y) [m] → 50 m × 50 m (본 시뮬 WAVE_SURFACE_HALF 와 같음)
PATCH_SNAP = 5.0              # 패치를 옮기는 단위 [m] (격자 간격의 배수)
PATCH_FADE = 5.0              # 가장자리에서 파도가 0 으로 줄어드는 폭 [m]
GRID_DEFAULT = 0.25           # 감지·거품 격자 [m]
MESH_DEFAULT = 0.25           # 화면 메시 격자 [m]
START_FOCUS = (-14.0, -18.0)  # 처음 패치 중심 (첫 바위 근처)

# ── 거품 그림 (최적화 3) ─────────────────────────────────────────
FOAM_BLOCK = 2                # 거품 칸 FOAM_BLOCK x FOAM_BLOCK 개를 그림 한 장으로 (장수 1/4)
FOAM_LOD_DIST = 40.0          # 카메라에서 이보다 먼 거품은 안 그림 [m]
FOAM_MAX = 500                # 거품 그림 최대 장수 (넘으면 진하고 가까운 것부터)

# ── 최적화 2: 부력·물 저항용 진폭 격자 ─────────────────────────────
AMP_GRID = 0.5                # 진폭 격자 간격 [m] (그늘·반사 경계가 이보다 훨씬 넓어 보간 오차 작음)
AMP_MARGIN = 10.0             # 패치보다 이만큼 넓게 (배가 패치 가장자리에 있어도 격자 안)

# ── 최적화 3: 효과 양·주기 ─────────────────────────────────────────
FX_EVERY_H = 8                # 효과 갱신 간격 [스텝] (8 x 2 ms = 60 Hz, 화면 프레임과 같음. 시험 맵은 4)
SPRAY_N_H = dict(sheet=6, drop=12, mist=4)   # 철썩 한 번에 나오는 물보라 조각 (시험 맵 10/22/6)


def _rock_inside(r, x, y):
    yaw = math.radians(r["yaw"])
    dx, dy = x - r["pos"][0], y - r["pos"][1]
    u = dx * math.cos(yaw) + dy * math.sin(yaw)
    v = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return (np.abs(u) <= r["half"][0]) & (np.abs(v) <= r["half"][1])


def terrain_z(x, y):
    """지형 윗면 높이 (식으로 계산 — 광선보다 훨씬 빠름. 패치를 옮길 때마다 다시 구함)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    z = np.select([x < SHORE_X - FLAT_WIDTH, x < SHORE_X, x < SHORE_X + BEACH_WIDTH],
                  [SEABED_TOP, FLAT_TOP, BEACH_TOP], LAND_TOP).astype(float)
    for r in ROCKS:
        z = np.where(_rock_inside(r, x, y), np.maximum(z, r["top"]), z)
    return z


class HSea(L.Sea):
    """큰 격자용: 바위 반사파처럼 범위가 좁은 성분은 진폭이 의미 있는 칸에서만 계산 (결과는 같음)."""
    ACTIVE_MIN = 1e-4          # 이보다 작은 진폭 [m] 은 0 으로 봄
    _amp = None                # (x0, y0, step, nx, ny, [그룹별 진폭 격자 (성분, ny, nx)])

    def set_amp_grid(self, x0, x1, y0, y1, step=AMP_GRID):
        """최적화 2: 그룹별 진폭(그늘·반사 범위 포함, 시간과 무관)을 격자에 미리 계산.
        부력·물 저항처럼 점 몇백 개를 매 스텝 묻는 경우 이 격자에서 보간 → 그늘·반사 계산을 매번 안 함."""
        self._amp = None
        gx = np.arange(x0, x1 + step, step)
        gy = np.arange(y0, y1 + step, step)
        X, Y = np.meshgrid(gx, gy)
        grids = [np.ascontiguousarray(L.Sea._static(self, gi, X, Y)) for gi in range(len(self.groups))]
        self.__dict__.pop("_cache", None)
        self._amp = (x0, y0, step, len(gx), len(gy), grids)
        # 모든 그룹의 성분을 한 줄로 이어 붙임 → 점 질의 한 번에 계산 (그룹마다 따로 부르는 비용 없앰)
        cat = lambda i: np.concatenate([g[i] for g in self.groups])
        kx, ky, w, ph = cat(2), cat(3), cat(4), cat(5)
        k = np.hypot(kx, ky)
        self._amp_all = (np.ascontiguousarray(np.concatenate(grids)), kx, ky, w, ph, k)

    def _points_amp(self, x, y):
        """점들의 모든 성분 진폭 (성분, 점) — 격자 밖이면 None."""
        amp = getattr(self, "_amp", None)
        if amp is None:
            return None
        x0, y0, step, nx, ny, _ = amp
        G = self._amp_all[0]
        if x.size == 0:                                      # 빈 질의 (예: 바닥에 닿은 칸 없음)
            return np.zeros((G.shape[0], 0))
        u, v = (x - x0) / step, (y - y0) / step
        if u.min() < 0 or v.min() < 0 or u.max() > nx - 1.001 or v.max() > ny - 1.001:
            return None
        i, j = u.astype(int), v.astype(int)
        fu, fv = u - i, v - j
        return (G[:, j, i] * (1 - fu) * (1 - fv) + G[:, j, i + 1] * fu * (1 - fv)
                + G[:, j + 1, i] * (1 - fu) * fv + G[:, j + 1, i + 1] * fu * fv)

    def _fast(self, x, t, y, deriv):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(0.0 if y is None else y, float))
        xf, yf = x.ravel(), y.ravel()
        A = self._points_amp(xf, yf)
        if A is None:
            return None
        _, kx, ky, w, ph, _ = self._amp_all
        phase = kx[:, None] * xf + ky[:, None] * yf - (w * t - ph)[:, None]
        out = ((A * -w[:, None]) * np.cos(phase)).sum(0) if deriv else (A * np.sin(phase)).sum(0)
        return out.reshape(x.shape)

    def orbital(self, x, y, z, t):
        """부모와 같은 식 (깊은 물 선형 파), 성분 한꺼번에."""
        xs, ys, zs = (np.atleast_1d(np.asarray(v, float)) for v in (x, y, z))
        A = self._points_amp(xs.ravel(), ys.ravel()) if xs.size <= 1000 else None
        if A is None:
            return super().orbital(x, y, z, t)
        _, kx, ky, w, ph, k = self._amp_all
        xf, yf, zf = xs.ravel(), ys.ravel(), zs.ravel()
        theta = kx[:, None] * xf + ky[:, None] * yf - (w * t - ph)[:, None]
        eta = (A * np.sin(theta)).sum(0)
        Aw = A * w[:, None] * np.exp(k[:, None] * np.minimum(zf - eta, 0.0))
        horiz = Aw * np.sin(theta)
        u = np.stack([(horiz * (kx / k)[:, None]).sum(0), (horiz * (ky / k)[:, None]).sum(0),
                      (-Aw * np.cos(theta)).sum(0)])
        return u.reshape((3,) + np.shape(x))

    def _static(self, gi, x, y):
        amp = self._amp
        if amp is None or x.size > 1000:
            return super()._static(gi, x, y)
        x0, y0, step, nx, ny, grids = amp
        if x.size == 0:
            return super()._static(gi, x, y)
        u, v = (x - x0) / step, (y - y0) / step
        if u.min() < 0 or v.min() < 0 or u.max() > nx - 1.001 or v.max() > ny - 1.001:
            return super()._static(gi, x, y)                 # 격자 밖 → 정확히 계산
        i, j = u.astype(int), v.astype(int)
        fu, fv = u - i, v - j
        G = grids[gi]
        return (G[:, j, i] * (1 - fu) * (1 - fv) + G[:, j, i + 1] * fu * (1 - fv)
                + G[:, j + 1, i] * (1 - fu) * fv + G[:, j + 1, i + 1] * fu * fv)

    def _sum(self, x, t, y, deriv):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(0.0 if y is None else y, float))
        if x.size <= 1000:
            fast = self._fast(x, t, y, deriv)                 # 최적화 2: 점 질의 (부력·노즐·물 저항)
            return fast if fast is not None else super()._sum(x, t, y, deriv)
        out = np.zeros(x.size)
        cache = self.__dict__.setdefault("_active", {})
        for gi, (window, a, kx, ky, w, ph, _) in enumerate(self.groups):
            key = (gi, x.shape, float(x.flat[0]), float(x.flat[-1]), float(y.flat[0]), float(y.flat[-1]))
            hit = cache.get(key)
            if hit is None:
                A = self._static(gi, x, y).reshape(len(a), -1)
                idx = np.nonzero(A.max(axis=0) > self.ACTIVE_MIN)[0] if window is not None else slice(None)
                hit = cache[key] = (idx, np.ascontiguousarray(A[:, idx]), x.ravel()[idx], y.ravel()[idx])
            idx, A, xs, ys = hit
            phase = kx[:, None] * xs + ky[:, None] * ys - (w * t - ph)[:, None]
            if deriv:
                out[idx] += np.einsum("ij,ij->j", A * (-w)[:, None], np.cos(phase))
            else:
                out[idx] += np.einsum("ij,ij->j", A, np.sin(phase))
        return out.reshape(x.shape)


def make_hanagae_sea(kind):
    sea = L.make_sea(kind)
    sea.__class__ = HSea
    sea.add_wall((SHORE_X, 0.0), (-1.0, 0.0), R_SEAWALL, name="seawall")          # 해안선 턱 (끝없이 긴 벽)
    for r in ROCKS:
        yaw = math.radians(r["yaw"])
        ex = np.array([math.cos(yaw), math.sin(yaw)])
        face = np.array(r["pos"]) - r["half"][0] * ex
        sea.add_wall(face, -ex, L.R_ROCK, half_len=r["half"][1], name=r["name"], shadowed=False)
        sea.add_shadow(r["pos"], yaw, r["half"][0], r["half"][1])
    return sea


def _box(name, x0, x1, y0, y1, z0, z1, mat, collide=True):
    c = f'contype="8" conaffinity="7"' if collide else 'contype="0" conaffinity="0"'
    return (f'<geom name="{name}" type="box" size="{(x1 - x0) / 2} {(y1 - y0) / 2} {(z1 - z0) / 2}" '
            f'pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {(z0 + z1) / 2}" material="{mat}" group="1" {c}/>')


def build_xml(sea, half, mesh_grid):
    nx, ny = int(round(2 * half[0] / mesh_grid)) + 1, int(round(2 * half[1] / mesh_grid)) + 1
    amax = sea.amax
    y0, y1 = -SEA_HALF_Y, SEA_HALF_Y
    terrain = "\n    ".join([
        _box("seabed", SEA_WEST, SHORE_X - FLAT_WIDTH, y0, y1, SEABED_BOTTOM, SEABED_TOP, "seabed"),
        _box("tidal_flat", SHORE_X - FLAT_WIDTH, SHORE_X, y0, y1, GROUND_BOTTOM, FLAT_TOP, "flat"),
        _box("beach", SHORE_X, SHORE_X + BEACH_WIDTH, y0, y1, GROUND_BOTTOM, BEACH_TOP, "sand"),
        _box("land", SHORE_X + BEACH_WIDTH, LAND_EAST, y0, y1, GROUND_BOTTOM, LAND_TOP, "grass"),
    ] + [f'<geom name="{r["name"]}" type="box" size="{r["half"][0]} {r["half"][1]} {(r["top"] - SEABED_BOTTOM) / 2}" '
         f'pos="{r["pos"][0]} {r["pos"][1]} {(r["top"] + SEABED_BOTTOM) / 2}" euler="0 0 {r["yaw"]}" '
         f'material="rock" group="1" contype="8" conaffinity="7"/>' for r in ROCKS])
    # 패치 바깥 평평한 바다: 상자 4개 (패치를 옮길 때 위치·크기를 다시 맞춤)
    ring = "\n    ".join(f'<geom name="sea_ring{i}" type="box" size="1 1 1" pos="0 0 -50" material="sea" '
                         f'group="0" contype="0" conaffinity="0"/>' for i in range(4))
    sprites = "\n    ".join(
        f'<texture name="spr_{n}" type="2d" file="spr_{n}.png" nchannel="4"/>'
        f'<material name="spr_{n}" texture="spr_{n}" rgba="1 1 1 1"/>' for n in L.SPRITE_NAMES)
    return f"""
<mujoco model="hanagae_lab">
  <option timestep="0.002"/>
  <visual><global offwidth="1280" offheight="720"/><map zfar="2000"/><quality shadowsize="0"/></visual>
  <asset>
    <hfield name="wave" nrow="{ny}" ncol="{nx}" size="{half[0]} {half[1]} {2 * amax + L.BULGE_MAX} 0.05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".6 .75 .9" rgb2=".95 .97 1" width="64" height="64"/>
    <material name="sand" rgba=".86 .78 .55 1"/>
    <material name="flat" rgba=".45 .38 .22 1"/>
    <material name="grass" rgba=".28 .42 .24 1"/>
    <material name="seabed" rgba=".04 .14 .22 1"/>
    <material name="rock" rgba=".45 .43 .42 1"/>
    <material name="sea" rgba=".15 .42 .62 {L.WATER_ALPHA}" specular=".6" shininess=".8"/>
    {sprites}
  </asset>
  <worldbody>
    <light pos="-40 0 40" dir="0 0 -1" directional="true" diffuse=".8 .8 .8" castshadow="false"/>
    {terrain}
    <geom name="water" type="hfield" hfield="wave" pos="0 0 {-amax}" material="sea" group="0" contype="0" conaffinity="0"/>
    {ring}
  </worldbody>
</mujoco>"""


class HanagaeLab(L.SplashLab):
    def __init__(self, mode="fx", fx=None, sea=None, seed=0, robot=False, throttle=None, drag_fix=True,
                 autopilot=True, full=False, grid=GRID_DEFAULT, mesh_grid=MESH_DEFAULT, focus=START_FOCUS):
        L.GRID = grid                                   # 효과 쪽(거품 크기 등)도 이 격자를 씀
        L.FX_EVERY = FX_EVERY_H                         # 최적화 3
        L.SPRAY_N = dict(SPRAY_N_H)
        L.SHORE_ON = False                              # 경사 해안 감쇠는 시험 맵 전용 (여기는 수직 턱)
        L.ROCK = dict(pos=(*ROCKS[0]["pos"], 0.0), half=(*ROCKS[0]["half"], 1.0))   # 로봇 출발·목표 = 첫 바위
        if full:                                        # 패치 = 바다 전체 (+ 해안 12 m)
            self.half = ((SHORE_X + BEACH_WIDTH - SEA_WEST) / 2, SEA_HALF_Y)
            focus = ((SEA_WEST + SHORE_X + BEACH_WIDTH) / 2, 0.0)
        else:
            self.half = PATCH_HALF
        self.full, self.grid, self.mesh_grid = full, grid, mesh_grid
        self.center = np.array(self._snap(focus) if not full else focus, float)
        self.focus = np.array(focus, float)
        sea = sea or make_hanagae_sea("three")
        self.timing = dict(mesh=0.0, recenter=0.0, n_recenter=0)
        orig_build = L.build_xml
        L.build_xml = lambda mode_, flex_dof, sea_: build_xml(sea_, self.half, mesh_grid)
        try:
            super().__init__(mode, "full", seed, fx, sea, robot=robot, throttle=throttle,
                             drag_fix=drag_fix, autopilot=autopilot)
        finally:
            L.build_xml = orig_build
        self.water_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "water")
        self.ring_gid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"sea_ring{i}") for i in range(4)]
        self._place_patch()
        if robot:
            self._contact_names = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, r["name"]): "rock" for r in ROCKS}
            for n in ("beach", "land", "tidal_flat", "seabed"):
                self._contact_names[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)] = "beach"

    # ── 패치 좌표 ────────────────────────────────────────────────
    def _snap(self, xy):
        return np.round(np.asarray(xy, float) / PATCH_SNAP) * PATCH_SNAP

    def _set_grids(self):
        cx, cy = self.center
        hx, hy = self.half
        self.vx = cx + np.linspace(-hx, hx, self.vcol)
        self.vy = cy + np.linspace(-hy, hy, self.vrow)
        self.ncol, self.nrow = int(round(2 * hx / self.grid)) + 1, int(round(2 * hy / self.grid)) + 1
        self.gx = cx + np.linspace(-hx, hx, self.ncol)
        self.gy = cy + np.linspace(-hy, hy, self.nrow)
        self.vterrain = terrain_z(self.vx[None, :], self.vy[:, None])
        fx_ = np.clip((hx - np.abs(self.vx - cx)) / PATCH_FADE, 0, 1)
        fy_ = np.clip((hy - np.abs(self.vy - cy)) / PATCH_FADE, 0, 1)
        sm = lambda f: f * f * (3 - 2 * f)
        self.vfade = 1.0 if self.full else sm(fy_)[:, None] * sm(fx_)[None, :]

    def _terrain_heights(self):
        """(부모 __init__ 이 부름) 패치 격자를 잡고 지형 높이를 식으로 계산."""
        self._set_grids()
        return terrain_z(self.gx[None, :], self.gy[:, None])

    def _place_patch(self):
        """물 패치 geom 과 바깥 바다 상자 4개 위치를 지금 패치 중심에 맞춤."""
        cx, cy = self.center
        hx, hy = self.half
        if isinstance(self.sea, HSea):                     # 최적화 2: 부력·물 저항용 진폭 격자도 패치 따라
            M = AMP_MARGIN
            self.sea.set_amp_grid(cx - hx - M, cx + hx + M, cy - hy - M, cy + hy + M)
        pos = (cx, cy, -self.sea.amax)
        self.m.geom_pos[self.water_gid] = pos
        self.d.geom_xpos[self.water_gid] = pos
        px0, px1, py0, py1 = cx - hx, cx + hx, cy - hy, cy + hy
        sx0, sx1, sy0, sy1 = SEA_WEST, SHORE_X, -SEA_HALF_Y, SEA_HALF_Y
        ix0, ix1 = max(px0, sx0), min(px1, sx1)
        boxes = [(sx0, min(px0, sx1), sy0, sy1),                  # 서쪽
                 (max(px1, sx0), sx1, sy0, sy1),                  # 동쪽 (해안 쪽)
                 (ix0, ix1, sy0, max(py0, sy0)),                  # 남쪽
                 (ix0, ix1, min(py1, sy1), sy1)]                  # 북쪽
        for gid, (x0, x1, y0, y1) in zip(self.ring_gid, boxes):
            if self.full or x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
                self.m.geom_size[gid] = (1e-3, 1e-3, 1e-3)
                p = (0.0, 0.0, -50.0)
            else:
                self.m.geom_size[gid] = ((x1 - x0) / 2, (y1 - y0) / 2, (0.0 - SEABED_TOP) / 2)
                p = ((x0 + x1) / 2, (y0 + y1) / 2, SEABED_TOP / 2)
            self.m.geom_pos[gid] = p
            self.d.geom_xpos[gid] = p
            self.m.geom_rbound[gid] = float(np.linalg.norm(self.m.geom_size[gid]))

    def _recenter(self, new_center):
        """패치를 new_center 로 옮김. 감지·거품 배열은 칸 수만큼 밀어서 이어지게."""
        t0 = time.perf_counter()
        shift = (np.asarray(new_center) - self.center) / self.grid
        dc, dr = int(round(shift[0])), int(round(shift[1]))
        self.center = np.asarray(new_center, float)
        self._set_grids()
        self.terrain = terrain_z(self.gx[None, :], self.gy[:, None])
        for name, fill in (("last_hit", -1e9), ("last_break", -1e9), ("foam", 0.0)):
            a = getattr(self, name)
            out = np.full_like(a, fill)
            r0, r1 = max(dr, 0), a.shape[0] + min(dr, 0)
            c0, c1 = max(dc, 0), a.shape[1] + min(dc, 0)
            if r1 > r0 and c1 > c0:
                out[r0 - dr:r1 - dr, c0 - dc:c1 - dc] = a[r0:r1, c0:c1]
            setattr(self, name, out)
        self.prev_wet = None
        self.sea.__dict__.pop("_cache", None)           # 그늘·반사 진폭은 새 격자에서 다시 계산
        self.sea.__dict__.pop("_active", None)
        self._band = None                               # 감지 띠도 새 격자에서 다시
        self._place_patch()
        self.timing["recenter"] += time.perf_counter() - t0
        self.timing["n_recenter"] += 1

    def update_water_mesh(self):
        t0 = time.perf_counter()
        if not self.full:
            target = self.d.xpos[self.boat_id][:2] if self.robot is not None and self.follow_boat else self.focus
            if np.max(np.abs(np.asarray(target) - self.center)) > PATCH_SNAP * 0.75:
                self._recenter(self._snap(target))
        t = self.d.time
        if self.vcol == self.ncol and self.vrow == self.nrow:
            eta, _ = L.wave_grid(self.vx, self.vy, t)
        else:
            if not hasattr(self, "_up"):
                self._up = (L._catmull_rom_matrix(self.gy - self.center[1], self.vy - self.center[1]),
                            L._catmull_rom_matrix(self.gx - self.center[0], self.vx - self.center[0]))
            coarse, _ = L.wave_grid(self.gx, self.gy, t)
            eta = self._up[0] @ coarse @ self._up[1].T
        if len(self.bumps):                                          # 솟는 물: 각각 주변(3σ)만 계산
            lift = np.zeros_like(eta)
            for bx, by, born, H in self.bumps:
                tau = t - born
                if 0 <= tau < L.BULGE_T:
                    env = math.sin(math.pi * tau / L.BULGE_T) ** 1.5
                    c0, c1 = np.searchsorted(self.vx, (bx - 3 * L.BULGE_SX, bx + 3 * L.BULGE_SX))
                    r0, r1 = np.searchsorted(self.vy, (by - 3 * L.BULGE_SY, by + 3 * L.BULGE_SY))
                    if c1 <= c0 or r1 <= r0:
                        continue
                    sub = lift[r0:r1, c0:c1]
                    np.maximum(sub, H * env * np.exp(-((self.vx[None, c0:c1] - bx) / L.BULGE_SX) ** 2
                                                     - ((self.vy[r0:r1, None] - by) / L.BULGE_SY) ** 2), out=sub)
            eta = eta + lift
        eta = eta * self.vfade                                       # 가장자리 → 평평한 바다와 이음
        eta = np.where(self.vterrain > 0, np.minimum(eta, self.vterrain - 0.05), eta)   # 땅 속으로는 안 보이게
        amax = self.sea.amax
        adr = self.m.hfield_adr[self.hfield_id]
        data = np.clip((eta + amax) / (2 * amax + L.BULGE_MAX), 0, 1)
        self.m.hfield_data[adr:adr + self.vrow * self.vcol] = data.ravel()
        self.timing["mesh"] += time.perf_counter() - t0
        return self.hfield_id

    follow_boat = True

    # ── 최적화 1: 철썩 감지는 "젖었다 말랐다 할 수 있는 칸"에서만 ─────────
    def _set_band(self):
        """칸마다 파도가 닿을 수 있는 최대 높이 A(성분 진폭 합, 반사·그늘 포함)를 구해
        지형이 −A 보다 낮으면 늘 잠김, +A 보다 높으면 늘 마름 → 그 사이(띠)만 매번 계산."""
        X, Y = np.meshgrid(self.gx, self.gy)
        A = sum(np.abs(self.sea._static(gi, X, Y)).sum(axis=0) for gi in range(len(self.sea.groups)))
        margin = 0.05
        self._always_wet = self.terrain < -A - margin
        always_dry = self.terrain >= A + margin
        band = ~self._always_wet & ~always_dry
        # 철썩 칸 = 젖은 칸 중 "바로 앞(+x)이 마른" 칸 → 늘 잠긴 칸이라도 앞 칸이 늘 잠긴 게 아니면 계산해야 함
        ahead_not_wet = np.zeros_like(band)
        ahead_not_wet[:, :-1] = ~self._always_wet[:, 1:]
        self._band = np.flatnonzero(band | (ahead_not_wet & ~always_dry))
        self._bx, self._by = X.ravel()[self._band], Y.ravel()[self._band]
        self.sea.__dict__.pop("_cache", None)            # 큰 격자 진폭은 띠 계산에 필요 없음 (메모리)

    def detect_hits(self):
        if getattr(self, "_band", None) is None:
            self._set_band()
        t = self.d.time
        eta = np.zeros(self.terrain.shape)
        rise = np.zeros(self.terrain.shape)
        if len(self._band):
            eta.flat[self._band] = self.sea.eta(self._bx, t, self._by)
            rise.flat[self._band] = self.sea.rise(self._bx, t, self._by)
        self._grid = (t, eta, rise)
        wet = self._always_wet.copy()
        wet.flat[self._band] = eta.flat[self._band] > self.terrain.flat[self._band]
        dry_ahead = np.zeros_like(wet)
        dry_ahead[:, :-1] = ~wet[:, 1:]
        hit = wet & dry_ahead & (rise > L.RISE_TH) & (t - self.last_hit > L.COOLDOWN)
        rows, cols = np.nonzero(hit)
        self.last_hit[rows, cols] = t
        self._wet = wet
        return [(self.gx[c], self.gy[r], eta[r, c], rise[r, c], r, c) for r, c in zip(rows, cols)]

    # ── 최적화 2·3: 그리기 ─────────────────────────────────────────
    def draw_effects(self, scn, cam_pos):
        """2: 수면 높이는 감지 때 계산한 것을 다시 씀 (부모는 격자 전체를 다시 계산했음).
        3: 거품 그림은 2x2 칸을 한 장으로 묶고, 멀리 있는 것은 생략, 장수 상한."""
        orig_grid, grid = L.wave_grid, getattr(self, "_grid", None)
        if grid is not None:
            L.wave_grid = lambda gx, gy, t: (grid[1], grid[2]) if gx is self.gx else orig_grid(gx, gy, t)
        foamtex = "foamtex" in self.fx
        self.fx.discard("foamtex")
        t0 = time.perf_counter()
        try:
            super().draw_effects(scn, cam_pos)
        finally:
            L.wave_grid = orig_grid
            if foamtex:
                self.fx.add("foamtex")
        if foamtex and grid is not None:
            self._draw_foam_blocks(scn, cam_pos, grid[1])
        self.draw_ms = 1000 * (time.perf_counter() - t0)

    def _draw_foam_blocks(self, scn, cam_pos, eta_g):
        b = FOAM_BLOCK
        nr, nc = (self.nrow // b) * b, (self.ncol // b) * b
        F = self.foam[:nr, :nc].reshape(nr // b, b, nc // b, b)
        Fb = F.max(axis=(1, 3))
        br, bc = np.nonzero(Fb > L.FOAM_MIN)
        if not len(br):
            return
        cr, cc = br * b + b // 2, bc * b + b // 2                        # 묶음 가운데 칸
        x, y = self.gx[cc], self.gy[cr]
        dist = np.hypot(x - cam_pos[0], y - cam_pos[1])
        keep = dist < FOAM_LOD_DIST
        br, bc, cr, cc, x, y, dist = br[keep], bc[keep], cr[keep], cc[keep], x[keep], y[keep], dist[keep]
        f = Fb[br, bc]
        if len(f) > FOAM_MAX:                                            # 진하고 가까운 것부터
            order = np.argsort(-(f / (1 + dist / 10)))[:FOAM_MAX]
            br, bc, cr, cc, x, y, f = br[order], bc[order], cr[order], cc[order], x[order], y[order], f[order]
        z = np.maximum(eta_g[cr, cc], self.terrain[cr, cc]) + 0.02
        j = self.foam_jitter[cr, cc]
        off = self.foam_offset[cr, cc] * b
        matid = self.spr_mat["foam"]
        g0 = scn.ngeom
        for i in range(len(f)):
            if g0 + i >= scn.maxgeom:
                break
            g = scn.geoms[g0 + i]
            half = L.GRID * 0.85 * b * j[i]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([half, half, 0.0005]),
                                np.array([x[i] + off[i, 0], y[i] + off[i, 1], z[i]]), self.foam_mat[cr[i], cc[i]],
                                np.array([1, 1, 1, min(0.99, 1.1 * f[i])], np.float32))
            L._decor(g, 0.85)
            g.matid = matid
            scn.ngeom = g0 + i + 1

    def _check_contacts(self):
        """부모와 같되, 바위 3개·해안 지형을 모두 기록."""
        now = {}
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            if self.boat_id not in (b1, b2):
                continue
            other = con.geom2 if b1 == self.boat_id else con.geom1
            name = self._contact_names.get(other)
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


def run_viewer(lab):
    import mujoco.viewer
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE = 265, 264, 263, 262, 32

    def on_key(key):
        r = lab.robot
        if r is None:
            return
        if key in (KEY_UP, KEY_DOWN):
            r.set_command(float(np.clip(r.throttle + (0.1 if key == KEY_UP else -0.1), -1, 1)), r.steer)
        elif key in (KEY_LEFT, KEY_RIGHT):
            lab.autopilot = False
            r.set_command(r.throttle, float(np.clip(r.steer + (0.2 if key == KEY_LEFT else -0.2), -1, 1)))
        elif key == KEY_SPACE:
            r.set_command(0.0, 0.0)
        elif key == ord("A"):
            lab.autopilot = True
        print(f"[배] 전진 {r.throttle:+.1f} · 조향 {r.steer:+.1f} · 자동조종 {'켬' if lab.autopilot else '끔'}")

    dt = lab.m.opt.timestep
    steps_per_frame = max(1, round((1 / 60) / dt))
    with mujoco.viewer.launch_passive(lab.m, lab.d, key_callback=on_key) as v:
        v.opt.geomgroup[3] = 1
        v.cam.lookat[:] = (*lab.focus, 0.2)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 30, 30, -25
        if lab.robot is not None:
            print("[배] 방향키 ↑↓ 전진·후진, ←→ 조향(자동조종 꺼짐), A 자동조종 켬, 스페이스 멈춤. 카메라가 배를 따라감")
        else:
            print("[물] 패치(50 m)가 카메라 보는 곳(lookat)을 따라감. 마우스 오른쪽 드래그로 이동")
        acc = dict(step=0.0, mesh=0.0, draw=0.0, sync=0.0, n=0)
        t_report = time.perf_counter()
        while v.is_running():
            w0 = time.perf_counter()
            with v.lock():
                for _ in range(steps_per_frame):
                    lab.step()
                if lab.robot is not None:
                    v.cam.lookat[:2] = lab.d.xpos[lab.boat_id][:2]
                else:
                    lab.focus = np.array(v.cam.lookat[:2])
                w1 = time.perf_counter()
                hid = lab.update_water_mesh()
                w2 = time.perf_counter()
                v.user_scn.ngeom = 0
                if lab.fx:
                    lab.draw_effects(v.user_scn, L.cam_position(v.cam))
                w3 = time.perf_counter()
            v.update_hfield(hid)
            v.sync()
            w4 = time.perf_counter()
            acc["step"] += w1 - w0; acc["mesh"] += w2 - w1; acc["draw"] += w3 - w2; acc["sync"] += w4 - w3
            acc["n"] += 1
            if w4 - t_report > 2.0:
                n = acc["n"]
                print(f"[렉] 물리+효과 {1000 * acc['step'] / n:.1f} · 메시 {1000 * acc['mesh'] / n:.1f} · "
                      f"효과 그리기 {1000 * acc['draw'] / n:.1f} · 화면 전송(sync) {1000 * acc['sync'] / n:.1f} ms/프레임 "
                      f"| 실제 {n / (w4 - t_report):.0f} fps | 패치 중심 {lab.center.round(0)}", flush=True)
                acc = dict(step=0.0, mesh=0.0, draw=0.0, sync=0.0, n=0)
                t_report = w4
            lag = steps_per_frame * dt - (time.perf_counter() - w0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    root = L.REPO_ROOT
    if root not in sys.path:
        sys.path.insert(0, root)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "fx"], default="fx")
    ap.add_argument("--fx", default="spray+foamtex+bulge")
    ap.add_argument("--sea", choices=["three", "team", "regular", "irregular"], default="three")
    ap.add_argument("--full", action="store_true", help="패치 대신 바다 전체를 계산 (렉 비교용)")
    ap.add_argument("--grid", type=float, default=GRID_DEFAULT, help="감지·거품 격자 [m]")
    ap.add_argument("--mesh-grid", type=float, default=MESH_DEFAULT, help="화면 메시 격자 [m]")
    ap.add_argument("--robot", action="store_true")
    ap.add_argument("--throttle", type=float, default=None)
    ap.add_argument("--no-autopilot", action="store_true")
    ap.add_argument("--no-drag-fix", action="store_true")
    args = ap.parse_args()
    fx = set(args.fx.split("+")) if args.mode == "fx" else set()
    lab = HanagaeLab(args.mode, fx=fx, sea=make_hanagae_sea(args.sea), robot=args.robot, throttle=args.throttle,
                     drag_fix=not args.no_drag_fix, autopilot=not args.no_autopilot, full=args.full,
                     grid=args.grid, mesh_grid=args.mesh_grid)
    print(f"[맵] 하나개 크기 · 물 {'바다 전체' if args.full else '패치 %gx%g m' % (2 * lab.half[0], 2 * lab.half[1])}"
          f" · 메시 {lab.vcol}x{lab.vrow} ({args.mesh_grid} m) · 감지 {lab.ncol}x{lab.nrow} ({args.grid} m)")
    run_viewer(lab)
