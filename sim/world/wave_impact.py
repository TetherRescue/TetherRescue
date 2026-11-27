"""해안·암초에 닿는 파도의 화면 효과.

수면 메시는 높이만 따라가서 모래·바위를 통과한다. 여기서는
  - 육지와 암초 안으로 들어간 꼭짓점을 경계로 접고
  - 수면이 빠르게 올라갈 때 물보라·거품·짧은 솟음을 그린다.
부력에 쓰는 수면 높이는 바꾸지 않는다.
"""

import math

import numpy as np

from sim import config
from sim.world.coast_map import REEF_CENTER, REEF_LENGTH, REEF_WIDTH, SHORE_X

RISE_TH = 0.25          # 이 속도(m/s)보다 빠르게 올라올 때만 철썩
COOLDOWN = 0.75
MAX_HITS = 4
MAX_DROPS = 80
DROP_LIFE = 0.7
FOAM_LIFE = 1.15
BULGE_T = 0.55
BULGE_SX = 1.3
BULGE_SY = 2.4
_G = 9.81
_WHITE = np.array([0.93, 0.96, 0.98], dtype=np.float32)


def _field(fn, x, y, t):
    return np.asarray(fn(x, y, t), dtype=float).reshape(-1)


class WaveImpact:
    """splash sea(eta, rise)가 모래사장·돌섬에 부딪힐 때의 물보라."""

    def __init__(self, sea, *, shore_x=SHORE_X, reef_center=REEF_CENTER,
                 reef_length=REEF_LENGTH, reef_width=REEF_WIDTH):
        self.sea = sea
        self.shore_x = float(shore_x)
        self.beach_top = config.WaterParams.SURFACE_Z + 0.4
        self.reef = (float(reef_center[0]), float(reef_center[1]), float(reef_length) / 2, float(reef_width) / 2)
        shore_y = np.linspace(-42.0, 42.0, 15)
        ang = np.linspace(-1.05, 1.05, 7)
        rx, ry = self.reef[2] + 0.4, self.reef[3] + 0.35
        self.sx = np.concatenate([
            np.full(shore_y.shape, self.shore_x - 0.35),
            self.reef[0] - rx * np.cos(ang),
        ])
        self.sy = np.concatenate([shore_y, self.reef[1] + ry * np.sin(ang)])
        self.kind = np.concatenate([np.zeros(len(shore_y)), np.ones(len(ang))]).astype(int)
        self.cool = np.full(len(self.sx), -1e9)
        self.drops = np.zeros((0, 8))
        self.foam = np.zeros((0, 5))
        self.bumps = np.zeros((0, 4))
        self._t = None
        self._rng = np.random.default_rng(1)

    def update(self, t):
        """물보라를 한 프레임 진행하고, 새로 부딪힌 곳을 만든다."""
        t = float(t)
        dt = 0.0 if self._t is None else t - self._t
        if dt < 0:
            self.drops = np.zeros((0, 8))
            self.foam = np.zeros((0, 5))
            self.bumps = np.zeros((0, 4))
            self.cool[:] = -1e9
            dt = 0.0
        self._t = t
        if dt > 0 and len(self.drops):
            self.drops[:, 0] += self.drops[:, 3] * dt
            self.drops[:, 1] += self.drops[:, 4] * dt
            self.drops[:, 2] += self.drops[:, 5] * dt
            self.drops[:, 5] -= _G * dt
            age = t - self.drops[:, 6]
            under = self.drops[:, 2] < _field(self.sea.eta, self.drops[:, 0], self.drops[:, 1], t) - 0.04
            self.drops = self.drops[(age < self.drops[:, 7]) & ~(under & (self.drops[:, 5] < 0))]
        if len(self.foam):
            self.foam = self.foam[t - self.foam[:, 3] < self.foam[:, 4]]
        if len(self.bumps):
            self.bumps = self.bumps[t - self.bumps[:, 2] < BULGE_T]
        if not hasattr(self.sea, 'rise'):
            return
        eta = _field(self.sea.eta, self.sx, self.sy, t)
        rise = _field(self.sea.rise, self.sx, self.sy, t)
        ready = np.flatnonzero((rise > RISE_TH) & (eta > 0.02) & (t - self.cool > COOLDOWN))
        if len(ready) == 0:
            return
        ready = ready[np.argsort(rise[ready])[-MAX_HITS:]]
        for i in ready:
            self.cool[i] = t
            self._spawn(float(self.sx[i]), float(self.sy[i]), float(eta[i]), float(rise[i]), t, int(self.kind[i]))

    def _spawn(self, x, y, eta, rise, t, kind):
        rng = self._rng
        strength = min(rise / 0.6, 1.5)
        seaward = -1.0 if kind == 0 or x <= self.reef[0] else 1.0
        drops = []
        for _ in range(4):
            drops.append((
                x + rng.uniform(-0.12, 0.12), y + rng.uniform(-0.2, 0.2), eta + 0.05,
                seaward * rng.uniform(0.4, 1.8) * strength,
                rng.uniform(-0.7, 0.7) * strength,
                rng.uniform(1.8, 3.6) * strength,
                t, DROP_LIFE,
            ))
        self.drops = np.vstack([self.drops, drops])[-MAX_DROPS:]
        self.foam = np.vstack([self.foam, [[x, y, eta + 0.02, t, FOAM_LIFE]]])[-40:]
        self.bumps = np.vstack([self.bumps, [[x + 0.15 * seaward, y, t, 0.18 + 0.16 * strength]]])[-24:]

    def dress(self, verts, t):
        """수면 메시가 모래·바위 속을 지나지 않게 접고, 부딪힌 자국만 살짝 솟게 한다."""
        out = np.array(verts, dtype=float, copy=True)
        if len(self.bumps):
            lift = np.zeros(len(out))
            for bx, by, born, height in self.bumps:
                tau = float(t) - born
                if 0 <= tau < BULGE_T:
                    env = math.sin(math.pi * tau / BULGE_T) ** 1.5
                    lift = np.maximum(lift, height * env * np.exp(
                        -((out[:, 0] - bx) / BULGE_SX) ** 2 - ((out[:, 1] - by) / BULGE_SY) ** 2))
            out[:, 2] += lift
        land = out[:, 0] > self.shore_x
        if land.any():
            out[land, 0] = self.shore_x - 0.02
            out[land, 2] = np.minimum(out[land, 2], self.beach_top - 0.02)
        cx, cy, rx, ry = self.reef
        dx = out[:, 0] - cx
        dy = out[:, 1] - cy
        near = np.hypot(dx, dy) < 1e-3
        dx = np.where(near, -rx * 0.5, dx)
        dy = np.where(near, 0.0, dy)
        nx, ny = dx / rx, dy / ry
        radius = np.hypot(nx, ny)
        inside = radius < 1.0
        if inside.any():
            scale = np.ones_like(radius)
            scale[inside] = 1.0 / np.maximum(radius[inside], 1e-3)
            out[inside, 0] = cx + dx[inside] * scale[inside]
            out[inside, 1] = cy + dy[inside] * scale[inside]
            out[inside, 2] = np.minimum(out[inside, 2], config.WaterParams.SURFACE_Z + 0.02)
        return out

    def draw(self, scene, t):
        """viewer.user_scn 에 물보라와 거품을 덧그린다. 이미 있는 geom 은 지우지 않는다."""
        import mujoco

        t = float(t)
        eye = np.eye(3).ravel()
        if len(self.foam) and scene.ngeom < scene.maxgeom:
            z = _field(self.sea.eta, self.foam[:, 0], self.foam[:, 1], t) + 0.03
            age = (t - self.foam[:, 3]) / self.foam[:, 4]
            for i in range(len(self.foam)):
                if scene.ngeom >= scene.maxgeom:
                    return
                fade = max(0.0, 1.0 - float(age[i]))
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                    np.array([0.55, 0.32, 0.035]),
                    np.array([self.foam[i, 0], self.foam[i, 1], z[i]]),
                    eye, np.append(_WHITE, np.float32(0.72 * fade)),
                )
                scene.ngeom += 1
        for drop in self.drops:
            if scene.ngeom >= scene.maxgeom:
                return
            age = (t - drop[6]) / drop[7]
            radius = 0.06 + 0.05 * (1.0 - age)
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius),
                drop[:3], eye, np.append(_WHITE, np.float32(0.9 * max(0.0, 1.0 - age))),
            )
            scene.ngeom += 1
