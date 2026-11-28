"""Sprite-based splash for the personal lab (port of tools/splash_lab spray + foamtex + bulge).

Visual only: nothing here touches forces. Uses the sprite PNGs from tools/splash_lab/sprites
(made by tools/splash_lab/sprites.py on first use).

What it draws
  rock face  : surface bulges up the rock, then at the top a burst of water-sheet / droplet / mist
               sprites flies seaward under gravity + air drag, fading as it falls back
  beach line : small bulge, a few low droplets + mist, foam patch on the swash
  break line : (shallow water) whitewater foam patches drifting to the shore, some low mist

Lag control (all measured per frame in `ms`)
  - hits are tested only at a few sample points (rock ring, beach line, break line), not on a grid
  - every sprite is advanced in closed form (no per-step integration), all numpy-vectorised
  - hard caps on live sprites / foam patches; one sea.eta call per frame for all sprites
"""
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SPLASH_LAB = os.path.join(REPO_ROOT, 'tools', 'splash_lab')
for path in (REPO_ROOT, SPLASH_LAB):
    if path not in sys.path:
        sys.path.insert(0, path)

from sim import config  # noqa: E402
from sim.world.wave_impact import WaveImpact  # noqa: E402
import sprites as sprite_files  # noqa: E402  (tools/splash_lab/sprites.py)

G = 9.81
# hit detection
RISE_TH = 0.25          # surface must rise faster than this [m/s]
COOLDOWN = 0.9          # same sample point cannot fire again within [s]
IMPACT_DELAY = 0.2      # bulge climbs the rock, spray fires at the top after this [s]
MAX_EVENTS = 3          # new events per frame at most
BREAK_SPRAY_CHANCE = 0.25
# bulge (used by WaveImpact.dress through self.bumps = (x, y, born, height))
BULGE_ROCK = (0.30, 0.30)   # height = a + b * strength [m]
BULGE_BEACH = 0.10
RUNUP_GAIN = 0.6        # crest climbs the seaward rock face by this fraction of its height (visual)
RUNUP_REACH = 1.5       # run-up fades out this far from the rock [m]
# beach swash: the wave at a reference line just offshore is replayed up the sand slope with a delay
SWASH_REF = -3.0        # x of the reference line [m] (seaward of the waterline x = 0)
SWASH_SPEED = 2.0       # how fast the water sheet runs up / back down the slope [m/s]
SWASH_GAIN = 1.3        # run-up height = gain x wave height at the reference line
SWASH_MAX = 0.12        # thickest the water sheet on the sand can be [m]
SWASH_MIN = 0.012       # thinner than this = dry sand
SWASH_HIDE = 0.05       # dry: water mesh is tucked this far under the sand [m]
SWASH_HISTORY = 15.0    # [s] of reference-line history kept (longer than the slowest run-up)
# spray pieces per rock event (beach gets a small share)
SPRAY_N = dict(sheet=6, drop=12, mist=4)
SPRAY_EMIT = 0.3        # pieces come out over this time, not all at once [s]
MAX_SPRAY = 260
SPLASH_UP, SPLASH_OUT, SPLASH_DRAG = 0.75, 0.8, 1.5
SPLASH_FADE_IN, MIST_FADE_IN = 0.15, 0.06
SPLASH_ALPHA = 1.3     # airborne spray opacity boost (thin sprites over blue water read as grey otherwise)
SPLASH_LAND_FADE = 0.3  # start fading this high above the water while falling [m]
SPLASH_FALL_FADE = 1.5  # fully faded at this falling speed [m/s] (disperses at the top, no "diving")
SPLASH_FALL_GROW = 0.8
SPLASH_LIFT = 0.8       # keep the bottom edge of a sprite above the surface
# foam patches: x y born life size yaw vx alpha
MAX_FOAM = 90
FOAM_LIFE = (1.6, 2.6)
FOAM_LIFT = 0.07         # foam above the true surface: the 0.5 m mesh cuts across troughs (chord above curve)
WHITEWATER_SPEED = 0.35  # foam from breaking waves drifts shoreward [m/s]

VIEW_COS = 0.35        # draw() skips pieces more than ~70 deg off the view direction
SPRITE_NAMES = sprite_files.NAMES   # puff, burst, foam, drop0-2, sheet0-1, mist0-1


def _decor(geom, emission):
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)   # no shadows
    geom.emission = emission
    geom.specular = 0.0
    geom.shininess = 0.0
    geom.reflectance = 0.0


def cam_position(cam):
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.array(cam.lookat) - cam.distance * fwd


def cam_forward(cam):
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def billboard(pos, cam_pos, roll=0.0):
    """Rotation (row-major 9) whose local z faces the camera."""
    z = cam_pos - pos
    z /= max(np.linalg.norm(z), 1e-9)
    x = np.cross([0.0, 0.0, 1.0], z)
    nx = np.linalg.norm(x)
    x = x / nx if nx > 1e-6 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        x, y = c * x + s * y, -s * x + c * y
    return np.column_stack([x, y, z]).ravel()


class SplashFX(WaveImpact):
    """Module part (build/bind: sprite textures) + Water.impact (dress: mesh fold + bulge)."""

    REQUIRES = {}

    def __init__(self, sea, *, shore_x, rock_center, rock_half, half_y, break_x, seed=0,
                 bed_slope=0.08, beach_width=12.0, beach_top=0.8, land_top=1.0):
        super().__init__(sea, shore_x=shore_x, reef_center=rock_center,
                         reef_length=2 * rock_half[0], reef_width=2 * rock_half[1])
        self.profile = (bed_slope, beach_width, beach_top, land_top)
        self.ref_y = np.linspace(-half_y - 1, half_y + 1, 105)
        self._hist_t, self._hist_eta = [], []
        # the run-up history is touched by the mesh thread (swash mesh) AND the main thread (sprites):
        # without a lock the time list and the height list could differ in length (IndexError, 2026-11-28)
        import threading
        self._hist_lock = threading.Lock()
        self.rng = np.random.default_rng(seed)
        shore_y = np.linspace(-half_y + 2, half_y - 2, 11)
        break_y = np.arange(-half_y + 1.5, half_y - 1.4, 1.5)
        ang = np.linspace(-1.1, 1.1, 9)
        rx, ry = self.reef[2] + 0.4, self.reef[3] + 0.35
        self.sx = np.concatenate([np.full(shore_y.shape, shore_x - 0.35), self.reef[0] - rx * np.cos(ang),
                                  np.full(break_y.shape, break_x)])
        self.sy = np.concatenate([shore_y, self.reef[1] + ry * np.sin(ang), break_y])
        self.kind = np.concatenate([np.zeros(len(shore_y)), np.ones(len(ang)),
                                    np.full(len(break_y), 2)]).astype(int)
        self.cool = np.full(len(self.sx), -1e9)
        self.pending = []                    # (t_fire, x, y, strength, top)
        self.spray = np.zeros((0, 17))       # x0 y0 z0 vx vy vz born life size0 grow mat spin roll0 drag align g alpha
        self.foam = np.zeros((0, 8))         # x y born life size yaw vx alpha
        self.mat = {}
        self.ms = 0.0
        # False when the water around the rock is computed (swe_rock.py): no faked run-up / bulge there
        self.rock_runup = True

    # ── module shape ──────────────────────────────────────────
    def build(self, spec):
        sprite_files.load_sprites()          # makes the PNGs if missing
        for name in SPRITE_NAMES:
            tex = f'spr_{name}'
            spec.add_texture(name=tex, type=mujoco.mjtTexture.mjTEXTURE_2D,
                             file=os.path.join(sprite_files.DIR, f'{name}.png'), nchannel=4)
            mat = spec.add_material(name=tex)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex

    def bind(self, model):
        for name in SPRITE_NAMES:
            mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f'spr_{name}')
            assert mid >= 0, f'sprite material spr_{name} missing'
            self.mat[name] = mid

    def add_forces(self, model, data, acc):
        pass

    # ── wave mesh dressing (called by Water.update_surface_visual) ──
    def dress(self, verts, t):
        """Fold the mesh out of the beach/rock and add bulges, WITHOUT flattening at the rock.

        Team WaveImpact.dress clamps folded rock vertices to z <= 0.02 m, which made the water
        touching the rock look still. Here a folded vertex keeps its wave height, and on the
        seaward face it gets run-up: water climbing the rock as the crest arrives.
        """
        out = np.array(verts, dtype=float, copy=True)
        if len(self.bumps):
            lift = np.zeros(len(out))
            for bx, by, born, height in self.bumps:
                tau = float(t) - born
                if 0 <= tau < 0.55:
                    env = math.sin(math.pi * tau / 0.55) ** 1.5
                    np.maximum(lift, height * env * np.exp(-((out[:, 0] - bx) / 1.3) ** 2
                                                           - ((out[:, 1] - by) / 2.4) ** 2), out=lift)
            out[:, 2] += lift
        # Beach: no fold into a wall at the waterline. Water slides up the sand and back (swash),
        # and wherever the sand is dry the mesh is tucked just under it.
        band = out[:, 0] > SWASH_REF
        if band.any():
            out[band, 2] = self.visible_z(out[band, 0], out[band, 1], out[band, 2], t)
        # Rock: the mesh is NOT folded. Folding vertices onto an ellipse that does not match the real
        # (lumpy) rock left a visible water rim around it. The rock is opaque, so it simply hides the
        # water inside it and the true waterline is where the two surfaces cross.
        # Run-up: crests climb the seaward face, fading smoothly with distance from the rock.
        cx, cy, rx, ry = self.reef
        nx, ny = (out[:, 0] - cx) / rx, (out[:, 1] - cy) / ry
        radius = np.hypot(nx, ny)
        near = radius < 1.0 + RUNUP_REACH / min(rx, ry)
        if self.rock_runup and near.any():
            r = np.maximum(radius[near], 1e-3)
            gap = np.maximum(r - 1.0, 0.0) * min(rx, ry)              # ~ distance outside the rock [m]
            seaward = np.clip(-nx[near] / r, 0.0, 1.0) ** 0.7
            fade = np.exp(-(gap / (0.5 * RUNUP_REACH)) ** 2)
            out[near, 2] += RUNUP_GAIN * seaward * fade * np.maximum(out[near, 2], 0.0)
        return out

    # ── beach swash ───────────────────────────────────────────
    def terrain_z(self, x):
        """Sand / seabed height of the lab map along x (straight shore at x = shore_x)."""
        slope, width, top, land = self.profile
        u = np.asarray(x, float) - self.shore_x
        return np.where(u < 0, slope * u, np.where(u < width, top * u / width, land))

    def _record(self, t):
        """Keep the wave at the reference line over time (one cheap 105-point query per new time)."""
        if self._hist_t and t <= self._hist_t[-1]:
            if t < self._hist_t[-1] - 1.0:                       # time went back (reset)
                self._hist_t, self._hist_eta = [], []
            else:
                return
        self._hist_t.append(t)
        self._hist_eta.append(np.asarray(self.sea.eta(np.full(len(self.ref_y), SWASH_REF), self.ref_y, t), float))
        while self._hist_t[-1] - self._hist_t[0] > SWASH_HISTORY:
            self._hist_t.pop(0)
            self._hist_eta.pop(0)

    def _swash_level(self, x, y, t):
        """Water level replayed up the slope: reference wave delayed by the travel time from SWASH_REF."""
        with self._hist_lock:
            self._record(t)
            T = np.asarray(self._hist_t)
            H = np.asarray(self._hist_eta)
        tq = t - np.maximum(x - SWASH_REF, 0.0) / SWASH_SPEED
        ti = np.clip(np.searchsorted(T, tq) - 1, 0, len(T) - 1)
        tn = np.minimum(ti + 1, len(T) - 1)
        span = np.where(tn > ti, T[tn] - T[ti], 1.0)
        ft = np.clip((tq - T[ti]) / span, 0.0, 1.0)
        yi = np.clip((y - self.ref_y[0]) / (self.ref_y[1] - self.ref_y[0]), 0, len(self.ref_y) - 1.001)
        j = yi.astype(int)
        fy = yi - j
        h0 = H[ti, j] * (1 - fy) + H[ti, j + 1] * fy
        h1 = H[tn, j] * (1 - fy) + H[tn, j + 1] * fy
        return SWASH_GAIN * (h0 * (1 - ft) + h1 * ft)

    def swash_water(self, x, y, eta, t):
        """Water level before the sand clamp: wave seaward of SWASH_REF, blended into the run-up level."""
        x, y, eta = (np.asarray(v, float) for v in (x, y, eta))
        level = self._swash_level(x, y, t)
        b = np.clip((x - SWASH_REF) / (self.shore_x - SWASH_REF), 0.0, 1.0)
        b = b * b * (3 - 2 * b)
        return (1 - b) * eta + b * level

    def visible_z(self, x, y, eta, t):
        """Shown water height near/over the beach: never through the sand.

        Seaward of SWASH_REF: the wave itself. Between SWASH_REF and the waterline the wave blends
        into the replayed run-up level; on the sand the water is a thin sheet that exists only
        where the run-up level is above the sand. Dry places are tucked under the sand.
        """
        x, y, eta = (np.asarray(v, float) for v in (x, y, eta))
        water = self.swash_water(x, y, eta, t)
        ground = self.terrain_z(x)
        depth = water - ground
        on_sand = x > self.shore_x
        shown = np.where(on_sand, ground + np.minimum(depth, SWASH_MAX), water)
        return np.where(depth > SWASH_MIN, shown, ground - SWASH_HIDE)

    # ── visible surface = wave + bulge ────────────────────────
    def surface_z(self, x, y, t):
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = np.asarray(self.sea.eta(x, y, t), float).reshape(x.shape)
        if len(self.bumps):
            lift = np.zeros_like(z)
            for bx, by, born, h in self.bumps:
                tau = t - born
                if 0 <= tau < 0.55:
                    env = math.sin(math.pi * tau / 0.55) ** 1.5
                    np.maximum(lift, h * env * np.exp(-((x - bx) / 1.3) ** 2 - ((y - by) / 2.4) ** 2), out=lift)
            z = z + lift
        band = x > SWASH_REF
        if band.any():
            z = np.array(z, dtype=float, copy=True)
            z[band] = np.maximum(self.visible_z(x[band], y[band], z[band], t), self.terrain_z(x[band]))
        # same run-up as dress(): otherwise spray/foam near the rock sit below the raised mesh
        if not self.rock_runup:
            return z
        cx, cy, rx, ry = self.reef
        nx, ny = (x - cx) / rx, (y - cy) / ry
        r = np.maximum(np.hypot(nx, ny), 1e-3)
        gap = np.maximum(r - 1.0, 0.0) * min(rx, ry)
        seaward = np.clip(-nx / r, 0.0, 1.0) ** 0.7
        fade = np.exp(-(gap / (0.5 * RUNUP_REACH)) ** 2)
        return z + RUNUP_GAIN * seaward * fade * np.maximum(z, 0.0)

    # ── per frame ────────────────────────────────────────────
    def update(self, t):
        import time
        t0 = time.perf_counter()
        t = float(t)
        if self._t is not None and t < self._t:          # time went back (reset)
            self.spray, self.foam, self.bumps = np.zeros((0, 17)), np.zeros((0, 8)), np.zeros((0, 4))
            self.pending, self.cool[:] = [], -1e9
        self._t = t
        if len(self.spray):
            self.spray = self.spray[t - self.spray[:, 6] < self.spray[:, 7]]
        if len(self.foam):
            self.foam = self.foam[t - self.foam[:, 2] < self.foam[:, 3]]
        if len(self.bumps):
            self.bumps = self.bumps[t - self.bumps[:, 2] < 0.55]

        eta = np.asarray(self.sea.eta(self.sx, self.sy, t), float).reshape(-1)
        rise = np.asarray(self.sea.rise(self.sx, self.sy, t), float).reshape(-1)
        ready = np.flatnonzero((rise > RISE_TH) & (eta > 0.02) & (t - self.cool > COOLDOWN))
        for i in ready[np.argsort(rise[ready])[-MAX_EVENTS:]]:
            self.cool[i] = t
            self._hit(float(self.sx[i]), float(self.sy[i]), float(eta[i]), float(rise[i]), t, int(self.kind[i]))

        fire = [p for p in self.pending if p[0] <= t]
        self.pending = [p for p in self.pending if p[0] > t]
        for _, x, y, s, top in fire:
            self._emit_spray(x, y, s, top, t, scale=1.0)
        self.ms = 1000 * (time.perf_counter() - t0)

    def _hit(self, x, y, eta, rise, t, kind):
        U = self.rng.uniform
        s = min(rise / 0.6, 1.5)
        if kind == 1:                                     # rock: bulge climbs, spray fires at the top
            h = BULGE_ROCK[0] + BULGE_ROCK[1] * s
            if self.rock_runup:
                self.bumps = np.vstack([self.bumps, [[x - 0.1, y, t, h]]])[-24:]
            self.pending.append((t + IMPACT_DELAY, x, y, s, h))
            self._add_foam(x - 0.3, y, t, size=U(0.5, 0.8), vx=-0.1)
        elif kind == 0:                                   # beach: small swash
            self.bumps = np.vstack([self.bumps, [[x - 0.15, y, t, BULGE_BEACH * (0.6 + 0.4 * s)]]])[-24:]
            self._emit_spray(x, y, 0.45 * s, 0.1, t, scale=0.25)
            for _ in range(2):
                self._add_foam(x - U(0.2, 0.9), y + U(-.6, .6), t, size=U(0.45, 0.7), vx=0.0)
        else:                                             # breaking line: whitewater drifts ashore
            for _ in range(2):
                self._add_foam(x + U(-.3, .3), y + U(-.6, .6), t, size=U(0.5, 0.85), vx=WHITEWATER_SPEED)
            if self.rng.random() < BREAK_SPRAY_CHANCE:
                self._emit_spray(x, y, 0.4 * s, 0.05, t, scale=0.2, kinds=('mist',))

    def _add_foam(self, x, y, t, size, vx):
        U = self.rng.uniform
        row = [[x, y, t, U(*FOAM_LIFE), size, U(0, 2 * math.pi), vx, U(0.75, 1.0)]]
        self.foam = np.vstack([self.foam, row])[-MAX_FOAM:]

    def _emit_spray(self, x, y, s, top, t, scale=1.0, kinds=('sheet', 'drop', 'mist')):
        """One event -> water sheets, droplet clusters and mist (same motion model as splash_lab)."""
        U = self.rng.uniform
        f = 0.6 + 0.6 * s
        mats = {k: [self.mat[n] for n in SPRITE_NAMES if n.rstrip('0123456789') == k] for k in kinds}
        rows = []
        for kind in kinds:
            for _ in range(max(1, int(round(SPRAY_N[kind] * scale)))):
                born = t + U(0, SPRAY_EMIT) + (U(0.0, 0.1) if kind == 'mist' else 0.0)
                px, py = x - 0.15 + U(-.1, .1), y + U(-.5, .5)
                bz = U(0.02, 0.3)        # height above the visible surface, resolved at draw time via z0 offset
                mat = float(self.rng.choice(mats[kind]))
                if kind == 'sheet':      # thin water sheet: fast, stretched along its flight
                    v = (U(-2.4, -0.6) * f * SPLASH_OUT, U(-1.2, 1.2) * f * SPLASH_OUT, U(4.0, 7.0) * f * SPLASH_UP)
                    rows.append((px, py, bz, *v, born, U(0.7, 1.0), U(0.3, 0.45), 1.0, mat,
                                 0.0, 0.0, 0.6 * SPLASH_DRAG, 1, G, 0.95))
                elif kind == 'drop':     # droplet cluster: spreads wide, spins, falls
                    v = (U(-3.0, -0.4) * f * SPLASH_OUT, U(-1.8, 1.8) * f * SPLASH_OUT, U(3.0, 6.5) * f * SPLASH_UP)
                    rows.append((px, py, bz, *v, born, U(0.9, 1.4), U(0.22, 0.38), 0.7, mat,
                                 U(-3, 3), U(0, 2 * math.pi), 0.4 * SPLASH_DRAG, 0, G, 0.9))
                else:                    # mist: late, floats up slowly, grows
                    v = (U(-0.6, -0.1), U(-0.3, 0.3), U(0.6, 1.4) * f)
                    rows.append((px - 0.2, py, bz + top * U(0.2, 0.7), *v, born, U(0.5, 0.8),
                                 U(0.4, 0.65) * f, 1.4, mat, U(-.6, .6), U(0, 2 * math.pi), 2.0, 0, 0.4, 0.4))
        # z0 column holds "height above the surface at the emit point": convert once to world z
        rows = np.array(rows, float)
        rows[:, 2] += self.surface_z(rows[:, 0], rows[:, 1], t)
        self.spray = np.vstack([self.spray, rows])[-MAX_SPRAY:]

    def draw(self, scene, t, cam_pos=None, cam_fwd=None):
        import time
        t0 = time.perf_counter()
        t = float(t)
        cam_pos = np.array([0.0, 0.0, 20.0]) if cam_pos is None else np.asarray(cam_pos, float)
        g = [scene.ngeom]

        def seen(x, y, z):   # skip pieces behind / far outside the view (each one costs Python time)
            if cam_fwd is None:
                return np.ones(len(x), bool)
            d = np.stack([x - cam_pos[0], y - cam_pos[1], z - cam_pos[2]], -1)
            return d @ np.asarray(cam_fwd, float) > VIEW_COS * np.linalg.norm(d, axis=1)

        def add(size, pos, mat9, alpha, emission, matid):
            if g[0] >= scene.maxgeom:
                return
            geom = scene.geoms[g[0]]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(size, float), np.asarray(pos, float),
                                mat9, np.array([1, 1, 1, min(0.99, alpha)], np.float32))
            _decor(geom, emission)
            geom.matid = matid
            g[0] += 1

        # foam patches lying on the surface (one eta call for all)
        if len(self.foam):
            F = self.foam
            tau = t - F[:, 2]
            x = np.minimum(F[:, 0] + F[:, 6] * tau, self.shore_x - 0.1)
            z = self.surface_z(x, F[:, 1], t) + FOAM_LIFT
            u = tau / F[:, 3]
            alpha = F[:, 7] * np.clip(tau / 0.15, 0, 1) * (1 - u) ** 1.2
            size = F[:, 4] * (1 + 0.35 * u)
            for i in np.nonzero(seen(x, F[:, 1], z))[0]:
                c, s_ = math.cos(F[i, 5]), math.sin(F[i, 5])
                add((size[i], size[i], 0.0005), (x[i], F[i, 1], z[i]),
                    np.array([c, -s_, 0, s_, c, 0, 0, 0, 1.0]), alpha[i], 0.85, self.mat['foam'])

        # spray pieces: closed-form motion with linear air drag
        if len(self.spray):
            P = self.spray
            tau = t - P[:, 6]
            live = tau >= 0
            P, tau = P[live], tau[live]
            if len(P):
                c, gg = P[:, 13], P[:, 15]
                e = np.exp(-c * tau)
                k = (1 - e) / c
                pos = np.stack([P[:, 0] + P[:, 3] * k, P[:, 1] + P[:, 4] * k,
                                P[:, 2] + (P[:, 5] + gg / c) * k - gg * tau / c], axis=1)
                vel = np.stack([P[:, 3] * e, P[:, 4] * e, (P[:, 5] + gg / c) * e - gg / c], axis=1)
                u = np.clip(tau / P[:, 7], 0, 1)
                mist = gg < 1.0
                alpha = (np.minimum(P[:, 16] * SPLASH_ALPHA, 1.0) * np.clip(tau / np.where(mist, MIST_FADE_IN, SPLASH_FADE_IN), 0, 1)
                         * np.where(mist, (1 - u) ** 1.5, 1 - u ** 2))
                sea = self.surface_z(pos[:, 0], pos[:, 1], t)
                falling = vel[:, 2] < 0
                alpha[falling] *= np.clip((pos[falling, 2] - sea[falling]) / SPLASH_LAND_FADE, 0, 1)
                fall = np.clip(-vel[:, 2] / SPLASH_FALL_FADE, 0, 1)
                alpha[~mist] *= (1 - fall[~mist]) ** 1.5
                alpha[pos[:, 2] < sea - 0.02] = 0
                size = P[:, 8] * (1 + P[:, 9] * tau) * (1 + SPLASH_FALL_GROW * fall * ~mist)
                pos[:, 2] = np.maximum(pos[:, 2], sea + SPLASH_LIFT * size)
                for i in np.nonzero((alpha > 0.01) & seen(pos[:, 0], pos[:, 1], pos[:, 2]))[0]:
                    if P[i, 14] > 0.5:   # sheet: stand along the flight direction
                        zc = cam_pos - pos[i]
                        zc /= max(np.linalg.norm(zc), 1e-9)
                        yv = vel[i] - (vel[i] @ zc) * zc
                        ny = np.linalg.norm(yv)
                        if ny < 1e-6:
                            continue
                        yv /= ny
                        mat9 = np.column_stack([np.cross(yv, zc), yv, zc]).ravel()
                        stretch = 1 + 0.12 * max(vel[i][2], 0.0)
                        half = (size[i] * 0.6, size[i] * stretch, 0.0005)
                    else:
                        mat9 = billboard(pos[i], cam_pos, roll=P[i, 12] + P[i, 11] * tau[i])
                        half = (size[i], size[i], 0.0005)
                    add(half, pos[i], mat9, alpha[i], 0.95, int(P[i, 10]))
        scene.ngeom = g[0]
        self.ms += 1000 * (time.perf_counter() - t0)
