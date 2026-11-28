"""Beach water look for the lab (visual only): see-through shallows, wet sand, lace foam.

Reference: a photo of swash on a sandy beach (user, 2026-11-28):
  - the shallower the water, the more the sand shows through (sand -> turquoise -> sea),
  - a white lace foam band rides the front of each run-up, and lace streaks stay behind as it drains,
  - sand that was just covered is darker (wet) for a few seconds.

How, without shaders (MuJoCo draws one colour per geom):
  1. A finer swash mesh (SWASH_STEP) over the beach band replaces the 0.5 m sea mesh there. Its
     texture coordinate u is set per vertex from the water depth, and the texture is a colour ramp
     (dry sand | wet sand | thin water ... deep sea). So the colour follows the depth, per pixel.
  2. Dry vertices that were wet a moment ago sit 4 mm above the sand with the wet-sand colour.
  3. Lace foam = flat decals with a procedural cellular foam texture (Worley edges), placed along
     the run-up front every frame plus a short-lived trail left behind.
"""
import math
import os
import struct
import zlib

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEX_DIR = os.path.join(HERE, 'textures')

SWASH_X = (-9.0, 4.0)    # [m] band covered by the swash mesh (main sea mesh kept just under it)
SWASH_STEP = 0.25        # [m]
DEPTH_FULL = 0.6         # [m] depth at which the colour reaches the open-sea colour (x ~ -7.5 m)
WET_SAND_TAU = 4.0       # [s] wet sand dries (fades back to dry colour) with this time constant
WET_SAND_LIFT = 0.004    # [m] wet-sand shell above the sand
U_WET = 0.12             # texture u: [0, U_WET] = dry -> wet sand, [U_WET, 1] = water 0 -> DEPTH_FULL

# colour ramp (u, rgb): matches SAND_RGBA at u = 0 and SEA_RGBA at u = 1
RAMP = ((0.00, (.76, .68, .50)), (U_WET - .005, (.56, .47, .33)), (U_WET + .01, (.70, .64, .50)),
        (0.30, (.58, .66, .56)), (0.50, (.30, .60, .57)), (0.72, (.14, .47, .49)), (1.00, (.06, .32, .45)))

LACE_ROW_EVERY = 4       # one foam station per 4 swash-mesh rows (1.0 m along the shore)
LACE_RANGE = 28.0        # [m] stations farther than this from the camera are skipped
LACE_VIEW_COS = 0.35     # stations outside ~70 deg of the view direction are skipped (off screen)
LACE_TRAIL_DT = (0.25, 0.45)   # [s] between trail decals at one station
LACE_TRAIL_LIFE = (1.6, 3.2)   # [s]
MAX_LACE = 120           # trail decals alive at once
LACE_LIFT = 0.012        # [m] above the shown surface (decals are also tilted to it)


# ── procedural textures (written once into tools/my_lab/textures) ──
def _png(path, img):
    h, w, c = img.shape
    raw = b''.join(b'\x00' + img[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    color = {3: 2, 4: 6}[c]
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, color, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def _ramp_png(path, width=256, height=4):
    u = np.linspace(0, 1, width)
    us = np.array([p[0] for p in RAMP])
    rgb = np.stack([np.interp(u, us, [p[1][k] for p in RAMP]) for k in range(3)], -1)
    img = np.repeat((rgb * 255).round().astype(np.uint8)[None], height, axis=0)
    _png(path, img)


def _worley(n, cells, rng):
    """F1, F2 distances (tileable) for an n x n image with cells x cells jittered points."""
    pts = (np.stack(np.meshgrid(np.arange(cells), np.arange(cells)), -1).reshape(-1, 2)
           + rng.uniform(0.1, 0.9, (cells * cells, 2))) / cells
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n
    best1 = np.full((n, n), 9.0)
    best2 = np.full((n, n), 9.0)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for px, py in pts + (ox, oy):
                d = np.hypot(xx - px, yy - py)
                best2 = np.where(d < best1, best1, np.minimum(best2, d))
                best1 = np.minimum(best1, d)
    return best1 * cells, best2 * cells


def _lace_png(path, seed, n=192):
    """White foam lace: thick bubbly rims around dark holes, soft round outline, alpha-only shape."""
    rng = np.random.default_rng(seed)
    f1, f2 = _worley(n, 7, rng)
    g1, g2 = _worley(n, 15, rng)
    # thick, bubbly bands between big cells; the band width wanders so holes vary in size
    h1, h2 = _worley(n, 3, rng)
    width = 0.42 + 0.30 * (h1 / h1.max())
    rim = np.clip(1 - (f2 - f1) / width, 0, 1) ** 0.7
    fine = np.clip(1 - (g2 - g1) / 0.35, 0, 1)                # bubbles: small holes inside the bands
    lace = np.clip(1.15 * rim - 0.35 * (1 - fine) * rim, 0, 1)
    lace *= np.clip(rng.normal(1.0, 0.12, (n, n)), 0.6, 1.2)
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1
    r = np.hypot(xx, yy * 1.05) + 0.12 * (f1 / 1.0 - 0.5)    # ragged outline
    mask = np.clip((1.0 - r) / 0.35, 0, 1) ** 1.5
    alpha = np.clip(lace * mask * 1.25, 0, 1)
    img = np.zeros((n, n, 4), np.uint8)
    img[..., :3] = (np.clip(0.93 + 0.07 * lace, 0, 1) * 255).astype(np.uint8)[..., None]
    img[..., 3] = (alpha * 255).astype(np.uint8)
    _png(path, img)


TEXTURES = {'beach_ramp': lambda p: _ramp_png(p),
            'beach_lace0': lambda p: _lace_png(p, 11), 'beach_lace1': lambda p: _lace_png(p, 23),
            'beach_lace2': lambda p: _lace_png(p, 37)}
TEX_VERSION = 2   # bump when a generator changes (textures are cached on disk)


def make_textures(force=False):
    os.makedirs(TEX_DIR, exist_ok=True)
    stamp = os.path.join(TEX_DIR, 'version.txt')
    old = open(stamp).read().strip() if os.path.exists(stamp) else ''
    force = force or old != str(TEX_VERSION)
    for name, fn in TEXTURES.items():
        path = os.path.join(TEX_DIR, name + '.png')
        if force or not os.path.exists(path):
            fn(path)
    with open(stamp, 'w') as f:
        f.write(str(TEX_VERSION))


class BeachFX:
    """Module part (build/bind) + per-frame update (mesh thread) + draw (decals, main thread)."""

    REQUIRES = {}

    def __init__(self, fx, *, half_y, seed=0):
        self.fx = fx                      # SplashFX: swash level, terrain profile
        self.half_y = half_y
        self.rng = np.random.default_rng(seed + 7)
        x = np.arange(SWASH_X[0], SWASH_X[1] + 1e-9, SWASH_STEP)
        y = np.arange(-half_y, half_y + 1e-9, SWASH_STEP)
        self.X, self.Y = np.meshgrid(x, y)
        self.ground = fx.terrain_z(self.X)
        self.wet_mem = np.zeros(self.X.shape)
        self.t_last = None
        self.mesh = None
        self.shown = self.ground.copy()                  # surface for decals (water or sand)
        self.front = np.full(len(y), np.nan)             # run-up front x per row
        self.front_prev = self.front.copy()
        self.stations = np.arange(0, len(y), LACE_ROW_EVERY)
        self.next_trail = np.zeros(len(self.stations))
        self.trail = np.zeros((0, 7))                    # x y born life size yaw tex
        self.mat = {}
        self.ms = 0.0

    # ── module shape ─────────────────────────────────────────
    def build(self, spec):
        make_textures()
        for name in TEXTURES:
            spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                             file=os.path.join(TEX_DIR, name + '.png'), nchannel=4 if 'lace' in name else 3)
            mat = spec.add_material(name=name)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
            if name == 'beach_ramp':
                # same lighting as a plain geom (the sand): dry-sand texels then look exactly like the sand
                mat.specular, mat.shininess, mat.reflectance = 0.5, 0.5, 0.0
        ny, nx = self.X.shape
        verts = np.stack([self.X, self.Y, self.ground - 0.05 + 0.01 * np.sin(self.X)], -1).reshape(-1, 3)
        idx = np.arange(nx * ny).reshape(ny, nx)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
        faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        tex = np.zeros((nx * ny, 2))
        tex[:, 1] = 0.5
        spec.add_mesh(name='lab_swash', uservert=verts.ravel().tolist(), userface=faces.ravel().tolist(),
                      usertexcoord=tex.ravel().tolist(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        from sim import config
        spec.worldbody.add_geom(name='lab_swash', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='lab_swash',
                                material='beach_ramp', rgba=(1, 1, 1, 1), contype=0, conaffinity=0,
                                group=config.Groups.VISUAL)

    def bind(self, model):
        for name in TEXTURES:
            self.mat[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            assert self.mat[name] >= 0, name
        mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, 'lab_swash')
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'lab_swash')
        assert mesh_id >= 0 and geom_id >= 0
        n = self.X.size
        # vertex i <-> normal i <-> texcoord i (so all three can be rewritten in the same order)
        assert model.mesh_vertnum[mesh_id] == n and model.mesh_normalnum[mesh_id] == n
        assert model.mesh_texcoordnum[mesh_id] == n
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
        self.mesh = (int(mesh_id), model.mesh_vertadr[mesh_id], model.mesh_normaladr[mesh_id],
                     model.mesh_texcoordadr[mesh_id], rot.reshape(3, 3), model.geom_pos[geom_id].copy())

    def add_forces(self, model, data, acc):
        pass

    # ── per frame, mesh thread ───────────────────────────────
    def update(self, model, eta_grid, t):
        """eta_grid(x, y) -> sea level (bilinear on the main mesh grid, already computed this frame)."""
        import time
        t0 = time.perf_counter()
        fx = self.fx
        X, Y, ground = self.X, self.Y, self.ground
        eta = eta_grid(X, Y)
        water = fx.swash_water(X, Y, eta, t)          # sea level blended into the run-up level
        depth = water - ground
        wet = depth > fx_swash_min()
        on_sand = X > fx.shore_x
        # on the sand, only water still joined to the sea is drawn. The replayed run-up level can sit
        # above the sand higher up while the backwash below is already dry; that left flat ponds with
        # stair-step outlines on the beach (render 2026-11-28). They show as wet sand instead.
        connected = np.cumprod(wet | ~on_sand, axis=1).astype(bool)
        wet &= connected
        z_water = np.where(on_sand, ground + np.minimum(depth, fx_swash_max()), water)
        # wet sand memory
        dt = 0.0 if self.t_last is None or t < self.t_last else t - self.t_last
        self.t_last = t
        self.wet_mem = np.where(wet, 1.0, self.wet_mem * math.exp(-dt / WET_SAND_TAU))
        # dry places get a sand-coloured shell (dry colour = sand), never a hole: no visible edge on the sand
        z = np.where(wet, z_water, ground + WET_SAND_LIFT)
        # wet sand shown slightly blurred: the 0.25 m cells otherwise draw a staircase edge
        m = self.wet_mem
        pad = np.pad(m, 1, mode='edge')
        soft = (4 * m + pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 8
        # colour change eases in from zero slope at DEPTH_FULL: a ramp that starts abruptly reads as a
        # line on the water (it sat right on the mesh edge at x = -6 m, render 2026-11-28)
        f = 1 - (1 - np.clip(depth / DEPTH_FULL, 0, 1)) ** 2.2
        u = np.where(wet, U_WET + (1 - U_WET) * f, U_WET * soft)
        u = np.clip(u, 0.004, 0.996)   # stay off the texture edge: linear filtering wraps u=0 onto u=1 (sea)
        # normals from the grid
        dz_dy, dz_dx = np.gradient(z, SWASH_STEP)
        normals = np.stack([-dz_dx, -dz_dy, np.ones_like(z)], -1).reshape(-1, 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        mesh_id, vadr, nadr, tadr, rot, pos = self.mesh
        verts = np.stack([X, Y, z], -1).reshape(-1, 3)
        n = len(verts)
        model.mesh_vert[vadr:vadr + n] = (verts - pos) @ rot
        model.mesh_normal[nadr:nadr + n] = normals @ rot
        model.mesh_texcoord[tadr:tadr + n, 0] = u.ravel()
        model.mesh_texcoord[tadr:tadr + n, 1] = 0.5
        # front per row: last wet vertex of the water connected to the sea (scan from the sea side)
        last = np.cumprod(wet, axis=1).sum(axis=1) - 1
        rows = np.arange(len(last))
        ok = (last >= 0) & (last < X.shape[1] - 1)
        front = np.full(len(last), np.nan)
        # sub-cell: where the depth crosses SWASH_MIN between the last wet and the first dry vertex
        i = np.clip(last, 0, X.shape[1] - 2)
        d0, d1 = depth[rows, i], depth[rows, i + 1]
        frac = np.clip((d0 - fx_swash_min()) / np.maximum(d0 - d1, 1e-6), 0, 1)
        front[ok] = X[rows[ok], i[ok]] + frac[ok] * SWASH_STEP
        self.front_prev, self.front = self.front, front
        self.shown = np.maximum(z, ground + WET_SAND_LIFT)
        self.slope = np.gradient(self.shown, SWASH_STEP)   # (d/dy, d/dx) for tilting decals
        self.ms = 1000 * (time.perf_counter() - t0)
        return mesh_id

    def _shown_at(self, x, y, grid=None):
        u = np.clip((x - SWASH_X[0]) / SWASH_STEP, 0, self.X.shape[1] - 1.001)
        v = np.clip((y + self.half_y) / SWASH_STEP, 0, self.X.shape[0] - 1.001)
        i, j = u.astype(int), v.astype(int)
        fu, fv = u - i, v - j
        S = self.shown if grid is None else grid
        return (S[j, i] * (1 - fu) * (1 - fv) + S[j, i + 1] * fu * (1 - fv)
                + S[j + 1, i] * (1 - fu) * fv + S[j + 1, i + 1] * fu * fv)

    def _frames(self, x, y, yaw):
        """Rotation (row-major 9 each) lying on the shown surface, turned by yaw about its normal."""
        sy, sx = self.slope
        gx, gy = self._shown_at(x, y, sx), self._shown_at(x, y, sy)
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        a = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], -1)
        a -= (a * n).sum(1, keepdims=True) * n
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = np.cross(n, a)
        return np.stack([a, b, n], -1).reshape(-1, 9)   # columns = local x, y, z

    # ── per frame, main thread: lace decals ──────────────────
    def _visible(self, x, y, z, cam_pos, cam_fwd):
        d = np.stack([x - cam_pos[0], y - cam_pos[1], z - cam_pos[2]], -1)
        dist = np.linalg.norm(d, axis=-1)
        ok = np.hypot(d[..., 0], d[..., 1]) < LACE_RANGE
        if cam_fwd is not None:
            ok &= (d @ np.asarray(cam_fwd, float)) > LACE_VIEW_COS * np.maximum(dist, 1e-6)
        return ok

    def draw(self, scene, t, cam_pos, cam_fwd=None):
        import time
        t0 = time.perf_counter()
        U = self.rng.uniform
        front = self.front
        ys = self.Y[self.stations, 0]
        fx_st = front[self.stations]
        near = np.isfinite(fx_st)
        near[near] = self._visible(fx_st[near], ys[near], np.zeros(near.sum()), cam_pos, cam_fwd)
        # trail: drop a decal at the front now and then; it stays where it fell and fades
        if len(self.trail):
            self.trail = self.trail[t - self.trail[:, 2] < self.trail[:, 3]]
        due = np.nonzero(near & (t >= self.next_trail))[0]
        if len(due):
            k = len(due)
            rows = np.column_stack([fx_st[due] - U(0.0, 0.5, k), ys[due] + U(-.35, .35, k), np.full(k, t),
                                    U(*LACE_TRAIL_LIFE, k), U(0.45, 0.8, k), U(0, 2 * math.pi, k),
                                    self.rng.integers(0, 3, k)])
            self.trail = np.vstack([self.trail, rows])[-MAX_LACE:]
            self.next_trail[due] = t + U(*LACE_TRAIL_DT, k)
        g = scene.ngeom

        def add(x, y, z, hx, hy, mat9, alpha, tex):
            nonlocal g
            if g >= scene.maxgeom:
                return
            geom = scene.geoms[g]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, np.array([hx, hy, 0.0005]),
                                np.array([x, y, z]), mat9, np.array([1, 1, 1, min(alpha, 0.99)], np.float32))
            geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            geom.emission, geom.specular, geom.shininess, geom.reflectance = 0.55, 0.0, 0.0, 0.0
            geom.matid = self.mat[f'beach_lace{int(tex)}']
            g += 1

        if len(self.trail):
            T = self.trail
            age = (t - T[:, 2]) / T[:, 3]
            alpha = np.clip((t - T[:, 2]) / 0.2, 0, 1) * (1 - age) ** 1.3 * 0.9
            z = self._shown_at(T[:, 0], T[:, 1]) + LACE_LIFT
            size = T[:, 4] * (1 + 0.25 * age)            # lace spreads thin as it drains
            R = self._frames(T[:, 0], T[:, 1], T[:, 5])
            show = self._visible(T[:, 0], T[:, 1], z, cam_pos, cam_fwd) & (alpha > 0.03)
            for k in np.nonzero(show)[0]:
                add(T[k, 0], T[k, 1], z[k], size[k], size[k] * 0.8, R[k], alpha[k], T[k, 6])
        # leading edge: a lace band right on the front, wider where the water is running up
        idx = np.nonzero(near)[0]
        if len(idx):
            x = fx_st[idx]
            run = np.nan_to_num(x - self.front_prev[self.stations][idx], nan=0.0)
            xs = x - 0.15
            z = self._shown_at(xs, ys[idx]) + LACE_LIFT + 0.002
            R = self._frames(xs, ys[idx], 0.15 * np.sin(idx * 1.7))
            for k, s in enumerate(idx):
                w = 0.35 + np.clip(run[k] * 8, 0, 0.35)
                add(xs[k], ys[s], z[k], w, LACE_ROW_EVERY * SWASH_STEP * 0.75, R[k], 0.95, s % 3)
        scene.ngeom = g
        self.ms_draw = 1000 * (time.perf_counter() - t0)


def fx_swash_min():
    import splash_fx
    return splash_fx.SWASH_MIN


def fx_swash_max():
    import splash_fx
    return splash_fx.SWASH_MAX
