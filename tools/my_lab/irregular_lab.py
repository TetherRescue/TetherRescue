"""Personal irregular-wave lab: small sea + sloped beach with the team water system.

Lab only (sim/ does not import it). Uses the real team modules:
  sim.world.world.build_world / advance_world   (assembly, step order, ForceAccumulator)
  sim.water.water.Water                          (voxel buoyancy, drag, wave surface mesh)
  sim.robot.robot.Robot                          (boat, throttle 0 by default)
  sim.world.human.Human                          (optional floating person)
and tools/my_lab/my_sea.MySea (subclass of sim.water.sea.Sea) for the waves.

Map (world ENU, waterline x = 0):
  sea      x in [SEA_WEST, 0], seabed rises with BED_SLOPE toward the shore
  beach    x in [0, BEACH_WIDTH], sand rises to BEACH_TOP
  land     x beyond the beach
Terrain is visual only (no contact), same as the team coast maps.

Run (repo root, venv on):
  python tools/my_lab/irregular_lab.py                          # irregular, Hs 0.6 m, Tp 4 s, seed 1004
  python tools/my_lab/irregular_lab.py --hs 1.0 --tp 6 --spread 20 --seed 2024
  python tools/my_lab/irregular_lab.py --sea three              # compare with run_sim waves
  python tools/my_lab/irregular_lab.py --human --throttle 0.3
  python tools/my_lab/irregular_lab.py --headless 60            # no viewer, print statistics
"""
import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (REPO_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from sim import config  # noqa: E402
from sim.robot.robot import Robot  # noqa: E402
from sim.water.sea import R_BEACH  # noqa: E402
from sim.water import water as W  # noqa: E402
from sim.water.water import Water, wave_height  # noqa: E402
from sim.world.human import Human  # noqa: E402
from sim.water.sea import R_ROCK  # noqa: E402
from splash_fx import SplashFX, cam_forward, cam_position  # noqa: E402
from sim.world.world import advance_world, build_world  # noqa: E402
from my_sea import MySea  # noqa: E402
from swe_rock import SweSea, ellipsoid_top  # noqa: E402
from beach_fx import BeachFX, SWASH_X  # noqa: E402

# ── map ───────────────────────────────────────────────────────
SEA_WEST = -40.0          # west end of the sea [m]
HALF_Y = 25.0             # map half width [m]
BED_SLOPE = 0.08          # seabed slope toward the shore [m/m] (also used for shore attenuation)
BEACH_WIDTH = 12.0        # sand from waterline to land [m]
BEACH_TOP = 0.8           # sand height at the land edge [m]
LAND_WIDTH = 15.0
LAND_TOP = 1.0
BOTTOM_Z = -6.0           # terrain is filled down to here (no gaps when viewed from the side)
SEA_BOUNDS = (SEA_WEST, 0.0, -HALF_Y, HALF_Y)
# Opaque on purpose: MuJoCo sorts transparent geoms by their CENTER distance, and the 50 m water mesh
# center was often nearer the camera than the spray sprites -> water drawn over the spray, which then
# looked under the water (2026-11-28). Opaque water is drawn first; the sprites blend on top of it.
SEA_RGBA = (.06, .32, .45, 1.0)
SAND_RGBA = (.76, .68, .50, 1)
BED_RGBA = (.10, .22, .28, 1)
LAND_RGBA = (.30, .44, .26, 1)

# rock island (ellipse footprint, solid like the Hanagae reef: boat/person bounce off it)
ROCK_CENTER = (-18.0, -5.0)   # offshore (~1.5 m deep) so waves reach it unbroken and pile up on its face
ROCK_HALF_X = 2.0          # along the wave direction [m]
ROCK_HALF_Y = 2.6          # across [m]
# (dx, dy, z, rx, ry, rz, tilt rad, rgba); z relative to the still water level
ROCK_PARTS = (
    (0.0, 0.0, .15, 1.6, 1.9, 1.5, .35, (.45, .42, .38, 1)),
    (-.9, .9, -.35, 1.1, 1.2, 1.0, -.5, (.38, .36, .33, 1)),
    (.8, -1.0, -.25, 1.0, 1.1, .95, .6, (.52, .48, .43, 1)),
    (.3, .3, .8, .6, .5, .5, .4, (.58, .55, .5, 1)),
)

ROCK_Z_SCALE = 0.55   # vertical scale of ROCK_PARTS (1.0 = top 1.66 m). Set by --rock-scale.
# Measured 2026-11-28 with the shallow-water solver, Hs 1.0: in front of the rock (1.6 m deep) the team
# Sea caps the waves by depth (crest max 0.79 m, Hs 1.5 and 2.0 give the same 1.22 m), so a 1.66 m
# rock was wet on 2 % of its dry area on average. 0.55 (top ~0.9 m): overtopping now and then.
ROCK_TILT_AXIS = np.array([.2, 1, .15])

# shallow-water box around the rock (swe_rock.py), kept small on purpose: the shallow-water waves are
# non-dispersive, so the farther they travel inside the box the more they drift from the analytic sea
SWE_BOUNDS = (-27.0, -9.0, -14.0, 4.0)
FILM_MIN = 0.02     # [m] water thinner than this on the rock is not drawn
FILM_HIDE = 0.06    # [m] dry rock: the film / sea mesh is tucked this far under the rock surface
FILM_UNDER = 0.03   # [m] film cells over open water sit just below the sea mesh (that one is drawn)
BOAT_GAP = 0.02     # [m] wave mesh stays this far under the deck around the boat

BOAT_XY = (-25.0, 4.0)
PATCH_CENTER = (-14.0, 0.0)   # wave-mesh patch (+-25 m): x -39..11 covers sea + swash zone
HUMAN_XY = (-20.0, 6.0)
SEEDS = (1004, 2024, 3000)
PROBE_XY = (-28.0, 8.0)   # deep-water point for measured Hs
PRINT_EVERY = 2.0         # [s]
HS_WINDOW = 1000          # samples (50 Hz) for the measured Hs = last 20 s


WAVE_MAX = 3.0        # slider top: Hs [m] for irregular seas, otherwise strength factor
WAVE_KEY_STEP = 0.1  # keyboard step (same unit as the slider)
KEY_WAVE_DOWN, KEY_WAVE_UP = 44, 46   # ',' and '.' (GLFW key codes)


class WaveKnob:
    """Wave-strength slider in the viewer's right panel (Control section).

    MuJoCo's passive viewer has no Python API for custom widgets, so this adds one dummy actuator
    whose slider we read every frame. It produces no force: zero gain/bias on a slide joint of a
    massless-looking, gravity-compensated body with no geoms, parked under the map.
    Visual/experiment helper only (not a team part).
    """

    REQUIRES = {}
    NAME = 'wave_Hs_m'

    def __init__(self, value, label):
        self.value = float(value)
        self.NAME = label
        self.actuator_id = None

    def build(self, spec):
        body = spec.worldbody.add_body(name='lab_wave_knob', pos=[0, 0, -20], gravcomp=1,
                                       mass=1e-3, inertia=[1e-6, 1e-6, 1e-6])
        body.add_joint(name='lab_wave_knob', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                       range=[-0.01, 0.01], damping=1.0)
        act = spec.add_actuator(name=self.NAME, target='lab_wave_knob', trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                ctrlrange=[0.0, WAVE_MAX], ctrllimited=True)
        act.gainprm[0] = 0.0      # no force at any slider value
        act.biasprm[:3] = [0.0, 0.0, 0.0]

    def bind(self, model):
        self.actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.NAME)
        assert self.actuator_id >= 0, 'wave knob actuator missing'

    def add_forces(self, model, data, acc):
        pass


class LabMap:
    """Visual terrain part (module shape: REQUIRES / build / bind / add_forces)."""

    REQUIRES = {}

    def build(self, spec):
        surface = config.WaterParams.SURFACE_Z
        z_toe = surface - BED_SLOPE * (0.0 - SEA_WEST)
        # seabed + beach as one extruded profile (x, z), top surface only changes slope at the waterline
        _extruded(spec, 'lab_shore', [(SEA_WEST, z_toe), (0.0, surface), (BEACH_WIDTH, surface + BEACH_TOP)],
                  BOTTOM_Z, SAND_RGBA)
        # darker plate just under the seabed top so the deep part reads as water depth
        _extruded(spec, 'lab_bed_tint', [(SEA_WEST, z_toe - .02), (-3.0, surface - BED_SLOPE * 3.0 - .02)],
                  z_toe - .3, BED_RGBA, half_y=HALF_Y - .01)   # sides not coplanar with the sand (z-fight stripes)
        _extruded(spec, 'lab_land', [(BEACH_WIDTH, surface + LAND_TOP), (BEACH_WIDTH + LAND_WIDTH, surface + LAND_TOP)],
                  BOTTOM_Z, LAND_RGBA)
        spec.worldbody.add_light(pos=(0, 0, 30), dir=(.3, 0, -1), diffuse=(.75, .75, .75),
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False)
        if self.rock:
            _add_rock(spec, surface)
        spec.add_texture(name='lab_sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[.55, .72, .9], rgb2=[.93, .96, 1],
                         width=64, height=64)

    def __init__(self, rock=True):
        self.rock = rock

    def bind(self, model):
        pass

    def add_forces(self, model, data, acc):
        pass


def rock_parts(surface=0.0):
    """[(center, size, quat, rgba)] of the rock ellipsoids, heights scaled by ROCK_Z_SCALE."""
    cx, cy = ROCK_CENTER
    out = []
    for dx, dy, z, rx, ry, rz, tilt, rgba in ROCK_PARTS:
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, ROCK_TILT_AXIS, tilt)
        out.append(((cx + dx, cy + dy, surface + z * ROCK_Z_SCALE), (rx, ry, rz * ROCK_Z_SCALE), quat, rgba))
    return out


def rock_top(x, y, surface=0.0):
    """Rock surface height above (x, y); -inf where there is no rock."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    top = np.full(np.broadcast(x, y).shape, -np.inf)
    for center, size, quat, _ in rock_parts(surface):
        top = np.maximum(top, ellipsoid_top(center, size, quat, x, y))
    return top


def lab_bed(x, y, rock=True):
    """Seabed (+ rock) height of the lab map for the shallow-water solver."""
    surface = config.WaterParams.SURFACE_Z
    bed = surface + BED_SLOPE * np.minimum(np.asarray(x, float), 0.0) + 0.0 * np.asarray(y, float)
    return np.maximum(bed, rock_top(x, y, surface)) if rock else bed


def _add_rock(spec, surface):
    """Rock island = overlapping tilted ellipsoids, solid (same contact setup as the Hanagae reef)."""
    contacts = config.Contacts
    solid = dict(group=config.Groups.PHYSICS, contype=contacts.ENV, conaffinity=contacts.ENV_AFFINITY,
                 solref=[.006, 1], solimp=[.95, .99, .001, .5, 2], priority=1)
    for i, (center, size, quat, rgba) in enumerate(rock_parts(surface)):
        spec.worldbody.add_geom(name=f'lab_rock_{i}', type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                                pos=center, size=size, quat=quat, rgba=rgba, **solid)
    _add_rock_base(spec, surface)


ROCK_BASE_SIDES = 64


def _add_rock_base(spec, surface):
    """Vertical wall under each rock ellipsoid, from its outline (seen from above) down to the seabed.

    The ellipsoids are widest at their middle and narrower below, so in a wave trough the hollow
    under that bulge showed through as stripes (render 2026-11-28). A height-field fill at 0.1 m
    left a saw-tooth edge; a prism on the exact outline does not.
    """
    for i, (center, size, quat, _) in enumerate(rock_parts(surface)):
        ang = np.linspace(0, 2 * np.pi, ROCK_BASE_SIDES, endpoint=False)
        ux, uy = np.cos(ang), np.sin(ang)
        lo, hi = np.zeros_like(ang), np.full_like(ang, 2 * max(size))
        for _ in range(30):                                   # outline radius: last r that still hits
            mid = 0.5 * (lo + hi)
            hit = np.isfinite(ellipsoid_top(center, size, quat, center[0] + mid * ux, center[1] + mid * uy))
            lo, hi = np.where(hit, mid, lo), np.where(hit, hi, mid)
        r = lo * 0.995
        x, y = center[0] + r * ux, center[1] + r * uy
        top = ellipsoid_top(center, size, quat, x, y) - 0.02
        bottom = surface + BED_SLOPE * np.minimum(x, 0.0) - 0.05
        n = ROCK_BASE_SIDES
        verts = np.concatenate([np.stack([x, y, top], -1), np.stack([x, y, bottom], -1)])
        k = np.arange(n)
        k1 = (k + 1) % n
        faces = np.concatenate([np.stack([k, k + n, k1 + n], -1), np.stack([k, k1 + n, k1], -1)])
        spec.add_mesh(name=f'lab_rock_base_{i}_mesh', uservert=verts.ravel().tolist(),
                      userface=faces.ravel().tolist(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        spec.worldbody.add_geom(name=f'lab_rock_base_{i}', type=mujoco.mjtGeom.mjGEOM_MESH,
                                meshname=f'lab_rock_base_{i}_mesh', rgba=(.36, .34, .31, 1),
                                contype=0, conaffinity=0, group=config.Groups.VISUAL)


BREAK_DEPTH = 0.5          # waves break (whitewater) around this depth [m]


def _extruded(spec, name, top_xz, z_bottom, rgba, half_y=None):
    """Prism along y: polygon = top polyline + floor at z_bottom, extruded over [-HALF_Y, HALF_Y]."""
    poly = list(top_xz) + [(top_xz[-1][0], z_bottom), (top_xz[0][0], z_bottom)]
    m = len(poly)
    half_y = HALF_Y if half_y is None else half_y
    verts = [(x, y, z) for y in (-half_y, half_y) for x, z in poly]
    faces = []
    for i in range(1, m - 1):                       # both caps (fan, polygon is convex)
        faces += [(0, i, i + 1), (m, m + i + 1, m + i)]
    for i in range(m):                              # sides
        j = (i + 1) % m
        faces += [(i, j, m + j), (i, m + j, m + i)]
    v = np.asarray(verts, float)
    center = v.mean(axis=0)
    oriented = []
    for a, b, c in faces:                           # make every normal point outward
        n = np.cross(v[b] - v[a], v[c] - v[a])
        oriented.append((a, b, c) if np.dot(n, (v[a] + v[b] + v[c]) / 3 - center) >= 0 else (a, c, b))
    spec.add_mesh(name=name + '_mesh', uservert=v.ravel().tolist(),
                  userface=np.asarray(oriented, int).ravel().tolist(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name + '_mesh',
                            rgba=rgba, contype=0, conaffinity=0, group=config.Groups.VISUAL)


LAB_DT = 0.004   # [s] physics step of THIS lab only (team config.Sim.TIMESTEP = 0.002 is untouched).
# Measured 2026-11-28: 0.004 s ran 30 s without NaN, heave std 0.241 m (vs ~0.25 m at 0.002 s),
# headless 2.97x vs 1.54x realtime. --dt 0.002 reproduces the team setting.


class LabWater(Water):
    """Team Water + optional wave water velocity in the drag (experiment only).

    flow='none'    : team model as is (drag vs. WaterParams.WATER_VELOCITY_W only)
    flow='orbital' : drag vs. wave orbital velocity at each body's COM (Sea.orbital)
    flow='stokes'  : drag vs. mean Stokes drift at each body's COM (MySea.stokes_drift)
    Do not combine orbital + stokes: a body that follows the orbital motion already picks up
    part of the Stokes drift, adding both would count it twice.
    Implementation: the team add_forces reads WATER_VELOCITY_W once per call, so targets are
    run one by one with that value set to (base current + wave flow at the body). Restored after.
    """

    # team WaterParams.REQUIRES says max_dt 0.002; this lab allows its own LAB_DT (see above)
    REQUIRES = {**config.WaterParams.REQUIRES, 'max_dt': LAB_DT}

    def __init__(self, *a, flow='none', **kw):
        super().__init__(*a, **kw)
        self.flow = flow
        self.extra_meshes = []   # more mesh ids to upload each frame (rock water film, swash mesh)
        self.beach = None        # BeachFX: finer swash mesh with depth colours (beach_fx.py)
        self._film = None
        self._main_hide = None

    @property
    def swe(self):
        return getattr(self.waves, 'swe', None)

    # ── rock water film (shallow-water cells over the rock, finer than the 0.5 m sea mesh) ──
    def _film_cells(self):
        swe = self.swe
        rock = rock_top(swe.X, swe.Y, config.WaterParams.SURFACE_Z) > config.WaterParams.SURFACE_Z - 0.2
        rows, cols = np.nonzero(rock)
        j0, j1 = max(rows.min() - 2, 0), min(rows.max() + 3, swe.ny)
        i0, i1 = max(cols.min() - 2, 0), min(cols.max() + 3, swe.nx)
        return (slice(j0, j1), slice(i0, i1)), rock[j0:j1, i0:i1]

    def build(self, spec):
        super().build(spec)
        if self.swe is None:
            return
        (sj, si), rock = self._film_cells()
        X, Y, b = self.swe.X[sj, si], self.swe.Y[sj, si], self.swe.b[sj, si]
        ny, nx = X.shape
        verts = np.stack([X, Y, b - FILM_HIDE], axis=-1).reshape(-1, 3)   # rock-shaped: not flat (inertia)
        idx = np.arange(nx * ny).reshape(ny, nx)
        a, bb, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
        faces = np.concatenate([np.stack([a, bb, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        spec.add_mesh(name='lab_rock_film', uservert=verts.ravel().tolist(), userface=faces.ravel().tolist(),
                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        spec.worldbody.add_geom(name='lab_rock_film', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='lab_rock_film',
                                rgba=list(self.sea_rgba), contype=0, conaffinity=0, group=config.Groups.VISUAL)

    def bind(self, model):
        super().bind(model)
        self._film, self.extra_meshes = None, []
        if self.swe is None:
            return self.bind_beach()
        mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, 'lab_rock_film')
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'lab_rock_film')
        assert mesh_id >= 0 and geom_id >= 0, 'rock film mesh missing'
        (sj, si), rock = self._film_cells()
        count = rock.size
        assert model.mesh_vertnum[mesh_id] == count and model.mesh_normalnum[mesh_id] == count
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
        self._film = (mesh_id, model.mesh_vertadr[mesh_id], model.mesh_normaladr[mesh_id],
                      rot.reshape(3, 3), model.geom_pos[geom_id].copy(), sj, si, rock)
        self.extra_meshes = [int(mesh_id)]

    def bind_beach(self):
        if self.beach is not None and self.beach.mesh is not None and self.beach.mesh[0] not in self.extra_meshes:
            self.extra_meshes.append(self.beach.mesh[0])

    def _update_film(self, model):
        mesh_id, vadr, nadr, rot, pos, sj, si, rock = self._film
        swe = self.swe
        h, b = swe.h[sj, si], swe.b[sj, si]          # one snapshot (the solver swaps arrays, never edits)
        wet = h > FILM_MIN
        # dry cell: under the rock surface AND no higher than the wet water next to it. Otherwise, in a
        # trough, the film made a wall from the rock rim down to the water (render 2026-11-28)
        level = np.where(wet, b + h, np.nan)
        pad = np.pad(level, 1, constant_values=np.nan)
        nb = np.stack([pad[1 + dj:pad.shape[0] - 1 + dj, 1 + di:pad.shape[1] - 1 + di]
                       for dj in (-1, 0, 1) for di in (-1, 0, 1)])
        near = np.where(np.isnan(nb).all(0), np.inf, np.nanmin(np.where(np.isnan(nb), np.inf, nb), 0))
        z = np.where(wet, b + h, np.minimum(b - FILM_HIDE, near))
        z = np.where(rock | ~wet, z, z - FILM_UNDER)
        dz_dy, dz_dx = np.gradient(z, swe.dx)
        normals = np.stack([-dz_dx, -dz_dy, np.ones_like(z)], -1).reshape(-1, 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        verts = np.stack([swe.X[sj, si], swe.Y[sj, si], z], -1).reshape(-1, 3)
        n = len(verts)
        model.mesh_vert[vadr:vadr + n] = (verts - pos) @ rot
        model.mesh_normal[nadr:nadr + n] = normals @ rot

    def _hide_in_rock(self, verts, n_grid):
        """Sea-mesh vertices inside the rock go under its surface; the film draws the water up there."""
        if self._main_hide is None or self._main_hide[0] != n_grid:
            top = rock_top(verts[:n_grid, 0], verts[:n_grid, 1], config.WaterParams.SURFACE_Z)
            mask = top > config.WaterParams.SURFACE_Z - 0.2
            self._main_hide = (n_grid, np.nonzero(mask)[0], top[mask] - FILM_HIDE)
        _, idx, z = self._main_hide
        verts[idx, 2] = np.minimum(verts[idx, 2], z)
        return verts

    def _clear_boat(self, verts, data):
        """Keep the wave mesh under the deck so the water never shows through the boat.

        The boat rides the waves but is only 0.2 m tall and 1.5 m long; across it the 0.5 m mesh can
        stand higher than the deck. Every vertex up to one mesh diagonal around the hull footprint is
        held under the (tilted) deck plane: then no mesh triangle above the hull can rise through it.
        """
        if self.body_id is None:
            return verts
        hull = config.Hull
        p = np.array(data.xpos[self.body_id])
        R = np.array(data.xmat[self.body_id]).reshape(3, 3)
        margin = W.WAVE_SURFACE_STEP * math.sqrt(2.0)
        local = (verts[:, :2] - p[:2]) @ R[:2, :2]
        near = (np.abs(local[:, 0]) <= hull.LENGTH / 2 + margin) & (np.abs(local[:, 1]) <= hull.BEAM / 2 + margin)
        if not near.any():
            return verts
        top = p + R[:, 2] * (hull.HEIGHT / 2)
        n = R[:, 2]
        deck = top[2] - (n[0] * (verts[near, 0] - top[0]) + n[1] * (verts[near, 1] - top[1])) / max(n[2], 0.2)
        verts[near, 2] = np.minimum(verts[near, 2], deck - BOAT_GAP)
        return verts

    def update_surface_visual(self, model, data, viewer=None, center_xy=None):
        """Same mesh as the team Water, but the normals come from the height grid itself.

        Team wave_surface_mesh evaluates the sea 3x per frame for a Sea object (heights + two
        shifted grids for a finite-difference slope). Here: heights once, slope = np.gradient of
        that grid (central difference, same 0.5 m spacing). Visual only.
        """
        if self._wave_mesh is None or not isinstance(self.waves, MySea):
            return super().update_surface_visual(model, data, viewer, center_xy)
        if center_xy is None:
            center_xy = PATCH_CENTER   # fixed: the 50 m patch covers the whole sea AND the beach
        step = W.WAVE_SURFACE_STEP
        center = np.round(np.asarray(center_xy, dtype=float) / step) * step
        axis = W.wave_surface_grid()
        gx, gy = np.meshgrid(center[0] + axis, center[1] + axis)
        fade, dfade_dx, dfade_dy = W._wave_surface_fade()
        surface = config.WaterParams.SURFACE_Z
        rise = np.asarray(self.waves.eta(gx, gy, data.time), float) - surface
        dh_dy, dh_dx = np.gradient(rise, step)
        heights = surface + fade * rise
        dz_dx = fade * dh_dx + rise * dfade_dx
        dz_dy = fade * dh_dy + rise * dfade_dy
        normals = np.stack([-dz_dx, -dz_dy, np.ones_like(heights)], axis=-1).reshape(-1, 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        verts = np.stack([gx, gy, heights], axis=-1).reshape(-1, 3)
        if self.sea_bounds is not None:
            corners = W.sea_ring_corners(self.sea_bounds, center)
            verts = np.vstack([verts, np.column_stack([corners, np.full(len(corners), surface)])])
            normals = np.vstack([normals, np.tile([0.0, 0.0, 1.0], (len(corners), 1))])
        if self.impact is not None:
            verts = self.impact.dress(verts, data.time)
        if self.swe is not None:
            verts = self._hide_in_rock(verts, gx.size)
            if self.swe.t is not None:   # solver starts at the first physics step
                self._update_film(model)
        if self.beach is not None and self.beach.mesh is not None:
            self.bind_beach()
            x0, y0 = gx[0, 0], gy[0, 0]

            H = heights   # this frame's sea mesh heights INCLUDING the edge fade (same water as the sea mesh)

            def eta_grid(x, y):   # bilinear on the 0.5 m sea grid -> passes exactly through its vertices
                u = np.clip((x - x0) / step, 0, gx.shape[1] - 1.001)
                v = np.clip((y - y0) / step, 0, gx.shape[0] - 1.001)
                i, j = u.astype(int), v.astype(int)
                fu, fv = u - i, v - j
                return (H[j, i] * (1 - fu) * (1 - fv) + H[j, i + 1] * fu * (1 - fv)
                        + H[j + 1, i] * (1 - fu) * fv + H[j + 1, i + 1] * fu * fv)
            self.beach.update(model, eta_grid, data.time)
            # The swash mesh draws the beach band. The coarse sea mesh stays just UNDER it (not dived
            # to the seabed: that left a visible step/sliver at x = -6 m seen from the map edge, and
            # the unfaded swash heights did not meet the faded sea mesh at the patch edge, 2026-11-28).
            band = np.nonzero(verts[:gx.size, 0] > SWASH_X[0] + 0.1)[0]
            under = self.beach._shown_at(verts[band, 0], verts[band, 1]) - 0.03
            verts[band, 2] = np.minimum(verts[band, 2], under)
        verts = self._clear_boat(verts, data)
        mesh_id, vert_adr, normal_adr, rot, pos = self._wave_mesh
        count = len(verts)
        model.mesh_vert[vert_adr:vert_adr + count] = (verts - pos) @ rot
        model.mesh_normal[normal_adr:normal_adr + count] = normals @ rot
        if viewer is not None:
            viewer.update_mesh(mesh_id)
            for extra in self.extra_meshes:
                viewer.update_mesh(extra)
        return int(mesh_id)

    def flow_w(self, data, body_id):
        if self.flow == 'none' or not isinstance(self.waves, MySea):
            return np.zeros(3)
        x, y, z = data.xipos[body_id]
        if self.flow == 'orbital':
            return np.asarray(self.waves.orbital(x, y, z, data.time), float).reshape(3)
        return np.asarray(self.waves.stokes_drift(x, y, z), float).reshape(3)

    def add_forces(self, model, data, acc):
        # the shallow-water box steps with the physics (buoyancy below reads its water level)
        advance = getattr(self.waves, 'advance', None)
        if advance is not None:
            advance(data.time)
        if self.flow == 'none':
            return super().add_forces(model, data, acc)
        params = config.WaterParams
        base, targets = params.WATER_VELOCITY_W, self.targets
        boat_volume = 0.0
        try:
            for target in targets:
                self.targets = [target]
                params.WATER_VELOCITY_W = tuple(np.asarray(base, float) + self.flow_w(data, target['body_id']))
                super().add_forces(model, data, acc)
                if target['body_id'] == self.body_id:
                    boat_volume = self.submerged_volume
        finally:
            self.targets, params.WATER_VELOCITY_W = targets, base
            self.submerged_volume = boat_volume


def make_sea(args):
    if args.sea == 'calm':
        return config.WaterParams.WAVES_CALM
    use_swe = not args.no_swe and not args.no_rock
    cls = SweSea if use_swe else MySea
    if args.sea == 'irregular':
        sea = cls('irregular', hs=args.hs, tp=args.tp, spread_deg=args.spread, n=args.n, seed=args.seed)
    else:
        sea = cls(args.sea)
    if not args.no_shore:
        sea.add_shore((0.0, 0.0), (-1.0, 0.0), BED_SLOPE)
    if not args.no_reflect:
        sea.add_wall((0.0, 0.0), (-1.0, 0.0), R_BEACH, name='beach')
    if not args.no_rock and not use_swe:
        # seaward face of the rock reflects, and the rock casts a wave shadow behind it (same as Hanagae)
        if not args.no_reflect:
            sea.add_wall((ROCK_CENTER[0] - ROCK_HALF_X, ROCK_CENTER[1]), (-1.0, 0.0), R_ROCK,
                         half_len=ROCK_HALF_Y, name='rock', shadowed=False)
        sea.add_shadow(ROCK_CENTER, 0.0, ROCK_HALF_X, ROCK_HALF_Y)
    sea.reuse = not args.exact_sea
    if not args.exact_sea:
        # amplitudes (shadow / reflection window / shore) once on a grid -> much cheaper eta per step
        sea.set_amp_grid(SEA_WEST - 2.0, 2.0, -HALF_Y - 2.0, HALF_Y + 2.0)
    if use_swe:
        # rock reflection / shadow / overtopping are computed in this box (swe_rock.py), not faked
        sea.attach_swe(SWE_BOUNDS, lab_bed, surface=config.WaterParams.SURFACE_Z)
    return sea


def build_lab(args):
    waves = make_sea(args)
    surface = config.WaterParams.SURFACE_Z
    draft = config.Expect.DRAFT or 0.0
    boat_z = surface + config.Hull.HEIGHT / 2 - draft
    robot = Robot(position=(BOAT_XY[0], BOAT_XY[1], boat_z), yaw=0.0)   # facing +x (the shore)
    floating = [config.Names.HUMAN_BODY] if args.human else []
    water = LabWater(waves=waves, surface_visual=False, wave_visual=bool(waves), sea_bounds=SEA_BOUNDS,
                     sea_rgba=SEA_RGBA, floating_bodies=floating, flow=args.flow)
    parts = [water, LabMap(rock=not args.no_rock), robot]
    if isinstance(waves, MySea) and not args.no_splash:
        # visual only: folds the mesh at the beach/rock, bulge + sprite spray + foam (tools/my_lab/splash_fx.py)
        rock_center = ROCK_CENTER if not args.no_rock else (SEA_WEST - 50.0, 0.0)   # far away = no rock hits
        water.impact = SplashFX(waves, shore_x=0.0, rock_center=rock_center, rock_half=(ROCK_HALF_X, ROCK_HALF_Y),
                                half_y=HALF_Y, break_x=-BREAK_DEPTH / BED_SLOPE, seed=args.seed,
                                bed_slope=BED_SLOPE, beach_width=BEACH_WIDTH, beach_top=BEACH_TOP, land_top=LAND_TOP)
        water.impact.rock_runup = getattr(waves, 'swe', None) is None
        parts.append(water.impact)   # adds the sprite textures to the model
        if not args.no_beach_fx:
            water.beach = BeachFX(water.impact, half_y=HALF_Y, seed=args.seed)
            parts.append(water.beach)
    if args.human:
        parts.append(Human(position=(HUMAN_XY[0], HUMAN_XY[1], surface - .03)))
    knob = None
    if isinstance(waves, MySea):
        base = args.hs if args.sea == 'irregular' else 1.0
        knob = WaveKnob(base, 'wave_Hs_m' if args.sea == 'irregular' else 'wave_strength')
        knob.base = base
        parts.append(knob)
    # build_world(timestep=...) is the team's way to pass an experiment-only value (see its docstring)
    world = build_world(parts=parts, timestep=getattr(args, 'dt', None))
    if knob is not None:
        world.data.ctrl[knob.actuator_id] = knob.base
    world.knob = knob
    robot.set_command(throttle=args.throttle)
    return world, water, robot, waves


class Stats:
    """Boat heave / roll / pitch and measured Hs at a deep-water probe."""

    def __init__(self, world, water, robot, waves):
        self.world, self.water, self.robot, self.waves = world, water, robot, waves
        self.eta, self.z, self.roll, self.pitch = [], [], [], []
        self.start_xy = world.data.xpos[robot.body_id][:2].copy()
        self.start_t = world.data.time

    def sample(self):
        d = self.world.data
        bid = self.robot.body_id
        self.eta.append(float(wave_height(PROBE_XY[0], PROBE_XY[1], d.time, self.waves)) if self.waves else 0.0)
        self.z.append(float(d.xpos[bid][2]))
        rot = d.xmat[bid].reshape(3, 3)
        self.roll.append(math.degrees(math.atan2(rot[2, 1], rot[2, 2])))
        self.pitch.append(math.degrees(-math.asin(np.clip(rot[2, 0], -1, 1))))

    def line(self):
        d = self.world.data
        recent = self.eta[-HS_WINDOW:]   # last ~20 s, so a slider change shows up quickly
        hs = 4 * float(np.std(recent)) if len(recent) > 1 else 0.0
        bid = self.robot.body_id
        finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        dxy = d.xpos[bid][:2] - self.start_xy
        drift = dxy / max(d.time - self.start_t, 1e-9)
        return (f't={d.time:6.1f}s  Hs_meas={hs:.2f}m  boat dx={dxy[0]:+6.2f} dy={dxy[1]:+5.2f}m'
                f' (mean {drift[0]:+.3f},{drift[1]:+.3f} m/s) z={self.z[-1]:+.3f}'
                f'  heave_std={np.std(self.z):.3f}m  |roll|max={np.max(np.abs(self.roll)):.1f}deg'
                f'  |pitch|max={np.max(np.abs(self.pitch)):.1f}deg  finite={finite}')


def run_headless(world, water, robot, waves, seconds):
    stats = Stats(world, water, robot, waves)
    every = max(1, int(round(0.02 / world.model.opt.timestep)))    # sample at 50 Hz
    next_print = PRINT_EVERY
    wall = time.perf_counter()
    step = 0
    while world.data.time < seconds:
        advance_world(world)
        step += 1
        if step % every == 0:
            stats.sample()
        if world.data.time >= next_print:
            print(stats.line())
            if water.swe is not None:
                print(f'  swe {water.swe.steps} steps, rock wet {100 * rock_wet_fraction(water.swe):.0f} %')
            next_print += PRINT_EVERY
    elapsed = time.perf_counter() - wall
    print(f'done: {step} steps in {elapsed:.1f}s wall ({world.data.time / elapsed:.2f}x realtime)')


def _patch_viewer_close():
    """Make the window X button actually end the program (MuJoCo 3.17 passive viewer).

    Measured 2026-11-28 (WM_DELETE_WINDOW sent to the window): after X the render loop ends but
    exitrequest stays 0, so is_running() keeps returning True, and a main thread waiting inside
    viewer.update_mesh() never returns (nobody performs the upload) -> the program hangs.
    A later run (faster mesh uploads) also hung: the upload thread sat in update_mesh() holding the
    viewer mutex, so the main loop (viewer.lock()) and the viewer's destroy() both blocked forever.
    Fix: closing the window ends this lab program. When render_loop() returns, flush output and
    terminate the process right there (os._exit), so no pending upload or lock can keep it alive.
    """
    import mujoco.viewer as mv

    if getattr(mv._Simulate, '_lab_exit_on_close', False):
        return

    class _Simulate(mv._Simulate):
        _lab_exit_on_close = True

        def render_loop(self):
            try:
                super().render_loop()
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)

    mv._Simulate = _Simulate


class _MeshUploader:
    """Computes AND uploads the wave mesh on a daemon thread, in parallel with the physics.

    Measured 2026-11-28 (Hs 1.0, rock + beach walls): mesh 21.6 ms/frame vs physics 26.6 ms per
    1/30 s of sim. The mesh is visual only (contype 0) and numpy releases the GIL in the big array
    math, so doing it off the main thread hides most of its cost. A new mesh starts only when the
    previous one has been uploaded (vertices are never rewritten while being read).
    """

    def __init__(self, viewer, water, model, data):
        self.viewer, self.water, self.model, self.data = viewer, water, model, data
        self.thread = None

    def idle(self):
        return self.thread is None or not self.thread.is_alive()

    def _work(self):
        mesh_id = self.water.update_surface_visual(self.model, self.data)
        if mesh_id is not None:
            self.viewer.update_mesh(mesh_id)
            for extra in self.water.extra_meshes:
                self.viewer.update_mesh(extra)

    def start(self):
        import threading
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()


def rock_wet_fraction(swe):
    """Share of the rock that sticks out of still water that is now covered by > FILM_MIN of water."""
    if swe.t is None:
        return 0.0
    dry = swe.b > config.WaterParams.SURFACE_Z + 0.1
    return float(((swe.h > FILM_MIN) & dry).sum()) / max(int(dry.sum()), 1)


def overtop_foam(water, t, per_frame=2):
    """White foam where water is pouring over the rock (thin, fast sheet on top of it)."""
    swe, fx = water.swe, water.impact
    if swe.t is None:            # solver starts at the first physics step
        return
    h, qx, qy = swe.h, swe.qx, swe.qy
    ux = 0.5 * (qx[:, :-1] + qx[:, 1:]) / np.maximum(h, 1e-3)
    uy = 0.5 * (qy[:-1, :] + qy[1:, :]) / np.maximum(h, 1e-3)
    pour = (swe.b > config.WaterParams.SURFACE_Z) & (h > 0.04) & (np.hypot(ux, uy) > 0.6)
    j, i = np.nonzero(pour)
    if len(i) == 0:
        return
    U = fx.rng.uniform
    for k in fx.rng.choice(len(i), size=min(per_frame, len(i)), replace=False):
        fx._add_foam(float(swe.xc[i[k]]) + U(-.1, .1), float(swe.yc[j[k]]) + U(-.1, .1), t,
                     size=U(0.35, 0.6), vx=float(np.clip(ux[j[k], i[k]], -1, 1)) * 0.5)


def run_viewer(world, water, robot, waves, follow=False):
    import mujoco.viewer

    _patch_viewer_close()
    stats = Stats(world, water, robot, waves)
    model, data = world.model, world.data
    dt = model.opt.timestep
    knob = getattr(world, 'knob', None)

    def on_key(key):
        # keyboard backup for the slider: ',' weaker / '.' stronger
        if knob is not None and key in (KEY_WAVE_DOWN, KEY_WAVE_UP):
            step = WAVE_KEY_STEP if key == KEY_WAVE_UP else -WAVE_KEY_STEP
            data.ctrl[knob.actuator_id] = float(np.clip(data.ctrl[knob.actuator_id] + step, 0.0, WAVE_MAX))

    if knob is not None:
        print(f"wave strength: right panel > Control > '{knob.NAME}' slider, or keys ',' (down) '.' (up)")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        uploader = _MeshUploader(viewer, water, model, data)
        # hull shapes are Groups.PHYSICS (3); the viewer shows only groups 0-2 by default
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        # start between the boat and the rock, looking toward the shore; after that the mouse owns the camera
        viewer.cam.lookat[:] = ((BOAT_XY[0] + ROCK_CENTER[0]) / 2, (BOAT_XY[1] + ROCK_CENTER[1]) / 2, 0.0)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 22, 20, -20
        start_wall, start_sim = time.perf_counter(), data.time
        next_print = PRINT_EVERY
        fx_ms, fx_n, frames, last_wall, last_sim = [], [], 0, start_wall, data.time
        while viewer.is_running():
            frame_end = time.perf_counter() + 1 / 60
            if uploader.idle():   # wave mesh for this frame is computed while the physics below runs
                uploader.start()
            with viewer.lock():
                # Physics stays INSIDE the lock on purpose. This PC renders on the CPU (llvmpipe); measured
                # 2026-11-28: stepping outside the lock let the render thread run freely and fight the
                # physics for CPU -> 19-22 fps, sim/real 0.62-0.72 (vs 29-38 fps, ~1.0 inside the lock).
                target = start_sim + (time.perf_counter() - start_wall)
                for _ in range(int(min(target - data.time, 1 / 30) / dt)):
                    advance_world(world)
                stats.sample()
                if knob is not None:   # slider / keys -> live wave strength
                    strength = float(data.ctrl[knob.actuator_id]) / knob.base
                    if abs(strength - waves.strength) > 1e-6:
                        waves.strength = strength
                        print(f'  wave {knob.NAME} = {data.ctrl[knob.actuator_id]:.2f}'
                              f' (x{strength:.2f})', flush=True)
                if follow:   # camera follows the boat horizontally (mouse still controls angle/zoom)
                    viewer.cam.lookat[:2] = data.xpos[robot.body_id][:2]
                if water.impact is not None:
                    water.impact.update(data.time)
                    if water.swe is not None:
                        overtop_foam(water, data.time)
                    viewer.user_scn.ngeom = 0
                    water.impact.draw(viewer.user_scn, data.time, cam_position(viewer.cam), cam_forward(viewer.cam))
                    if water.beach is not None:   # after the reset (it was erased every frame before)
                        water.beach.draw(viewer.user_scn, data.time, cam_position(viewer.cam), cam_forward(viewer.cam))
                    fx_ms.append(water.impact.ms)
                    fx_n.append(viewer.user_scn.ngeom)
            viewer.sync()
            frames += 1
            if data.time >= next_print:
                print(stats.line())
                now = time.perf_counter()
                if water.swe is not None:
                    print(f'  swe {water.swe.ms:.1f} ms/call, {water.swe.steps} steps,'
                          f' rock wet {100 * rock_wet_fraction(water.swe):.0f} %', flush=True)
                if fx_ms:
                    print(f'  fx {np.mean(fx_ms):.1f} ms/frame (max {np.max(fx_ms):.1f}), sprites {np.mean(fx_n):.0f}'
                          f' (max {np.max(fx_n)}), {frames / (now - last_wall):.0f} fps,'
                          f' sim/real {(data.time - last_sim) / (now - last_wall):.2f}', flush=True)
                fx_ms.clear(), fx_n.clear()
                frames, last_wall, last_sim = 0, now, data.time
                next_print += PRINT_EVERY
            pause = frame_end - time.perf_counter()
            if pause > 0:
                time.sleep(pause)


# OpenGL renderer for the viewer (Mesa on WSL2). Measured 2026-11-28 on this PC: llvmpipe (CPU)
# 32-40 fps vs NVIDIA via d3d12 40-50 fps (physics / mesh math stay on the CPU either way)
RENDER_ENV = {
    'cpu': {'LIBGL_ALWAYS_SOFTWARE': '1'},
    'gpu': {'GALLIUM_DRIVER': 'd3d12', 'MESA_D3D12_DEFAULT_ADAPTER_NAME': 'NVIDIA'},
    'igpu': {'GALLIUM_DRIVER': 'd3d12', 'MESA_D3D12_DEFAULT_ADAPTER_NAME': 'Intel'},
}


def gl_renderer_name():
    """Name of the OpenGL renderer a new window gets with the current environment (None if unknown)."""
    try:
        import glfw
        from OpenGL.GL import GL_RENDERER, glGetString
    except ImportError:
        return None
    if not glfw.init():
        return None
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(16, 16, 'probe', None, None)
        if not win:
            return None
        glfw.make_context_current(win)
        name = glGetString(GL_RENDERER)
        glfw.destroy_window(win)
        return name.decode() if name else None
    finally:
        glfw.terminate()


def main():
    global ROCK_Z_SCALE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sea', default='irregular', choices=('irregular', 'three', 'team', 'regular', 'calm'))
    p.add_argument('--hs', type=float, default=0.6, help='significant wave height [m] (irregular)')
    p.add_argument('--tp', type=float, default=4.0, help='peak period [s] (irregular)')
    p.add_argument('--spread', type=float, default=30.0, help='directional spread [deg] (irregular)')
    p.add_argument('--n', type=int, default=16, help='number of components (irregular)')
    p.add_argument('--seed', type=int, default=SEEDS[0])
    p.add_argument('--no-shore', action='store_true', help='no shoaling attenuation near the beach')
    p.add_argument('--no-reflect', action='store_true', help='no beach reflection')
    p.add_argument('--human', action='store_true', help='add a floating person')
    p.add_argument('--no-rock', action='store_true', help='no rock island')
    p.add_argument('--no-swe', action='store_true',
                   help='analytic rock (reflection window + shadow + faked run-up) instead of the shallow-water box')
    p.add_argument('--rock-scale', type=float, default=ROCK_Z_SCALE,
                   help=f'rock height factor (1.0 = top 1.66 m, default {ROCK_Z_SCALE})')
    p.add_argument('--exact-sea', action='store_true', help='no amplitude grid (slow, for checking the speed-up)')
    p.add_argument('--no-splash', action='store_true', help='no splash / spray effect')
    p.add_argument('--no-beach-fx', action='store_true', help='no depth colours / wet sand / lace foam on the beach')
    p.add_argument('--throttle', type=float, default=0.0, help='boat throttle (-1 ~ 1)')
    p.add_argument('--flow', default='none', choices=('none', 'orbital', 'stokes'),
                   help='wave water velocity in the drag (experiment, see LabWater)')
    p.add_argument('--follow', action='store_true', help='camera tracks the boat (default: free camera, mouse only)')
    p.add_argument('--dt', type=float, default=LAB_DT,
                   help=f'physics step [s] for this lab (default {LAB_DT}; team config {config.Sim.TIMESTEP})')
    p.add_argument('--render', choices=tuple(RENDER_ENV), default=None,
                   help='viewer OpenGL: cpu (llvmpipe), gpu (NVIDIA via d3d12), igpu (Intel). default: system default')
    p.add_argument('--headless',type=float, default=None, metavar='SECONDS', help='run without viewer')
    args = p.parse_args()
    ROCK_Z_SCALE = args.rock_scale

    world, water, robot, waves = build_lab(args)
    print(f'dt={world.model.opt.timestep} (team config {config.Sim.TIMESTEP})')
    print(f'sea={args.sea} hs={args.hs} tp={args.tp} spread={args.spread} n={args.n} seed={args.seed} '
          f'shore={not args.no_shore} reflect={not args.no_reflect} human={args.human} flow={args.flow} '
          f'swe={getattr(waves, "swe", None) is not None} rock_top={rock_top(*ROCK_CENTER).max() if not args.no_rock else 0:.2f}m')
    if isinstance(waves, MySea):
        u = waves.stokes_drift(BOAT_XY[0], BOAT_XY[1], 0.0).reshape(3)
        print(f'surface Stokes drift at boat start (hand formula): ({u[0]:+.3f}, {u[1]:+.3f}) m/s')
    if args.headless is not None:
        run_headless(world, water, robot, waves, args.headless)
    else:
        if args.render is not None:
            for key in ('LIBGL_ALWAYS_SOFTWARE', 'GALLIUM_DRIVER', 'MESA_D3D12_DEFAULT_ADAPTER_NAME'):
                os.environ.pop(key, None)
            os.environ.update(RENDER_ENV[args.render])   # read by Mesa when the viewer window is created
        print(f'renderer: {gl_renderer_name() or "unknown"}', flush=True)
        run_viewer(world, water, robot, waves, follow=args.follow)


if __name__ == '__main__':
    main()
