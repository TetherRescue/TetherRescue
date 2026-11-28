"""Local shallow-water solver around the rock (personal lab, visual + boat buoyancy).

Why: the analytic Sea can only fake the rock (reflection window + shadow + mesh run-up). Here the
water around the rock is computed: waves pile up on the face, pour OVER the rock when the crest is
higher than the rock, run off the back and leave a calmer lee. Nothing about the rock is tuned in.

Scheme: LISFLOOD-ACC local-inertia shallow water (Bates et al. 2010, J. Hydrol. 387) with the
theta-weighted flux of de Almeida et al. 2012 (WRR 48) and Manning friction. Staggered grid:
depth h at cell centres, unit-width flux qx / qy on cell faces. Wetting / drying comes out of the
face flow depth max(eta) - max(bed), so overtopping a rock crest needs no special case.
Limits: depth-averaged (no overturning breakers, no airborne water -> sprites do that), shallow-
water wave speed sqrt(g h) instead of the full dispersion (short components travel a bit fast).

Boundary: a sponge band (SPONGE m) relaxes eta and flux toward the incoming analytic sea (the same
Sea WITHOUT the rock), so waves enter and the rock's reflection leaves without bouncing back.
"""
import math

import mujoco
import numpy as np

G = 9.81
DX = 0.25          # [m] cell size
SPONGE = 2.0       # [m] relaxation band at the domain edge
CFL = 0.7          # dt = CFL * dx / (sqrt(g h_max) + 1 m/s)
DT_MAX = 0.04      # [s]
THETA = 0.7        # flux smoothing (1 = none); <1 damps the checkerboard mode of the explicit scheme
MANNING = 0.015    # [s/m^(1/3)] bed friction (smooth sand / rock)
WET = 1e-3         # [m] faces shallower than this carry no flow
SAMPLE_WET = 0.01  # [m] cells shallower than this are "dry" when sampling the water level
BREAK_GAMMA = 0.55 # crest above still water > this x still depth -> the wave breaks there
BREAK_TAU = 0.4    # [s] flux damping time in breaking cells (energy loss, mass is kept)


def ellipsoid_top(center, size, quat, x, y):
    """Highest z of a (rotated) ellipsoid above each (x, y); -inf where the vertical misses it."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    R = R.reshape(3, 3)
    r = np.asarray(size, float)
    p0 = np.stack([x - center[0], y - center[1], np.full_like(x, -center[2])], axis=-1)
    q0 = p0 @ R / r                  # local, scaled: q = R^T (p - c) / r  (row vectors)
    d = R[2, :] / r                  # d q / d z
    a = float(d @ d)
    b = 2.0 * (q0 @ d)
    c = np.einsum('...i,...i->...', q0, q0) - 1.0
    disc = b * b - 4 * a * c
    top = (-b + np.sqrt(np.maximum(disc, 0.0))) / (2 * a)
    return np.where(disc >= 0, top + 0.0, -np.inf)


class RockSWE:
    """Shallow water on [x0, x1] x [y0, y1].

    bed(x, y) -> z. target(x, y, t) -> (eta, ex, ey): incoming level and the same waves weighted by
    their travel direction (sum of a sin(th) k/|k|). The incoming flux is q = sqrt(g h) * (ex, ey):
    the speed waves really have in this solver. Using the analytic omega/k (6.2 m/s deep-water speed
    for Tp 4 s vs sqrt(g h) = 4.3 m/s at 1.9 m) pushed in too much water and grew the waves inside
    (measured Hs 1.97 m in the middle of a rock-free box for Hs 1.0 at the edge).
    """

    def __init__(self, bounds, bed, target, surface=0.0, dx=DX):
        x0, x1, y0, y1 = bounds
        self.bounds, self.dx, self.surface = bounds, dx, surface
        self.nx, self.ny = int(round((x1 - x0) / dx)), int(round((y1 - y0) / dx))
        self.xc = x0 + (np.arange(self.nx) + 0.5) * dx
        self.yc = y0 + (np.arange(self.ny) + 0.5) * dx
        X, Y = np.meshgrid(self.xc, self.yc)
        self.X, self.Y = X, Y
        self.b = bed(X, Y)
        self.still = np.maximum(surface - self.b, 0.0)   # still-water depth (0 on the rock)
        self.target = target
        # sponge weight: 1 at the edge, 0 at SPONGE inside
        d = np.minimum.reduce([X - x0, x1 - X, Y - y0, y1 - Y])
        self.w = np.clip(1.0 - (d - 0.5 * dx) / SPONGE, 0.0, 1.0) ** 2
        band = self.w > 0
        grow = band.copy()                       # one more cell so every band face has both neighbours
        grow[1:, :] |= band[:-1, :]
        grow[:-1, :] |= band[1:, :]
        grow[:, 1:] |= band[:, :-1]
        grow[:, :-1] |= band[:, 1:]
        self.band_idx = np.nonzero(grow.ravel())[0]
        self._c_band = np.sqrt(G * np.maximum(self.still.ravel()[self.band_idx], 0.05))
        pad = np.pad(self.w, 1, mode='edge')
        self.wfx = np.maximum(pad[1:-1, :-1], pad[1:-1, 1:])   # (ny, nx+1)
        self.wfy = np.maximum(pad[:-1, 1:-1], pad[1:, 1:-1])   # (ny+1, nx)
        self.t = None
        self.steps = 0
        self.ms = 0.0

    # ── state ────────────────────────────────────────────────
    def reset(self, t):
        """Start from the incoming sea everywhere (no 'calm box' that waves first have to fill)."""
        eta, ex, ey = self.target(self.X.ravel(), self.Y.ravel(), t)
        eta, ex, ey = (v.reshape(self.X.shape) for v in (eta, ex, ey))
        c = np.sqrt(G * np.maximum(self.still, 0.05))
        qx, qy = c * ex, c * ey
        self.h = np.maximum(eta - self.b, 0.0)
        dry = self.h <= WET
        qx, qy = np.where(dry, 0.0, qx), np.where(dry, 0.0, qy)
        self.qx = np.zeros((self.ny, self.nx + 1))
        self.qy = np.zeros((self.ny + 1, self.nx))
        self.qx[:, 1:-1] = 0.5 * (qx[:, :-1] + qx[:, 1:])
        self.qy[1:-1, :] = 0.5 * (qy[:-1, :] + qy[1:, :])
        self.rate = np.zeros_like(self.h)
        self.breaking = np.zeros(self.h.shape, bool)
        self.level = self._level(self.h)
        self.t = float(t)

    def _level(self, h):
        """Water level for sampling: eta where wet, NaN where dry (so dry rock never lifts a wave)."""
        return np.where(h > SAMPLE_WET, self.b + h, np.nan)

    def _targets(self, t):
        n = self.X.size
        eta, qx, qy = (np.zeros(n) for _ in range(3))
        i = self.band_idx
        eta[i], ex, ey = self.target(self.X.ravel()[i], self.Y.ravel()[i], t)
        c = self._c_band
        qx[i], qy[i] = c * ex, c * ey
        eta, qx, qy = (v.reshape(self.X.shape) for v in (eta, qx, qy))
        pad_x, pad_y = np.pad(qx, ((0, 0), (1, 1)), mode='edge'), np.pad(qy, ((1, 1), (0, 0)), mode='edge')
        return eta, 0.5 * (pad_x[:, :-1] + pad_x[:, 1:]), 0.5 * (pad_y[:-1, :] + pad_y[1:, :])

    def advance(self, t):
        """Step the water up to time t (called every physics step; usually does nothing)."""
        import time
        # self.t may run up to one step ahead of t; only a real jump back (sim reset) restarts the box
        if self.t is None or t < self.t - 2 * DT_MAX:
            self.reset(t)
            return
        t0 = time.perf_counter()
        while self.t < t - 1e-9:
            dt = min(CFL * self.dx / (math.sqrt(G * max(float(self.h.max()), 0.05)) + 1.0), DT_MAX)
            if self.t + dt > t + DT_MAX:          # never run ahead of the caller by more than one step
                break
            self._step(dt, self.t + dt)
            self.t += dt
            self.steps += 1
        self.ms = 1000 * (time.perf_counter() - t0)

    def _face_flux(self, q, eta_l, eta_r, b_l, b_r, q_prev, q_next, dt):
        hf = np.maximum(eta_l, eta_r) - np.maximum(b_l, b_r)
        slope = (eta_r - eta_l) / self.dx
        q_s = THETA * q + 0.5 * (1.0 - THETA) * (q_prev + q_next)
        hs = np.maximum(hf, WET)
        out = (q_s - G * hs * dt * slope) / (1.0 + G * dt * MANNING ** 2 * np.abs(q) / hs ** (7.0 / 3.0))
        return np.where(hf > WET, out, 0.0)

    def _step(self, dt, t_new):
        h, b, dx = self.h, self.b, self.dx
        eta = b + h
        qx, qy = self.qx, self.qy
        nqx, nqy = np.zeros_like(qx), np.zeros_like(qy)
        # interior faces (neighbour fluxes for theta: edge faces reuse themselves)
        pqx = np.pad(qx, ((0, 0), (1, 1)), mode='edge')
        nqx[:, 1:-1] = self._face_flux(qx[:, 1:-1], eta[:, :-1], eta[:, 1:], b[:, :-1], b[:, 1:],
                                       pqx[:, 1:-3], pqx[:, 3:-1], dt)
        pqy = np.pad(qy, ((1, 1), (0, 0)), mode='edge')
        nqy[1:-1, :] = self._face_flux(qy[1:-1, :], eta[:-1, :], eta[1:, :], b[:-1, :], b[1:, :],
                                       pqy[1:-3, :], pqy[3:-1, :], dt)
        # depth-limited breaking (the plain shallow-water equations never break, so waves on the
        # shoaling bed kept growing: measured Hs 1.97 m in 1.2 m of water for Hs 1.0 offshore)
        brk = (eta - self.surface > BREAK_GAMMA * self.still) & (self.still > 0)
        if brk.any():
            damp = 1.0 / (1.0 + dt / BREAK_TAU)
            bx = np.pad(brk, ((0, 0), (1, 1)))
            by = np.pad(brk, ((1, 1), (0, 0)))
            nqx *= np.where(bx[:, :-1] | bx[:, 1:], damp, 1.0)
            nqy *= np.where(by[:-1, :] | by[1:, :], damp, 1.0)
        self.breaking = brk
        # sponge: pull fluxes and levels toward the incoming sea (edge faces = fully prescribed)
        eta_t, qx_t, qy_t = self._targets(t_new)
        nqx += self.wfx * (qx_t - nqx)
        nqy += self.wfy * (qy_t - nqy)
        # no cell may lose more water than it has (keeps h >= 0 on drying rock / sand)
        out = (np.maximum(nqx[:, 1:], 0) + np.maximum(-nqx[:, :-1], 0)
               + np.maximum(nqy[1:, :], 0) + np.maximum(-nqy[:-1, :], 0)) * dt / dx
        f = np.where(out > h, h / np.maximum(out, 1e-12), 1.0)
        fx = np.pad(f, ((0, 0), (1, 1)), constant_values=1.0)
        fy = np.pad(f, ((1, 1), (0, 0)), constant_values=1.0)
        nqx *= np.where(nqx > 0, fx[:, :-1], fx[:, 1:])
        nqy *= np.where(nqy > 0, fy[:-1, :], fy[1:, :])
        h_new = h + dt / dx * (nqx[:, :-1] - nqx[:, 1:] + nqy[:-1, :] - nqy[1:, :])
        h_new = np.maximum(h_new, 0.0)
        h_new += self.w * (np.maximum(eta_t - b, 0.0) - h_new)
        self.qx, self.qy = nqx, nqy
        self.rate = (h_new - h) / dt
        self.h = h_new                         # swapped, never written in place: the mesh thread reads it
        self.level = self._level(h_new)

    # ── queries ──────────────────────────────────────────────
    def contains(self, x, y):
        x0, x1, y0, y1 = self.bounds
        h = 0.5 * self.dx
        return (x >= x0 + h) & (x <= x1 - h) & (y >= y0 + h) & (y <= y1 - h)

    def _corners(self, x, y):
        u = np.clip((x - self.xc[0]) / self.dx, 0, self.nx - 1.001)
        v = np.clip((y - self.yc[0]) / self.dx, 0, self.ny - 1.001)
        i, j = u.astype(int), v.astype(int)
        fu, fv = u - i, v - j
        return i, j, ((1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv)

    def sample(self, x, y):
        """Water level at (x, y): bilinear over the WET corners only; all dry -> bed (rock surface)."""
        level, b = self.level, self.b
        i, j, w = self._corners(x, y)
        idx = ((j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1))
        num, den, bed = 0.0, 0.0, 0.0
        for (jj, ii), wk in zip(idx, w):
            lv = level[jj, ii]
            wet = ~np.isnan(lv)
            num = num + np.where(wet, wk * np.nan_to_num(lv), 0.0)
            den = den + np.where(wet, wk, 0.0)
            bed = bed + wk * b[jj, ii]
        return np.where(den > 1e-6, num / np.maximum(den, 1e-6), bed)

    def sample_rate(self, x, y):
        i, j, (w00, w10, w01, w11) = self._corners(x, y)
        r = self.rate
        return r[j, i] * w00 + r[j, i + 1] * w10 + r[j + 1, i] * w01 + r[j + 1, i + 1] * w11


# ── sea with the solver patched in ───────────────────────────
from my_sea import MySea  # noqa: E402
from sim.water.sea import Sea  # noqa: E402


class SweSea(MySea):
    """MySea (built WITHOUT the rock wall / shadow) + RockSWE inside its domain.

    eta / rise inside the domain come from the solver, so buoyancy, drag, the wave mesh, swash and
    sprites all see the same water. Outside: the analytic sea, which is also the sponge target.
    advance(t) must be called from the physics thread (LabWater.add_forces does it every step).
    orbital / stokes_drift stay analytic (no rock effect in the --flow experiment).
    """

    swe = None

    def attach_swe(self, bounds, bed, surface=0.0):
        self._tcache = {}
        self.swe = RockSWE(bounds, bed, self.incoming, surface=surface)
        return self

    def _amps(self, x, y):
        """Time-free part of every component at these points, as matrices (n_components, n_points).

        sin(th0 - w t) = sin(th0) cos(w t) - cos(th0) sin(w t), th0 = kx x + ky y + phase, so a query
        is two small matrix-vector products per output instead of one sin per component and point.
        """
        key = (x.size, float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
        hit = self._tcache.get(key)
        if hit is None:
            rows_s, rows_c, ws = [], [], []
            for gi, (_, _, kx, ky, w, ph, _) in enumerate(self.groups):
                A = np.asarray(Sea._static(self, gi, x, y), float).reshape(len(kx), -1)
                th0 = kx[:, None] * x + ky[:, None] * y + ph[:, None]
                k = np.hypot(kx, ky)
                for scale in (np.ones_like(k), kx / k, ky / k):   # eta, direction-weighted eta (x, y)
                    rows_s.append(A * np.sin(th0) * scale[:, None])
                    rows_c.append(A * np.cos(th0) * scale[:, None])
                ws.append(w)
            S = [np.vstack(rows_s[j::3]) for j in range(3)]
            C = [np.vstack(rows_c[j::3]) for j in range(3)]
            hit = (np.concatenate(ws), S, C)
            if len(self._tcache) > 8:
                self._tcache.clear()
            self._tcache[key] = hit
        return hit

    def incoming(self, x, y, t):
        """Analytic sea without the rock: (eta, ex, ey). RockSWE turns (ex, ey) into flux with sqrt(g h)."""
        x, y = np.ravel(x).astype(float), np.ravel(y).astype(float)
        w, S, C = self._amps(x, y)
        cw, sw = np.cos(w * t), np.sin(w * t)
        # einsum, not @: BLAS threads cost ~2.6 ms for this tiny product on this PC and then spin (measured)
        eta, qx, qy = (np.einsum('i,ij->j', cw, S[j]) - np.einsum('i,ij->j', sw, C[j]) for j in range(3))
        k = self.strength
        return self.swe.surface + k * eta, k * qx, k * qy

    def advance(self, t):
        if self.swe is not None:
            self.swe.advance(float(t))

    def _split(self, x, y, t, inner, outer):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        if self.swe is None or self.swe.t is None:
            return outer(x, y, t)
        inside = self.swe.contains(x, y)
        if not inside.any():
            return outer(x, y, t)
        if inside.all():
            return inner(x, y)
        out = np.empty(x.shape)
        out[inside] = inner(x[inside], y[inside])
        out[~inside] = np.asarray(outer(x[~inside], y[~inside], t), float).reshape(-1)
        return out

    # MySea.eta wraps these with the small-query reuse (my_sea.REUSE_*)
    def _eta_now(self, x, y, t):
        return self._split(x, y, t, self.swe.sample if self.swe else None, super()._eta_now)

    def _rise_now(self, x, y, t):
        return self._split(x, y, t, self.swe.sample_rate if self.swe else None, super()._rise_now)
