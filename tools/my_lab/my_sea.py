"""Sea experiments for the irregular-wave lab (shared for review, not used by sim/).

MySea subclasses sim.water.sea.Sea so that sim.water.water still recognises it
(water._sea() uses isinstance). Override methods here to try new irregular-wave
ideas without touching the shared sim/water/sea.py.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.water.sea import Sea  # noqa: E402


import threading  # noqa: E402

import numpy as np  # noqa: E402


SHADOW_FLOOR = 0.3   # part of each wave that still bends (diffracts) into the lee of a rock
AMP_GRID = 0.25      # [m] grid for the precomputed (time-independent) amplitudes
ACTIVE_MIN = 1e-4    # [m] amplitudes below this are skipped on large grids (reflections are local)
POINTS_MAX = 1000    # queries up to this many points use the amplitude grid (buoyancy, drag, sprites)
# Reuse of small eta queries (physics): the boat asks for the water level at its voxels every 2 ms
# step (and robot.py asks again for its intakes), but waves change on a ~4 s scale. Measured
# 2026-11-28: eta = 60 % of a physics step. A fresh answer also stores d(eta)/dt; within REUSE_DT
# and while the points moved less than REUSE_DX, the answer is eta + rate * (t - t0).
REUSE_DT = 0.010     # [s]
REUSE_DX = 0.03      # [m]
REUSE_SLOTS = 4      # cached query sets per point count (different callers can share a size)


class MySea(Sea):
    """Sea + experimental wave-drift helpers + speed-ups. Results match Sea up to grid interpolation.

    Speed-up (ported from tools/splash_lab/hanagae_lab.py HSea, 2026-11-28):
      Profiling 500 physics steps: Sea.eta = 58 % of a step, and 2/3 of that was Sea._static
      (rock shadow, wall window, shore factor) recomputed every step although it does not change
      with time. set_amp_grid() computes it once on a grid; point queries interpolate it and sum
      all groups' components in one numpy expression.
    """

    _amp = None
    strength = 1.0   # live wave-strength factor (viewer slider). eta, rise, orbital scale linearly, drift as ^2
    reuse = True     # False: every query is computed fresh (--exact-sea)

    def _eta_now(self, x, y, t):
        return Sea.eta(self, x, y, t) * self.strength

    def _rise_now(self, x, y, t):
        return Sea.rise(self, x, y, t) * self.strength

    def eta(self, x, y, t):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        n = x.size
        if not self.reuse or n == 0 or n > POINTS_MAX:
            return self._eta_now(x, y, t)
        local = self.__dict__.get('_tls')
        if local is None:
            local = self.__dict__.setdefault('_tls', threading.local())   # caches are per thread
        slots = getattr(local, 'slots', None)
        if slots is None:
            slots = local.slots = {}
        t = float(t)
        entries = slots.setdefault(n, [])
        for e in entries:
            t0, s0, x0, y0, eta0, rate0 = e
            if (s0 == self.strength and 0.0 <= t - t0 < REUSE_DT and x0.shape == x.shape
                    and np.abs(x - x0).max() < REUSE_DX and np.abs(y - y0).max() < REUSE_DX):
                return eta0 + rate0 * (t - t0)
        eta0 = np.asarray(self._eta_now(x, y, t), float).reshape(x.shape)
        rate0 = np.asarray(self._rise_now(x, y, t), float).reshape(x.shape)
        entries.insert(0, (t, self.strength, x.copy(), y.copy(), eta0, rate0))
        del entries[REUSE_SLOTS:]
        return eta0

    def rise(self, x, y, t):
        return self._rise_now(x, y, t)

    def orbital(self, x, y, z, t):
        return super().orbital(x, y, z, t) * self.strength

    @property
    def amax(self):
        return super().amax * max(self.strength, 1.0)   # never smaller than the built mesh range

    def set_amp_grid(self, x0, x1, y0, y1, step=AMP_GRID):
        """Precompute every group's amplitude field. Call after all walls/shadows/shores are added."""
        self._amp = None
        gx, gy = np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step)
        X, Y = np.meshgrid(gx, gy)
        grids = [np.ascontiguousarray(Sea._static(self, gi, X, Y)) for gi in range(len(self.groups))]
        self.__dict__.pop('_cache', None)
        self.__dict__.pop('_active', None)
        self._amp = (x0, y0, step, len(gx), len(gy), grids)
        # per group: where is it non-zero at all? (wall reflections only live near their wall)
        self._grp = [(G, G.max(axis=0) > ACTIVE_MIN, g[2], g[3], g[4], g[5]) for G, g in zip(grids, self.groups)]
        return self

    @staticmethod
    def _bilinear(G, amp, x, y):
        x0, y0, step, nx, ny, _ = amp
        u, v = (x - x0) / step, (y - y0) / step
        if u.min() < 0 or v.min() < 0 or u.max() > nx - 1.001 or v.max() > ny - 1.001:
            return None
        i, j = u.astype(int), v.astype(int)
        fu, fv = u - i, v - j
        return (G[:, j, i] * (1 - fu) * (1 - fv) + G[:, j, i + 1] * fu * (1 - fv)
                + G[:, j + 1, i] * (1 - fu) * fv + G[:, j + 1, i + 1] * fu * fv)

    def _static(self, gi, x, y):
        if self._amp is None or x.size == 0 or x.size > POINTS_MAX:
            return super()._static(gi, x, y)
        out = self._bilinear(self._amp[5][gi], self._amp, x, y)
        return super()._static(gi, x, y) if out is None else out

    def _sum(self, x, t, y, deriv):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(0.0 if y is None else y, float))
        if self._amp is not None and 0 < x.size <= POINTS_MAX:
            xf, yf = x.ravel(), y.ravel()
            x0, y0, step, nx, ny, _ = self._amp
            u, v = (xf - x0) / step, (yf - y0) / step
            if u.min() >= 0 and v.min() >= 0 and u.max() <= nx - 1.001 and v.max() <= ny - 1.001:
                i, j = u.astype(int), v.astype(int)
                fu, fv = u - i, v - j
                w00, w10, w01, w11 = (1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv
                out = np.zeros(xf.size)
                for G, active, kx, ky, w, ph in self._grp:
                    if not active[j, i].any():          # e.g. rock reflection far from the boat
                        continue
                    A = G[:, j, i] * w00 + G[:, j, i + 1] * w10 + G[:, j + 1, i] * w01 + G[:, j + 1, i + 1] * w11
                    phase = kx[:, None] * xf + ky[:, None] * yf - (w * t - ph)[:, None]
                    if deriv:
                        out += ((A * -w[:, None]) * np.cos(phase)).sum(0)
                    else:
                        out += (A * np.sin(phase)).sum(0)
                return out.reshape(x.shape)
        if x.size <= POINTS_MAX:
            return super()._sum(x, t, y, deriv)
        # large grids (wave mesh): skip points where a local group (wall reflection) is ~0
        out = np.zeros(x.size)
        cache = self.__dict__.setdefault('_active', {})
        for gi, (window, a, kx, ky, w, ph, _) in enumerate(self.groups):
            key = (gi, x.shape, float(x.flat[0]), float(x.flat[-1]), float(y.flat[0]), float(y.flat[-1]))
            hit = cache.get(key)
            if hit is None:
                A = super()._static(gi, x, y).reshape(len(a), -1)
                idx = np.nonzero(A.max(axis=0) > ACTIVE_MIN)[0] if window is not None else slice(None)
                if len(cache) > 64:
                    cache.clear()
                hit = cache[key] = (idx, np.ascontiguousarray(A[:, idx]), x.ravel()[idx], y.ravel()[idx])
            idx, A, xs, ys = hit
            phase = kx[:, None] * xs + ky[:, None] * ys - (w * t - ph)[:, None]
            if deriv:
                out[idx] += np.einsum('ij,ij->j', A * (-w)[:, None], np.cos(phase))
            else:
                out[idx] += np.einsum('ij,ij->j', A, np.sin(phase))
        return out.reshape(x.shape)

    def _shadow_factor(self, kx, ky, x, y):
        """Rock shadow with a diffraction floor: the lee is calmer, not dead flat.

        Team Sea leaves ~10 % of the wave right behind a 5 m rock (measured Hs 0.10 m vs 0.92 m
        in open water, 2026-11-28). A rock much smaller than the wavelength cannot block that much,
        so keep SHADOW_FLOOR of the amplitude everywhere in the shadow.
        """
        base = super()._shadow_factor(kx, ky, x, y)
        return SHADOW_FLOOR + (1.0 - SHADOW_FLOOR) * base

    def stokes_drift(self, x, y, z):
        """Mean wave-induced drift (Stokes drift) [m/s] at (x, y, z), world (u, v, 0).

        Deep-water, per component: u_s = a^2 * omega * k * exp(2 k z) along the travel direction.
        Amplitudes include shore attenuation / reef shadow (Sea._static). z is clipped to <= 0.
        Approximation: components are summed independently (no cross terms).
        """
        x, y = (np.asarray(v, float) for v in (x, y))
        z = np.minimum(np.asarray(z, float), 0.0)
        sh = (-1,) + (1,) * x.ndim
        drift = np.zeros((3,) + x.shape)
        for gi, (_, amp, kx, ky, omega, phase, _) in enumerate(self.groups):
            k = np.hypot(kx, ky)
            a = self._static(gi, x, y)
            speed = a ** 2 * (omega * k).reshape(sh) * np.exp(2 * k.reshape(sh) * z)
            drift[0] += np.sum(speed * (kx / k).reshape(sh), axis=0)
            drift[1] += np.sum(speed * (ky / k).reshape(sh), axis=0)
        return drift * self.strength ** 2
