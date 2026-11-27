"""반사·그늘·해안 감쇠가 있는 수면. 화면 시험 도구와 본 시뮬이 이 파일만 쓴다.

성분 하나 = (진폭 m, 파장 m, 진행 방향 rad, 위상 rad). 숫자는 config.WaterParams.
"""

import math

import numpy as np

from sim import config

# regular: 시험 도구가 처음 쓰던 사인파 하나. three·team 은 config 를 읽는다.
REGULAR_AMP = 0.30
REGULAR_LENGTH = 6.0
REGULAR_PERIOD = 3.0
# 반사 (벽)
R_ROCK = 0.8          # 바위(수직 벽) 반사율
R_BEACH = 0.3         # 모래사장(경사면) 반사율
ROCK_SPREAD_DEG = 35  # 바위 반사파가 퍼지는 각
ROCK_DECAY = 8.0      # 바위 반사파가 약해지는 거리 [m]
# 해안: 파고가 SHORE_GAMMA x (수심 + SHORE_SWASH) 를 넘지 못하게 줄인다.
SHORE_GAMMA = 0.8
SHORE_SWASH = 0.35
SHADOW_EDGE = 0.15    # 바위 바로 뒤 그늘 경계의 번짐 [m]
SHADOW_BLUR = 0.35    # 멀어질수록 경계가 번지는 정도
SHADOW_KEEP = 2.0     # 그늘이 유지되는 거리 = 이 값 x 바위 반폭 + 4 w0²/λ


class Sea:
    """수면 = 사인파 성분들의 합 + 벽 반사 + 바위 그늘.

    three     : config.WaterParams.WAVES_THREE
    team      : config.WaterParams.WAVES_DEFAULT
    regular   : 사인파 하나 (REGULAR_*)
    irregular : Pierson–Moskowitz. 시드가 같으면 같은 바다.
    """

    def __init__(self, kind='regular', hs=2 * REGULAR_AMP, tp=REGULAR_PERIOD, spread_deg=30.0, n=16, seed=0):
        self.kind = kind
        if kind in ('regular', 'three', 'team'):
            comps = {
                'regular': ((REGULAR_AMP, REGULAR_LENGTH, 0.0, 0.0),),
                'three': config.WaterParams.WAVES_THREE,
                'team': config.WaterParams.WAVES_DEFAULT,
            }[kind]
            a = np.array([c[0] for c in comps])
            k = 2 * math.pi / np.array([c[1] for c in comps])
            th = np.array([c[2] for c in comps])
            gravity = config.Physics.GRAVITY
            w = np.array([2 * math.pi / REGULAR_PERIOD]) if kind == 'regular' else np.sqrt(gravity * k)
            ph = np.array([c[3] for c in comps]) + math.pi / 2
            a_ref = float(np.sqrt(np.sum(a ** 2)))
            rise_ref = float(np.sqrt(np.sum((a * w) ** 2)))
        else:
            rng = np.random.default_rng(seed)
            fp = 1 / tp
            edges = np.linspace(0.65 * fp, 2.6 * fp, n + 1)
            f = 0.5 * (edges[:-1] + edges[1:]) + rng.uniform(-.3, .3, n) * np.diff(edges)
            df = np.diff(edges)
            spectrum = (5 / 16) * hs ** 2 * fp ** 4 * f ** -5 * np.exp(-1.25 * (fp / f) ** 4)
            a = np.sqrt(2 * spectrum * df)
            a *= (hs / 4) / math.sqrt(0.5 * np.sum(a ** 2))
            w = 2 * math.pi * f
            k = w ** 2 / config.Physics.GRAVITY
            th = np.radians(spread_deg) * np.clip(rng.normal(0, 0.5, n), -1.5, 1.5)
            ph = rng.uniform(0, 2 * math.pi, n)
            a_ref = hs / 2
            rise_ref = a_ref * 2 * math.pi / tp
        self.a_ref, self.rise_ref = a_ref, rise_ref
        self.groups = [(None, a, k * np.cos(th), k * np.sin(th), w, ph, True)]
        self.walls = []
        self.shadows = []
        self.shores = []

    def scaled(self, factor):
        """입사·반사 진폭을 처음 만든 바다의 factor 배로 둔다. 여러 번 불러도 곱이 쌓이지 않는다."""
        factor = float(factor)
        if not np.isfinite(factor) or factor < 0:
            raise ValueError('파도 배율은 0 이상이어야 합니다.')
        if not hasattr(self, '_base_amps'):
            self._base_amps = [np.array(group[1], copy=True) for group in self.groups]
            self._base_refs = (self.a_ref, self.rise_ref)
        for group, base in zip(self.groups, self._base_amps):
            group[1][:] = base * factor
        self.a_ref = self._base_refs[0] * factor
        self.rise_ref = self._base_refs[1] * factor
        self.__dict__.pop('_cache', None)
        self.__dict__.pop('_active', None)
        return self

    def add_shore(self, point, seaward, slope):
        """경사 해안: point 에서 수심 0, seaward 방향으로 갈수록 slope 로 깊어짐."""
        n = np.asarray(seaward, float)
        self.shores.append((np.asarray(point, float), n / np.linalg.norm(n), float(slope)))
        self.__dict__.pop('_cache', None)

    def _shore_factor(self, x, y):
        """해안 감쇠 (0~1). 깊은 곳은 1."""
        f = np.ones(np.broadcast(x, y).shape)
        a_sig = 1.4 * self.a_ref
        for p, n, slope in self.shores:
            depth = slope * ((x - p[0]) * n[0] + (y - p[1]) * n[1])
            f = np.minimum(f, np.clip(SHORE_GAMMA * (depth + SHORE_SWASH) / (2 * a_sig), 0.0, 1.0))
        return f

    def add_shadow(self, center, yaw, half_x, half_y):
        """파도를 가리는 박스. 성분마다 진행 방향 뒤쪽에 그늘이 생긴다."""
        self.shadows.append((np.asarray(center, float), yaw, half_x, half_y))

    def _shadow_factor(self, kx, ky, x, y):
        """성분마다 그 위치에서 남는 진폭 비율 (1 = 그늘 없음)."""
        sh = (-1,) + (1,) * x.ndim
        k = np.hypot(kx, ky)
        th = np.arctan2(ky, kx).reshape(sh)
        lam = (2 * math.pi / k).reshape(sh)
        factor = np.ones((len(kx),) + x.shape)
        for center, yaw, hx, hy in self.shadows:
            rel = th - yaw
            w0 = hx * np.abs(np.sin(rel)) + hy * np.abs(np.cos(rel))
            hd = hx * np.abs(np.cos(rel)) + hy * np.abs(np.sin(rel))
            dx, dy = x - center[0], y - center[1]
            along = np.cos(th) * dx + np.sin(th) * dy
            lat = -np.sin(th) * dx + np.cos(th) * dy
            behind = np.maximum(along - hd, 0.0)
            sigma = SHADOW_EDGE + SHADOW_BLUR * np.sqrt(lam * behind)
            lateral = 0.5 * (1 - np.tanh((np.abs(lat) - w0) / sigma))
            started = 0.5 * (1 + np.tanh((along + hd) / 0.1))
            recover = np.exp(-behind / (SHADOW_KEEP * w0 + 4 * w0 ** 2 / lam))
            factor *= 1 - started * lateral * recover
        return factor

    @property
    def amax(self):
        """입사 진폭 합 x (1 + 반사율 합). 화면 범위용."""
        return float(np.sum(np.abs(self.groups[0][1]))) * (1 + sum(reflect for _, reflect in self.walls))

    def add_wall(self, center, normal, reflect, half_len=None, spread_deg=ROCK_SPREAD_DEG, decay=ROCK_DECAY,
                 name='wall', shadowed=True):
        """벽 하나. normal 은 벽에서 바다 쪽. half_len 이 없으면 끝없이 긴 벽."""
        _, amp, kx, ky, omega, phase, _ = self.groups[0]
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        c = np.asarray(center, float)
        kn = kx * n[0] + ky * n[1]
        hit = kn < 0
        if not hit.any():
            return
        kx_r, ky_r = kx - 2 * kn * n[0], ky - 2 * kn * n[1]
        phase_r = phase + 2 * kn * (n @ c)
        if half_len is None:
            window, gain = None, reflect
        else:
            t_dir = np.array([-n[1], n[0]])
            tan_s = math.tan(math.radians(spread_deg))

            def window(x, y):
                dx, dy = x - c[0], y - c[1]
                s = dx * n[0] + dy * n[1]
                u = dx * t_dir[0] + dy * t_dir[1]
                sp = np.maximum(s, 0.0)
                width = half_len + sp * tan_s
                lateral = 0.5 * (1 - np.tanh((np.abs(u) - width) / 0.4))
                front = 0.5 * (1 + np.tanh((s + 0.3) / 0.1))
                return reflect * front * lateral * np.sqrt(half_len / width) * np.exp(-sp / decay)

            gain = 1.0
        self.groups.append((window, amp[hit] * gain, kx_r[hit], ky_r[hit], omega[hit], phase_r[hit], shadowed))
        self.walls.append((name, reflect))
        self.__dict__.pop('_base_amps', None)

    def _static(self, gi, x, y):
        """그룹 gi 의 시간 무관 진폭. 큰 격자는 한 번만 계산한다."""
        window, amp, kx, ky, _, _, shadowed = self.groups[gi]
        key = None
        if x.size > 1000:
            key = (gi, x.shape, float(x.flat[0]), float(x.flat[-1]), float(y.flat[0]), float(y.flat[-1]))
            hit = getattr(self, '_cache', {}).get(key)
            if hit is not None:
                return hit
        sh = (-1,) + (1,) * x.ndim
        amplitude = np.broadcast_to(amp.reshape(sh), (len(amp),) + x.shape)
        if shadowed and self.shadows:
            amplitude = amplitude * self._shadow_factor(kx, ky, x, y)
        if window is not None:
            amplitude = amplitude * window(x, y)
        if self.shores:
            amplitude = amplitude * self._shore_factor(x, y)
        if key is not None:
            self.__dict__.setdefault('_cache', {})[key] = amplitude
        return amplitude

    def _sum(self, x, t, y, deriv):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(0.0 if y is None else y, float))
        sh = (-1,) + (1,) * x.ndim
        out = np.zeros(x.shape)
        for gi, (_, amp, kx, ky, omega, phase, _) in enumerate(self.groups):
            angle = kx.reshape(sh) * x + ky.reshape(sh) * y - omega.reshape(sh) * t + phase.reshape(sh)
            amplitude = self._static(gi, x, y)
            if deriv:
                out += np.einsum('i...,i...->...', amplitude * (-omega).reshape(sh), np.cos(angle))
            else:
                out += np.einsum('i...,i...->...', amplitude, np.sin(angle))
        return out

    def orbital(self, x, y, z, t):
        """(x, y, z) 의 물 입자 속도 (u, v, w) [m/s]."""
        x, y, z = (np.asarray(v, float) for v in (x, y, z))
        depth = np.minimum(z - self.eta(x, y, t), 0.0)
        sh = (-1,) + (1,) * x.ndim
        velocity = np.zeros((3,) + x.shape)
        for gi, (_, amp, kx, ky, omega, phase, _) in enumerate(self.groups):
            k = np.hypot(kx, ky)
            theta = kx.reshape(sh) * x + ky.reshape(sh) * y - omega.reshape(sh) * t + phase.reshape(sh)
            amplitude = self._static(gi, x, y) * omega.reshape(sh) * np.exp(k.reshape(sh) * depth)
            horiz = amplitude * np.sin(theta)
            velocity[0] += np.sum(horiz * (kx / k).reshape(sh), axis=0)
            velocity[1] += np.sum(horiz * (ky / k).reshape(sh), axis=0)
            velocity[2] += np.sum(-amplitude * np.cos(theta), axis=0)
        return velocity

    def eta(self, x, y, t):
        """수면 높이 [m]. wave_height(x, y, t) 와 인자 순서가 같다."""
        return self._sum(x, t, y, False)

    def rise(self, x, y, t):
        """수면이 올라가는 속도 [m/s]."""
        return self._sum(x, t, y, True)
