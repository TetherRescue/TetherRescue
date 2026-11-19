"""2 로봇: 구명환 발사대 (담당: 동경).

발사 전: 구명환을 배 홈 위 발사대(config.Launcher.MOUNT_B)에 weld 로 고정한다.
발사: weld 를 풀고 PUSH_SECONDS 동안 일정한 힘으로 민다. 같은 크기의 반동은 배가 받는다.
  힘은 ForceAccumulator 로만 넣는다 (계약). 위치·속도를 직접 바꾸지 않는다.
겨누기: 공기 저항 없는 포물선. 잡는 점(Lifebuoy.GRIP_L)이 목표(사람 손)에 떨어지도록 환 중심을 겨눈다.

build 순서: Robot → (Human) → Lifebuoy(attached=False) → Launcher. 발사대가 환의 시작 위치를 배 기준으로 바꾼다.
"""
import math

import numpy as np

from sim import config
from sim.config import Launcher as Cfg, Names

LOADED, PUSHING, FLYING = 'LOADED', 'PUSHING', 'FLYING'


def ballistic_speed(horizontal, rise, angle):
    """앙각 angle [rad] 로 쏴서 수평 horizontal [m] 만큼 가 rise [m] 높이(+ 위)에 닿는 속력 [m/s].

    v² = g·R² / (2·cos²θ·(R·tanθ − Δh)). 그 높이에 닿을 수 없으면 ValueError.
    """
    g = config.Physics.GRAVITY
    reach = horizontal * math.tan(angle) - rise
    if horizontal <= 0 or reach <= 0:
        raise ValueError('이 앙각으로는 목표에 닿을 수 없습니다.')
    return math.sqrt(g * horizontal ** 2 / (2 * math.cos(angle) ** 2 * reach))


class Launcher:
    REQUIRES = {}

    def __init__(self, *, ring_site=None):
        """ring_site: 목표(손)에 맞출 환의 점. 기본 = 잡는 점 Names.LIFEBUOY_GRIP_L_SITE."""
        self.ring_site_name = ring_site or Names.LIFEBUOY_GRIP_L_SITE
        self.state = LOADED
        self.eq_id = None
        self._force = None
        self._push_end = None
        self.shot = None   # 마지막 발사 기록 (시각, 거리, 속력, 겨눈 점)

    # ── 조립 ─────────────────────────────────────────────────
    def build(self, spec):
        import mujoco

        boat, ring = spec.body(Names.BOAT_BODY), spec.body(Names.LIFEBUOY_BODY)
        if boat is None or ring is None:
            raise ValueError('Launcher.build 전에 Robot·Lifebuoy build 가 필요합니다.')
        if spec.equality(Names.HUMAN_RING_WORN_EQ) is not None:
            raise ValueError('발사대에는 착용하지 않은 구명환(Lifebuoy(attached=False))이 필요합니다.')
        # 환 시작 위치 = 배 위치·자세 기준 발사대 자리
        boat_quat = np.asarray(boat.quat, dtype=float)
        offset = np.empty(3)
        mujoco.mju_rotVecQuat(offset, np.asarray(Cfg.MOUNT_B, dtype=float), boat_quat)
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, boat_quat, np.asarray(Cfg.MOUNT_QUAT_B, dtype=float))
        ring.pos = np.asarray(boat.pos, dtype=float) + offset
        ring.quat = quat
        # 배 ↔ 환 weld. relpose = 배 기준 환 위치·자세 (Lifebuoy 착용 weld 와 같은 data 배치)
        eq = spec.add_equality(name=Names.LAUNCHER_MOUNT_EQ, type=mujoco.mjtEq.mjEQ_WELD,
                               objtype=mujoco.mjtObj.mjOBJ_BODY, name1=Names.BOAT_BODY,
                               name2=Names.LIFEBUOY_BODY, active=True)
        eq.data[:] = 0
        eq.data[3:6] = Cfg.MOUNT_B
        eq.data[6:10] = Cfg.MOUNT_QUAT_B
        eq.data[10] = 1
        eq.solref[:] = [.02, 1]

    def bind(self, model):
        try:
            self.eq_id = model.equality(Names.LAUNCHER_MOUNT_EQ).id
            self.boat_body = model.body(Names.BOAT_BODY).id
            self.ring_body = model.body(Names.LIFEBUOY_BODY).id
            self.ring_site = model.site(self.ring_site_name).id
        except KeyError as error:
            raise ValueError('Launcher.bind 전에 배·구명환·발사대 build·compile 이 필요합니다.') from error
        if model.site_bodyid[self.ring_site] != self.ring_body:
            raise ValueError('겨눌 점은 구명환에 있어야 합니다.')
        self.ring_mass = float(model.body_subtreemass[self.ring_body])
        self._model = model
        self.state = LOADED

    def reset(self, model, data):
        """발사 전 상태로. mj_resetData 는 weld 를 eq_active0(켜짐)으로 되돌린다."""
        if model is not self._model:
            raise ValueError('이 발사대에 bind 된 모델이 필요합니다.')
        data.eq_active[self.eq_id] = True
        self.state = LOADED
        self._force = self._push_end = None
        self.shot = None

    @property
    def loaded(self):
        return self.state == LOADED

    # ── 겨누기·발사 ───────────────────────────────────────────
    def aim(self, data, target_w, angle=None):
        """target_w 에 잡는 점이 떨어지게 하는 환 중심 발사 속도 (월드, 배에 대한 속도 아님). 호출 전 mj_forward.

        target_w 높이는 무시하고 수면 + AIM_HEIGHT 에서 환 중심이 닿는다고 본다 (사람 손은 수면 근처).
        미는 동안 환이 움직이는 거리(속도 × PUSH_SECONDS / 2)만큼 출발점을 앞당겨 한 번 고친다.
        """
        angle = math.radians(Cfg.ANGLE_DEG) if angle is None else angle
        center = data.xpos[self.ring_body]
        grip_offset = data.site_xpos[self.ring_site] - center
        aim = np.asarray(target_w, dtype=float) - grip_offset
        aim[2] = config.WaterParams.SURFACE_Z + Cfg.AIM_HEIGHT
        start = center.copy()
        for _ in range(2):
            flat = aim[:2] - start[:2]
            distance = float(np.linalg.norm(flat))
            speed = ballistic_speed(distance, aim[2] - start[2], angle)
            heading = math.atan2(flat[1], flat[0])
            velocity = speed * np.array([math.cos(angle) * math.cos(heading),
                                         math.cos(angle) * math.sin(heading), math.sin(angle)])
            start = center + velocity * Cfg.PUSH_SECONDS / 2
        return velocity, distance, aim

    def fire(self, model, data, velocity_w):
        """weld 를 풀고 다음 PUSH_SECONDS 동안 환을 velocity_w 가 되도록 민다 (반동은 배)."""
        if model is not self._model:
            raise ValueError('이 발사대에 bind 된 모델이 필요합니다.')
        if not self.loaded:
            raise RuntimeError('이미 발사했습니다.')
        velocity_w = np.asarray(velocity_w, dtype=float)
        if velocity_w.shape != (3,) or not np.isfinite(velocity_w).all():
            raise ValueError('발사 속도는 유한한 3차원 벡터여야 합니다.')
        from sim.frames import body_velocity_w
        current, _ = body_velocity_w(model, data, self.ring_body)
        gravity = np.array([0., 0., self.ring_mass * float(np.linalg.norm(model.opt.gravity))])
        # 일정한 힘 F 로 T 초 밀면 Δv = (F − 무게) · T / m → F = m · Δv / T + 무게
        self._force = self.ring_mass * (velocity_w - current) / Cfg.PUSH_SECONDS + gravity
        self._push_end = float(data.time) + Cfg.PUSH_SECONDS
        data.eq_active[self.eq_id] = False
        self.state = PUSHING

    def add_forces(self, model, data, acc):
        if self.state != PUSHING:
            return
        if data.time >= self._push_end - 1e-12:
            self.state = FLYING
            return
        mount_w = data.xpos[self.boat_body] + data.xmat[self.boat_body].reshape(3, 3) @ np.asarray(Cfg.MOUNT_B)
        acc.add(self.ring_body, self._force.copy())
        acc.add(self.boat_body, -self._force, point_w=mount_w)   # 반동
