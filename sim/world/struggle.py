"""[시험용] 물에 빠진 사람의 발버둥 (가영, 2026-11-20, 브랜치 try/weather).

두 부분으로 나눈다.
  물리: 몸통에 위아래로 출렁이는 힘과 좌우로 뒤척이는 토크를 더한다 (ForceAccumulator). 머리가 오르내리고 몸이 흔들린다.
  화면: 팔은 물을 내리치듯 휘젓고 다리는 번갈아 찬다. 팔다리 고정 형상은 그대로 두고(부력·접촉 유지) 화면에서만
        숨긴 뒤 움직이는 팔다리를 그린다. 관절 팔(articulated_arms)은 쓰지 않는다: buoyancy.body_voxels가 관절로 붙은
        팔 몸체를 사람 부피에서 빼서 사람이 가라앉기 때문.
제자리: 발버둥 치는 동안 처음 자리로 되돌리는 약한 수평 힘(선헤엄으로 버티기)을 더해 떠밀려 가지 않는다.
살려달라는 손짓: HELP_PERIOD마다 HELP_SECONDS 동안 오른팔을 머리 위로 높이 들어 좌우로 흔들고, 머리 위에 HELP! 글자를
  띄운다 (MuJoCo 화면 글꼴이 한글을 지원하지 않아 영어). 나머지 시간은 두 팔로 물을 휘젓는다.
구명환이 손 가까이 오면 잦아들고 (잡기 상대 속도 조건을 지키려고), 손으로 잡으면 멈춘다.
"""

import math

import numpy as np

from sim import config


class Struggle:
    REQUIRES = {}
    FREQUENCY = .9          # 발버둥 빠르기 [Hz] (팔 한 번 휘젓는 주기 약 1.1초)
    HEAVE_FORCE = 110.0     # 위아래 출렁임 힘 진폭 [N]. 70 kg 사람 머리가 약 5~8 cm 오르내린다 (시험용 가정값)
    ROLL_TORQUE = 12.0      # 좌우 뒤척임 토크 진폭 [N·m] (시험용 가정값)
    HOLD_STIFFNESS = 40.0   # 제자리 유지 [N/m]: 처음 자리에서 떨어진 거리만큼 되돌리는 힘 (시험용 가정값)
    HOLD_DAMPING = 60.0     # 제자리 유지 감쇠 [N/(m/s)]
    HOLD_MAX = 25.0         # 제자리 유지 힘 상한 [N]: 사람이 헤엄쳐 낼 만한 정도. 파도·줄에 끌리는 것까지 막지는 않는다
    HELP_PERIOD = 6.0       # 살려달라는 손짓 주기 [s]
    HELP_SECONDS = 2.5      # 그중 손을 들고 흔드는 시간 [s]
    HELP_WAVE = 1.4         # 들어 올린 손을 흔드는 빠르기 [Hz]
    CALM_DISTANCE = (.6, 1.8)   # 구명환-손 거리가 이 범위 [m] 안으로 들어오면 발버둥이 1 → 0으로 잦아든다
    LIMBS = ('arm_l', 'arm_r', 'forearm_l', 'forearm_r', 'hand_l', 'hand_r',
             'leg_l', 'leg_r', 'shin_l', 'shin_r', 'foot_l', 'foot_r')
    SKIN, SHIRT, TROUSERS, SHOES = (.72, .48, .32, 1), (.12, .39, .52, 1), (.10, .14, .20, 1), (.055, .065, .075, 1)

    def __init__(self, human, grasp, *, seed=0):
        if human.articulated_arms:
            raise ValueError("발버둥은 고정 팔 사람 모델에 쓴다 (관절 팔은 부력에서 빠짐).")
        self.human, self.grasp = human, grasp
        self.phase = np.random.default_rng(seed).uniform(0, 2 * math.pi)
        self.limb_geoms = {}
        self._rgba = None
        self._hidden = False
        self._home = None

    def build(self, spec):
        """새 형상이 없다."""

    def bind(self, model):
        names = config.Names
        detail = names.HUMAN_DETAIL_PREFIX
        full = {'arm_l': names.HUMAN_ARM_L_GEOM, 'arm_r': names.HUMAN_ARM_R_GEOM,
                'leg_l': names.HUMAN_LEG_L_GEOM, 'leg_r': names.HUMAN_LEG_R_GEOM}
        self.limb_geoms = {limb: model.geom(full.get(limb, f"{detail}_{limb}")).id for limb in self.LIMBS}
        self._rgba = {gid: model.geom_rgba[gid].copy() for gid in self.limb_geoms.values()}

    def intensity(self, data):
        """발버둥 세기 0~1. 잡았으면 0, 구명환이 손 가까이 오면 줄어든다."""
        if self.grasp.is_held(data):
            return 0.
        gap = float(np.linalg.norm(data.site_xpos[self.grasp.hand_site] - data.site_xpos[self.grasp.ring_site]))
        near, far = self.CALM_DISTANCE
        return float(np.clip((gap - near) / (far - near), 0., 1.))

    def add_forces(self, model, data, acc):
        k = self.intensity(data)
        if k <= 0:
            return
        w = 2 * math.pi * self.FREQUENCY
        t = data.time + self.phase / w
        body = self.human.body_id
        axis = data.xmat[body].reshape(3, 3)[:, 0]   # 몸 긴 방향(머리 쪽)
        heave = k * self.HEAVE_FORCE * math.sin(w * t)
        roll = k * self.ROLL_TORQUE * math.sin(.5 * w * t)
        # 제자리 유지: 처음 자리(발버둥 시작 위치)로 되돌리는 수평 스프링·감쇠, 크기 상한 HOLD_MAX.
        position = data.xpos[body, :2]
        if self._home is None:
            self._home = position.copy()
        velocity = np.zeros(6)
        import mujoco
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        hold = -self.HOLD_STIFFNESS * (position - self._home) - self.HOLD_DAMPING * velocity[3:5]
        size = np.linalg.norm(hold)
        if size > self.HOLD_MAX:
            hold *= self.HOLD_MAX / size
        acc.add(body, [k * hold[0], k * hold[1], heave], torque_w=roll * axis)

    def help_weight(self, t):
        """손 드는 정도 0~1. 주기마다 HELP_SECONDS 동안 1, 앞뒤 0.4초에 걸쳐 부드럽게 오르내린다."""
        x = (t % self.HELP_PERIOD)
        ramp = .4
        up = np.clip(x / ramp, 0, 1) * np.clip((self.HELP_SECONDS - x) / ramp, 0, 1)
        return float(up * up * (3 - 2 * up))

    # ── 화면 ──────────────────────────────────────────────
    def limb_segments(self, data, k):
        """(시작점, 끝점, 반지름, 색) 목록 [월드 좌표]. k = 발버둥 세기."""
        cfg = config.Human
        s, g = self.human.length_scale, self.human.girth_scale
        rotation, origin = data.xmat[self.human.body_id].reshape(3, 3), data.xpos[self.human.body_id]
        w = 2 * math.pi * self.FREQUENCY
        t = data.time + self.phase / w

        def world(point):
            return origin + rotation @ point

        segments = []
        for side, sign, offset in (('l', 1, 0.), ('r', -1, math.pi)):
            # 팔: 옆으로 뻗은 자세에서 위(물 밖)로 들어 올렸다가 머리 쪽으로 내리친다. 두 팔은 반 박자 어긋난다.
            shoulder = np.asarray(getattr(cfg, f"ARM_{side.upper()}_FROMTO_B")[0]) * s
            elbow0 = np.asarray(getattr(cfg, f"ELBOW_{side.upper()}_B")) * s
            wrist0 = np.asarray(getattr(cfg, f"ARM_{side.upper()}_FROMTO_B")[1]) * s
            upper, fore = np.linalg.norm(elbow0 - shoulder), np.linalg.norm(wrist0 - elbow0)
            phase = w * t + offset
            lift = k * (.55 + .65 * math.sin(phase))          # 위로 드는 각 [rad]
            sweep = k * .7 * math.cos(phase)                  # 머리 쪽으로 쓸어내리는 각 [rad]
            bend = .35 + k * .5 * (1 + math.sin(phase + 1.2)) / 2
            rest = (elbow0 - shoulder) / upper
            if side == 'r':
                # 살려달라는 손짓: 오른팔을 거의 수직으로 들고 (팔꿈치 거의 폄) 좌우로 흔든다.
                h = k * self.help_weight(t)
                wave = .45 * math.sin(2 * math.pi * self.HELP_WAVE * t)
                lift = (1 - h) * lift + h * 1.35
                sweep = (1 - h) * sweep + h * (.25 + wave)
                bend = (1 - h) * bend + h * .1
            d_upper = self._turn(rest, sign, lift, sweep)
            d_fore = self._turn(rest, sign, lift + bend, sweep + .3 * k)
            elbow = shoulder + upper * d_upper
            wrist = elbow + fore * d_fore
            segments += [(world(shoulder), world(elbow), cfg.ARM_RADIUS * g, self.SHIRT),
                         (world(elbow), world(wrist), cfg.ARM_RADIUS * .78 * g, self.SKIN),
                         (world(wrist), world(wrist + .06 * s * d_fore), .045 * s, self.SKIN)]
            # 다리: 번갈아 위아래로 찬다 (자유형 발차기처럼).
            hip, ankle0 = (np.asarray(p) * s for p in getattr(cfg, f"LEG_{side.upper()}_FROMTO_B"))
            knee0 = np.asarray(getattr(cfg, f"KNEE_{side.upper()}_B")) * s
            thigh, shin = np.linalg.norm(knee0 - hip), np.linalg.norm(ankle0 - knee0)
            kick = k * .45 * math.sin(1.6 * w * t + offset)
            knee_bend = k * .35 * (1 + math.sin(1.6 * w * t + offset + .8)) / 2
            d_thigh = np.array([-math.cos(kick), sign * .05, math.sin(kick)])
            d_shin = np.array([-math.cos(kick - knee_bend), sign * .03, math.sin(kick - knee_bend)])
            knee = hip + thigh * d_thigh / np.linalg.norm(d_thigh)
            ankle = knee + shin * d_shin / np.linalg.norm(d_shin)
            segments += [(world(hip), world(knee), cfg.LEG_RADIUS * g, self.TROUSERS),
                         (world(knee), world(ankle), cfg.LEG_RADIUS * .7 * g, self.TROUSERS),
                         (world(ankle), world(ankle + np.array([-.1, 0, .05]) * s), .05 * s, self.SHOES)]
        return segments

    @staticmethod
    def _turn(rest, sign, lift, sweep):
        """옆으로 뻗은 방향 rest를 위(+z)로 lift, 머리 쪽(+x)으로 sweep만큼 돌린 단위 방향 (몸 좌표)."""
        lateral = np.array([0., sign, 0.])
        base = rest - lateral * (rest @ lateral)        # rest의 머리 쪽 성분
        side = math.cos(lift) * math.cos(sweep)
        direction = side * lateral + math.sin(sweep) * np.array([1., 0., 0.]) + math.sin(lift) * np.array([0., 0., 1.])
        direction = direction + .3 * base
        return direction / np.linalg.norm(direction)

    def draw(self, scene, model, data):
        """발버둥 중이면 고정 팔다리를 숨기고 움직이는 팔다리를 그린다. 멈추면 고정 팔다리를 다시 보인다."""
        import mujoco

        k = self.intensity(data)
        hide = k > 0
        if hide != self._hidden:
            for gid, rgba in self._rgba.items():
                model.geom_rgba[gid] = rgba if not hide else (*rgba[:3], 0.)
            self._hidden = hide
        if not hide:
            return
        for start, end, radius, color in self.limb_segments(data, k):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.asarray(color, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
            scene.ngeom += 1
        w = 2 * math.pi * self.FREQUENCY
        if k * self.help_weight(data.time + self.phase / w) > .5 and scene.ngeom < scene.maxgeom:
            # 손 흔드는 동안 머리 위에 HELP! 글자
            head = data.xpos[self.human.body_id] + data.xmat[self.human.body_id].reshape(3, 3) @ (
                np.asarray(config.Human.HEAD_POS_B) * self.human.length_scale)
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.full(3, .1), head + [0, 0, .9],
                                np.eye(3).ravel(), np.array([1, .2, .1, 1], dtype=np.float32))
            geom.label = 'HELP!'
            scene.ngeom += 1
