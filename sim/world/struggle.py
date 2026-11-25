"""[시험용] 물에 빠진 사람의 발버둥: 선 자세로 허우적대기 (가영, 2026-11-20, 브랜치 dev/kayoung-weather).

사람은 누워 떠 있지 않고 물속에 선 자세(머리·어깨만 물 밖)로 발버둥 친다.
  자세: 시작할 때 몸을 세워 놓고 (몸 긴 축 = 위), 선헤엄으로 버티는 토크로 세운 채 유지한다. 얼굴은 facing 방향
        (배가 오는 쪽)을 본다. 구명환을 잡으면 LAY_SECONDS 동안 뒤로 눕고, 견인 내내 누운 자세를 유지한 채 끌려간다.
  물리: 선헤엄으로 머리를 물 위로 받치는 힘, 몸통에 위아래로 출렁이는 힘(머리가 물 위아래로 오르내림)과 좌우·앞뒤로 흔들리는 기울기를 더한다.
  화면: 팔은 수면에서 물을 휘저으며 내리치고, 다리는 물속에서 번갈아 찬다. 팔다리 고정 형상(누운 자세용)은
        그대로 두고(부력·접촉 유지) 화면에서만 숨긴 뒤 움직이는 팔다리를 그린다. 관절 팔(articulated_arms)은
        쓰지 않는다: buoyancy.body_voxels가 관절로 붙은 팔 몸체를 사람 부피에서 빼서 사람이 가라앉기 때문.
  잡는 손: 손 잡기 점(grip site)을 선 자세에서 손이 있는 곳(수면 근처, 몸 옆)으로 옮긴다. 배는 이 점을 겨눈다.
제자리: 발버둥 치는 동안 처음 자리로 되돌리는 약한 수평 힘(선헤엄으로 버티기)을 더해 떠밀려 가지 않는다.
공황 주기: PANIC_PERIOD마다 허우적대다 힘이 빠져 머리까지 잠기고(두 팔을 위로 뻗음) 헐떡이며 치고 올라온다. 좌우로 두리번거린다.
살려달라는 손짓: HELP_PERIOD마다 HELP_SECONDS 동안 오른팔을 머리 위로 높이 들어 좌우로 흔들고, 머리 위에 HELP! 글자를
  띄운다 (MuJoCo 화면 글꼴이 한글을 지원하지 않아 영어). 나머지 시간은 두 팔로 물을 휘젓는다.
구명환이 손 가까이 오면 잦아들고 (잡기 상대 속도 조건을 지키려고), 손으로 잡으면 멈춘다.

몸 좌표 (Human과 같음): +x 머리 쪽, +y 왼쪽, +z 얼굴 쪽. 선 자세에서는 +x가 위, +z가 앞이다.
"""

import math

import numpy as np

from sim import config


class Struggle:
    REQUIRES = {}
    FREQUENCY = 2.0         # 발버둥 빠르기 [Hz]. .9 → 1.5 → 2.0 (2026-11-23: 다급해 보이게, 팔 한 번 내리치는 주기 약 0.5초)
    LEG_FREQUENCY = 1.1     # 다리 페달 빠르기 [Hz]. 팔과 따로, 더 느리게 (다리는 무겁다, 시험용 가정값)
    # 2026-11-23 가영: 걸어 나오는 것처럼 보여서 더 허우적대게 — 더 낮게 잠기고, 불규칙하게 크게 출렁이고, 더 흔들린다.
    HEAVE_FORCE = 160.0     # 위아래 출렁임 힘 진폭 [N]. 110 → 160: 머리가 수면에 잠겼다 나왔다 한다 (시험용 가정값)
    SWAY = .22              # 좌우·앞뒤로 흔들리는 기울기 [rad]. .12 → .22 (시험용 가정값)
    POSE_STIFFNESS = 400.0  # 자세 유지 [N·m/rad] (선헤엄으로 버티기, 시험용 가정값)
    POSE_DAMPING = 120.0    # [N·m/(rad/s)]
    POSE_MAX = 250.0        # 자세 유지 토크 상한 [N·m]
    TREAD_HEAD_Z = .10      # 선헤엄으로 받치는 머리 중심 높이: 수면 위 [m]. .18 → .10: 얼굴이 겨우 물 밖, 어깨는 물속
    TREAD_STIFFNESS = 400.0 # 선헤엄 받침 [N/m] (시험용 가정값)
    TREAD_DAMPING = 120.0   # [N/(m/s)]
    TREAD_MAX = 160.0       # 선헤엄 받침 힘 상한 [N] (위로만)
    # 구명환을 잡은 뒤 뒤로 눕는 시간 [s] (선 채로 끌리면 물 저항이 커 견인이 느려진다). 2.5 → 1.5 (2026-11-25)
    # 2026-11-25 가영 '허우적 사람 견인이 느림': 예전엔 누운 뒤 2초에 걸쳐 자세 유지 토크를 풀었는데, 그러면 다리가 가라앉아
    #   몸이 다시 반쯤 섰다 (견인 중 세로에서 평균 57°, 가만히 있는 사람은 89°). 선 몸이 물을 정면으로 받아 견인이 29.1초
    #   (가만히 있는 사람 21.5초, calm). 견인 내내 누운 자세를 유지하고 눕는 시간을 줄이면 82° · 24.6초, 머리는 물 밖 그대로.
    LAY_SECONDS = 1.5
    START_DEPTH = .53       # 선 자세 시작 높이: 몸 원점(허리)이 수면 아래 이만큼 [m]. 머리 중심이 TREAD_HEAD_Z 근처
    HOLD_STIFFNESS = 40.0   # 제자리 유지 [N/m]: 처음 자리에서 떨어진 거리만큼 되돌리는 힘 (시험용 가정값)
    HOLD_DAMPING = 60.0     # 제자리 유지 감쇠 [N/(m/s)]
    HOLD_MAX = 25.0         # 제자리 유지 힘 상한 [N]: 사람이 헤엄쳐 낼 만한 정도. 파도·줄에 끌리는 것까지 막지는 않는다
    HELP_PERIOD = 4.0       # 살려달라는 손짓 주기 [s]. 6 → 4 (2026-11-23: 더 자주 구조 요청)
    HELP_SECONDS = 2.5      # 그중 손을 들고 흔드는 시간 [s]
    HELP_WAVE = 1.4         # 들어 올린 손을 흔드는 빠르기 [Hz]
    # 공황 주기 (2026-11-23, 긴박해 보이게): PANIC_PERIOD마다 허우적대다(FRANTIC) → 힘이 빠져 가라앉고(SINK, 머리가 물에 잠김)
    #   → 헐떡이며 세게 치고 올라온다(GASP). 시험용 가정값.
    PANIC_PERIOD = 7.0      # [s]
    SINK_START = 5.0        # 주기 안에서 가라앉기 시작하는 때 [s]
    SINK_SECONDS = 1.3      # 가라앉는 시간 [s]: 선헤엄 받침이 SINK_LIFT 배로 줄어 머리가 수면 아래로
    SINK_LIFT = .1          # 가라앉는 동안 남는 선헤엄 받침 비율
    SINK_PUSH = 60.0       # 가라앉는 동안 아래로 잠기는 힘 [N] (힘이 빠져 물에 빨려 들어가는 것을 흉내, 시험용 가정값)
    GASP_BOOST = 1.5        # 올라오며 헐떡일 때 동작 크기 배율
    # 구명환을 잡은 뒤 (2026-11-23 가영: 잡은 뒤 물에 잠기지 않게): 사람이 구명환에 매달려 머리를 물 밖에 유지하는 힘.
    #   머리 점에 위로만 건다 (뒤로 누워도 머리 쪽이 들림). 견인이 끝날 때까지 유지. 시험용 가정값.
    HELD_HEAD_Z = .14       # 잡은 뒤 유지할 머리 중심 높이: 수면 위 [m]
    HELD_STIFFNESS = 1200.0 # [N/m]
    HELD_DAMPING = 250.0    # [N/(m/s)]
    HELD_MAX = 450.0        # 위로 받치는 힘 상한 [N]
    LEAN = math.radians(20)  # 발버둥 중 몸을 앞으로 기울이는 각 (헤엄치려는 자세, 시험용 가정값)
    LOOK = .45              # 도움을 찾아 좌우로 두리번거리는 몸 돌림 [rad]
    CALM_DISTANCE = (.6, 1.8)   # 구명환-손 거리가 이 범위 [m] 안으로 들어오면 발버둥이 1 → 0으로 잦아든다
    # 팔 쉬는 자세 (선 자세 몸 좌표 각도 [rad]): 옆으로 뻗어 수면 근처에서 약간 앞으로. 손 잡기 점이 여기 손끝에 온다.
    ARM_REST = dict(lift=.05, swing=.35, bend=.35)
    LIMBS = ('arm_l', 'arm_r', 'forearm_l', 'forearm_r', 'hand_l', 'hand_r',
             'leg_l', 'leg_r', 'shin_l', 'shin_r', 'foot_l', 'foot_r')
    SKIN, SHIRT, TROUSERS, SHOES = (.72, .48, .32, 1), (.12, .39, .52, 1), (.10, .14, .20, 1), (.055, .065, .075, 1)

    def __init__(self, human, grasp, *, facing=0.0, seed=0):
        """facing: 선 자세에서 얼굴이 보는 수평 방향 [rad] (월드)."""
        if human.articulated_arms:
            raise ValueError("발버둥은 고정 팔 사람 모델에 쓴다 (관절 팔은 부력에서 빠짐).")
        self.human, self.grasp = human, grasp
        self.facing = float(facing)
        self.phase = np.random.default_rng(seed).uniform(0, 2 * math.pi)
        self.limb_geoms = {}
        self._rgba = None
        self._hidden = False
        self._home = None
        self._held_since = None

    # ── 조립 ──────────────────────────────────────────────
    def build(self, spec):
        """사람 시작 자세를 세우고, 손 잡기 점을 선 자세의 손끝으로 옮긴다 (Human 부품 다음에 조립)."""
        body = spec.body(config.Names.HUMAN_BODY)
        if body is None or self.human.length_scale is None:
            raise ValueError("Struggle은 Human 다음에 조립해야 합니다.")
        # 월드 회전 = 수평 회전(yaw) · y축 -90° (몸 +x → 위, 몸 +z(얼굴) → 수평). 얼굴이 facing을 보려면 yaw = facing + π.
        yaw = self.facing + math.pi
        q_yaw = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        q_up = np.array([math.cos(-math.pi / 4), 0, math.sin(-math.pi / 4), 0])
        body.quat[:] = self._quat_mul(q_yaw, q_up)
        body.pos[2] = config.WaterParams.SURFACE_Z - self.START_DEPTH
        s = self.human.length_scale
        for side, name in (('l', config.Names.HUMAN_GRIP_L_SITE), ('r', config.Names.HUMAN_GRIP_R_SITE)):
            site = spec.site(name)
            if site is not None:
                _, _, wrist, direction = self._arm_points(side, s, **self.ARM_REST)
                site.pos[:] = wrist + .04 * s * direction

    @staticmethod
    def _quat_mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    def bind(self, model):
        names = config.Names
        detail = names.HUMAN_DETAIL_PREFIX
        full = {'arm_l': names.HUMAN_ARM_L_GEOM, 'arm_r': names.HUMAN_ARM_R_GEOM,
                'leg_l': names.HUMAN_LEG_L_GEOM, 'leg_r': names.HUMAN_LEG_R_GEOM}
        self.limb_geoms = {limb: model.geom(full.get(limb, f"{detail}_{limb}")).id for limb in self.LIMBS}
        self._rgba = {gid: model.geom_rgba[gid].copy() for gid in self.limb_geoms.values()}

    # ── 물리 ──────────────────────────────────────────────
    def panic(self, t):
        """(동작 크기 배율, 가라앉는 정도 0~1). t = 위상이 더해진 시간 [s]."""
        x = t % self.PANIC_PERIOD
        end = self.SINK_START + self.SINK_SECONDS
        if x < self.SINK_START:
            return 1.0, 0.0
        if x < end:
            u = (x - self.SINK_START) / self.SINK_SECONDS
            sink = math.sin(math.pi * min(u * 1.6, 1.) / 2)          # 빠르게 힘이 빠지고 그대로 잠김
            return 1. - .6 * sink, sink
        u = (x - end) / (self.PANIC_PERIOD - end)                   # 헐떡이며 치고 올라옴 → 다시 허우적
        return 1. + (self.GASP_BOOST - 1.) * math.sin(math.pi * u), 0.0

    def intensity(self, data):
        """발버둥 세기 0~1. 잡았으면 0, 구명환이 손 가까이 오면 줄어든다."""
        if self.grasp.is_held(data):
            return 0.
        gap = float(np.linalg.norm(data.site_xpos[self.grasp.hand_site] - data.site_xpos[self.grasp.ring_site]))
        near, far = self.CALM_DISTANCE
        return float(np.clip((gap - near) / (far - near), 0., 1.))

    def pose_schedule(self, data):
        """(눕는 정도 0~1, 자세 유지 세기 0~1). 잡기 전 (0, 1). 잡은 뒤 LAY_SECONDS 동안 뒤로 눕고, 그 뒤로도 자세 유지
        토크를 그대로 두어 누운 채 끌려간다 (풀면 다리가 가라앉아 몸이 다시 서고 견인이 느려진다)."""
        if not self.grasp.is_held(data):
            self._held_since = None
            return 0., 1.
        if self._held_since is None:
            self._held_since = data.time
        after = data.time - self._held_since
        x = float(np.clip(after / self.LAY_SECONDS, 0., 1.))
        lay = x * x * (3 - 2 * x)
        return lay, 1.

    def target_rotation(self, lay, sway=(0., 0.), look=0.):
        """목표 자세 회전 행렬 (열 = 몸 x·y·z 축, 월드 좌표). lay=0 선 자세(얼굴 facing), lay=1 뒤로 누움(얼굴 위)."""
        # 발버둥 중엔 앞으로 LEAN만큼 기울인다 (꼿꼿이 서면 바닥을 딛고 선 것처럼 보임, 2026-11-23). 잡은 뒤엔 뒤로 눕는다.
        theta = lay * math.pi / 2 - (1 - lay) * self.LEAN
        front = np.array([math.cos(self.facing), math.sin(self.facing), 0.])
        up = np.array([0., 0., 1.])
        x = math.cos(theta) * up - math.sin(theta) * front     # 머리 쪽: 위 → 뒤로 넘어감
        z = math.cos(theta) * front + math.sin(theta) * up     # 얼굴: 앞 → 하늘
        target = np.column_stack([x, np.cross(z, x), z])
        a, b = sway
        tilt_x = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        tilt_y = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
        turn = np.array([[math.cos(look), -math.sin(look), 0], [math.sin(look), math.cos(look), 0], [0, 0, 1]])
        return turn @ tilt_x @ tilt_y @ target

    def add_forces(self, model, data, acc):
        import mujoco

        k = self.intensity(data)
        lay, u = self.pose_schedule(data)
        held = self.grasp.is_held(data)
        if held:
            self._hold_head_up(model, data, acc)
        if k <= 0 and u <= 0:
            return
        w = 2 * math.pi * self.FREQUENCY
        t = data.time + self.phase / w
        body = self.human.body_id
        rotation = data.xmat[body].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        spin, linear = velocity[:3], velocity[3:]
        force, torque = np.zeros(3), np.zeros(3)
        amp, sink = self.panic(t)
        sink *= k   # 구명환이 가까우면 가라앉지 않는다 (잡을 수 있게)
        if k > 0:
            # 박자가 다른 두 출렁임을 섞어 일정한 걸음 박자처럼 보이지 않게 한다 (합 진폭 최대 HEAVE_FORCE × 배율)
            force[2] = k * amp * self.HEAVE_FORCE * (.65 * math.sin(w * t) + .35 * math.sin(1.73 * w * t + 2.))
            # 제자리 유지: 처음 자리(발버둥 시작 위치)로 되돌리는 수평 스프링·감쇠, 크기 상한 HOLD_MAX.
            position = data.xpos[body, :2]
            if self._home is None:
                self._home = position.copy()
            hold = -self.HOLD_STIFFNESS * (position - self._home) - self.HOLD_DAMPING * linear[:2]
            size = np.linalg.norm(hold)
            if size > self.HOLD_MAX:
                hold *= self.HOLD_MAX / size
            force[:2] = k * hold
        if u > 0:
            # 선헤엄: 머리가 물 위에 있도록 위로 받치는 힘 (위로만, 상한 TREAD_MAX). 누우면 구명환이 받치므로 뺀다.
            head = data.xpos[body] + rotation @ (np.asarray(config.Human.HEAD_POS_B) * self.human.length_scale)
            lift = (self.TREAD_STIFFNESS * (config.WaterParams.SURFACE_Z + self.TREAD_HEAD_Z - head[2])
                    - self.TREAD_DAMPING * linear[2])
            force[2] += (1 - lay) * (1 - (1 - self.SINK_LIFT) * sink) * float(np.clip(lift, 0., self.TREAD_MAX))
            force[2] -= (1 - lay) * self.SINK_PUSH * sink
            # 자세 유지: 목표 자세(선 자세, 발버둥 중엔 좌우·앞뒤로 조금 흔들림 → 잡은 뒤 뒤로 누움)로 돌리는 토크.
            sway = (k * self.SWAY * (.7 * math.sin(.5 * w * t) + .3 * math.sin(1.3 * w * t + .4)),
                    k * self.SWAY * (.7 * math.sin(.37 * w * t + 1) + .3 * math.sin(1.1 * w * t + 2.2)))
            look = k * self.LOOK * (.7 * math.sin(.21 * w * t + .3) + .3 * math.sin(.53 * w * t + 1.7))
            error = self.target_rotation(lay, sway, look) @ rotation.T
            angle = math.acos(float(np.clip((np.trace(error) - 1) / 2, -1., 1.)))
            axis = np.array([error[2, 1] - error[1, 2], error[0, 2] - error[2, 0], error[1, 0] - error[0, 1]])
            size = np.linalg.norm(axis)
            if size > 1e-9:
                axis /= size
            elif angle > 1.:   # 정확히 반대(180°) 자세: 몸 y축으로 돌린다
                axis = rotation[:, 1]
            pose = self.POSE_STIFFNESS * angle * axis - self.POSE_DAMPING * spin
            size = np.linalg.norm(pose)
            if size > self.POSE_MAX:
                pose *= self.POSE_MAX / size
            torque = u * pose
        acc.add(body, force, torque_w=torque)

    def _hold_head_up(self, model, data, acc):
        """구명환을 잡은 뒤: 머리 중심이 HELD_HEAD_Z 아래로 내려가면 머리 점을 위로 받친다 (위로만, 상한 HELD_MAX)."""
        import mujoco

        body = self.human.body_id
        rotation = data.xmat[body].reshape(3, 3)
        head = data.xpos[body] + rotation @ (np.asarray(config.Human.HEAD_POS_B) * self.human.length_scale)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        spin, linear = velocity[:3], velocity[3:]
        head_vz = linear[2] + np.cross(spin, head - data.xipos[body])[2]
        up = (self.HELD_STIFFNESS * (config.WaterParams.SURFACE_Z + self.HELD_HEAD_Z - head[2])
              - self.HELD_DAMPING * head_vz)
        up = float(np.clip(up, 0., self.HELD_MAX))
        if up > 0:
            acc.add(body, np.array([0., 0., up]), point_w=head)

    def help_weight(self, t):
        """손 드는 정도 0~1. 주기마다 HELP_SECONDS 동안 1, 앞뒤 0.4초에 걸쳐 부드럽게 오르내린다."""
        x = (t % self.HELP_PERIOD)
        ramp = .4
        up = np.clip(x / ramp, 0, 1) * np.clip((self.HELP_SECONDS - x) / ramp, 0, 1)
        return float(up * up * (3 - 2 * up))

    # ── 화면 ──────────────────────────────────────────────
    @staticmethod
    def _direction(sign, lift, swing):
        """옆(±y)으로 뻗은 방향을 위(+x)로 lift, 앞(+z, 얼굴 쪽)으로 swing만큼 돌린 단위 방향 (몸 좌표)."""
        return np.array([math.sin(lift), sign * math.cos(lift) * math.cos(swing), math.cos(lift) * math.sin(swing)])

    @classmethod
    def _arm_points(cls, side, s, lift, swing, bend):
        """선 자세 팔 (어깨, 팔꿈치, 손목, 아래팔 방향) [몸 좌표]. bend: 팔꿈치에서 앞으로 더 굽히는 각."""
        cfg = config.Human
        sign = 1 if side == 'l' else -1
        shoulder = np.asarray(getattr(cfg, f"ARM_{side.upper()}_FROMTO_B")[0]) * s
        elbow0 = np.asarray(getattr(cfg, f"ELBOW_{side.upper()}_B")) * s
        wrist0 = np.asarray(getattr(cfg, f"ARM_{side.upper()}_FROMTO_B")[1]) * s
        upper, fore = np.linalg.norm(elbow0 - shoulder), np.linalg.norm(wrist0 - elbow0)
        elbow = shoulder + upper * cls._direction(sign, lift, swing)
        d_fore = cls._direction(sign, lift - .3 * bend, swing + bend)
        return shoulder, elbow, elbow + fore * d_fore, d_fore

    def limb_segments(self, data, k):
        """(시작점, 끝점, 반지름, 색) 목록 [월드 좌표]. k = 발버둥 세기 (0이면 쉬는 자세)."""
        cfg = config.Human
        s, g = self.human.length_scale, self.human.girth_scale
        rotation, origin = data.xmat[self.human.body_id].reshape(3, 3), data.xpos[self.human.body_id]
        w = 2 * math.pi * self.FREQUENCY
        t = data.time + self.phase / w
        rest = self.ARM_REST
        amp, sink = self.panic(t)
        sink *= k
        calm = k            # 손짓(HELP)은 원래 세기로
        k = k * amp         # 공황 주기: 허우적 → 가라앉으며 작아짐 → 헐떡이며 크게

        def world(point):
            return origin + rotation @ point

        segments = []
        for side, sign, offset in (('l', 1, 0.), ('r', -1, math.pi)):
            # 팔: 위로 높이 들었다가 수면을 세게 내리친다(lift). 박자가 다른 사인을 섞어 불규칙하게, 두 팔은 어긋나게.
            #   (2026-11-23: 앞뒤로 일정하게 휘저으면 걷는 팔처럼 보여서 위아래 내리치기 위주로 바꿈)
            phase = w * t + offset
            slap = .6 * math.sin(phase) + .4 * math.sin(1.9 * phase + (1.1 if side == 'l' else 2.6))
            lift = rest['lift'] + k * (.45 + .75 * slap)
            swing = rest['swing'] + k * (.25 * math.cos(.7 * phase + .5) + .15)
            bend = rest['bend'] + k * .5 * (1 + math.sin(1.3 * phase + 1.2)) / 2
            if side == 'r':
                # 살려달라는 손짓: 오른팔을 머리 위로 곧게 들고 좌우로 흔든다.
                h = calm * self.help_weight(t) * (1 - sink)   # 가라앉는 동안은 손 흔들기 대신 위로 뻗기
                wave = .35 * math.sin(2 * math.pi * self.HELP_WAVE * t)
                lift = (1 - h) * lift + h * (1.3 + wave)
                swing = (1 - h) * swing + h * .1
                bend = (1 - h) * bend + h * .05
            # 가라앉을 때: 두 팔을 머리 위로 뻗어 허공을 잡으려 한다 (물에 잠기는 사람의 몸짓)
            reach = 1.15 + .15 * math.sin(3.1 * phase)
            lift = (1 - sink) * lift + sink * reach
            swing = (1 - sink) * swing + sink * .25
            bend = (1 - sink) * bend + sink * .3
            shoulder, elbow, wrist, d_fore = self._arm_points(side, s, lift, swing, bend)
            segments += [(world(shoulder), world(elbow), cfg.ARM_RADIUS * g, self.SHIRT),
                         (world(elbow), world(wrist), cfg.ARM_RADIUS * .78 * g, self.SKIN),
                         (world(wrist), world(wrist + .06 * s * d_fore), .045 * s, self.SKIN)]
            # 다리: 두 다리를 반 박자 어긋나게 자전거 페달 밟듯 부드럽게 돌린다 (물에서 버티는 발차기).
            #   2026-11-23: 팔 박자에 묶여 너무 빠르고(초당 2.8번) 들쭉날쭉해 부자연스러웠음 → 다리는 팔과 따로
            #   LEG_FREQUENCY로 느리게, 사인 하나로 매끄럽게. 허벅지가 앞으로 올라올 때 무릎이 더 굽는다.
            hip, ankle0 = (np.asarray(p) * s for p in getattr(cfg, f"LEG_{side.upper()}_FROMTO_B"))
            knee0 = np.asarray(getattr(cfg, f"KNEE_{side.upper()}_B")) * s
            thigh, shin = np.linalg.norm(knee0 - hip), np.linalg.norm(ankle0 - knee0)
            pedal = 2 * math.pi * self.LEG_FREQUENCY * data.time + self.phase + (0. if side == 'l' else math.pi)
            leg = min(k, 1.2)
            # 2026-11-23: 다리를 아래로 쭉 뻗으면 바닥에 선 것처럼 보여서, 무릎을 크게 접어 발이 엉덩이 뒤로 올라오게 한다.
            kick = -.05 + leg * .3 * (1 + math.sin(pedal)) / 2             # 허벅지를 앞으로 든 각 [rad] (음수면 뒤로)
            knee_bend = 1.05 + leg * .45 * (1 + math.sin(pedal + .9)) / 2  # 무릎 굽힘 [rad]: 발이 뒤로 올라옴
            spread = .1                                                     # 옆으로 살짝 벌림
            d_thigh = np.array([-math.cos(kick), sign * math.sin(spread), math.sin(kick)])
            d_thigh /= np.linalg.norm(d_thigh)
            d_shin = np.array([-math.cos(kick - knee_bend), sign * .5 * math.sin(spread), math.sin(kick - knee_bend)])
            d_shin /= np.linalg.norm(d_shin)
            knee = hip + thigh * d_thigh
            ankle = knee + shin * d_shin
            # 발: 정강이에 수직인 앞쪽(+z) 방향과 정강이 방향을 섞어 발끝이 살짝 펴지게 (정강이를 따라 함께 돈다)
            forward = np.array([0., 0., 1.]) - d_shin[2] * d_shin
            forward /= np.linalg.norm(forward)
            d_foot = .75 * forward + .66 * d_shin
            d_foot /= np.linalg.norm(d_foot)
            segments += [(world(hip), world(knee), cfg.LEG_RADIUS * g, self.TROUSERS),
                         (world(knee), world(ankle), cfg.LEG_RADIUS * .7 * g, self.TROUSERS),
                         (world(ankle), world(ankle + .13 * s * d_foot), .05 * s, self.SHOES)]
        return segments

    SPLASH_DROPS = 18       # 손 하나당 물보라 방울 수 (화면 표시만). 10 → 18

    def splash_drops(self, data, k):
        """손이 수면 근처에 있을 때 튀는 물보라 방울 [(중심, 반지름)] (월드). 화면 표시만, 물리 없음."""
        if k <= 0:
            return []
        surface = config.WaterParams.SURFACE_Z
        drops = []
        segments = self.limb_segments(data, k)
        # limb_segments 순서: 한쪽마다 [위팔, 아래팔, 손, 허벅지, 정강이, 발] → 손은 2번, 8번
        for side, index in ((0, 2), (1, 8)):
            hand = segments[index][1]
            near = 1 - abs(hand[2] - surface) / .15   # 수면 ±15 cm 안에서만, 가까울수록 크게
            if near <= 0:
                continue
            seed = int(data.time * 12) * 2 + side   # 1/12초마다 방울 모양이 바뀐다
            rng = np.random.default_rng(seed)
            for _ in range(self.SPLASH_DROPS):
                offset = rng.uniform([-.35, -.35, 0.], [.35, .35, .6]) * near
                drops.append((np.array([hand[0], hand[1], surface]) + offset, .018 + .02 * rng.random() * near))
        return drops

    def draw(self, scene, model, data):
        """고정 팔다리(누운 자세용)를 숨기고 선 자세로 움직이는 팔다리를 그린다. 잡은 뒤에는 쉬는 자세로 그린다."""
        import mujoco

        k = self.intensity(data)
        if not self._hidden:
            for gid, rgba in self._rgba.items():
                model.geom_rgba[gid] = (*rgba[:3], 0.)
            self._hidden = True
        for start, end, radius, color in self.limb_segments(data, k):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.asarray(color, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
            scene.ngeom += 1
        for center, radius in self.splash_drops(data, k):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius), center, np.eye(3).ravel(),
                                np.array([.92, .96, 1., .8], dtype=np.float32))
            scene.ngeom += 1
        w = 2 * math.pi * self.FREQUENCY
        t = data.time + self.phase / w
        sinking = k * self.panic(t)[1] > .3
        if (k * self.help_weight(t) > .5 or sinking) and scene.ngeom < scene.maxgeom:
            # 손 흔드는 동안·가라앉는 동안 머리 위에 HELP! 글자. 빨강·노랑으로 초당 3번 깜빡인다 (2026-11-23: 긴박하게)
            blink = (1, .15, .1, 1) if int(data.time * 6) % 2 == 0 else (1, .9, .1, 1)
            head = data.xpos[self.human.body_id] + data.xmat[self.human.body_id].reshape(3, 3) @ (
                np.asarray(config.Human.HEAD_POS_B) * self.human.length_scale)
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.full(3, .1), head + [0, 0, .6],
                                np.eye(3).ravel(), np.array(blink, dtype=np.float32))
            geom.label = 'HELP!!' if sinking else 'HELP!'
            scene.ngeom += 1
