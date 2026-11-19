"""2 로봇: 선체·추진기·조향·전복 판정·구명환 윈치 (담당: 동경)."""
import mujoco
import numpy as np

from sim import config, frames
from sim.config import Hull, Names, Sensors, Thruster, Winch

SITE_SIZE = (0.03,)   # 점(site) 표시 크기 [m]. 그림용이라 물리에 영향 없음
HULL_RGBA = (0.9, 0.4, 0.1, 1.0)
# A형 선체 충돌 형상 이름 (왼쪽 다리, 오른쪽 다리, 연결부, 뱃머리). config.Hull.SHAPE == "a_frame" 일 때 생김
A_FRAME_GEOMS = ("hull_leg_l", "hull_leg_r", "hull_bridge", "hull_bow")

# 윈치 상태: 감음(짧게 끌고 감) → 푸는 중 → 다 풀림(= 건네줌). delivery_ready 는 PAID_OUT 일 때만 True
STOWED, PAYING_OUT, PAID_OUT = 'STOWED', 'PAYING_OUT', 'PAID_OUT'


class Robot:
    REQUIRES = {}  # 예: {"max_dt": 0.004}

    def __init__(self, surface_height=None, *, position=None, yaw=0.0, tow_point_b=None, winch=False):
        """surface_height: 수면 높이 함수 (x, y, t) -> z, 예: Water().wave_height. x, y 는 배열로 받음.

        None 이면 1주차 가정 "노즐은 항상 물속" (WaterAPI.USE_IN_WEEK1 = False).
        주면 노즐이 수면 위로 나온 스텝에는 그 노즐 추력을 0 으로 한다 (Thruster.FULL_THRUST_DEPTH 주석).
        winch: True 면 선미 로프(Names.TETHER_TENDON)의 길이를 배가 조절한다.
          처음엔 Winch.STOW_LENGTH 로 짧게 감아 구명환을 끌고 가고, pay_out() 으로 PAYOUT_LENGTH 까지 풀어 건네준다.
          세계에 Tether(enabled=True, attached=True) 가 있어야 한다 (로프 자체·장력 계산은 가영 tether.py).
        """
        self.winch = bool(winch)
        self.delivery_state = STOWED if self.winch else None
        self._model = None
        self.throttle = 0.0   # 전진 명령 (−1 최대 후진 ~ +1 최대 전진)
        self.steer = 0.0      # 조향 명령 (+ 왼쪽으로 돎)
        self.surface_height = surface_height
        self.position = None if position is None else self._vector(position, 'position')
        if not np.isfinite(yaw):
            raise ValueError('yaw는 유한한 각도여야 합니다.')
        self.yaw = float(yaw)   # 처음 방향 [rad]. build 때 몸체 자세로 넣음 (발사대처럼 배 기준으로 놓는 부품용)
        self.tow_point_b = None if tow_point_b is None else self._vector(tow_point_b, 'tow_point_b')

    @staticmethod
    def _vector(value, name):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f'{name}는 유한한 좌표 3개여야 합니다.')
        return value.copy()

    @property
    def _commands(self):
        # 기존 접근 제어·감시 시험에서 읽는 정규화 좌우 명령 호환값.
        return np.clip([self.throttle - self.steer, self.throttle + self.steer], *Thruster.CMD_RANGE)

    def set_command(self, throttle=0.0, steer=0.0, *, forward=None, steering=None):
        """추진 명령을 저장한다 (태수 PhaseManager 가 부름). 범위 밖이면 CMD_RANGE 로 자름."""
        if forward is not None:
            throttle = forward
        if steering is not None:
            steer = steering
        if not np.isfinite([throttle, steer]).all():
            raise ValueError('추진 명령은 유한한 값이어야 합니다.')
        lo, hi = Thruster.CMD_RANGE
        self.throttle = float(np.clip(throttle, lo, hi))
        self.steer = float(np.clip(steer, lo, hi))

    def thrusts(self):
        """현재 명령 → (왼쪽, 오른쪽) 추력 [N]. 앞(+x_b)이 양수.

        좌우 추력 차이로 조향: 왼쪽 = 전진 − 조향, 오른쪽 = 전진 + 조향 (오른쪽이 세면 왼쪽으로 돎).
        합이 범위를 넘으면 (예: 전진 1 + 조향 1 = 2) 추진기마다 CMD_RANGE 로 자름.
        """
        lo, hi = Thruster.CMD_RANGE
        cmds = np.clip((self.throttle - self.steer, self.throttle + self.steer), lo, hi)
        # 전진은 MAX_FORWARD, 후진은 MAX_REVERSE 비율 (명령 −1 → −MAX_REVERSE)
        return tuple(float(c * (Thruster.MAX_FORWARD if c >= 0 else Thruster.MAX_REVERSE)) for c in cmds)

    def build(self, spec):
        """세계 조립 때 자기 몸(body/geom 등)을 spec 에 추가한다.

        만드는 것 (이름은 전부 config.Names):
          boat      배 몸체. 원점 = 선체 상자 중심 (Frames.BODY_ORIGIN)
          boat_free 자유 관절 (이동 3 + 회전 3)
          hull      선체를 감싸는 상자. 물이 이 크기를 읽어 부력 복셀을 만듦.
                    A형(Hull.SHAPE)이면 투명·충돌 없음이고 실제 형상은 A_FRAME_GEOMS (다리·연결부·뱃머리)
          thr_l/r   추진기 점 (Thruster.POS_L_B/R_B)
          tow_point 줄 묶는 점 (Hull.TOW_POINT_B)
          cam_bow   앞을 보는 카메라 (Sensors.CAMERA_POS_B). 일반 영상만, 열화상은 THERMAL_ENABLED 때
          sonar     소나 자리 점 (Sensors.SONAR_POS_B). 1주차는 자리만 (SONAR_ENABLED = False)
        시작 높이는 정수 흘수만큼 잠긴 자리 → 물을 켜도 처음부터 거의 가만히 떠 있음.
        """
        if Thruster.METHOD != Thruster.ACC:
            raise NotImplementedError(f"Thruster.METHOD={Thruster.METHOD!r}: 지금은 ACC 만 구현 (actuator 없음)")
        if Hull.OUTBOARD_SEPARATE:
            raise NotImplementedError("Hull.OUTBOARD_SEPARATE=True 는 아직 구현 안 함 (2026-11-12 회의: False)")

        draft = config.Expect.DRAFT or 0.0
        start_z = config.WaterParams.SURFACE_Z + Hull.HEIGHT / 2 - draft   # 바닥이 수면 아래 draft
        body = spec.worldbody.add_body(name=Names.BOAT_BODY, pos=(0.0, 0.0, start_z) if self.position is None else self.position,
                                       quat=(np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)))
        body.add_freejoint(name=Names.BOAT_FREE)

        # 질량은 geom 밀도로 자동 계산하지 않고 config 값을 그대로 넣음 (물의 흘수 정답값과 같은 숫자여야 함)
        body.explicitinertial = True
        body.mass = Hull.MASS
        body.ipos = Hull.COM
        body.inertia = Hull.INERTIA

        # 감싸는 상자: 물이 이 크기로 복셀을 만든 뒤 모양(hull_contains) 밖을 버린다. 직육면체일 때만 이 상자가 곧 선체
        box = Hull.SHAPE == "box"
        body.add_geom(
            name=Names.HULL_GEOM,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(Hull.LENGTH / 2, Hull.BEAM / 2, Hull.HEIGHT / 2),   # MuJoCo box 는 반 길이
            rgba=HULL_RGBA if box else (0, 0, 0, 0),
            group=config.Groups.PHYSICS if box else config.Groups.VISUAL,
            contype=config.Contacts.BOAT if box else 0,
            conaffinity=config.Contacts.BOAT_AFFINITY if box else 0,
        )
        if not box:
            self._add_a_frame(spec, body)

        body.add_site(name=Names.THRUSTER_L_SITE, pos=Thruster.POS_L_B, size=SITE_SIZE)
        body.add_site(name=Names.THRUSTER_R_SITE, pos=Thruster.POS_R_B, size=SITE_SIZE)
        body.add_site(name=Names.TOW_POINT_SITE, pos=Hull.TOW_POINT_B if self.tow_point_b is None else self.tow_point_b, size=SITE_SIZE)
        body.add_site(name=Names.SONAR_SITE, pos=Sensors.SONAR_POS_B, size=SITE_SIZE)

        # MuJoCo 카메라는 자기 −z 로 보고 +y 가 화면 위. 몸체(FLU)로 쓰면
        #   화면 오른쪽 = −y_b, 화면 위 = (sin p, 0, cos p), 보는 방향 = (cos p, 0, −sin p)  (p = 아래로 숙인 각)
        pitch = Sensors.CAMERA_PITCH_DOWN or 0.0   # 비전 담당과 합의 전엔 수평
        cam = body.add_camera(name=Names.CAMERA_BOW, pos=Sensors.CAMERA_POS_B)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = (0.0, -1.0, 0.0, np.sin(pitch), 0.0, np.cos(pitch))
        if Sensors.CAMERA_FOVY is not None:   # 합의 전엔 MuJoCo 기본 45°
            cam.fovy = Sensors.CAMERA_FOVY
        return body

    @staticmethod
    def _add_a_frame(spec, body):
        """A형 선체의 실제 충돌·표시 형상: 다리 2개 + 연결부 + 뱃머리(사다리꼴 기둥 메시). 모양은 config.Hull."""
        L, B, H = Hull.LENGTH, Hull.BEAM, Hull.HEIGHT
        stern, slot_end = -L / 2, -L / 2 + Hull.SLOT_DEPTH
        leg_w = (B - Hull.SLOT_WIDTH) / 2
        leg_y = B / 2 - leg_w / 2
        if Hull.MOON_POOL is not None:
            raise NotImplementedError("Hull.MOON_POOL 충돌 형상은 아직 없음 (물 복셀만 지원)")
        boxes = {
            # 다리: 선미 ~ 뱃머리 시작. 연결부: 홈 앞끝 ~ 뱃머리 시작, 다리 사이
            A_FRAME_GEOMS[0]: ((stern + Hull.BOW_START_X) / 2, leg_y, (Hull.BOW_START_X - stern) / 2, leg_w / 2),
            A_FRAME_GEOMS[1]: ((stern + Hull.BOW_START_X) / 2, -leg_y, (Hull.BOW_START_X - stern) / 2, leg_w / 2),
            A_FRAME_GEOMS[2]: ((slot_end + Hull.BOW_START_X) / 2, 0.0, (Hull.BOW_START_X - slot_end) / 2, Hull.SLOT_WIDTH / 2),
        }
        common = dict(rgba=HULL_RGBA, group=config.Groups.PHYSICS,
                      contype=config.Contacts.BOAT, conaffinity=config.Contacts.BOAT_AFFINITY)
        for name, (x, y, hx, hy) in boxes.items():
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=(x, y, 0.0), size=(hx, hy, H / 2), **common)
        verts = []
        for z in (-H / 2, H / 2):
            verts += [Hull.BOW_START_X, B / 2, z, Hull.BOW_START_X, -B / 2, z,
                      L / 2, -Hull.BOW_TIP_HALF_BEAM, z, L / 2, Hull.BOW_TIP_HALF_BEAM, z]
        mesh = spec.add_mesh()
        mesh.name, mesh.uservert = A_FRAME_GEOMS[3] + "_mesh", verts
        body.add_geom(name=A_FRAME_GEOMS[3], type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, **common)

    def bind(self, model):
        """이름 → id 를 한 번만 찾아 캐시한다. 없으면 assert 로 멈춘다.

        스텝마다 이름으로 찾으면 느리고, 이름이 틀려도 −1 이 조용히 들어가 엉뚱한 부품에 힘이 걸린다.
        """
        def find(obj, name):
            i = mujoco.mj_name2id(model, obj, name)
            assert i >= 0, f"모델에 {name!r} 없음 — build() 를 했는지, config.Names 와 같은지 확인"
            return i

        self.body_id = find(mujoco.mjtObj.mjOBJ_BODY, Names.BOAT_BODY)
        free_id = find(mujoco.mjtObj.mjOBJ_JOINT, Names.BOAT_FREE)
        assert model.jnt_type[free_id] == mujoco.mjtJoint.mjJNT_FREE, f"{Names.BOAT_FREE!r} 가 자유 관절이 아님"
        # 자유 관절 칸 위치: qpos[adr:adr+7] = 위치 3 + 쿼터니언 4, qvel[adr:adr+6] = 속도 3 + 각속도 3
        self.qpos_adr = model.jnt_qposadr[free_id]
        self.qvel_adr = model.jnt_dofadr[free_id]
        self.hull_geom_id = find(mujoco.mjtObj.mjOBJ_GEOM, Names.HULL_GEOM)
        self.thr_l_site_id = find(mujoco.mjtObj.mjOBJ_SITE, Names.THRUSTER_L_SITE)
        self.thr_r_site_id = find(mujoco.mjtObj.mjOBJ_SITE, Names.THRUSTER_R_SITE)
        self.tow_site_id = find(mujoco.mjtObj.mjOBJ_SITE, Names.TOW_POINT_SITE)
        self.sonar_site_id = find(mujoco.mjtObj.mjOBJ_SITE, Names.SONAR_SITE)
        self.camera_id = find(mujoco.mjtObj.mjOBJ_CAMERA, Names.CAMERA_BOW)
        self.joint_id = free_id
        self.geom_id = self.hull_geom_id
        # 실제로 부딪히는 배 형상 (간격·관통 검사용). A형이면 다리·연결부·뱃머리, 직육면체면 hull 하나
        self.collision_geom_ids = tuple(g for g in range(model.ngeom)
                                        if model.geom_bodyid[g] == self.body_id and model.geom_contype[g])
        assert self.collision_geom_ids, "배에 충돌 형상이 없음"
        self.site_ids = (self.thr_l_site_id, self.thr_r_site_id)

        if self.winch:
            self.tendon_id = find(mujoco.mjtObj.mjOBJ_TENDON, Names.TETHER_TENDON)
            self._model = model
            self.reset_winch(model)   # 처음엔 짧게 감긴 상태

        # 물은 무게를 모델에서 읽음 → 누가 모델을 바꿔 config 와 달라지면 흘수 정답값이 조용히 틀림
        assert abs(model.body_mass[self.body_id] - Hull.MASS) < 1e-9, (
            f"모델 배 무게 {model.body_mass[self.body_id]} kg ≠ config.Hull.MASS {Hull.MASS} kg")

    # ── 구명환 윈치: 짧게 끌고 가다가 줄을 풀어 건네줌 ─────────────────
    def _check_winch(self, model=None):
        if not self.winch:
            raise ValueError('winch=True 로 만든 로봇만 줄을 감고 풀 수 있습니다.')
        if model is not None and model is not self._model:
            raise ValueError('이 로봇이 bind 된 모델이 필요합니다.')

    def rope_length(self, model):
        """지금 윈치가 내준 로프 기준 길이 [m] (= tendon 스프링 기준 길이, 이보다 길게 늘어나야 장력)."""
        self._check_winch(model)
        return float(model.tendon_lengthspring[self.tendon_id, 1])

    def _set_rope_length(self, model, length):
        # tether.py 는 매 스텝 이 값을 읽어 장력을 계산한다 → 여기만 바꾸면 줄이 풀린다
        model.tendon_lengthspring[self.tendon_id, :] = (0.0, length)

    def reset_winch(self, model):
        """줄을 STOW_LENGTH 로 감은 처음 상태로. mj_resetData 는 모델 값을 되돌리지 않으므로 리셋 때 함께 부른다."""
        self._check_winch(model)
        self._set_rope_length(model, Winch.STOW_LENGTH)
        self.delivery_state = STOWED

    def pay_out(self):
        """건네주기 시작: 다음 스텝부터 PAYOUT_SPEED 로 PAYOUT_LENGTH 까지 줄을 푼다. 이미 시작했으면 False."""
        self._check_winch()
        if self.delivery_state != STOWED:
            return False
        self.delivery_state = PAYING_OUT
        return True

    def release_winch(self, length):
        """윈치를 자유롭게 풀어 줄 기준 길이를 length 로 바로 늘린다 (구명환 발사용). 다 풀린 것으로 본다.

        발사한 환이 날아가는 동안 줄이 당기지 않게 한다. 감는 것은 하지 않는다 (지금보다 짧으면 ValueError).
        """
        self._check_winch(self._model)
        if not np.isfinite(length) or length < self.rope_length(self._model):
            raise ValueError('풀 길이는 지금 줄 길이 이상의 유한한 값이어야 합니다.')
        self._set_rope_length(self._model, float(length))
        self.delivery_state = PAID_OUT

    @property
    def delivery_ready(self):
        """'건네줬다' 신호 (CONTRACT delivery_ready). 줄을 다 푼 뒤 True.

        True 라고 바로 잡히지는 않음 — 사람 쪽 HumanGrasp.try_grab 이 손과 환의 거리·속도를 다시 확인한다.
        """
        return self.delivery_state == PAID_OUT

    def _update_winch(self, model):
        if self.delivery_state != PAYING_OUT:
            return
        length = min(Winch.PAYOUT_LENGTH, self.rope_length(model) + Winch.PAYOUT_SPEED * model.opt.timestep)
        self._set_rope_length(model, length)
        if length >= Winch.PAYOUT_LENGTH:
            self.delivery_state = PAID_OUT

    def nozzles_submerged(self, data):
        """(왼쪽, 오른쪽) 노즐이 수면 아래인지. surface_height 가 없으면 항상 (True, True)."""
        if self.surface_height is None:
            return (True, True)
        pos = data.site_xpos[[self.thr_l_site_id, self.thr_r_site_id]]
        surface = self.surface_height(pos[:, 0], pos[:, 1], data.time)   # 두 점을 한 번에 (WAVE_HEIGHT_VECTORIZED)
        return tuple(bool(z < s) for z, s in zip(pos[:, 2], surface))

    def add_forces(self, model, data, acc):
        """스텝마다 힘을 acc(ForceAccumulator)에 더한다. 월드 좌표.

        추력은 배 앞쪽(+x_b) 방향 → body_to_world 로 월드 방향으로 돌리고,
        추진기 점의 월드 위치(site_xpos)에 건다. 점이 무게중심에서 벗어나 있으니 돌림힘은 acc 가 계산.
        """
        for site_id, thrust, wet in zip((self.thr_l_site_id, self.thr_r_site_id), self.thrusts(),
                                        self.nozzles_submerged(data)):
            if not wet:
                continue   # 노즐이 공기 중이면 물을 못 밀어 추력 없음
            force_w = frames.body_to_world(data, self.body_id, (thrust, 0.0, 0.0))
            acc.add(self.body_id, force_w, point_w=data.site_xpos[site_id].copy())
        if self.winch:
            self._update_winch(model)
