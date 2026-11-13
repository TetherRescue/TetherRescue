"""2 로봇: 선체·추진기·조향·전복 판정 (담당: 동경)."""
import mujoco
import numpy as np

from sim import config, frames
from sim.config import Hull, Names, Thruster

SITE_SIZE = (0.03,)   # 점(site) 표시 크기 [m]. 그림용이라 물리에 영향 없음


class Robot:
    REQUIRES = {}  # 예: {"max_dt": 0.004}

    def __init__(self, surface_height=None):
        """surface_height: 수면 높이 함수 (x, y, t) -> z, 예: Water().wave_height. x, y 는 배열로 받음.

        None 이면 1주차 가정 "노즐은 항상 물속" (WaterAPI.USE_IN_WEEK1 = False).
        주면 노즐이 수면 위로 나온 스텝에는 그 노즐 추력을 0 으로 한다 (Thruster.FULL_THRUST_DEPTH 주석).
        """
        self.throttle = 0.0   # 전진 명령 (−1 최대 후진 ~ +1 최대 전진)
        self.steer = 0.0      # 조향 명령 (+ 왼쪽으로 돎)
        self.surface_height = surface_height

    def set_command(self, throttle, steer=0.0):
        """추진 명령을 저장한다 (태수 PhaseManager 가 부름). 범위 밖이면 CMD_RANGE 로 자름."""
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
          hull      선체 상자. 물이 이 크기를 읽어 부력을 계산
          thr_l/r   추진기 점 (Thruster.POS_L_B/R_B)
          tow_point 줄 묶는 점 (Hull.TOW_POINT_B)
        시작 높이는 정수 흘수만큼 잠긴 자리 → 물을 켜도 처음부터 거의 가만히 떠 있음.
        """
        if Thruster.METHOD != Thruster.ACC:
            raise NotImplementedError(f"Thruster.METHOD={Thruster.METHOD!r}: 지금은 ACC 만 구현 (actuator 없음)")
        if Hull.OUTBOARD_SEPARATE:
            raise NotImplementedError("Hull.OUTBOARD_SEPARATE=True 는 아직 구현 안 함 (2026-11-12 회의: False)")

        draft = config.Expect.DRAFT or 0.0
        start_z = config.WaterParams.SURFACE_Z + Hull.HEIGHT / 2 - draft   # 바닥이 수면 아래 draft
        body = spec.worldbody.add_body(name=Names.BOAT_BODY, pos=(0.0, 0.0, start_z))
        body.add_freejoint(name=Names.BOAT_FREE)

        # 질량은 geom 밀도로 자동 계산하지 않고 config 값을 그대로 넣음 (물의 흘수 정답값과 같은 숫자여야 함)
        body.explicitinertial = True
        body.mass = Hull.MASS
        body.ipos = Hull.COM
        body.inertia = Hull.INERTIA

        hull = body.add_geom(
            name=Names.HULL_GEOM,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(Hull.LENGTH / 2, Hull.BEAM / 2, Hull.HEIGHT / 2),   # MuJoCo box 는 반 길이
            rgba=(0.9, 0.4, 0.1, 1.0),
        )
        if config.Groups.PHYSICS is not None:   # 가영 님이 정하기 전엔 MuJoCo 기본값(0)
            hull.group = config.Groups.PHYSICS

        body.add_site(name=Names.THRUSTER_L_SITE, pos=Thruster.POS_L_B, size=SITE_SIZE)
        body.add_site(name=Names.THRUSTER_R_SITE, pos=Thruster.POS_R_B, size=SITE_SIZE)
        body.add_site(name=Names.TOW_POINT_SITE, pos=Hull.TOW_POINT_B, size=SITE_SIZE)
        return body

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

        # 물은 무게를 모델에서 읽음 → 누가 모델을 바꿔 config 와 달라지면 흘수 정답값이 조용히 틀림
        assert abs(model.body_mass[self.body_id] - Hull.MASS) < 1e-9, (
            f"모델 배 무게 {model.body_mass[self.body_id]} kg ≠ config.Hull.MASS {Hull.MASS} kg")

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
