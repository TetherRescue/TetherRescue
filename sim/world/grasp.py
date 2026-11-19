"""사람 손과 독립 구명환의 잡기·놓기. 로프는 계속 구명환에 연결한다."""

from dataclasses import dataclass
import math

import numpy as np

from sim import config


@dataclass(frozen=True)
class GraspSettings:
    max_distance: float = .08  # m, 손과 환의 잡기 site 간격
    max_relative_speed: float = .1  # m/s, 잡는 점의 상대 속도
    max_relative_angular_speed: float = .5  # rad/s, 급회전 중 강제 결합 방지

    def __post_init__(self):
        for value in vars(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('잡기 허용 오차는 유한한 양수여야 합니다.')


class HumanGrasp:
    """전달 신호와 거리·상대 속도를 확인해 비활성 weld를 켠다.

    손가락·팔 관절·접촉의 실제 잡기 동작이 아닌 고정 잡기 근사다.
    build 순서: Human(grip_sites=True) → Lifebuoy(attached=False, grip_sites=True) → HumanGrasp.
    """
    REQUIRES = {'max_dt': .002}

    def __init__(self, *, hand='left', settings=None, ring_site=None):
        if hand not in ('left', 'right'):
            raise ValueError("hand는 left 또는 right여야 합니다.")
        self.hand = hand
        self.settings = GraspSettings() if settings is None else settings
        self.hand_name = config.Names.HUMAN_GRIP_L_SITE if hand == 'left' else config.Names.HUMAN_GRIP_R_SITE
        self.ring_name = config.Names.LIFEBUOY_GRIP_L_SITE if hand == 'left' else config.Names.LIFEBUOY_GRIP_R_SITE
        if ring_site is not None:
            self.ring_name = ring_site
        self.eq_id = None
        self._model = None

    def build(self, spec):
        import mujoco

        for name in (self.hand_name, self.ring_name):
            if spec.site(name) is None:
                raise ValueError(f'잡기 제약 생성 전에 {name} site가 필요합니다.')
        equality = spec.add_equality(name=config.Names.HUMAN_RING_GRASP_EQ,
                                     type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                     name1=spec.site(self.hand_name).parent.name, name2=config.Names.LIFEBUOY_BODY,
                                     active=False)
        equality.data[:] = 0
        equality.data[10] = 1  # 회전도 고정하는 weld의 torque scale
        equality.solref[:] = [.02, 1]

    def bind(self, model):
        import mujoco

        self.human_body = model.body(config.Names.HUMAN_BODY).id
        self.ring_body = model.body(config.Names.LIFEBUOY_BODY).id
        for body in (self.human_body, self.ring_body):
            joint = int(model.body_jntadr[body])
            if (model.body_parentid[body] != 0 or model.body_jntnum[body] != 1 or joint < 0
                    or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE)):
                raise ValueError('사람과 구명환은 각각 독립된 자유 강체여야 합니다.')
        self.hand_site = model.site(self.hand_name).id
        self.ring_site = model.site(self.ring_name).id
        self.grasp_body = int(model.site_bodyid[self.hand_site])
        ancestor = self.grasp_body
        while ancestor != 0 and ancestor != self.human_body:
            ancestor = int(model.body_parentid[ancestor])
        if (ancestor != self.human_body or model.site_bodyid[self.ring_site] != self.ring_body):
            raise ValueError('손·구명환 잡기 site의 소유 몸체를 확인하세요.')
        self.eq_id = model.equality(config.Names.HUMAN_RING_GRASP_EQ).id
        self._initial_data = model.eq_data[self.eq_id].copy()
        self._model = model

    def _check(self, model):
        if model is not self._model or self.eq_id is None:
            raise ValueError('이 모델에 build·bind된 잡기 제약이 필요합니다.')

    def is_held(self, data):
        if self.eq_id is None:
            raise RuntimeError('HumanGrasp.bind를 먼저 호출하세요.')
        return bool(data.eq_active[self.eq_id])

    def metrics(self, model, data):
        """호출 전 mj_forward로 현재 위치·속도를 갱신한다."""
        import mujoco

        self._check(model)
        velocities = []
        for site in (self.hand_site, self.ring_site):
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0)
            velocities.append(velocity)
        return dict(held=self.is_held(data),
                    distance=float(np.linalg.norm(data.site_xpos[self.hand_site] - data.site_xpos[self.ring_site])),
                    relative_speed=float(np.linalg.norm(velocities[0][3:] - velocities[1][3:])),
                    relative_angular_speed=float(np.linalg.norm(velocities[0][:3] - velocities[1][:3])))

    def try_grab(self, model, data, *, delivery_ready=False):
        """전달 완료 신호가 있어도 손 근처·저속일 때만 연결한다. 위치·속도는 변경하지 않는다.

        호출 전 mj_forward, 호출 후 힘을 읽기 전 다시 mj_forward가 필요하다.
        """
        self._check(model)
        if not isinstance(delivery_ready, bool):
            raise ValueError('delivery_ready는 bool 신호여야 합니다.')
        if self.is_held(data):
            return True
        if not delivery_ready:
            return False
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            return False
        state = self.metrics(model, data)
        if not all(math.isfinite(state[key]) for key in ('distance', 'relative_speed', 'relative_angular_speed')):
            return False
        if (state['distance'] > self.settings.max_distance
                or state['relative_speed'] > self.settings.max_relative_speed
                or state['relative_angular_speed'] > self.settings.max_relative_angular_speed):
            return False
        # 초기 배치가 아니라 실제 받은 순간의 상대 자세를 유지한다.
        rotation = data.xmat[self.grasp_body].reshape(3, 3)
        relative_position = rotation.T @ (data.xpos[self.ring_body] - data.xpos[self.grasp_body])
        inverse = data.xquat[self.grasp_body] * np.array([1, -1, -1, -1])
        import mujoco
        relative_quaternion = np.empty(4)
        mujoco.mju_mulQuat(relative_quaternion, inverse, data.xquat[self.ring_body])
        model.eq_data[self.eq_id, :3] = 0
        model.eq_data[self.eq_id, 3:6] = relative_position
        model.eq_data[self.eq_id, 6:10] = relative_quaternion
        data.eq_active[self.eq_id] = True
        return True

    def release(self, model, data):
        """놓기. 힘을 읽기 전 mj_forward로 제약 상태를 갱신한다."""
        self._check(model)
        was_held = self.is_held(data)
        data.eq_active[self.eq_id] = False
        return was_held

    def reset(self, model, data):
        self.release(model, data)
        model.eq_data[self.eq_id] = self._initial_data

    def add_forces(self, model, data, acc):
        """잡기 제약 힘은 엔진이 계산한다. 외력에 중복 합산하지 않는다."""
