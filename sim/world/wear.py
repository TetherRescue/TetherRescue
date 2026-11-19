"""양팔을 곧게 나란히 뻗어 동시에 구명환을 통과하는 관절 제어.

접근 이동은 별도 시험 장치/전달부의 역할이다. 실제 pose는 물리 엔진만 적분한다.
"""
import numpy as np

from sim import config


def smooth(value):
    value = float(np.clip(value, 0, 1))
    return value**3 * (10 - 15 * value + 6 * value**2)


class SelfWear:
    REQUIRES = {'max_dt': .002}

    def __init__(self, human, grasp):
        self.human, self.grasp = human, grasp
        self.phase, self.phase_start = 'WAITING', None
        self.progress = 0.
        self._next_control = 0.

    def build(self, spec):
        import mujoco
        eq = spec.add_equality(name=config.Names.HUMAN_RING_WORN_EQ,
                               type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                               name1=config.Names.HUMAN_BODY, name2=config.Names.LIFEBUOY_BODY, active=False)
        eq.data[:] = 0
        eq.data[10] = 1
        eq.solref[:] = [.02, 1]

    def bind(self, model):
        if not self.human.articulated_arms:
            raise ValueError('착용 동작에는 관절 팔이 필요합니다.')
        self.model, self.body = model, self.human.body_id
        self.ring = model.body(config.Names.LIFEBUOY_BODY).id
        self.eq = model.equality(config.Names.HUMAN_RING_WORN_EQ).id
        self.initial_eq = model.eq_data[self.eq].copy()
        self.qadr = {s: model.jnt_qposadr[self.human.arm_joints[s]] for s in ('l', 'r')}
        self.sites = {'l': model.site(config.Names.HUMAN_GRIP_L_SITE).id,
                      'r': model.site(config.Names.HUMAN_GRIP_R_SITE).id}
        self.target = {s: np.zeros(7) for s in ('l', 'r')}
        self.extended = {}
        for side in ('l', 'r'):
            shoulder = np.array([.5, .14 if side == 'l' else -.14, 0.])
            upper = np.asarray(getattr(config.Human, f'ELBOW_{side.upper()}_B')) - shoulder
            # Rx Ry Rz 순서. 어깨만 회전해 위팔 방향을 몸의 +x로 맞춘다.
            self.extended[side] = np.array([0., np.arctan2(upper[2], np.linalg.norm(upper[:2])),
                                           -np.arctan2(upper[1], upper[0]), 0., 0., 0., 0.])

    def _local_hand(self, data, side):
        rotation = data.xmat[self.body].reshape(3, 3)
        return rotation.T @ (data.site_xpos[self.sites[side]] - data.xpos[self.body])

    def _enter(self, phase, data):
        self.phase, self.phase_start = phase, float(data.time)

    def start(self, data, *, delivery_ready=False):
        if self.phase != 'WAITING':
            return True
        if not delivery_ready:
            return False
        self._enter('ALIGN_ARMS', data)
        self._next_control = float(data.time)
        return True

    def _arms_ready(self, data):
        return all(np.max(np.abs(data.qpos[self.qadr[s]] - self.extended[s])) < .025 for s in ('l', 'r'))

    def measurements(self, data):
        rotation = data.xmat[self.body].reshape(3, 3)
        relative = rotation.T @ (data.xpos[self.ring] - data.xpos[self.body])
        normal = rotation.T @ data.xmat[self.ring].reshape(3, 3)[:, 0]
        return dict(phase=self.phase, center=relative,
                    position_error=float(np.linalg.norm(relative - config.Lifebuoy.POSITION_IN_HUMAN_B)),
                    axis_error=float(np.arccos(np.clip(normal[0], -1, 1))),
                    worn=bool(data.eq_active[self.eq]),
                    ready_to_tow=self.phase == 'WORN' and bool(data.eq_active[self.eq]))

    def update(self, data):
        import mujoco
        if self.phase == 'WAITING':
            return
        elapsed = float(data.time) - self.phase_start
        if elapsed > 20 and self.phase != 'WORN':
            raise RuntimeError(f'착용 단계 {self.phase}에서 목표 자세에 도달하지 못했습니다.')
        if data.time + 1e-12 < self._next_control:
            return
        self._next_control = float(data.time) + 1 / config.Sim.CONTROL_HZ
        for side in ('l', 'r'):
            goal = self.extended[side] * (smooth(elapsed / 4.) if self.phase == 'ALIGN_ARMS' else 1.)
            self.target[side] += np.clip(goal - self.target[side], -.025, .025)
            data.ctrl[self.human.arm_actuators[side]] = self.target[side]
        if self.phase == 'ALIGN_ARMS':
            if elapsed >= 4 and self._arms_ready(data):
                self._enter('INSERT_BOTH', data)
        elif self.phase == 'INSERT_BOTH':
            # 시험 장치에는 접근 진행률만 전달한다. 실제 양팔 자세가 틀어지면 접근을 멈춘다.
            if self._arms_ready(data):
                self.progress = max(self.progress, smooth(elapsed / 7.))
            state = self.measurements(data)
            if (elapsed >= 9 and self.progress == 1. and self._arms_ready(data)
                    and state['position_error'] < .015 and state['axis_error'] < .03):
                self.model.eq_data[self.eq, :3] = 0
                self.model.eq_data[self.eq, 3:6] = state['center']
                mujoco.mju_mulQuat(self.model.eq_data[self.eq, 6:10],
                                   data.xquat[self.body] * [1, -1, -1, -1], data.xquat[self.ring])
                data.eq_active[self.eq] = True
                self._enter('WORN', data)

    def reset(self, data):
        self.grasp.reset(self.model, data)
        data.eq_active[self.eq] = False
        self.model.eq_data[self.eq] = self.initial_eq
        self.phase, self.phase_start = 'WAITING', None
        self.progress, self._next_control = 0., 0.
        for side in ('l', 'r'):
            self.target[side][:] = 0
            data.ctrl[self.human.arm_actuators[side]] = 0

    def add_forces(self, model, data, acc):
        """팔은 내부 actuator, 착용 지지는 equality에서 처리한다."""
