"""잡기·양팔 착용 시험 장면. GUI와 테스트가 같은 장면을 사용한다.

시험용 외력 장치는 배 제어와 구분하며 실제 장면에서는 전달·접근 제어로 대체한다.
"""
import mujoco
import numpy as np

from sim import config
from sim.frames import body_velocity_w
from sim.grasp import HumanGrasp
from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.tether import Tether
from sim.wear import SelfWear
from sim.world import advance_world, build_world


class PullFixture:
    """배 대신 쓰는 10kg 시험 질점. 외력 제어로 연결점을 이동하며 위치를 강제 수정하지 않는다."""
    REQUIRES = {}

    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)
        self.target = self.position.copy()

    def build(self, spec):
        body = spec.worldbody.add_body(name=config.Names.TEST_TOW_BODY, pos=self.position)
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.06, 0, 0], mass=10.,
                      rgba=[.2, .6, 1., 1.], contype=0, conaffinity=0)
        body.add_site(name=config.Names.TOW_POINT_SITE, size=[.015, 0, 0])

    def bind(self, model):
        self.body_id = model.body(config.Names.TEST_TOW_BODY).id

    def add_forces(self, model, data, acc):
        velocity, _ = body_velocity_w(model, data, self.body_id)
        force = np.clip(150 * (self.target - data.xpos[self.body_id]) - 80 * velocity, -150, 150)
        # 중력 시험에서도 연결점 높이를 지지하는 외부 시험 장치다.
        force -= model.body_mass[self.body_id] * model.opt.gravity
        acc.add(self.body_id, force)


class WearApproachFixture:
    """중력 0에서 전달된 환을 지지하고 사람의 접근을 외력으로 재현하는 시험 장치.

    배 제어·수영 모델이 아니다. 착용 완료 시 지지를 모두 해제해 자유롭게 견인한다.
    """
    REQUIRES = {}

    def __init__(self, wear):
        self.wear = wear

    def build(self, spec):
        pass

    def bind(self, model):
        self.human = model.body(config.Names.HUMAN_BODY).id
        self.ring = model.body(config.Names.LIFEBUOY_BODY).id
        self.human_start = model.body_pos[self.human].copy()
        self.ring_start = model.body_pos[self.ring].copy()
        self.displacement = self.ring_start - self.human_start - config.Lifebuoy.POSITION_IN_HUMAN_B

    def add_forces(self, model, data, acc):
        if self.wear.phase == 'WORN':
            return
        for body, target, stiffness, damping, torque_k, torque_d in (
                (self.human, self.human_start + self.displacement * self.wear.progress, 1000., 530., 200., 40.),
                (self.ring, self.ring_start, 400., 60., 30., 8.)):
            velocity, angular = body_velocity_w(model, data, body)
            error = np.empty(3)
            mujoco.mju_subQuat(error, np.array([1., 0., 0., 0.]), data.xquat[body])
            # 시험의 기준 자세는 월드 단위 자세다. 힘과 토크에 상한을 둔다.
            force = np.clip(stiffness * (target - data.xpos[body]) - damping * velocity, -200., 200.)
            torque = np.clip(torque_k * error - torque_d * angular, -80., 80.)
            acc.add(body, force, torque_w=torque)


class GraspDemo:
    def __init__(self, *, water=False, receive=True):
        self.water = water
        self.receive = receive
        self.human = Human(position=(0, 0, -.02 if water else 1.), grip_sites=True)
        hand = np.asarray(config.Human.GRIP_L_POINT_B) + np.asarray(self.human.position)
        ring_grip = np.asarray(config.Lifebuoy.GRIP_L_POINT_B)
        position = hand - ring_grip + [.02, 0, 0]
        self.ring = Lifebuoy(attached=False, position=position, grip_sites=True)
        self.fixture = PullFixture(position + config.Lifebuoy.TOW_POINT_B + np.array([1.5, 0, 0]))
        self.grasp = HumanGrasp()
        self.rope = Tether(enabled=True, attached=True)
        parts = [self.fixture, self.human, self.ring, self.rope, self.grasp]
        if water:
            from sim.water import Water
            parts.insert(0, Water(waves=(), floating_bodies=[config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY],
                                  include_boat=False, surface_visual=True))
        self.world = build_world(parts=parts, gravity=None if water else 0.)
        self.world.model.vis.headlight.ambient[:] = [.3, .3, .3]
        self.phase = 'WAITING' if self.receive else 'NOT_RECEIVED'
        self._human_start = self.world.data.xpos[self.human.body_id].copy()
        self.grab_time = None
        self.release_time = None

    def reset(self):
        mujoco.mj_resetData(self.world.model, self.world.data)
        self.grasp.reset(self.world.model, self.world.data)
        self.rope.reset(self.world.model)
        self.fixture.target[:] = self.fixture.position
        self.phase = 'WAITING' if self.receive else 'NOT_RECEIVED'
        self.grab_time = self.release_time = None
        mujoco.mj_forward(self.world.model, self.world.data)

    def step(self):
        world = self.world
        mujoco.mj_forward(world.model, world.data)
        now = float(world.data.time)
        receive_at = 0. if self.water else 2.
        release_at = receive_at + 6.
        if self.phase == 'WAITING' and now + 1e-12 >= receive_at:
            if not self.grasp.try_grab(world.model, world.data, delivery_ready=True):
                raise RuntimeError('시험 초기 배치가 잡기 조건을 만족하지 않습니다.')
            self.grab_time = now
            self.phase = 'HELD'
        if self.phase == 'HELD' and now + 1e-12 >= release_at:
            self.grasp.release(world.model, world.data)
            self.release_time = now
            self.phase = 'RELEASED'
        self.fixture.target[:] = self.fixture.position + [.4 * max(0., now - receive_at), 0, 0]
        mujoco.mj_forward(world.model, world.data)
        advance_world(world)
        if (world.data.warning.number.any() or not all(np.isfinite(state).all() for state in
                                                     (world.data.qpos, world.data.qvel, world.data.qacc))):
            raise RuntimeError('잡기 독립 시험의 물리 상태가 비정상입니다.')

    def readings(self):
        state = self.grasp.metrics(self.world.model, self.world.data)
        state.update(phase=self.phase, tension=self.rope.tension(self.world.model, self.world.data),
                     human_movement=float(np.linalg.norm(self.world.data.xpos[self.human.body_id] - self._human_start)))
        return state


class WearDemo:
    def __init__(self, *, contact=True):
        self.human = Human(grip_sites=True, articulated_arms=True, ring_contact=contact)
        # 정면에 전달된 환. 양팔을 함께 뻗고 사람이 접근해 구멍을 통과한다.
        position = np.array([1.20, 0., 1.])
        self.ring = Lifebuoy(attached=False, position=position, grip_sites=True, human_contact=contact)
        self.fixture = PullFixture((2.0, .4, 1.))
        self.rope = Tether(enabled=True, attached=True)
        self.grasp = HumanGrasp(ring_site=config.Names.LIFEBUOY_GRIP_R_SITE)
        self.wear = SelfWear(self.human, self.grasp)
        self.approach_fixture = WearApproachFixture(self.wear)
        self.world = build_world(parts=[self.fixture, self.human, self.ring, self.rope, self.grasp, self.wear, self.approach_fixture], gravity=0)
        self.world.model.vis.headlight.ambient[:] = [.3, .3, .3]

    def step(self):
        world = self.world
        mujoco.mj_forward(world.model, world.data)
        if world.data.time >= 1 and self.wear.phase == 'WAITING':
            if not self.wear.start(world.data, delivery_ready=True):
                raise RuntimeError('받기 조건을 만족하지 못했습니다.')
        self.wear.update(world.data)
        mujoco.mj_forward(world.model, world.data)
        advance_world(world)
        if world.data.warning.number.any() or not all(np.isfinite(state).all() for state in
                                                     (world.data.qpos, world.data.qvel, world.data.qacc)):
            raise RuntimeError('착용 시험의 물리 상태가 비정상입니다.')

    def reset(self):
        mujoco.mj_resetData(self.world.model, self.world.data)
        self.wear.reset(self.world.data)
        self.rope.reset(self.world.model)
        self.fixture.target[:] = self.fixture.position
        mujoco.mj_forward(self.world.model, self.world.data)
