"""잡기·양팔 착용·배 충돌 시험 장면. GUI와 테스트가 같은 장면을 사용한다.

시험용 외력 장치는 배 제어와 구분하며 실제 장면에서는 전달·접근 제어로 대체한다.
"""
import functools
import math

import mujoco
import numpy as np

from sim import config
from sim.frames import body_velocity_w
from sim.world.grasp import HumanGrasp
from sim.world.human import Human
from sim.world.lifebuoy import Lifebuoy
from sim.robot.robot import Robot
from sim.world.tether import Tether
from sim.water.water import Water
from sim.world.wear import SelfWear
from sim.world.world import advance_world, build_world


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
                      rgba=[.2, .6, 1., 1.], group=config.Groups.VISUAL, contype=0, conaffinity=0)
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
            from sim.water.water import Water
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


# ─────────────────────────────────────────────────────────────
# 배 ↔ 사람 충돌 시험 (tests/world/test_contact.py, sim/world/view_contact.py)
# ─────────────────────────────────────────────────────────────
# 접근 단계의 사람은 아직 구명환이 없다 (구명환은 배에 실려 있다가 전달된다).
CONTACT_SETTLE_SECONDS = 5.0   # 사람이 물에 자리 잡는 시간 (측정: 5초 뒤 속도 약 0.003 m/s)
_FLOATING = (config.Names.HUMAN_BODY,)
_FREE_JOINTS = (config.Names.HUMAN_FREE,)


def _floating_human():
    return Human(position=(0, 0, config.WaterParams.SURFACE_Z - .03))


@functools.lru_cache(maxsize=1)
def _settled_pose():
    human = _floating_human()
    world = build_world(parts=[Water(waves=(), include_boat=False, floating_bodies=list(_FLOATING)), human])
    model, data = world.model, world.data
    for _ in range(round(CONTACT_SETTLE_SECONDS / model.opt.timestep)):
        advance_world(world)
    if np.linalg.norm(data.qvel) >= 0.05:
        raise RuntimeError('사람이 물에 자리 잡지 못했습니다.')
    return {name: tuple(data.qpos[model.jnt_qposadr[model.joint(name).id]:][:7]) for name in _FREE_JOINTS}


def settled_victim_pose():
    """배 없이 구명환 없는 사람만 잔잔한 물에 띄워 자리 잡은 자세 {자유 관절 이름: qpos 7개}.

    계산은 한 번만 하고 재사용한다.
    """
    return {name: np.array(qpos) for name, qpos in _settled_pose().items()}


def min_geom_distance(model, data, geoms_a, geoms_b, limit=5.0):
    """두 형상 묶음 사이 가장 가까운 부호 있는 거리 [m]. 겹치면 음수."""
    fromto = np.zeros(6)
    return min(mujoco.mj_geomDistance(model, data, a, b, limit, fromto) for a in geoms_a for b in geoms_b)


class ContactDemo:
    """물에 자리 잡은 사람(구명환 없음)을 향해 배가 직진해 부딪히는 장면.

    실제 흐름에서 배는 사람에게 닿지 않고 멈춰야 한다. 이 장면은 그 접근 제어가 실패했을 때의 최악 상황이다.

    heading_deg: 배가 달리는 방향 (0 = 발 쪽에서, 90 = 옆에서, 180 = 머리 쪽에서).
    speed: 시작 순간 배 속도 [m/s]. thrust: 계속 내는 전진 명령 (0~1). gap: 출발 시 최소 형상 간격 [m].
    관통은 배 원점 위치가 아니라 형상 사이 부호 있는 거리로 판정한다. 옆에서 밀면 배가 마찰로
    회전해 사람 둘레를 닿은 채 돌아가는데, 이는 정상 물리다.
    """
    HEADINGS = {'feet': 0., 'side': 90., 'head': 180.}
    DISTANCE_EVERY = 10   # 이 스텝마다 실제 형상 겹침을 따로 잰다 (접촉 검출과 별개)

    def __init__(self, heading_deg=0., *, speed=1.0, thrust=0.0, gap=0.15, surface_visual=False):
        if not np.isfinite([heading_deg, speed, thrust, gap]).all() or speed < 0 or gap <= 0:
            raise ValueError('방향·속도·추력·간격은 유한해야 하고 속도는 0 이상, 간격은 양수여야 합니다.')
        self.speed, self.thrust = float(speed), float(thrust)
        yaw = math.radians(heading_deg)
        self.direction = np.array([math.cos(yaw), math.sin(yaw), 0.])
        draft = config.Hull.MASS / (config.Physics.RHO_WATER * config.Hull.LENGTH * config.Hull.BEAM)
        start = -(config.Hull.LENGTH / 2 + 2.) * self.direction   # 충분히 멀리 둔 뒤 아래에서 간격을 맞춘다
        self.robot = Robot(position=(start[0], start[1],
                                     config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - draft))
        self.human = _floating_human()
        water = Water(waves=(), floating_bodies=list(_FLOATING), surface_visual=surface_visual)
        self.world = build_world(parts=[water, self.robot, self.human])
        model, data = self.world.model, self.world.data

        boat_q = model.jnt_qposadr[self.robot.joint_id]
        data.qpos[boat_q + 3:boat_q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for name, qpos in settled_victim_pose().items():
            address = model.jnt_qposadr[model.joint(name).id]
            data.qpos[address:address + 7] = qpos
        mujoco.mj_forward(model, data)

        self.boat_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == self.robot.body_id}
        self.victim_geoms = [g for g in range(model.ngeom) if model.geom_group[g] == config.Groups.PHYSICS
                             and model.geom_contype[g] and g not in self.boat_geoms]
        # 배를 진행 방향으로 옮겨 실제 최소 간격을 gap에 맞춘다. 다리처럼 비스듬히 물속으로 내려간
        # 형상은 옮긴 만큼 간격이 줄지 않으므로 오차가 1mm 안이 될 때까지 반복한다.
        for _ in range(50):
            error = self.gap() - gap
            if abs(error) < 1e-3:
                break
            data.qpos[boat_q:boat_q + 3] += error * self.direction
            mujoco.mj_forward(model, data)
        if abs(self.gap() - gap) > .01 or self.boat_contacts():
            raise RuntimeError('배를 사람과 떨어진 출발 위치에 놓지 못했습니다.')

        boat_mass = model.body_subtreemass[self.robot.body_id]
        victim_mass = sum(model.body_subtreemass[model.body(name).id] for name in _FLOATING)
        # 정지한 상대에게 완전 탄성 충돌했을 때 상대가 얻을 수 있는 최대 속도
        self.elastic_limit = 2 * boat_mass / (boat_mass + victim_mass) * self.speed
        self._dof = slice(model.jnt_dofadr[self.human.joint_id], model.jnt_dofadr[self.human.joint_id] + 3)
        self._initial = mujoco.MjData(model)
        mujoco.mj_copyData(self._initial, model, data)
        self.reset()

    def gap(self):
        return min_geom_distance(self.world.model, self.world.data, self.boat_geoms, self.victim_geoms)

    def boat_contacts(self):
        data = self.world.data
        return [c for c in data.contact[:data.ncon] if c.geom1 in self.boat_geoms or c.geom2 in self.boat_geoms]

    def reset(self):
        """출발 배치로 되돌린다. 기록도 초기화한다."""
        model, data = self.world.model, self.world.data
        mujoco.mj_copyData(data, model, self._initial)
        boat_v = model.jnt_dofadr[self.robot.joint_id]
        data.qvel[boat_v:boat_v + 3] = self.speed * self.direction
        self.robot.set_command(self.thrust, 0.)
        mujoco.mj_forward(model, data)
        self.human_start = data.xipos[self.human.body_id].copy()
        self.human_v0 = data.qvel[self._dof].copy()
        self.hit, self.max_penetration, self.human_dv, self._steps = False, 0., 0., 0

    def step(self):
        model, data = self.world.model, self.world.data
        advance_world(self.world)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError('충돌 시험의 물리 상태가 비정상입니다 (NaN/Inf).')
        contacts = self.boat_contacts()
        self.hit = self.hit or bool(contacts)
        for contact in contacts:
            self.max_penetration = max(self.max_penetration, -contact.dist)
        self._steps += 1
        if self._steps % self.DISTANCE_EVERY == 0:
            self.max_penetration = max(self.max_penetration, -self.gap())
        self.human_dv = max(self.human_dv, float(np.dot(data.qvel[self._dof] - self.human_v0, self.direction)))

    def readings(self):
        data = self.world.data
        contacts = self.boat_contacts()
        boat_v = self.world.model.jnt_dofadr[self.robot.joint_id]
        return dict(
            time=float(data.time),
            contacts=len(contacts),
            penetration=max([0.] + [-c.dist for c in contacts]),
            max_penetration=self.max_penetration,
            hit=self.hit,
            human_dv=self.human_dv,
            elastic_limit=self.elastic_limit,
            push=float(np.dot(data.xipos[self.human.body_id] - self.human_start, self.direction)),
            boat_speed=float(np.linalg.norm(data.qvel[boat_v:boat_v + 2])),
            warnings=int(sum(int(w.number) for w in data.warning)),
        )
