"""구명환을 짧은 로프로 끌고 가서 줄을 풀어 건네주는 구조 시나리오.

접근(선미 정렬 → 후진) → 줄 풀기(건네주기) → 사람이 잡기 → 견인 → 정지.

2026-11-16 가영·동경 합의: 배가 구명환을 짧게 감은 로프로 끌고 가다가, 사람 앞에서 선미를 대고 멈춘 뒤
줄을 풀어 건네준다. 줄은 끊지 않고 그대로 견인에 쓴다.
기존 rescue_mission.py (환이 처음부터 사람에게 고정, 도착 후 로프 연결) 는 초기 견인 근사로 그대로 둔다.

- 배 쪽 (robot.py, 동경): 윈치 — 로프 길이 감기·풀기, delivery_ready 신호
- 받는 쪽 (grasp.py, 가영): HumanGrasp.try_grab(delivery_ready=...) 가 거리·속도를 확인해 손에 연결
- 로프 (tether.py, 가영): 처음부터 배 tow_point ↔ 환 lifebuoy_tow_point 에 연결

주의 (2026-11-16): 사람 모델은 혼자서는 뜨지 않는다 (가영 수정 예정). 그 전까지 이 시나리오는
extra_parts 로 시험용 부유 장치를 넣어야 끝까지 돈다.
"""

from dataclasses import dataclass, replace
from enum import Enum
import math

import mujoco
import numpy as np

from sim import config
from sim.frames import body_velocity_w
from sim.grasp import GraspSettings, HumanGrasp
from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.rescue_approach import RescueApproach, RescueApproachSettings
from sim.rescue_mission import RescueMissionSettings
from sim.robot import Robot
from sim.tether import Tether
from sim.watchdog import FailureCode, MissionWatchdog, WatchdogSettings
from sim.water import Water
from sim.world import build_world


class DeliveryStage(str, Enum):
    APPROACH = 'APPROACH'
    DELIVER = 'DELIVER'
    TOW = 'TOW'
    COMPLETE = 'COMPLETE'
    FAILED = 'FAILED'


# Watchdog 단계 시간 제한은 APPROACH / ATTACH / TOW 이름으로 걸린다. 건네주기 = 사람에게 연결되는 단계 → ATTACH
_WATCHDOG_STAGE = {DeliveryStage.DELIVER: 'ATTACH'}

# 끌려가는 환의 자세 (배 몸체 기준, w x y z): 환 로컬 축 (x = 구멍 축, y, z) → 배 (z_b, x_b, y_b).
# 환이 수면에 눕고, 잡는 점(Lifebuoy.GRIP_L_POINT_B)은 배에서 먼 쪽(사람 쪽), 로프 점은 배 쪽 테두리에 온다
TRAIL_QUAT_B = (0.5, -0.5, -0.5, -0.5)


@dataclass(frozen=True)
class DeliverySettings:
    """시연용 가정값. 거리 m, 속도 m/s."""

    # 건네줄 때 선미 로프 점 ↔ 사람 손 사이 간격. 사이에 환(바깥 지름 0.8 m)이 들어갈 자리
    delivery_gap: float = 1.1
    gap_tolerance: float = .15
    # 후진 직전 줄 펴기: 앞으로 최대 이 거리까지 가며, 환 로프 점이 선미 뒤 일직선(옆으로 이 값 이내)에 오면 멈춤.
    # 이유 (2026-11-16 시험): 선미 정렬 중 제자리 선회로 느슨한 줄의 환이 배 옆으로 밀려나, 후진해도 사람 앞에 안 옴
    straighten_distance: float = 2.0
    straighten_lateral: float = .15
    # 사람이 팔을 뻗어 환을 잡을 수 있는 거리 (손 site ↔ 환 잡는 점). HumanGrasp 기본 0.08 m 보다 넓게 둠:
    # 물에 뜬 사람이 손을 뻗는 동작은 아직 없어서, 이 범위 안에 오면 잡았다고 본다 (고정 잡기 근사)
    reach: float = .35
    grab_relative_speed: float = .25
    # 건넬 거리: 후진 중 선체 표면 ↔ 사람 표면 간격이 이 값 이하가 되면 그 자리에 멈춰 줄을 푼다.
    # 이유 (2026-11-17 동경): 배는 구명환을 건넬 만큼만 다가가고 사람에게 붙지 않는다.
    #   떠 있는 사람의 손을 끝까지 쫓으면 선체가 사람을 밀고, 밀린 손을 다시 쫓느라 멈추지 못했다.
    handover_clearance: float = .5

    def __post_init__(self):
        for value in vars(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('전달 설정은 유한한 양수여야 합니다.')


class DeliveryMission:
    def __init__(self, world, robot, grasp, rope, *, tow_heading=0., settings=None, delivery=None,
                 watchdog_settings=None):
        if not robot.winch:
            raise ValueError('winch=True 인 로봇이 필요합니다.')
        if rope not in world.parts or not rope.attached:
            raise ValueError('전달 시나리오의 로프는 처음부터 배와 환에 연결되어 있어야 합니다.')
        self.world, self.robot, self.grasp, self.rope = world, robot, grasp, rope
        self.settings = RescueMissionSettings() if settings is None else settings
        if self.settings.tow_distance <= config.Winch.PAYOUT_LENGTH + self.settings.minimum_tow_progress:
            raise ValueError('견인 거리는 푼 로프 길이와 최소 사람 진행 거리의 합보다 커야 합니다.')
        self.delivery = DeliverySettings() if delivery is None else delivery
        approach_settings = RescueApproachSettings(connection_gap=self.delivery.delivery_gap,
                                                   gap_tolerance=self.delivery.gap_tolerance)
        # 환은 배가 끌고 다니므로 다가갈 기준은 사람 손, 선체 간격 검사 대상은 사람만
        self.approach = RescueApproach(world, robot, tow_heading=tow_heading, settings=approach_settings,
                                       target_site=grasp.hand_name, victim_bodies=(config.Names.HUMAN_BODY,))
        self.controller = self.approach.controller
        self.human_body = world.model.body(config.Names.HUMAN_BODY).id
        self.watchdog = MissionWatchdog(watchdog_settings)
        self._initial = mujoco.MjData(world.model)
        mujoco.mj_copyData(self._initial, world.model, world.data)
        self.reset()

    # ── 상태 ─────────────────────────────────────────────────
    @property
    def failed(self):
        return self.watchdog.failure is not None

    @property
    def failure(self):
        return self.watchdog.failure

    @property
    def arrived(self):
        return self.stage == DeliveryStage.COMPLETE

    @property
    def display_data(self):
        return self.world.data

    @property
    def target_xy(self):
        return self.controller.target_xy

    @property
    def target_heading(self):
        return self.approach.target_heading

    @property
    def ring_site(self):
        return self.world.model.site(config.Names.LIFEBUOY_TOW_SITE).id

    def reset(self):
        """출발 배치(환은 짧은 로프로 선미 뒤에)로 되돌린다."""
        model, data = self.world.model, self.world.data
        mujoco.mj_copyData(data, model, self._initial)
        self.robot.reset_winch(model)
        self.grasp.reset(model, data)
        self.robot.set_command(0, 0)
        mujoco.mj_forward(model, data)
        self.approach.reset()
        self.stage = DeliveryStage.APPROACH
        self._straighten = 'PENDING'   # PENDING → ACTIVE → DONE (후진 직전 한 번)
        self.payout_time = self.grab_time = None
        self._human_start = None
        self._complete_since = None
        self.watchdog.reset(data.time, self.stage.value)

    def readings(self):
        model, data = self.world.model, self.world.data
        state = self.approach.readings()
        grab = self.grasp.metrics(model, data)
        progress = (0.0 if self._human_start is None else float(
            (data.xpos[self.human_body, :2] - self._human_start) @ self.approach.direction))
        velocity, _ = body_velocity_w(model, data, self.human_body)
        sub = 'STRAIGHTEN' if self._straighten == 'ACTIVE' else self.approach.stage.value
        phase = f'APPROACH / {sub}' if self.stage == DeliveryStage.APPROACH else self.stage.value
        state.update(phase=phase, winch=self.robot.delivery_state, rope_length=self.robot.rope_length(model),
                     held=grab['held'], hand_gap=grab['distance'], hand_relative_speed=grab['relative_speed'],
                     tension=self.rope.tension(model, data), tow_progress=progress,
                     human_speed=float(np.linalg.norm(velocity[:2])))
        if self.failed:
            state.update(failure_code=self.failure.code.value, failure_message=self.failure.message,
                         failure_time=self.failure.time)
        return state

    # ── 한 스텝 ──────────────────────────────────────────────
    def _watch_stage(self):
        return _WATCHDOG_STAGE.get(self.stage, self.stage.value)

    def _stop(self):
        self.stage = DeliveryStage.FAILED
        self.robot.set_command(0, 0)
        self.controller.forward = self.controller.steering = 0.
        return False

    def step(self):
        """제어 + 물리 한 스텝. 실패하면 False (명령 0 으로 멈춤)."""
        if self.failed:
            self.robot.set_command(0, 0)
            return False
        world = self.world
        if self.watchdog.check(world.model, world.data, self.robot.body_id, self._watch_stage()):
            return self._stop()
        try:
            self._step_active()
        except RuntimeError as error:
            self.watchdog.trip(FailureCode.CONTROL_ERROR, world.data.time, self._watch_stage(), str(error))
            return self._stop()
        if self.watchdog.check(world.model, world.data, self.robot.body_id, self._watch_stage()):
            return self._stop()
        return True

    # ── 후진 직전 줄 펴기 ─────────────────────────────────────
    def _ring_behind_stern(self):
        """(환 로프 점의 선미 뒤 거리, 환 중심의 옆 거리) [m], 배 몸체 기준. 뒤 거리 > 0 이면 선미 뒤.

        옆 거리는 로프 점(테두리)이 아니라 환 중심으로 잰다: 테두리만 일직선이고 환은 옆에 있을 수 있음.
        """
        data = self.world.data
        rotation = data.xmat[self.robot.body_id].reshape(3, 3)
        stern = data.site_xpos[self.robot.tow_site_id]
        rope_end = rotation.T @ (data.site_xpos[self.ring_site] - stern)
        center = rotation.T @ (data.xpos[self.world.model.site_bodyid[self.ring_site]] - stern)
        return float(-rope_end[0]), float(abs(center[1]))

    def _begin_straighten(self):
        """선미 정렬을 마친 자리에서 지금 방향 그대로 앞으로 가 줄을 팽팽하게 편다."""
        data = self.world.data
        self._straighten = 'ACTIVE'
        self._straighten_target = (data.xpos[self.robot.body_id, :2]
                                   + self.delivery.straighten_distance * self.approach.direction)
        self.controller.target_xy[:] = self._straighten_target
        self.controller.reset()

    def _step_straighten(self):
        behind, lateral = self._ring_behind_stern()
        taut = behind >= .8 * self.robot.rope_length(self.world.model)
        if (taut and lateral <= self.delivery.straighten_lateral) or self.controller.arrived:
            self._straighten = 'DONE'   # 다시 접근(후진)으로. 목표는 approach.step 이 사람 손 기준으로 다시 잡음
            self.controller.reset()
            self.approach._ready_since = None
            self.approach.step()
            return
        self.controller.target_xy[:] = self._straighten_target
        self.controller.step()

    def _hold_here(self):
        """지금 위치·견인 방향으로 목표를 고정한다. 이후 사람이 움직여도 쫓아가지 않는다."""
        data = self.world.data
        self.controller.target_xy[:] = data.xpos[self.robot.body_id, :2]
        self.controller.settings = replace(self.approach.navigation, max_speed=self.approach.settings.docking_speed)
        self.controller.reset()

    def _start_tow(self):
        data = self.world.data
        self.grab_time = float(data.time)
        self._human_start = data.xpos[self.human_body, :2].copy()
        self.controller.target_xy[:] = (data.xpos[self.robot.body_id, :2]
                                        + self.settings.tow_distance * self.approach.direction)
        self.controller.settings = replace(
            self.approach.navigation, max_speed=self.settings.tow_speed,
            capture_radius=self.approach.navigation.position_tolerance,
            reapproach_radius=max(.5, 2 * self.approach.navigation.position_tolerance))
        self.controller.reset()
        self.stage = DeliveryStage.TOW

    def _step_active(self):
        model, data = self.world.model, self.world.data
        mujoco.mj_forward(model, data)
        if self.stage == DeliveryStage.APPROACH:
            if self._straighten == 'ACTIVE':
                self._step_straighten()
                return
            self.approach.step()   # 안에서 controller.step() → 물리 한 스텝
            if self.approach.stage.value == 'BACK_IN' and self._straighten == 'PENDING':
                self._begin_straighten()
                return
            at_handover = (self._straighten == 'DONE'
                           and self.approach.readings()['clearance'] <= self.delivery.handover_clearance)
            if at_handover or self.approach.arrived:   # 건넬 거리 → 그 자리 유지하며 줄 풀기 시작
                self._hold_here()
                self.robot.pay_out()
                self.payout_time = float(data.time)
                self.stage = DeliveryStage.DELIVER
            return
        self.approach.check_clearance()
        if self.stage == DeliveryStage.DELIVER:
            # 배는 그 자리 유지. 줄을 다 풀면(신호 True) 손 근처·저속일 때 사람이 잡음
            if self.grasp.try_grab(model, data, delivery_ready=self.robot.delivery_ready):
                mujoco.mj_forward(model, data)
                self._start_tow()
        self.controller.step()
        if self.stage in (DeliveryStage.TOW, DeliveryStage.COMPLETE):
            state = self.readings()
            complete = (self.controller.arrived and state['held']
                        and state['tow_progress'] >= self.settings.minimum_tow_progress
                        and state['human_speed'] <= self.settings.human_speed_tolerance
                        and state['tension'] <= self.settings.residual_tension)
            if complete:
                if self._complete_since is None:
                    self._complete_since = float(data.time)
                if data.time - self._complete_since >= self.settings.completion_hold_seconds:
                    self.stage = DeliveryStage.COMPLETE
            else:
                self._complete_since = None
                if self.stage == DeliveryStage.COMPLETE:
                    self.stage = DeliveryStage.TOW


def trail_lifebuoy(model, data, robot):
    """환을 선미 뒤 수면에 눕혀, 감아 둔 로프(STOW_LENGTH)보다 5 cm 느슨한 자리에 놓는다 (시작 배치).

    배 위치·방향을 정한 뒤 부른다. qpos0 도 같이 바꿔 리셋해도 같은 배치로 시작한다.
    """
    joint = model.joint(config.Names.LIFEBUOY_FREE).id
    q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    boat_pos = data.qpos[robot.qpos_adr:robot.qpos_adr + 3]
    boat_quat = data.qpos[robot.qpos_adr + 3:robot.qpos_adr + 7]
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, boat_quat, np.asarray(TRAIL_QUAT_B))
    # 환 로프 점 = 환 중심 + (R, 0, 튜브) [배 기준]. 이 점을 선미 로프 점 뒤 (STOW − 0.05) m 에
    ring = config.Lifebuoy
    tow_x = config.Hull.TOW_POINT_B[0] - (config.Winch.STOW_LENGTH - .05)
    center_b = np.array([tow_x - ring.CENTERLINE_RADIUS, 0., 0.])
    offset = np.empty(3)
    mujoco.mju_rotVecQuat(offset, center_b, boat_quat)
    position = boat_pos + offset
    position[2] = config.WaterParams.SURFACE_Z + .03   # 가벼운 환은 높게 뜬다
    for target in (data.qpos, model.qpos0):
        target[q:q + 3] = position
        target[q + 3:q + 7] = quat
    data.qvel[v:v + 6] = 0
    mujoco.mj_forward(model, data)


def build_delivery_mission(*, start=(6., 3.), victim=(0., 0.), yaw=0., tow_heading=0., tow_distance=6.,
                           hand='left', delivery=None, watchdog_settings=None, extra_parts=()):
    """짧은 로프로 구명환을 끄는 배·물에 뜬 사람으로 전달 시나리오를 만든다. 각도 rad.

    extra_parts: 시험 장치 (예: 사람 부유 수정 전 임시 부유 장치). 사람 다음에 조립한다.
    """
    start, victim = np.asarray(start, dtype=float), np.asarray(victim, dtype=float)
    if (start.shape != (2,) or victim.shape != (2,) or not np.isfinite(start).all()
            or not np.isfinite(victim).all() or not math.isfinite(yaw) or not math.isfinite(tow_heading)):
        raise ValueError('출발·사람 위치와 방향은 유한한 값이어야 합니다.')
    delivery = DeliverySettings() if delivery is None else delivery
    draft = config.Hull.MASS / (config.Physics.RHO_WATER * config.Hull.LENGTH * config.Hull.BEAM)
    robot = Robot(position=(*start, config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - draft),
                  tow_point_b=config.Hull.TOW_POINT_B, winch=True)
    human = Human(position=(*victim, config.WaterParams.SURFACE_Z - .03), grip_sites=True)
    water = Water(waves=(), floating_bodies=[config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY],
                  surface_visual=True)
    ring = Lifebuoy(attached=False, grip_sites=True)
    rope = Tether(enabled=True, attached=True)
    grasp = HumanGrasp(hand=hand, settings=GraspSettings(max_distance=delivery.reach,
                                                         max_relative_speed=delivery.grab_relative_speed))
    world = build_world(parts=[water, robot, human, *extra_parts, ring, rope, grasp])
    model, data = world.model, world.data
    address = model.jnt_qposadr[robot.joint_id]
    quaternion = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    model.qpos0[address + 3:address + 7] = quaternion
    data.qpos[address + 3:address + 7] = quaternion
    mujoco.mj_forward(model, data)
    trail_lifebuoy(model, data, robot)   # 배 방향을 정한 뒤 환을 선미 뒤에
    if watchdog_settings is None:
        # 줄을 푸는 시간(4 s)과 환이 손 근처에서 잦아들 시간을 둔다 (기본 ATTACH 5 s 는 로프 연결용)
        watchdog_settings = WatchdogSettings(attach_timeout=30.)
    settings = RescueMissionSettings(tow_distance=tow_distance)
    return DeliveryMission(world, robot, grasp, rope, tow_heading=tow_heading, settings=settings,
                           delivery=delivery, watchdog_settings=watchdog_settings)
