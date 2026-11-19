"""월미도 구조 흐름: 구조 로봇이 구명환을 싣고(배 뒤 짧은 줄) 가서 건네주고, 사람이 잡으면 해변 쪽으로 견인 (담당: 가영).

순서: APPROACH → DELIVER → GRAB → TOW → COMPLETE (실패 시 FAILED)
  APPROACH 해변 물가에서 출발해 사람 둘레를 돌아 해변 쪽을 보고 선 뒤, 선미를 사람 손 쪽으로 천천히 댄다
           (기존 RescueApproach, 목표점 = 사람 손). 배 뒤 줄 끝의 구명환이 사람 손 근처에 온다.
  DELIVER  동경 Robot 윈치로 줄을 PAYOUT_LENGTH까지 푼다 (delivery_ready).
  GRAB     사람이 손을 뻗어 가까운 구명환을 끌어당긴다(ReachAssist). 손·환 조건이 맞으면 HumanGrasp로 잡는다.
  TOW      사람을 해변 쪽으로 끌고 온다. 끝에서 정지·장력 해소를 확인하면 COMPLETE.

배의 접근·정지 제어는 기존 RescueApproach·PhaseManager, 윈치는 동경 Robot(winch=True)을 그대로 쓴다 (그 파일들은 바꾸지 않음).
구명환을 던지는 장치는 아직 없다 (동경 담당 예정). 지금은 배 뒤에 끌고 가다 줄을 풀어 건네준다.
"""

from dataclasses import dataclass, replace
import math

import mujoco
import numpy as np

from sim import config
from sim.frames import body_velocity_w
from sim.robot.phase_manager import ApproachSettings
from sim.robot.rescue_approach import RescueApproach, RescueApproachSettings


@dataclass(frozen=True)
class DeliverySettings:
    capture_radius: float = .2       # 경유점 포착 [m] (팀 기본 0.05)
    position_tolerance: float = .3   # 도착 위치 허용 [m] (팀 기본 0.2)
    reapproach_radius: float = .5    # 이보다 멀어지면 다시 접근 [m] (팀 기본 0.21)
    # 속도 (이 시나리오만, 팀 기본값은 그대로). 이동 최대 속도·최대 힘·후진 접근 속도.
    transit_speed: float = 1.0       # [m/s] (팀 기본 1.0)
    max_force: float = 60.0          # [N] 전진 힘 상한 (팀 기본 60, 추진기 합 최대 100)
    docking_speed: float = .3        # [m/s] 마지막 후진 (팀 기본 0.3)
    reach_radius: float = 1.6        # 사람이 손을 뻗어 구명환을 끌어당길 수 있는 거리 [m]
    reach_force: float = 15.0        # 끌어당기는 힘 상한 [N] (사람 팔 힘 근사, 시험값)
    tow_speed: float = .4            # 견인 중 배 목표 속도 상한 [m/s] (RescueMission과 같은 값)
    minimum_tow_progress: float = 2.0
    human_speed_tolerance: float = .05
    residual_tension: float = 10.0
    completion_hold_seconds: float = 1.0
    # 단계별 시뮬레이션 시간 제한 [s]
    approach_timeout: float = 200.0
    deliver_timeout: float = 20.0
    grab_timeout: float = 40.0
    tow_timeout: float = 200.0


class ReachAssist:
    """사람이 손을 뻗어 물에 뜬 구명환을 끌어당기는 동작의 근사 (사람 쪽 부품, 가영).

    켜져 있고 손–구명환 잡기점이 reach_radius 안이면, 구명환 잡기점에 손 쪽 힘을, 사람 손에 같은 크기 반대 힘을 준다
    (사람이 당긴 만큼 사람도 끌려가므로 운동량이 보존된다). 팔 관절·수영은 재현하지 않는다.
    """
    REQUIRES = {}

    def __init__(self, settings, *, hand='left'):
        self.settings = settings
        self.hand = hand
        self.active = False

    def build(self, spec):
        pass

    def bind(self, model):
        names = config.Names
        self.hand_site = model.site(names.HUMAN_GRIP_L_SITE if self.hand == 'left' else names.HUMAN_GRIP_R_SITE).id
        self.ring_site = model.site(names.LIFEBUOY_GRIP_L_SITE if self.hand == 'left' else names.LIFEBUOY_GRIP_R_SITE).id
        self.human_body = model.body(names.HUMAN_BODY).id
        self.ring_body = model.body(names.LIFEBUOY_BODY).id

    def gap(self, data):
        return float(np.linalg.norm(data.site_xpos[self.hand_site] - data.site_xpos[self.ring_site]))

    def add_forces(self, model, data, acc):
        if not self.active or self.gap(data) > self.settings.reach_radius:
            return
        hand, ring = data.site_xpos[self.hand_site].copy(), data.site_xpos[self.ring_site].copy()
        relative = np.empty(6)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, self.ring_site, relative, 0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, self.hand_site, velocity, 0)
        force = 40. * (hand - ring) - 25. * (relative[3:] - velocity[3:])
        size = np.linalg.norm(force)
        if size > self.settings.reach_force:
            force *= self.settings.reach_force / size
        acc.add(self.ring_body, force, point_w=ring)
        acc.add(self.human_body, -force, point_w=hand)


class WolmidoDelivery:
    """구명환을 실은 구조 로봇의 접근·전달·잡기·견인. step() 한 번 = 물리 한 스텝."""

    STAGES = ('APPROACH', 'DELIVER', 'GRAB', 'TOW', 'COMPLETE', 'FAILED')

    def __init__(self, world, robot, rope, grasp, reach, *, tow_heading, tow_distance, settings=None):
        self.world, self.robot, self.rope, self.grasp, self.reach = world, robot, rope, grasp, reach
        self.settings = DeliverySettings() if settings is None else settings
        self.tow_heading = float(tow_heading)
        self.direction = np.array([math.cos(tow_heading), math.sin(tow_heading)])
        self.tow_distance = float(tow_distance)
        model, data = world.model, world.data
        mujoco.mj_forward(model, data)
        self.human_body = model.body(config.Names.HUMAN_BODY).id
        self.ring_body = model.body(config.Names.LIFEBUOY_BODY).id
        self.stern_site = model.site(config.Names.TOW_POINT_SITE).id
        hand_name = model.site(reach.hand_site).name
        # 구명환을 1m 줄로 끌고 가면 감속할 때마다 환이 배를 당겨 5cm 안에 머물기 어렵다 (2026-11-19 측정: 경유점 0.1~0.3m에서
        # 140초 맴돎). 이 시나리오만 포착·도착 허용 범위를 넓힌다. 팀 기본값(ApproachSettings)은 바꾸지 않는다.
        navigation = replace(ApproachSettings(), capture_radius=self.settings.capture_radius,
                             position_tolerance=self.settings.position_tolerance,
                             reapproach_radius=self.settings.reapproach_radius,
                             max_speed=self.settings.transit_speed, max_force=self.settings.max_force)
        self.approach = RescueApproach(world, robot, tow_heading=tow_heading, target_site=hand_name,
                                       navigation=navigation,
                                       settings=replace(RescueApproachSettings(),
                                                        docking_speed=self.settings.docking_speed))
        self.controller, self.navigation = self.approach.controller, self.approach.navigation
        self.stage = 'APPROACH'
        self.stage_since = float(data.time)
        self.failure = None
        self.grab_time = None
        self._human_start = None
        self._complete_since = None
        self._last_good = mujoco.MjData(model)
        mujoco.mj_copyData(self._last_good, model, data)

    @property
    def failed(self):
        return self.stage == 'FAILED'

    @property
    def display_data(self):
        return self._last_good

    def _fail(self, message):
        self.stage, self.failure = 'FAILED', message
        self.robot.set_command(0, 0)
        self.reach.active = False

    def _enter(self, stage):
        self.stage, self.stage_since = stage, float(self.world.data.time)

    def readings(self):
        model, data = self.world.model, self.world.data
        human = data.xpos[self.human_body, :2]
        velocity, _ = body_velocity_w(model, data, self.human_body)
        progress = 0. if self._human_start is None else float((human - self._human_start) @ self.direction)
        return dict(stage=self.stage, time=float(data.time), delivery=self.robot.delivery_state,
                    rope_length=self.robot.rope_length(model), hand_ring_gap=self.reach.gap(data),
                    held=self.grasp.is_held(data), tension=self.rope.tension(model, data),
                    tow_progress=progress, human_speed=float(np.linalg.norm(velocity[:2])),
                    boat_target_distance=float(np.linalg.norm(self.controller.target_xy
                                                              - data.xpos[self.robot.body_id, :2])))

    def step(self):
        """한 스텝 진행. 실패 뒤에는 False를 돌려주고 물리를 진행하지 않는다."""
        if self.failed:
            return False
        model, data, s = self.world.model, self.world.data, self.settings
        mujoco.mj_forward(model, data)
        elapsed = data.time - self.stage_since
        limits = dict(APPROACH=s.approach_timeout, DELIVER=s.deliver_timeout, GRAB=s.grab_timeout, TOW=s.tow_timeout)
        if self.stage in limits and elapsed > limits[self.stage]:
            self._fail(f'{self.stage} 단계가 {limits[self.stage]:g}초를 넘었습니다.')
            return False
        if self.stage == 'APPROACH' and self.approach.arrived:
            self.robot.pay_out()
            self._enter('DELIVER')
        elif self.stage == 'DELIVER' and self.robot.delivery_ready:
            self.reach.active = True
            self._enter('GRAB')
        elif self.stage == 'GRAB' and self.grasp.try_grab(model, data, delivery_ready=self.robot.delivery_ready):
            self.reach.active = False
            self.grab_time = float(data.time)
            self._human_start = data.xpos[self.human_body, :2].copy()
            self.controller.target_xy[:] = data.xpos[self.robot.body_id, :2] + self.tow_distance * self.direction
            self.controller.settings = replace(self.navigation, max_speed=s.tow_speed,
                                               capture_radius=self.navigation.position_tolerance,
                                               reapproach_radius=max(.5, 2 * self.navigation.position_tolerance))
            self.controller.reset()
            self._enter('TOW')
        try:
            if self.stage == 'TOW' or self.stage == 'COMPLETE':
                self.approach.check_clearance()
                self.controller.step()   # 제어 갱신 + 외력 합산 + mj_step 한 번
            else:
                self.approach.step()     # 접근·건네주기·잡기 동안 선미 위치 유지 (내부에서 mj_step 한 번)
        except RuntimeError as error:
            self._fail(str(error))
            return False
        if data.warning.number.any() or not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc)):
            self._fail('물리 상태가 비정상입니다 (NaN·경고).')
            return False
        mujoco.mj_copyData(self._last_good, model, data)
        if self.stage in ('TOW', 'COMPLETE'):
            state = self.readings()
            done = (self.controller.arrived and state['tow_progress'] >= s.minimum_tow_progress
                    and state['human_speed'] <= s.human_speed_tolerance and state['tension'] <= s.residual_tension)
            if done:
                self._complete_since = self._complete_since if self._complete_since is not None else float(data.time)
                if data.time - self._complete_since >= s.completion_hold_seconds:
                    self.stage = 'COMPLETE'
            else:
                self._complete_since = None
                if self.stage == 'COMPLETE':
                    self.stage = 'TOW'
        return True
