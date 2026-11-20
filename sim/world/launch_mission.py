"""구명환 발사 구조 시나리오: 접근 → 선미를 사람 쪽으로 정렬 → 발사 → 사람이 잡기 → 견인 → 정지.

2026-11-19 결정 (동경): 줄을 풀어 건네는 대신 배 홈 위 발사대에서 구명환을 쏜다.
- 배 쪽 (robot/launcher.py, robot.py 윈치, 동경): 발사대 고정·겨누기·발사, 발사 순간 윈치 자유 풀림
- 받는 쪽 (grasp.py, 가영): HumanGrasp.try_grab(delivery_ready=...) 가 거리·속도를 확인해 손에 연결
- 로프 (tether.py, 가영): 처음부터 배 tow_point ↔ 환 lifebuoy_tow_point 에 연결 (발사 전엔 느슨)
기존 rescue_mission.py (환을 처음부터 사람이 착용, 도착 후 로프 연결) 는 그대로 둔다.

근사: 공기 저항·바람 없음. 사람이 팔을 뻗는 동작은 없고, 잡는 점이 손에서 reach 안이고 느리면 잡았다고 본다.
"""

from dataclasses import dataclass, replace
from enum import Enum
import math

import numpy as np

from sim import config
from sim.frames import body_velocity_w
from sim.robot.launcher import Launcher, LOADED
from sim.robot.phase_manager import ApproachSettings
from sim.robot.rescue_approach import RescueApproach, RescueApproachSettings, RescueStage
from sim.world.rescue_mission import RescueMissionSettings
from sim.world.watchdog import FailureCode, MissionWatchdog, WatchdogSettings


class LaunchStage(str, Enum):
    APPROACH = 'APPROACH'
    LAUNCH = 'LAUNCH'      # 발사 후 환이 물에 떨어질 때까지
    GRAB = 'GRAB'          # 떨어진 환을 사람이 잡을 때까지
    TOW = 'TOW'
    COMPLETE = 'COMPLETE'
    FAILED = 'FAILED'


# Watchdog 단계 시간 제한은 APPROACH / ATTACH / TOW 이름으로 걸린다. 발사·잡기 = 사람에게 연결되는 단계 → ATTACH
_WATCHDOG_STAGE = {LaunchStage.LAUNCH: 'ATTACH', LaunchStage.GRAB: 'ATTACH'}


@dataclass(frozen=True)
class LaunchSettings:
    """시연용 가정값. 거리 m, 속도 m/s."""

    # 사람이 팔을 뻗어 환을 잡을 수 있는 거리 (손 site ↔ 환 잡는 점). HumanGrasp 기본 0.08 m 보다 넓게:
    # 팔 뻗는 동작이 아직 없어서 이 범위 안에 오면 잡았다고 본다 (고정 잡기 근사). 실험(2026-11-19) 기준
    reach: float = .45
    grab_relative_speed: float = .25
    # 물에 닿았다고 보는 환 중심 높이 (수면 위) [m]
    landing_height: float = .06
    # 발사 자리에 멈추는 허용 오차. 발사대가 손을 향해 다시 겨누므로 5 cm 정밀 정지가 필요 없다.
    # 이유 (2026-11-19 측정): 5 cm·8°·1 초 유지로는 정렬에만 14~19 초. 아래 값으로 발사 12~26 초, 착수 오차 그대로
    # [실험값 — 11/23 회의에서 확정]
    stop_radius: float = .3
    stop_tolerance: float = .4
    stop_heading_deg: float = 15.
    stop_hold_seconds: float = .2

    def __post_init__(self):
        for value in vars(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('발사 설정은 유한한 양수여야 합니다.')


class LaunchMission:
    def __init__(self, world, robot, launcher, grasp, rope, *, tow_heading=0., settings=None, launch=None,
                 watchdog_settings=None):
        import mujoco

        if not robot.winch:
            raise ValueError('winch=True 인 로봇이 필요합니다.')
        if rope not in world.parts or not rope.attached or launcher not in world.parts:
            raise ValueError('발사 시나리오의 로프는 처음부터 연결, 발사대는 조립되어 있어야 합니다.')
        self.world, self.robot, self.launcher, self.grasp, self.rope = world, robot, launcher, grasp, rope
        # 견인 속도 상한 0.4 → 0.7 m/s (2026-11-19 측정: 사람 실제 속도 최고 0.23 m/s, 최대 장력 158 N,
        # 머리 높이 변화는 0.4 와 0.3 cm 차이). 1.0 이면 장력 221 N. [실험값 — 11/23 회의에서 확정]
        self.settings = RescueMissionSettings(tow_distance=10., tow_speed=.7) if settings is None else settings
        self.launch = LaunchSettings() if launch is None else launch
        launch_nav = self.launch
        navigation = replace(ApproachSettings(), capture_radius=launch_nav.stop_radius,
                             position_tolerance=launch_nav.stop_tolerance,
                             reapproach_radius=launch_nav.stop_tolerance + .3,
                             heading_tolerance=math.radians(launch_nav.stop_heading_deg),
                             speed_tolerance=.1, yaw_speed_tolerance=.1,
                             arrival_hold_seconds=launch_nav.stop_hold_seconds)
        approach = RescueApproachSettings(staging_gap=config.Launcher.STAGING_GAP,
                                          connection_gap=config.Launcher.STAGING_GAP - 1.)
        # 환은 배에 실려 있으므로 다가갈 기준은 사람 손, 선체 간격 검사 대상은 사람만
        self.approach = RescueApproach(world, robot, tow_heading=tow_heading, settings=approach, navigation=navigation,
                                       target_site=grasp.hand_name, victim_bodies=(config.Names.HUMAN_BODY,))
        self.controller = self.approach.controller
        self.human_body = world.model.body(config.Names.HUMAN_BODY).id
        self.watchdog = MissionWatchdog(WatchdogSettings(attach_timeout=20.)
                                        if watchdog_settings is None else watchdog_settings)
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
        return self.stage == LaunchStage.COMPLETE

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

    @property
    def minimum_observed_clearance(self):
        return self.approach.minimum_observed_clearance

    def reset(self):
        """출발 배치(환은 발사대에 고정)로 되돌린다."""
        import mujoco

        model, data = self.world.model, self.world.data
        mujoco.mj_copyData(data, model, self._initial)
        self.robot.reset_winch(model)
        self.robot.reset_inversion(model)
        self.launcher.reset(model, data)
        self.grasp.reset(model, data)
        self.robot.set_command(0, 0)
        mujoco.mj_forward(model, data)
        self.approach.reset()
        self.stage = LaunchStage.APPROACH
        self.launch_time = self.landing = self.grab_time = None
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
        phase = f'APPROACH / {self.approach.stage.value}' if self.stage == LaunchStage.APPROACH else self.stage.value
        state.update(phase=phase, launcher=self.launcher.state, rope_length=self.robot.rope_length(model),
                     held=grab['held'], hand_gap=grab['distance'], hand_relative_speed=grab['relative_speed'],
                     tension=self.rope.tension(model, data), tow_progress=progress,
                     human_speed=float(np.linalg.norm(velocity[:2])),
                     shot=self.launcher.shot, landing=self.landing, inverted=self.robot.inverted)
        if self.failed:
            state.update(failure_code=self.failure.code.value, failure_message=self.failure.message,
                         failure_time=self.failure.time)
        return state

    # ── 한 스텝 ──────────────────────────────────────────────
    def _watch_stage(self):
        return _WATCHDOG_STAGE.get(self.stage, self.stage.value)

    def _stop(self):
        self.stage = LaunchStage.FAILED
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

    def _hold_here(self):
        """지금 위치·견인 방향으로 목표를 고정한다. 발사 반동이나 사람 움직임을 쫓아가지 않는다."""
        data = self.world.data
        self.controller.target_xy[:] = data.xipos[self.robot.body_id, :2]
        self.controller.settings = replace(self.approach.navigation, max_speed=self.approach.settings.docking_speed)
        self.controller.reset()

    def _fire(self):
        model, data = self.world.model, self.world.data
        self._hold_here()
        hand = data.site_xpos[self.grasp.hand_site]
        velocity, distance, aim = self.launcher.aim(data, hand)
        self.launcher.fire(model, data, velocity)
        self.robot.release_winch(distance + config.Launcher.ROPE_SLACK)
        self.launch_time = float(data.time)
        self.launcher.shot = dict(time=self.launch_time, distance=distance,
                                  speed=float(np.linalg.norm(velocity)), aim=aim.copy())
        self.stage = LaunchStage.LAUNCH

    def _check_landing(self):
        data = self.world.data
        ring = self.launcher.ring_body
        velocity, _ = body_velocity_w(self.world.model, data, ring)
        if velocity[2] < 0 and data.xpos[ring, 2] <= config.WaterParams.SURFACE_Z + self.launch.landing_height:
            self.landing = dict(time=float(data.time),
                                grip_to_hand=self.grasp.metrics(self.world.model, data)['distance'],
                                impact_speed=float(np.linalg.norm(velocity)))
            self.stage = LaunchStage.GRAB

    def _start_tow(self):
        data = self.world.data
        self.grab_time = float(data.time)
        self._human_start = data.xpos[self.human_body, :2].copy()
        self.controller.target_xy[:] = (data.xipos[self.robot.body_id, :2]
                                        + self.settings.tow_distance * self.approach.direction)
        self.controller.settings = replace(
            self.approach.navigation, max_speed=self.settings.tow_speed,
            capture_radius=self.approach.navigation.position_tolerance,
            reapproach_radius=max(.5, 2 * self.approach.navigation.position_tolerance))
        self.controller.reset()
        self.stage = LaunchStage.TOW

    def _step_active(self):
        import mujoco

        model, data = self.world.model, self.world.data
        mujoco.mj_forward(model, data)
        if self.stage == LaunchStage.APPROACH:
            # 선미 정렬(ALIGN_STERN)이 끝나 후진(BACK_IN)으로 넘어가는 순간 = 발사 자리. 후진하지 않고 쏜다
            if self.approach.stage == RescueStage.BACK_IN and self.launcher.state == LOADED:
                self._fire()
            else:
                self.approach.step()   # 안에서 controller.step() → 물리 한 스텝
                return
        self.approach.check_clearance()
        if self.stage == LaunchStage.LAUNCH:
            self._check_landing()
        if self.stage == LaunchStage.GRAB:
            if self.grasp.try_grab(model, data, delivery_ready=self.robot.delivery_ready):
                mujoco.mj_forward(model, data)
                self._start_tow()
        self.controller.step()
        if self.stage in (LaunchStage.TOW, LaunchStage.COMPLETE):
            state = self.readings()
            complete = (self.controller.arrived and state['held']
                        and state['tow_progress'] >= self.settings.minimum_tow_progress
                        and state['human_speed'] <= self.settings.human_speed_tolerance
                        and state['tension'] <= self.settings.residual_tension)
            if complete:
                if self._complete_since is None:
                    self._complete_since = float(data.time)
                if data.time - self._complete_since >= self.settings.completion_hold_seconds:
                    self.stage = LaunchStage.COMPLETE
            else:
                self._complete_since = None
                if self.stage == LaunchStage.COMPLETE:
                    self.stage = LaunchStage.TOW


def build_launch_mission(*, start=(8., 3.), victim=(0., 0.), yaw=0., tow_heading=0., tow_distance=10.,
                         hand='left', launch=None, watchdog_settings=None):
    """구명환을 발사대에 실은 배·물에 뜬 사람으로 발사 시나리오를 만든다. 각도 rad."""
    import mujoco

    from sim.robot.robot import Robot
    from sim.water.water import Water
    from sim.world.grasp import GraspSettings, HumanGrasp
    from sim.world.human import Human
    from sim.world.lifebuoy import Lifebuoy
    from sim.world.tether import Tether
    from sim.world.world import build_world

    start, victim = np.asarray(start, dtype=float), np.asarray(victim, dtype=float)
    if (start.shape != (2,) or victim.shape != (2,) or not np.isfinite(start).all()
            or not np.isfinite(victim).all() or not math.isfinite(yaw) or not math.isfinite(tow_heading)):
        raise ValueError('출발·사람 위치와 방향은 유한한 값이어야 합니다.')
    launch = LaunchSettings() if launch is None else launch
    height = config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - config.Expect.DRAFT
    # 배 방향은 build 전에 정해야 발사대가 환을 배 기준 자리에 놓는다
    robot = Robot(position=(*start, height), yaw=yaw, winch=True)
    human = Human(position=(*victim, config.WaterParams.SURFACE_Z - .03), grip_sites=True)
    water = Water(waves=(), floating_bodies=[config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY],
                  surface_visual=True)
    ring = Lifebuoy(attached=False, grip_sites=True)
    rope = Tether(enabled=True, attached=True)
    grasp = HumanGrasp(hand=hand, settings=GraspSettings(max_distance=launch.reach,
                                                         max_relative_speed=launch.grab_relative_speed))
    launcher = Launcher(ring_site=grasp.ring_name)
    world = build_world(parts=[water, robot, human, ring, rope, grasp, launcher])
    mujoco.mj_forward(world.model, world.data)
    settings = RescueMissionSettings(tow_distance=tow_distance, tow_speed=.7)
    return LaunchMission(world, robot, launcher, grasp, rope, tow_heading=tow_heading, settings=settings,
                         launch=launch, watchdog_settings=watchdog_settings)
