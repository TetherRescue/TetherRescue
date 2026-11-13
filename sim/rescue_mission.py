"""구명환 접근 → 로프 연결 → 견인 → 정지 유지의 시험 시나리오."""

from dataclasses import dataclass, replace
from enum import Enum
import math

import numpy as np

from sim import config
from sim.frames import body_velocity_w
from sim.rescue_approach import RescueApproach, RescueStage
from sim.watchdog import MissionWatchdog, FailureCode


class MissionStage(str, Enum):
    APPROACH = 'APPROACH'
    ATTACH = 'ATTACH'
    TOW = 'TOW'
    COMPLETE = 'COMPLETE'
    FAILED = 'FAILED'


@dataclass(frozen=True)
class RescueMissionSettings:
    tow_distance: float = 6.0  # 연결 순간의 선체 원점에서 견인 방향으로 이동할 거리
    tow_speed: float = .4
    minimum_tow_progress: float = 2.0  # 사람의 실제 진행 거리가 이 이상이어야 완료
    human_speed_tolerance: float = .05
    residual_tension: float = 10.0
    completion_hold_seconds: float = 1.0

    def __post_init__(self):
        for value in vars(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('견인 설정은 유한한 양수여야 합니다.')


class RescueMission:
    def __init__(self, world, robot, rope, *, tow_heading=0., settings=None, approach_settings=None, watchdog_settings=None):
        if (rope not in world.parts or not rope.enabled or rope.tendon_id is None
                or rope.attached or rope.initially_attached):
            raise ValueError('처음부터 미연결인 tendon을 생성·bind해 전달하세요.')
        self.world, self.robot, self.rope = world, robot, rope
        self.settings = RescueMissionSettings() if settings is None else settings
        rest = world.model.tendon_lengthspring[rope.tendon_id, 1]
        if self.settings.tow_distance <= rest + self.settings.minimum_tow_progress:
            raise ValueError('견인 거리는 로프 기준 길이와 최소 사람 진행 거리의 합보다 커야 합니다.')
        self.approach = RescueApproach(world, robot, tow_heading=tow_heading, settings=approach_settings)
        self.controller = self.approach.controller
        self.human_body = world.model.body(config.Names.HUMAN_BODY).id
        self.stage = MissionStage.APPROACH
        self.attachment_time = None
        self._human_start = None
        self._complete_since = None
        self._last_time = float(world.data.time)
        self.watchdog = MissionWatchdog(watchdog_settings)
        self.watchdog.reset(world.data.time, self.stage.value)
        self._failure_display = None
        self._save_valid_state()

    @property
    def failed(self):
        return self.watchdog.failure is not None

    @property
    def failure(self):
        return self.watchdog.failure

    @property
    def display_data(self):
        return self._failure_display if self.failed else self.world.data

    @property
    def arrived(self):
        return self.stage == MissionStage.COMPLETE

    @property
    def target_xy(self):
        return self.controller.target_xy

    @property
    def target_heading(self):
        return self.approach.target_heading

    @property
    def ring_site(self):
        return self.approach.ring_site

    @property
    def minimum_observed_clearance(self):
        return self.approach.minimum_observed_clearance

    def reset(self):
        """월드가 출발 배치로 리셋된 뒤 모델의 로프 계수도 미연결 상태로 복원한다."""
        if MissionWatchdog.invalid_state(self.world.data):
            raise ValueError('먼저 mj_resetData로 정상 출발 상태를 복원하세요.')
        self.rope.reset(self.world.model)
        self.approach.reset()
        self.stage = MissionStage.APPROACH
        self.attachment_time = None
        self._human_start = None
        self._complete_since = None
        self._last_time = float(self.world.data.time)
        self.watchdog.reset(self.world.data.time, self.stage.value)
        self._failure_display = None
        self._save_valid_state()

    def readings(self):
        if self.failed:
            state = self._last_readings.copy()
            state.update(phase=self.stage.value, failure_code=self.failure.code.value,
                         failure_message=self.failure.message, failure_time=self.failure.time,
                         attached=self.rope.attached, forward=0., steering=0.)
            return state
        state = self.approach.readings()
        progress = (0.0 if self._human_start is None else float(
            (self.world.data.xpos[self.human_body, :2] - self._human_start) @ self.approach.direction))
        velocity, _ = body_velocity_w(self.world.model, self.world.data, self.human_body)
        state.update(phase=(f'APPROACH / {self.approach.stage.value}'
                            if self.stage == MissionStage.APPROACH else self.stage.value),
                     attached=self.rope.attached, tension=self.rope.tension(self.world.model, self.world.data),
                     tow_progress=progress, human_speed=float(np.linalg.norm(velocity[:2])))
        return state

    def _can_attach(self):
        state = self.approach.readings()
        nav, docking = self.approach.navigation, self.approach.settings
        return (self.controller.arrived
                and state['distance'] <= nav.position_tolerance
                and state['speed'] <= nav.speed_tolerance
                and abs(state['heading_error']) <= nav.heading_tolerance
                and abs(state['yaw_speed']) <= nav.yaw_speed_tolerance
                and abs(state['endpoint_gap'] - docking.connection_gap) <= docking.gap_tolerance
                and state['relative_speed'] <= docking.relative_speed_tolerance)

    def _save_valid_state(self):
        data = self.world.data
        self._valid_pose = (data.qpos.copy(), data.act.copy(), float(data.time))
        self._last_readings = self.readings()

    def _stop(self):
        """추진·외력을 비우고 실패를 고정한다. 실제 물리 상태는 보정하지 않는다."""
        import mujoco

        self.stage = MissionStage.FAILED
        self.robot.set_command(0, 0)
        self.controller.forward = self.controller.steering = 0.
        self.world.acc.reset()
        self.world.acc.apply(self.world.model, self.world.data)
        # 수치 오류가 난 실제 data를 렌더러에 전달하지 않는다.
        if self.failure.code in (FailureCode.CAPSIZE, FailureCode.STAGE_TIMEOUT,
                                 FailureCode.MISSION_TIMEOUT):
            data = self.world.data
            pose = (data.qpos, data.act, data.time)
        else:
            pose = self._valid_pose
        display = mujoco.MjData(self.world.model)
        display.qpos[:] = pose[0]
        display.act[:] = pose[1]
        display.time = pose[2]
        mujoco.mj_forward(self.world.model, display)
        self._failure_display = display
        return False

    def step(self):
        import mujoco

        if self.failed:
            self.robot.set_command(0, 0)
            return False
        world = self.world
        # 건강한 실행의 명시적 초기 배치 리셋만 자동으로 감지한다.
        if (math.isfinite(world.data.time) and world.data.time < self._last_time - 1e-12
                and world.data.time == 0 and np.array_equal(world.data.qpos, world.model.qpos0)
                and MissionWatchdog.invalid_state(world.data) is None):
            self.reset()
        if self.watchdog.check(world.model, world.data, self.robot.body_id, self.stage.value):
            return self._stop()
        try:
            self._step_active()
        except RuntimeError as error:
            self.watchdog.trip(FailureCode.CONTROL_ERROR, world.data.time, self.stage.value, str(error))
            return self._stop()
        if self.watchdog.check(world.model, world.data, self.robot.body_id, self.stage.value):
            return self._stop()
        self._save_valid_state()
        return True

    def _step_active(self):
        import mujoco

        world = self.world
        mujoco.mj_forward(world.model, world.data)
        if self.stage == MissionStage.APPROACH:
            self.approach.step()
            if self.approach.arrived:
                self.stage = MissionStage.ATTACH
        else:
            self.approach.check_clearance()
            if self.stage == MissionStage.ATTACH:
                # READY 이후 상태가 바뀌었다면 이전 판정만으로 연결하지 않는다.
                if not self._can_attach():
                    self.stage = MissionStage.APPROACH
                    self.approach.stage = RescueStage.BACK_IN
                    self.approach._ready_since = None
                    self.controller.reset()
                    self.approach.step()
                    self._last_time = float(world.data.time)
                    return
                self.rope.attach(world.model)
                self.attachment_time = float(world.data.time)
                self._human_start = world.data.xpos[self.human_body, :2].copy()
                self.controller.target_xy[:] = (world.data.xpos[self.robot.body_id, :2]
                                                + self.settings.tow_distance * self.approach.direction)
                # 견인 중에는 20cm 허용 범위에서 일찍 정렬한다. 마지막 5cm를
                # 추적하며 선미가 크게 회전하면 뒤따르는 사람과 간격이 줄 수 있다.
                self.controller.settings = replace(
                    self.approach.navigation, max_speed=self.settings.tow_speed,
                    capture_radius=self.approach.navigation.position_tolerance,
                    reapproach_radius=max(.5, 2 * self.approach.navigation.position_tolerance))
                self.controller.reset()
                self.stage = MissionStage.TOW
            self.controller.step()
            state = self.readings()
            complete = (self.controller.arrived and self.rope.attached
                        and state['tow_progress'] >= self.settings.minimum_tow_progress
                        and state['human_speed'] <= self.settings.human_speed_tolerance
                        and state['tension'] <= self.settings.residual_tension)
            if complete:
                if self._complete_since is None:
                    self._complete_since = float(world.data.time)
                if world.data.time - self._complete_since >= self.settings.completion_hold_seconds:
                    self.stage = MissionStage.COMPLETE
            else:
                self._complete_since = None
                if self.stage == MissionStage.COMPLETE:
                    self.stage = MissionStage.TOW
        self._last_time = float(world.data.time)
