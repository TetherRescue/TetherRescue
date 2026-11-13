"""구명환의 현재 연결점으로 접근해 선미를 정렬하는 독립 시험 제어."""

from dataclasses import dataclass, replace
from enum import Enum
import math

import numpy as np

from sim import config
from sim.phase_manager import PhaseManager, ApproachSettings


class RescueStage(str, Enum):
    TRANSIT = 'TRANSIT'
    ALIGN_STERN = 'ALIGN_STERN'
    BACK_IN = 'BACK_IN'
    READY = 'READY'


@dataclass(frozen=True)
class RescueApproachSettings:
    # 모두 시연용 가정값. route_radius는 선체가 방향을 바꿀 공간을 포함한다.
    route_radius: float = 4.5
    staging_gap: float = 3.0
    connection_gap: float = .7
    waypoint_tolerance: float = .5
    docking_speed: float = .2
    gap_tolerance: float = .15
    relative_speed_tolerance: float = .06
    ready_hold_seconds: float = 1.0
    minimum_clearance: float = .1

    def __post_init__(self):
        for value in vars(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('구조 접근 설정은 유한한 양수여야 합니다.')
        if self.connection_gap <= self.gap_tolerance or self.staging_gap <= self.connection_gap:
            raise ValueError('연결 간격·허용 오차·대기 간격을 확인하세요.')


class RescueApproach:
    """바깥 경로 → 선미 정렬 → 저속 후진 → 연결 대기. 로프는 연결하지 않는다.

    목표 위치는 현재 구명환 site를 따라 갱신한다. 목표 방향은 지정한 견인 방향이다.
    고정된 사람이 아닌 강체 부유 시험이며 전역 장애물 회피 계획기는 아니다.
    """

    def __init__(self, world, robot, *, tow_heading=0., settings=None, navigation=None):
        import mujoco

        if robot not in world.parts or robot.body_id is None:
            raise ValueError('world에 조립·bind된 로봇이 필요합니다.')
        self.world, self.robot = world, robot
        self.settings = RescueApproachSettings() if settings is None else settings
        if not math.isfinite(tow_heading):
            raise ValueError('견인 방향은 유한한 각도여야 합니다.')
        self.target_heading = math.atan2(math.sin(tow_heading), math.cos(tow_heading))
        try:
            self.ring_site = world.model.site(config.Names.LIFEBUOY_TOW_SITE).id
            self.stern_site = world.model.site(config.Names.TOW_POINT_SITE).id
            human_body = world.model.body(config.Names.HUMAN_BODY).id
        except KeyError as error:
            raise ValueError('구명환·선미 연결점이 먼저 조립되어야 합니다.') from error
        if world.model.site_bodyid[self.stern_site] != robot.body_id:
            raise ValueError('선미 연결점은 로봇에 속해야 합니다.')
        self.navigation = ApproachSettings() if navigation is None else navigation
        self.direction = np.array([math.cos(self.target_heading), math.sin(self.target_heading)])
        self._stern_b = world.model.site_pos[self.stern_site].copy()
        if self._stern_b[0] >= 0:
            raise ValueError('선미 연결점은 로봇의 뒤쪽에 있어야 합니다.')
        self._staging_distance = self.settings.staging_gap - self._stern_b[0]
        if self.settings.route_radius < self._staging_distance:
            raise ValueError('바깥 경로 반지름은 대기 위치 거리 이상이어야 합니다.')
        mujoco.mj_forward(world.model, world.data)
        self._plan_route()
        self.controller = PhaseManager(world, robot, self._route_target(),
                                       target_heading=self.target_heading, settings=self.navigation)
        self.stage = RescueStage.TRANSIT
        self._ready_since = None
        victim_bodies = {human_body, int(world.model.site_bodyid[self.ring_site])}
        self.victim_geoms = tuple(i for i in range(world.model.ngeom)
                                  if world.model.geom_bodyid[i] in victim_bodies)
        self._clearance = self.clearance()
        self.minimum_observed_clearance = self._clearance
        self._next_clearance_time = float(world.data.time)
        self._last_time = float(world.data.time)

    def reset(self):
        """월드가 출발 배치로 리셋된 뒤 경로·대기 판정을 초기화한다."""
        import mujoco

        mujoco.mj_forward(self.world.model, self.world.data)
        self._plan_route()
        self.stage = RescueStage.TRANSIT
        self._ready_since = None
        self.controller.settings = self.navigation
        self.controller.target_xy[:] = self._route_target()
        self.controller.reset()
        self._clearance = self.clearance()
        self.minimum_observed_clearance = self._clearance
        self._next_clearance_time = float(self.world.data.time)
        self._last_time = float(self.world.data.time)

    def clearance(self):
        """현재 hull과 사람·구명환 사이 최소 표면 간격. 파생 위치 갱신 후 호출한다."""
        import mujoco

        return min(mujoco.mj_geomDistance(self.world.model, self.world.data,
                                         self.robot.geom_id, geom, 10., None)
                   for geom in self.victim_geoms)

    def check_clearance(self):
        """현재 파생 위치에서 간격을 제어 주기로 검사한다. 견인 단계에서도 재사용한다."""
        world = self.world
        if world.data.time + 1e-12 >= self._next_clearance_time:
            self._clearance = self.clearance()
            self.minimum_observed_clearance = min(self.minimum_observed_clearance, self._clearance)
            self._next_clearance_time = float(world.data.time) + self.controller.control_period
            if self._clearance < self.settings.minimum_clearance:
                self.robot.set_command(0, 0)
                raise RuntimeError('선체와 사람·구명환의 간격이 너무 작아 접근 시험을 중단했습니다.')

    def _plan_route(self):
        offset = self.world.data.xpos[self.robot.body_id, :2] - self.world.data.site_xpos[self.ring_site, :2]
        radius = float(np.linalg.norm(offset))
        if radius < self.settings.route_radius:
            raise ValueError('이 시험은 배를 구명환에서 route_radius 이상 떨어뜨려 시작해야 합니다.')
        start_angle = math.atan2(offset[1], offset[0])
        turn = math.atan2(math.sin(self.target_heading - start_angle),
                          math.cos(self.target_heading - start_angle))
        count = max(1, math.ceil(abs(turn) / math.radians(30)))
        self._angles = list(np.linspace(start_angle, start_angle + turn, count + 1))
        self._route_index = 0

    @property
    def arrived(self):
        return self.stage == RescueStage.READY

    @property
    def target_xy(self):
        return self.controller.target_xy

    def _body_target(self, gap):
        # yaw만의 초기 추정 대신 현재 roll/pitch도 반영해 실제 선미 site를 목표에 맞춘다.
        rotation = self.world.data.xmat[self.robot.body_id].reshape(3, 3)
        # 목표 yaw와 현재 yaw의 차이만 회전한다. 수면 위 선체의 작은 기울기는 유지한다.
        current_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        angle = self.target_heading - current_yaw
        yaw_rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
        stern_offset = yaw_rotation @ (rotation @ self._stern_b)[:2]
        return self.world.data.site_xpos[self.ring_site, :2] + gap * self.direction - stern_offset

    def _route_target(self):
        angle = self._angles[self._route_index]
        return (self.world.data.site_xpos[self.ring_site, :2]
                + self.settings.route_radius * np.array([math.cos(angle), math.sin(angle)]))

    def readings(self):
        import mujoco

        state = self.controller.readings()
        data, model = self.world.data, self.world.model
        start, end = data.site_xpos[self.stern_site], data.site_xpos[self.ring_site]
        velocities = []
        for site in (self.stern_site, self.ring_site):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0)
            velocities.append(velocity[3:])
        state.update(phase=self.stage.value, endpoint_gap=float(np.linalg.norm(start - end)),
                     relative_speed=float(np.linalg.norm(velocities[0] - velocities[1])),
                     clearance=self._clearance)
        return state

    def step(self):
        import mujoco

        world = self.world
        mujoco.mj_forward(world.model, world.data)
        if world.data.time < self._last_time - 1e-12:
            self.reset()
        self.check_clearance()
        # 이 제어기는 물리 상태를 리셋하거나 사람 위치를 직접 바꾸지 않는다.
        if self.stage == RescueStage.TRANSIT:
            self.controller.target_xy[:] = self._route_target()
            if np.linalg.norm(self.controller.target_xy - world.data.xpos[self.robot.body_id, :2]) < self.settings.waypoint_tolerance:
                if self._route_index + 1 < len(self._angles):
                    self._route_index += 1
                else:
                    self.stage = RescueStage.ALIGN_STERN
                self.controller.reset()
        if self.stage == RescueStage.ALIGN_STERN:
            self.controller.target_xy[:] = self._body_target(self.settings.staging_gap)
            if self.controller.arrived:
                self.stage = RescueStage.BACK_IN
                self.controller.settings = replace(self.navigation, max_speed=self.settings.docking_speed)
                self.controller.reset()
        if self.stage in (RescueStage.BACK_IN, RescueStage.READY):
            self.controller.target_xy[:] = self._body_target(self.settings.connection_gap)
        self.controller.step()
        if self.stage in (RescueStage.BACK_IN, RescueStage.READY):
            state = self.readings()
            ready = (self.controller.arrived
                     and abs(state['endpoint_gap'] - self.settings.connection_gap) <= self.settings.gap_tolerance
                     and state['relative_speed'] <= self.settings.relative_speed_tolerance)
            if ready:
                if self._ready_since is None:
                    self._ready_since = float(world.data.time)
                if world.data.time - self._ready_since >= self.settings.ready_hold_seconds:
                    self.stage = RescueStage.READY
            else:
                self._ready_since = None
                self.stage = RescueStage.BACK_IN
        self._last_time = float(world.data.time)
