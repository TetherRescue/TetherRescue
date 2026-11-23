"""하나개 출항 시나리오: 육지 대기 → 출항 → 익수자 GPS 자동 표기 → 항해."""

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

from sim.robot.phase_manager import ApproachSettings, PhaseManager
from sim.world.coast_map import (
    SHORE_X, VICTIM_RANGE, build_hanagae_world, land_boat_pose, water_launch_pose, _set_boat_pose,
)


class SortieStage(str, Enum):
    STOWED = 'STOWED'          # 지상에 대기
    LAUNCHING = 'LAUNCHING'    # 사람이 배를 물에 띄움
    TRANSIT = 'TRANSIT'        # GPS 목표로 항해
    ARRIVED = 'ARRIVED'        # 익수자 근처 도착


def _long_range_approach():
    return ApproachSettings(
        max_speed=1.5,
        distance_gain=.25,
        capture_radius=3.0,
        position_tolerance=5.0,
        reapproach_radius=8.0,
        arrival_hold_seconds=1.0,
    )


@dataclass(frozen=True)
class SortieSettings:
    launch_delay: float = 3.0       # 지상 대기 [s]
    launch_hold: float = 1.0        # 출항 후 GPS 찍기 전 수면 안정 [s]
    victim_range: float = VICTIM_RANGE
    approach: ApproachSettings = field(default_factory=_long_range_approach)

    def __post_init__(self):
        if not math.isfinite(self.launch_delay) or self.launch_delay < 0:
            raise ValueError('launch_delay는 0 이상이어야 합니다.')
        if not math.isfinite(self.launch_hold) or self.launch_hold < 0:
            raise ValueError('launch_hold는 0 이상이어야 합니다.')
        if not math.isfinite(self.victim_range) or self.victim_range <= 0:
            raise ValueError('victim_range는 유한한 양수여야 합니다.')


class RescueSortie:
    """전체 틀의 한 에피소드. GPS 입력은 생략하고 익수자 좌표를 자동으로 찍는다."""

    def __init__(self, *, waves=None, settings=None):
        self.settings = SortieSettings() if settings is None else settings
        self.world, self.robot, self.water, self.human = build_hanagae_world(
            waves=waves, boat_on_land=True, victim_range=self.settings.victim_range)
        self.stage = SortieStage.STOWED
        self.gps_fix = None
        self.controller = None
        self._stow_pos, self._stow_yaw = land_boat_pose()
        self._launch_pos, self._launch_yaw = water_launch_pose()
        self._afloat_since = None
        self._hold_boat(self._stow_pos, self._stow_yaw)

    @property
    def arrived(self):
        return self.stage == SortieStage.ARRIVED

    def readings(self):
        boat = self.world.data.xpos[self.robot.body_id]
        victim = self.world.data.xpos[self.human.body_id]
        state = dict(stage=self.stage.value, time=float(self.world.data.time),
                     boat_xy=boat[:2].copy(), victim_xy=victim[:2].copy(),
                     gps_xy=None if self.gps_fix is None else self.gps_fix.copy(),
                     shore_gap=float(abs(victim[0] - SHORE_X)),
                     distance_to_gps=None, phase=None, forward=0., steering=0.)
        if self.controller is not None:
            nav = self.controller.readings()
            state.update(distance_to_gps=nav['distance'], phase=nav['phase'],
                         forward=nav['forward'], steering=nav['steering'])
        return state

    def step(self):
        import mujoco

        world = self.world
        if self.stage == SortieStage.STOWED:
            self._hold_boat(self._stow_pos, self._stow_yaw)
            world.acc.reset()
            world.acc.apply(world.model, world.data)
            mujoco.mj_step(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)
            self._hold_boat(self._stow_pos, self._stow_yaw)
            if world.data.time + 1e-12 >= self.settings.launch_delay:
                self._launch()
            return True

        if self.stage == SortieStage.LAUNCHING:
            from sim.world.world import advance_world
            advance_world(world)
            if self._afloat_since is None:
                self._afloat_since = float(world.data.time)
            if world.data.time - self._afloat_since + 1e-12 >= self.settings.launch_hold:
                self._mark_gps_and_transit()
            return True

        if self.stage == SortieStage.TRANSIT:
            self.controller.step()
            if self.controller.arrived:
                self.stage = SortieStage.ARRIVED
            return True

        self.controller.step()
        return True

    def _launch(self):
        """발견자가 배를 물에 띄운 직후. 운반 연출은 생략하고 수면 배치로 대체한다."""
        _set_boat_pose(self.world, self.robot, self._launch_pos, self._launch_yaw)
        self.robot.set_command(0, 0)
        self.stage = SortieStage.LAUNCHING
        self._afloat_since = float(self.world.data.time)

    def _mark_gps_and_transit(self):
        """GPS 수동 입력 생략: 현재 익수자 수평 좌표를 목표로 찍는다."""
        victim = self.world.data.xpos[self.human.body_id, :2].copy()
        self.gps_fix = victim
        boat = self.world.data.xpos[self.robot.body_id, :2]
        heading = math.atan2(victim[1] - boat[1], victim[0] - boat[0])
        self.controller = PhaseManager(
            self.world, self.robot, victim, target_heading=heading, settings=self.settings.approach)
        self.stage = SortieStage.TRANSIT

    def _hold_boat(self, position, yaw):
        """육지 충돌이 없으므로 대기 중에는 자세를 강제한다."""
        _set_boat_pose(self.world, self.robot, position, yaw)
        self.robot.set_command(0, 0)
