"""목표 위치·방향으로 접근해 정지하는 첫 단계 제어.

구조 대상 접근·로프 연결·견인 단계는 아직 포함하지 않는다.
"""

from dataclasses import dataclass, fields
from enum import Enum
import math

import numpy as np

from sim import config


class Phase(str, Enum):
    APPROACH = "APPROACH"
    SETTLE = "SETTLE"
    ARRIVED = "ARRIVED"


@dataclass(frozen=True)
class ApproachSettings:
    """잔잔한 물의 현재 선체에 맞춘 시험용 제어값. 거리 m, 속도 m/s, 각도 rad."""

    max_speed: float = .5
    distance_gain: float = .18
    speed_gain: float = 60.0  # N / (m/s)
    position_gain: float = 30.0  # 정지 유지 N/m
    heading_gain: float = 1.4  # 목표 yaw 속도 / 방향 오차
    yaw_speed_gain: float = 12.0  # N·m / (rad/s)
    max_yaw_speed: float = .6
    max_force: float = 60.0
    max_brake_force: float = 30.0
    max_yaw_torque: float = 6.0
    capture_radius: float = .05
    position_tolerance: float = .20
    reapproach_radius: float = .21
    speed_tolerance: float = .05
    heading_tolerance: float = math.radians(8)
    yaw_speed_tolerance: float = .03
    arrival_hold_seconds: float = 1.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name}은 유한한 양수여야 합니다.")
        if not self.capture_radius <= self.position_tolerance < self.reapproach_radius:
            raise ValueError("capture_radius <= position_tolerance < reapproach_radius가 필요합니다.")
        if self.heading_tolerance >= math.pi:
            raise ValueError("방향 허용 오차는 pi보다 작아야 합니다.")


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class PhaseManager:
    """50Hz 명령 갱신과 물리 스텝을 관리한다. 목표는 선체 원점의 월드 xy와 yaw.

    ARRIVED도 저속 제동·위치 유지를 계속한다. 허용 범위를 벗어나면 재접근한다.
    """

    def __init__(self, world, robot, target_xy, *, target_heading=0.0, settings=None):
        if robot not in world.parts or robot.body_id is None:
            raise ValueError("world에 조립·bind된 로봇을 전달하세요.")
        target = np.asarray(target_xy, dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all() or not math.isfinite(target_heading):
            raise ValueError("목표는 유한한 xy 좌표 2개와 방향이어야 합니다.")
        if config.Sim.CONTROL_HZ is None or not math.isfinite(config.Sim.CONTROL_HZ) or config.Sim.CONTROL_HZ <= 0:
            raise ValueError("Sim.CONTROL_HZ에 유한한 양수를 지정하세요.")
        self.world, self.robot = world, robot
        self.target_xy = target.copy()
        self.target_heading = _angle(target_heading)
        self.settings = ApproachSettings() if settings is None else settings
        self.control_period = 1 / config.Sim.CONTROL_HZ
        left, right = robot.site_ids
        self._lever = world.model.site_pos[left, 1] - world.model.site_pos[right, 1]
        if self._lever <= 0 or not np.isfinite(self._lever):
            raise ValueError("왼쪽 추진기 y가 오른쪽보다 커야 합니다.")
        for maximum in (config.Thruster.MAX_FORWARD, config.Thruster.MAX_REVERSE):
            if not math.isfinite(maximum) or maximum <= 0:
                raise ValueError("추진기 최대 추력은 유한한 양수여야 합니다.")
        self.reset()

    def reset(self):
        """월드 리셋 또는 새 접근 시험 시작 후 호출한다. 물리 상태는 바꾸지 않는다."""
        self.phase = Phase.APPROACH
        self.arrival_seconds = 0.0
        self._last_control_time = None
        self._next_control_time = float(self.world.data.time)
        self.forward = self.steering = 0.0
        self.robot.set_command(0, 0)

    @property
    def arrived(self):
        return self.phase == Phase.ARRIVED

    def readings(self):
        """현재 파생 위치·속도에서 표시용 상태를 읽는다. 호출 전 mj_forward가 필요하다."""
        from sim.frames import body_velocity_w

        data, model = self.world.data, self.world.model
        rotation = data.xmat[self.robot.body_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        velocity, omega = body_velocity_w(model, data, self.robot.body_id)
        error = self.target_xy - data.xpos[self.robot.body_id, :2]
        return dict(distance=float(np.linalg.norm(error)), speed=float(np.linalg.norm(velocity[:2])),
                    heading_error=_angle(self.target_heading - yaw), yaw_speed=float(omega[2]),
                    yaw=yaw, error=error, velocity=velocity[:2], phase=self.phase.value,
                    forward=self.forward, steering=self.steering)

    def _set_force(self, force, torque):
        # 정·역추력 최대값이 달라 명령 ±같은 값은 순수 회전이 아니다.
        # 먼저 좌우 힘[N]으로 분배한 뒤 각 추진기의 정규화 명령으로 역변환한다.
        # 요 토크를 먼저 가능한 범위로 제한하고, 두 노즐 한계 안에서 합력을 배분한다.
        # 좌우를 각각 잘라 버리면 원하는 순수 회전이 전진 운동으로 바뀐다.
        maximum, minimum = config.Thruster.MAX_FORWARD, -config.Thruster.MAX_REVERSE
        torque = float(np.clip(torque, -.5 * self._lever * (maximum - minimum),
                               .5 * self._lever * (maximum - minimum)))
        difference = 2 * torque / self._lever
        force = float(np.clip(force, 2 * minimum + abs(difference), 2 * maximum - abs(difference)))
        forces = np.array([(force - difference) / 2, (force + difference) / 2])
        commands = forces / np.where(forces >= 0, config.Thruster.MAX_FORWARD,
                                      config.Thruster.MAX_REVERSE)
        self.forward = float((commands[0] + commands[1]) / 2)
        self.steering = float((commands[1] - commands[0]) / 2)
        self.robot.set_command(self.forward, self.steering)

    def _update_control(self, elapsed):
        state, settings = self.readings(), self.settings
        if not np.isfinite(self.world.data.qpos).all() or not np.isfinite(self.world.data.qvel).all():
            self.robot.set_command(0, 0)
            raise RuntimeError("비정상 물리 상태로 접근 제어를 중단했습니다.")
        if self.phase != Phase.APPROACH and state['distance'] > settings.reapproach_radius:
            self.phase = Phase.APPROACH
        if self.phase == Phase.APPROACH and state['distance'] <= settings.capture_radius:
            self.phase = Phase.SETTLE
        direction = np.array([math.cos(state['yaw']), math.sin(state['yaw'])])
        surge_speed = float(state['velocity'] @ direction)
        if self.phase == Phase.APPROACH:
            desired_speed = min(settings.max_speed, settings.distance_gain * state['distance'])
            desired_velocity = desired_speed * state['error'] / state['distance']
            force_w = settings.speed_gain * (desired_velocity - state['velocity'])
            heading = math.atan2(force_w[1], force_w[0])
            heading_error = _angle(heading - state['yaw'])
            # 필요한 힘이 뒤쪽이면 후진 추력을 사용해 제동한다.
            if abs(heading_error) > math.pi / 2:
                heading_error = _angle(heading_error + math.pi)
            force = float(force_w @ direction)
        else:
            heading_error = state['heading_error']
            force = settings.position_gain * float(state['error'] @ direction) - settings.speed_gain * surge_speed
        yaw_speed = float(np.clip(settings.heading_gain * heading_error,
                                  -settings.max_yaw_speed, settings.max_yaw_speed))
        torque = settings.yaw_speed_gain * (yaw_speed - state['yaw_speed'])
        self._set_force(float(np.clip(force, -settings.max_brake_force, settings.max_force)),
                        float(np.clip(torque, -settings.max_yaw_torque, settings.max_yaw_torque)))
        stable = (self.phase != Phase.APPROACH
                  and state['distance'] <= settings.position_tolerance
                  and state['speed'] <= settings.speed_tolerance
                  and abs(state['heading_error']) <= settings.heading_tolerance
                  and abs(state['yaw_speed']) <= settings.yaw_speed_tolerance)
        self.arrival_seconds = self.arrival_seconds + elapsed if stable else 0.0
        if self.arrival_seconds + 1e-12 >= settings.arrival_hold_seconds:
            self.phase = Phase.ARRIVED
        elif self.phase == Phase.ARRIVED:
            self.phase = Phase.SETTLE

    def step(self):
        """위치 갱신 → 주기적 명령 → reset/add/apply → mj_step. 월드를 리셋하지 않는다."""
        import mujoco

        world = self.world
        mujoco.mj_forward(world.model, world.data)
        now = float(world.data.time)
        if self._last_control_time is not None and now < self._last_control_time - 1e-12:
            self.reset()
        if now + 1e-12 >= self._next_control_time:
            elapsed = 0.0 if self._last_control_time is None else now - self._last_control_time
            self._update_control(elapsed)
            self._last_control_time = now
            self._next_control_time = now + self.control_period
        world.acc.reset()
        for part in world.parts:
            part.add_forces(world.model, world.data, world.acc)
        world.acc.apply(world.model, world.data)
        mujoco.mj_step(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)
