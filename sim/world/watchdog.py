"""시뮬레이션 시간·비정상 물리 상태를 감시하는 latched Watchdog. 전복 판정은 없앴다 (2026-11-19).

추론 응답이 제한된 실제 시간 안에 오지 않으면 InferenceWatchdog가 안전 모드로 고정한다.
안전 모드는 추진 명령을 0으로 두며, 로프를 끊거나 고장을 복구하지 않는다.
"""

from dataclasses import dataclass, fields
from enum import Enum
import math
import time

import numpy as np


class FailureCode(str, Enum):
    STAGE_TIMEOUT = 'STAGE_TIMEOUT'
    MISSION_TIMEOUT = 'MISSION_TIMEOUT'
    INVALID_STATE = 'INVALID_STATE'
    ENGINE_WARNING = 'ENGINE_WARNING'
    CONTROL_ERROR = 'CONTROL_ERROR'
    INFERENCE_TIMEOUT = 'INFERENCE_TIMEOUT'


@dataclass(frozen=True)
class WatchdogSettings:
    """잔잔한 물 시험용 기준. 전복 판정은 2026-11-19 동경 결정으로 없앴다 (기울기로 임무를 멈추지 않음)."""

    approach_timeout: float = 150.0
    attach_timeout: float = 5.0
    tow_timeout: float = 100.0
    mission_timeout: float = 240.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field.name}은 유한한 양수여야 합니다.')


@dataclass(frozen=True)
class Failure:
    code: FailureCode
    stage: str
    time: float
    message: str


class MissionWatchdog:
    def __init__(self, settings=None):
        self.settings = WatchdogSettings() if settings is None else settings
        self.reset()

    def reset(self, now=0.0, stage='APPROACH'):
        if not math.isfinite(now) or now < 0:
            raise ValueError('Watchdog 시작 시각은 유한한 0 이상 값이어야 합니다.')
        self.failure = None
        self._mission_start = self._stage_start = self._last_time = float(now)
        self._stage = stage

    @staticmethod
    def invalid_state(data):
        if not math.isfinite(data.time) or data.time < 0:
            return 'Invalid simulation time'
        for name in ('qpos', 'qvel', 'qacc', 'act', 'ctrl', 'xfrc_applied', 'qfrc_applied',
                     'xpos', 'xmat', 'site_xpos'):
            if not np.isfinite(getattr(data, name)).all():
                return f'Non-finite {name}'
        return None

    def trip(self, code, now, stage, message):
        if self.failure is None:
            timestamp = float(now) if math.isfinite(now) else self._last_time
            self.failure = Failure(code, stage, timestamp, message)
        return self.failure

    def check(self, model, data, body_id, stage):
        """mj_forward 전후에 호출한다. 최초 실패 이유를 reset 전까지 유지한다."""
        import mujoco

        if self.failure is not None:
            return self.failure
        invalid = self.invalid_state(data)
        if invalid:
            return self.trip(FailureCode.INVALID_STATE, data.time, stage, invalid)
        now = float(data.time)
        if now < self._last_time - 1e-12:
            return self.trip(FailureCode.INVALID_STATE, now, stage, 'Simulation time moved backwards without reset')
        warning_ids = np.flatnonzero(data.warning.number)
        if len(warning_ids):
            warning = mujoco.mjtWarning(int(warning_ids[0])).name
            return self.trip(FailureCode.ENGINE_WARNING, now, stage, f'MuJoCo {warning}')
        joint = int(model.body_jntadr[body_id])
        if joint < 0 or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError('자세 검사는 자유 관절이 있는 선체가 필요합니다.')
        address = model.jnt_qposadr[joint]
        quat = data.qpos[address + 3:address + 7]
        norm = float(np.linalg.norm(quat))
        if not math.isfinite(norm) or norm < 1e-12:
            return self.trip(FailureCode.INVALID_STATE, now, stage, 'Invalid boat quaternion')
        if stage != self._stage:
            self._stage = stage
            self._stage_start = now
        self._last_time = now
        if stage != 'COMPLETE' and now - self._mission_start + 1e-12 >= self.settings.mission_timeout:
            return self.trip(FailureCode.MISSION_TIMEOUT, now, stage,
                             f'Mission exceeded {self.settings.mission_timeout:g} simulation seconds')
        timeout = {'APPROACH': self.settings.approach_timeout,
                   'ATTACH': self.settings.attach_timeout, 'TOW': self.settings.tow_timeout}.get(stage)
        if timeout is not None and now - self._stage_start + 1e-12 >= timeout:
            return self.trip(FailureCode.STAGE_TIMEOUT, now, stage,
                             f'{stage} exceeded {timeout:g} simulation seconds')
        return None


@dataclass(frozen=True)
class InferenceWatchdogSettings:
    """추론 응답을 기다릴 실제 시간. 시연용이며 제어 주기와 합의된 값이 아니다."""

    response_timeout: float = 0.5

    def __post_init__(self):
        if not math.isfinite(self.response_timeout) or self.response_timeout <= 0:
            raise ValueError('response_timeout은 유한한 양수여야 합니다.')


class InferenceWatchdog:
    """요청 시각을 기록하고, 응답이 없으면 안전 모드(추진 0)로 고정한다.

    시각은 시뮬레이션 data.time이 아니라 실제 시간이다. clock를 바꾸면 시험이 대기하지 않아도 된다.
    대기 중인 요청이 없으면 check는 시계를 읽지 않는다.
    """

    def __init__(self, settings=None, clock=None):
        self.settings = InferenceWatchdogSettings() if settings is None else settings
        if clock is not None and not callable(clock):
            raise ValueError('clock는 시각을 반환하는 함수여야 합니다.')
        self.clock = time.monotonic if clock is None else clock
        self.reset()

    def reset(self):
        self.failure = None
        self._requested_at = None
        self._last_now = None

    @property
    def safe_mode(self):
        return self.failure is not None

    @property
    def command(self):
        """안전 모드면 (전진 0, 조향 0). 아니면 None이며 호출자가 정책 출력을 쓴다."""
        return (0.0, 0.0) if self.safe_mode else None

    def request(self, now=None):
        """추론 요청을 기록한다. 이미 기다리는 중이면 최초 시각을 유지한다."""
        if self.failure is not None:
            return self.failure
        if self._requested_at is None:
            self._requested_at = self._now(now)
        return None

    def respond(self, now=None):
        """제시간 응답이면 대기를 끝낸다. 늦으면 응답을 폐기하고 안전 모드로 고정한다."""
        if self.failure is not None:
            return self.failure
        if self._requested_at is None:
            raise ValueError('추론 요청 없이 응답을 기록할 수 없습니다.')
        now = self._now(now)
        waited = now - self._requested_at
        self._requested_at = None
        if waited + 1e-12 >= self.settings.response_timeout:
            return self._trip(now, waited)
        return None

    def check(self, now=None):
        """아직 응답이 없고 제한 시간을 넘기면 안전 모드로 고정한다."""
        if self.failure is not None:
            return self.failure
        if self._requested_at is None:
            return None
        now = self._now(now)
        waited = now - self._requested_at
        if waited + 1e-12 >= self.settings.response_timeout:
            return self._trip(now, waited)
        return None

    def apply(self, robot):
        """안전 모드일 때만 추진 명령을 0으로 만든다. 물리 상태는 바꾸지 않는다."""
        if not self.safe_mode:
            return False
        robot.set_command(0.0, 0.0)
        return True

    def _trip(self, now, waited):
        if self.failure is None:
            self.failure = Failure(
                FailureCode.INFERENCE_TIMEOUT, 'INFERENCE', float(now),
                f'Inference response exceeded {self.settings.response_timeout:g}s wall time '
                f'(waited {waited:.3f}s)')
        self._requested_at = None
        return self.failure

    def _now(self, now):
        if now is None:
            now = self.clock()
        now = float(now)
        if not math.isfinite(now):
            raise ValueError('추론 시각은 유한한 값이어야 합니다.')
        if self._last_now is not None and now < self._last_now - 1e-12:
            raise ValueError('추론 시각이 역행했습니다.')
        self._last_now = now
        return now
