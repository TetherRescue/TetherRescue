"""시뮬레이션 시간·전복·비정상 물리 상태를 감시하는 latched Watchdog.

추론 응답의 wall-clock 지연을 감시하는 기능은 아직 포함하지 않는다.
"""

from dataclasses import dataclass, fields
from enum import Enum
import math

import numpy as np


class FailureCode(str, Enum):
    STAGE_TIMEOUT = 'STAGE_TIMEOUT'
    MISSION_TIMEOUT = 'MISSION_TIMEOUT'
    CAPSIZE = 'CAPSIZE'
    INVALID_STATE = 'INVALID_STATE'
    ENGINE_WARNING = 'ENGINE_WARNING'
    CONTROL_ERROR = 'CONTROL_ERROR'


@dataclass(frozen=True)
class WatchdogSettings:
    """잔잔한 물 시험용 기준. 미확정 config.Capsize 계약값을 변경하지 않는다."""

    approach_timeout: float = 150.0
    attach_timeout: float = 5.0
    tow_timeout: float = 100.0
    mission_timeout: float = 240.0
    roll_limit: float = math.radians(60)
    pitch_limit: float = math.radians(60)

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field.name}은 유한한 양수여야 합니다.')
        if max(self.roll_limit, self.pitch_limit) > math.pi / 2:
            raise ValueError('전복 기준은 90도 이하여야 합니다.')


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
            raise ValueError('전복 검사는 자유 관절이 있는 선체가 필요합니다.')
        address = model.jnt_qposadr[joint]
        quat = data.qpos[address + 3:address + 7]
        norm = float(np.linalg.norm(quat))
        if not math.isfinite(norm) or norm < 1e-12:
            return self.trip(FailureCode.INVALID_STATE, now, stage, 'Invalid boat quaternion')
        # xmat이 아직 갱신되지 않았어도 qpos에서 직접 자세를 판정한다.
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quat / norm)
        rotation = matrix.reshape(3, 3)
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
        if abs(roll) >= self.settings.roll_limit or abs(pitch) >= self.settings.pitch_limit:
            return self.trip(FailureCode.CAPSIZE, now, stage,
                             f'Boat roll={math.degrees(roll):.1f} deg, pitch={math.degrees(pitch):.1f} deg')
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
