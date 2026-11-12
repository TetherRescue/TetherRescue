"""🔒 ForceAccumulator (담당: 가영).

data.xfrc_applied / data.qfrc_applied 에는 이 파일만 씁니다.
각 모듈은 add() 로 힘을 더하기만 하고, apply() 는 스텝마다 한 번 호출됩니다.
"""

import operator

import numpy as np


class ForceAccumulator:
    def __init__(self):
        # add() 시점에는 무게중심을 모르므로 작용점도 함께 보관한다.
        self._loads = []

    def reset(self):
        """스텝 시작 시 누적 기록을 비운다. MuJoCo 배열은 apply()에서 갱신한다."""
        self._loads.clear()

    def add(self, body_id, force_w, torque_w=None, point_w=None):
        """월드 좌표 힘[N]·토크[N·m]·작용점[m]을 기록한다.

        point_w 생략 시 힘은 몸체 무게중심에 작용한다.
        torque_w는 작용점에 따른 토크에 더하는 별도의 순수 토크다.
        """
        body_id = operator.index(body_id)
        if body_id <= 0:
            raise ValueError("body_id는 월드(0)를 제외한 양의 정수여야 합니다.")
        force = self._vector(force_w, "force_w")
        torque = np.zeros(3) if torque_w is None else self._vector(torque_w, "torque_w")
        point = None if point_w is None else self._vector(point_w, "point_w")
        self._loads.append((body_id, force, torque, point))

    @staticmethod
    def _vector(value, name):
        vector = np.array(value, dtype=float, copy=True)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{name}는 유한한 숫자 3개로 된 벡터여야 합니다.")
        return vector

    def apply(self, model, data):
        """현재 자세의 무게중심을 기준으로 합산해 외력 배열을 교체한다.

        호출 전에 mj_forward 등으로 data.xipos를 현재 자세에 맞춰야 한다.
        반복 호출해도 힘이 중복되지 않으며, 누적 기록은 reset()까지 유지한다.
        """
        loads = np.zeros((model.nbody, 6))
        for body_id, force, torque, point in self._loads:
            if body_id >= model.nbody:
                raise ValueError(f"모델에 body_id={body_id}인 몸체가 없습니다.")
            loads[body_id, :3] += force
            loads[body_id, 3:] += torque
            if point is not None:
                # 몸체 원점(xpos)이 아니라 무게중심(xipos)에서 작용점까지의 거리.
                arm_w = point - data.xipos[body_id]
                loads[body_id, 3:] += np.cross(arm_w, force)

        # xfrc_applied 순서: [Fx, Fy, Fz, Tx, Ty, Tz].
        # 외력은 이 파일만 소유한다. 같은 힘을 qfrc에도 넣으면 중복 적용된다.
        data.xfrc_applied[:] = loads
        data.qfrc_applied[:] = 0.0
