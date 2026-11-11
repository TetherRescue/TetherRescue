"""🔒 ForceAccumulator (담당: 가영).

data.xfrc_applied / data.qfrc_applied 에는 이 파일만 씁니다.
각 모듈은 add() 로 힘을 더하기만 하고, apply() 는 스텝마다 한 번 호출됩니다.
"""


class ForceAccumulator:
    def reset(self):
        """스텝 시작 시 모든 힘을 0으로."""
        # TODO(가영)

    def add(self, body_id, force_w, torque_w=None, point_w=None):
        """월드 좌표 힘을 더한다. point_w 가 있으면 그 점에 작용하는 힘."""
        # TODO(가영)

    def apply(self, model, data):
        """모은 힘을 MuJoCo 에 한 번에 쓴다."""
        # TODO(가영)
