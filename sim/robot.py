"""2 로봇: 선체·추진기·조향·전복 판정 (담당: 동경)."""


class Robot:
    REQUIRES = {}  # 예: {"max_dt": 0.004}

    def build(self, spec):
        """세계 조립 때 자기 몸(body/geom 등)을 spec 에 추가한다."""
        # TODO(동경)

    def bind(self, model):
        """이름 → id 를 한 번만 찾아 캐시한다. 없으면 assert 로 멈춘다."""
        # TODO(동경)

    def add_forces(self, model, data, acc):
        """스텝마다 힘을 acc(ForceAccumulator)에 더한다. 월드 좌표."""
        # TODO(동경)
