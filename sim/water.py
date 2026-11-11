"""1 물: 파도·복셀 부력·저항·부가질량 (담당: 미현)."""


class Water:
    REQUIRES = {}  # 예: {"max_dt": 0.004}

    def build(self, spec):
        """세계 조립 때 자기 몸(body/geom 등)을 spec 에 추가한다."""
        # TODO(미현)

    def bind(self, model):
        """이름 → id 를 한 번만 찾아 캐시한다. 없으면 assert 로 멈춘다."""
        # TODO(미현)

    def add_forces(self, model, data, acc):
        """스텝마다 힘을 acc(ForceAccumulator)에 더한다. 월드 좌표."""
        # TODO(미현)
