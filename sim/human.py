"""3 세계 조립: 사람 몸 모델 (담당: 가영)."""

import numpy as np

from sim import config


class Human:
    # 이 단순 강체만으로 추가 시간 간격 제한을 확정하지 않는다.
    REQUIRES = {}

    def __init__(self, position=(0.0, 0.0, 1.0)):
        # 기본 위치는 독립 시험용: 수면 기준 1m 위, 몸의 긴 방향은 월드 +x.
        # 실제 구조 장면의 배치는 world 생성 시 별도로 전달한다.
        self.position = position
        self.body_id = None
        self.joint_id = None
        self.geom_id = None

    def build(self, spec):
        """사람을 상자 한 개와 자유 관절로 만들어 설계도에 추가한다."""
        import mujoco

        cfg = config.Human
        if cfg.SHAPE != "box":
            raise ValueError("현재 Human은 SHAPE='box'만 지원합니다.")
        dimensions = np.array([cfg.LENGTH, cfg.WIDTH, cfg.HEIGHT], dtype=float)
        mass = float(cfg.MASS)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("사람의 길이·폭·두께는 유한한 양수여야 합니다.")
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("사람 질량은 유한한 양수여야 합니다.")
        position = self._vector(self.position, "position")
        com = self._vector(cfg.COM, "Human.COM")

        body = spec.worldbody.add_body(name=config.Names.HUMAN_BODY, pos=position)
        body.add_joint(name=config.Names.HUMAN_FREE, type=mujoco.mjtJoint.mjJNT_FREE)

        # 질량·무게중심·관성은 body에서 한 번만 지정한다.
        # 균일 상자의 중심 관성 공식. COM 변경 시에도 이 관성을 사용하는 단순 근사다.
        length, width, height = dimensions
        body.explicitinertial = True
        body.mass = mass
        body.ipos[:] = com
        body.inertia[:] = mass / 12 * np.array([
            width**2 + height**2,
            length**2 + height**2,
            length**2 + width**2,
        ])
        body.add_geom(
            name=config.Names.HUMAN_GEOM,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=dimensions / 2,  # MuJoCo box는 전체 크기가 아닌 반길이를 받는다.
            mass=0.0,  # 질량은 위 body에 이미 지정했으므로 다시 더하지 않는다.
            group=0 if config.Groups.PHYSICS is None else config.Groups.PHYSICS,
            contype=0, conaffinity=0,  # 독립 시험용: 충돌 규칙 합의 전에는 접촉을 끈다.
        )

    @staticmethod
    def _vector(value, name):
        vector = np.array(value, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{name}는 유한한 숫자 3개의 좌표여야 합니다.")
        return vector

    def bind(self, model):
        """컴파일된 모델에서 몸체·관절·모양 ID를 찾아 저장한다."""
        try:
            body_id = model.body(config.Names.HUMAN_BODY).id
            joint_id = model.joint(config.Names.HUMAN_FREE).id
            geom_id = model.geom(config.Names.HUMAN_GEOM).id
        except KeyError as error:
            raise ValueError("Human.bind 전에 사람 모델의 build·compile이 필요합니다.") from error
        self.body_id, self.joint_id, self.geom_id = body_id, joint_id, geom_id

    def add_forces(self, model, data, acc):
        """현재 추가 외력 없음. 중력은 MuJoCo가 처리한다.

        부력·물 저항은 담당 범위 합의 후 연결한다. 중력을 다시 add하지 않는다.
        """
        pass
