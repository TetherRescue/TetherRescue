"""3 세계 조립: 사람 몸 모델 (담당: 가영)."""

import numpy as np

from sim import config


class Human:
    # 이 단순 강체만으로 추가 시간 간격 제한을 확정하지 않는다.
    REQUIRES = {}

    def __init__(self, position=(0.0, 0.0, 1.0), *, shape=None):
        # 기본 위치는 독립 시험용: 수면 기준 1m 위, 몸의 긴 방향은 월드 +x.
        # 실제 구조 장면의 배치는 world 생성 시 별도로 전달한다.
        self.position = position
        self.shape = config.Human.SHAPE if shape is None else shape
        self.body_id = None
        self.joint_id = None
        self.geom_id = None
        self.geom_ids = {}

    def build(self, spec):
        """사람의 고정된 형상과 전체를 이동·회전시키는 자유 관절을 만든다."""
        import mujoco

        cfg = config.Human
        if self.shape not in ("box", "humanoid"):
            raise ValueError("Human shape은 'box' 또는 'humanoid'여야 합니다.")
        mass = float(cfg.MASS)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("사람 질량은 유한한 양수여야 합니다.")
        position = self._vector(self.position, "position")

        body = spec.worldbody.add_body(name=config.Names.HUMAN_BODY, pos=position)
        body.add_joint(name=config.Names.HUMAN_FREE, type=mujoco.mjtJoint.mjJNT_FREE)

        if self.shape == "humanoid":
            self._build_humanoid(body, mass)
        else:
            self._build_box(body, mass)

    @staticmethod
    def _geom_options():
        return dict(
            group=0 if config.Groups.PHYSICS is None else config.Groups.PHYSICS,
            contype=0, conaffinity=0,  # 공통 충돌 규칙 합의 후 접촉을 활성화한다.
        )

    def _build_box(self, body, mass):
        import mujoco

        cfg = config.Human
        dimensions = np.array([cfg.LENGTH, cfg.WIDTH, cfg.HEIGHT], dtype=float)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("사람의 길이·폭·두께는 유한한 양수여야 합니다.")
        com = self._vector(cfg.COM, "Human.COM")

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
            **self._geom_options(),
        )

    def _build_humanoid(self, body, mass):
        import mujoco

        cfg, names = config.Human, config.Names
        fractions = np.asarray(cfg.MASS_FRACTIONS, dtype=float)
        if (fractions.shape != (6,) or not np.isfinite(fractions).all()
                or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0)):
            raise ValueError("Human.MASS_FRACTIONS는 합이 1인 양수 6개여야 합니다.")
        parts = [
            (names.HUMAN_GEOM, cfg.TORSO_RADIUS, cfg.TORSO_FROMTO_B, [0.12, 0.5, 0.65, 1]),
            (names.HUMAN_HEAD_GEOM, cfg.HEAD_RADIUS, None, [0.85, 0.65, 0.48, 1]),
            (names.HUMAN_ARM_L_GEOM, cfg.ARM_RADIUS, cfg.ARM_L_FROMTO_B, [0.85, 0.65, 0.48, 1]),
            (names.HUMAN_ARM_R_GEOM, cfg.ARM_RADIUS, cfg.ARM_R_FROMTO_B, [0.85, 0.65, 0.48, 1]),
            (names.HUMAN_LEG_L_GEOM, cfg.LEG_RADIUS, cfg.LEG_L_FROMTO_B, [0.15, 0.2, 0.35, 1]),
            (names.HUMAN_LEG_R_GEOM, cfg.LEG_RADIUS, cfg.LEG_R_FROMTO_B, [0.15, 0.2, 0.35, 1]),
        ]
        for fraction, (name, radius, endpoints, color) in zip(fractions, parts):
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("사람 부위의 반지름은 유한한 양수여야 합니다.")
            if endpoints is None:
                geometry = dict(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                pos=self._vector(cfg.HEAD_POS_B, "HEAD_POS_B"))
            else:
                points = np.asarray(endpoints, dtype=float)
                if (points.shape != (2, 3) or not np.isfinite(points).all()
                        or np.linalg.norm(points[1] - points[0]) == 0):
                    raise ValueError("캡슐 양 끝점은 유한하고 서로 다른 3차원 좌표여야 합니다.")
                geometry = dict(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=points.ravel())
            # 부위마다 질량을 나눠 준다. body에 70kg을 별도로 더하지 않는다.
            body.add_geom(name=name, size=[radius, 0, 0], mass=mass * fraction,
                          rgba=color, **geometry, **self._geom_options())
        # MuJoCo가 geom 분포로 전체 무게중심·관성을 계산한다.

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
            names = [config.Names.HUMAN_GEOM]
            if self.shape == "humanoid":
                names += [config.Names.HUMAN_HEAD_GEOM, config.Names.HUMAN_ARM_L_GEOM,
                          config.Names.HUMAN_ARM_R_GEOM, config.Names.HUMAN_LEG_L_GEOM,
                          config.Names.HUMAN_LEG_R_GEOM]
            geom_ids = {name: model.geom(name).id for name in names}
        except KeyError as error:
            raise ValueError("Human.bind 전에 사람 모델의 build·compile이 필요합니다.") from error
        self.body_id, self.joint_id, self.geom_id = body_id, joint_id, geom_id
        self.geom_ids = geom_ids

    def add_forces(self, model, data, acc):
        """현재 추가 외력 없음. 중력은 MuJoCo가 처리한다.

        부력·물 저항은 담당 범위 합의 후 연결한다. 중력을 다시 add하지 않는다.
        """
        pass
