"""캡슐을 원형으로 배치한 구명환. 첫 단계는 사람과의 고정 연결 시험이다."""

import operator

import numpy as np

from sim import config


class Lifebuoy:
    REQUIRES = {}

    def __init__(self, *, attached=True, position=(0.0, 0.0, 1.0)):
        self.attached = attached
        self.position = position
        self.body_id = None
        self.site_id = None

    @staticmethod
    def _vector(value, name):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{name}는 유한한 숫자 3개의 좌표여야 합니다.")
        return vector

    def build(self, spec):
        import mujoco

        cfg = config.Lifebuoy
        radius, tube, mass = float(cfg.CENTERLINE_RADIUS), float(cfg.TUBE_RADIUS), float(cfg.MASS)
        segments = operator.index(cfg.SEGMENTS)
        if not np.isfinite([radius, tube, mass]).all() or min(radius, tube, mass) <= 0:
            raise ValueError("구명환 반지름·튜브 반지름·질량은 유한한 양수여야 합니다.")
        if segments < 3 or tube >= radius * np.cos(np.pi / segments):
            raise ValueError("구명환은 캡슐 3개 이상이고 가운데 구멍이 열려 있어야 합니다.")
        tow_point = self._vector(cfg.TOW_POINT_B, "Lifebuoy.TOW_POINT_B")
        if self.attached:
            parent = spec.body(config.Names.HUMAN_BODY)
            if parent is None:
                raise ValueError("고정 연결 전에 Human.build()가 필요합니다.")
            position = self._vector(cfg.POSITION_IN_HUMAN_B, "Lifebuoy.POSITION_IN_HUMAN_B")
        else:
            parent = spec.worldbody
            position = self._vector(self.position, "position")
        body = parent.add_body(name=config.Names.LIFEBUOY_BODY, pos=position)
        if not self.attached:
            body.add_joint(name=config.Names.LIFEBUOY_FREE, type=mujoco.mjtJoint.mjJNT_FREE)
        # 자식 body에 관절이 없으면 부모에 고정된다. 접촉으로 몸을 지지하는 구현은 후속 작업.
        for index in range(segments):
            angles = 2 * np.pi * np.array([index, index + 1]) / segments
            endpoints = np.column_stack((np.zeros(2), radius * np.cos(angles), radius * np.sin(angles)))
            body.add_geom(
                name=f"{config.Names.LIFEBUOY_GEOM_PREFIX}_{index}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=endpoints.ravel(), size=[tube, 0, 0], mass=mass / segments,
                rgba=[1.0, 0.3, 0.05, 1.0] if index % 3 else [0.95, 0.95, 0.95, 1.0],
                group=0 if config.Groups.PHYSICS is None else config.Groups.PHYSICS,
                contype=0, conaffinity=0,
            )
        body.add_site(
            name=config.Names.LIFEBUOY_TOW_SITE, pos=tow_point,
            size=[0.02, 0, 0], rgba=[0.1, 0.3, 1.0, 1.0],
        )

    def bind(self, model):
        try:
            body_id = model.body(config.Names.LIFEBUOY_BODY).id
            site_id = model.site(config.Names.LIFEBUOY_TOW_SITE).id
        except KeyError as error:
            raise ValueError("Lifebuoy.bind 전에 구명환 build·compile이 필요합니다.") from error
        self.body_id, self.site_id = body_id, site_id

    def add_forces(self, model, data, acc):
        """추가 외력 없음. 부력·물 저항·실제 로프는 이후 연결한다."""
        pass
