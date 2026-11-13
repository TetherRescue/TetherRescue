"""구명환은 항상 독립 자유 강체다. 착용 시험만 해제 가능한 몸통 지지를 켠다."""
import operator
import numpy as np
from sim import config


class Lifebuoy:
    REQUIRES = {}

    def __init__(self, *, attached=False, position=None, grip_sites=False, human_contact=True):
        self.attached = attached  # 이미 착용한 시험의 초기 지지 여부. body를 합치지는 않는다.
        self.grip_sites = grip_sites
        self.human_contact = human_contact
        self.position = position
        self.body_id = self.site_id = self.worn_eq_id = None

    @staticmethod
    def _vector(value, name):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f'{name}는 유한한 숫자 3개의 좌표여야 합니다.')
        return vector

    def build(self, spec):
        import mujoco
        cfg, names = config.Lifebuoy, config.Names
        radius, tube, mass = float(cfg.CENTERLINE_RADIUS), float(cfg.TUBE_RADIUS), float(cfg.MASS)
        segments = operator.index(cfg.SEGMENTS)
        if not np.isfinite([radius, tube, mass]).all() or min(radius, tube, mass) <= 0:
            raise ValueError('구명환 반지름·튜브 반지름·질량은 유한한 양수여야 합니다.')
        if segments < 3 or tube >= radius * np.cos(np.pi / segments):
            raise ValueError('구명환은 캡슐 3개 이상이고 가운데 구멍이 열려 있어야 합니다.')
        position = self._vector(cfg.INDEPENDENT_POSITION_W if self.position is None else self.position, 'position')
        quat = np.array([1., 0., 0., 0.])
        if self.attached:
            human = spec.body(names.HUMAN_BODY)
            if human is None:
                raise ValueError('착용 지지 전에 Human.build()가 필요합니다.')
            if self.position is not None:
                raise ValueError('attached=True의 초기 위치는 사람 기준으로 계산합니다.')
            offset = np.empty(3)
            mujoco.mju_rotVecQuat(offset, self._vector(cfg.POSITION_IN_HUMAN_B, 'POSITION_IN_HUMAN_B'), human.quat)
            position = np.asarray(human.pos) + offset
            quat = np.asarray(human.quat).copy()
        body = spec.worldbody.add_body(name=names.LIFEBUOY_BODY, pos=position, quat=quat)
        body.add_freejoint(name=names.LIFEBUOY_FREE)
        for index in range(segments):
            angles = 2 * np.pi * np.array([index, index + 1]) / segments
            endpoints = np.column_stack((np.zeros(2), radius * np.cos(angles), radius * np.sin(angles)))
            body.add_geom(name=f'{names.LIFEBUOY_GEOM_PREFIX}_{index}', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=endpoints.ravel(), size=[tube, 0, 0], mass=mass / segments,
                          rgba=[1., .3, .05, 1.] if index % 3 else [.95, .95, .95, 1.],
                          group=config.Groups.PHYSICS,
                          contype=config.Contacts.LIFEBUOY if self.human_contact else 0,
                          conaffinity=config.Contacts.LIFEBUOY_AFFINITY if self.human_contact else 0,
                          solref=[.006, 1], solimp=[.95, .99, .001, .5, 2])
        body.add_site(name=names.LIFEBUOY_TOW_SITE, pos=self._vector(cfg.TOW_POINT_B, 'TOW_POINT_B'),
                      size=[.02, 0, 0], rgba=[.1, .3, 1., 1.])
        if self.grip_sites:
            for name, point in ((names.LIFEBUOY_GRIP_L_SITE, cfg.GRIP_L_POINT_B),
                                (names.LIFEBUOY_GRIP_R_SITE, cfg.GRIP_R_POINT_B)):
                body.add_site(name=name, pos=self._vector(point, name), size=[.014, 0, 0], rgba=[.2, 1., .3, 1.])
        if self.attached:
            eq = spec.add_equality(name=names.HUMAN_RING_WORN_EQ, type=mujoco.mjtEq.mjEQ_WELD,
                                   objtype=mujoco.mjtObj.mjOBJ_BODY, name1=names.HUMAN_BODY,
                                   name2=names.LIFEBUOY_BODY, active=True)
            eq.data[:] = 0
            eq.data[3:6] = cfg.POSITION_IN_HUMAN_B
            eq.data[6] = eq.data[10] = 1
            eq.solref[:] = [.02, 1]

    def bind(self, model):
        try:
            self.body_id = model.body(config.Names.LIFEBUOY_BODY).id
            self.site_id = model.site(config.Names.LIFEBUOY_TOW_SITE).id
        except KeyError as error:
            raise ValueError('Lifebuoy.bind 전에 구명환 build·compile이 필요합니다.') from error
        self._model = model
        try:
            self.worn_eq_id = model.equality(config.Names.HUMAN_RING_WORN_EQ).id
        except KeyError:
            self.worn_eq_id = None

    def is_worn(self, data):
        return self.worn_eq_id is not None and bool(data.eq_active[self.worn_eq_id])

    def release_worn(self, model, data):
        """착용 지지만 해제한다. 잡기 제약·로프·실제 위치와 속도는 바꾸지 않는다."""
        if model is not self._model:
            raise ValueError('이 구명환에 bind된 모델이 필요합니다.')
        was_worn = self.is_worn(data)
        if self.worn_eq_id is not None:
            data.eq_active[self.worn_eq_id] = False
        return was_worn

    def add_forces(self, model, data, acc):
        """외력은 Water와 Tether가, 착용 지지는 엔진 equality가 계산한다."""
