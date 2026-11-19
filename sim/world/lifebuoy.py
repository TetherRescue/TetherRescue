"""구명환은 항상 독립 자유 강체다. 사람 몸에 끼우지 않고, 손으로 잡는 연결은 HumanGrasp가 맡는다."""
import operator
import numpy as np
from sim import config


class Lifebuoy:
    REQUIRES = {}

    def __init__(self, *, position=None, grip_sites=False, human_contact=True, held_by=None, hand='left',
                 attached=False):
        # 구명환은 사람 몸에 끼우지 않는다 (2026-11-19 가영 결정: 착용 모드 삭제).
        # attached=False는 기존 호출(다른 팀원 시험 포함)과의 호환용으로만 받는다.
        if attached:
            raise ValueError('구명환 착용(attached=True)은 삭제되었습니다. 손에 쥔 시험은 held_by=human을 쓰세요.')
        # held_by: 처음부터 손에 쥐고 있는 시험이면 그 Human(grip_sites=True)을 넘긴다.
        #   위치는 build에서 그 손의 잡기 site 옆으로 계산하고, 실제 연결은 HumanGrasp(held=True)가 한다.
        if hand not in ('left', 'right'):
            raise ValueError("hand는 left 또는 right여야 합니다.")
        if held_by is not None and position is not None:
            raise ValueError('held_by를 쓰면 초기 위치는 손 기준으로 계산합니다.')
        self.held_by = held_by
        self.hand = hand
        self.grip_sites = grip_sites or held_by is not None
        self.human_contact = human_contact
        self.position = position
        self.body_id = self.site_id = None

    @staticmethod
    def held_tow_point_b(hand='left', length_scale=1.0):
        """처음부터 손에 쥔 환의 로프 연결점을 사람 로컬 좌표로 (배치 계산용). build의 배치와 같은 식."""
        cfg, human = config.Lifebuoy, config.Human
        hand_point, ring_grip = ((human.GRIP_L_POINT_B, cfg.GRIP_L_POINT_B) if hand == 'left'
                                 else (human.GRIP_R_POINT_B, cfg.GRIP_R_POINT_B))
        return (np.asarray(hand_point, dtype=float) * length_scale - np.asarray(ring_grip, dtype=float)
                + [.02, 0., 0.] + np.asarray(cfg.TOW_POINT_B, dtype=float))

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
        if self.held_by is not None:
            human = spec.body(names.HUMAN_BODY)
            hand_name = names.HUMAN_GRIP_L_SITE if self.hand == 'left' else names.HUMAN_GRIP_R_SITE
            hand = spec.site(hand_name)
            if human is None or hand is None or hand.parent.name != names.HUMAN_BODY:
                raise ValueError('손에 쥔 구명환은 먼저 Human(grip_sites=True)를 build해야 합니다 (고정 팔 모드).')
            ring_grip = cfg.GRIP_L_POINT_B if self.hand == 'left' else cfg.GRIP_R_POINT_B
            # 손 잡기점과 환 테두리 잡기점을 맞추고, 형상이 겹치지 않게 머리 쪽(+x)으로 2cm 띄운다.
            local = np.asarray(hand.pos) - self._vector(ring_grip, 'GRIP_POINT_B') + [.02, 0., 0.]
            offset = np.empty(3)
            mujoco.mju_rotVecQuat(offset, local, human.quat)
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

    def bind(self, model):
        try:
            self.body_id = model.body(config.Names.LIFEBUOY_BODY).id
            self.site_id = model.site(config.Names.LIFEBUOY_TOW_SITE).id
        except KeyError as error:
            raise ValueError('Lifebuoy.bind 전에 구명환 build·compile이 필요합니다.') from error
        self._model = model

    def add_forces(self, model, data, acc):
        """외력은 Water와 Tether가, 손 잡기는 HumanGrasp equality가 계산한다."""
