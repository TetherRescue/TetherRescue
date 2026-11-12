"""3 세계 조립: 줄 (tendon 스프링, 끊김) (담당: 가영)."""

import numpy as np

from sim import config


class Tether:
    REQUIRES = {}

    def __init__(self, *, enabled=False):
        # 연결 시점 합의 전: 기본 세계는 미연결, 시험에서는 enabled=True로 명시한다.
        self.enabled = enabled
        self.tendon_id = None
        self.site_ids = None
        self.body_ids = None
        self.damping = None

    def build(self, spec):
        """실제 존재하는 선미·구명환 site를 인장 전용 스프링으로 연결한다."""
        if not self.enabled:
            return
        values = np.asarray([config.Tether.REST_LENGTH, config.Tether.STIFFNESS,
                             config.Tether.DAMPING], dtype=float)
        if (not np.isfinite(values).all() or values[0] <= 0
                or values[1] <= 0 or values[2] < 0):
            raise ValueError("줄 길이·강성은 양수, 감쇠는 0 이상의 유한한 값이어야 합니다.")
        names = (config.Names.TOW_POINT_SITE, config.Names.LIFEBUOY_TOW_SITE)
        for name in names:
            if spec.site(name) is None:
                raise ValueError(f"Tether.build 전에 연결점 '{name}'을 생성해야 합니다.")
        rest_length, stiffness, self.damping = values
        tendon = spec.add_tendon(name=config.Names.TETHER_TENDON)
        # 0~기준길이는 스프링 힘이 없는 구간. 길이가 더 길 때만 엔진이 당긴다.
        tendon.springlength[:] = (0.0, rest_length)
        tendon.stiffness[:] = (stiffness, 0.0, 0.0)
        # 내장 감쇠는 느슨할 때도 작용할 수 있으므로 add_forces에서 인장 보정만 계산한다.
        tendon.damping[:] = 0.0
        tendon.width = 0.008
        tendon.rgba[:] = (0.95, 0.85, 0.3, 1.0)
        for name in names:
            tendon.wrap_site(name)

    def bind(self, model):
        """tendon과 양 끝점·몸체 ID를 찾아 저장한다."""
        if not self.enabled:
            return
        try:
            tendon_id = model.tendon(config.Names.TETHER_TENDON).id
            site_ids = (model.site(config.Names.TOW_POINT_SITE).id,
                        model.site(config.Names.LIFEBUOY_TOW_SITE).id)
        except KeyError as error:
            raise ValueError("Tether.bind 전에 연결점·tendon의 build·compile이 필요합니다.") from error
        self.tendon_id = tendon_id
        self.site_ids = site_ids
        self.body_ids = tuple(int(model.site_bodyid[site]) for site in site_ids)

    def _forces(self, model, data):
        if self.tendon_id is None:
            raise RuntimeError("연결된 Tether의 bind를 먼저 호출하세요.")
        length = float(data.ten_length[self.tendon_id])
        rest_length = model.tendon_lengthspring[self.tendon_id, 1]
        stiffness = model.tendon_stiffness[self.tendon_id]
        spring = stiffness * max(0.0, length - rest_length)
        # Ldot>0은 늘어남. 줄이 줄어들 때도 총장력은 음수(압축)가 될 수 없다.
        tension = (max(0.0, spring + self.damping * data.ten_velocity[self.tendon_id])
                   if length > rest_length else 0.0)
        return float(tension), float(spring)

    def tension(self, model, data):
        """현재 상태의 총장력[N]. 호출 전 mj_forward로 위치·속도를 갱신한다."""
        return self._forces(model, data)[0] if self.enabled else 0.0

    def add_forces(self, model, data, acc):
        """엔진 스프링에 더할 감쇠 보정만 전달한다. 스프링 힘 자체는 중복하지 않는다."""
        if not self.enabled:
            return
        tension, spring = self._forces(model, data)
        correction = tension - spring
        if correction == 0.0:
            return
        start, end = (data.site_xpos[site] for site in self.site_ids)
        distance = np.linalg.norm(end - start)
        if distance == 0.0:
            return
        force = correction * (end - start) / distance
        # 당기는 힘과 반작용을 양쪽 작용점에 더한다. 월드 고정점(body 0)은 생략한다.
        for body, point, value in zip(self.body_ids, (start, end), (force, -force)):
            if body != 0:
                acc.add(body, value, point_w=point)
