"""2 로봇: 계약 선체·차동 추진. 전복·수면 연동은 후속 작업."""

import numpy as np

from sim import config


class Robot:
    REQUIRES = {}

    def __init__(self, position=(4.0, 0.0, 1.0), *, tow_point_b=None):
        self.position = self._vector(position, "position")
        # 선미 연결 위치는 미합의: 시험에서 명시한 경우에만 site를 만든다.
        self.tow_point_b = None if tow_point_b is None else self._vector(tow_point_b, "tow_point_b")
        self.body_id = self.joint_id = self.geom_id = None
        self.site_ids = None
        self._commands = np.zeros(2)

    @staticmethod
    def _vector(value, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError(f"{name}는 유한한 좌표 3개여야 합니다.")
        return result.copy()

    def build(self, spec):
        import mujoco

        hull, names = config.Hull, config.Names
        dimensions = self._vector((hull.LENGTH, hull.BEAM, hull.HEIGHT), "Hull dimensions")
        inertia = self._vector(hull.INERTIA, "Hull.INERTIA")
        mass = float(hull.MASS)
        if np.any(dimensions <= 0) or np.any(inertia <= 0) or not np.isfinite(mass) or mass <= 0:
            raise ValueError("선체 크기·질량·관성은 유한한 양수여야 합니다.")
        if hull.SHAPE != "box" or hull.OUTBOARD_SEPARATE is not False:
            raise ValueError("현재 로봇은 box 선체와 통합 선외기 질량만 지원합니다.")
        if (config.Thruster.METHOD != config.Thruster.ACC
                or config.Thruster.STEERING != "differential"
                or config.WaterAPI.USE_IN_WEEK1 is not False):
            raise ValueError("현재 로봇은 acc 차동 추진·수면 연동 비활성 계약만 지원합니다.")
        body = spec.worldbody.add_body(name=names.BOAT_BODY, pos=self.position)
        body.add_joint(name=names.BOAT_FREE, type=mujoco.mjtJoint.mjJNT_FREE)
        body.explicitinertial = True
        body.mass = mass
        body.ipos[:] = self._vector(hull.COM, "Hull.COM")
        body.inertia[:] = inertia
        body.add_geom(name=names.HULL_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=dimensions / 2, mass=0, rgba=[.15, .4, .7, 1],
                      group=0 if config.Groups.PHYSICS is None else config.Groups.PHYSICS,
                      contype=0, conaffinity=0)
        for name, position in ((names.THRUSTER_L_SITE, config.Thruster.POS_L_B),
                               (names.THRUSTER_R_SITE, config.Thruster.POS_R_B)):
            body.add_site(name=name, pos=self._vector(position, name), size=[.04, 0, 0],
                          rgba=[.9, .25, .1, 1])
        if self.tow_point_b is not None:
            body.add_site(name=names.TOW_POINT_SITE, pos=self.tow_point_b,
                          size=[.04, 0, 0], rgba=[.2, .8, 1, 1])

    def bind(self, model):
        names = config.Names
        try:
            self.body_id = model.body(names.BOAT_BODY).id
            self.joint_id = model.joint(names.BOAT_FREE).id
            self.geom_id = model.geom(names.HULL_GEOM).id
            self.site_ids = tuple(model.site(name).id for name in
                                  (names.THRUSTER_L_SITE, names.THRUSTER_R_SITE))
        except KeyError as error:
            raise ValueError("Robot.bind 전에 선체·추진기의 build·compile이 필요합니다.") from error

    def set_command(self, forward=0.0, steering=0.0):
        """정규화 명령. 양의 steering은 좌회전, 혼합 후 좌우 각각 포화한다."""
        values = np.asarray([forward, steering], dtype=float)
        lo, hi = config.Thruster.CMD_RANGE
        if not np.isfinite(values).all() or np.any(values < lo) or np.any(values > hi):
            raise ValueError(f"전진·조향 명령은 {lo}~{hi}의 유한한 값이어야 합니다.")
        self._commands[:] = np.clip([forward - steering, forward + steering], lo, hi)

    def add_forces(self, model, data, acc):
        from sim.frames import body_to_world

        if self.site_ids is None:
            raise RuntimeError("Robot.bind를 먼저 호출하세요.")
        for command, site in zip(self._commands, self.site_ids):
            maximum = config.Thruster.MAX_FORWARD if command >= 0 else config.Thruster.MAX_REVERSE
            force_w = body_to_world(data, self.body_id, [command * maximum, 0, 0])
            acc.add(self.body_id, force_w, point_w=data.site_xpos[site])
