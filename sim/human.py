"""3 세계 조립: 사람 몸 모델 (담당: 가영)."""

import numpy as np

from sim import config


class Human:
    # 이 단순 강체만으로 추가 시간 간격 제한을 확정하지 않는다.
    REQUIRES = {}

    def __init__(self, position=(0.0, 0.0, 1.0), *, shape=None, grip_sites=False, articulated_arms=False, ring_contact=False):
        # 기본 위치는 독립 시험용: 수면 기준 1m 위, 몸의 긴 방향은 월드 +x.
        # 실제 구조 장면의 배치는 world 생성 시 별도로 전달한다.
        self.position = position
        self.shape = config.Human.SHAPE if shape is None else shape
        self.grip_sites = grip_sites or articulated_arms
        self.articulated_arms = articulated_arms
        self.ring_contact = ring_contact
        self.arm_joints = {}
        self.arm_actuators = {}
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
            self._build_humanoid(spec, body, mass)
            if self.grip_sites and not self.articulated_arms:
                for name, position in ((config.Names.HUMAN_GRIP_L_SITE, cfg.GRIP_L_POINT_B),
                                        (config.Names.HUMAN_GRIP_R_SITE, cfg.GRIP_R_POINT_B)):
                    body.add_site(name=name, pos=self._vector(position, name), size=[.014, 0, 0],
                                  rgba=[.2, 1, .3, 1])
        else:
            if self.grip_sites or self.articulated_arms:
                raise ValueError("잡기 연결점은 humanoid 형상에서만 지원합니다.")
            self._build_box(body, mass)

    def _geom_options(self):
        contact = dict(solref=[.006, 1], solimp=[.95, .99, .001, .5, 2]) if self.ring_contact else {}
        return dict(
            **contact,
            group=0 if config.Groups.PHYSICS is None else config.Groups.PHYSICS,
            contype=1 if self.ring_contact else 0, conaffinity=2 if self.ring_contact else 0,
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

    def _build_humanoid(self, spec, body, mass):
        import mujoco

        cfg, names = config.Human, config.Names
        fractions = np.asarray(cfg.MASS_FRACTIONS, dtype=float)
        if (fractions.shape != (6,) or not np.isfinite(fractions).all()
                or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0)):
            raise ValueError("Human.MASS_FRACTIONS는 합이 1인 양수 6개여야 합니다.")
        skin = [0.72, 0.48, 0.32, 1]
        shirt = [0.12, 0.39, 0.52, 1]
        trousers = [0.10, 0.14, 0.20, 1]
        shoes = [0.055, 0.065, 0.075, 1]
        prefix = names.HUMAN_DETAIL_PREFIX

        def ellipsoid(name, position, size, part_mass, color):
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                          pos=self._vector(position, name), size=self._vector(size, name),
                          mass=part_mass, rgba=color, **self._geom_options())

        def capsule(name, endpoints, radius, part_mass, color):
            points = np.asarray(endpoints, dtype=float)
            if (points.shape != (2, 3) or not np.isfinite(points).all()
                    or np.linalg.norm(points[1] - points[0]) == 0
                    or not np.isfinite(radius) or radius <= 0):
                raise ValueError("사람 캡슐은 서로 다른 유한한 끝점과 양수 반지름이 필요합니다.")
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=points.ravel(), size=[radius, 0, 0],
                          mass=part_mass, rgba=color, **self._geom_options())

        # 어깨와 허리가 구분되는 단면. 단일 캡슐의 둥근 소시지 윤곽을 대체한다.
        sections = np.asarray(cfg.TORSO_SECTIONS_B, dtype=float)
        if (sections.ndim != 2 or sections.shape[1] != 3 or len(sections) < 3
                or not np.isfinite(sections).all() or np.any(sections[:, 1:] <= 0)
                or np.any(np.diff(sections[:, 0]) <= 0)):
            raise ValueError("TORSO_SECTIONS_B는 x 오름차순의 (x, 반폭, 반두께) 단면이어야 합니다.")
        angles = np.arange(16) * 2 * np.pi / 16
        vertices = np.array([[x, width * np.cos(angle), depth * np.sin(angle)]
                             for x, width, depth in sections for angle in angles])
        mesh_name = f"{prefix}_torso_mesh"
        spec.add_mesh(name=mesh_name, uservert=vertices.ravel())
        body.add_geom(name=names.HUMAN_GEOM, type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=mesh_name, mass=mass * fractions[0], rgba=shirt,
                      **self._geom_options())
        head = self._vector(cfg.HEAD_POS_B, "HEAD_POS_B")
        ellipsoid(names.HUMAN_HEAD_GEOM, head, cfg.HEAD_RADII,
                  mass * fractions[1], skin)
        # 목·얼굴·머리카락은 질량 없는 표시 형상. 얼굴은 사람 로컬 +z 방향.
        capsule(f"{prefix}_neck", [(.42, 0, 0), (.51, 0, 0)], .048, 0, skin)
        ellipsoid(f"{prefix}_hair", head + [.018, 0, -.018], [.118, .086, .086],
                  0, [.07, .045, .03, 1])
        for side in (-1, 1):
            ellipsoid(f"{prefix}_eye_{side}", head + [.024, side * .036, .091],
                      [.012, .010, .006], 0, [.045, .035, .025, 1])
            ellipsoid(f"{prefix}_ear_{side}", head + [-.006, side * .084, .005],
                      [.024, .014, .032], 0, skin)
        ellipsoid(f"{prefix}_nose", head + [-.005, 0, .103],
                  [.024, .016, .025], 0, skin)
        ellipsoid(f"{prefix}_mouth", head + [-.052, 0, .089],
                  [.006, .027, .004], 0, [.34, .14, .12, 1])

        for index, (side, arm_name, leg_name, arm, leg) in enumerate([
            ("l", names.HUMAN_ARM_L_GEOM, names.HUMAN_LEG_L_GEOM,
             cfg.ARM_L_FROMTO_B, cfg.LEG_L_FROMTO_B),
            ("r", names.HUMAN_ARM_R_GEOM, names.HUMAN_LEG_R_GEOM,
             cfg.ARM_R_FROMTO_B, cfg.LEG_R_FROMTO_B),
        ]):
            sign = 1 if side == "l" else -1
            shoulder, wrist = np.asarray(arm, dtype=float)
            elbow = self._vector(getattr(cfg, f"ELBOW_{side.upper()}_B"), "elbow")
            arm_mass = mass * fractions[2 + index]
            if self.articulated_arms:
                self._build_arm(spec, body, side, shoulder, elbow, wrist, arm_mass, arm_name)
            else:
                capsule(arm_name, [shoulder, elbow], cfg.ARM_RADIUS, arm_mass * .55, shirt)
                capsule(f"{prefix}_forearm_{side}", [elbow, wrist], cfg.ARM_RADIUS * .78,
                        arm_mass * .40, skin)
                ellipsoid(f"{prefix}_hand_{side}", wrist + [-.015, sign * .035, 0],
                          [.037, .065, .022], arm_mass * .05, skin)
            hip, ankle = np.asarray(leg, dtype=float)
            knee = self._vector(getattr(cfg, f"KNEE_{side.upper()}_B"), "knee")
            leg_mass = mass * fractions[4 + index]
            capsule(leg_name, [hip, knee], cfg.LEG_RADIUS, leg_mass * .57, trousers)
            capsule(f"{prefix}_shin_{side}", [knee, ankle], cfg.LEG_RADIUS * .70,
                    leg_mass * .38, trousers)
            ellipsoid(f"{prefix}_foot_{side}", ankle + [-.040, 0, .035],
                      [.080, .045, .045], leg_mass * .05, shoes)
        # 분절에 질량을 분배해 총 70kg 유지. articulated_arms 모드만 팔 관절을 추가한다.

    def _build_arm(self, spec, root, side, shoulder, elbow, wrist, mass, arm_name):
        """어깨 3·팔꿈치 1·손목 3 관절. 내부 토크는 엔진 actuator로 구동한다."""
        import mujoco

        names, cfg = config.Names, config.Human
        # 관절 팔의 0도는 실제 일직선이다. 고정 팔의 기존 형상은 유지한다.
        upper_vector = elbow - shoulder
        forearm_vector = upper_vector / np.linalg.norm(upper_vector) * np.linalg.norm(wrist - elbow)
        wrist = elbow + forearm_vector
        elbow_axis = np.cross(upper_vector, [0., 0., 1.])
        elbow_axis /= np.linalg.norm(elbow_axis)
        upper = root.add_body(name=f"{names.HUMAN_ARM_BODY_PREFIX}_{side}_upper", pos=shoulder)
        forearm = upper.add_body(name=f"{names.HUMAN_ARM_BODY_PREFIX}_{side}_forearm", pos=elbow - shoulder)
        hand = forearm.add_body(name=f"{names.HUMAN_ARM_BODY_PREFIX}_{side}_hand", pos=wrist - elbow)
        for link, label, axes in ((upper, 'shoulder', np.eye(3)),
                                  (forearm, 'elbow', [elbow_axis]),
                                  (hand, 'wrist', np.eye(3))):
            for index, axis in enumerate(axes):
                name = f"{names.HUMAN_ARM_JOINT_PREFIX}_{side}_{label}_{index}"
                # 로컬 시험 범위. 팔꿈치 역꺾임·여러 바퀴 회전을 막는다.
                limits = ((0., 2.55) if label == 'elbow' else
                          ((-3.1, 3.1), (-1.5, 1.5), (-3.1, 3.1))[index] if label == 'shoulder'
                          else (-1.7, 1.7))
                joint_range = np.rad2deg(limits) if spec.compiler.degree else limits
                link.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis,
                               limited=True, range=joint_range, damping=[2., 0, 0], armature=.02,
                               solref_limit=[.008, 1])
                actuator = spec.add_actuator(name=f"{names.HUMAN_ARM_ACTUATOR_PREFIX}_{side}_{label}_{index}",
                                             trntype=mujoco.mjtTrn.mjTRN_JOINT, target=name,
                                             gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                                             biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                                             forcelimited=True,
                                             forcerange=[-22., 22.] if label == 'wrist' else [-40., 40.])
                actuator.gainprm[0] = 110.
                actuator.biasprm[:3] = [0., -110., -20.]
        skin, shirt = [.72, .48, .32, 1], [.12, .39, .52, 1]
        prefix = names.HUMAN_DETAIL_PREFIX
        upper.add_geom(name=arm_name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=np.r_[np.zeros(3), elbow - shoulder], size=[cfg.ARM_RADIUS, 0, 0],
                       mass=mass * .55, rgba=shirt, **self._geom_options())
        forearm.add_geom(name=f"{prefix}_forearm_{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                         fromto=np.r_[np.zeros(3), wrist - elbow], size=[cfg.ARM_RADIUS * .78, 0, 0],
                         mass=mass * .4, rgba=skin, **self._geom_options())
        sign = 1 if side == 'l' else -1
        hand.add_geom(name=f"{prefix}_hand_{side}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      pos=[-.015, sign * .035, 0], size=[.037, .065, .022],
                      mass=mass * .05, rgba=skin, **self._geom_options())
        grip_name = names.HUMAN_GRIP_L_SITE if side == 'l' else names.HUMAN_GRIP_R_SITE
        hand.add_site(name=grip_name, pos=[.022, sign * .035, 0], size=[.014, 0, 0], rgba=[.2, 1, .3, 1])

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
            bodies = {body_id}
            for i in range(body_id + 1, model.nbody):
                if int(model.body_parentid[i]) in bodies:
                    bodies.add(i)
            geom_ids = {model.geom(i).name: i for i in range(model.ngeom)
                        if model.geom_bodyid[i] in bodies
                        and model.geom(i).name.startswith(('human_', config.Names.HUMAN_GEOM))}
        except KeyError as error:
            raise ValueError("Human.bind 전에 사람 모델의 build·compile이 필요합니다.") from error
        self.body_id, self.joint_id, self.geom_id = body_id, joint_id, geom_id
        self.geom_ids = geom_ids
        if self.articulated_arms:
            for side in ('l', 'r'):
                self.arm_joints[side] = [i for i in range(model.njnt)
                                         if model.joint(i).name.startswith(f"{config.Names.HUMAN_ARM_JOINT_PREFIX}_{side}_")]
                self.arm_actuators[side] = [i for i in range(model.nu)
                                            if model.actuator(i).name.startswith(f"{config.Names.HUMAN_ARM_ACTUATOR_PREFIX}_{side}_")]

    def add_forces(self, model, data, acc):
        """현재 추가 외력 없음. 중력은 MuJoCo가 처리한다.

        실험 부력·물 저항은 Water가 명시적 대상으로 선택해 적용한다. 중력을 다시 add하지 않는다.
        """
        pass
