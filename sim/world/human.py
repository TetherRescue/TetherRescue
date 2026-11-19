"""3 세계 조립: 사람 몸 모델 (담당: 가영)."""

import numpy as np

from sim import config


class Human:
    # 이 단순 강체만으로 추가 시간 간격 제한을 확정하지 않는다.
    REQUIRES = {}

    def __init__(self, position=(0.0, 0.0, 1.0), *, yaw=0.0, shape=None, size=None, grip_sites=False, articulated_arms=False,
                 ring_contact=True):
        # 기본 위치는 독립 시험용: 수면 기준 1m 위, 몸의 긴 방향은 월드 +x.
        # 실제 구조 장면의 배치는 world 생성 시 별도로 전달한다.
        self.position = position
        # yaw: 수평면 회전 [rad]. 0이면 머리가 월드 +x. 구조 장면에서 몸 방향을 정할 때 쓴다.
        self.yaw = float(yaw)
        self.shape = config.Human.SHAPE if shape is None else shape
        self.size = config.Human.SIZE if size is None else size
        if self.size not in config.Human.SIZES:
            raise ValueError(f"Human size는 {tuple(config.Human.SIZES)} 중 하나여야 합니다.")
        # 키·질량·비율은 build에서 그때의 config로 계산한다.
        self.height = self.mass = self.length_scale = self.girth_scale = None
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
        # config가 생성 뒤 바뀌어도(시험의 monkeypatch 등) build 시점 값을 쓴다.
        self.height, self.mass, self.length_scale, self.girth_scale = self._size_scales(self.size)
        mass = self.mass
        position = self._vector(self.position, "position")
        s = self.length_scale

        if not np.isfinite(self.yaw):
            raise ValueError("yaw는 유한한 각도여야 합니다.")
        quat = [np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)]
        body = spec.worldbody.add_body(name=config.Names.HUMAN_BODY, pos=position, quat=quat)
        body.add_joint(name=config.Names.HUMAN_FREE, type=mujoco.mjtJoint.mjJNT_FREE)

        if self.shape == "humanoid":
            self._build_humanoid(spec, body, mass)
            if self.grip_sites and not self.articulated_arms:
                for name, position in ((config.Names.HUMAN_GRIP_L_SITE, cfg.GRIP_L_POINT_B),
                                        (config.Names.HUMAN_GRIP_R_SITE, cfg.GRIP_R_POINT_B)):
                    body.add_site(name=name, pos=self._vector(position, name) * s, size=[.014, 0, 0],
                                  rgba=[.2, 1, .3, 1])
        else:
            if self.grip_sites or self.articulated_arms:
                raise ValueError("잡기 연결점은 humanoid 형상에서만 지원합니다.")
            self._build_box(body, mass)

    @staticmethod
    def _size_scales(size):
        """몸집 이름 → (키 m, 질량 kg, 길이 비율 s, 굵기 비율 s·k). config.Human.SIZE·SIZES 참고."""
        cfg = config.Human
        if size not in cfg.SIZES:
            raise ValueError(f"Human size는 {tuple(cfg.SIZES)} 중 하나여야 합니다.")
        height, mass = cfg.SIZES[size]
        reference = float(cfg.REFERENCE_HEIGHT)
        height = reference if height is None else float(height)
        mass = float(cfg.MASS) if mass is None else float(mass)
        if not np.isfinite([height, mass, reference]).all() or min(height, mass, reference) <= 0:
            raise ValueError("사람 질량·키는 유한한 양수여야 합니다.")
        s = height / reference
        # 부피 ∝ s³k² 를 질량비에 맞춰 평균 밀도를 유지한다.
        k = np.sqrt(mass / float(cfg.MASS) / s**3) if np.isfinite(cfg.MASS) and cfg.MASS > 0 else 1.0
        return height, mass, s, s * k

    def _geom_options(self, *, visual=False):
        if visual:
            return dict(group=config.Groups.VISUAL, contype=0, conaffinity=0)
        contact = dict(solref=[.006, 1], solimp=[.95, .99, .001, .5, 2]) if self.ring_contact else {}
        return dict(
            **contact,
            group=config.Groups.PHYSICS,
            contype=config.Contacts.HUMAN if self.ring_contact else 0,
            conaffinity=config.Contacts.HUMAN_AFFINITY if self.ring_contact else 0,
        )

    def _build_box(self, body, mass):
        import mujoco

        cfg = config.Human
        dimensions = np.array([cfg.LENGTH * self.length_scale, cfg.WIDTH * self.girth_scale,
                               cfg.HEIGHT * self.girth_scale], dtype=float)
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
        arm_split = self._segment_fractions("ARM_SEGMENT_FRACTIONS")
        leg_split = self._segment_fractions("LEG_SEGMENT_FRACTIONS")
        skin = [0.72, 0.48, 0.32, 1]
        shirt = [0.12, 0.39, 0.52, 1]
        trousers = [0.10, 0.14, 0.20, 1]
        shoes = [0.055, 0.065, 0.075, 1]
        prefix = names.HUMAN_DETAIL_PREFIX
        # s: 길이 비율(위치·머리·손발·얼굴), g: 몸통·팔다리 굵기 비율. medium은 둘 다 1.
        s, g = self.length_scale, self.girth_scale

        def ellipsoid(name, position, size, part_mass, color):
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                          pos=self._vector(position, name), size=self._vector(size, name),
                          mass=part_mass, rgba=color, **self._geom_options(visual=part_mass == 0))

        def capsule(name, endpoints, radius, part_mass, color):
            points = np.asarray(endpoints, dtype=float)
            if (points.shape != (2, 3) or not np.isfinite(points).all()
                    or np.linalg.norm(points[1] - points[0]) == 0
                    or not np.isfinite(radius) or radius <= 0):
                raise ValueError("사람 캡슐은 서로 다른 유한한 끝점과 양수 반지름이 필요합니다.")
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=points.ravel(), size=[radius, 0, 0],
                          mass=part_mass, rgba=color, **self._geom_options(visual=part_mass == 0))

        # 어깨와 허리가 구분되는 단면. 단일 캡슐의 둥근 소시지 윤곽을 대체한다.
        sections = np.asarray(cfg.TORSO_SECTIONS_B, dtype=float)
        if (sections.ndim != 2 or sections.shape[1] != 3 or len(sections) < 3
                or not np.isfinite(sections).all() or np.any(sections[:, 1:] <= 0)
                or np.any(np.diff(sections[:, 0]) <= 0)):
            raise ValueError("TORSO_SECTIONS_B는 x 오름차순의 (x, 반폭, 반두께) 단면이어야 합니다.")
        sections = sections * [s, g, g]
        angles = np.arange(16) * 2 * np.pi / 16
        vertices = np.array([[x, width * np.cos(angle), depth * np.sin(angle)]
                             for x, width, depth in sections for angle in angles])
        mesh_name = f"{prefix}_torso_mesh"
        spec.add_mesh(name=mesh_name, uservert=vertices.ravel())
        body.add_geom(name=names.HUMAN_GEOM, type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=mesh_name, mass=mass * fractions[0], rgba=shirt,
                      **self._geom_options())
        head = self._vector(cfg.HEAD_POS_B, "HEAD_POS_B") * s
        ellipsoid(names.HUMAN_HEAD_GEOM, head, np.asarray(cfg.HEAD_RADII, dtype=float) * s,
                  mass * fractions[1], skin)
        # 목·얼굴·머리카락은 질량 없는 표시 형상. 얼굴은 사람 로컬 +z 방향.
        capsule(f"{prefix}_neck", np.array([(.42, 0, 0), (.51, 0, 0)]) * s, .048 * g, 0, skin)
        ellipsoid(f"{prefix}_hair", head + np.array([.018, 0, -.018]) * s, np.array([.118, .086, .086]) * s,
                  0, [.07, .045, .03, 1])
        for side in (-1, 1):
            ellipsoid(f"{prefix}_eye_{side}", head + np.array([.024, side * .036, .091]) * s,
                      np.array([.012, .010, .006]) * s, 0, [.045, .035, .025, 1])
            ellipsoid(f"{prefix}_ear_{side}", head + np.array([-.006, side * .084, .005]) * s,
                      np.array([.024, .014, .032]) * s, 0, skin)
        ellipsoid(f"{prefix}_nose", head + np.array([-.005, 0, .103]) * s,
                  np.array([.024, .016, .025]) * s, 0, skin)
        ellipsoid(f"{prefix}_mouth", head + np.array([-.052, 0, .089]) * s,
                  np.array([.006, .027, .004]) * s, 0, [.34, .14, .12, 1])

        for index, (side, arm_name, leg_name, arm, leg) in enumerate([
            ("l", names.HUMAN_ARM_L_GEOM, names.HUMAN_LEG_L_GEOM,
             cfg.ARM_L_FROMTO_B, cfg.LEG_L_FROMTO_B),
            ("r", names.HUMAN_ARM_R_GEOM, names.HUMAN_LEG_R_GEOM,
             cfg.ARM_R_FROMTO_B, cfg.LEG_R_FROMTO_B),
        ]):
            sign = 1 if side == "l" else -1
            shoulder, wrist = np.asarray(arm, dtype=float) * s
            elbow = self._vector(getattr(cfg, f"ELBOW_{side.upper()}_B"), "elbow") * s
            arm_mass = mass * fractions[2 + index]
            if self.articulated_arms:
                self._build_arm(spec, body, side, shoulder, elbow, wrist, arm_mass, arm_name)
            else:
                capsule(arm_name, [shoulder, elbow], cfg.ARM_RADIUS * g, arm_mass * arm_split[0], shirt)
                capsule(f"{prefix}_forearm_{side}", [elbow, wrist], cfg.ARM_RADIUS * .78 * g,
                        arm_mass * arm_split[1], skin)
                ellipsoid(f"{prefix}_hand_{side}", wrist + np.array([-.015, sign * .035, 0]) * s,
                          np.array([.037, .065, .022]) * s, arm_mass * arm_split[2], skin)
            hip, ankle = np.asarray(leg, dtype=float) * s
            knee = self._vector(getattr(cfg, f"KNEE_{side.upper()}_B"), "knee") * s
            leg_mass = mass * fractions[4 + index]
            capsule(leg_name, [hip, knee], cfg.LEG_RADIUS * g, leg_mass * leg_split[0], trousers)
            capsule(f"{prefix}_shin_{side}", [knee, ankle], cfg.LEG_RADIUS * .70 * g,
                    leg_mass * leg_split[1], trousers)
            ellipsoid(f"{prefix}_foot_{side}", ankle + np.array([-.040, 0, .035]) * s,
                      np.array([.080, .045, .045]) * s, leg_mass * leg_split[2], shoes)
        # 분절에 질량을 분배해 몸집별 총질량 유지. articulated_arms 모드만 팔 관절을 추가한다.

    def _build_arm(self, spec, root, side, shoulder, elbow, wrist, mass, arm_name):
        """어깨 3·팔꿈치 1·손목 3 관절. 내부 토크는 엔진 actuator로 구동한다."""
        import mujoco

        names, cfg = config.Names, config.Human
        arm_split = self._segment_fractions("ARM_SEGMENT_FRACTIONS")
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
        s, g = self.length_scale, self.girth_scale
        upper.add_geom(name=arm_name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=np.r_[np.zeros(3), elbow - shoulder], size=[cfg.ARM_RADIUS * g, 0, 0],
                       mass=mass * arm_split[0], rgba=shirt, **self._geom_options())
        forearm.add_geom(name=f"{prefix}_forearm_{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                         fromto=np.r_[np.zeros(3), wrist - elbow], size=[cfg.ARM_RADIUS * .78 * g, 0, 0],
                         mass=mass * arm_split[1], rgba=skin, **self._geom_options())
        sign = 1 if side == 'l' else -1
        hand.add_geom(name=f"{prefix}_hand_{side}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      pos=np.array([-.015, sign * .035, 0]) * s, size=np.array([.037, .065, .022]) * s,
                      mass=mass * arm_split[2], rgba=skin, **self._geom_options())
        grip_name = names.HUMAN_GRIP_L_SITE if side == 'l' else names.HUMAN_GRIP_R_SITE
        hand.add_site(name=grip_name, pos=np.array([.022, sign * .035, 0]) * s, size=[.014, 0, 0],
                      rgba=[.2, 1, .3, 1])

    @staticmethod
    def _segment_fractions(name):
        """팔·다리 안쪽 세 분절의 질량 비율 (합 1인 양수 3개)."""
        fractions = np.asarray(getattr(config.Human, name), dtype=float)
        if (fractions.shape != (3,) or not np.isfinite(fractions).all()
                or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0)):
            raise ValueError(f"Human.{name}는 합이 1인 양수 3개여야 합니다.")
        return fractions

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
