"""1 물: 파도·복셀 부력·저항·부가질량 (담당: 미현).

설정값은 전부 config.WaterParams(물 설정)·config.Physics(밀도·중력)·config.Hull(선체)에서 읽는다.
물성 계수는 config에서 읽고, 복셀 적분의 수치 연산은 이 모듈에서 수행한다.
"""

import math

import numpy as np

from sim import config


# ─────────────────────────────────────────────────────────────
# 파도: 수면 높이
# ─────────────────────────────────────────────────────────────
def wave_height(x, y, t, waves=config.WaterParams.WAVES_DEFAULT):
    """(x, y) 위치, t 초에서의 수면 높이 z [m] (월드 좌표, ENU).

    x, y 는 숫자 하나도, numpy 배열도 된다 (WaterAPI.WAVE_HEIGHT_VECTORIZED = True).
      - 숫자를 넣으면 숫자(float) 하나가 나온다.
      - 배열을 넣으면 같은 모양의 배열이 나온다. 예: 복셀 128개 위치를 한 번에 물어볼 수 있다.
    t 는 숫자 하나 (지금 시각).

    파도 = 사인파 성분들의 합. 성분 하나 = (진폭, 파장, 진행 방향, 위상)
      높이 = 진폭 · cos(파수 · (진행 방향으로 잰 거리) − 각진동수 · t + 위상)
      파수 = 2π / 파장,  각진동수 = sqrt(g · 파수)  ("깊은 물" 공식)
    waves=WaterParams.WAVES_CALM(빈 튜플)이면 어디서나 수면 높이 = SURFACE_Z.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.full(np.broadcast(x, y).shape, config.WaterParams.SURFACE_Z, dtype=float)

    g = config.Physics.GRAVITY
    for amp, length, direction, phase in waves:
        k = 2.0 * math.pi / length       # 파수: 1 m 당 물결이 몇 rad 만큼 바뀌나
        omega = math.sqrt(g * k)         # 각진동수: 1 초 당 몇 rad 만큼 오르내리나
        along = x * math.cos(direction) + y * math.sin(direction)  # 진행 방향으로 잰 거리
        z = z + amp * np.cos(k * along - omega * t + phase)

    return float(z) if z.ndim == 0 else z


# ─────────────────────────────────────────────────────────────
# 회전 감쇠: 감쇠비 → 계수
# ─────────────────────────────────────────────────────────────
def restoring_stiffness(mass):
    """배를 살짝 기울였을 때 되돌아오려는 힘의 세기 k [N·m/rad] = (roll, pitch).

    직육면체 선체(config.Hull) 공식:
      흘수 T = m / (rho · L · B)            # 물에 잠기는 깊이
      KB = T / 2                            # 바닥 → 부력 중심 (잠긴 부분의 가운데)
      BM = 폭² / (12 · T)                   # 부력 중심 → 메타센터 (roll 은 폭 B, pitch 는 길이 L)
      KG = HEIGHT / 2 + COM_z               # 바닥 → 무게중심 (원점이 상자 중심, COM_z 는 원점 기준)
      GM = KB + BM − KG                     # 메타센터 높이: 클수록 오뚝이처럼 잘 일어남
      k  = m · g · GM

    mass 는 MuJoCo 모델에서 읽은 배 무게(body_subtreemass)를 넣는다.
    GM ≤ 0 이면 배가 스스로 일어나지 못해 뒤집히는 설계라 바로 에러.
    """
    rho = config.Physics.RHO_WATER
    g = config.Physics.GRAVITY
    L, B, H = config.Hull.LENGTH, config.Hull.BEAM, config.Hull.HEIGHT

    draft = mass / (rho * L * B)
    kb = draft / 2.0
    kg = H / 2.0 + config.Hull.COM[2]

    stiffness = []
    for name, width in (("roll", B), ("pitch", L)):
        gm = kb + width ** 2 / (12.0 * draft) - kg
        if gm <= 0:
            raise ValueError(
                f"{name} 메타센터 높이 GM = {gm:.3f} m ≤ 0 → 기울면 되돌아오지 못하고 뒤집힘. "
                "무게중심(Hull.COM)을 낮추거나 선체 크기를 확인할 것."
            )
        stiffness.append(mass * g * gm)
    return tuple(stiffness)


def rot_drag_linear(mass, inertia):
    """회전 속도에 비례하는 저항 계수 [N·m·s/rad] = (roll, pitch, yaw).

    roll·pitch: 감쇠비 WaterParams.ROT_DAMPING_RATIO 에서 c = 2 · zeta · sqrt(k · I) 로 계산.
    yaw: 되돌아오려는 힘이 없어 감쇠비를 쓸 수 없으므로 WaterParams.ROT_DRAG_LINEAR[2] 를 그대로 쓴다.

    mass    = 배 무게 [kg]           (MuJoCo model.body_subtreemass)
    inertia = (roll, pitch, yaw) 관성 [kg·m²]  (MuJoCo model.body_inertia)
    """
    coeffs = []
    for zeta, k, inertia_axis in zip(config.WaterParams.ROT_DAMPING_RATIO, restoring_stiffness(mass), inertia):
        coeffs.append(2.0 * zeta * math.sqrt(k * inertia_axis))
    coeffs.append(config.WaterParams.ROT_DRAG_LINEAR[2])
    return tuple(coeffs)


# ─────────────────────────────────────────────────────────────
# 물 모듈 본체
# ─────────────────────────────────────────────────────────────
class Water:
    # 요구사항은 config 한 곳에만 적고 여기서는 가져다 쓴다 (값을 두 곳에 두지 않기).
    REQUIRES = config.WaterParams.REQUIRES

    def __init__(self, waves=config.WaterParams.WAVES_DEFAULT, *, floating_bodies=(), surface_visual=False, include_boat=True):
        self.waves = waves
        self.surface_visual = surface_visual
        self.include_boat = include_boat
        # 사람·구명환의 실험 부력은 명시적으로 선택한다. 선체 부력은 기본 적용.
        self.floating_bodies = tuple(floating_bodies)
        self.targets = []

    def wave_height(self, x, y, t):
        """이 물 객체의 파도로 계산한 수면 높이. 설명은 모듈 함수 wave_height 참고."""
        return wave_height(x, y, t, self.waves)

    def build(self, spec):
        """선택한 수면 표시는 접촉·질량이 없는 평면이다 (파도 변형은 미표시)."""
        if self.surface_visual:
            import mujoco
            spec.worldbody.add_geom(name=config.Names.WATER_SURFACE_GEOM,
                                    type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    pos=[0, 0, config.WaterParams.SURFACE_Z], size=[50, 50, .1],
                                    rgba=[.06, .32, .45, .4], contype=0, conaffinity=0,
                                    group=0 if config.Groups.VISUAL is None else config.Groups.VISUAL)
            spec.worldbody.add_light(pos=[0, 0, 8], dir=[0, 0, -1], diffuse=[.7, .7, .7])

    def bind(self, model):
        """선체 복셀과 명시한 고정 연결 몸체의 합집합 복셀을 준비한다."""
        from sim.buoyancy import body_voxels
        import mujoco

        if config.WaterParams.ADDED_MASS_ENABLED is not False:
            raise ValueError("부가질량은 아직 지원하지 않습니다.")
        self.targets = []
        seen = set()
        if self.include_boat:
            body_id = model.body(config.Names.BOAT_BODY).id
            geom_id = model.geom(config.Names.HULL_GEOM).id
            if model.geom_bodyid[geom_id] != body_id or int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError("선체 부력에는 boat body에 속한 box hull이 필요합니다.")
            size = model.geom_size[geom_id].copy()
            counts = np.asarray(config.WaterParams.VOXEL_RES, dtype=int)
            if counts.shape != (3,) or np.any(counts < 1):
                raise ValueError("VOXEL_RES는 양의 정수 3개여야 합니다.")
            cell = 2 * size / counts
            axes = [-size[i] + (np.arange(counts[i]) + .5) * cell[i] for i in range(3)]
            points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
            # hull geom의 위치·회전도 body 좌표로 반영한다.
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
            rotation = rotation.reshape(3, 3)
            if not np.allclose(rotation, np.eye(3)):
                raise ValueError("현재 선체 부력은 body 축과 정렬된 box hull만 지원합니다.")
            points += model.geom_pos[geom_id]
            mass = model.body_subtreemass[body_id]
            linear = np.array([*config.WaterParams.DRAG_LINEAR[:2],
                               2 * config.WaterParams.HEAVE_DAMPING_RATIO * np.sqrt(
                                   mass * config.Physics.RHO_WATER * np.linalg.norm(model.opt.gravity)
                                   * 4 * size[0] * size[1])])
            self.targets.append(self._target(body_id, points, cell / 2, 2 * size, linear,
                                            rot_drag_linear(mass, model.body_inertia[body_id]),
                                            config.WaterParams.DRAG_CD,
                                            config.WaterParams.ROT_DRAG_QUAD))
            seen.add(body_id)
        for name in self.floating_bodies:
            extra_id = model.body(name).id
            if extra_id in seen or extra_id == 0:
                raise ValueError("부력 대상 몸체가 중복되거나 world입니다.")
            # 부모와 자식을 동시에 지정하면 같은 부피에 힘을 두 번 넣게 된다.
            for other in seen:
                for child, ancestor in ((extra_id, other), (other, extra_id)):
                    while child and child != ancestor:
                        child = int(model.body_parentid[child])
                    if child == ancestor:
                        raise ValueError("부력 대상은 서로 독립된 몸체여야 합니다.")
            seen.add(extra_id)
            points, half, dimensions = body_voxels(model, extra_id, config.RescueWater.VOXEL_SIZE)
            self.targets.append(self._target(extra_id, points, half, dimensions,
                                            config.RescueWater.DRAG_LINEAR,
                                            config.RescueWater.ROT_DRAG_LINEAR,
                                            config.RescueWater.DRAG_CD,
                                            config.RescueWater.ROT_DRAG_QUAD))

    @staticmethod
    def _target(body_id, points, half, dimensions, linear, angular, cd, angular_quad):
        coefficients = np.asarray([linear, angular, cd, angular_quad], dtype=float)
        if coefficients.shape != (4, 3) or not np.isfinite(coefficients).all() or np.any(coefficients < 0):
            raise ValueError("물 저항 계수는 유한한 0 이상 값이어야 합니다.")
        return dict(body_id=body_id, points=points, half=half,
                    cell_volume=float(np.prod(half * 2)), dimensions=dimensions,
                    linear=coefficients[0], angular=coefficients[1],
                    cd=coefficients[2], angular_quad=coefficients[3])

    def add_forces(self, model, data, acc):
        """부력은 잠긴 복셀 중심에, 저항은 현재 물 상대 속도에 적용한다.

        회전된 복셀의 수직 투영 높이로 부분 침수를 근사한다.
        파도는 수면 높이만 바꾸며 물 입자의 파도 속도는 아직 계산하지 않는다.
        """
        from sim.frames import body_to_world, world_to_body, point_body_to_world, body_velocity_w

        gravity = float(np.linalg.norm(model.opt.gravity))
        if model.opt.gravity[0] != 0 or model.opt.gravity[1] != 0 or model.opt.gravity[2] > 0:
            raise ValueError("물 모듈은 월드 -z 방향 중력만 지원합니다.")
        rho = config.Physics.RHO_WATER
        current = np.asarray(config.WaterParams.WATER_VELOCITY_W, dtype=float)
        if current.shape != (3,) or not np.isfinite(current).all():
            raise ValueError("물 속도는 유한한 월드 벡터여야 합니다.")
        for target in self.targets:
            body_id = target['body_id']
            points_w = point_body_to_world(data, body_id, target['points'])
            rotation = data.xmat[body_id].reshape(3, 3)
            height = 2 * (np.abs(rotation[2]) @ target['half'])
            surface = self.wave_height(points_w[:, 0], points_w[:, 1], data.time)
            wet = np.clip(.5 + (surface - points_w[:, 2]) / height, 0, 1)
            total_wet = float(wet.sum())
            if total_wet == 0:
                continue
            volume = total_wet * target['cell_volume']
            # 부분 침수 복셀의 잠긴 부분 중심. 직립 box에서는 정확한 부력 중심이다.
            centers_w = points_w.copy()
            centers_w[:, 2] += (wet - 1) * height / 2
            center_w = np.sum(centers_w * wet[:, None], axis=0) / total_wet
            acc.add(body_id, [0, 0, rho * gravity * volume], point_w=center_w)
            velocity_w, omega_w = body_velocity_w(model, data, body_id)
            velocity_b = world_to_body(data, body_id, velocity_w - current)
            omega_b = world_to_body(data, body_id, omega_w)
            wet_ratio = total_wet / len(wet)
            # 평형 잠김량에서는 감쇠비 계약값이 그대로 적용되고, 물 밖에서는 사라진다.
            damping_scale = min(1., volume * rho / model.body_subtreemass[body_id])
            dimensions = target['dimensions']
            areas = np.array([dimensions[1] * dimensions[2], dimensions[0] * dimensions[2],
                              dimensions[0] * dimensions[1]]) * wet_ratio
            drag_b = (-target['linear'] * damping_scale * velocity_b
                      - .5 * rho * target['cd'] * areas * np.abs(velocity_b) * velocity_b)
            torque_b = (-target['angular'] * omega_b
                        - target['angular_quad'] * np.abs(omega_b) * omega_b) * damping_scale
            acc.add(body_id, body_to_world(data, body_id, drag_b),
                    torque_w=body_to_world(data, body_id, torque_b))
