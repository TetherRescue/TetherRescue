"""하나개 해수욕장 참고 해안 맵 (담당: 가영).

동쪽 끝에 육지를 붙이고, 서쪽 넓은 바다에 익수자·배를 둔다.
실측 GIS·밀물 수면 변화·바닥 충돌은 포함하지 않는다.
맵 형상은 Groups.VISUAL·비접촉이다. 물은 Water의 부력만 쓰고 수면 표시는 맵이 맡는다.
"""

import math

import mujoco
import numpy as np

from sim import config


# 시연용 배치 (m). ENU: +x 동(육지), −x 서(바다), +y 북.
SHORE_X = 0.0                 # 해안선
LAND_EAST = 80.0              # 육지 동쪽 끝
LAND_DEPTH = 8.0              # 지면이 수면 아래로 내려가는 두께 (떠 보이지 않게)
LAND_TOP = 1.5                # 내륙 지면 높이
BEACH_WIDTH = 12.0
FLAT_WIDTH = 25.0             # 해안 앞 갯벌
SEA_WEST = -260.0             # 바다 서쪽 끝 (익수자 200m + 여유)
SEA_HALF_Y = 80.0
VICTIM_RANGE = 200.0          # 해안선에서 익수자까지 거리
SEA_BOUNDS = (SEA_WEST, SHORE_X, -SEA_HALF_Y, SEA_HALF_Y)   # 수면을 그리는 범위 (x0, x1, y0, y1)
SEA_RGBA = (.06, .32, .45, .45)


def _visual_box(spec, *, name, pos, size, rgba):
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=size,
        rgba=rgba,
        contype=0,
        conaffinity=0,
        group=config.Groups.VISUAL,
    )


def _box_from_bounds(spec, *, name, x0, x1, y0, y1, z0, z1, rgba):
    """모서리 좌표로 상자 중심·반크기를 잡아 지면이 끊기지 않게 붙인다."""
    pos = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    size = ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)
    _visual_box(spec, name=name, pos=pos, size=size, rgba=rgba)


class CoastalMap:
    """해안 풍경 조립 부품. Water(surface_visual=False)·Robot·Human과 함께 쓴다."""

    REQUIRES = {}

    def __init__(self, *, scene='hanagae', wave_amplitude=0.0):
        """wave_amplitude: 파도 진폭 합 [m]. 0 보다 크면 맵은 바다 판을 그리지 않고 갯벌을 파도 골 아래로 내린다.
        파도가 있을 때 수면은 Water(wave_visual=True, sea_bounds=SEA_BOUNDS) 가 한 겹으로 그린다
        (배 주변 파도 패치 + 둘레 평평한 바다를 메시 하나로). 맵 판까지 겹치면 "물 위에 물"처럼 보였다.
        """
        if scene != 'hanagae':
            raise ValueError('지원하는 장면은 hanagae뿐입니다.')
        self.scene = scene
        self.wave_amplitude = float(wave_amplitude)
        self._bathers = []

    def build(self, spec):
        surface = config.WaterParams.SURFACE_Z
        y0, y1 = -SEA_HALF_Y, SEA_HALF_Y

        # 넓은 수면 (Water 기본 50m 평면 대신 맵이 표시). 파도가 있으면 Water 가 그린다
        if self.wave_amplitude <= 0:
            _visual_box(spec, name='hanagae_water_surface',
                        pos=((SEA_WEST + SHORE_X) / 2, 0, surface),
                        size=((SHORE_X - SEA_WEST) / 2, SEA_HALF_Y, .05),
                        rgba=SEA_RGBA)
        # 햇빛처럼 평행광, 그림자 끔 (2026-11-23 렌더 확인):
        #   - 스포트라이트는 꼭짓점마다 밝기를 계산해서, 꼭짓점이 멀리 있는 큰 상자(바다 판)는 거의 안 비추고
        #     꼭짓점이 촘촘한 파도 높이맵만 밝게 비춰 → 배 주변 50 m 네모가 밝게 튀어 보였다.
        #   - 그림자를 켜면 파도 높이맵에 자잘한 줄무늬(shadow acne)가 생긴다.
        spec.worldbody.add_light(pos=(SHORE_X - 40, 0, 40), dir=(0, 0, -1), diffuse=(.75, .75, .75),
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False)

        # 해저: 모래 경사 서쪽 끝에서 이어진다
        _box_from_bounds(spec, name='hanagae_seabed',
                         x0=SEA_WEST, x1=SHORE_X - FLAT_WIDTH, y0=y0, y1=y1,
                         z0=surface - 4.0, z1=surface - .4,
                         rgba=(.04, .14, .22, 1))

        # 모래는 한 층. 육지 쪽 끝에서 바다 쪽으로 직선으로 내려간다.
        _sand_slope(spec, y0=y0, y1=y1)

        # 내륙: 맵 동쪽 끝까지 한 덩어리
        _box_from_bounds(spec, name='hanagae_land',
                         x0=SHORE_X + BEACH_WIDTH, x1=LAND_EAST, y0=y0, y1=y1,
                         z0=surface - LAND_DEPTH, z1=surface + LAND_TOP,
                         rgba=(.28, .42, .24, 1))

        # 하나개로 (육지 위)
        road_x = SHORE_X + BEACH_WIDTH + 4.0
        _box_from_bounds(spec, name='hanagae_road',
                         x0=road_x - 1.2, x1=road_x + 1.2, y0=y0 + 5, y1=y1 - 5,
                         z0=surface + LAND_TOP, z1=surface + LAND_TOP + .08,
                         rgba=(.22, .22, .24, 1))
        _box_from_bounds(spec, name='hanagae_road_curve',
                         x0=road_x, x1=min(LAND_EAST - 2, road_x + 20),
                         y0=y1 - 10, y1=y1 - 7.5,
                         z0=surface + LAND_TOP, z1=surface + LAND_TOP + .08,
                         rgba=(.22, .22, .24, 1))

        buildings = (
            ('hanagae_shop_a', road_x + 5, 20, 1.4, 2.0, 1.5, .7, (.55, .58, .62, 1)),
            ('hanagae_shop_b', road_x + 6, -15, 1.2, 2.2, 1.6, .6, (.5, .55, .6, 1)),
            ('hanagae_art_center', road_x + 18, 45, 2.2, 3.0, 2.5, 1.1, (.62, .64, .7, 1)),
            ('hanagae_zipline', road_x + 10, -50, 3.0, 1.2, 1.2, 1.5, (.7, .45, .2, 1)),
        )
        for name, x, y, h, sx, sy, sz, rgba in buildings:
            _visual_box(spec, name=name,
                        pos=(x, y, surface + LAND_TOP + h),
                        size=(sx, sy, sz), rgba=rgba)

        # 해안 앞 고립 구간 표시
        spec.worldbody.add_site(name='hanagae_hazard',
                                pos=(SHORE_X - 8.0, 5.0, surface + .5),
                                size=(.5, .5, .5), rgba=(.95, .75, .1, 1),
                                type=mujoco.mjtGeom.mjGEOM_SPHERE)
        _add_beachgoers(spec, surface)
        _add_reef(spec, surface)

    def bind(self, model):
        """물놀이 몸체의 처음 자세를 기억한다. 화면마다 이 자세에서 팔·몸을 움직인다."""
        self._bathers = []
        for index, (*_place, pose, _scale) in enumerate(_BEACHGOERS):
            body = model.body(f'hanagae_bather_{index}')
            joints = []
            for part in ('arm_l', 'arm_r'):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'hanagae_bather_{index}_{part}')
                if jid >= 0:
                    joints.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))
            self._bathers.append(dict(mocap=int(body.mocapid[0]), pose=pose,
                                      pos=model.body(body.id).pos.copy(), quat=model.body(body.id).quat.copy(),
                                      joints=joints))

    def add_forces(self, model, data, acc):
        self.animate(model, data)

    def animate(self, model, data):
        """data.time 기준으로 물장구·헤엄 자세를 갱신한다. 힘과 익수자 물리는 바꾸지 않는다."""
        if not self._bathers:
            return
        t = float(data.time)
        for index, bather in enumerate(self._bathers):
            cycle = t * 2.2 + index * 0.6
            pose = bather['pose']
            pos = np.array(bather['pos'], dtype=float)
            pos[2] += .28 * math.sin(cycle * 2)
            if pose in ('paddle', 'breast'):
                pos[0] += 1.4 * math.sin(cycle)
            data.mocap_pos[bather['mocap']] = pos
            quat = np.array(bather['quat'], dtype=float)
            if pose == 'float':
                quat = _quat_mul(_axis_quat((1, 0, 0), .5 * math.sin(cycle)), quat)
            elif pose == 'dive':
                quat = _quat_mul(_axis_quat((0, 1, 0), 1.1 * math.sin(cycle * .8)), quat)
            elif pose == 'splash':
                quat = _quat_mul(_axis_quat((0, 1, 0), .45 * math.sin(cycle * 2)), quat)
            data.mocap_quat[bather['mocap']] = quat
            for side, (qadr, dadr) in enumerate(bather['joints']):
                if pose == 'splash':
                    angle = 1.4 * math.sin(cycle * 3 + side * 1.5)
                elif pose == 'breast':
                    angle = 1.2 * math.sin(cycle * 2)
                elif pose == 'wave' and side == 1:
                    angle = 1.3 * math.sin(cycle * 2)
                elif pose == 'paddle':
                    angle = 1.1 * math.sin(cycle * 2 + side * math.pi)
                else:
                    angle = .9 * math.sin(cycle * 2 + side * math.pi)
                data.qpos[qadr] = angle
                data.qvel[dadr] = 0.0


# 얕은 바다에서 물놀이. (x, y, yaw, pose, scale). 가슴~목까지 잠기게 발을 수면 아래에 둔다.
# 출항 직선(y=0)과 암초 자리는 비워 둔다.
_BEACHGOERS = (
    (-8, -16, math.pi, 'splash', 1.0),
    (-14, -8, math.pi * .9, 'paddle', 1.0),
    (-6, 8, math.pi, 'tread', 1.0),
    (-18, 14, math.pi * 1.1, 'float', 1.0),
    (-11, 22, math.pi, 'dive', 1.0),
    (-22, -22, math.pi * .8, 'breast', 1.0),
    (-9, -28, math.pi, 'splash', .9),
    (-16, 28, math.pi, 'ball', 1.0),
    (-7, 34, math.pi * 1.2, 'wave', .7),
    (-24, 6, math.pi, 'paddle', 1.0),
)
# 암초: 수평 길이는 선체 길이의 약 5배. 바닥은 수면 아래, 꼭대기는 수면 위라 배가 타고 넘지 못한다.
REEF_CENTER = (-110.0, 0.0)
REEF_LENGTH = 5 * config.Hull.LENGTH
REEF_WIDTH = 4.0
REEF_BOTTOM = -2.2
REEF_TOP = 1.3
_SHIRTS = (
    (.75, .22, .18, 1), (.2, .45, .75, 1), (.95, .8, .2, 1), (.15, .55, .35, 1), (.55, .25, .7, 1),
    (.9, .45, .2, 1), (.25, .7, .75, 1), (.85, .15, .4, 1), (.35, .35, .85, 1), (.7, .55, .3, 1),
)
_SWIMMING = ('paddle', 'float', 'breast', 'dive')


def _add_beachgoers(spec, surface):
    """얕은 바다의 물놀이 10명. 형상은 익수자와 같은 add_visual_human."""
    from sim.world.human import add_visual_human, lying_quat, upright_quat

    for index, (x, y, yaw, pose, _scale) in enumerate(_BEACHGOERS):
        # 헤엄은 수면에 눕히고, 물장구는 발을 물속에 둔 선 자세.
        if pose in _SWIMMING:
            z, quat = surface - .05, lying_quat(yaw)
        else:
            z, quat = surface - .15, upright_quat(yaw)
        add_visual_human(spec, f'hanagae_bather_{index}', (x, y, z), quat, shirt=_SHIRTS[index])


def _add_reef(spec, surface):
    """항로 한가운데 돌섬. 기울어진 바위가 겹친 모양이고, 배·사람은 통과하지 못한다."""
    contacts = config.Contacts
    cx, cy = REEF_CENTER
    # (dx, dy, z, rx, ry, rz, 기울기 rad, 색). z는 수면 기준. 전체 길이는 선체의 약 5배.
    rocks = (
        (0.0, 0.0, .15, 2.1, 1.5, 1.7, .35, (.45, .42, .38, 1)),
        (-2.3, .7, -.35, 1.5, 1.1, 1.15, -.5, (.38, .36, .33, 1)),
        (2.1, -.6, -.2, 1.35, 1.15, 1.05, .6, (.52, .48, .43, 1)),
        (.4, 1.5, -.55, 1.1, .9, .85, .2, (.34, .33, .31, 1)),
        (-.9, -1.35, -.7, 1.25, .85, .7, -.25, (.4, .38, .35, 1)),
        (.8, .2, .85, .7, .55, .55, .4, (.58, .55, .5, 1)),
    )
    solid = dict(group=config.Groups.PHYSICS, contype=contacts.ENV, conaffinity=contacts.ENV_AFFINITY,
                 solref=[.006, 1], solimp=[.95, .99, .001, .5, 2], priority=1)
    for index, (dx, dy, z, rx, ry, rz, tilt, rgba) in enumerate(rocks):
        spec.worldbody.add_geom(
            name=f'hanagae_reef_{index}', type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=(cx + dx, cy + dy, surface + z), size=(rx, ry, rz),
            quat=_axis_quat((.2, 1, .15), tilt), rgba=rgba, **solid,
        )


def _axis_quat(axis, angle):
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, dtype=float), float(angle))
    return quat


def _quat_mul(left, right):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    return out


def sand_top_xz():
    """모래 경사의 육지 쪽 끝 (x, z). 잔디 윗면과 같은 높이에서 시작한다."""
    return (SHORE_X + BEACH_WIDTH, config.WaterParams.SURFACE_Z + LAND_TOP)


def sand_toe_z():
    """모래 경사의 바다 쪽 끝 높이. 해저보다 깊게 들어가 사선이 이어진다."""
    return config.WaterParams.SURFACE_Z - 2.2


def sand_height(x):
    """모래 윗면 높이. 동쪽 끝에서 서쪽 끝까지 한 직선이다."""
    x_top, z_top = sand_top_xz()
    x_toe = SHORE_X - FLAT_WIDTH
    z_toe = sand_toe_z()
    return z_top + (float(x) - x_top) / (x_toe - x_top) * (z_toe - z_top)


def _sand_slope(spec, *, y0, y1):
    """모래 한 덩어리. 윗면만 육지에서 바다로 곧게 내려가고, 속은 바닥까지 채운다."""
    import mujoco

    x_top, z_top = sand_top_xz()
    x_toe, z_toe = SHORE_X - FLAT_WIDTH, sand_toe_z()
    z_bot = config.WaterParams.SURFACE_Z - LAND_DEPTH
    # 남쪽(y0) 0~3, 북쪽(y1) 4~7. 각 끝: 위·육지, 위·바다, 아래·바다, 아래·육지.
    verts = []
    for y in (y0, y1):
        verts += [(x_top, y, z_top), (x_toe, y, z_toe), (x_toe, y, z_bot), (x_top, y, z_bot)]
    # 위에서 보면 반시계. 윗면 법선이 하늘을 본다.
    faces = [
        (0, 4, 5), (0, 5, 1),   # 경사 윗면
        (0, 3, 7), (0, 7, 4),   # 육지 쪽
        (1, 5, 6), (1, 6, 2),   # 바다 쪽
        (3, 2, 6), (3, 6, 7),   # 바닥
        (0, 1, 2), (0, 2, 3),   # 남쪽 단면
        (4, 7, 6), (4, 6, 5),   # 북쪽 단면
    ]
    spec.add_mesh(name='hanagae_beach_mesh', uservert=np.asarray(verts, dtype=float).ravel().tolist(),
                  userface=np.asarray(faces, dtype=int).ravel().tolist(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    spec.worldbody.add_geom(
        name='hanagae_beach', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='hanagae_beach_mesh',
        rgba=(.86, .78, .55, 1), contype=0, conaffinity=0, group=config.Groups.VISUAL,
    )


def victim_spawn_position(*, range_m=VICTIM_RANGE):
    """해안선에서 서쪽으로 range_m만큼 떨어진 수면 근처."""
    if not math.isfinite(range_m) or range_m <= 0:
        raise ValueError('익수자 거리는 유한한 양수여야 합니다.')
    return (SHORE_X - float(range_m), 0.0, config.WaterParams.SURFACE_Z - .03)


def land_boat_pose():
    """모래 경사 위 대기 위치. 바다(−x)를 본다. 육지 충돌이 없어 시나리오가 자세를 붙잡는다."""
    x = SHORE_X + BEACH_WIDTH * .45
    z = sand_height(x) + config.Hull.HEIGHT / 2 + .05
    return (x, 0.0, z), math.pi


def water_launch_pose():
    """출항 직후 해안 앞 수면 위치. 익수자(−x)를 본다."""
    draft = config.Expect.DRAFT or 0.0
    z = config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - draft
    return (SHORE_X - 20.0, 0.0, z), math.pi


def boat_spawn_pose():
    """맵 미리보기용: 해안 앞 바다 기본 배치."""
    return water_launch_pose()


def _set_boat_pose(world, robot, position, yaw):
    address = world.model.jnt_qposadr[robot.joint_id]
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    world.model.qpos0[address:address + 3] = position
    world.model.qpos0[address + 3:address + 7] = quat
    world.data.qpos[address:address + 3] = position
    world.data.qpos[address + 3:address + 7] = quat
    world.data.qvel[:] = 0
    mujoco.mj_forward(world.model, world.data)


def hanagae_splash_sea():
    """철썩 시험의 --sea three. 해안 턱과 돌섬 반사를 포함한다."""
    from sim.water.sea import R_ROCK, Sea

    sea = Sea('three')
    sea.add_wall((SHORE_X, 0.0), (-1.0, 0.0), 0.6, name='seawall')
    sea.add_wall(REEF_CENTER, (1.0, 0.0), R_ROCK, half_len=REEF_LENGTH / 2,
                 name='reef', shadowed=False)
    sea.add_shadow(REEF_CENTER, 0.0, REEF_LENGTH / 2, REEF_WIDTH / 2)
    return sea


def build_hanagae_world(*, waves=None, boat_on_land=False, victim_range=VICTIM_RANGE, splash_sea=False):
    """동쪽 육지 + 서쪽 바다에 배·익수자를 조립한다."""
    from sim.robot.robot import Robot
    from sim.water.water import Water, wave_amplitude
    from sim.world.victim import make_victim
    from sim.world.world import build_world

    if waves is None:
        waves = config.WaterParams.WAVES_CALM
    if splash_sea:
        waves = hanagae_splash_sea()
    position, yaw = land_boat_pose() if boat_on_land else water_launch_pose()
    # Water의 50m 기본 평면은 끈다. 잔잔하면 맵이 넓은 수면 판을, 파도가 있으면 Water가 배 주변 패치와 평평한 둘레를 한 겹으로 그린다.
    water = Water(waves=waves, surface_visual=False, wave_visual=True, sea_bounds=SEA_BOUNDS, sea_rgba=SEA_RGBA,
                  floating_bodies=[config.Names.HUMAN_BODY])
    if splash_sea:
        from sim.world.wave_impact import WaveImpact
        water.impact = WaveImpact(waves)
    robot = Robot(surface_height=water.wave_height, position=position)
    # 익수자는 해안(동쪽)을 본다. 구명환은 아직 없으므로 발버둥만 켠다.
    victim_xy = victim_spawn_position(range_m=victim_range)
    human, victim = make_victim(victim_xy[:2], 0.0)
    parts = [water, CoastalMap(wave_amplitude=wave_amplitude(waves)), robot, human, victim]
    if boat_on_land:
        from sim.world.shore_call import hanagae_shore_call
        parts.append(hanagae_shore_call(victim_xy[:2]))
    world = build_world(parts=parts)
    _set_boat_pose(world, robot, position, yaw)
    return world, robot, water, human
