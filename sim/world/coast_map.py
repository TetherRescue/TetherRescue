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

    def __init__(self, *, scene='hanagae'):
        if scene != 'hanagae':
            raise ValueError('지원하는 장면은 hanagae뿐입니다.')
        self.scene = scene

    def build(self, spec):
        surface = config.WaterParams.SURFACE_Z
        y0, y1 = -SEA_HALF_Y, SEA_HALF_Y

        # 넓은 수면 (Water 기본 50m 평면 대신 맵이 표시)
        _visual_box(spec, name='hanagae_water_surface',
                    pos=((SEA_WEST + SHORE_X) / 2, 0, surface),
                    size=((SHORE_X - SEA_WEST) / 2, SEA_HALF_Y, .05),
                    rgba=(.06, .32, .45, .45))
        spec.worldbody.add_light(pos=(SHORE_X - 40, 0, 40), dir=(0, 0, -1), diffuse=(.75, .75, .75))

        # 해저: 수면에서 아래로 이어지게
        _box_from_bounds(spec, name='hanagae_seabed',
                         x0=SEA_WEST, x1=SHORE_X - FLAT_WIDTH, y0=y0, y1=y1,
                         z0=surface - 4.0, z1=surface - .4,
                         rgba=(.04, .14, .22, 1))

        # 갯벌(고립 구간): 해안 바로 앞, 수면 아래이지만 육지와 이어짐
        _box_from_bounds(spec, name='hanagae_tidal_flat',
                         x0=SHORE_X - FLAT_WIDTH, x1=SHORE_X, y0=y0, y1=y1,
                         z0=surface - LAND_DEPTH, z1=surface - .05,
                         rgba=(.45, .38, .22, 1))

        # 모래사장: 해안선에서 내륙으로, 바닥은 수면 아래까지 내려 띄우지 않음
        _box_from_bounds(spec, name='hanagae_beach',
                         x0=SHORE_X, x1=SHORE_X + BEACH_WIDTH, y0=y0, y1=y1,
                         z0=surface - LAND_DEPTH, z1=surface + .4,
                         rgba=(.86, .78, .55, 1))

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

    def bind(self, model):
        return None

    def add_forces(self, model, data, acc):
        return None


def victim_spawn_position(*, range_m=VICTIM_RANGE):
    """해안선에서 서쪽으로 range_m만큼 떨어진 수면 근처."""
    if not math.isfinite(range_m) or range_m <= 0:
        raise ValueError('익수자 거리는 유한한 양수여야 합니다.')
    return (SHORE_X - float(range_m), 0.0, config.WaterParams.SURFACE_Z - .03)


def land_boat_pose():
    """모래사장 위 대기 위치. 바다(−x)를 본다. 육지 충돌이 없어 시나리오가 자세를 붙잡는다."""
    beach_top = config.WaterParams.SURFACE_Z + .4
    z = beach_top + config.Hull.HEIGHT / 2 + .05
    return (SHORE_X + BEACH_WIDTH * .45, 0.0, z), math.pi


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


def build_hanagae_world(*, waves=None, boat_on_land=False, victim_range=VICTIM_RANGE):
    """동쪽 육지 + 서쪽 바다에 배·익수자를 조립한다."""
    from sim.robot.robot import Robot
    from sim.water.water import Water
    from sim.world.human import Human
    from sim.world.world import build_world

    if waves is None:
        waves = config.WaterParams.WAVES_CALM
    position, yaw = land_boat_pose() if boat_on_land else water_launch_pose()
    # 맵이 넓은 수면을 그리므로 Water의 50m 기본 평면은 끈다.
    water = Water(waves=waves, surface_visual=False,
                  floating_bodies=[config.Names.HUMAN_BODY])
    robot = Robot(surface_height=water.wave_height, position=position)
    human = Human(position=victim_spawn_position(range_m=victim_range))
    world = build_world(parts=[water, CoastalMap(), robot, human])
    _set_boat_pose(world, robot, position, yaw)
    return world, robot, water, human
