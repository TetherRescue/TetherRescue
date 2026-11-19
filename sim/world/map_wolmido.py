"""월미도 해안 맵: 방파제·산책로·경사 사석·바위 (담당: 가영).

세계 조립 부품(build·bind·add_forces)이다. 지형은 모두 월드에 고정된 형상이며 질량·외력이 없다.
치수·환경 값은 config.Wolmido의 시험용 가정값이다. 실측 도면·수심 자료가 아니다.

간조는 수면 높이(물 담당 config)를 바꾸지 않고 지형 전체를 tide_drop만큼 올려서 표현한다.
"""

import math

import numpy as np

from sim import config


class WolmidoMap:
    REQUIRES = {}

    def __init__(self, *, tide_drop=0.0):
        tide_drop = float(tide_drop)
        if not math.isfinite(tide_drop) or tide_drop < 0:
            raise ValueError("tide_drop은 0 이상의 유한한 값이어야 합니다 (간조로 내려간 수면 m).")
        self.tide_drop = tide_drop
        self.geom_ids = {}

    @staticmethod
    def environment(name):
        """config.Wolmido.ENVIRONMENTS의 환경 하나 (사본)."""
        try:
            return dict(config.Wolmido.ENVIRONMENTS[name])
        except KeyError:
            raise ValueError(f"환경은 {tuple(config.Wolmido.ENVIRONMENTS)} 중 하나여야 합니다.") from None

    def _solid(self):
        # 콘크리트·바위는 단단하므로 사람·구명환과 같은 짧은 접촉 시간 상수를 쓴다 (기본 0.02s는 최대 추력에 1cm 파고듦).
        return dict(group=config.Groups.PHYSICS, contype=config.Contacts.ENV,
                    conaffinity=config.Contacts.ENV_AFFINITY, solref=[.006, 1], solimp=[.95, .99, .001, .5, 2],
                    priority=1)  # 두 형상 값을 섞지 않고 지형 값을 쓴다 (선체 기본 0.02s와 평균되면 7mm 파고듦)

    @staticmethod
    def _visual():
        return dict(group=config.Groups.VISUAL, contype=0, conaffinity=0)

    def build(self, spec):
        import mujoco

        cfg, names = config.Wolmido, config.Names
        face, dz = float(cfg.WALL_FACE_X), self.tide_drop
        half_length = cfg.WALL_LENGTH / 2
        concrete, stone, paving = [.62, .62, .6, 1], [.36, .34, .31, 1], [.72, .66, .55, 1]
        world = spec.worldbody
        box = mujoco.mjtGeom.mjGEOM_BOX

        # 방파제는 해변 구간(BEACH_Y)에서 끊긴 두 덩어리. 바다 쪽 면이 x = face.
        segments = self.wall_segments()
        height = cfg.WALL_TOP_Z - cfg.WALL_BOTTOM_Z
        top_z = cfg.WALL_TOP_Z + dz
        for index, (y0, y1) in enumerate(segments):
            suffix = "" if index == 0 else "_north"
            middle, half = (y0 + y1) / 2, (y1 - y0) / 2
            world.add_geom(name=names.WOLMIDO_WALL_GEOM + suffix, type=box,
                           pos=[face + cfg.LAND_DEPTH / 2, middle, cfg.WALL_BOTTOM_Z + height / 2 + dz],
                           size=[cfg.LAND_DEPTH / 2, half, height / 2], rgba=concrete, **self._solid())
            # 산책로 포장 (표시용, 방파제 윗면 바로 위)
            world.add_geom(name=f"{names.WOLMIDO_PREFIX}_promenade{suffix}", type=box,
                           pos=[face + 3.0, middle, top_z + .005], size=[2.8, half, .005],
                           rgba=paving, **self._visual())
            # 경사 사석: 위 끝(방파제 면)에서 바다 쪽 아래 끝까지 기울어진 판.
            self._slab(world, mujoco, names.WOLMIDO_RIPRAP_GEOM + suffix,
                       (face + cfg.RIPRAP_TOP[0], cfg.RIPRAP_TOP[1] + dz), (face + cfg.RIPRAP_TOE[0], cfg.RIPRAP_TOE[1] + dz),
                       (y0, y1), cfg.RIPRAP_THICKNESS, stone)
            # 난간 (표시용): 기둥 2m 간격 + 위 손잡이
            rail_x = face + .25
            for y in np.arange(y0 + 1, y1, 2.0):
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                               fromto=[rail_x, y, top_z, rail_x, y, top_z + cfg.RAILING_HEIGHT],
                               size=[.03, 0, 0], rgba=[.2, .25, .3, 1], **self._visual())
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                           fromto=[rail_x, y0 + 1, top_z + cfg.RAILING_HEIGHT, rail_x, y1 - 1, top_z + cfg.RAILING_HEIGHT],
                           size=[.03, 0, 0], rgba=[.2, .25, .3, 1], **self._visual())
        for index, (x, y, radius) in enumerate(cfg.BOULDERS):
            world.add_geom(name=f"{names.WOLMIDO_PREFIX}_boulder_{index}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                           pos=[face + x, y, cfg.BOULDER_Z + dz], size=[radius, radius * 1.2, radius * .75],
                           rgba=[.30 + .02 * index, .29, .27, 1], **self._solid())

        # 모래 해변: 방파제 사이를 완만한 경사로 채우고, 뒤쪽 육지는 해변 위 끝 높이.
        b0, b1 = cfg.BEACH_Y
        sand = [.86, .78, .58, 1]
        beach_top_z = cfg.BEACH_TOP[1] + dz
        land = cfg.LAND_DEPTH - cfg.BEACH_TOP[0]
        world.add_geom(name=f"{names.WOLMIDO_PREFIX}_beach_land", type=box,
                       pos=[face + cfg.BEACH_TOP[0] + land / 2, (b0 + b1) / 2, (beach_top_z + cfg.WALL_BOTTOM_Z + dz) / 2],
                       size=[land / 2, (b1 - b0) / 2, (beach_top_z - cfg.WALL_BOTTOM_Z - dz) / 2], rgba=sand, **self._solid())
        self._slab(world, mujoco, f"{names.WOLMIDO_PREFIX}_beach",
                   (face + cfg.BEACH_TOP[0], beach_top_z), (face + cfg.BEACH_TOE[0], cfg.BEACH_TOE[1] + dz),
                   (b0, b1), cfg.BEACH_THICKNESS, sand)
        # 방파제 사다리 (표시용): 산책로에서 지금 수면 1m 아래까지 (간조에도 물에 닿는다)
        ladder_bottom = config.WaterParams.SURFACE_Z - 1.0
        for side in (-.25, .25):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                           fromto=[face - .08, cfg.LADDER_Y + side, top_z, face - .08, cfg.LADDER_Y + side, ladder_bottom],
                           size=[.025, 0, 0], rgba=[.85, .75, .1, 1], **self._visual())
        for z in np.arange(ladder_bottom, top_z, .35):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                           fromto=[face - .08, cfg.LADDER_Y - .25, z, face - .08, cfg.LADDER_Y + .25, z],
                           size=[.02, 0, 0], rgba=[.85, .75, .1, 1], **self._visual())

        self._build_details(world, mujoco, face, dz, top_z, half_length)
        # 낮 장면 조명 (표시용): 넓은 맵에서 수면 위 점조명 하나로는 어두워 해 방향 조명을 더한다.
        world.add_light(name=f"{config.Names.WOLMIDO_PREFIX}_sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                        pos=[-30, -20, 40],
                        dir=[.45, .3, -.84], diffuse=[.75, .72, .66], specular=[.1, .1, .1], castshadow=False)
        spec.visual.headlight.ambient = [.32, .32, .34]
        spec.add_texture(name=f"{config.Names.WOLMIDO_PREFIX}_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[.55, .72, .9], rgb2=[.92, .95, 1.],
                         width=256, height=256)

    @staticmethod
    def wall_segments():
        """방파제가 있는 y 구간 두 개 (해변 구간 BEACH_Y를 뺀 나머지)."""
        cfg = config.Wolmido
        half = cfg.WALL_LENGTH / 2
        return ((-half, cfg.BEACH_Y[0]), (cfg.BEACH_Y[1], half))

    def _slab(self, world, mujoco, name, top, toe, y_range, thickness, rgba):
        """윗면이 top–toe 선(x, z)인 기울어진 판. y축 회전은 +x를 -z 쪽으로 돌리므로 각의 부호를 반대로 준다."""
        top, toe = np.asarray(top, dtype=float), np.asarray(toe, dtype=float)
        slope = top - toe
        angle = math.atan2(slope[1], slope[0])
        normal = np.array([-math.sin(angle), math.cos(angle)])
        center = (top + toe) / 2 - normal * thickness / 2
        y0, y1 = y_range
        world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[center[0], (y0 + y1) / 2, center[1]],
                       quat=[math.cos(-angle / 2), 0, math.sin(-angle / 2), 0],
                       size=[float(np.hypot(*slope)) / 2, (y1 - y0) / 2, thickness / 2], rgba=rgba, **self._solid())

    def beach_x_at_depth(self, depth):
        """해변 경사에서 물 깊이가 depth[m]인 지점의 월드 x (지금 조위 기준)."""
        cfg = config.Wolmido
        (tx, tz), (ex, ez) = cfg.BEACH_TOP, cfg.BEACH_TOE
        target = config.WaterParams.SURFACE_Z - depth - self.tide_drop  # 만조 기준 높이로 환산
        if not ez <= target <= tz:
            raise ValueError("그 깊이는 해변 경사 범위 밖입니다.")
        return cfg.WALL_FACE_X + tx + (target - tz) / (ez - tz) * (ex - tx)

    def _build_details(self, world, mujoco, face, dz, top_z, half_length):
        """테트라포드·친수 계단·선착장·부표(충돌)와 바닥·가로등·벤치·대관람차(표시)."""
        cfg, prefix = config.Wolmido, config.Names.WOLMIDO_PREFIX
        box, capsule, cylinder = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER
        surface = config.WaterParams.SURFACE_Z
        tetrapod_color = [.70, .69, .66, 1]

        # 테트라포드: 중심에서 정사면체 꼭짓점 방향으로 다리 4개. 놓인 방향은 개수마다 조금씩 돌린다.
        legs = np.array([[0, 0, 1], [.943, 0, -.333], [-.471, .816, -.333], [-.471, -.816, -.333]])
        for index, (x, y) in enumerate(cfg.TETRAPODS):
            yaw, tilt = .9 * index, .35 * (-1) ** index
            rotation = (np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
                        @ np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]]))
            center = np.array([face + x, y, cfg.TETRAPOD_Z + dz])
            for leg_index, leg in enumerate(legs):
                end = center + rotation @ leg * cfg.TETRAPOD_LEG
                world.add_geom(name=f"{prefix}_tetrapod_{index}_{leg_index}", type=capsule,
                               fromto=[*center, *end], size=[cfg.TETRAPOD_RADIUS, 0, 0],
                               rgba=tetrapod_color, **self._solid())

        # 계단식 친수 호안: 위에서부터 한 단씩 바다 쪽으로 내려간다.
        y0, y1 = cfg.STEPS_Y
        for step in range(cfg.STEP_COUNT):
            step_top = cfg.WALL_TOP_Z - (step + 1) * cfg.STEP_RISE + dz
            reach = (step + 1) * cfg.STEP_RUN
            bottom = cfg.RIPRAP_TOP[1] + dz - 1.0
            world.add_geom(name=f"{prefix}_step_{step}", type=box,
                           pos=[face - reach / 2, (y0 + y1) / 2, (step_top + bottom) / 2],
                           size=[reach / 2, (y1 - y0) / 2, (step_top - bottom) / 2],
                           rgba=[.66 - .02 * step, .64 - .02 * step, .6, 1], **self._solid())

        # 선착장: 폰툰(물에 뜸, 간조에도 수면 높이), 말뚝(바닥에 박힘), 산책로에서 내려가는 연결 다리.
        (px, py), (length, width, height) = cfg.PONTOON_CENTER, cfg.PONTOON_SIZE
        pontoon_top = surface + cfg.PONTOON_FREEBOARD
        world.add_geom(name=f"{prefix}_pontoon", type=box, pos=[face + px, py, pontoon_top - height / 2],
                       size=[length / 2, width / 2, height / 2], rgba=[.55, .58, .62, 1], **self._solid())
        for index, (x, y) in enumerate(cfg.PILES):
            bottom, top = cfg.SEABED_Z + dz, cfg.PILE_TOP_Z + dz
            world.add_geom(name=f"{prefix}_pile_{index}", type=cylinder, pos=[face + x, y, (bottom + top) / 2],
                           size=[cfg.PILE_RADIUS, (top - bottom) / 2, 0], rgba=[.25, .22, .2, 1], **self._solid())
        start = np.array([face + 1.5, top_z])
        end = np.array([face + px + length / 2 - .5, pontoon_top])
        run = start - end
        angle = math.atan2(run[1], run[0])
        span = float(np.hypot(*run))
        middle = (start + end) / 2
        world.add_geom(name=f"{prefix}_gangway", type=box, pos=[middle[0], py, middle[1] - .05],
                       quat=[math.cos(-angle / 2), 0, math.sin(-angle / 2), 0],
                       size=[span / 2, .7, .05], rgba=[.4, .42, .45, 1], **self._solid())
        for side in (-.7, .7):
            world.add_geom(type=capsule, fromto=[start[0], py + side, start[1] + 1, end[0], py + side, end[1] + 1],
                           size=[.03, 0, 0], rgba=[.2, .25, .3, 1], **self._visual())
        (bx, by), (blen, bwid, bhgt) = cfg.MOORED_BOAT['center'], cfg.MOORED_BOAT['size']
        world.add_geom(name=f"{prefix}_moored_boat", type=box, pos=[face + bx, by, surface + bhgt / 2 - .5],
                       size=[blen / 2, bwid / 2, bhgt / 2], rgba=[.92, .92, .9, 1], **self._solid())
        world.add_geom(type=box, pos=[face + bx - .8, by, surface + bhgt - .5 + .4],
                       size=[1.2, bwid / 2 - .2, .4], rgba=[.15, .35, .6, 1], **self._visual())

        # 항로 표지 부표: 물에 떠 있는 원통(충돌) + 위 표시
        for index, (x, y) in enumerate(cfg.BUOYS):
            world.add_geom(name=f"{prefix}_buoy_{index}", type=cylinder, pos=[face + x, y, surface + .3],
                           size=[cfg.BUOY_RADIUS, .6, 0], rgba=[.9, .75, .05, 1], **self._solid())
            world.add_geom(type=capsule, fromto=[face + x, y, surface + .9, face + x, y, surface + 2.6],
                           size=[.08, 0, 0], rgba=[.1, .1, .1, 1], **self._visual())

        # 표시용: 바다 바닥, 가로등, 벤치, 구명환 보관함, 대관람차
        world.add_geom(type=box, pos=[face - 30, 0, cfg.SEABED_Z + dz - .5], size=[30, half_length, .5],
                       rgba=[.62, .56, .42, 1], **self._visual())
        lamp_x = face + 4.5
        for y in np.arange(-half_length + 5, half_length, cfg.LAMP_SPACING):
            if cfg.BEACH_Y[0] - 1 < y < cfg.BEACH_Y[1] + 1:
                continue  # 해변 구간에는 산책로가 없다
            world.add_geom(type=capsule, fromto=[lamp_x, y, top_z, lamp_x, y, top_z + 4.0], size=[.06, 0, 0],
                           rgba=[.15, .15, .17, 1], **self._visual())
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[lamp_x - .3, y, top_z + 4.05], size=[.18, 0, 0],
                           rgba=[1, .95, .7, 1], **self._visual())
        for y in cfg.BENCH_Y:
            world.add_geom(type=box, pos=[face + 3.5, y, top_z + .45], size=[.25, .9, .05],
                           rgba=[.55, .35, .2, 1], **self._visual())
            world.add_geom(type=box, pos=[face + 3.75, y, top_z + .75], size=[.04, .9, .3],
                           rgba=[.55, .35, .2, 1], **self._visual())
        for y in cfg.LIFEBUOY_BOX_Y:
            world.add_geom(type=box, pos=[face + .55, y, top_z + .7], size=[.15, .45, .55],
                           rgba=[.85, .15, .1, 1], **self._visual())
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=[face + .38, y, top_z + .75],
                           size=[.03, .3, .3], rgba=[1, .45, .1, 1], **self._visual())
        (wx, wy), radius = cfg.FERRIS_WHEEL['center'], cfg.FERRIS_WHEEL['radius']
        hub = np.array([face + wx, wy, top_z + radius + 2.0])
        for k in range(24):
            a0, a1 = 2 * math.pi * k / 24, 2 * math.pi * (k + 1) / 24
            p0 = hub + radius * np.array([0, math.cos(a0), math.sin(a0)])
            p1 = hub + radius * np.array([0, math.cos(a1), math.sin(a1)])
            world.add_geom(type=capsule, fromto=[*p0, *p1], size=[.08, 0, 0], rgba=[.85, .85, .9, 1], **self._visual())
            if k % 2 == 0:
                world.add_geom(type=capsule, fromto=[*hub, *p0], size=[.03, 0, 0], rgba=[.7, .7, .75, 1],
                               **self._visual())
                world.add_geom(type=box, pos=p0 - [0, 0, .7], size=[.5, .4, .4],
                               rgba=[(.9, .2, .2, 1), (.2, .5, .9, 1), (.95, .8, .1, 1)][k // 2 % 3], **self._visual())
        for side in (-1, 1):
            world.add_geom(type=capsule, fromto=[*hub, face + wx, wy + side * 4.0, top_z], size=[.15, 0, 0],
                           rgba=[.6, .6, .65, 1], **self._visual())

    def bind(self, model):
        prefix = (config.Names.WOLMIDO_PREFIX, config.Names.WOLMIDO_WALL_GEOM, config.Names.WOLMIDO_RIPRAP_GEOM)
        self.geom_ids = {model.geom(i).name: i for i in range(model.ngeom)
                         if model.geom(i).name.startswith(prefix) and model.geom_contype[i]}
        if config.Names.WOLMIDO_WALL_GEOM not in self.geom_ids:
            raise ValueError("WolmidoMap.bind 전에 맵 build·compile이 필요합니다.")

    def add_forces(self, model, data, acc):
        """고정 지형이라 외력이 없다. 접촉력은 엔진이 계산한다."""
