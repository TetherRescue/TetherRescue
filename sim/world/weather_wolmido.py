"""[시험용] 월미도 날씨 환경 3개: 맑음 · 비 · 안개 (가영, 2026-11-20, 브랜치 try/weather).

기존 파일(config.py·view_wolmido.py)은 바꾸지 않고 이 파일 안에 설정을 둔다. 확정되면 config.Wolmido로 옮긴다.
비·안개·하늘·조명은 화면 표시만 바꾼다. 세 날씨 모두 같은 파도(미현 님 기본 파도의 절반 높이, WAVES_WEATHER)다.
2026-11-23 가영: 폭풍(큰 파도)은 뺐다. 화면에서 비와 차이가 잘 안 보여서, 눈에 확 보이는 날씨만 남긴다.
바람 힘은 넣지 않는다: config.Owner.WIND = 물 담당(미현).
"""

import math

import numpy as np

from sim import config

# 날씨 파도 (2026-11-23 가영): 미현 님 기본 파도(WaterParams.WAVES_DEFAULT, 진폭 합 0.17 m) 모양 그대로, 높이만 WAVE_SCALE 배.
#   그대로(1.0)는 동경 님 접근·견인 제어가 파도 속에서 도착 판정을 못 맞춰 STAGE_TIMEOUT (오늘 확인한 선미 정렬 문제).
#   동경 님 쪽이 고쳐지면 1.0으로 올릴 것.
WAVE_SCALE = 0.5
WAVES_WEATHER = tuple((amp * WAVE_SCALE, length, heading, phase)
                      for amp, length, heading, phase in config.WaterParams.WAVES_DEFAULT)

# 날씨 환경 3개. 같은 맵·같은 사람 위치에서 날씨만 다르다.
#   waves: 파도 성분 튜플. rain: 화면 빗줄기 수 (0 = 비 없음), rain_speed: 떨어지는 속도 [m/s]
#   visibility: 안개 가시거리 [m] (None = 안개 없음). 카메라에서 이 거리의 물체가 안개 색에 95% 덮인다.
#     가까울수록 옅다: 덮이는 정도 = 1 - exp(-3 (거리 - FOG_NEAR) / (visibility - FOG_NEAR))
#   light: 조명 밝기 배율 (맑음 1). sky: 하늘·안개 색 (None = 맵 기본 하늘). sky_mix: 맵 하늘을 sky 색으로 섞는 비율
#   mist: 물 위에 낮게 깔려 흘러가는 물안개 덩어리 수. water_rgba: 수면 색
# 안개는 MuJoCo 기본 안개(mjRND_FOG)를 쓰지 않는다 (2026-11-24 확인: WSL 그래픽에서 안개 색이 안 먹고 검게 나옴).
#   대신 카메라 앞에 반투명 안개 막을 거리별로 여러 겹 그린다(draw_fog). 멀수록 여러 겹 뒤라 더 뿌옇다.
WEATHERS = {
    # 1. 맑음: 밝은 하늘
    "clear": dict(waves=WAVES_WEATHER, rain=0, rain_speed=0.0, visibility=None, light=1.0,
                  sky=None, sky_mix=0.0, mist=0, water_rgba=(.06, .32, .45, .6)),
    # 2. 비: 굵은 비, 어두운 하늘, 빗속 가시거리 약 70 m
    "rain": dict(waves=WAVES_WEATHER, rain=900, rain_speed=9.0, visibility=110.0, light=.45,
                  sky=(.36, .39, .42), sky_mix=.85, mist=0, water_rgba=(.13, .22, .25, .75)),
    # 3. 안개: 짙은 안개 (가시거리 45 m. 기본 카메라 거리 16 m에서 배·사람은 절반쯤 뿌옇고 먼 방파제·하늘은 안 보인다).
    #    하늘은 완전히 회색, 수면도 회색빛, 물 위에 물안개가 흘러간다. 안개 막은 스스로 빛나게 그려 화면에서 밝게 보이므로
    #    색은 어두운 푸른 회색으로 둔다 (2026-11-24 가영: .7~.76 은 너무 하얗다).
    "fog": dict(waves=WAVES_WEATHER, rain=0, rain_speed=0.0, visibility=45.0, light=.8,
                sky=(.55, .6, .63), sky_mix=1.0, mist=40, water_rgba=(.3, .38, .42, .75)),
}


def weather(name):
    """WEATHERS의 날씨 하나 (사본)."""
    try:
        return dict(WEATHERS[name])
    except KeyError:
        raise ValueError(f"날씨는 {tuple(WEATHERS)} 중 하나여야 합니다.") from None


class WolmidoWeather:
    """하늘·조명·안개를 바꾸고 비를 그리는 표시 전용 부품. 힘·질량·접촉이 없다."""

    REQUIRES = {}
    SKY_TEXTURE = f"{config.Names.WOLMIDO_PREFIX}_sky"
    RAIN_BOX = (30.0, 30.0, 12.0)   # 카메라 중심 둘레 빗줄기 영역 (x, y 너비, 높이) [m]
    DROP_LENGTH = .5                # 빗줄기 하나 길이 [m]
    FOG_NEAR = 4.0                  # 카메라에서 안개가 시작되는 거리 [m]
    FOG_LAYERS = 50                 # 안개 막 수. 적으면 막이 수면과 만나는 줄이 보인다 (뒤에 완전히 덮는 막 하나 더)
    MIST_BOX = 50.0                 # 물안개가 흘러 다니는 카메라 둘레 영역 너비 [m]
    MIST_DRIFT = (.6, .25)          # 물안개가 흘러가는 속도 (x, y) [m/s]

    def __init__(self, name, *, seed=0):
        self.name = name
        self.setting = weather(name)
        rng = np.random.default_rng(seed)
        # 빗줄기마다 고정된 수평 위치·시작 높이. 시간에 따라 아래로 떨어지고 바닥에 닿으면 위로 되돌아간다.
        #   실행 중에 날씨를 바꿀 수 있게(apply) 가장 많은 날씨 기준으로 만들어 두고 앞에서부터 rain 개만 쓴다.
        most = max(int(setting['rain']) for setting in WEATHERS.values())
        self._drops = rng.uniform(-.5, .5, size=(most, 3)) * self.RAIN_BOX
        # 물안개 덩어리: 수평 위치(영역 비율), 반지름 [m], 높이 [m], 진하기
        most = max(int(setting['mist']) for setting in WEATHERS.values())
        self._mist = np.column_stack([rng.uniform(0, 1, size=(most, 2)), rng.uniform(3., 7., most),
                                      rng.uniform(.4, 1.4, most), rng.uniform(.06, .14, most)])
        self._base = None

    def build(self, spec):
        """조립 전에는 바꿀 것이 없다. 하늘·조명·안개는 컴파일된 모델에서 bind가 바꾼다."""

    def bind(self, model):
        """처음 모델 값(맑음 기준 조명·하늘·안개·수면 색)을 저장하고 지금 날씨를 적용한다."""
        import mujoco

        texture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, self.SKY_TEXTURE)
        sky_pixels = None
        if texture >= 0:
            start = model.tex_adr[texture]
            size = model.tex_height[texture] * model.tex_width[texture] * model.tex_nchannel[texture]
            sky_pixels = (slice(start, start + size), int(model.tex_nchannel[texture]),
                          model.tex_data[start:start + size].copy())
        # 물 수면 표시 형상: 파도 수면(WaveSurface)과 물 모듈의 평평한 수면 판 중 있는 것
        water = [g for g in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                             for name in ("wave_surface", config.Names.WATER_SURFACE_GEOM)) if g >= 0]
        self._base = dict(
            light_diffuse=model.light_diffuse.copy(), light_specular=model.light_specular.copy(),
            head_ambient=np.array(model.vis.headlight.ambient), head_diffuse=np.array(model.vis.headlight.diffuse),
            haze=np.array(model.vis.rgba.haze),
            texture=texture, sky_pixels=sky_pixels,
            water={g: model.geom_rgba[g].copy() for g in water})
        self.apply(model, self.name)

    def apply(self, model, name):
        """날씨를 name으로 바꾼다 (실행 중에도 가능). 처음 저장한 값에서 다시 계산하므로 여러 번 바꿔도 쌓이지 않는다.
        표시만 바꾸고 물리는 그대로다. 하늘 그림이 바뀌면 그 texture id를 돌려준다 (뷰어에서 update_texture 필요)."""
        if self._base is None:
            raise ValueError("WolmidoWeather.apply 전에 bind가 필요합니다.")
        self.setting, self.name = weather(name), name
        base = self._base
        light, sky, visibility = self.setting['light'], self.setting['sky'], self.setting['visibility']
        model.light_diffuse[:] = base['light_diffuse'] * light
        model.light_specular[:] = base['light_specular'] * light
        model.vis.headlight.ambient[:] = base['head_ambient'] * light
        model.vis.headlight.diffuse[:] = base['head_diffuse'] * light
        updated = None
        if base['sky_pixels'] is not None:
            where, channels, original = base['sky_pixels']
            pixels = original.reshape(-1, channels).astype(float)
            if sky is not None:
                # 맵 기본 하늘(파란 그라데이션)을 날씨 하늘색 쪽으로 sky_mix만큼 섞는다.
                mix = self.setting['sky_mix']
                pixels[:, :3] = (1 - mix) * pixels[:, :3] + mix * 255 * np.asarray(sky)
            model.tex_data[where] = np.clip(pixels, 0, 255).astype(np.uint8).ravel()
            updated = base['texture']
        # 하늘 아래쪽 지평선 흐림(haze)을 안개 색으로. 안개 자체는 draw_fog가 그린다.
        model.vis.rgba.haze[:] = base['haze'] if visibility is None else [*self.fog_color, 1.]
        for geom, rgba in base['water'].items():
            model.geom_rgba[geom, :3] = self.setting['water_rgba'][:3]   # 투명도는 원래 형상 값 그대로
        return updated

    def add_forces(self, model, data, acc):
        """표시 전용이라 힘이 없다."""

    @property
    def foggy(self):
        """MuJoCo 기본 안개(mjRND_FOG)를 켤지. 쓰지 않는다 (WSL 그래픽에서 검게 나옴) — 안개는 draw_fog로 그린다."""
        return False

    @property
    def fog_color(self):
        sky = self.setting['sky']
        return tuple(sky) if sky is not None else (.8, .8, .8)

    @staticmethod
    def _add(scene, kind, size, pos, mat, rgba):
        """사용자 장면에 스스로 빛나는(조명 영향 없는) 표시 형상 하나. 장면이 가득 차면 False."""
        import mujoco

        if scene.ngeom >= scene.maxgeom:
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
        geom.emission, geom.specular, geom.shininess = 1., 0., 0.
        scene.ngeom += 1
        return True

    def fog_layers(self, camera):
        """카메라(MjvCamera: lookat·distance·azimuth·elevation) 앞 안개 막 [(거리 m, 진하기)]. 안개 없으면 []."""
        visibility = self.setting['visibility']
        if visibility is None:
            return []
        # 안개 시작부터 가시거리의 1.5배까지 막을 고르게 세운다. 막 한 장이 간격(step) 만큼의 안개를 맡으면
        #   거리 d까지 겹친 막이 1 - exp(-k (d - FOG_NEAR)) 만큼 덮는다. k는 가시거리에서 95%가 되게 3 / (V - near).
        near, far = self.FOG_NEAR, 1.5 * visibility
        k = 3. / (visibility - near)
        distances = np.linspace(near, far, self.FOG_LAYERS)
        alpha = 1. - math.exp(-k * (distances[1] - distances[0]))
        return [(float(d), alpha) for d in distances] + [(far + 1., 1.)]

    def draw_fog(self, scene, camera, t=0.):
        """안개 막(카메라를 향한 반투명 판)과 물안개 덩어리를 사용자 장면에 그린다. 안개 없는 날씨면 아무것도 안 한다."""
        import mujoco

        layers = self.fog_layers(camera)
        if not layers:
            return
        azimuth, elevation = math.radians(camera.azimuth), math.radians(camera.elevation)
        forward = np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth),
                            math.sin(elevation)])
        eye = np.asarray(camera.lookat, float) - camera.distance * forward
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        mat = np.column_stack([right, up, forward])   # 판의 z축 = 시선 방향
        color = self.fog_color
        for distance, alpha in layers:
            half = 1.5 * distance + 5.   # 화면을 다 덮을 만큼 넓게
            if not self._add(scene, mujoco.mjtGeom.mjGEOM_BOX, (half, half, .001), eye + distance * forward, mat,
                             (*color, alpha)):
                return
        # 물안개: 카메라 중심 둘레 MIST_BOX 안을 바람 방향으로 흘러가고, 넘어가면 반대편에서 다시 들어온다.
        count = int(self.setting['mist'])
        center = np.asarray(camera.lookat, float)[:2]
        drift = np.asarray(self.MIST_DRIFT) * t / self.MIST_BOX
        for fx, fy, radius, height, alpha in self._mist[:count]:
            # 카메라와 함께 움직이지 않게 월드 격자 기준으로 감싼다
            offset = ((np.array([fx, fy]) + drift - center / self.MIST_BOX) % 1. - .5) * self.MIST_BOX
            pos = (*(center + offset), config.WaterParams.SURFACE_Z + height)
            if not self._add(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, (radius, radius * .7, .5), pos, np.eye(3),
                             (*color, alpha)):
                return

    def rain_segments(self, center, t):
        """t 초에 center 둘레에 보일 빗줄기 (시작점, 끝점) 배열, 모양 (개수, 2, 3)."""
        own = self._drops[:int(self.setting['rain'])]
        if not len(own):
            return np.zeros((0, 2, 3))
        height = self.RAIN_BOX[2]
        top = np.array([*center[:2], config.WaterParams.SURFACE_Z + height])
        drops = top + own * [1, 1, 0]
        drops[:, 2] -= (own[:, 2] + height / 2 + self.setting['rain_speed'] * t) % height
        return np.stack([drops, drops + [0, 0, self.DROP_LENGTH]], axis=1)

    def draw_rain(self, scene, center, t):
        """빗줄기를 뷰어 사용자 장면에 그린다. 장면이 가득 차면 남는 빗줄기는 그리지 않는다."""
        import mujoco

        color = np.array([.75, .8, .86, .45], dtype=np.float32)
        for start, end in self.rain_segments(center, t):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.5, start, end)
            scene.ngeom += 1
