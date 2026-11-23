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
#   visibility: 안개 가시거리 [m] (None = 안개 없음). 이 거리에서 물체가 안개 색에 완전히 덮인다
#   light: 조명 밝기 배율 (맑음 1). sky: 하늘·안개 색 (None = 맵 기본 하늘). water_rgba: 파도 수면 색
WEATHERS = {
    # 1. 맑음: 밝은 하늘
    "clear": dict(waves=WAVES_WEATHER, rain=0, rain_speed=0.0, visibility=None, light=1.0,
                  sky=None, water_rgba=(.06, .32, .45, .6)),
    # 2. 비: 굵은 비, 어두운 하늘, 빗속 가시거리 약 70 m
    "rain": dict(waves=WAVES_WEATHER, rain=900, rain_speed=9.0, visibility=70.0, light=.45,
                  sky=(.36, .39, .42), water_rgba=(.13, .22, .25, .75)),
    # 3. 안개: 짙은 안개 (가시거리 25 m. 출발점에서 사람까지 약 36 m라 처음엔 사람이 안 보인다)
    "fog": dict(waves=WAVES_WEATHER, rain=0, rain_speed=0.0, visibility=25.0, light=.75,
                sky=(.78, .8, .82), water_rgba=(.06, .32, .45, .6)),
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

    def __init__(self, name, *, seed=0):
        self.name = name
        self.setting = weather(name)
        rng = np.random.default_rng(seed)
        count = int(self.setting['rain'])
        # 빗줄기마다 고정된 수평 위치·시작 높이. 시간에 따라 아래로 떨어지고 바닥에 닿으면 위로 되돌아간다.
        self._drops = rng.uniform(-.5, .5, size=(count, 3)) * self.RAIN_BOX

    def build(self, spec):
        """조립 전에는 바꿀 것이 없다. 하늘·조명·안개는 컴파일된 모델에서 bind가 바꾼다."""

    def bind(self, model):
        import mujoco

        light, sky, visibility = self.setting['light'], self.setting['sky'], self.setting['visibility']
        model.light_diffuse[:] *= light
        model.light_specular[:] *= light
        model.vis.headlight.ambient[:] = np.asarray(model.vis.headlight.ambient) * light
        model.vis.headlight.diffuse[:] = np.asarray(model.vis.headlight.diffuse) * light
        if sky is not None:
            texture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, self.SKY_TEXTURE)
            if texture >= 0:
                # 맵 기본 하늘(파란 그라데이션)을 날씨 하늘색 쪽으로 85% 섞는다.
                start = model.tex_adr[texture]
                size = model.tex_height[texture] * model.tex_width[texture] * model.tex_nchannel[texture]
                pixels = model.tex_data[start:start + size].reshape(-1, model.tex_nchannel[texture]).astype(float)
                pixels[:, :3] = .15 * pixels[:, :3] + .85 * 255 * np.asarray(sky)
                model.tex_data[start:start + size] = np.clip(pixels, 0, 255).astype(np.uint8).ravel()
        if visibility is not None:
            # 안개 거리는 모델 크기(stat.extent) 배수로 적는다. 가시거리의 20%부터 옅어지기 시작한다.
            extent = float(model.stat.extent)
            model.vis.map.fogstart = .2 * visibility / extent
            model.vis.map.fogend = visibility / extent
            color = [*(sky if sky is not None else (.8, .8, .8)), 1.]
            model.vis.rgba.fog[:] = color
            model.vis.rgba.haze[:] = color

    def add_forces(self, model, data, acc):
        """표시 전용이라 힘이 없다."""

    @property
    def foggy(self):
        return self.setting['visibility'] is not None

    def rain_segments(self, center, t):
        """t 초에 center 둘레에 보일 빗줄기 (시작점, 끝점) 배열, 모양 (개수, 2, 3)."""
        if not len(self._drops):
            return np.zeros((0, 2, 3))
        height = self.RAIN_BOX[2]
        top = np.array([*center[:2], config.WaterParams.SURFACE_Z + height])
        drops = top + self._drops * [1, 1, 0]
        drops[:, 2] -= (self._drops[:, 2] + height / 2 + self.setting['rain_speed'] * t) % height
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
