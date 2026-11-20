"""파도 수면 표시: 물 담당 wave_height로 계산한 수면을 높이맵(hfield) 형상으로 그린다 (담당: 가영).

물 모듈의 수면 표시(surface_visual)는 평평한 판이라 파도가 보이지 않는다. 이 부품은 같은 넓이의 높이맵을
접촉·질량 없는 표시 형상으로 두고, 화면을 그릴 때마다 update()로 지금 시각의 수면 높이를 채운다.
물리(부력·저항)는 바꾸지 않는다. 높이는 sim.water.water.wave_height 하나만 쓴다 (값을 두 곳에서 계산하지 않기).
"""

import numpy as np

from sim import config


class WaveSurface:
    REQUIRES = {}
    NAME = "wave_surface"
    # 물 모듈 수면 판과 같은 색. 파도 모양이 그림자로 보이도록 조금 덜 투명하게 한다.
    RGBA = (.06, .32, .45, .6)

    def __init__(self, waves, *, half_size=50.0, spacing=.5, base=.05, rgba=None):
        """waves: config.WaterParams의 파도 성분 튜플. half_size: 표시 넓이 반 [m] (물 판과 같은 50m).
        spacing: 격자 간격 [m]. 가장 짧은 파장(기본 파도 2m)에 4칸 이상 들어가게 0.5m."""
        from sim.water.water import wave_height

        self.waves = tuple(waves)
        if not self.waves:
            raise ValueError("파도가 없으면 물 모듈의 평평한 수면 표시를 쓰세요.")
        if half_size <= 0 or spacing <= 0 or base <= 0:
            raise ValueError("half_size·spacing·base는 양수여야 합니다.")
        self.half_size, self.base = float(half_size), float(base)
        self.rgba = self.RGBA if rgba is None else tuple(rgba)
        self.count = int(round(2 * half_size / spacing)) + 1
        axis = np.linspace(-half_size, half_size, self.count)
        # hfield 자료는 행 = y, 열 = x 순서다.
        self.x, self.y = np.meshgrid(axis, axis)
        self._height = lambda t: wave_height(self.x, self.y, t, self.waves)
        # 표시 높이 범위: 수면 ± 진폭 합. 이 범위를 hfield의 0~1로 맞춘다.
        self.amplitude = float(sum(abs(amp) for amp, *_ in self.waves))
        self.hfield_id = None

    def build(self, spec):
        import mujoco

        surface = config.WaterParams.SURFACE_Z
        spec.add_hfield(name=self.NAME, nrow=self.count, ncol=self.count,
                        size=[self.half_size, self.half_size, 2 * self.amplitude, self.base],
                        userdata=self._normalized(0.0).ravel().tolist())
        spec.worldbody.add_geom(name=self.NAME, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=self.NAME,
                                pos=[0, 0, surface - self.amplitude], rgba=list(self.rgba),
                                group=config.Groups.VISUAL, contype=0, conaffinity=0)
        # 물 모듈 수면 표시를 끄면 그쪽 조명도 빠지므로 같은 조명을 둔다.
        spec.worldbody.add_light(pos=[0, 0, 8], dir=[0, 0, -1], diffuse=[.7, .7, .7])

    def bind(self, model):
        import mujoco

        self.hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, self.NAME)
        if self.hfield_id < 0:
            raise ValueError("WaveSurface.bind 전에 build·compile이 필요합니다.")

    def add_forces(self, model, data, acc):
        """표시 전용이라 힘이 없다."""

    def _normalized(self, t):
        low = config.WaterParams.SURFACE_Z - self.amplitude
        return np.clip((self._height(t) - low) / (2 * self.amplitude), 0., 1.)

    def update(self, model, t):
        """t 초의 수면 높이로 높이맵을 채운다. 뷰어에서는 이어서 viewer.update_hfield(hfield_id)를 부른다."""
        start, rows, cols = (model.hfield_adr[self.hfield_id], model.hfield_nrow[self.hfield_id],
                             model.hfield_ncol[self.hfield_id])
        model.hfield_data[start:start + rows * cols] = self._normalized(t).ravel()

    def surface_z(self, model, data, row, col):
        """격자 (row, col) 칸에 표시된 수면 높이 [m] (시험용)."""
        size = model.hfield_size[self.hfield_id]
        value = model.hfield_data[model.hfield_adr[self.hfield_id] + row * self.count + col]
        return data.geom_xpos[model.geom(self.NAME).id][2] + value * size[2]
