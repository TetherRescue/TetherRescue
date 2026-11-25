"""TetherRescue 전체 시나리오 실행: python -m sim.run_sim

기본은 MuJoCo 창과 설정 창을 함께 엽니다. 로그만 보려면 --headless.
설정 창에서 맵은 하나개·월미도 둘 중 하나를 고르고, 파도·물때·날씨는 따로 고릅니다.
"""

import argparse
from dataclasses import dataclass
import math
import os
import time
import tkinter as tk

import numpy as np

# WSL/WSLg에서 창이 안 뜨면 EGL 대신 glfw를 쓴다.
os.environ.setdefault('MUJOCO_GL', 'glfw')

from sim import config
from sim.world.sortie import RescueSortie, SortieSettings

MAPS = ('hanagae', 'wolmido')
TIDES = ('high', 'low')
WEATHERS = ('clear', 'rain', 'fog')


@dataclass(frozen=True)
class SceneSetup:
    map: str = 'hanagae'
    waves: bool = False
    tide: str = 'high'
    weather: str = 'clear'

    def __post_init__(self):
        if self.map not in MAPS:
            raise ValueError(f'맵은 {MAPS} 중 하나여야 합니다.')
        if self.tide not in TIDES:
            raise ValueError(f'물때는 {TIDES} 중 하나여야 합니다.')
        if self.weather not in WEATHERS:
            raise ValueError(f'날씨는 {WEATHERS} 중 하나여야 합니다.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', choices=MAPS, default='hanagae')
    parser.add_argument('--headless', action='store_true', help='창 없이 로그만 출력')
    parser.add_argument('--timeout', type=float, default=400, help='headless 제한 시간 [초]')
    parser.add_argument('--launch-delay', type=float, default=3.0, help='하나개: 지상 대기 [초]')
    parser.add_argument('--victim-range', type=float, default=200.0, help='하나개: 해안–익수자 거리 [m]')
    parser.add_argument('--waves', action='store_true', help='파도 켜고 시작')
    parser.add_argument('--tide', choices=TIDES, default='high', help='월미도 물때')
    parser.add_argument('--weather', choices=WEATHERS, default='clear', help='월미도 날씨')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout은 유한한 양수여야 합니다.')
    setup = SceneSetup(args.map, args.waves, args.tide, args.weather)
    if args.headless:
        try:
            run_headless(open_session(setup, args), args.timeout)
        except RuntimeError as error:
            parser.exit(1, f'{error}\n')
        return
    try:
        run_with_panel(setup, args)
    except Exception as error:
        parser.exit(1, f'화면을 열지 못했습니다: {error}\n'
                       f'DISPLAY={os.environ.get("DISPLAY")!r}, MUJOCO_GL={os.environ.get("MUJOCO_GL")!r}\n')


def open_session(setup, args):
    if setup.map == 'hanagae':
        waves = config.WaterParams.WAVES_DEFAULT if setup.waves else config.WaterParams.WAVES_CALM
        settings = SortieSettings(launch_delay=args.launch_delay, victim_range=args.victim_range)
        return HanagaeSession(setup, RescueSortie(waves=waves, settings=settings))
    from sim.world.view_wolmido import build_wolmido
    waves = config.WaterParams.WAVES_DEFAULT if setup.waves else ()
    tide = 0.0 if setup.tide == 'high' else config.Wolmido.LOW_TIDE_DROP
    mission, _map = build_wolmido('calm', weather=setup.weather, struggle=True,
                                  wave_set=waves, tide_drop=tide)
    return WolmidoSession(setup, mission)


class HanagaeSession:
    speed = 1.0
    foggy = False

    def __init__(self, setup, sortie):
        self.setup = setup
        self.title = 'HANAGAE'
        self.sortie = sortie
        self.world = sortie.world
        self.look = sortie.world.data.xpos[sortie.robot.body_id].copy()
        self.distance = 35.0

    def step(self):
        self.sortie.step()
        return self.sortie.arrived

    def sync_visuals(self, viewer):
        import mujoco
        from sim.world.coast_map import CoastalMap
        coast = next(part for part in self.world.parts if isinstance(part, CoastalMap))
        coast.animate(self.world.model, self.world.data)
        mujoco.mj_forward(self.world.model, self.world.data)
        self._draw_gps(viewer.user_scn)
        return self.sortie.water.update_surface_visual(self.world.model, self.world.data), None

    def lines(self):
        state = self.sortie.readings()
        gps = '—' if state['gps_xy'] is None else f"{state['gps_xy'][0]:.1f}, {state['gps_xy'][1]:.1f}"
        dist = '—' if state['distance_to_gps'] is None else f"{state['distance_to_gps']:.1f} m"
        return (f"{state['stage']}\n{state['time']:.1f} s\n{gps}\n{dist}\n"
                f"{state['boat_xy'][0]:.1f} m\n{state['victim_xy'][0]:.1f} m")

    def _draw_gps(self, scene):
        import mujoco
        scene.ngeom = 0
        gps = self.sortie.gps_fix
        if gps is None:
            return
        start = np.array([*gps, config.WaterParams.SURFACE_Z + .4])
        end = start + np.array([0, 0, 1.2])
        geom = scene.geoms[0]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), [.2, 1., .3, 1.])
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .08, start, end)
        scene.ngeom = 1


class WolmidoSession:
    speed = 2.0

    def __init__(self, setup, mission):
        from sim.world.weather_wolmido import WolmidoWeather
        self.setup = setup
        self.title = 'WOLMIDO'
        self.mission = mission
        self.world = mission.world
        self.sky = next((part for part in mission.world.parts if isinstance(part, WolmidoWeather)), None)
        self.foggy = bool(self.sky and self.sky.foggy)
        boat = self.world.data.xpos[mission.robot.body_id]
        person = self.world.data.xpos[mission.human_body]
        self.look = (boat + person) / 2
        self.distance = 40.0

    def step(self):
        if not self.mission.failed and not self.mission.arrived:
            self.mission.step()
        return self.mission.arrived or self.mission.failed

    def sync_visuals(self, viewer):
        from sim.world.rope_visual import draw_rope
        from sim.world.struggle import Struggle
        from sim.world.wave_visual import WaveSurface
        world = self.world
        viewer.user_scn.ngeom = 0
        draw_rope(viewer.user_scn, world.model, world.data, self.mission.rope)
        struggle = next((part for part in world.parts if isinstance(part, Struggle)), None)
        wave = next((part for part in world.parts if isinstance(part, WaveSurface)), None)
        if struggle is not None:
            struggle.draw(viewer.user_scn, world.model, world.data)
        if self.sky is not None:
            self.sky.draw_fog(viewer.user_scn, viewer.cam, world.data.time)
            self.sky.draw_rain(viewer.user_scn, viewer.cam.lookat, world.data.time)
        if wave is not None:
            wave.update(world.model, world.data.time)
            return None, wave.hfield_id
        return None, None

    def lines(self):
        state = self.mission.readings()
        status = (self.mission.failure.code.value if self.mission.failed
                  else 'COMPLETE' if self.mission.arrived else state['phase'])
        return (f"{status}\n{self.world.data.time:.1f} s\n{state['launcher']}\n"
                f"{state['hand_gap']:.2f} m\n{state['tension']:.1f} N\n{state['tow_progress']:.2f} m")


def run_headless(session, timeout):
    world = session.world
    while world.data.time < timeout:
        done = session.step()
        if world.data.warning.number.any() or not all(
                np.isfinite(s).all() for s in (world.data.qpos, world.data.qvel, world.data.qacc)):
            raise RuntimeError(f'{session.title}: 비정상 물리 상태가 발생했습니다.')
        if done:
            print(f'{session.title}: {world.data.time:.1f}s\n{session.lines()}')
            return
    raise RuntimeError(f'{timeout:g}초 안에 {session.title}을 끝내지 못했습니다.')


def run_with_panel(setup, args):
    """설정 창은 유지하고, 고른 값이 바뀌면 시뮬레이션 창만 다시 연다."""
    state = {'setup': setup, 'quit': False}
    root = tk.Tk()
    root.title('TetherRescue')
    root.geometry('280x420')
    _build_panel(root, state)

    def on_close():
        state['quit'] = True

    root.protocol('WM_DELETE_WINDOW', on_close)
    current = setup
    try:
        while not state['quit']:
            session = open_session(current, args)
            print(f'맵 {session.title}: 파도 {"on" if current.waves else "off"}')
            nxt = run_viewer(session, root, state)
            if nxt is None:
                break
            current = nxt
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def _build_panel(root, state):
    setup = state['setup']
    picks = {
        'map': tk.StringVar(value=setup.map),
        'waves': tk.StringVar(value='on' if setup.waves else 'off'),
        'tide': tk.StringVar(value=setup.tide),
        'weather': tk.StringVar(value=setup.weather),
    }
    only_wolmido = []

    def chosen():
        return SceneSetup(picks['map'].get(), picks['waves'].get() == 'on',
                          picks['tide'].get(), picks['weather'].get())

    def refresh(*_):
        wolmido = picks['map'].get() == 'wolmido'
        for widget in only_wolmido:
            widget.configure(state='normal' if wolmido else 'disabled')
        state['setup'] = chosen()

    def group(title, key, options):
        frame = tk.LabelFrame(root, text=title, padx=8, pady=4)
        frame.pack(fill='x', padx=10, pady=6)
        made = []
        for value, label in options:
            button = tk.Radiobutton(frame, text=label, variable=picks[key], value=value, command=refresh)
            button.pack(anchor='w')
            made.append(button)
        return made

    group('Map', 'map', (('hanagae', 'Hanagae'), ('wolmido', 'Wolmido')))
    group('Waves', 'waves', (('off', 'Off'), ('on', 'On')))
    only_wolmido += group('Tide (Wolmido)', 'tide', (('high', 'High'), ('low', 'Low')))
    only_wolmido += group('Weather (Wolmido)', 'weather', (('clear', 'Clear'), ('rain', 'Rain'), ('fog', 'Fog')))
    refresh()


def run_viewer(session, root, state):
    """시뮬레이션 창. 설정이 바뀌면 그 설정을, 창을 닫으면 None을 돌려준다."""
    import mujoco
    import mujoco.viewer

    world = session.world
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.opt.geomgroup[config.Groups.VISUAL] = 1
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = session.foggy
        # 처음 시점만 둔다. 이후 휠·드래그를 매 프레임 덮어쓰지 않는다.
        viewer.cam.lookat[:] = session.look
        viewer.cam.distance = session.distance
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -25
        # 맵이 넓으면 휠 줌 하한이 수 m에서 멈춘다. 사람 가까이까지, 바다 끝까지 보이게 범위를 연다.
        world.model.stat.extent = 8.0
        world.model.vis.map.znear = 0.001
        world.model.vis.map.zfar = 80.0
        budget = 0.0
        frame = 1 / 60
        while viewer.is_running():
            try:
                root.update()
            except tk.TclError:
                return None
            if state['quit']:
                return None
            if state['setup'] != session.setup:
                return state['setup']
            started = time.monotonic()
            with viewer.lock():
                while budget < frame * session.speed:
                    session.step()
                    budget += world.model.opt.timestep
                budget = max(0.0, budget - frame * session.speed)
                mesh_id, hfield_id = session.sync_visuals(viewer)
            if mesh_id is not None:
                viewer.update_mesh(mesh_id)
            if hfield_id is not None:
                viewer.update_hfield(hfield_id)
            waves = 'on' if session.setup.waves else 'off'
            extra = '' if session.setup.map == 'hanagae' else f"\n{session.setup.tide} / {session.setup.weather}"
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               f'{session.title}  x{session.speed:g}\nWaves{extra}\n\nStatus',
                               f'\n{waves}{extra}\n\n{session.lines()}')])
            viewer.sync()
            time.sleep(max(0.0, frame - (time.monotonic() - started)))
    return None


if __name__ == '__main__':
    main()
