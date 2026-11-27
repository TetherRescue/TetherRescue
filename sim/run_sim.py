"""TetherRescue 전체 시나리오 실행: python -m sim.run_sim

기본은 MuJoCo 창과 설정 창을 함께 엽니다. 로그만 보려면 --headless.
설정 창은 맵과 파도만 먼저 보여 준다. 물때·날씨는 월미도를 고른 뒤에 나타난다.
"""

import argparse
from dataclasses import dataclass
import math
import os
import signal
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
    parser.add_argument('--waves', action='store_true',
                        help='파도 켜고 시작 (철썩 시험의 --sea three: 너울·풍랑·잔물결)')
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
        waves = () if setup.waves else config.WaterParams.WAVES_CALM
        settings = SortieSettings(launch_delay=args.launch_delay, victim_range=args.victim_range)
        return HanagaeSession(setup, RescueSortie(waves=waves, settings=settings, splash_sea=setup.waves))
    from sim.world.view_wolmido import build_wolmido
    waves = config.WaterParams.WAVES_THREE if setup.waves else ()
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
        from sim.world.struggle import Struggle
        coast.animate(self.world.model, self.world.data)
        mujoco.mj_forward(self.world.model, self.world.data)
        impact = getattr(self.sortie.water, 'impact', None)
        if impact is not None:
            impact.update(float(self.world.data.time))
        self._draw_gps(viewer.user_scn)
        struggle = next((part for part in self.world.parts if isinstance(part, Struggle)), None)
        if struggle is not None:
            struggle.draw(viewer.user_scn, self.world.model, self.world.data)
        if impact is not None:
            impact.draw(viewer.user_scn, float(self.world.data.time))
        box = getattr(self, '_weather_box', None)
        if box is not None:
            box.draw_fog(viewer.user_scn, viewer.cam, self.world.data.time)
            box.draw_rain(viewer.user_scn, viewer.cam.lookat, self.world.data.time)
        return self.sortie.water.update_surface_visual(self.world.model, self.world.data), None

    def lines(self):
        from sim.world.coast_map import SHORE_X
        state = self.sortie.readings()
        stage = {'STOWED': 'on the beach', 'LAUNCHING': 'entering water',
                 'TRANSIT': 'heading out', 'ARRIVED': 'at the person'}.get(state['stage'], state['stage'])
        boat = state['boat_xy'][0]
        place = f'{-boat:.0f} m offshore' if boat < SHORE_X else f'{boat - SHORE_X:.0f} m inland'
        goal = 'not set yet' if state['distance_to_gps'] is None else f"{state['distance_to_gps']:.0f} m left"
        return f"Stage  {stage}\nTime   {state['time']:.1f} s\nBoat   {place}\nGoal   {goal}"

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
        return (f"Phase  {status}\nTime   {self.world.data.time:.1f} s\n"
                f"Tow    {state['tow_progress']:.1f} m")


def apply_controls(session, state):
    """파도 높이·배 속도·날씨를 지금 시뮬레이션에 바로 반영한다. 창을 다시 열지 않는다."""
    _apply_wave_height(session, state.get('wave_height', 0.0))
    _apply_boat_speed(session, state.get('boat_speed', 1.5))
    return _apply_weather(session, state.get('weather', session.setup.weather))


def _apply_wave_height(session, meters):
    """가장 큰 물결의 평균 대비 높이 [m]. 0이면 수면이 평평하다."""
    meters = max(0.0, float(meters))
    for part in session.world.parts:
        if getattr(part, 'waves', None) is None:
            continue
        _scale_part_waves(part, meters)


def _scale_part_waves(part, meters):
    base = getattr(part, '_wave_base', None)
    if base is None:
        from sim.water.sea import Sea
        waves = part.waves
        if isinstance(waves, Sea):
            peak = float(np.max(np.abs(waves.groups[0][1]))) if waves.groups[0][1].size else 0.2
            part._wave_base = ('sea', None, peak or 0.2)
        elif waves:
            peak = max(abs(amp) for amp, *_rest in waves) or 0.2
            part._wave_base = ('tuple', tuple(waves), peak)
        else:
            base_waves = config.WaterParams.WAVES_THREE
            peak = max(abs(amp) for amp, *_rest in base_waves)
            part._wave_base = ('tuple', base_waves, peak)
        base = part._wave_base
    kind, data, peak = base
    scale = 0.0 if peak <= 0 else meters / peak
    if kind == 'sea':
        part.waves.scaled(scale)
    else:
        part.waves = tuple((amp * scale, length, direction, phase) for amp, length, direction, phase in data)


def _apply_boat_speed(session, speed):
    """자동 항해 최고 속도 [m/s]."""
    from dataclasses import replace
    speed = float(np.clip(speed, 0.2, 3.0))
    sortie = getattr(session, 'sortie', None)
    if sortie is not None:
        sortie.settings = replace(
            sortie.settings, approach=replace(sortie.settings.approach, max_speed=speed))
        if sortie.controller is not None:
            sortie.controller.settings = replace(sortie.controller.settings, max_speed=speed)
        return
    mission = getattr(session, 'mission', None)
    if mission is None:
        return
    mission.settings = replace(mission.settings, tow_speed=speed)
    controller = getattr(mission, 'controller', None)
    if controller is not None and getattr(controller, 'settings', None) is not None:
        controller.settings = replace(controller.settings, max_speed=speed)


def _apply_weather(session, name):
    """날씨 표시를 바꾼다. 월미도는 하늘·조명까지, 하나개는 비·안개만 그린다."""
    sky = getattr(session, 'sky', None)
    if sky is not None and getattr(sky, '_base', None) is not None:
        if sky.name != name:
            return sky.apply(session.world.model, name)
        return None
    from sim.world.weather_wolmido import WolmidoWeather
    box = getattr(session, '_weather_box', None)
    if box is None or box.name != name:
        session._weather_box = WolmidoWeather(name)
    return None


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


def _arm_interrupt():
    """Ctrl+C는 프로세스를 바로 죽인다. 파이썬에서 받으면 화면 잠금 중에 처리가 밀려 창이 남는다.

    Ctrl+Z는 무시한다. 일시정지로 두면 창만 얼어 있다.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if hasattr(signal, 'SIGTSTP'):
        signal.signal(signal.SIGTSTP, signal.SIG_IGN)


def _request_stop(state, signum, _frame):
    """창의 종료 버튼이 루프를 빠져나오게 표시한다. 터미널 Ctrl+C는 이 함수를 쓰지 않는다."""
    state['quit'] = True
    if signum == getattr(signal, 'SIGTSTP', None):
        return


def run_with_panel(setup, args):
    """설정 창은 유지하고, 고른 값이 바뀌면 시뮬레이션 창만 다시 연다."""
    state = {'setup': setup, 'quit': False, 'viewer': None}
    _ensure_korean_font()
    root = tk.Tk()
    state['root'] = root
    root.title('TetherRescue')
    _build_panel(root, state)

    def on_close():
        state['quit'] = True

    root.protocol('WM_DELETE_WINDOW', on_close)
    _arm_interrupt()
    current = setup
    try:
        while not state['quit']:
            _arm_interrupt()
            session = open_session(current, args)
            print(f'맵 {session.title}: 파도 {"on" if current.waves else "off"}')
            nxt = run_viewer(session, root, state)
            if nxt is None:
                break
            current = nxt
    finally:
        state['viewer'] = None
        try:
            root.destroy()
        except tk.TclError:
            pass


_HOVER_DELAY_MS = 1000
_HOVER = {'win': None, 'owner': None, 'job': None, 'job_owner': None}
_HANGUL = None


def _hangul_draw():
    """맑은 고딕으로 글자 그림을 만든다. Tk 기본 글꼴은 한글이 빈칸으로 나온다."""
    global _HANGUL
    if _HANGUL is not None:
        return _HANGUL
    import ctypes
    ft = ctypes.CDLL('libfreetype.so.6')
    cairo = ctypes.CDLL('libcairo.so.2')
    ft.FT_Init_FreeType.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    ft.FT_Init_FreeType.restype = ctypes.c_int
    ft.FT_New_Face.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)]
    ft.FT_New_Face.restype = ctypes.c_int
    library, face = ctypes.c_void_p(), ctypes.c_void_p()
    if ft.FT_Init_FreeType(ctypes.byref(library)) != 0:
        return None
    path = '/mnt/c/Windows/Fonts/malgun.ttf'
    if not os.path.isfile(path):
        path = os.path.join(os.path.expanduser('~'), '.local', 'share', 'fonts', 'malgun.ttf')
    if ft.FT_New_Face(library, path.encode(), 0, ctypes.byref(face)) != 0:
        return None
    cairo.cairo_ft_font_face_create_for_ft_face.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cairo.cairo_ft_font_face_create_for_ft_face.restype = ctypes.c_void_p
    font = cairo.cairo_ft_font_face_create_for_ft_face(face, 0)
    if not font:
        return None
    cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    cairo.cairo_image_surface_create.restype = ctypes.c_void_p
    cairo.cairo_create.argtypes = [ctypes.c_void_p]
    cairo.cairo_create.restype = ctypes.c_void_p
    cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
    cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
    cairo.cairo_set_font_face.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cairo.cairo_set_font_size.argtypes = [ctypes.c_void_p, ctypes.c_double]
    cairo.cairo_set_source_rgb.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    cairo.cairo_rectangle.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    cairo.cairo_fill.argtypes = [ctypes.c_void_p]
    cairo.cairo_move_to.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    cairo.cairo_show_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    cairo.cairo_surface_write_to_png_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    cairo.cairo_surface_write_to_png_stream.restype = ctypes.c_int

    class Ext(ctypes.Structure):
        _fields_ = [('xb', ctypes.c_double), ('yb', ctypes.c_double), ('w', ctypes.c_double),
                    ('h', ctypes.c_double), ('xa', ctypes.c_double), ('ya', ctypes.c_double)]

    cairo.cairo_text_extents.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(Ext)]
    measure_surface = cairo.cairo_image_surface_create(0, 8, 8)
    measure = cairo.cairo_create(measure_surface)
    cairo.cairo_set_font_face(measure, font)
    cairo.cairo_set_font_size(measure, 15)
    _HANGUL = {'cairo': cairo, 'font': font, 'measure': measure, 'Ext': Ext, 'ctypes': ctypes}
    return _HANGUL


def _hangul_photo(widget, text, *, bg='#d9d9d9', fg='#1c1c1c', max_width=280, pad_width=None, size=15):
    """한글이 포함된 문자열을 그림으로 만든다. 없으면 None."""
    import ctypes
    draw = _hangul_draw()
    if draw is None or text.isascii():
        return None
    cairo, measure, Ext = draw['cairo'], draw['measure'], draw['Ext']
    cairo.cairo_set_font_size(measure, size)

    def width(line):
        ext = Ext()
        cairo.cairo_text_extents(measure, line.encode(), ctypes.byref(ext))
        return ext.w, ext

    lines, current = [], ''
    for word in text.split(' '):
        trial = word if not current else f'{current} {word}'
        if width(trial)[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    extents = [width(line)[1] for line in lines]
    line_h = max(int(item.h) + 6 for item in extents)
    width_px = pad_width or min(max_width, max(int(item.w) for item in extents) + 16)
    height = line_h * len(lines) + 10
    surface = cairo.cairo_image_surface_create(0, max(width_px, 1), max(height, 1))
    context = cairo.cairo_create(surface)
    red, green, blue = _rgb(bg)
    cairo.cairo_set_source_rgb(context, red, green, blue)
    cairo.cairo_rectangle(context, 0, 0, width_px, height)
    cairo.cairo_fill(context)
    cairo.cairo_set_font_face(context, draw['font'])
    cairo.cairo_set_font_size(context, size)
    fred, fgreen, fblue = _rgb(fg)
    cairo.cairo_set_source_rgb(context, fred, fgreen, fblue)
    for index, (line, ext) in enumerate(zip(lines, extents)):
        cairo.cairo_move_to(context, 2 - ext.xb, 2 + index * line_h - ext.yb)
        cairo.cairo_show_text(context, line.encode())
    buf = bytearray()
    writer = draw['ctypes'].CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint)

    def write(_closure, data, length):
        buf.extend(ctypes.string_at(data, length))
        return 0

    callback = writer(write)
    cairo.cairo_surface_write_to_png_stream(surface, callback, None)
    cairo.cairo_destroy(context)
    cairo.cairo_surface_destroy(surface)
    photo = tk.PhotoImage(data=bytes(buf), master=widget.winfo_toplevel())
    images = getattr(widget.winfo_toplevel(), '_hangul_images', None)
    if images is None:
        widget.winfo_toplevel()._hangul_images = images = []
    images.append(photo)
    return photo


def _rgb(color):
    color = str(color).strip()
    if color.startswith('#') and len(color) == 7:
        return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (0.85, 0.85, 0.85)


def _paint(widget, shown, bg=None, fg='#1c1c1c', pad_width=None):
    """한글 문자열이면 그림으로 바꿔 붙인다. 위젯 text 값은 그대로 둔다."""
    if shown.isascii():
        return
    if bg is None:
        try:
            bg = widget.cget('background')
        except tk.TclError:
            bg = '#f6f3ee'
    photo = _hangul_photo(widget, shown, bg=bg, fg=fg, pad_width=pad_width)
    if photo is not None:
        widget.configure(image=photo, bg=bg, activebackground=bg)


def _ensure_korean_font():
    """창을 만들기 전에 맑은 고딕을 쓸 수 있게 해 둔다. 없으면 그냥 넘어간다."""
    src = os.path.join('/mnt/c/Windows/Fonts', 'malgun.ttf')
    dest_dir = os.path.join(os.path.expanduser('~'), '.local', 'share', 'fonts')
    dest = os.path.join(dest_dir, 'malgun.ttf')
    if os.path.exists(dest) or not os.path.isfile(src):
        return
    os.makedirs(dest_dir, exist_ok=True)
    os.symlink(src, dest)
    import subprocess
    subprocess.run(['fc-cache', '-f', dest_dir], check=False)


def _apply_korean_font(root):
    import tkinter.font as tkfont
    families = set(tkfont.families(root))
    family = next((name for name in ('Malgun Gothic', '맑은 고딕', 'Noto Sans CJK KR', 'NanumGothic')
                   if name in families), None)
    if family is None:
        return None
    root.option_add('*Font', f'{{{family}}} 10')
    font = tkfont.Font(root=root, family=family, size=10)
    root._korean_font = font
    return font


def _hover(widget, text):
    """마우스를 약 1초 올려 두면 설명 하나를 보여 준다. 지나가면 만들지 않는다."""

    def cancel():
        if _HOVER['job_owner'] is not widget:
            return
        job = _HOVER['job']
        _HOVER['job'] = None
        _HOVER['job_owner'] = None
        if job is not None:
            try:
                widget.after_cancel(job)
            except tk.TclError:
                pass

    def close_tip():
        if _HOVER['owner'] is not widget:
            return
        win = _HOVER['win']
        _HOVER['win'] = None
        _HOVER['owner'] = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass

    def show():
        _HOVER['job'] = None
        _HOVER['job_owner'] = None
        if _HOVER['win'] is not None and _HOVER['owner'] is not widget:
            old = _HOVER['win']
            _HOVER['win'] = None
            _HOVER['owner'] = None
            try:
                old.destroy()
            except tk.TclError:
                pass
        if _HOVER['win'] is not None:
            return
        win = tk.Toplevel(widget)
        win.overrideredirect(True)
        win.geometry(f'+{widget.winfo_rootx() + 16}+{widget.winfo_rooty() + widget.winfo_height() + 2}')
        font = getattr(widget.winfo_toplevel(), '_korean_font', None)
        label = tk.Label(win, text=text, justify='left', background='#fffde7', relief='solid',
                         borderwidth=1, padx=4, pady=4)
        if font is not None:
            label.configure(font=font)
        _paint(label, text, bg='#fffde7')
        label.pack()
        _HOVER['win'] = win
        _HOVER['owner'] = widget

    def enter(_event):
        cancel_other()
        _HOVER['job_owner'] = widget
        _HOVER['job'] = widget.after(_HOVER_DELAY_MS, show)

    def cancel_other():
        job = _HOVER['job']
        owner = _HOVER['job_owner']
        _HOVER['job'] = None
        _HOVER['job_owner'] = None
        if job is not None and owner is not None:
            try:
                owner.after_cancel(job)
            except tk.TclError:
                pass

    def leave(_event):
        cancel()
        close_tip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


def _build_panel(root, state):
    _apply_korean_font(root)
    root.configure(bg='#efeae2')
    root.geometry('320x640')
    setup = state['setup']
    picks = {
        'map': tk.StringVar(value=setup.map),
        'waves': tk.StringVar(value='on' if setup.waves else 'off'),
        'tide': tk.StringVar(value=setup.tide),
        'weather': tk.StringVar(value=setup.weather),
    }
    choices = []

    def chosen():
        return SceneSetup(picks['map'].get(), picks['waves'].get() == 'on',
                          picks['tide'].get(), picks['weather'].get())

    def group(title, key, options):
        frame = tk.Frame(root, bg='#efeae2')
        caption = tk.Label(frame, text=title, anchor='w', bg='#efeae2')
        caption.pack(fill='x', padx=12, pady=(8, 2))
        _paint(caption, title, bg='#efeae2')
        for value, label, help_text in options:
            button = tk.Button(frame, text=label, anchor='w', relief='flat', bd=0,
                               command=lambda k=key, v=value: picks[k].set(v) or refresh())
            button.pack(fill='x', padx=12, pady=2)
            _hover(button, help_text)
            choices.append((key, value, button, label))
        return frame

    def refresh(*_):
        if picks['map'].get() == 'wolmido':
            for frame in extra:
                frame.pack(fill='x')
        else:
            for frame in extra:
                frame.pack_forget()
        for key, value, button, label in choices:
            selected = picks[key].get() == value
            bg = '#245c94' if selected else '#f7f4ef'
            fg = '#ffffff' if selected else '#1c1c1c'
            mark = '●  ' if selected else '○  '
            _paint(button, mark + label, bg=bg, fg=fg, pad_width=280)
        names = {'hanagae': '하나개', 'wolmido': '월미도', 'off': '잔잔', 'on': '출렁임',
                 'high': '만조', 'low': '간조', 'clear': '맑음', 'rain': '비', 'fog': '안개'}
        now = f"지금  {names[picks['map'].get()]}  ·  {names[picks['waves'].get()]}"
        if picks['map'].get() == 'wolmido':
            now += f"  ·  {names[picks['tide'].get()]}  ·  {names[picks['weather'].get()]}"
        _paint(status, now, bg='#efeae2', pad_width=280)
        state['setup'] = chosen()

    state.setdefault('paused', False)
    state.setdefault('reset', False)
    state.setdefault('wave_height', 0.2 if setup.waves else 0.0)
    state.setdefault('boat_speed', 1.5)
    state.setdefault('weather', setup.weather)
    status = tk.Label(root, text='지금', anchor='w', bg='#efeae2')
    status.pack(fill='x', padx=12, pady=(10, 0))
    run = tk.Frame(root, bg='#efeae2')
    run.pack(fill='x', padx=10, pady=(8, 0))
    controls = (
        ('Pause', '일시정지', lambda: state.update(paused=True)),
        ('Run', '계속', lambda: state.update(paused=False)),
        ('Reset', '처음부터', lambda: state.update(paused=False, reset=True)),
        ('Quit', '종료', lambda: state.update(quit=True)),
    )
    for key, shown, command in controls:
        button = tk.Button(run, text=key, command=command, relief='flat')
        button.pack(side='left', padx=2)
        _paint(button, shown, bg='#d9d3c7')
    def slider(caption, key, low, high, step, help_text):
        label = tk.Label(root, text=caption, anchor='w', bg='#efeae2')
        label.pack(fill='x', padx=12, pady=(10, 0))
        _paint(label, caption, bg='#efeae2', pad_width=280)
        variable = tk.DoubleVar(value=state[key])
        variable.trace_add('write', lambda *_args, name=key, var=variable: state.__setitem__(name, float(var.get())))
        scale = tk.Scale(root, from_=low, to=high, resolution=step, orient='horizontal',
                         variable=variable, bg='#efeae2', highlightthickness=0, length=280)
        scale.pack(padx=12)
        _hover(scale, help_text)
        return scale

    slider('파도 높이  0–1.5 m', 'wave_height', 0.0, 1.5, 0.05,
           '수면이 평균보다 올라가는 높이입니다. 0이면 잔잔하고, 올릴수록 물결이 커집니다.')
    slider('배 속도  0.2–3 m/s', 'boat_speed', 0.2, 3.0, 0.1,
           '배가 사람 쪽으로 가는 최고 속도입니다. 바로 반영되고, 창을 다시 열지 않습니다.')
    weather_row = tk.Frame(root, bg='#efeae2')
    weather_caption = tk.Label(weather_row, text='날씨', anchor='w', bg='#efeae2')
    weather_caption.pack(fill='x', padx=12, pady=(8, 2))
    _paint(weather_caption, '날씨', bg='#efeae2')
    weather_buttons = []

    def paint_weather():
        for value, button, label in weather_buttons:
            selected = state.get('weather') == value
            bg = '#245c94' if selected else '#f7f4ef'
            fg = '#ffffff' if selected else '#1c1c1c'
            mark = '●  ' if selected else '○  '
            _paint(button, mark + label, bg=bg, fg=fg, pad_width=84)

    def pick_weather(name):
        state['weather'] = name
        paint_weather()

    for value, label, help_text in (
        ('clear', '맑음', '비와 안개가 없습니다.'),
        ('rain', '비', '비가 내립니다. 화면만 바뀌고 배의 움직임은 같습니다.'),
        ('fog', '안개', '먼 곳이 흐려집니다. 화면만 바뀝니다.'),
    ):
        button = tk.Button(weather_row, text=label, relief='flat', bd=0,
                           command=lambda name=value: pick_weather(name))
        button.pack(side='left', padx=4)
        _hover(button, help_text)
        weather_buttons.append((value, button, label))
    weather_row.pack(fill='x')
    paint_weather()
    group('맵', 'map', (
        ('hanagae', '하나개', '동쪽 모래사장입니다. 배는 모래에서 출발해 약 200 m 밖 사람에게 갑니다.'),
        ('wolmido', '월미도', '섬 구조입니다. 물때는 이 맵을 고른 뒤에 아래에 나옵니다.'),
    )).pack(fill='x')
    group('파도', 'waves', (
        ('off', '잔잔', '수면이 평평합니다. 해안이나 바위에 철썩이지 않습니다.'),
        ('on', '출렁임', '수면이 움직입니다. 하나개에서는 모래와 돌섬에 닿으면 물보라가 납니다.'),
    )).pack(fill='x')
    extra = [
        group('물때', 'tide', (
            ('high', '만조', '월미도만. 물이 방파제 높이까지 차 있습니다.'),
            ('low', '간조', '월미도만. 물이 빠져 해안이 더 드러납니다.'),
        )),
    ]
    state['extra'] = extra
    refresh()


def run_viewer(session, root, state):
    """시뮬레이션 창. 설정이 바뀌면 그 설정을, 창을 닫으면 None을 돌려준다."""
    import mujoco
    import mujoco.viewer

    world = session.world
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        state['viewer'] = viewer
        try:
            return _drive_viewer(viewer, session, root, state, world)
        finally:
            state['viewer'] = None


def _drive_viewer(viewer, session, root, state, world):
    import mujoco

    viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
    viewer.opt.geomgroup[config.Groups.VISUAL] = 1
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = session.foggy
    # 처음 시점만 둔다. 이후 lookat·distance·azimuth는 마우스가 바꾼 값을 덮어쓰지 않는다.
    viewer.cam.lookat[:] = session.look
    viewer.cam.distance = session.distance
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -25
    # 휠 줌 한계는 맵 크기(stat.extent)의 0.01~100배다. 하나개 맵은 약 400 m라
    # 기본값이면 4 m보다 가까이 못 가고, 너무 줄이면 드래그가 거의 안 움직인다.
    # 40 m면 가장 가까이 0.4 m, 가장 멀리 4 km.
    world.model.stat.extent = 40.0
    world.model.vis.map.znear = 0.001
    world.model.vis.map.zfar = 100.0
    budget = 0.0
    frame = 1 / 60
    while viewer.is_running() and not state['quit']:
        _arm_interrupt()
        try:
            root.update()
        except tk.TclError:
            state['quit'] = True
            return None
        if state['quit']:
            return None
        if state.get('reset'):
            state['reset'] = False
            state['paused'] = False
            return session.setup
        if state['setup'] != session.setup:
            return state['setup']
        started = time.monotonic()
        # 물리와 화면 갱신을 나눠 잠가서, 그 사이에 휠·드래그가 들어온다.
        texture = apply_controls(session, state)
        if not state.get('paused'):
            with viewer.lock():
                while budget < frame * session.speed and not state['quit'] and not state.get('paused'):
                    session.step()
                    budget += world.model.opt.timestep
                budget = max(0.0, budget - frame * session.speed)
        if state['quit']:
            return None
        with viewer.lock():
            mesh_id, hfield_id = session.sync_visuals(viewer)
        if state['quit']:
            return None
        if mesh_id is not None:
            viewer.update_mesh(mesh_id)
        if hfield_id is not None:
            viewer.update_hfield(hfield_id)
        if texture is not None:
            viewer.update_texture(texture)
        waves = f"wave {state.get('wave_height', 0):.2f}m  boat {state.get('boat_speed', 1.5):.1f}m/s  {state.get('weather', 'clear')}"
        sky = '' if session.setup.map == 'hanagae' else f'   {session.setup.tide}   {session.setup.weather}'
        viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           f'{session.title}   {waves}{sky}\n{session.lines()}', '')])
        viewer.sync()
        time.sleep(max(0.0, frame - (time.monotonic() - started)))
    return None


if __name__ == '__main__':
    main()
