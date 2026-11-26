"""최종 실행이 맵 둘과 파도·물때·날씨 설정을 따로 여는지 확인한다."""
import argparse
import signal

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim.run_sim import MAPS, SceneSetup, _build_panel, _request_stop, apply_controls, open_session


def _args():
    return argparse.Namespace(launch_delay=.05, victim_range=30.)


def test_ctrl_c_kills_the_process_instead_of_waiting_on_the_viewer_lock():
    from sim.run_sim import _arm_interrupt
    previous = {name: signal.getsignal(getattr(signal, name))
                for name in ('SIGINT', 'SIGTERM', 'SIGTSTP')}
    try:
        _arm_interrupt()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_DFL
        assert signal.getsignal(signal.SIGTSTP) is signal.SIG_IGN
    finally:
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)


def test_terminal_stop_marks_quit_without_closing_under_the_lock():
    closed = []

    class Viewer:
        def close(self):
            closed.append(True)

    state = {'quit': False, 'viewer': Viewer()}
    _request_stop(state, signal.SIGTSTP, None)
    assert state['quit'] is True
    assert closed == []


def test_settings_panel_explains_choices_and_hides_wolmido_extras():
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('display 없음')
    root.withdraw()

    def packed(frame):
        try:
            frame.pack_info()
        except tk.TclError:
            return False
        return True

    def walk(widget):
        yield widget
        for child in widget.winfo_children():
            yield from walk(child)

    try:
        state = {'setup': SceneSetup('hanagae', waves=False)}
        _build_panel(root, state)
        assert all(not packed(frame) for frame in state['extra'])

        def text_of(widget):
            try:
                return widget.cget('text')
            except tk.TclError:
                return ''

        buttons = [widget for widget in walk(root) if isinstance(widget, tk.Button)]
        name = next(widget for widget in buttons if text_of(widget) == '하나개')
        assert name.cget('image')
        assert all(widget.bind('<Enter>') for widget in buttons if text_of(widget) in ('하나개', '월미도', '잔잔', '출렁임'))
        root.deiconify()
        root.update()

        def tips():
            return [child for child in name.winfo_children() if isinstance(child, tk.Toplevel)]

        def pump(seconds):
            import time
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                root.update()
                time.sleep(0.05)

        name.event_generate('<Enter>')
        name.event_generate('<Leave>')
        pump(1.2)
        assert tips() == []
        name.unbind('<Leave>')
        name.event_generate('<Enter>')
        root.update()
        assert tips() == []
        pump(1.15)
        assert tips()
        assert '모래' in tips()[0].winfo_children()[0].cget('text')
        next(widget for widget in buttons if text_of(widget) == 'Pause').invoke()
        assert state['paused'] is True
        next(widget for widget in buttons if text_of(widget) == '출렁임').invoke()
        assert state['setup'].waves is True
        next(widget for widget in buttons if text_of(widget) == '월미도').invoke()
        scales = [widget for widget in walk(root) if isinstance(widget, tk.Scale)]
        assert len(scales) == 2
        scales[0].set(0.4)
        scales[1].set(2.0)
        assert state['wave_height'] == pytest.approx(0.4)
        assert state['boat_speed'] == pytest.approx(2.0)
        next(widget for widget in buttons if text_of(widget) == '비').invoke()
        assert state['weather'] == 'rain'
        assert all(packed(frame) for frame in state['extra'])
        assert state['setup'].map == 'wolmido'
    finally:
        root.destroy()


def test_wave_height_and_boat_speed_change_the_running_scene():
    session = open_session(SceneSetup('hanagae', waves=False), _args())
    apply_controls(session, {'wave_height': 0.4, 'boat_speed': 2.0, 'weather': 'rain'})
    amps = [amp for amp, *_rest in session.sortie.water.waves]
    assert max(amps) == pytest.approx(0.4)
    assert session.sortie.settings.approach.max_speed == pytest.approx(2.0)
    assert session._weather_box.name == 'rain'
    assert session.setup.waves is False
    apply_controls(session, {'wave_height': 0.0, 'boat_speed': 1.5, 'weather': 'clear'})
    amps = [amp for amp, *_rest in session.sortie.water.waves]
    assert max(amps) == pytest.approx(0.0)


def test_two_maps_and_wolmido_settings_are_separate():
    assert MAPS == ('hanagae', 'wolmido')
    hanagae = open_session(SceneSetup('hanagae', waves=False), _args())
    assert hanagae.step() is False
    assert np.isfinite(hanagae.world.data.qpos).all()
    wolmido = open_session(SceneSetup('wolmido', waves=True, tide='low', weather='rain'), _args())
    assert wolmido.world.model.geom('wolmido_wall').id >= 0
    assert wolmido.sky.name == 'rain'
    assert wolmido.step() is False
    assert np.isfinite(wolmido.world.data.qpos).all()