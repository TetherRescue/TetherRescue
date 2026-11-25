"""월미도 해안 구조 시나리오: python -m sim.world.view_wolmido [--env calm|waves|low_tide] [--weather clear|rain|fog]
[--calm-person] [--speed 1.5] [--headless].

사람은 바다 한가운데에 구명환 없이 떠 있다. 구조 로봇(동경 A형 선체)은 구명환을 발사대에 싣고 해변 물가에서 출발해
사람 둘레를 돌아 선미를 사람 쪽으로 두고 멈춘 뒤 구명환을 쏜다(동경 Launcher·LaunchMission). 사람이 잡으면
해변 쪽으로 끌고 와 물가 10m 앞에서 멈춘다. 화면은 기본 1.5배속으로 보여 준다 (--speed). 2배속은 물리 계산이 못 따라간다 (2026-11-25 측정 x1.8~2.2 한계).
환경 3개는 같은 맵에서 조건만 다르다 (config.Wolmido.ENVIRONMENTS):
  calm      만조·잔잔한 물
  waves     만조·파도(물 담당 기본 파도)
  low_tide  간조·잔잔한 물. 해변이 넓게 드러나 물가가 사람 쪽으로 다가오고, 방파제 앞 바위가 수면 위로 나온다
[시험, 브랜치 dev/kayoung-weather, 가영 2026-11-24] 날씨와 사람 발버둥을 이 장면에 함께 넣었다.
  날씨(sim.world.weather_wolmido): clear 맑음 · rain 비 · fog 안개. 화면(하늘·조명·안개·빗줄기·수면 색)만 바꾸고
    파도·물때는 위 환경이 정한다. 화면을 보는 중에 F6 맑음 · F7 비 · F8 안개로 바로 바꿀 수 있다
    (F1~F5·글자 키는 MuJoCo 뷰어가 이미 쓴다).
  사람(sim.world.struggle): 구명환을 잡기 전까지 물속에 선 자세로 허우적대고, 주기마다 손을 흔들며 HELP!를 띄운다.
    화면 실행은 기본으로 켜고, --calm-person 으로 끈다. build_wolmido는 기본으로 끈다 (기존 시험과 같은 물리).
"""

import argparse
import math

import mujoco
import numpy as np

from sim import config
from sim.world.map_wolmido import WolmidoMap


def beach_start(world_map, victim):
    """해변 물가의 배 출발 위치 (x, y), 사람 쪽을 보는 방향 [rad], 견인 방향 [rad]·거리 [m]."""
    cfg = config.Wolmido
    start = np.array([world_map.beach_x_at_depth(cfg.BEACH_START_DEPTH), cfg.BEACH_START_Y])
    to_victim = np.asarray(victim, dtype=float) - start
    distance = float(np.linalg.norm(to_victim))
    if distance <= cfg.BEACH_TOW_STOP + 4:
        raise ValueError("사람이 해변 출발 지점에 너무 가깝습니다.")
    yaw = math.atan2(to_victim[1], to_victim[0])
    # 사람을 해변 출발 지점 쪽으로 끌고 온다. BEACH_TOW_STOP 앞까지, 최대 BEACH_TOW_MAX만큼.
    return start, yaw, yaw + math.pi, min(distance - cfg.BEACH_TOW_STOP, cfg.BEACH_TOW_MAX)


WEATHER_KEYS = {295: 'clear', 296: 'rain', 297: 'fog'}   # GLFW 키 코드: F6, F7, F8


def build_wolmido(env='calm', *, weather=None, struggle=False, launch=None, watchdog_settings=None,
                  wave_set=None, tide_drop=None):
    """월미도 맵·환경에 동경 구명환 발사 시나리오(LaunchMission)를 조립한다. (LaunchMission, WolmidoMap)을 돌려준다.

    배는 해변 물가에서 사람 쪽을 보고 출발하고, 구명환은 배 발사대에 실려 있다. 사람은 바다 한가운데에 구명환 없이 떠 있다.
    조립 순서·부품은 동경 build_launch_mission과 같고, 맵(WolmidoMap)과 환경의 파도만 더한다 (그 파일은 바꾸지 않음).
    weather: 날씨 이름(None이면 날씨 부품 없음). 표시만 바꾼다. struggle=True면 사람이 허우적대고 손을 흔든다(힘이 더해짐).
    환경에 파도가 있으면 평평한 수면 판 대신 파도 모양 수면(WaveSurface)을 그린다 (표시만).
    """
    from sim.robot.launcher import Launcher
    from sim.robot.robot import Robot
    from sim.water.water import Water
    from sim.world.grasp import GraspSettings, HumanGrasp
    from sim.world.human import Human
    from sim.world.launch_mission import LaunchMission, LaunchSettings
    from sim.world.lifebuoy import Lifebuoy
    from sim.world.rescue_mission import RescueMissionSettings
    from sim.world.struggle import Struggle
    from sim.world.tether import Tether
    from sim.world.wave_visual import WaveSurface
    from sim.world.weather_wolmido import WolmidoWeather
    from sim.world.world import build_world

    setting = WolmidoMap.environment(env)
    launch = LaunchSettings() if launch is None else launch
    if tide_drop is None:
        tide_drop = setting['tide_drop']
    if not math.isfinite(tide_drop) or tide_drop < 0:
        raise ValueError('tide_drop은 0 이상이어야 합니다.')
    world_map = WolmidoMap(tide_drop=tide_drop)
    start, yaw, tow_heading, tow_distance = beach_start(world_map, setting['victim'])
    surface = config.WaterParams.SURFACE_Z
    robot = Robot(position=(*start, surface + config.Hull.HEIGHT / 2 - config.Expect.DRAFT), yaw=yaw, winch=True)
    human = Human(position=(*setting['victim'], surface - .03), grip_sites=True)
    waves = getattr(config.WaterParams, setting['waves']) if wave_set is None else tuple(wave_set)
    water = Water(waves=waves, floating_bodies=[config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY],
                  surface_visual=not waves)
    ring = Lifebuoy(grip_sites=True)
    rope = Tether(enabled=True, attached=True)
    grasp = HumanGrasp(settings=GraspSettings(max_distance=launch.reach, max_relative_speed=launch.grab_relative_speed))
    launcher = Launcher(ring_site=grasp.ring_name)
    parts = [water, robot, human, ring, rope, grasp, launcher, world_map]
    if waves:
        parts.append(WaveSurface(waves))
    if struggle:
        parts.append(Struggle(human, grasp, facing=tow_heading))   # 선 자세, 얼굴은 배가 오는 해변 쪽
    if weather is not None:
        parts.append(WolmidoWeather(weather))
    world = build_world(parts=parts)
    mujoco.mj_forward(world.model, world.data)
    mission = LaunchMission(world, robot, launcher, grasp, rope, tow_heading=tow_heading,
                            settings=RescueMissionSettings(tow_distance=tow_distance, tow_speed=.7),
                            launch=launch, watchdog_settings=watchdog_settings)
    return mission, world_map


def boat_terrain_clearance(manager, world_map):
    """배 선체와 맵 지형 사이 최소 형상 거리 [m]. 겹치면 음수."""
    import mujoco

    model, data = manager.world.model, manager.world.data
    boat = [g for g in range(model.ngeom) if model.geom_bodyid[g] == manager.robot.body_id and model.geom_contype[g]]
    return min(mujoco.mj_geomDistance(model, data, a, b, 10., None)
               for a in boat for b in world_map.geom_ids.values())


def run_headless(mission, timeout):
    data = mission.world.data
    while data.time < timeout and not mission.arrived:
        if mission.step() is False:
            failure = mission.failure
            raise RuntimeError(f'STOPPED: {failure.code.value} at {failure.time:.1f}s [{failure.stage}] - {failure.message}')
    if not mission.arrived:
        raise RuntimeError(f'{timeout:g}초 안에 구조를 끝내지 못했습니다 ({mission.readings()["phase"]}).')
    state = mission.readings()
    shot, landing = state['shot'], state['landing']
    return (f"발사 {shot['time']:.1f}초 (거리 {shot['distance']:.2f} m) → 착수 잡는 점-손 {landing['grip_to_hand']:.2f} m → "
            f"잡음 {mission.grab_time:.1f}초 → 견인 완료 {data.time:.1f}초, 사람 진행 {state['tow_progress']:.2f} m")


def _part(world, kind):
    return next((part for part in world.parts if isinstance(part, kind)), None)


# 파도 수면 표시 갱신 (아래 upload_wave). viewer.update_hfield 는 렌더 한 장을 기다려서, 렌더가 무거울수록 오래 걸린다
# (2026-11-25 WSL 소프트웨어 렌더 측정: 맑음 60~76 ms, 비·안개 약 190 ms — 반투명 안개 막이 수면 위에 겹침).
# 매 장 부르면 화면이 4~12 장/초로 떨어져 waves 가 x1 에서도 x0.2~0.4 로 보였다.
WAVE_INTERVAL = .2   # 최소 간격 [s]. .1 은 waves x1 이 x0.4~0.8, .2 는 x0.8~1.0
WAVE_SHARE = .35     # 수면 올리기에 쓸 시간 몫의 상한: 간격 ≥ 한 번 걸린 시간 / WAVE_SHARE


def upload_wave(viewer, wave, started):
    """바뀐 수면 높이를 화면에 올리고 다음에 올릴 시각 [time.monotonic 기준 s]을 돌려준다."""
    import time

    before = time.monotonic()
    viewer.update_hfield(wave.hfield_id)
    return started + max(WAVE_INTERVAL, (time.monotonic() - before) / WAVE_SHARE)


def run_viewer(mission, title, speed=1.):
    import time
    import mujoco.viewer
    from sim.world.rope_visual import draw_rope
    from sim.world.struggle import Struggle
    from sim.world.view_launch import FRAME, TEXT_INTERVAL, Playback
    from sim.world.wave_visual import WaveSurface
    from sim.world.weather_wolmido import WolmidoWeather

    world = mission.world
    wave, struggle, sky = _part(world, WaveSurface), _part(world, Struggle), _part(world, WolmidoWeather)
    asked = []   # 키 콜백(뷰어 스레드)이 넣은 날씨 이름. 화면 루프가 lock 안에서 꺼내 적용한다

    def on_key(key):
        if sky is not None and key in WEATHER_KEYS:
            asked.append(WEATHER_KEYS[key])

    with mujoco.viewer.launch_passive(world.model, world.data, key_callback=on_key) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 16., 150., -32.
        if sky is not None:
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = sky.foggy
        # 흐른 실제 시간 × 배속만큼 진행하고, 글자는 TEXT_INTERVAL 마다만 바꾼다 (동경 view_launch 의 Playback 그대로).
        # 예전처럼 매 장 set_texts 를 부르면 WSL 에서 렌더를 기다리느라 배속보다 훨씬 느리게 보였다 (2026-11-25)
        playback = Playback(speed, world.model.opt.timestep)
        text_due, shown = 0., (time.monotonic(), world.data.time)
        wave_due = 0.   # 파도 수면은 upload_wave 가 정한 간격마다만 다시 올린다
        while viewer.is_running():
            started = time.monotonic()
            texture = None
            with viewer.lock():
                if asked:
                    texture = sky.apply(world.model, asked.pop())
                    asked.clear()
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = sky.foggy
                    text_due = 0.   # 바뀐 날씨 이름을 바로 보여 준다
                if not mission.failed and not mission.arrived:
                    playback.advance(mission.step)
                viewer.cam.lookat[:] = (world.data.xpos[mission.robot.body_id] + world.data.xpos[mission.human_body]) / 2
                viewer.user_scn.ngeom = 0   # 매 화면 줄·팔다리·빗줄기를 새로 그린다
                draw_rope(viewer.user_scn, world.model, world.data, mission.rope)
                if struggle is not None:
                    struggle.draw(viewer.user_scn, world.model, world.data)
                if sky is not None:
                    sky.draw_fog(viewer.user_scn, viewer.cam, world.data.time)
                    sky.draw_rain(viewer.user_scn, viewer.cam.lookat, world.data.time)
                wave_now = wave is not None and started >= wave_due
                if wave_now:
                    wave.update(world.model, world.data.time)
            if texture is not None:
                viewer.update_texture(texture)   # 바뀐 하늘색을 화면에 올린다
            if wave_now:
                wave_due = upload_wave(viewer, wave, started)
            if started >= text_due:
                actual = (world.data.time - shown[1]) / max(started - shown[0], 1e-9)   # 실제로 나온 배속
                text_due, shown = started + TEXT_INTERVAL, (started, world.data.time)
                state = mission.readings()
                landing = state['landing']
                status = (mission.failure.code.value if mission.failed else 'COMPLETE' if mission.arrived else state['phase'])
                weather = '' if sky is None else 'Weather\n'
                weather_value = '' if sky is None else f'{sky.name}  (F6 clear / F7 rain / F8 fog)\n'
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   f'{title} (x{speed:g}, now x{actual:.1f})\n{weather}Phase\nTime\nLauncher\nGrip-hand gap\nLanding miss\nHeld\n'
                                   'Tension\nPerson towed',
                                   f"\n{weather_value}{status}\n{world.data.time:.1f} s\n{state['launcher']}\n{state['hand_gap']:.2f} m\n"
                                   + (f"{landing['grip_to_hand']:.2f} m" if landing else '-')
                                   + f"\n{state['held']}\n{state['tension']:.1f} N\n{state['tow_progress']:.2f} m")])
            viewer.sync()
            time.sleep(max(0., FRAME - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', choices=tuple(config.Wolmido.ENVIRONMENTS), default='calm')
    parser.add_argument('--weather', choices=tuple(WEATHER_KEYS.values()), default='clear',
                        help='시작 날씨 (화면에서 F6·F7·F8로 바꿀 수 있음)')
    parser.add_argument('--calm-person', action='store_true', help='사람 허우적대기·손 흔들기 끄기')
    parser.add_argument('--speed', type=float, default=1.5, help='화면 재생 배속 (기본 1.5배, headless 에는 영향 없음)')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--timeout', type=float, default=240, help='headless 제한 시뮬레이션 시간 [초]')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed 는 양수여야 합니다.')
    mission, world_map = build_wolmido(args.env, weather=args.weather, struggle=not args.calm_person)
    if args.headless:
        try:
            print(f'[{args.env}] ' + run_headless(mission, args.timeout))
        except RuntimeError as error:
            parser.exit(1, f'[{args.env}] {error}\n')
        return
    run_viewer(mission, f'WOLMIDO / {args.env}', args.speed)


if __name__ == '__main__':
    main()
