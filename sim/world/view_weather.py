"""[시험용] 월미도 날씨 구조 시나리오: python -m sim.world.view_weather [--weather clear|storm|fog] [--tide high|low]
[--speed 2] [--headless]

view_wolmido와 같은 구조 흐름(동경 Launcher·LaunchMission)에 날씨만 더한다 (브랜치 try/weather, 가영 2026-11-20).
  clear  맑음: 잔잔한 물, 밝은 하늘
  storm  비/폭풍: 큰 파도(진폭 합 0.2 m, 화면에도 파도 모양 수면), 굵은 비, 어두운 하늘
  fog    안개: 잔잔한 물, 가시거리 25 m
물때(--tide)는 날씨와 따로 고른다. 사람은 구명환을 잡기 전까지 발버둥 친다 (--calm-person 으로 끔). 기존 view_wolmido·config는 바꾸지 않는다.
"""

import argparse
import math

import mujoco
import numpy as np

from sim import config
from sim.world.map_wolmido import WolmidoMap
from sim.world.view_wolmido import beach_start, run_headless
from sim.world.weather_wolmido import WEATHERS, WolmidoWeather

VICTIM = (-38.0, 4.0)   # view_wolmido 환경들과 같은 사람 위치 [m]
TIDES = {"high": 0.0, "low": config.Wolmido.LOW_TIDE_DROP}


def build_weather(name='clear', tide='high', *, struggle=True, launch=None, watchdog_settings=None):
    """월미도 맵·날씨에 동경 구명환 발사 시나리오를 조립한다. (LaunchMission, WolmidoMap, WolmidoWeather)를 돌려준다.

    조립 순서·부품은 view_wolmido.build_wolmido와 같고, 날씨 부품과 (파도가 있으면) 파도 수면 표시,
    struggle=True면 사람 발버둥(sim.world.struggle)을 더한다.
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
    from sim.world.world import build_world

    if tide not in TIDES:
        raise ValueError(f"물때는 {tuple(TIDES)} 중 하나여야 합니다.")
    sky = WolmidoWeather(name)
    waves = sky.setting['waves']
    launch = LaunchSettings() if launch is None else launch
    world_map = WolmidoMap(tide_drop=TIDES[tide])
    start, yaw, tow_heading, tow_distance = beach_start(world_map, VICTIM)
    surface = config.WaterParams.SURFACE_Z
    robot = Robot(position=(*start, surface + config.Hull.HEIGHT / 2 - config.Expect.DRAFT), yaw=yaw, winch=True)
    human = Human(position=(*VICTIM, surface - .03), grip_sites=True)
    # 파도가 있으면 물 모듈의 평평한 수면 판 대신 파도 모양 수면을 그린다 (물 모듈 설정 값만 고르고 파일은 그대로).
    water = Water(waves=waves, floating_bodies=[config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY],
                  surface_visual=not waves)
    ring = Lifebuoy(grip_sites=True)
    rope = Tether(enabled=True, attached=True)
    grasp = HumanGrasp(settings=GraspSettings(max_distance=launch.reach, max_relative_speed=launch.grab_relative_speed))
    launcher = Launcher(ring_site=grasp.ring_name)
    parts = [water, robot, human, ring, rope, grasp, launcher, world_map, sky]
    if waves:
        parts.append(WaveSurface(waves, rgba=sky.setting['water_rgba']))
    if struggle:
        parts.append(Struggle(human, grasp))
    world = build_world(parts=parts)
    mujoco.mj_forward(world.model, world.data)
    mission = LaunchMission(world, robot, launcher, grasp, rope, tow_heading=tow_heading,
                            settings=RescueMissionSettings(tow_distance=tow_distance, tow_speed=.7),
                            launch=launch, watchdog_settings=watchdog_settings)
    return mission, world_map, sky


def run_viewer(mission, sky, title, speed=1.):
    import time
    import mujoco.viewer
    from sim.world.rope_visual import draw_rope
    from sim.world.struggle import Struggle
    from sim.world.wave_visual import WaveSurface

    world = mission.world
    wave = next((part for part in world.parts if isinstance(part, WaveSurface)), None)
    struggle = next((part for part in world.parts if isinstance(part, Struggle)), None)
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 16., 150., -32.
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = sky.foggy   # 뷰어 장면에 안개 켜기
        frame, budget = 1 / 60, 0.
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                if not mission.failed and not mission.arrived:
                    budget = min(budget + frame * speed, 2 * frame * speed)
                    while budget >= world.model.opt.timestep:
                        if mission.step() is False:
                            break
                        budget -= world.model.opt.timestep
                viewer.cam.lookat[:] = (world.data.xpos[mission.robot.body_id] + world.data.xpos[mission.human_body]) / 2
                viewer.user_scn.ngeom = 0   # 매 화면 줄·빗줄기를 새로 그린다
                draw_rope(viewer.user_scn, world.model, world.data, mission.rope)
                if struggle is not None:
                    struggle.draw(viewer.user_scn, world.model, world.data)
                sky.draw_rain(viewer.user_scn, viewer.cam.lookat, world.data.time)
                if wave is not None:
                    wave.update(world.model, world.data.time)
            if wave is not None:
                viewer.update_hfield(wave.hfield_id)   # 바뀐 수면 높이를 화면에 올린다
            state = mission.readings()
            landing = state['landing']
            status = (mission.failure.code.value if mission.failed else 'COMPLETE' if mission.arrived else state['phase'])
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               f'{title} (x{speed:g})\nPhase\nTime\nLauncher\nGrip-hand gap\nLanding miss\nHeld\n'
                               'Tension\nPerson towed',
                               f"\n{status}\n{world.data.time:.1f} s\n{state['launcher']}\n{state['hand_gap']:.2f} m\n"
                               + (f"{landing['grip_to_hand']:.2f} m" if landing else '-')
                               + f"\n{state['held']}\n{state['tension']:.1f} N\n{state['tow_progress']:.2f} m")])
            viewer.sync()
            time.sleep(max(0., frame - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weather', choices=tuple(WEATHERS), default='clear')
    parser.add_argument('--tide', choices=tuple(TIDES), default='high')
    parser.add_argument('--speed', type=float, default=2., help='화면 재생 배속 (기본 2배, headless 에는 영향 없음)')
    parser.add_argument('--calm-person', action='store_true', help='사람 발버둥 끄기')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--timeout', type=float, default=240, help='headless 제한 시뮬레이션 시간 [초]')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed 는 양수여야 합니다.')
    mission, _, sky = build_weather(args.weather, args.tide, struggle=not args.calm_person)
    label = f'{args.weather}/{args.tide}'
    if args.headless:
        try:
            print(f'[{label}] ' + run_headless(mission, args.timeout))
        except RuntimeError as error:
            parser.exit(1, f'[{label}] {error}\n')
        return
    run_viewer(mission, sky, f'WOLMIDO / {label}', args.speed)


if __name__ == '__main__':
    main()
