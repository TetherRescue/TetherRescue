"""월미도 해안 구조 시나리오: python -m sim.world.view_wolmido [--env calm|waves|low_tide] [--speed 2] [--headless].

사람은 바다 한가운데에 구명환 없이 떠 있다. 구조 로봇(동경 A형 선체)은 구명환을 발사대에 싣고 해변 물가에서 출발해
사람 둘레를 돌아 선미를 사람 쪽으로 두고 멈춘 뒤 구명환을 쏜다(동경 Launcher·LaunchMission). 사람이 잡으면
해변 쪽으로 끌고 와 물가 10m 앞에서 멈춘다. 화면은 기본 2배속으로 보여 준다 (--speed).
환경 3개는 같은 맵에서 조건만 다르다 (config.Wolmido.ENVIRONMENTS):
  calm      만조·잔잔한 물
  waves     만조·파도(물 담당 기본 파도)
  low_tide  간조·잔잔한 물. 해변이 넓게 드러나 물가가 사람 쪽으로 다가오고, 방파제 앞 바위가 수면 위로 나온다
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


def build_wolmido(env='calm', *, launch=None, watchdog_settings=None):
    """월미도 맵·환경에 동경 구명환 발사 시나리오(LaunchMission)를 조립한다. (LaunchMission, WolmidoMap)을 돌려준다.

    배는 해변 물가에서 사람 쪽을 보고 출발하고, 구명환은 배 발사대에 실려 있다. 사람은 바다 한가운데에 구명환 없이 떠 있다.
    조립 순서·부품은 동경 build_launch_mission과 같고, 맵(WolmidoMap)과 환경의 파도만 더한다 (그 파일은 바꾸지 않음).
    """
    from sim.robot.launcher import Launcher
    from sim.robot.robot import Robot
    from sim.water.water import Water
    from sim.world.grasp import GraspSettings, HumanGrasp
    from sim.world.human import Human
    from sim.world.launch_mission import LaunchMission, LaunchSettings
    from sim.world.lifebuoy import Lifebuoy
    from sim.world.rescue_mission import RescueMissionSettings
    from sim.world.tether import Tether
    from sim.world.world import build_world

    setting = WolmidoMap.environment(env)
    launch = LaunchSettings() if launch is None else launch
    world_map = WolmidoMap(tide_drop=setting['tide_drop'])
    start, yaw, tow_heading, tow_distance = beach_start(world_map, setting['victim'])
    surface = config.WaterParams.SURFACE_Z
    robot = Robot(position=(*start, surface + config.Hull.HEIGHT / 2 - config.Expect.DRAFT), yaw=yaw, winch=True)
    human = Human(position=(*setting['victim'], surface - .03), grip_sites=True)
    water = Water(waves=getattr(config.WaterParams, setting['waves']),
                  floating_bodies=[config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY], surface_visual=True)
    ring = Lifebuoy(grip_sites=True)
    rope = Tether(enabled=True, attached=True)
    grasp = HumanGrasp(settings=GraspSettings(max_distance=launch.reach, max_relative_speed=launch.grab_relative_speed))
    launcher = Launcher(ring_site=grasp.ring_name)
    world = build_world(parts=[water, robot, human, ring, rope, grasp, launcher, world_map])
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


def run_viewer(mission, title, speed=1.):
    import time
    import mujoco.viewer
    from sim.world.rope_visual import draw_rope

    world = mission.world
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 16., 150., -32.
        frame, budget = 1 / 60, 0.
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                if not mission.failed and not mission.arrived:
                    # 계산이 배속을 못 따라가면 밀린 시간은 버린다 (동경 view_launch와 같은 방식)
                    budget = min(budget + frame * speed, 2 * frame * speed)
                    while budget >= world.model.opt.timestep:
                        if mission.step() is False:
                            break
                        budget -= world.model.opt.timestep
                draw_rope(viewer.user_scn, world.model, world.data, mission.rope)
                viewer.cam.lookat[:] = (world.data.xpos[mission.robot.body_id] + world.data.xpos[mission.human_body]) / 2
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
    parser.add_argument('--env', choices=tuple(config.Wolmido.ENVIRONMENTS), default='calm')
    parser.add_argument('--speed', type=float, default=2., help='화면 재생 배속 (기본 2배, headless 에는 영향 없음)')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--timeout', type=float, default=240, help='headless 제한 시뮬레이션 시간 [초]')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed 는 양수여야 합니다.')
    mission, world_map = build_wolmido(args.env)
    if args.headless:
        try:
            print(f'[{args.env}] ' + run_headless(mission, args.timeout))
        except RuntimeError as error:
            parser.exit(1, f'[{args.env}] {error}\n')
        return
    run_viewer(mission, f'WOLMIDO / {args.env}', args.speed)


if __name__ == '__main__':
    main()
