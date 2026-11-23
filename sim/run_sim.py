"""TetherRescue 전체 시나리오 실행: python -m sim.run_sim

기본은 MuJoCo 창으로 시뮬레이션을 보여 줍니다. 로그만 보려면 --headless.

흐름:
  1. 배가 모래사장(지상)에 대기
  2. 익수자 발견 후 출항(물에 띄움)
  3. GPS는 생략하고 익수자 좌표를 자동으로 찍어 그쪽으로 항해
"""

import argparse
import math
import os
import time

import numpy as np

# WSL/WSLg에서 창이 안 뜨면 EGL 대신 glfw를 쓴다.
os.environ.setdefault('MUJOCO_GL', 'glfw')

from sim import config
from sim.world.sortie import RescueSortie, SortieSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true',
                        help='창 없이 로그만 출력 (화면 확인용 아님)')
    parser.add_argument('--timeout', type=float, default=400,
                        help='headless 제한 시뮬레이션 시간 [초]')
    parser.add_argument('--launch-delay', type=float, default=3.0,
                        help='지상 대기 시간 [초]')
    parser.add_argument('--victim-range', type=float, default=200.0,
                        help='해안에서 익수자까지 거리 [m]')
    parser.add_argument('--waves', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout은 유한한 양수여야 합니다.')
    waves = config.WaterParams.WAVES_DEFAULT if args.waves else config.WaterParams.WAVES_CALM
    settings = SortieSettings(launch_delay=args.launch_delay, victim_range=args.victim_range)
    sortie = RescueSortie(waves=waves, settings=settings)
    if args.headless:
        try:
            run_headless(sortie, args.timeout)
        except RuntimeError as error:
            parser.exit(1, f'{error}\n')
        return
    print('MuJoCo 창을 엽니다. 닫으면 종료됩니다. (로그만 보려면 --headless)')
    try:
        run_viewer(sortie)
    except Exception as error:
        parser.exit(1, f'화면을 열지 못했습니다: {error}\n'
                       f'DISPLAY={os.environ.get("DISPLAY")!r}, MUJOCO_GL={os.environ.get("MUJOCO_GL")!r}\n')


def run_headless(sortie, timeout):
    while sortie.world.data.time < timeout:
        sortie.step()
        world = sortie.world
        if world.data.warning.number.any() or not all(
                np.isfinite(s).all() for s in (world.data.qpos, world.data.qvel, world.data.qacc)):
            raise RuntimeError('전체 시나리오에 비정상 물리 상태가 발생했습니다.')
        if sortie.arrived:
            state = sortie.readings()
            print(f"도착: {state['time']:.1f}s, GPS 거리 {state['distance_to_gps']:.2f}m, "
                  f"익수자 ({state['victim_xy'][0]:.1f}, {state['victim_xy'][1]:.1f})")
            return
    state = sortie.readings()
    raise RuntimeError(f'{timeout:g}초 안에 익수자 GPS에 도착하지 못했습니다 '
                       f"(단계 {state['stage']}, GPS 거리 {state['distance_to_gps']}).")


def run_viewer(sortie):
    import mujoco
    import mujoco.viewer

    world = sortie.world
    boat0 = world.data.xpos[sortie.robot.body_id]
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.opt.geomgroup[config.Groups.VISUAL] = 1
        # 초기 시점만 잡고, 이후 휠·드래그는 사용자가 조작한다 (매 프레임 덮어쓰지 않음).
        viewer.cam.lookat[:] = boat0
        viewer.cam.distance = 35
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -25
        budget = 0.0
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                while budget < 1 / 60:
                    sortie.step()
                    budget += world.model.opt.timestep
                budget = max(0.0, budget - 1 / 60)
                state = sortie.readings()
                _draw_gps(viewer.user_scn, state)
                wave_mesh = sortie.water.update_surface_visual(world.model, world.data)
            if wave_mesh is not None:   # 파도 수면 메시를 화면에 올린다 (lock 밖)
                viewer.update_mesh(wave_mesh)
            gps = '—' if state['gps_xy'] is None else f"{state['gps_xy'][0]:.1f}, {state['gps_xy'][1]:.1f}"
            dist = '—' if state['distance_to_gps'] is None else f"{state['distance_to_gps']:.1f} m"
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'TETHER RESCUE\nStage\nTime\nGPS target\nDist to GPS\n'
                               'Boat x\nVictim x\n\nWheel: zoom\nDrag: orbit\nGreen: GPS mark',
                               f"\n{state['stage']}\n{state['time']:.1f} s\n{gps}\n{dist}\n"
                               f"{state['boat_xy'][0]:.1f} m\n{state['victim_xy'][0]:.1f} m")])
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - started)))


def _draw_gps(scene, state):
    import mujoco

    scene.ngeom = 0
    if state['gps_xy'] is None:
        return
    start = np.array([*state['gps_xy'], config.WaterParams.SURFACE_Z + .4])
    end = start + np.array([0, 0, 1.2])
    geom = scene.geoms[0]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), [.2, 1., .3, 1.])
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .08, start, end)
    scene.ngeom = 1


if __name__ == '__main__':
    main()
