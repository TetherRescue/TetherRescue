"""구명환 발사 구조 시연: python -m sim.world.view_launch [--headless] [--speed 3].

접근 → 선미를 사람 쪽으로 정렬 → 구명환 발사 → 사람이 잡기 → 견인 → 정지.
"""
import argparse
import math
import time

import mujoco
import numpy as np

from sim import config
from sim.world.launch_mission import build_launch_mission
from sim.world.rope_visual import draw_rope
from sim.world.view_approach import decorate_goal


def run_headless(mission, timeout):
    data = mission.world.data
    while data.time < timeout:
        if mission.step() is False:
            failure = mission.failure
            raise RuntimeError(f'STOPPED: {failure.code.value} at {failure.time:.3f}s '
                               f'[{failure.stage}] - {failure.message}')
        if mission.arrived:
            state = mission.readings()
            shot, landing = state['shot'], state['landing']
            print(f"발사 {shot['time']:.2f}초 (거리 {shot['distance']:.2f} m, 속력 {shot['speed']:.2f} m/s) → "
                  f"착수 잡는 점-손 {landing['grip_to_hand']:.2f} m, 착수 속력 {landing['impact_speed']:.1f} m/s → "
                  f"잡음 {mission.grab_time:.2f}초 → 견인 완료 {data.time:.2f}초, 사람 진행 {state['tow_progress']:.2f} m, "
                  f"최소 표면 간격 {mission.minimum_observed_clearance:.2f} m")
            return
    raise RuntimeError(f'{timeout:g}초 안에 시나리오를 완료하지 못했습니다. ({mission.readings()["phase"]})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--speed', type=float, default=1., help='화면 재생 배속 (headless 에는 영향 없음)')
    # 기본 = 사람에게서 50 m 떨어져 사람을 보고 출발 (2026-11-19 동경: 최소 50 m 에서 다가가는 장면)
    parser.add_argument('--start', type=float, nargs=2, default=(50, 0), metavar=('X', 'Y'))
    parser.add_argument('--victim', type=float, nargs=2, default=(0, 0), metavar=('X', 'Y'))
    parser.add_argument('--yaw', type=float, default=180, help='배의 출발 방향 [도]')
    parser.add_argument('--heading', type=float, default=0, help='견인 방향 [도]')
    parser.add_argument('--timeout', type=float, default=200, help='headless 제한 시뮬레이션 시간 [초]')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed 는 양수여야 합니다.')
    mission = build_launch_mission(start=args.start, victim=args.victim, yaw=math.radians(args.yaw),
                                   tow_heading=math.radians(args.heading))
    if args.headless:
        try:
            run_headless(mission, args.timeout)
        except RuntimeError as error:
            parser.exit(1, f'{error}\n')
        return
    import mujoco.viewer

    world = mission.world
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.cam.distance = 9.
        viewer.cam.azimuth = 100
        viewer.cam.elevation = -35
        frame = 1 / 60
        budget = 0.
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                if not mission.failed and not mission.arrived:
                    # 계산이 배속을 못 따라가면 밀린 시간을 버린다 (화면이 멈춘 듯 쌓이지 않게)
                    budget = min(budget + frame * args.speed, 2 * frame * args.speed)
                    while budget >= world.model.opt.timestep:
                        if mission.step() is False:
                            break
                        budget -= world.model.opt.timestep
                decorate_goal(mission, viewer.user_scn)
                draw_rope(viewer.user_scn, world.model, world.data, mission.rope)
                boat, person = world.data.xpos[mission.robot.body_id], world.data.xpos[mission.human_body]
                viewer.cam.lookat[:] = (boat + person) / 2
                viewer.cam.distance = max(9., .9 * float(np.linalg.norm(boat[:2] - person[:2])) + 5.)   # 둘 다 보이게
            state = mission.readings()
            landing = state['landing']
            status = (f"{mission.failure.code.value}" if mission.failed else
                      'COMPLETE' if mission.arrived else state['phase'])
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               f'LIFEBUOY LAUNCH (x{args.speed:g})\nPhase\nTime\nLauncher\nGrip-hand gap\n'
                               'Landing miss\nHeld\nTension\nPerson progress\nUpside down',
                               f"\n{status}\n{world.data.time:.1f} s\n{state['launcher']}\n{state['hand_gap']:.2f} m\n"
                               + (f"{landing['grip_to_hand']:.2f} m" if landing else '-')
                               + f"\n{state['held']}\n{state['tension']:.1f} N\n{state['tow_progress']:.2f} m"
                               f"\n{state['inverted']}")])
            viewer.sync()
            time.sleep(max(0., frame - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
