"""하나개 해수욕장 참고 맵: python -m sim.world.view_hanagae [--headless].

동쪽 끝에 육지를 붙이고, 해안에서 약 200m 떨어진 바다에 익수자를 둔다.
물은 sim.water.Water 부력·저항을 쓰고, 배는 sim.robot.Robot 모델을 쓴다.
"""

import argparse
import time

import numpy as np

from sim import config
from sim.world.coast_map import SHORE_X, VICTIM_RANGE, build_hanagae_world
from sim.world.world import advance_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--waves', action='store_true', help='기본 파도를 켠다 (기본은 잔잔한 물)')
    args = parser.parse_args()
    waves = config.WaterParams.WAVES_DEFAULT if args.waves else config.WaterParams.WAVES_CALM
    world, robot, _water, human = build_hanagae_world(waves=waves)
    if args.headless:
        run_headless(world, robot, human)
        return
    run_viewer(world, robot, human)


def run_headless(world, robot, human):
    for _ in range(1000):
        advance_world(world)
        if not all(np.isfinite(state).all() for state in (world.data.qpos, world.data.qvel, world.data.qacc)):
            raise RuntimeError('하나개 맵 시연에 비정상 물리 상태가 발생했습니다.')
    if world.data.warning.number.any():
        raise RuntimeError('하나개 맵 시연에서 MuJoCo 경고가 발생했습니다.')
    gap = float(abs(world.data.xpos[human.body_id, 0] - SHORE_X))
    print(f'하나개 맵: 1000스텝 정상, 해안↔익수자 {gap:.1f} m '
          f'(목표 {VICTIM_RANGE:g} m)')


def run_viewer(world, robot, human):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.opt.geomgroup[config.Groups.VISUAL] = 1
        mid = (world.data.xpos[robot.body_id] + world.data.xpos[human.body_id]) / 2
        viewer.cam.lookat[:] = mid
        viewer.cam.distance = 220
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30
        budget = 0.0
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                while budget < 1 / 60:
                    advance_world(world)
                    budget += world.model.opt.timestep
                budget = max(0.0, budget - 1 / 60)
            boat = world.data.xpos[robot.body_id]
            victim = world.data.xpos[human.body_id]
            shore_gap = abs(victim[0] - SHORE_X)
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'HANAGAE COAST MAP\nTime\nBoat x\nVictim x\nShore gap\n\n'
                               'Land on east edge\nVictim ~200 m offshore',
                               f'\n{world.data.time:.1f} s\n'
                               f'{boat[0]:.1f} m\n{victim[0]:.1f} m\n{shore_gap:.1f} m')])
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
