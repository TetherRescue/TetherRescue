"""배 → 로프 → 구명환·사람 수상 견인 시험. --dry로 기존 무중력 시험을 선택한다.

python -m sim.view_towing [--headless]
"""

import argparse
import time

import mujoco
import numpy as np

from sim import config
from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.robot import Robot
from sim.tether import Tether
from sim.world import advance_world, build_world


def build_towing_world():
    # 연결점·배치는 시험용 명시 인자이며 공유 config의 합의값이 아니다.
    # 부력이 없어 추진기와 COM 높이 차이에 의한 pitch도 발생한다.
    robot = Robot(position=(4, 0, 1), tow_point_b=(-config.Hull.LENGTH / 2, 0, 0))
    human = Human(position=(0, -.325, 1))
    rope = Tether(enabled=True)
    world = build_world(parts=[robot, human, Lifebuoy(attached=True), rope], gravity=0)
    robot.set_command(forward=.2)
    return world, robot, human, rope


def build_surface_towing_world():
    """잔잔한 물에서 중력·부력·저항·추력·로프를 함께 적용한다."""
    from sim.water import Water

    height = config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - config.Expect.DRAFT
    # 초기 줄은 느슨하다. 배가 이동해 줄을 팽팽하게 만드는 과정부터 관찰한다.
    ring_end_x = config.Lifebuoy.POSITION_IN_HUMAN_B[0] + config.Lifebuoy.TOW_POINT_B[0]
    boat_x = ring_end_x + .9 * config.Tether.REST_LENGTH + config.Hull.LENGTH / 2
    robot = Robot(position=(boat_x, 0, height),
                  tow_point_b=config.Hull.TOW_POINT_B)
    human = Human(position=(0, -config.Lifebuoy.TOW_POINT_B[1], -.03 + config.WaterParams.SURFACE_Z))
    water = Water(waves=config.WaterParams.WAVES_CALM,
                  floating_bodies=[config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY], surface_visual=True)
    rope = Tether(enabled=True)
    world = build_world(parts=[water, robot, human, Lifebuoy(attached=True), rope])
    robot.set_command(forward=.2)
    return world, robot, human, rope


def advance(world):
    """리셋 없이 현재 상태에서 한 스텝 실행한다."""
    advance_world(world)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--dry', action='store_true', help='기존 중력·물 없는 힘 전달 시험')
    args = parser.parse_args()
    world, robot, human, rope = build_towing_world() if args.dry else build_surface_towing_world()
    steps = 1000 if args.dry else 5000
    if args.headless:
        initial_x = world.data.xpos[human.body_id, 0]
        for _ in range(steps):
            advance(world)
            if not all(np.isfinite(state).all() for state in
                       (world.data.qpos, world.data.qvel, world.data.qacc)):
                raise RuntimeError('견인 시험에서 비정상 상태가 발생했습니다.')
        if world.data.warning.number.any():
            raise RuntimeError('견인 시험에서 MuJoCo 경고가 발생했습니다.')
        print(f'통합 견인 {steps}스텝: 사람 이동 {world.data.xpos[human.body_id, 0] - initial_x:.3f} m, '
              f'장력 {rope.tension(world.model, world.data):.1f} N '
              f'({"중력·부력·저항 없음" if args.dry else "중력·부력·물 저항 적용"})')
        return
    import mujoco.viewer

    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.cam.lookat[:] = (2, 0, 1 if args.dry else 0)
        viewer.cam.distance = 9
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
        while viewer.is_running():
            started = time.monotonic()
            # 수상 시험은 연속 실행한다. 기존 무중력 시험만 2초 구간을 반복한다.
            with viewer.lock():
                if args.dry and world.data.time >= 2:
                    mujoco.mj_resetData(world.model, world.data)
                for _ in range(round(1 / (60 * world.model.opt.timestep))):
                    advance(world)
                if not args.dry:
                    viewer.cam.lookat[:] = (world.data.xpos[robot.body_id] + world.data.xpos[human.body_id]) / 2
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               ('DRY TOWING TEST\nGravity / buoyancy / drag OFF\nRepeats every 2 seconds' if args.dry
                                else 'SURFACE TOWING TEST\nGravity / buoyancy / water drag ON\nContinuous simulation'),
                               f'Tension: {rope.tension(world.model, world.data):.1f} N')])
            viewer.sync()
            time.sleep(max(0, 1 / 60 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
