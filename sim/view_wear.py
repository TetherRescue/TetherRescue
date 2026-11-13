"""양팔을 곧게 뻗어 구명환에 동시에 넣는 동작 시험: python -m sim.view_wear."""

import argparse
import math
import time

import mujoco
import numpy as np

from sim.rope_visual import draw_rope
from sim.scenarios import WearDemo


def follow_camera(demo, camera):
    rotation = demo.world.data.xmat[demo.human.body_id].reshape(3, 3)
    direction = rotation @ np.array([1., 1., 1.6])
    camera.lookat[:] = demo.world.data.xpos[demo.human.body_id]
    camera.azimuth = 180 + math.degrees(math.atan2(direction[1], direction[0]))
    camera.elevation = -math.degrees(math.atan2(direction[2], np.linalg.norm(direction[:2])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--free-camera', action='store_true', help='사람 정면을 따라가는 자동 시점 회전 끄기')
    args = parser.parse_args()
    demo = WearDemo()
    if args.headless:
        for _ in range(25000):
            demo.step()
            if demo.wear.phase == 'WORN':
                print('착용 완료', demo.world.data.time, demo.wear.measurements(demo.world.data))
                return
        raise RuntimeError('50초 안에 착용 동작을 마치지 못했습니다.')
    import mujoco.viewer
    world = demo.world
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.cam.lookat[:] = (0, .2, 1)
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -40
        budget = 0.
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                while budget < 1 / 60:
                    demo.step()
                    budget += world.model.opt.timestep
                budget -= 1 / 60
                draw_rope(viewer.user_scn, world.model, world.data, demo.rope, clear=True)
                viewer.cam.lookat[:] = world.data.xpos[demo.human.body_id]
                if not args.free_camera:
                    follow_camera(demo, viewer.cam)
            state = demo.wear.measurements(world.data)
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'SELF DONNING - JOINT CONTROL\nPhase\nRing / torso error\nReady to tow\n'
                               'Gravity OFF | Guided approach | Rope connected',
                               f"\n{state['phase']}\n{state['position_error']:.3f} m\n{state['ready_to_tow']}")])
            viewer.sync()
            time.sleep(max(0., 1 / 60 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
