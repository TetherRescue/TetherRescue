"""사람·로프 연결 독립 시험: python -m sim.view_grasp [--headless] [--water].

배의 탑재·전달 동작은 구현하지 않는다. 시험용 이동 연결점과 전달 완료 신호만 입력한다.
"""

import argparse
import time

import mujoco
import numpy as np

from sim import config

from sim.frames import body_velocity_w
from sim.rope_visual import draw_rope
from sim.scenarios import GraspDemo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-grab', action='store_true', help='전달 신호 없이 구명환만 당기는 비교 시험')
    parser.add_argument('--water', action='store_true', help='중력·부력·저항을 켠 물리 안정성 시험')
    args = parser.parse_args()
    demo = GraspDemo(water=args.water, receive=not args.no_grab)
    if args.headless:
        maximum_tension = 0.
        held_progress = 0.
        released_velocity = None
        for _ in range(6000):
            previous = demo.phase
            demo.step()
            maximum_tension = max(maximum_tension, demo.rope.tension(demo.world.model, demo.world.data))
            if previous == 'HELD' and demo.phase == 'RELEASED':
                held_progress = float(demo.world.data.xpos[demo.human.body_id, 0] - demo._human_start[0])
                released_velocity = body_velocity_w(demo.world.model, demo.world.data, demo.human.body_id)[0]
        if not args.no_grab:
            if demo.grab_time is None or demo.release_time is None or held_progress < .05 or maximum_tension < 1:
                raise RuntimeError('잡기·견인·놓기 비교 조건을 만족하지 못했습니다.')
            if not args.water:
                velocity = body_velocity_w(demo.world.model, demo.world.data, demo.human.body_id)[0]
                # 회전하는 자유 강체의 4초 수치 적분 오차(약 수 um/s)를 허용한다.
                if not np.allclose(velocity, released_velocity, rtol=0, atol=1e-5):
                    raise RuntimeError('놓은 뒤 사람의 관성 운동이 유지되지 않았습니다.')
            print(f'잡기 독립 시험: {demo.grab_time:.2f}s 잡기 → {demo.release_time:.2f}s 놓기, '
                  f'잡은 동안 사람 진행 {held_progress:.3f}m, 최대 장력 {maximum_tension:.1f}N '
                  f'({"중력·부력·저항 ON" if args.water else "중력·물 OFF"})')
        else:
            movement = demo.readings()['human_movement']
            if not args.water and movement > 1e-8:
                raise RuntimeError('미수령 상태에서 사람이 로프에 끌려갔습니다.')
            print(f'미수령 비교: 잡기 연결 없음, 사람 이동 {movement:.6f}m, 최대 장력 {maximum_tension:.1f}N')
        return
    import mujoco.viewer

    world = demo.world
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.cam.lookat[:] = (1, .5, -.1 if args.water else 1.)
        viewer.cam.distance = 6.5
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -40
        budget = 0.
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                if world.data.time >= 12:
                    demo.reset()  # 비교용 반복 리셋이며 실제 전달 중 위치 변경은 아니다.
                while budget < 1 / 60:
                    demo.step()
                    budget += world.model.opt.timestep
                budget -= 1 / 60
                draw_rope(viewer.user_scn, world.model, world.data, demo.rope, clear=True)
                human_position = world.data.xpos[demo.human.body_id]
                fixture_position = world.data.xpos[demo.fixture.body_id]
                viewer.cam.lookat[:] = (human_position + fixture_position) / 2
                viewer.cam.distance = max(6.5, np.linalg.norm(fixture_position - human_position) + 3.)
            state = demo.readings()
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'HANDOVER TEST - NO BOAT\nPhase\nHand / ring gap\nRelative speed\n'
                               'Rope tension\nPerson movement\n\nBlue: pull fixture | Green: grip points\n'
                               + ('Gravity / buoyancy / drag ON' if args.water else 'Gravity / water OFF'),
                               f"\n{state['phase']}\n{state['distance']:.3f} m\n{state['relative_speed']:.3f} m/s\n"
                               f"{state['tension']:.1f} N\n{state['human_movement']:.3f} m")])
            viewer.sync()
            time.sleep(max(0., 1 / 60 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
