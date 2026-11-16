"""배가 물에 뜬 사람·구명환에 부딪히는 충돌 시험: python -m sim.view_contact [--side feet|side|head] [--push].

기본은 배가 1 m/s로 달려와 부딪히는 장면이고, --push는 정지 상태에서 최대 추력으로 계속 미는 장면이다.
장면은 sim/scenarios.py의 ContactDemo이며 tests/test_contact.py와 같다. 끝나면 잠시 멈췄다가 반복한다.
현재는 구명환을 이미 착용한 사람을 쓴다 (사람 모델 부피가 작아 구명환 없이는 가라앉음 — ContactDemo 설명 참고).
"""

import argparse
import math
import time

import mujoco
import numpy as np

from sim import config
from sim.scenarios import ContactDemo

HOLD_SECONDS = 1.5   # 한 번 끝난 뒤 결과를 보여 주며 멈추는 시간 (화면 전용)


def build_demo(args):
    speed, thrust = (0., 1.) if args.push else (args.speed, 0.)
    return ContactDemo(ContactDemo.HEADINGS[args.side], speed=speed, thrust=thrust,
                       gap=args.gap, surface_visual=not args.headless)


def describe(args):
    if args.push:
        return f'Full-thrust push from {args.side}'
    return f'Impact {args.speed:.1f} m/s from {args.side}'


def run_headless(demo, seconds):
    for _ in range(round(seconds / demo.world.model.opt.timestep)):
        demo.step()
    state = demo.readings()
    limit = f" (탄성 한계 {state['elastic_limit']:.3f})" if state['elastic_limit'] > 0 else ''
    print(f"충돌 {state['hit']}, 최대 겹침 {state['max_penetration'] * 1000:.1f} mm, "
          f"사람 속도 변화 {state['human_dv']:.3f} m/s{limit}, "
          f"사람 밀린 거리 {state['push']:.3f} m, 경고 {state['warnings']}")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=tuple(ContactDemo.HEADINGS), default='feet',
                        help='배가 부딪히는 쪽: feet(발), side(옆·구명환), head(머리)')
    parser.add_argument('--speed', type=float, default=1.0, help='충돌 속도 [m/s] (--push 아닐 때)')
    parser.add_argument('--push', action='store_true', help='정지 상태에서 최대 추력으로 계속 밀기')
    parser.add_argument('--gap', type=float, default=0.15, help='출발 시 최소 형상 간격 [m]')
    parser.add_argument('--seconds', type=float, default=None, help='한 번 실행 시간 [초] (기본: 충돌 3초, 밀기 6초)')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    seconds = args.seconds or (6. if args.push else 3.)
    if not math.isfinite(seconds) or seconds <= 0:
        parser.error('--seconds는 유한한 양수여야 합니다.')

    print('사람이 물에 자리 잡는 중... (약 5초 계산)')
    demo = build_demo(args)
    if args.headless:
        state = run_headless(demo, seconds)
        if state['warnings'] or not state['hit']:
            parser.exit(1, '충돌 시험이 비정상입니다.\n')
        return

    import mujoco.viewer

    world = demo.world
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        world.model.vis.map.force = 0.003          # 접촉력 화살표 길이 [m/N] (표시 전용)
        world.model.vis.scale.contactwidth = 0.05
        world.model.vis.scale.contactheight = 0.02
        side = np.array([-demo.direction[1], demo.direction[0]])
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = math.degrees(math.atan2(-side[1], -side[0]))   # 진행 방향 옆에서 본다
        viewer.cam.elevation = -30
        budget, finished_at = 0., None
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                if finished_at is None:
                    while budget < 1 / 60:
                        demo.step()
                        budget += world.model.opt.timestep
                    budget -= 1 / 60
                    if world.data.time >= seconds:
                        finished_at = time.monotonic()
                elif time.monotonic() - finished_at >= HOLD_SECONDS:
                    demo.reset()
                    budget, finished_at = 0., None
                center = (world.data.xpos[demo.robot.body_id] + world.data.xipos[demo.human.body_id]) / 2
                viewer.cam.lookat[:] = (center[0], center[1], 0.)
            state = demo.readings()
            limit = f" / {state['elastic_limit']:.2f}" if not args.push else ''
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'BOAT-HUMAN CONTACT TEST (ring already worn)\nMode\nTime\nBoat contacts\nOverlap now / max\n'
                               'Human dv' + (' / elastic limit' if not args.push else '') +
                               '\nHuman pushed\nBoat speed\n\nRed: contact force' +
                               ('\nFinished - restarting' if finished_at else ''),
                               f"\n{describe(args)}\n{state['time']:.2f} s\n{state['contacts']}\n"
                               f"{state['penetration'] * 1000:.1f} / {state['max_penetration'] * 1000:.1f} mm\n"
                               f"{state['human_dv']:.2f}{limit} m/s\n{state['push']:.2f} m\n"
                               f"{state['boat_speed']:.2f} m/s")])
            viewer.sync()
            time.sleep(max(0., 1 / 60 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
