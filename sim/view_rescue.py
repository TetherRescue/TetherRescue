"""사람·구명환에 접근·로프 연결·견인 후 정지: python -m sim.view_rescue [--headless]."""

import argparse
from dataclasses import replace
import math
import time

import mujoco
import numpy as np

from sim import config
from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.rescue_approach import RescueApproach
from sim.rescue_mission import RescueMission, RescueMissionSettings
from sim.rope_visual import draw_rope
from sim.robot import Robot
from sim.tether import Tether
from sim.view_approach import decorate_goal
from sim.water import Water
from sim.world import build_world
from sim.watchdog import WatchdogSettings


def _build_rescue_world(*, start, victim, yaw, tow_heading, rope):
    """같은 부유 장면을 접근 단독·전체 시나리오에서 사용한다."""
    start, victim = np.asarray(start, dtype=float), np.asarray(victim, dtype=float)
    if (start.shape != (2,) or victim.shape != (2,) or not np.isfinite(start).all()
            or not np.isfinite(victim).all() or not math.isfinite(yaw) or not math.isfinite(tow_heading)):
        raise ValueError('출발·사람 위치와 방향은 유한한 값이어야 합니다.')
    draft = config.Hull.MASS / (config.Physics.RHO_WATER * config.Hull.LENGTH * config.Hull.BEAM)
    robot = Robot(position=(*start, config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - draft),
                  tow_point_b=(-config.Hull.LENGTH / 2, 0, -.08))
    human = Human(position=(*victim, config.WaterParams.SURFACE_Z - .03))
    water = Water(waves=(), floating_bodies=[config.Names.HUMAN_BODY], surface_visual=True)
    world = build_world(parts=[water, robot, human, Lifebuoy(), rope])
    address = world.model.jnt_qposadr[robot.joint_id]
    quaternion = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    world.model.qpos0[address + 3:address + 7] = quaternion
    world.data.qpos[address + 3:address + 7] = quaternion
    mujoco.mj_forward(world.model, world.data)
    return world, robot


def build_rescue_demo(*, start=(6., 3.), victim=(0., 0.), yaw=0., tow_heading=0.):
    """기존 접근 단독 API: 로프를 생성하지 않는다. 각도 rad."""
    world, robot = _build_rescue_world(start=start, victim=victim, yaw=yaw,
                                        tow_heading=tow_heading, rope=Tether(enabled=False))
    return RescueApproach(world, robot, tow_heading=tow_heading)


def build_rescue_mission(*, start=(6., 3.), victim=(0., 0.), yaw=0., tow_heading=0., tow_distance=6., watchdog_settings=None):
    """처음에는 미연결인 tendon을 생성한 전체 구조 시험. 각도 rad."""
    rope = Tether(enabled=True, attached=False)
    world, robot = _build_rescue_world(start=start, victim=victim, yaw=yaw,
                                       tow_heading=tow_heading, rope=rope)
    return RescueMission(world, robot, rope, tow_heading=tow_heading,
                         settings=RescueMissionSettings(tow_distance=tow_distance), watchdog_settings=watchdog_settings)


def run_headless(manager, timeout):
    if isinstance(manager, RescueMission):
        manager.watchdog.settings = replace(manager.watchdog.settings,
                                             mission_timeout=min(timeout, manager.watchdog.settings.mission_timeout))
    ready_since = None
    while not math.isfinite(manager.world.data.time) or manager.world.data.time < timeout:
        manager.step()
        if isinstance(manager, RescueMission) and manager.failed:
            failure = manager.failure
            raise RuntimeError(f'STOPPED: {failure.code.value} at {failure.time:.3f}s '
                               f'[{failure.stage}] - {failure.message}')
        world = manager.world
        if world.data.warning.number.any() or not all(np.isfinite(state).all() for state in
                                                      (world.data.qpos, world.data.qvel, world.data.qacc)):
            raise RuntimeError('구조 접근 시험에 비정상 물리 상태가 발생했습니다.')
        if not manager.arrived:
            ready_since = None
        elif ready_since is None:
            ready_since = float(world.data.time)
        if ready_since is not None and world.data.time >= ready_since + 5:
            state = manager.readings()
            if isinstance(manager, RescueMission):
                print(f"접근·연결·견인 완료: {ready_since:.2f}초, 5초 추가 정지 유지, "
                      f"사람 진행 {state['tow_progress']:.3f}m, 장력 {state['tension']:.2f}N, "
                      f"최소 표면 간격 {manager.minimum_observed_clearance:.3f}m")
            else:
                print(f"연결 대기 성공: {ready_since:.2f}초, 5초 추가 유지, "
                      f"연결점 간격 {state['endpoint_gap']:.3f}m, 상대속도 {state['relative_speed']:.3f}m/s, "
                      f"최소 표면 간격 {manager.minimum_observed_clearance:.3f}m (로프 미연결)")
            return
    raise RuntimeError(f'{timeout:g}초 안에 시나리오와 추가 정지 유지를 완료하지 못했습니다.')


def decorate_rescue(manager, scene):
    decorate_goal(manager, scene)
    if not isinstance(manager, RescueMission):
        return
    draw_rope(scene, manager.world.model, manager.display_data, manager.rope)


def apply_failure_demo(manager, kind):
    """사용자가 명시적으로 요청한 중단 시연에만 상태·제한 시간을 변경한다."""
    if not isinstance(manager, RescueMission):
        raise ValueError('실패 시연은 전체 시나리오에서만 지원합니다.')
    if kind == 'timeout':
        manager.watchdog.settings = replace(manager.watchdog.settings, approach_timeout=1.)
    elif kind == 'capsize':
        address = manager.world.model.jnt_qposadr[manager.robot.joint_id]
        angle = math.radians(80)
        manager.world.data.qpos[address + 3:address + 7] = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]
    elif kind == 'nonfinite':
        manager.world.data.qpos[0] = float('nan')
    elif kind == 'warning':
        manager.world.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number += 1
    else:
        raise ValueError('알 수 없는 실패 시연입니다.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--failure-demo', choices=('timeout', 'capsize', 'nonfinite', 'warning'),
                        help='Watchdog 중단 동작을 확인하는 오류 주입 시연')
    parser.add_argument('--approach-only', action='store_true', help='기존 로프 없는 연결 대기 시험')
    parser.add_argument('--tow-distance', type=float, default=6., help='연결 후 배의 이동 거리 [m], 4m 초과')
    parser.add_argument('--start', type=float, nargs=2, default=(6, 3), metavar=('X', 'Y'))
    parser.add_argument('--victim', type=float, nargs=2, default=(0, 0), metavar=('X', 'Y'))
    parser.add_argument('--yaw', type=float, default=0, help='배의 출발 방향 [도]')
    parser.add_argument('--heading', type=float, default=0, help='연결 대기 시 견인 방향 [도]')
    parser.add_argument('--timeout', type=float, default=240, help='headless 제한 시뮬레이션 시간 [초]')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout은 유한한 양수여야 합니다.')
    builder = build_rescue_demo if args.approach_only else build_rescue_mission
    options = {} if args.approach_only else dict(tow_distance=args.tow_distance)
    manager = builder(start=args.start, victim=args.victim, yaw=math.radians(args.yaw),
                      tow_heading=math.radians(args.heading), **options)
    if args.failure_demo:
        if args.approach_only:
            parser.error('--failure-demo는 --approach-only와 함께 사용할 수 없습니다.')
        apply_failure_demo(manager, args.failure_demo)
    if args.headless:
        try:
            run_headless(manager, args.timeout)
        except RuntimeError as error:
            parser.exit(1, f'{error}\n')
        return
    import mujoco.viewer

    world = manager.world
    if isinstance(manager, RescueMission):
        manager.step()  # 초기 오류 주입도 렌더러에 전달하기 전에 검사한다.
    display = mujoco.MjData(world.model)
    mujoco.mj_copyData(display, world.model, manager.display_data if isinstance(manager, RescueMission) else world.data)
    with mujoco.viewer.launch_passive(world.model, display) as viewer:
        viewer.cam.lookat[:] = [*(np.asarray(args.start) + np.asarray(args.victim)) / 2, 0]
        viewer.cam.distance = max(10., np.linalg.norm(np.asarray(args.start) - np.asarray(args.victim)) + 4)
        viewer.cam.azimuth = 100
        viewer.cam.elevation = -35
        budget = 0.
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                if not (isinstance(manager, RescueMission) and manager.failed):
                    while budget < 1 / 60:
                        if manager.step() is False:
                            break
                        budget += world.model.opt.timestep
                    budget = max(0., budget - 1 / 60)
                source = manager.display_data if isinstance(manager, RescueMission) else world.data
                mujoco.mj_copyData(display, world.model, source)
                decorate_rescue(manager, viewer.user_scn)
                viewer.cam.lookat[:] = (display.xpos[manager.robot.body_id]
                                       + display.site_xpos[manager.ring_site]) / 2
            state = manager.readings()
            mission = isinstance(manager, RescueMission)
            extra_labels = '\nRope connected\nTension\nPerson progress' if mission else ''
            extra_values = (f"\n{state['attached']}\n{state['tension']:.1f} N\n{state['tow_progress']:.2f} m"
                            if mission else '')
            failure_label = '\nStop reason' if mission and manager.failed else ''
            failure_value = (f"\n{manager.failure.code.value} / {manager.failure.stage} @ {manager.failure.time:.2f}s"
                             if mission and manager.failed else '')
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'RESCUE SIMULATION\nPhase\nTime\nEndpoint gap\n'
                               'Relative speed\nSurface clearance' + extra_labels + failure_label + '\n\nGreen: current boat target',
                               f"\n{state['phase']}\n{display.time:.1f} s\n{state['endpoint_gap']:.3f} m\n"
                               f"{state['relative_speed']:.3f} m/s\n{state['clearance']:.3f} m" + extra_values + failure_value)])
            viewer.sync()
            time.sleep(max(0., 1 / 60 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
