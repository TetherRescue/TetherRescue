"""잔잔한 물에서 목표 위치·방향으로 접근 후 정지: python -m sim.view_approach."""

import argparse
import math
import time

import mujoco
import numpy as np

from sim import config
from sim.phase_manager import PhaseManager
from sim.robot import Robot
from sim.water import Water
from sim.world import build_world


class GoalMarker:
    REQUIRES = {}

    def __init__(self, target):
        self.target = target

    def build(self, spec):
        spec.worldbody.add_site(name=config.Names.APPROACH_TARGET_SITE,
                                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                pos=[*self.target, config.WaterParams.SURFACE_Z + .015],
                                size=[.2, .01, 0], rgba=[.2, 1., .3, .8])

    def bind(self, model):
        pass

    def add_forces(self, model, data, acc):
        pass


def build_approach_demo(*, start=(0., 0.), target=(4., 2.), yaw=0., target_heading=0.):
    """각도 인자는 rad. 목표는 선체 중심의 xy이며 도착 방향은 기본 월드 +x."""
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    if (start.shape != (2,) or target.shape != (2,) or not np.isfinite(start).all()
            or not np.isfinite(target).all() or not math.isfinite(yaw) or not math.isfinite(target_heading)):
        raise ValueError("출발·목표 좌표와 방향은 유한한 값이어야 합니다.")
    draft = config.Hull.MASS / (config.Physics.RHO_WATER * config.Hull.LENGTH * config.Hull.BEAM)
    height = config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - draft
    robot = Robot(position=(*start, height))
    world = build_world(parts=[Water(waves=(), surface_visual=True), robot, GoalMarker(target)])
    address = world.model.jnt_qposadr[robot.joint_id]
    world.data.qpos[address + 3:address + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(world.model, world.data)
    return PhaseManager(world, robot, target, target_heading=target_heading)


def decorate_goal(manager, scene):
    """목표의 도착 방향을 초록 화살표로 표시한다."""
    scene.ngeom = 0
    start = np.array([*manager.target_xy, config.WaterParams.SURFACE_Z + .35])
    end = start + .75 * np.array([math.cos(manager.target_heading), math.sin(manager.target_heading), 0])
    geom = scene.geoms[0]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                      np.eye(3).ravel(), [.2, 1., .3, 1.])
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .045, start, end)
    scene.ngeom = 1


def run_headless(manager, timeout):
    """도착 판정 후에도 8초를 더 실행해 정지 유지 여부를 확인한다."""
    world = manager.world
    first_arrival = None
    while world.data.time < timeout:
        manager.step()
        if world.data.warning.number.any() or not all(np.isfinite(state).all() for state in
                                                      (world.data.qpos, world.data.qvel, world.data.qacc)):
            raise RuntimeError("접근 시험 중 비정상 물리 상태가 발생했습니다.")
        if not manager.arrived:
            first_arrival = None
        elif first_arrival is None:
            first_arrival = float(world.data.time)
        if first_arrival is not None and world.data.time >= first_arrival + 8:
            if not manager.arrived:
                raise RuntimeError("도착 후 정지 유지 조건을 벗어났습니다.")
            reading = manager.readings()
            print(f"접근·정지 성공: {first_arrival:.2f}초에 도착, 8초 추가 유지, "
                  f"위치 오차 {reading['distance']:.3f}m, 속도 {reading['speed']:.3f}m/s, "
                  f"방향 오차 {math.degrees(reading['heading_error']):.2f}도")
            return
    raise RuntimeError(f"{timeout:g}초 안에 접근·정지 유지 시험을 완료하지 못했습니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--start', type=float, nargs=2, default=(0, 0), metavar=('X', 'Y'))
    parser.add_argument('--target', type=float, nargs=2, default=(4, 2), metavar=('X', 'Y'))
    parser.add_argument('--yaw', type=float, default=0, help='출발 방향 [도]')
    parser.add_argument('--heading', type=float, default=0, help='목표 도착 방향 [도]')
    parser.add_argument('--timeout', type=float, default=60, help='headless 제한 시뮬레이션 시간 [초]')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout은 유한한 양수여야 합니다.')
    manager = build_approach_demo(start=args.start, target=args.target,
                                 yaw=math.radians(args.yaw), target_heading=math.radians(args.heading))
    if args.headless:
        run_headless(manager, args.timeout)
        return
    import mujoco.viewer

    world = manager.world
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.cam.lookat[:] = [*(np.asarray(args.start) + np.asarray(args.target)) / 2, 0]
        viewer.cam.distance = max(8., np.linalg.norm(np.asarray(args.target) - np.asarray(args.start)) + 4)
        viewer.cam.azimuth = 100
        viewer.cam.elevation = -35
        budget = 0.
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                while budget < 1 / 60:
                    manager.step()
                    budget += world.model.opt.timestep
                budget -= 1 / 60
                decorate_goal(manager, viewer.user_scn)
            reading = manager.readings()
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'APPROACH AND STOP\nPhase\nTime\nPosition error\nSpeed\nHeading error\n'
                               '\nGreen: goal position and heading',
                               f"\n{reading['phase']}\n{world.data.time:.1f} s\n{reading['distance']:.3f} m\n"
                               f"{reading['speed']:.3f} m/s\n{math.degrees(reading['heading_error']):+.1f} deg")])
            viewer.sync()
            time.sleep(max(0., 1 / 60 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
