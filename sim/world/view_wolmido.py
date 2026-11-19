"""월미도 해안 구조 시나리오: python -m sim.world.view_wolmido [--env calm|waves|low_tide] [--headless].

사람은 바다 한가운데에 구명환 없이 떠 있다. 구조 로봇은 구명환을 배 뒤 짧은 줄(1m)에 달고 해변 물가에서 출발해
사람 둘레를 돌아 선미를 사람 손 쪽으로 대고, 윈치로 줄을 3m까지 풀어 구명환을 건넨다. 사람이 손을 뻗어 잡으면
해변 쪽으로 최대 8m 끌고 와 멈춘다 (이후 사람 구조대 인계 가정). 구명환을 던지는 장치는 아직 없다(동경 담당 예정).
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


def flat_ring_quat(yaw):
    """물에 눕힌 구명환 자세: 구멍 축이 위(z), 로컬 +y(로프 연결점 쪽)가 수평 yaw 방향."""
    lie = np.array([math.cos(-math.pi / 4), 0., math.sin(-math.pi / 4), 0.])   # y축 −90°: 구멍 축 x → z
    turn = np.array([math.cos((yaw - math.pi / 2) / 2), 0., 0., math.sin((yaw - math.pi / 2) / 2)])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, turn, lie)
    return quat


def build_wolmido(env='calm', *, settings=None):
    """구명환을 배 뒤 짧은 줄에 달고 해변 물가에서 출발하는 구조 시나리오(WolmidoDelivery)와 맵.

    사람은 바다 한가운데(config.Wolmido.ENVIRONMENTS의 victim)에 구명환 없이 떠 있고, 왼손이 해변 쪽을 본다.
    """
    from sim.robot.robot import Robot
    from sim.water.water import Water
    from sim.world.grasp import HumanGrasp
    from sim.world.human import Human
    from sim.world.lifebuoy import Lifebuoy
    from sim.world.tether import Tether
    from sim.world.wolmido_mission import DeliverySettings, ReachAssist, WolmidoDelivery
    from sim.world.world import build_world

    setting = WolmidoMap.environment(env)
    settings = DeliverySettings() if settings is None else settings
    world_map = WolmidoMap(tide_drop=setting['tide_drop'])
    start, yaw, tow_heading, tow_distance = beach_start(world_map, setting['victim'])
    surface = config.WaterParams.SURFACE_Z
    draft = config.Hull.MASS / (config.Physics.RHO_WATER * config.Hull.LENGTH * config.Hull.BEAM)
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-forward[1], forward[0]])
    robot = Robot(position=(*start, surface + config.Hull.HEIGHT / 2 - draft), tow_point_b=config.Hull.TOW_POINT_B,
                  winch=True)
    # 구명환: 선미 줄 묶는 점 뒤로 0.9m(감은 줄 1m보다 짧아 처음엔 느슨함)에 로프 연결점이 오게 물에 눕힌다.
    stern_b = np.asarray(config.Hull.TOW_POINT_B, dtype=float)
    stern = start + forward * stern_b[0] + left * stern_b[1]
    site = stern - forward * .9
    ring_quat = flat_ring_quat(yaw)
    offset = np.empty(3)
    mujoco.mju_rotVecQuat(offset, np.asarray(config.Lifebuoy.TOW_POINT_B, dtype=float), ring_quat)
    ring = Lifebuoy(position=(site[0] - offset[0], site[1] - offset[1], surface), quat=ring_quat, grip_sites=True)
    # 사람: 왼손이 견인 방향(해변 쪽)을 보도록 돌린다.
    human = Human(position=(*setting['victim'], surface - .03), grip_sites=True, yaw=tow_heading - math.pi / 2)
    rope = Tether(enabled=True, attached=True)
    grasp = HumanGrasp()
    reach = ReachAssist(settings)
    water = Water(waves=getattr(config.WaterParams, setting['waves']),
                  floating_bodies=[config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY], surface_visual=True)
    world = build_world(parts=[water, robot, human, ring, rope, grasp, reach, world_map])
    address = world.model.jnt_qposadr[robot.joint_id]
    quaternion = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    world.model.qpos0[address + 3:address + 7] = quaternion
    world.data.qpos[address + 3:address + 7] = quaternion
    mujoco.mj_forward(world.model, world.data)
    mission = WolmidoDelivery(world, robot, rope, grasp, reach, tow_heading=tow_heading, tow_distance=tow_distance,
                              settings=settings)
    return mission, world_map


def boat_terrain_clearance(manager, world_map):
    """배 선체와 맵 지형 사이 최소 형상 거리 [m]. 겹치면 음수."""
    import mujoco

    model, data = manager.world.model, manager.world.data
    boat = [g for g in range(model.ngeom) if model.geom_bodyid[g] == manager.robot.body_id and model.geom_contype[g]]
    return min(mujoco.mj_geomDistance(model, data, a, b, 10., None)
               for a in boat for b in world_map.geom_ids.values())


def run_headless(mission, timeout):
    while mission.world.data.time < timeout and mission.stage != 'COMPLETE':
        if mission.step() is False:
            raise RuntimeError(f'STOPPED [{mission.stage}] {mission.failure}')
    if mission.stage != 'COMPLETE':
        raise RuntimeError(f'{timeout:g}초 안에 구조를 끝내지 못했습니다 (단계 {mission.stage}).')
    state = mission.readings()
    return (f"구조 완료 {state['time']:.1f}초: 잡은 시각 {mission.grab_time:.1f}초, 사람 진행 {state['tow_progress']:.2f}m, "
            f"장력 {state['tension']:.1f}N")


def run_viewer(mission, title):
    import time
    import mujoco.viewer
    from sim.world.rope_visual import draw_rope

    world = mission.world
    display = mujoco.MjData(world.model)
    mujoco.mj_copyData(display, world.model, world.data)
    with mujoco.viewer.launch_passive(world.model, display) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 22., 150., -32.
        budget = 0.
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                while budget < 1 / 60 and mission.stage != 'COMPLETE' and not mission.failed:
                    mission.step()
                    budget += world.model.opt.timestep
                budget = max(0., budget - 1 / 60)
                mujoco.mj_copyData(display, world.model, mission.display_data)
                viewer.user_scn.ngeom = 0
                draw_rope(viewer.user_scn, world.model, display, mission.rope)
                boat, person = display.xpos[mission.robot.body_id], display.xpos[mission.human_body]
                viewer.cam.lookat[:] = (boat + person) / 2
            state = mission.readings()
            labels = 'Stage\nTime\nWinch\nRope length\nHand-ring gap\nHolding ring\nTension\nPerson towed'
            values = (f"\n{state['stage']}\n{state['time']:.1f} s\n{state['delivery']}\n{state['rope_length']:.2f} m\n"
                      f"{state['hand_ring_gap']:.2f} m\n{state['held']}\n{state['tension']:.1f} N\n{state['tow_progress']:.2f} m")
            if mission.failed:
                labels, values = labels + '\nStop reason', values + f"\n{mission.failure}"
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               title + '\n' + labels, values)])
            viewer.sync()
            time.sleep(max(0., 1 / 60 - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', choices=tuple(config.Wolmido.ENVIRONMENTS), default='calm')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--timeout', type=float, default=300, help='headless 제한 시뮬레이션 시간 [초]')
    args = parser.parse_args()
    mission, world_map = build_wolmido(args.env)
    if args.headless:
        try:
            print(f'[{args.env}] ' + run_headless(mission, args.timeout))
        except RuntimeError as error:
            parser.exit(1, f'[{args.env}] {error}\n')
        return
    run_viewer(mission, f'WOLMIDO / {args.env}')


if __name__ == '__main__':
    main()
