"""사람·구명환 고정 연결의 자유낙하 데모: python -m sim.view_human.

화면을 위한 독립 실행 예제다. 실제 구조 단계 제어는 PhaseManager의 담당이다.
"""

import argparse
import time

import mujoco
import numpy as np

from sim import config

from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.world import advance_world, build_world


def advance(world):
    # 바닥·부력이 없는 데모: 화면 밖으로 떨어지면 처음 위치에서 반복한다.
    if world.data.time >= 1.0:
        mujoco.mj_resetData(world.model, world.data)
    advance_world(world)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="창 없이 데모 실행을 검사")
    parser.add_argument("--human-only", action="store_true", help="구명환 없이 사람만 표시")
    args = parser.parse_args()
    parts = [Human()] if args.human_only else [Human(), Lifebuoy(attached=True)]
    world = build_world(parts=parts)
    if args.headless:
        run_headless(world)
        return
    run_viewer(world)


def run_headless(world):
    for _ in range(1000):
        advance(world)
        if not all(np.isfinite(state).all() for state in (world.data.qpos, world.data.qvel)):
            raise RuntimeError("데모의 위치·속도에 비정상 값이 발생했습니다.")
    print("데모: 1000스텝 정상 실행 (1초마다 초기 위치로 복귀)")


def run_viewer(world, *, lookat=(0, 0, -1.5), distance=8):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = distance
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -15
        # 렌더링 주기 약 60Hz, 물리 연산은 계약의 timestep(현재 500Hz)을 유지한다.
        target_time = 0.0
        while viewer.is_running():
            started = time.monotonic()
            while target_time < 1 / 60:
                advance(world)
                target_time += world.model.opt.timestep
            target_time -= 1 / 60
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
