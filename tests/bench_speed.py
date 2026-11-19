"""수상 견인 조립의 화면 없는 물리 스텝 속도 측정 (담당: 채연).

저장소 루트에서 python tests/bench_speed.py 로 실행한다.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from sim import config
from sim.world.view_towing import build_surface_towing_world
from sim.world.world import advance_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=100)
    args = parser.parse_args()
    if args.steps <= 0 or args.repeat <= 0 or args.warmup < 0:
        parser.error('steps와 repeat는 양수, warmup은 0 이상의 정수여야 합니다.')

    rates = []
    durations = []
    for _ in range(args.repeat):
        world, _, _, _ = build_surface_towing_world()
        for _ in range(args.warmup):
            advance_world(world)
        started = time.perf_counter()
        for _ in range(args.steps):
            advance_world(world)
        elapsed = time.perf_counter() - started
        if not all(np.isfinite(state).all() for state in
                   (world.data.qpos, world.data.qvel, world.data.qacc)):
            raise RuntimeError('측정 종료 상태에 NaN 또는 무한대가 있습니다.')
        if world.data.warning.number.any():
            raise RuntimeError('측정 중 MuJoCo 경고가 발생했습니다.')
        durations.append(elapsed)
        rates.append(args.steps / elapsed)

    median_rate = statistics.median(rates)
    print(json.dumps(dict(
        scenario='calm_surface_towing', python=platform.python_version(),
        platform=platform.platform(), mujoco=mujoco.__version__, numpy=np.__version__,
        timestep_s=world.model.opt.timestep,
        added_mass_enabled=config.WaterParams.ADDED_MASS_ENABLED,
        steps_per_repeat=args.steps, warmup_steps=args.warmup, repeats=args.repeat,
        wall_seconds=durations, steps_per_second=rates,
        median_steps_per_second=median_rate,
        median_realtime_factor=median_rate * world.model.opt.timestep,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
