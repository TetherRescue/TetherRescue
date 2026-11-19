"""정수 흘수의 복셀·시간 간격 민감도 측정.

실행: python tests/water/measure_water_convergence.py
config 변경은 각 측정의 메모리 안에서만 적용하고 복원한다.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from sim import config
from sim.frames import body_velocity_w
from sim.robot.robot import Robot
from sim.water.water import Water
from sim.world.world import advance_world, build_world


def measure(resolution, timestep, seconds, tail_seconds):
    theoretical_draft = config.Expect.DRAFT
    start_z = (config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2
               - theoretical_draft + .05)
    steps = round(seconds / timestep)
    tail_steps = round(tail_seconds / timestep)
    samples = []
    with patch.multiple(config.WaterParams, VOXEL_RES=resolution,
                        WATER_VELOCITY_W=(0., 0., 0.), ADDED_MASS_ENABLED=False):
        water = Water(waves=())
        robot = Robot(position=(0, 0, start_z))
        world = build_world(parts=[water, robot], timestep=timestep)
        for step in range(steps):
            advance_world(world)
            if not all(np.isfinite(state).all() for state in
                       (world.data.qpos, world.data.qvel, world.data.qacc)):
                raise RuntimeError('NONFINITE')
            if world.data.warning.number.any():
                raise RuntimeError('ENGINE_WARNING')
            if step >= steps - tail_steps:
                rotation = world.data.xmat[robot.body_id].reshape(3, 3)
                origin_draft = (config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2
                                - world.data.xpos[robot.body_id, 2])
                # volume은 마지막 스텝 외력 계산 시점(적분 직전)의 값이다.
                volume_draft = water.submerged_volume / config.Hull.WATERPLANE_AREA
                pitch = np.degrees(np.arctan2(-rotation[2, 0],
                                              np.hypot(rotation[0, 0], rotation[1, 0])))
                samples.append((origin_draft, volume_draft, pitch))
        values = np.asarray(samples)
        velocity, omega = body_velocity_w(world.model, world.data, robot.body_id)
        return dict(
            resolution=list(resolution), voxels=int(np.prod(resolution)),
            timestep_s=timestep, simulation_seconds=float(world.data.time),
            tail_seconds=tail_steps * timestep, theoretical_draft_m=theoretical_draft,
            mean_origin_draft_m=float(values[:, 0].mean()),
            origin_draft_range_m=float(np.ptp(values[:, 0])),
            origin_draft_error_m=float(values[:, 0].mean() - theoretical_draft),
            mean_volume_equivalent_draft_m=float(values[:, 1].mean()),
            mean_pitch_deg=float(values[:, 2].mean()),
            final_speed_m_s=float(np.linalg.norm(velocity)),
            final_angular_speed_rad_s=float(np.linalg.norm(omega)),
            warnings=world.data.warning.number.tolist(),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=8.)
    parser.add_argument('--tail-seconds', type=float, default=1.)
    args = parser.parse_args()
    if not (np.isfinite([args.seconds, args.tail_seconds]).all()
            and .002 <= args.tail_seconds <= args.seconds):
        parser.error('seconds와 tail-seconds는 유한해야 하며 0.002 ≤ tail-seconds ≤ seconds입니다.')
    for resolution in ((4, 2, 2), (8, 4, 4), (16, 8, 8), (32, 16, 16)):
        for timestep in (.002, .001, .0005):
            result = measure(resolution, timestep, args.seconds, args.tail_seconds)
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
