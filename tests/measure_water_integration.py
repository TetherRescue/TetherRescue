"""물·접근·견인의 수치 측정. 저장소 루트에서 python -m tests... 대신 아래 명령 사용.

python tests/measure_water_integration.py --case all
판정은 현재 모델의 수치 안정성과 기존 시나리오 완료 조건에 한정한다.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sim import config
from sim.robot import Robot
from sim.water import Water
from sim.world import advance_world, build_world
from sim.view_rescue import build_rescue_demo, build_rescue_mission
from sim.view_towing import build_surface_towing_world


def measure(case, nozzle_cutoff=True):
    waves = config.WaterParams.WAVES_DEFAULT if case.startswith('waves') else ()
    manager = None
    tow_human = tow_rope = None
    if case == 'waves_boat':
        water = Water(waves=waves)
        robot = Robot(surface_height=water.wave_height if nozzle_cutoff else None)
        world = build_world(parts=[water, robot])
        limit = 30.
    elif case == 'waves_towing':
        world, robot, tow_human, tow_rope = build_surface_towing_world()
        water = next(p for p in world.parts if isinstance(p, Water))
        water.waves = waves
        robot.surface_height = water.wave_height if nozzle_cutoff else None
        human_start = world.data.xpos[tow_human.body_id].copy()
        limit = 30.
    else:
        builder = build_rescue_mission if case.endswith('mission') else build_rescue_demo
        manager = builder()
        world, robot = manager.world, manager.robot
        water = next(p for p in world.parts if isinstance(p, Water))
        water.waves = waves
        robot.surface_height = water.wave_height if nozzle_cutoff else None
        limit = 240. if case.endswith('mission') else 180.
    result = dict(case=case, timestep_s=world.model.opt.timestep,
                  waves=waves, nozzle_cutoff=nozzle_cutoff, samples=0,
                  max_tilt_deg=0., max_abs_vz_m_s=0., max_tension_N=0.,
                  min_nozzle_depth_m=[math.inf, math.inf],
                  dry_counts=[0, 0], stages=[])
    done_since = None
    reason = None
    try:
        while world.data.time < limit:
            positions = world.data.site_xpos[[robot.thr_l_site_id, robot.thr_r_site_id]]
            depth = water.wave_height(positions[:, 0], positions[:, 1], world.data.time) - positions[:, 2]
            result['samples'] += 1
            result['min_nozzle_depth_m'] = np.minimum(result['min_nozzle_depth_m'], depth).tolist()
            result['dry_counts'] = (np.asarray(result['dry_counts']) + (depth <= 0)).tolist()
            if manager is None:
                if case == 'waves_boat':
                    robot.set_command(0. if world.data.time < 5. else 1.)
                advance_world(world)
            else:
                manager.step()
            data = world.data
            if not all(np.isfinite(getattr(data, n)).all() for n in ('qpos', 'qvel', 'qacc')):
                reason = 'NONFINITE'
                break
            if data.warning.number.any():
                reason = 'ENGINE_WARNING'
                break
            up_z = data.xmat[robot.body_id, 8]
            tilt = float(np.degrees(np.arccos(np.clip(up_z, -1, 1))))
            result['max_tilt_deg'] = max(result['max_tilt_deg'], tilt)
            result['max_abs_vz_m_s'] = max(result['max_abs_vz_m_s'], abs(float(data.qvel[robot.qvel_adr + 2])))
            if tow_rope is not None:
                result['max_tension_N'] = max(result['max_tension_N'], tow_rope.tension(world.model, data))
            if manager is not None:
                stage = manager.stage.value
                if not result['stages'] or result['stages'][-1]['stage'] != stage:
                    result['stages'].append(dict(stage=stage, time_s=float(data.time)))
                if hasattr(manager, 'rope'):
                    result['max_tension_N'] = max(result['max_tension_N'], manager.rope.tension(world.model, data))
                if getattr(manager, 'failed', False):
                    reason = manager.failure.code.value + ': ' + manager.failure.message
                    break
                if manager.arrived:
                    if done_since is None:
                        done_since = float(data.time)
                    if data.time >= done_since + 5.:
                        break
                else:
                    done_since = None
            elif tilt >= 60.:
                reason = 'TILT_60_DEG'  # 측정용 전체 기울기 한계. Watchdog의 축별 판정과 구분.
                break
    except (ValueError, RuntimeError, FloatingPointError) as error:
        reason = type(error).__name__ + ': ' + str(error)
    result['time_s'] = float(world.data.time)
    result['completed_at_s'] = done_since
    result['warnings'] = world.data.warning.number.tolist()
    result['final_velocity_m_s'] = world.data.qvel[robot.qvel_adr:robot.qvel_adr + 3].tolist()
    result['final_position_m'] = world.data.xpos[robot.body_id].tolist()
    if tow_human is not None:
        result['human_progress_x_m'] = float(world.data.xpos[tow_human.body_id, 0] - human_start[0])
        result['final_human_position_m'] = world.data.xpos[tow_human.body_id].tolist()
    if manager is not None:
        result['readings'] = manager.readings()
        result['minimum_observed_clearance_m'] = manager.minimum_observed_clearance
        success = done_since is not None and world.data.time >= done_since + 5.
        if not success and reason is None:
            reason = 'COMPLETION_HOLD_NOT_REACHED'
    else:
        success = world.data.time >= limit
    result['passed'] = bool(reason is None and success)
    result['failure'] = reason
    result['dry_fraction'] = (np.asarray(result['dry_counts']) / max(1, result['samples'])).tolist()
    return result


def main():
    cases = ('calm_approach', 'calm_mission', 'waves_boat', 'waves_towing', 'waves_approach', 'waves_mission')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', *cases), default='all')
    parser.add_argument('--no-cutoff', action='store_true', help='노즐은 항상 물속이라는 기존 기본 가정으로 비교')
    args = parser.parse_args()
    results = []
    for case in cases if args.case == 'all' else (args.case,):
        result = measure(case, nozzle_cutoff=not args.no_cutoff)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False,
                         default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                         else v.item() if isinstance(v, np.generic) else str(v)), flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
