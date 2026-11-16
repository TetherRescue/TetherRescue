"""배 ↔ 사람·구명환 실제 충돌 시험 (담당: 가영, 통합 충돌 확인).

config.Contacts 표에서 배·사람·구명환은 서로 부딪히게 되어 있다. test_world는 비트 설정만 보고,
여기서는 잔잔한 물 위에서 실제로 부딪혔을 때 물리가 멀쩡한지 확인한다.

장면: 물에 자리 잡은 사람(+고정 구명환)을 향해 배가 직진한다. 구조 접근(RescueApproach)은
0.1m 안에서 미리 멈추므로, 이 시험은 그 안전장치가 실패했을 때의 최악 상황을 대신한다.

2026-11-16 측정 (사람은 물에 5초 자리 잡은 자세, 배와 최소 간격 0.15m에서 출발):
  1 m/s 충돌(발·옆·머리 쪽): 겹침 최대 4.0mm, 사람 속도 변화 최대 0.18 m/s (완전 탄성 한계 0.43 m/s)
  최대 추력으로 5초 밀기(발·옆): 겹침 최대 5.1mm, 사람 약 1m 밀려남. 옆에서 밀면 배가 회전해 사람 둘레를 돈다.
  모든 경우 경고·NaN 없음 → 사람·구명환 solref(.006, 1)·solimp 설정을 바꾸지 않아도 된다.
"""

import math

import numpy as np
import pytest

from sim import config

mujoco = pytest.importorskip("mujoco")

from sim.human import Human  # noqa: E402
from sim.lifebuoy import Lifebuoy  # noqa: E402
from sim.robot import Robot  # noqa: E402
from sim.water import Water  # noqa: E402
from sim.world import build_world, advance_world  # noqa: E402

SETTLE_SECONDS = 5.0      # 사람이 물에 자리 잡는 시간 (측정: 약 4초 뒤 속도 < 0.01 m/s)
MAX_PENETRATION = 0.01    # m. 충돌 겹침 허용 (측정 최대 4mm)
MAX_PUSH_PENETRATION = 0.015  # m. 최대 추력으로 계속 밀 때 (측정 최대 5.1mm)
FLOATING = [config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY]
FREE_JOINTS = [config.Names.HUMAN_FREE, config.Names.LIFEBUOY_FREE]


def _human():
    return Human(position=(0, 0, config.WaterParams.SURFACE_Z - .03))


@pytest.fixture(scope="module")
def settled_human():
    """배 없이 사람·구명환만 물에 띄워 자리 잡은 자세.

    고정 연결 구명환은 자기 자유 관절을 가진 별도 body이고 weld로 사람에 붙어 있다.
    그래서 사람과 구명환의 qpos를 함께 저장한다 ({관절 이름: qpos 7개}).
    """
    human = _human()
    world = build_world(parts=[Water(waves=(), include_boat=False, floating_bodies=FLOATING),
                               human, Lifebuoy(attached=True)])
    model, data = world.model, world.data
    for _ in range(int(SETTLE_SECONDS / model.opt.timestep)):
        advance_world(world)
    assert np.linalg.norm(data.qvel) < 0.05, "사람이 아직 자리 잡지 않음"
    return {name: data.qpos[model.jnt_qposadr[model.joint(name).id]:][:7].copy() for name in FREE_JOINTS}


def _min_distance(model, data, geoms_a, geoms_b, limit=5.0):
    fromto = np.zeros(6)
    return min(mujoco.mj_geomDistance(model, data, a, b, limit, fromto) for a in geoms_a for b in geoms_b)


def _bump_scene(settled, heading_deg, gap=0.15):
    """자리 잡은 사람을 원점에 두고, heading 방향으로 달려올 배를 가장 가까운 형상 간격 gap[m]에 둔다."""
    yaw = math.radians(heading_deg)
    direction = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    draft = config.Hull.MASS / (config.Physics.RHO_WATER * config.Hull.LENGTH * config.Hull.BEAM)
    start = -(config.Hull.LENGTH / 2 + 2.0) * direction   # 일단 충분히 멀리 둔 뒤 아래에서 당긴다
    robot = Robot(position=(start[0], start[1], config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - draft))
    human = _human()
    world = build_world(parts=[Water(waves=(), floating_bodies=FLOATING), robot, human, Lifebuoy(attached=True)])
    model, data = world.model, world.data

    boat_q = model.jnt_qposadr[robot.joint_id]
    data.qpos[boat_q + 3:boat_q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    for name, qpos in settled.items():
        address = model.jnt_qposadr[model.joint(name).id]
        data.qpos[address:address + 7] = qpos
    mujoco.mj_forward(model, data)

    boat_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == robot.body_id}
    victim_geoms = [g for g in range(model.ngeom) if model.geom_group[g] == config.Groups.PHYSICS
                    and model.geom_contype[g] and g not in boat_geoms]
    for _ in range(5):   # 배를 진행 방향으로 옮겨 실제 최소 간격을 gap에 맞춘다
        data.qpos[boat_q:boat_q + 3] += (_min_distance(model, data, boat_geoms, victim_geoms) - gap) * direction
        mujoco.mj_forward(model, data)
    assert _min_distance(model, data, boat_geoms, victim_geoms) == pytest.approx(gap, abs=0.01)
    assert not _boat_contacts(data, boat_geoms)
    return world, robot, human, direction, boat_geoms


def _boat_contacts(data, boat_geoms):
    return [c for c in data.contact[:data.ncon] if c.geom1 in boat_geoms or c.geom2 in boat_geoms]


def _run(world, robot, human, direction, boat_geoms, seconds, distance_every=10):
    """실행하며 겹침·사람 속도 변화를 기록한다.

    관통 판정은 형상 사이 부호 있는 거리(mj_geomDistance, 겹치면 음수)로 한다.
    배 원점과 사람의 앞뒤 위치로 판정하면 안 된다: 한쪽 모서리로 밀면 배가 마찰로 회전해
    사람 둘레를 닿은 채로 돌아가는데(정상 물리), 그때 사람이 배 '뒤'에 있게 되기 때문이다.
    """
    model, data = world.model, world.data
    dof = slice(model.jnt_dofadr[human.joint_id], model.jnt_dofadr[human.joint_id] + 3)
    victim_geoms = [g for g in range(model.ngeom) if model.geom_group[g] == config.Groups.PHYSICS
                    and model.geom_contype[g] and g not in boat_geoms]
    human_start, human_v0 = data.xipos[human.body_id].copy(), data.qvel[dof].copy()
    record = dict(penetration=0.0, hit=False, human_dv=0.0)
    for step in range(int(seconds / model.opt.timestep)):
        advance_world(world)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "계산이 폭발함 (NaN/Inf)"
        for contact in _boat_contacts(data, boat_geoms):
            record["hit"] = True
            record["penetration"] = max(record["penetration"], -contact.dist)
        if step % distance_every == 0:   # 접촉 검출과 별개로 실제 형상 겹침도 확인 (관통 대비)
            gap = _min_distance(model, data, boat_geoms, victim_geoms)
            record["penetration"] = max(record["penetration"], -gap)
        record["human_dv"] = max(record["human_dv"], float(np.dot(data.qvel[dof] - human_v0, direction)))
    record["push"] = float(np.dot(data.xipos[human.body_id] - human_start, direction))
    record["warnings"] = sum(int(w.number) for w in data.warning)
    return record


# 배가 달리는 방향. 0 = 발 쪽에서, 90 = 옆(구명환)에서, 180 = 머리 쪽에서.
DIRECTIONS = {"feet": 0, "side": 90, "head": 180}


@pytest.mark.parametrize("heading_deg", DIRECTIONS.values(), ids=DIRECTIONS.keys())
def test_boat_impact_does_not_pass_through_or_add_energy(settled_human, heading_deg):
    speed = 1.0   # m/s. 구조 접근 목표 속도(0.2~0.4 m/s)보다 빠른 최악 상황
    world, robot, human, direction, boat_geoms = _bump_scene(settled_human, heading_deg)
    world.data.qvel[robot.qvel_adr:robot.qvel_adr + 3] = speed * direction
    record = _run(world, robot, human, direction, boat_geoms, seconds=3.0)

    model = world.model
    boat_mass = model.body_subtreemass[robot.body_id]
    human_mass = model.body_subtreemass[human.body_id] + model.body_subtreemass[  # 사람 + weld된 구명환
        model.body(config.Names.LIFEBUOY_BODY).id]
    elastic_limit = 2 * boat_mass / (boat_mass + human_mass) * speed   # 완전 탄성 충돌의 최대 속도

    assert record["hit"], "배가 사람·환에 닿지 않음 — 장면 배치를 확인할 것"
    assert record["penetration"] < MAX_PENETRATION
    assert 0 < record["human_dv"] < elastic_limit, "충돌로 에너지가 생겨남 (튕겨 나가는 속도가 너무 큼)"
    assert record["push"] > 0, "사람이 배가 미는 방향으로 밀려나지 않음"
    assert record["warnings"] == 0


@pytest.mark.parametrize("heading_deg", [DIRECTIONS["feet"], DIRECTIONS["side"]], ids=["feet", "side"])
def test_full_thrust_push_stays_stable(settled_human, heading_deg):
    # 최대 추력으로 계속 밀어도 뚫고 지나가지 않고 사람을 앞으로 밀어낸다.
    # 옆(구명환)에서 밀면 배가 마찰로 회전해 사람 둘레를 닿은 채 돌아간다 — 겹침만 작으면 정상.
    world, robot, human, direction, boat_geoms = _bump_scene(settled_human, heading_deg)
    robot.set_command(1.0, 0.0)
    record = _run(world, robot, human, direction, boat_geoms, seconds=5.0)

    assert record["hit"]
    assert record["penetration"] < MAX_PUSH_PENETRATION
    assert record["push"] > 0.3, "계속 미는데 사람이 거의 움직이지 않음 (끼임 의심)"
    assert record["warnings"] == 0
