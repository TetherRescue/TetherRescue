"""기본 세계 조립과 배·로프·구명환·사람의 무중력 힘 전달 시험.

수상 견인·정착 시험은 test_water_physics에서 실행한다.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim.world.view_towing import advance, build_towing_world
from sim.frames import body_velocity_w


def test_default_assembly_runs_thousand_steps_without_warning():
    from sim.world.world import advance_world, build_world

    world = build_world()
    for _ in range(1000):
        advance_world(world)
        for state in (world.data.qpos, world.data.qvel, world.data.qacc):
            assert np.isfinite(state).all()
        assert not world.data.warning.number.any()
    assert world.data.time == pytest.approx(1000 * world.model.opt.timestep)


def test_boat_tows_human_for_thousand_steps_without_reset():
    world, robot, human, rope = build_towing_world()
    assert rope.body_ids == (robot.body_id, world.model.body('lifebuoy').id)
    start = world.data.xpos[human.body_id].copy()
    max_tension = 0
    for _ in range(1000):
        advance(world)
        max_tension = max(max_tension, rope.tension(world.model, world.data))
        for state in (world.data.qpos, world.data.qvel, world.data.qacc):
            assert np.isfinite(state).all()
    assert world.data.time == pytest.approx(2)
    assert world.data.xpos[human.body_id, 0] > start[0] + .1
    assert max_tension > 0
    assert not world.data.warning.number.any()


def test_rope_reaction_preserves_total_linear_momentum_without_thrust():
    world, robot, human, rope = build_towing_world()
    # 실제 로프 강성(1만 N/m)에서는 무중력의 환·사람이 줄에 홱 당겨져 20 rad/s 넘게 돈다.
    # 기본 적분기(implicitfast, 0.002 s)는 빠른 회전의 원심 항에서 운동량 오차 ≈ 0.2 kg·m/s 를 만든다
    # (시간 간격 1/4 → 오차 1/4 로 줄어 수치 오차임을 확인, 2026-11-17).
    # 이 시험은 '줄 반작용이 같고 반대인가'를 보므로 회전 적분 오차가 작은 RK4 로 돌린다.
    world.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    robot.set_command(0)
    initial_distance = world.data.ten_length[rope.tendon_id]
    for _ in range(500):
        advance(world)
    # 몸체마다 원점 속도 × 질량은 무게중심이 원점에서 떨어진 몸체(A형 배, 카메라 팔)에서 틀리므로
    # MuJoCo 가 계산한 전체 무게중심 속도 × 전체 질량으로 잰다 (2026-11-19)
    mujoco.mj_subtreeVel(world.model, world.data)
    momentum = world.model.body_subtreemass[0] * world.data.subtree_linvel[0]
    # 총 322kg. RK4 측정 오차 ≈ 0.002 kg·m/s (잔여 속도 약 6e-6 m/s).
    # 2026-11-19: 카메라 회전 팔(경첩 하나)이 생겨 줄에 홱 당겨진 배가 빠르게 돌 때 오차 ≈ 0.008 (팔 모터를 꺼도 0.004)
    #   → 1e-2 허용 (속도로 약 3e-5 m/s). 팔 힘이 안쪽 힘이라는 건 tests/robot/test_inversion.py 가 1e-6 으로 확인.
    np.testing.assert_allclose(momentum, 0, atol=1e-2)
    assert world.data.ten_length[rope.tendon_id] < initial_distance
    assert world.data.xpos[robot.body_id, 0] < 4
    assert world.data.xpos[human.body_id, 0] > 0


def test_without_rope_thrust_does_not_move_human():
    from sim.world.human import Human
    from sim.world.lifebuoy import Lifebuoy
    from sim.robot.robot import Robot
    from sim.world.world import build_world

    robot, human = Robot(), Human()
    world = build_world(parts=[robot, human, Lifebuoy()], gravity=0)
    robot.set_command(.2)
    start = world.data.xpos[human.body_id].copy()
    boat_start = world.data.xpos[robot.body_id, 0]
    for _ in range(1000):
        advance(world)
    np.testing.assert_allclose(world.data.xpos[human.body_id], start, atol=1e-12)
    assert world.data.xpos[robot.body_id, 0] > boat_start + .1
