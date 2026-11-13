"""움직이는 배·로프·구명환·사람의 무중력 힘 전달 시험 (수상 시험은 test_water_physics)."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim.view_towing import advance, build_towing_world
from sim.frames import body_velocity_w


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
    robot.set_command(0)
    initial_distance = world.data.ten_length[rope.tendon_id]
    for _ in range(500):
        advance(world)
    momentum = np.zeros(3)
    for body_id in range(1, world.model.nbody):
        velocity, _ = body_velocity_w(world.model, world.data, body_id)
        momentum += world.model.body_mass[body_id] * velocity
    # 총 322kg의 회전 강체 적분 오차: 잔여 속도 약 3e-6 m/s 이하 허용.
    np.testing.assert_allclose(momentum, 0, atol=1e-3)
    assert world.data.ten_length[rope.tendon_id] < initial_distance
    assert world.data.xpos[robot.body_id, 0] < 4
    assert world.data.xpos[human.body_id, 0] > 0


def test_without_rope_thrust_does_not_move_human():
    from sim.human import Human
    from sim.lifebuoy import Lifebuoy
    from sim.robot import Robot
    from sim.world import build_world

    robot, human = Robot(), Human()
    world = build_world(parts=[robot, human, Lifebuoy()], gravity=0)
    robot.set_command(.2)
    start = world.data.xpos[human.body_id].copy()
    boat_start = world.data.xpos[robot.body_id, 0]
    for _ in range(1000):
        advance(world)
    np.testing.assert_allclose(world.data.xpos[human.body_id], start, atol=1e-12)
    assert world.data.xpos[robot.body_id, 0] > boat_start + .1
