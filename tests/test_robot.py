"""계약 질량·실제 추력 가속도·차동 조향 검증."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.robot import Robot
from sim.frames import body_velocity_w
from sim.view_towing import advance
from sim.world import build_world


def make_robot():
    robot = Robot()
    return build_world(parts=[robot], gravity=0), robot


def test_hull_mass_geometry_and_optional_tow_site():
    world, robot = make_robot()
    assert world.model.body_subtreemass[robot.body_id] == pytest.approx(config.Hull.MASS)
    np.testing.assert_allclose(world.model.geom_size[robot.geom_id],
                               np.array([config.Hull.LENGTH, config.Hull.BEAM, config.Hull.HEIGHT]) / 2)
    np.testing.assert_allclose(world.model.body_ipos[robot.body_id], config.Hull.COM)
    assert world.model.nsite == 2  # 미합의 tow_point는 기본 생성하지 않는다.


@pytest.mark.parametrize('command,force', [(1, 600), (-1, -300), (0, 0)])
def test_actual_forward_reverse_acceleration(command, force):
    world, robot = make_robot()
    robot.set_command(command)
    advance(world)
    assert body_velocity_w(world.model, world.data, robot.body_id)[0][0] / world.model.opt.timestep == pytest.approx(force / config.Hull.MASS)


def test_yaw_ninety_forward_accelerates_world_y():
    world, robot = make_robot()
    address = world.model.jnt_qposadr[robot.joint_id]
    world.data.qpos[address + 3:address + 7] = [2**-.5, 0, 0, 2**-.5]
    robot.set_command(1)
    advance(world)
    dof = world.model.jnt_dofadr[robot.joint_id]
    assert world.data.qvel[dof] == pytest.approx(0, abs=1e-12)
    assert body_velocity_w(world.model, world.data, robot.body_id)[0][1] / world.model.opt.timestep == pytest.approx(600 / config.Hull.MASS)


def test_positive_steering_turns_left_and_mixing_saturates():
    world, robot = make_robot()
    robot.set_command(steering=1)
    advance(world)
    dof = world.model.jnt_dofadr[robot.joint_id]
    assert world.data.qvel[dof + 5] > 0
    world, robot = make_robot()
    robot.set_command(forward=1, steering=1)
    advance(world)
    assert body_velocity_w(world.model, world.data, robot.body_id)[0][0] / world.model.opt.timestep == pytest.approx(300 / config.Hull.MASS)


@pytest.mark.parametrize('value', [1.1, -1.1, float('nan'), float('inf')])
def test_invalid_commands_rejected(value):
    with pytest.raises(ValueError):
        Robot().set_command(value)
