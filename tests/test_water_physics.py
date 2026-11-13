"""MuJoCo에서 부력·저항·흘수·수상 견인을 검증한다."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim import config
from sim.frames import body_velocity_w
from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.robot import Robot
from sim.view_towing import advance, build_surface_towing_world
from sim.water import Water
from sim.world import build_world


def boat_at(height, *, gravity=None):
    boat = Robot(position=(0, 0, height))
    water = Water(waves=())
    return build_world(parts=[water, boat], gravity=gravity), water, boat


def applied_loads(world, water):
    mujoco.mj_forward(world.model, world.data)
    world.acc.reset()
    water.add_forces(world.model, world.data, world.acc)
    world.acc.apply(world.model, world.data)
    return world.data.xfrc_applied.copy()


@pytest.mark.parametrize('height,volume', [
    (1., 0.), (-1., config.Hull.LENGTH * config.Hull.BEAM * config.Hull.HEIGHT),
    (0., config.Hull.LENGTH * config.Hull.BEAM * config.Hull.HEIGHT / 2),
    (config.Hull.HEIGHT / 2 - config.Expect.DRAFT, config.Hull.MASS / config.Physics.RHO_WATER),
])
def test_archimedes_force_matches_analytic_box(height, volume):
    world, water, boat = boat_at(height)
    loads = applied_loads(world, water)
    np.testing.assert_allclose(loads[boat.body_id, :3],
                               [0, 0, volume * config.Physics.RHO_WATER * config.Physics.GRAVITY], atol=1e-9)
    np.testing.assert_allclose(loads[boat.body_id, 3:], 0, atol=1e-9)


def test_dry_body_has_no_water_drag_or_torque():
    world, water, _ = boat_at(2.)
    world.data.qvel[:] = [1, 2, 3, .1, .2, .3]
    np.testing.assert_array_equal(applied_loads(world, water), 0)


def test_gravity_override_does_not_create_spurious_buoyancy():
    world, water, _ = boat_at(-1., gravity=0)
    np.testing.assert_array_equal(applied_loads(world, water), 0)


def test_actual_boat_settles_to_analytic_draft_without_drift():
    world, _, boat = boat_at(.15)
    for _ in range(3000):
        advance(world)
    draft = config.Hull.HEIGHT / 2 - world.data.xpos[boat.body_id, 2]
    assert draft == pytest.approx(config.Expect.DRAFT, abs=2e-4)
    np.testing.assert_allclose(world.data.xpos[boat.body_id, :2], 0, atol=1e-8)
    velocity, omega = body_velocity_w(world.model, world.data, boat.body_id)
    assert np.linalg.norm(velocity) < 1e-3 and np.linalg.norm(omega) < 1e-3
    assert not world.data.warning.number.any()


@pytest.mark.parametrize('axis', [0, 1])
def test_heel_creates_restoring_torque_and_settles(axis):
    world, water, boat = boat_at(config.Hull.HEIGHT / 2 - config.Expect.DRAFT)
    angle = np.deg2rad(8)
    quaternion = np.zeros(4)
    quaternion[0] = np.cos(angle / 2)
    quaternion[1 + axis] = np.sin(angle / 2)
    world.data.qpos[3:7] = quaternion
    assert applied_loads(world, water)[boat.body_id, 3 + axis] < 0
    for _ in range(2500):
        advance(world)
    up = world.data.xmat[boat.body_id].reshape(3, 3)[:, 2]
    assert np.arccos(np.clip(up[2], -1, 1)) < np.deg2rad(.3)
    assert not world.data.warning.number.any()


def test_current_relative_drag_and_yaw_frame(monkeypatch):
    monkeypatch.setattr(config.WaterParams, 'WATER_VELOCITY_W', (0, .5, 0))
    world, water, boat = boat_at(-1.)
    world.data.qpos[3:7] = [2**-.5, 0, 0, 2**-.5]
    loads = applied_loads(world, water)[boat.body_id]
    assert loads[1] > 0
    assert loads[0] == pytest.approx(0, abs=1e-10)
    # 같은 해류 속도로 움직이면 상대 속도와 수평 저항이 0이다.
    world.data.qvel[1] = .5
    np.testing.assert_allclose(applied_loads(world, water)[boat.body_id, :2], 0, atol=1e-10)


def test_drag_removes_energy_for_translation_and_rotation():
    world, water, boat = boat_at(-1.)
    world.data.qvel[:] = [1, -.3, 0, .2, -.1, .3]
    load = applied_loads(world, water)[boat.body_id]
    velocity, omega = body_velocity_w(world.model, world.data, boat.body_id)
    # 중력·부력과 무관한 수평 이동·회전의 수력 일률.
    assert np.dot(load[:2], velocity[:2]) + np.dot(load[3:], omega) < 0


def test_surface_towing_starts_slack_and_stays_afloat_for_ten_seconds():
    world, boat, human, rope = build_surface_towing_world()
    assert world.model.opt.gravity[2] == -config.Physics.GRAVITY
    assert rope.tension(world.model, world.data) == 0
    start = world.data.xpos[human.body_id].copy()
    maximum_tension = 0
    for i in range(5000):
        advance(world)
        maximum_tension = max(maximum_tension, rope.tension(world.model, world.data))
        for state in (world.data.qpos, world.data.qvel, world.data.qacc):
            assert np.isfinite(state).all()
        if i >= 1000:
            assert -.3 < world.data.xpos[boat.body_id, 2] < .4
            assert -.6 < world.data.xpos[human.body_id, 2] < .4
    assert world.data.time == pytest.approx(10)
    assert world.data.xpos[human.body_id, 0] > start[0] + 1
    assert maximum_tension > 1
    assert not world.data.warning.number.any()


def test_person_and_ring_float_without_thrust_or_rope():
    human = Human(position=(0, 0, -.03))
    boat = Robot(position=(4, 0, .1))
    water = Water(waves=(), floating_bodies=[config.Names.HUMAN_BODY])
    world = build_world(parts=[water, boat, human, Lifebuoy()])
    for _ in range(2500):
        advance(world)
    assert -.6 < world.data.xpos[human.body_id, 2] < .4
    assert np.linalg.norm(body_velocity_w(world.model, world.data, human.body_id)[0]) < .05
    assert world.model.body_subtreemass[human.body_id] == pytest.approx(72)


def test_overlapping_water_targets_are_rejected():
    water = Water(waves=(), floating_bodies=[config.Names.HUMAN_BODY, config.Names.LIFEBUOY_BODY])
    with pytest.raises(ValueError, match='독립'):
        build_world(parts=[water, Robot(), Human(), Lifebuoy()])
