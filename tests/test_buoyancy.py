"""형상 부피 근사와 겹침·좌표 변환 검사."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim.buoyancy import body_voxels
from sim import config
from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.world import build_world


def sphere_model(*, duplicate=False, rotated=False):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='probe', pos=[3, 2, 1])
    body.add_freejoint()
    if rotated:
        body.quat[:] = [2**-.5, 0, 0, 2**-.5]
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.2, 0, 0], mass=1, group=config.Groups.PHYSICS)
    if duplicate:
        child = body.add_body()
        child.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.2, 0, 0], mass=1, group=config.Groups.PHYSICS)
    return spec.compile()


def volume(model):
    points, half, _ = body_voxels(model, model.body('probe').id, .02)
    return np.prod(half * 2) * len(points)


def test_sphere_volume_matches_analytic_volume():
    assert volume(sphere_model()) == pytest.approx(4 / 3 * np.pi * .2**3, rel=.03)


def test_fixed_overlapping_geoms_are_counted_once():
    assert volume(sphere_model(duplicate=True)) == pytest.approx(volume(sphere_model()))


def test_initial_position_and_rotation_do_not_change_volume():
    assert volume(sphere_model(rotated=True)) == pytest.approx(volume(sphere_model()))


def test_human_ring_voxels_preserve_open_center_and_sufficient_volume():
    human = Human()
    world = build_world(parts=[human, Lifebuoy()])
    points, half, _ = body_voxels(world.model, human.body_id, .04)
    ring_id = world.model.body(config.Names.LIFEBUOY_BODY).id
    ring_points, ring_half, _ = body_voxels(world.model, ring_id, .04)
    displaced_mass = (np.prod(half * 2) * len(points) + np.prod(ring_half * 2) * len(ring_points)) * 1025
    assert 72 < displaced_mass < 110
    # 환 평면 중심에서 몸통 바깥·환 안쪽의 빈 공간은 채우지 않는다.
    empty = np.array([.35, 0, .18])
    assert not np.any(np.all(np.abs(points - empty) <= half, axis=1))


@pytest.mark.parametrize('spacing', [0, -.1, float('nan')])
def test_invalid_spacing_is_rejected(spacing):
    with pytest.raises(ValueError, match='간격'):
        body_voxels(sphere_model(), 1, spacing)
