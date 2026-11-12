"""외력의 합산·무게중심 기준 토크·스텝 사이 초기화 검증."""

from types import SimpleNamespace

import numpy as np
import pytest

from sim.forces import ForceAccumulator


@pytest.fixture
def scene():
    # 몸체 원점과 무게중심이 다른 상황을 표현한다.
    model = SimpleNamespace(nbody=3)
    data = SimpleNamespace(
        xipos=np.array([[0., 0., 0.], [2., 3., 4.], [0., 0., 0.]]),
        xfrc_applied=np.zeros((3, 6)),
        qfrc_applied=np.zeros(12),
    )
    return model, data


def test_sum_forces_and_keep_bodies_separate(scene):
    model, data = scene
    acc = ForceAccumulator()
    acc.add(1, [10, 0, 0])
    acc.add(1, [-3, 0, 8], torque_w=[0, 2, 0])
    acc.add(2, [0, 5, 0])
    acc.apply(model, data)
    np.testing.assert_allclose(data.xfrc_applied[1], [7, 0, 8, 0, 2, 0])
    np.testing.assert_allclose(data.xfrc_applied[2], [0, 5, 0, 0, 0, 0])


def test_offset_force_uses_center_of_mass_and_adds_explicit_torque(scene):
    model, data = scene
    acc = ForceAccumulator()
    # 무게중심에서 +y로 1m 떨어진 곳에 +x로 10N → z축 토크 -10N·m.
    acc.add(1, [10, 0, 0], torque_w=[0, 0, 2], point_w=[2, 4, 4])
    acc.apply(model, data)
    np.testing.assert_allclose(data.xfrc_applied[1], [10, 0, 0, 0, 0, -8])


def test_apply_replaces_old_forces_and_reset_clears_next_step(scene):
    model, data = scene
    acc = ForceAccumulator()
    data.xfrc_applied[:] = 99
    data.qfrc_applied[:] = 99
    acc.add(1, [10, 0, 0])
    acc.apply(model, data)
    acc.apply(model, data)
    assert data.xfrc_applied[1, 0] == 10
    np.testing.assert_array_equal(data.xfrc_applied[2], 0)
    np.testing.assert_array_equal(data.qfrc_applied, 0)
    acc.reset()
    acc.apply(model, data)
    np.testing.assert_array_equal(data.xfrc_applied, 0)


def test_add_copies_inputs(scene):
    model, data = scene
    acc = ForceAccumulator()
    force = np.array([10., 0., 0.])
    point = np.array([2., 4., 4.])
    acc.add(1, force, point_w=point)
    force[:] = 0
    point[:] = 0
    acc.apply(model, data)
    np.testing.assert_allclose(data.xfrc_applied[1], [10, 0, 0, 0, 0, -10])


@pytest.mark.parametrize("argument", ["force_w", "torque_w", "point_w"])
@pytest.mark.parametrize("bad", [[1, 2], [np.nan, 0, 0], [np.inf, 0, 0]])
def test_reject_invalid_vectors(argument, bad):
    kwargs = {"force_w": [0, 0, 0], argument: bad}
    with pytest.raises(ValueError):
        ForceAccumulator().add(1, **kwargs)


@pytest.mark.parametrize("body_id", [-1, 0, 3])
def test_reject_invalid_body_id(scene, body_id):
    model, data = scene
    acc = ForceAccumulator()
    with pytest.raises(ValueError):
        acc.add(body_id, [1, 0, 0])
        acc.apply(model, data)


def test_matches_mujoco_applyft_acceleration():
    """무게중심이 원점과 다른 몸체에서 MuJoCo의 공식 함수와 비교한다."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string('''
        <mujoco><option gravity="0 0 0"/><worldbody>
          <body pos="2 3 4"><freejoint/>
            <inertial pos="0.2 -0.1 0.3" mass="2" diaginertia="0.3 0.4 0.5"/>
            <geom type="sphere" size="0.1"/>
          </body>
        </worldbody></mujoco>
    ''')
    actual = mujoco.MjData(model)
    expected = mujoco.MjData(model)
    mujoco.mj_forward(model, actual)
    mujoco.mj_forward(model, expected)
    force, torque, point = [10., 3., 4.], [1., 2., 3.], [2., 4., 4.]
    acc = ForceAccumulator()
    acc.add(1, force, torque_w=torque, point_w=point)
    acc.apply(model, actual)
    mujoco.mj_applyFT(model, expected, force, torque, point, 1, expected.qfrc_applied)
    mujoco.mj_forward(model, actual)
    mujoco.mj_forward(model, expected)
    np.testing.assert_allclose(actual.qacc, expected.qacc, atol=1e-12)
