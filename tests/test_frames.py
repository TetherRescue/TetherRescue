"""frames.py 검사: 축 방향, 왕복 변환, 무게중심 속도 (담당: 동경).

물·중력 없이 상자 하나만 만들어 확인합니다.
"""
import math

import mujoco
import numpy as np

from sim import config, frames


def _box(yaw_deg, com_b=config.Hull.COM):
    spec = mujoco.MjSpec()
    spec.option.gravity = (0, 0, 0)
    body = spec.worldbody.add_body(name="b")
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(1.2, 0.5, 0.2))
    body.explicitinertial = True
    body.mass = config.Hull.MASS
    body.ipos = com_b
    body.inertia = config.Hull.INERTIA
    model = spec.compile()
    data = mujoco.MjData(model)
    half = math.radians(yaw_deg) / 2
    data.qpos[3:7] = (math.cos(half), 0, 0, math.sin(half))   # z 축 회전 (w, x, y, z)
    mujoco.mj_forward(model, data)
    return model, data, model.body("b").id


def test_yaw90_forward_is_world_y():
    _, data, bid = _box(90)
    np.testing.assert_allclose(frames.body_to_world(data, bid, (1, 0, 0)), (0, 1, 0), atol=1e-12)
    np.testing.assert_allclose(frames.body_to_world(data, bid, (0, 1, 0)), (-1, 0, 0), atol=1e-12)   # 왼쪽 = 월드 −x


def test_round_trip():
    _, data, bid = _box(37)
    v_b = np.array([0.3, -1.2, 0.5])
    np.testing.assert_allclose(frames.world_to_body(data, bid, frames.body_to_world(data, bid, v_b)), v_b, atol=1e-12)


def test_point_body_to_world():
    model, data, bid = _box(90)
    data.qpos[:3] = (5.0, 2.0, 0.0)
    mujoco.mj_kinematics(model, data)
    # 선미 바닥 점: 배가 +y 를 보고 있으니 뒤쪽(−x_b)은 월드 −y
    np.testing.assert_allclose(frames.point_body_to_world(data, bid, (-1.2, 0, -0.2)), (5.0, 0.8, -0.2), atol=1e-12)
    # 여러 점 (N, 3) 한 번에 = 한 점씩 한 결과와 같음
    pts_b = np.array([[1.0, 0.5, 0.0], [-0.3, -0.2, 0.1]])
    many = frames.point_body_to_world(data, bid, pts_b)
    assert many.shape == (2, 3)
    for p_b, p_w in zip(pts_b, many):
        np.testing.assert_allclose(p_w, frames.point_body_to_world(data, bid, p_b), atol=1e-12)


def test_velocity_is_at_com_in_world():
    model, data, bid = _box(90)
    v_origin_w = np.array([1.0, 2.0, 0.0])
    w_b = np.array([0.0, 0.0, 0.5])            # freejoint 각속도 qvel[3:6] 는 몸체 좌표
    data.qvel[:3] = v_origin_w                 # freejoint 선속도 qvel[0:3] 는 월드 좌표, body 원점 기준
    data.qvel[3:6] = w_b
    mujoco.mj_forward(model, data)

    lin_w, ang_w = frames.body_velocity_w(model, data, bid)

    w_w = frames.body_to_world(data, bid, w_b)
    r_w = data.xipos[bid] - data.xpos[bid]     # 원점 → 무게중심
    np.testing.assert_allclose(ang_w, w_w, atol=1e-12)
    np.testing.assert_allclose(lin_w, v_origin_w + np.cross(w_w, r_w), atol=1e-12)


def test_velocity_com_offset_matters():
    # 무게중심을 옆(+y)으로 옮기고 돌리면 원점 속도와 무게중심 속도가 달라야 함 (기준점이 실제로 COM 인지)
    model, data, bid = _box(0, com_b=(0.0, 0.3, 0.0))
    data.qvel[3:6] = (0, 0, 1.0)
    mujoco.mj_forward(model, data)
    lin_w, _ = frames.body_velocity_w(model, data, bid)
    np.testing.assert_allclose(lin_w, (-0.3, 0, 0), atol=1e-12)
