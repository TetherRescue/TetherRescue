"""장력·느슨함·단축 시 압축 방지를 실제 엔진의 가속도로 검증한다."""

import numpy as np
import pytest

from sim import config
from sim.forces import ForceAccumulator
from sim.tether import Tether
from sim.world import build_world

mujoco = pytest.importorskip("mujoco")

# 아래 기대값은 숫자로 적지 않고 config.Tether 값에서 계산한다.
# 그래서 강성·감쇠·기준 길이를 바꿔도 이 테스트는 그대로 쓸 수 있다.
END_MASS = 2.0   # 시험용 끝 물체 질량 [kg] (EndpointPair가 만드는 구)


def rest():
    return config.Tether.REST_LENGTH


def expected_tension(length, rate):
    """tether.py 규칙의 총장력 [N]: 늘어난 줄은 max(0, k·(L−L0) + c·L'), 느슨한 줄은 0."""
    extension = length - rest()
    if extension <= 0:
        return 0.0
    return max(0.0, config.Tether.STIFFNESS * extension + config.Tether.DAMPING * rate)


class EndpointPair:
    """구명환 대신 계산이 쉬운 2kg 구를 쓰는 로프 단위 시험 장면."""
    REQUIRES = {}

    def __init__(self, length, moving_anchor=False):
        self.length = length
        self.moving_anchor = moving_anchor

    def build(self, spec):
        if self.moving_anchor:
            start = spec.worldbody.add_body()
            start.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
            start.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=END_MASS,
                           contype=0, conaffinity=0)
        else:
            start = spec.worldbody
        start.add_site(name=config.Names.TOW_POINT_SITE)
        end = spec.worldbody.add_body(pos=[self.length, 0, 0])
        end.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        end.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=END_MASS,
                     contype=0, conaffinity=0)
        end.add_site(name=config.Names.LIFEBUOY_TOW_SITE)

    def bind(self, model):
        pass

    def add_forces(self, model, data, acc):
        pass


def _cancel_rate():
    """0.5m 늘어난 줄의 스프링을 감쇠가 정확히 두 배로 상쇄하는 줄어드는 속도 (총장력 0이 되는 경우)."""
    return -2 * config.Tether.STIFFNESS * 0.5 / config.Tether.DAMPING


# (끝점 거리, 길이 변화 속도). 길이는 기준 길이 L0에 대한 상대값으로 정한다.
CASES = {
    "slack-still": (lambda: 0.5 * rest(), lambda: 0.0),
    "slack-lengthening": (lambda: 0.5 * rest(), lambda: 5.0),
    "slack-shortening": (lambda: 0.5 * rest(), lambda: -5.0),
    "exactly-rest": (lambda: rest(), lambda: 0.0),
    "stretched-still": (lambda: rest() + 0.5, lambda: 0.0),
    "stretched-lengthening": (lambda: rest() + 0.5, lambda: 2.0),
    "stretched-shortening": (lambda: rest() + 0.5, lambda: -2.0),
    "stretched-fast-shortening": (lambda: rest() + 0.5, _cancel_rate),   # 감쇠가 스프링을 넘어도 밀지 않음
    "zero-length": (lambda: 0.0, lambda: 0.0),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_total_tension_and_actual_acceleration(case):
    length, velocity = case[0](), case[1]()
    tension = expected_tension(length, velocity)
    rope = Tether(enabled=True)
    world = build_world(parts=[EndpointPair(length), rope])
    world.data.qvel[0] = velocity
    mujoco.mj_forward(world.model, world.data)
    assert rope.tension(world.model, world.data) == pytest.approx(tension)
    acc = ForceAccumulator()
    rope.add_forces(world.model, world.data, acc)
    acc.apply(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    # 끝 물체의 x 가속도로 native 스프링+외력 보정의 합을 검증한다.
    assert world.data.qacc[0] == pytest.approx(-tension / END_MASS, abs=1e-10)
    np.testing.assert_array_equal(world.data.qfrc_applied, 0)


def test_force_and_reaction_on_two_free_bodies():
    rope = Tether(enabled=True)
    world = build_world(parts=[EndpointPair(rest() + 0.5, moving_anchor=True), rope])
    world.data.qvel[6] = 2  # 끝 물체가 멀어지는 속도
    mujoco.mj_forward(world.model, world.data)
    world.acc.reset()
    rope.add_forces(world.model, world.data, world.acc)
    world.acc.apply(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    acceleration = expected_tension(rest() + 0.5, 2.0) / END_MASS
    assert acceleration > 0
    assert world.data.qacc[0] == pytest.approx(acceleration)
    assert world.data.qacc[6] == pytest.approx(-acceleration)


def test_disabled_tether_does_not_need_endpoints():
    rope = Tether()
    world = build_world(parts=[rope])
    assert world.model.ntendon == 0
    assert rope.tension(world.model, world.data) == 0
    rope.add_forces(world.model, world.data, world.acc)


def test_missing_endpoint_fails_clearly():
    with pytest.raises(ValueError, match="tow_point"):
        build_world(parts=[Tether(enabled=True)])


@pytest.mark.parametrize("field,value", [("REST_LENGTH", 0), ("STIFFNESS", -1), ("DAMPING", -1)])
def test_invalid_parameters_are_rejected(monkeypatch, field, value):
    monkeypatch.setattr(config.Tether, field, value)
    with pytest.raises(ValueError):
        build_world(parts=[EndpointPair(rest() + 0.5), Tether(enabled=True)])


def test_actual_ring_pulls_human_and_runs_thousand_steps_without_reset():
    from sim.view_tether import build_demo_world

    world = build_demo_world()
    human_id = world.model.body(config.Names.HUMAN_BODY).id
    ring_id = world.model.body(config.Names.LIFEBUOY_BODY).id
    rope = world.parts[-1]
    masses = world.model.body_mass[[human_id, ring_id]]

    def center_of_mass():
        return masses @ world.data.xipos[[human_id, ring_id]] / masses.sum()

    initial_center = center_of_mass()
    for step in range(1000):
        mujoco.mj_forward(world.model, world.data)
        assert rope.tension(world.model, world.data) >= 0
        world.acc.reset()
        for part in world.parts:
            part.add_forces(world.model, world.data, world.acc)
        world.acc.apply(world.model, world.data)
        mujoco.mj_step(world.model, world.data)
        assert np.isfinite(world.data.qpos).all()
        assert np.isfinite(world.data.qvel).all()
        if step == 99:
            mujoco.mj_forward(world.model, world.data)
            center = center_of_mass()
            assert center[0] > initial_center[0]  # 고정점(+x) 쪽으로 사람도 이동
    assert world.data.time == pytest.approx(2.0)  # 데모 리셋 없이 연속 실행
    assert not world.data.warning.number.any()


def test_comparison_makes_slack_and_release_distinct_without_changing_shared_gravity():
    from sim.view_tether import PullComparison

    demo = PullComparison()
    np.testing.assert_array_equal(demo.world.model.opt.gravity, 0)
    assert config.Physics.GRAVITY == 9.81
    assert demo.readings()["length"] == pytest.approx(1.5)
    assert demo.readings()["tension"] == 0
    held = released = False
    for _ in range(3500):
        demo.advance()
        reading = demo.readings()
        assert reading["tension"] >= 0
        if demo.phase == "SLACK":
            assert abs(reading["displacement"]) < 1e-10
        elif demo.phase == "HELD":
            held = True
            assert reading["extension"] == pytest.approx(1.0)
            assert reading["tension"] == pytest.approx(config.Tether.STIFFNESS * reading["extension"])
            assert abs(reading["displacement"]) < 1e-10
        elif 3.5 < demo.clock < 4:
            released = True
            assert reading["displacement"] > 0.1
    assert held and released
    assert not demo.world.data.warning.number.any()


def test_detached_tendon_has_no_native_spring_or_damping_force():
    rope = Tether(enabled=True, attached=False)
    world = build_world(parts=[EndpointPair(rest() + 1.), rope], gravity=0)
    world.data.qvel[0] = 2.
    mujoco.mj_forward(world.model, world.data)
    rope.add_forces(world.model, world.data, world.acc)
    world.acc.apply(world.model, world.data)
    mujoco.mj_forward(world.model, world.data)
    assert rope.tension(world.model, world.data) == 0
    assert world.model.tendon_stiffness[rope.tendon_id] == 0
    assert world.data.qacc[0] == pytest.approx(0)
    np.testing.assert_array_equal(world.data.qfrc_passive, 0)


def test_runtime_attach_and_detach_preserve_state_and_change_actual_force():
    rope = Tether(enabled=True, attached=False)
    world = build_world(parts=[EndpointPair(rest() + 1.), rope], gravity=0)
    world.data.qvel[0] = 2.
    pulled = -expected_tension(rest() + 1., 2.) / END_MASS
    qpos, qvel = world.data.qpos.copy(), world.data.qvel.copy()
    spring_lengths = world.model.tendon_lengthspring.copy()
    for attached, acceleration in ((True, pulled), (True, pulled), (False, 0.)):
        rope.set_attached(world.model, attached)
        np.testing.assert_array_equal(world.data.qpos, qpos)
        np.testing.assert_array_equal(world.data.qvel, qvel)
        np.testing.assert_array_equal(world.model.tendon_lengthspring, spring_lengths)
        mujoco.mj_forward(world.model, world.data)
        world.acc.reset()
        rope.add_forces(world.model, world.data, world.acc)
        world.acc.apply(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)
        assert world.data.qacc[0] == pytest.approx(acceleration)
        assert rope.attached == attached


def test_attaching_slack_rope_does_not_produce_initial_force():
    rope = Tether(enabled=True, attached=False)
    world = build_world(parts=[EndpointPair(0.35 * rest()), rope], gravity=0)
    rope.attach(world.model)
    mujoco.mj_forward(world.model, world.data)
    assert rope.tension(world.model, world.data) == 0
    np.testing.assert_array_equal(world.data.qacc, 0)


def test_tether_reset_restores_initial_model_coefficients():
    rope = Tether(enabled=True, attached=False)
    world = build_world(parts=[EndpointPair(rest() + 1.), rope], gravity=0)
    rope.attach(world.model)
    assert world.model.tendon_stiffness[rope.tendon_id] > 0
    mujoco.mj_resetData(world.model, world.data)
    rope.reset(world.model)
    mujoco.mj_forward(world.model, world.data)
    assert not rope.attached
    np.testing.assert_array_equal(world.data.qacc, 0)


def test_unbuilt_or_wrong_model_cannot_be_attached():
    with pytest.raises(ValueError, match='생성'):
        Tether(enabled=False, attached=True)
    rope = Tether(enabled=True, attached=False)
    world = build_world(parts=[EndpointPair(rest() + 1.), rope])
    other = build_world(parts=[EndpointPair(rest() + 1.), Tether(enabled=True, attached=False)])
    with pytest.raises(ValueError, match='모델'):
        rope.attach(other.model)
    with pytest.raises(ValueError, match='모델'):
        Tether().attach(world.model)
