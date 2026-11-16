"""실제 MuJoCo에서 조립 순서·전역 계약·1000스텝 실행을 확인한다."""

import numpy as np
import pytest

from sim import config
from sim.world import build_world

mujoco = pytest.importorskip("mujoco")
SETTINGS = dict(timestep=0.002, gravity=9.81, integrator="implicitfast")


class Probe:
    REQUIRES = {"max_dt": 0.004}

    def __init__(self, name, events):
        self.name = name
        self.events = events

    def build(self, spec):
        self.events.append(f"build:{self.name}")
        body = spec.worldbody.add_body(name=self.name, pos=[0, 0, 1])
        body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0],
                      mass=2, contype=0, conaffinity=0)

    def bind(self, model):
        self.events.append(f"bind:{self.name}")
        self.body_id = model.body(self.name).id

    def add_forces(self, model, data, acc):
        acc.add(self.body_id, [4, 0, 0])


def test_all_builds_precede_all_binds_and_initialize_positions():
    events = []
    parts = [Probe("one", events), Probe("two", events)]
    world = build_world(parts=parts, **SETTINGS)
    assert events == ["build:one", "build:two", "bind:one", "bind:two"]
    assert world.parts == tuple(parts)
    np.testing.assert_allclose(world.data.xipos[parts[0].body_id], [0, 0, 1])
    assert world.model.opt.timestep == SETTINGS["timestep"]
    assert world.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    np.testing.assert_allclose(world.model.opt.gravity, [0, 0, -9.81])
    assert world.model.opt.density == world.model.opt.viscosity == 0


@pytest.mark.parametrize("moving", [False, True])
def test_thousand_steps_are_finite(moving):
    # 빈 모델뿐 아니라 실제로 움직이는 몸체에서도 확인한다.
    parts = [Probe("probe", [])] if moving else ()
    world = build_world(parts=parts, **SETTINGS)
    for _ in range(1000):
        mujoco.mj_forward(world.model, world.data)
        world.acc.reset()
        for part in world.parts:
            part.add_forces(world.model, world.data, world.acc)
        world.acc.apply(world.model, world.data)
        mujoco.mj_step(world.model, world.data)
        for state in (world.data.qpos, world.data.qvel, world.data.qacc):
            assert np.isfinite(state).all()
    elapsed = 1000 * SETTINGS["timestep"]
    assert world.data.time == pytest.approx(elapsed)
    assert not world.data.warning.number.any()
    if moving:
        # 4N / 2kg = 2m/s², 실행 시간에 비례해 속도가 증가한다.
        assert world.data.qvel[0] == pytest.approx(2.0 * elapsed)


def test_uses_config_when_arguments_are_omitted(monkeypatch):
    monkeypatch.setattr(config.Sim, "TIMESTEP", 0.002)
    monkeypatch.setattr(config.Physics, "GRAVITY", 9.8)
    monkeypatch.setattr(config.Sim, "INTEGRATOR", "Euler")
    world = build_world()
    assert world.model.opt.timestep == 0.002
    assert world.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert world.model.opt.gravity[2] == -9.8


def test_missing_contract_is_an_explicit_error(monkeypatch):
    monkeypatch.setattr(config.Sim, "TIMESTEP", None)
    with pytest.raises(ValueError, match="Sim.TIMESTEP.*미확정"):
        build_world()


def test_zero_gravity_can_be_requested_for_isolated_force_experiment():
    world = build_world(parts=(), gravity=0.0)
    np.testing.assert_array_equal(world.model.opt.gravity, 0)
    assert config.Physics.GRAVITY == 9.81


@pytest.mark.parametrize("field,value", [
    ("timestep", 0), ("timestep", float("nan")), ("gravity", -1),
    ("gravity", float("inf")), ("integrator", "unknown"),
])
def test_invalid_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        build_world(**{**SETTINGS, field: value})


def test_default_water_max_dt_is_checked():
    with pytest.raises(ValueError, match="max_dt"):
        build_world(**{**SETTINGS, "timestep": 0.005})


def test_module_max_dt_is_checked():
    with pytest.raises(ValueError, match="Probe.*max_dt"):
        build_world(parts=[Probe("probe", [])], **{**SETTINGS, "timestep": 0.005})


def test_module_cannot_enable_builtin_fluid():
    class BadModule(Probe):
        def build(self, spec):
            spec.option.density = 1000

    with pytest.raises(ValueError, match="전역 물리 설정"):
        build_world(parts=[BadModule("bad", [])], **SETTINGS)


# ── Geom 그룹·충돌 표 (config.Groups·Contacts) ──────────────────────────

class GeomProbe(Probe):
    """geom 하나의 group/contype/conaffinity를 지정해 조립 검사를 시험한다."""

    def __init__(self, name, **geom):
        super().__init__(name, [])
        self.geom = geom

    def build(self, spec):
        body = spec.worldbody.add_body(name=self.name, pos=[0, 0, 1])
        body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        body.add_geom(name=f"{self.name}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[0.1, 0, 0], mass=1, **self.geom)


def test_mjspec_default_geom_is_rejected():
    # 그룹·충돌을 지정하지 않은 geom(group 0, contype 1, conaffinity 1)은 표에 없는 조합이다.
    with pytest.raises(ValueError, match="probe_geom"):
        build_world(parts=[GeomProbe("probe")], **SETTINGS)


@pytest.mark.parametrize("geom", [
    dict(group=config.Groups.VISUAL, contype=config.Contacts.HUMAN, conaffinity=0),
    dict(group=config.Groups.SENSOR, contype=0, conaffinity=config.Contacts.BOAT),
    dict(group=config.Groups.PHYSICS, contype=config.Contacts.HUMAN,
         conaffinity=config.Contacts.HUMAN_AFFINITY | config.Contacts.HUMAN),  # 자기 충돌
    dict(group=config.Groups.PHYSICS, contype=8, conaffinity=8),               # 모르는 비트
    dict(group=1, contype=0, conaffinity=0),                                   # 모르는 그룹
])
def test_geom_outside_table_is_rejected(geom):
    with pytest.raises(ValueError, match="Groups·Contacts"):
        build_world(parts=[GeomProbe("probe", **geom)], **SETTINGS)


@pytest.mark.parametrize("geom", [
    dict(group=config.Groups.VISUAL, contype=0, conaffinity=0),
    dict(group=config.Groups.PHYSICS, contype=0, conaffinity=0),
    dict(group=config.Groups.PHYSICS, contype=config.Contacts.BOAT,
         conaffinity=config.Contacts.BOAT_AFFINITY),
])
def test_geom_inside_table_is_accepted(geom):
    build_world(parts=[GeomProbe("probe", **geom)], **SETTINGS)


def _collides(model, a, b):
    ta, aa = model.geom_contype[a], model.geom_conaffinity[a]
    tb, ab = model.geom_contype[b], model.geom_conaffinity[b]
    return bool((ta & ab) or (tb & aa))


def test_contact_table_pairs_in_assembled_world():
    # 기본 세계: 배·사람·구명환은 서로 부딪히고, 같은 종류끼리는 부딪히지 않는다.
    world = build_world(**SETTINGS)
    model = world.model

    def physics_geoms(body_prefix):
        return [g for g in range(model.ngeom)
                if model.geom_group[g] == config.Groups.PHYSICS
                and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      model.geom_bodyid[g]).startswith(body_prefix)]

    boat = physics_geoms(config.Names.BOAT_BODY)
    human = physics_geoms(config.Names.HUMAN_BODY)
    ring = physics_geoms(config.Names.LIFEBUOY_BODY)
    assert boat and human and ring
    assert _collides(model, boat[0], human[0])
    assert _collides(model, boat[0], ring[0])
    assert _collides(model, human[0], ring[0])
    assert not _collides(model, human[0], human[1])
    assert not _collides(model, ring[0], ring[1])
    visual = [g for g in range(model.ngeom) if model.geom_group[g] == config.Groups.VISUAL]
    assert all(not _collides(model, v, g) for v in visual for g in boat + human + ring)


def test_articulated_arms_have_no_self_contact():
    # 관절 팔 모드에서는 팔·몸통이 다른 body지만 같은 HUMAN 비트라 서로 밀어내지 않는다.
    from sim.human import Human

    world = build_world(parts=[Human(articulated_arms=True)], gravity=0)
    model = world.model
    human = [g for g in range(model.ngeom) if model.geom_group[g] == config.Groups.PHYSICS]
    assert len({model.geom_bodyid[g] for g in human}) > 1
    assert not any(_collides(model, a, b) for a in human for b in human if a < b)
