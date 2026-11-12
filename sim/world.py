"""3 세계 조립: MjSpec 으로 바다·로봇·사람·줄을 하나의 모델로 조립 (담당: 가영).

순서: 각 모듈 build → compile → 각 모듈 bind
timestep·integrator 같은 전역 설정은 이 파일에서만 정합니다.
"""

from dataclasses import dataclass
import math
from typing import Any

from sim import config
from sim.forces import ForceAccumulator


@dataclass
class World:
    """조립 결과. 스텝 실행 순서는 PhaseManager에서 관리한다."""

    spec: Any
    model: Any
    data: Any
    parts: tuple
    acc: ForceAccumulator


def _positive(value, name, *, allow_zero=False):
    if value is None:
        raise ValueError(f"{name}이 미확정입니다. config를 채우거나 인자로 명시하세요.")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        requirement = "0 이상의 유한한 값" if allow_zero else "유한한 양수"
        raise ValueError(f"{name}은 {requirement}이어야 합니다.")
    return value


def build_world(*, parts=None, timestep=None, gravity=None, integrator=None):
    """모듈 build → compile → bind 후 초기 상태까지 준비한다.

    인자를 생략하면 config의 계약값을 사용한다. 아직 None인 값은 임의로
    확정하지 않고 오류를 낸다. 실험 시에만 명시적 인자로 임시값을 전달한다.
    gravity=0은 중력 없는 독립 힘 실험에 사용할 수 있다 (공통 config는 유지).
    parts 생략 시 물·로봇·줄·사람의 현재 구현을 조립한다.
    """
    timestep = _positive(
        config.Sim.TIMESTEP if timestep is None else timestep, "Sim.TIMESTEP"
    )
    gravity = _positive(
        config.Physics.GRAVITY if gravity is None else gravity, "Physics.GRAVITY", allow_zero=True
    )
    integrator = config.Sim.INTEGRATOR if integrator is None else integrator
    if integrator not in ("Euler", "RK4", "implicit", "implicitfast"):
        raise ValueError("Sim.INTEGRATOR에 Euler/RK4/implicit/implicitfast를 명시하세요.")
    if config.Sim.BUILTIN_FLUID_OFF is not True:
        raise ValueError("중복 저항을 막으려면 Sim.BUILTIN_FLUID_OFF=True여야 합니다.")

    # import 스모크 테스트는 MuJoCo가 없는 환경에서도 실행할 수 있다.
    import mujoco

    if parts is None:
        from sim.water import Water
        from sim.robot import Robot
        from sim.tether import Tether
        from sim.human import Human
        from sim.lifebuoy import Lifebuoy

        # 실제 로봇 연결점·연결 시점 합의 전에는 기본 세계에 줄을 생성하지 않는다.
        parts = (Water(), Robot(), Human(), Lifebuoy(), Tether(enabled=False))
    parts = tuple(parts)
    for part in parts:
        max_dt = part.REQUIRES.get("max_dt")
        if max_dt is not None and timestep > _positive(max_dt, "REQUIRES.max_dt"):
            raise ValueError(f"{type(part).__name__}: timestep이 max_dt={max_dt}를 초과합니다.")

    spec = mujoco.MjSpec()
    spec.modelname = "TetherRescue"
    integrator_id = getattr(mujoco.mjtIntegrator, f"mjINT_{integrator.upper()}")
    spec.option.timestep = timestep
    spec.option.gravity[:] = (0.0, 0.0, -gravity)
    spec.option.integrator = integrator_id
    spec.option.density = 0.0
    spec.option.viscosity = 0.0

    for part in parts:
        part.build(spec)

    # 모듈이 전역 옵션을 바꿨다면 조립 단계에서 바로 드러나게 한다.
    if (
        spec.option.timestep != timestep
        or tuple(spec.option.gravity) != (0.0, 0.0, -gravity)
        or spec.option.integrator != integrator_id
        or spec.option.density != 0.0
        or spec.option.viscosity != 0.0
    ):
        raise ValueError("모듈의 build()가 world.py 소유의 전역 물리 설정을 변경했습니다.")

    model = spec.compile()
    for part in parts:
        part.bind(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return World(spec, model, data, parts, ForceAccumulator())
