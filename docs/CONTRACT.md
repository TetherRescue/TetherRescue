# 🔒 계약 (CONTRACT)

노션 "06. 모듈 통합 규칙 (계약)"의 확정본입니다.

- 담당: 섹션마다 그 값에 걸린 담당자가 정하고 씁니다 (물·로봇 섹션: 동경·미현).
- 통합 충돌 확인: 가영 — 이름 겹침, 공통 값(timestep·그룹)이 모든 모듈과 맞는지.
- 바꿀 때는 PR에 가영 + 그 섹션 담당자를 태그합니다.

<!-- TODO(섹션 담당): 이름 · 좌표계 · 단위 · 물리량 소유권 · 외력 주입 · 전역 설정 · Geom 그룹 -->

## 외력 주입 — ForceAccumulator 구현 규칙 (가영, 팀 리뷰용)

- 각 모듈은 `acc.add(body_id, force_w, torque_w=None, point_w=None)`로 외력을 전달합니다.
- 힘은 N, 토크는 N·m, 작용점은 m이며 모두 월드 좌표입니다. 몸체 ID는 월드(0)를 제외합니다.
- `point_w`가 없으면 힘은 몸체 무게중심에 작용합니다. 있으면
  `토크 = torque_w + (point_w - data.xipos[body_id]) × force_w`로 합산합니다.
- 한 스텝 순서: `reset → 모듈별 add → apply → mj_step`.
- 첫 스텝 및 자세를 직접 바꾼 뒤에는 `mj_forward` 등으로 현재 무게중심 위치를 갱신합니다.
  모듈이 위치 기반 힘을 계산하기 전에도 파생 위치가 현재 `qpos`와 일치해야 합니다.
  조립 테스트 루프는 매 스텝 시작에 `mj_forward`를 호출해 이를 보장합니다.
- `reset()`은 누적 기록만 비웁니다. `apply()`가 `xfrc_applied`를 현재 합산값으로 교체하고
  `qfrc_applied`를 0으로 비웁니다. 힘이 없으면 두 배열 모두 0이 됩니다.
- 두 외력 배열에는 `sim/forces.py`만 씁니다. 반복 `apply()`는 힘을 중복 누적하지 않습니다.
- MuJoCo 내장 중력·접촉·tendon 스프링 힘은 엔진이 처리하므로 같은 힘을 다시 `add()`하지 않습니다.

## 세계 조립 — 구현 규칙 (가영, 팀 리뷰용)

- `build_world()`는 모든 모듈의 `build(spec)` → `spec.compile()` → 모든 모듈의
  `bind(model)` 순서로 조립합니다. 이후 `MjData`를 만들고 `mj_forward`로 초기 상태를 갱신합니다.
- 반환된 `World`의 `model`, `data`, `parts`, `acc`를 단계 연결 담당자가 사용합니다.
  스텝 순서와 단계 전환 구현은 `phase_manager.py`의 담당 범위입니다.
- 기본 모듈 순서는 물 → 로봇 → 줄 → 사람입니다. 조립 테스트는 `parts` 인자로 대체 모듈을 넣을 수 있습니다.
- timestep·중력·integrator·내장 유체 옵션은 `world.py`에서만 설정합니다.
  모듈이 이를 변경하거나 timestep이 모듈의 `REQUIRES['max_dt']`보다 크면 조립을 중단합니다.
- 미확정 config 값은 자동으로 채우지 않습니다. 실험용 인자를 명시해서 실행하며 계약 확정값과 구분합니다.
- MjSpec 사용법: [MuJoCo 공식 Python 문서](https://mujoco.readthedocs.io/en/stable/python.html#model-editing).

## 사람·줄 설정 — 가영의 독립 구현용 초안 (팀 합의 전)

이 절은 위 문서의 확정 규칙과 구분되는 제안입니다. `human.py`에는 단순 사람 강체를 구현했고,
`tether.py`는 아직 빈 모듈입니다. 이름은 공유 예정이며 수치는 실측·문헌 근거 없이
단순 모델 구현을 시작하기 위해 정한 시험값입니다. 사람 강체의 크기·질량·자유낙하는 검사했으며,
실제 인체의 물리적 타당성과 물·로봇·줄 통합 안정성은 아직 검증하지 않았습니다.

| 구분 | 설정 | 초안과 의미 |
| --- | --- | --- |
| 사람 이름 | `Names.HUMAN_BODY/FREE/GEOM/TOW_SITE` | `human` / `human_free` / `human_geom` / `human_tow_point` |
| 줄 이름 | `Names.TETHER_TENDON` | `tow_tendon` |
| 사람 형태 | `Human.SHAPE` | 관절 없는 상자 하나 (`box`), 머리·팔다리·기도 모델은 후속 작업 |
| 전체 크기 | `Human.LENGTH/WIDTH/HEIGHT` | 1.7 × 0.4 × 0.2 m, MuJoCo box에 전달할 때는 각각 절반 |
| 질량·무게중심 | `Human.MASS/COM` | 70 kg, 상자 중심과 일치하는 단순화 |
| 사람 로컬 축 | 구현 제안 | +x 머리 쪽, +y 왼쪽, +z 몸의 앞면. 초기 월드 자세는 미정 |
| 줄 기준 길이 | `Tether.REST_LENGTH` | 2 m, 초기 두 연결점 거리와는 별개 |
| 강성·감쇠 | `Tether.STIFFNESS/DAMPING` | 100 N/m, 10 N·s/m, 검증 전 시험값 |

미확정 값은 `None`으로 유지합니다. 후속 구현에서 필요한 값은 명시적으로 검사하며,
0이나 False로 자동 해석하지 않습니다.

- `Human.TOW_POINT_B`: 사람 몸에서 줄을 묶는 위치 — 동경·미현과 확인.
- `Tether.ATTACH_ON_START`: 연결 시점 — 동경·태수와 확인.
- `Tether.BREAK_FORCE`: 끊김 모델 도입 여부와 기준 — 후속 범위 결정 필요.
- 사람의 부력·물 저항 계산 담당과 인계 방식 — 미현과 확인.
- `Groups` 번호 및 `contype`·`conaffinity` 충돌 규칙 — 공통 합의가 필요해 이번에는 유지.
- 로봇 쪽 연결점은 기존 합의 이름 `Names.TOW_POINT_SITE` (`tow_point`)를 사용하며 위치는 로봇 담당과 확인.

현재 사람 구현은 자유 관절과 상자 geom 한 개로 구성합니다. 질량·무게중심·관성은 body에서 지정하고,
geom에 질량을 중복 추가하지 않습니다. 관성은 균일 상자 공식으로 근사합니다.
`Human(position=(0, 0, 1))`의 기본 배치는 독립 시험용이며 실제 구조 장면의 배치 확정이 아닙니다.
충돌 합의 전에는 사람 geom의 `contype=0`, `conaffinity=0`으로 접촉을 끕니다.
그룹이 미정이면 로컬 시험에서만 그룹 0을 사용하며 `config.Groups`의 합의값은 변경하지 않습니다.
`bind()`는 몸체·자유 관절·geom의 ID를 저장하고, `add_forces()`는 현재 아무 외력도 추가하지 않습니다.
사람의 부력·저항·줄 연결점은 후속 작업입니다.
