# TetherRescue

수상 구조 로봇이 익수자에게 **접근 → 접촉 → 줄로 견인**하는 과정을 MuJoCo 물리 시뮬레이션으로 만듭니다.

## 팀 (1주차, 기록용)

| # | 작업 단위 | 담당 | 파일 | 브랜치 |
| --- | --- | --- | --- | --- |
| 1 | 물 | 미현 | `sim/water/water.py` | `dev/mihyun` |
| 2 | 로봇 | 동경 (팀장) | `sim/robot/robot.py` | `dev/dongkyeong` |
| 3 | 세계 조립 · 통합 충돌 확인 | 가영 | `sim/world/world.py` `config.py` `forces.py` `human.py` `tether.py` | `dev/kayoung` |
| 4 | 검증 | 채연 | `tests/` `docs/` | `dev/chaeyeon` |
| 5 | 단계 연결 · CI | 태수 | `sim/robot/phase_manager.py` `watchdog.py` | `dev/taesu` |

```
계약 합의 → 1 물 ∥ 2 로봇 → 3 세계 조립 → 5 단계 연결
             ↺ 4 검증은 1·2·3이 바뀔 때마다
```

## 구조

2주차부터 `sim/`과 `tests/`를 **물 · 로봇 · 세계 조립** 세 분야로 나눴습니다.

| 분야 | 담당 | 코드 | 테스트 |
| --- | --- | --- | --- |
| 물 | 미현 · 채연 | `sim/water/` (water, buoyancy) | `tests/water/` |
| 로봇 | 동경 | `sim/robot/` (robot, phase_manager, rescue_approach) | `tests/robot/` |
| 세계 조립 | 가영 · 태수 | `sim/world/` (world, human, lifebuoy, tether, grasp, wear, scenarios, rope_visual, rescue_mission, watchdog, view_*) | `tests/world/` |
| 공통 계약 | 전원 | `sim/config.py` `sim/forces.py` `sim/frames.py` | `tests/` 바로 아래 (smoke, forces, frames, integration, bench_speed) |

```
sim/
  config.py forces.py frames.py   계약 파일 (바꾸면 가영·영향 담당자 태그 + 카톡 공지)
  water/   물
  robot/   로봇
  world/   세계 조립, 미션, 시연 화면(view_*.py)
tests/
  water/ robot/ world/            분야별 테스트
docs/     CONTRACT.md(계약), PARAMETERS.md(파라미터 근거)
```

import는 전체 경로로 씁니다. 예: `from sim.water.water import Water`, `from sim.robot.robot import Robot`,
`from sim.world.world import build_world`. 시연은 `python -m sim.world.view_rescue`처럼 실행합니다.

## 실행

설치와 실행은 [HOW_TO_RUN.md](HOW_TO_RUN.md)를 보세요.
