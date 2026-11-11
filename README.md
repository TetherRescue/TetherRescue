# TetherRescue

수상 구조 로봇이 익수자에게 **접근 → 접촉 → 줄로 견인**하는 과정을 MuJoCo 물리 시뮬레이션으로 만듭니다.

## 팀 (1주차)

| # | 작업 단위 | 담당 | 파일 | 브랜치 |
| --- | --- | --- | --- | --- |
| 1 | 물 | 미현 | `sim/water.py` | `dev/mihyun` |
| 2 | 로봇 | 동경 (팀장) | `sim/robot.py` | `dev/dongkyeong` |
| 3 | 세계 조립 · 계약 | 가영 | `sim/world.py` `config.py` `forces.py` `human.py` `tether.py` | `dev/kayoung` |
| 4 | 검증 | 채연 | `tests/` `docs/` | `dev/chaeyeon` |
| 5 | 단계 연결 · CI | 태수 | `sim/phase_manager.py` `watchdog.py` | `dev/taesu` |

```
계약 합의 → 1 물 ∥ 2 로봇 → 3 세계 조립 → 5 단계 연결
             ↺ 4 검증은 1·2·3이 바뀔 때마다
```

## 구조

```
sim/      시뮬레이터 모듈 (config·forces는 계약 파일)
tests/    단위·통합 테스트, 속도 측정
docs/     CONTRACT.md(계약), PARAMETERS.md(파라미터 근거)
```

## 실행

설치와 실행은 [HOW_TO_RUN.md](HOW_TO_RUN.md)를 보세요.
