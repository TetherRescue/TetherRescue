# 실행 방법

사람·구명환·로프 연결 담당의 독립 검증은 아래 **사람이 구명환을 받고 놓는 시험**을 사용합니다.
팔로 환을 가져와 머리 쪽에서 몸통에 끼는 동작은 **사람이 스스로 구명환을 끼는 시험**에서 확인합니다.
기존 배 통합 시연은 사람에게 구명환을 고정한 견인 근사입니다. 배의 탑재·전달 구현과
새 잡기 연결을 합치기 전까지의 참고 시연이며, 이번 작업에서는 배 쪽 코드를 변경하지 않습니다.

자세한 내용은 노션 "03. 개발 환경 & 실행 가이드"를 기준으로 합니다.

## 전체 시나리오 (메인 실행)

지나가던 사람이 익수자를 발견하고, 배를 찾아, 물가에 밀어 띄운 뒤 출항합니다.
하나개·월미도 둘 다 같은 네 박자(걷기, 발견, 배로 달리기, 밀어 띄우기)입니다.
설정 창의 출항을 '생략'으로 두면 그 동작을 건너뛰고, 배가 이미 바다에 떠 있는 상태에서 시작합니다.
GPS 수동 입력은 아직 없고, 띄운 뒤 익수자 좌표를 자동으로 찍어 그쪽으로 항해합니다.

```bash
conda activate rescue
python -m sim.run_sim                 # 시뮬레이션 창 + 설정 창
python -m sim.run_sim --map wolmido --waves --tide low --weather rain
# 창 없이 로그만: python -m sim.run_sim --headless
```

설정 창 위쪽 슬라이더로 파도 높이(m)와 배 속도(m/s)를 바꿉니다. 날씨(맑음·비·안개)도 그 창에서 고르며, 이 세 가지는 시뮬 창을 다시 열지 않습니다.
이름에 마우스를 약 1초 올려 두면 한글 설명이 나옵니다.
물때·날씨는 월미도를 고른 뒤에만 나타납니다. 시뮬레이션 창 왼쪽 메뉴는 엔진 설정입니다.
값을 바꾸면 시뮬레이션 창만 다시 열립니다.

## 설치 (처음 한 번)

```bash
sudo apt install python3.12-venv -y
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# torch 는 CPU 빌드를 먼저 (그냥 설치하면 GPU 용 CUDA 패키지를 수 GB 받음)
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

ROS2와 같이 쓸 때:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -c "import rclpy, mujoco; print('ok')"
```

## 실행

| 무엇 | 명령 |
| --- | --- |
| 전체 테스트 | `pytest -q` |
| 통합 테스트 | `pytest tests/test_integration.py -q` |
| 속도 측정 | `python tests/bench_speed.py` |
| 화면 없이 렌더 | 명령 앞에 `MUJOCO_GL=egl` |

## 세계 조립 뼈대 확인 (가영)

MuJoCo가 설치된 가상환경에서 실행합니다. 화면을 띄우지 않으므로 렌더러 설정은 필요 없습니다.

```bash
source .venv/bin/activate
python -m pytest tests/world/test_world.py tests/test_forces.py -q
```

조립 함수만 사용하려면:

```python
from sim.world.world import build_world

# 회의에서 확정한 config 값(dt=0.002초, 중력=9.81, implicitfast)을 사용합니다.
world = build_world()
print(world.model.nbody, world.data.time)
```

현재 기본 조립에는 몸통·머리·양팔·양다리를 고정한 사람과 구명환이 포함됩니다.
물 모듈은 선체 복셀 부력·이동/회전 저항을 적용하고, 로봇은 계약 선체·좌우 추진기를 조립합니다.
사람·구명환의 부력은 아래 수상 견인 데모에서 명시적으로 활성화합니다.
기본 세계의 로프는 명시적으로 미연결이며 아래 독립 데모에서 시험합니다.
테스트는 빈 세계, 움직이는 시험용 몸체, 사람의 자유낙하에서 각각 1,000스텝을 확인합니다.
인체·줄·부력 모델의 실제 구조 상황에 대한 물리 정확성은 후속 검증 대상입니다.
설정 인자를 생략하면 `config.py`의 값을 사용하고, 미확정(`None`)이면 오류로 알립니다.

기존 상자 시험 모드로 크기·질량을 확인하려면:

```python
from sim.world.human import Human
from sim.world.world import build_world

human = Human(position=(0, 0, 1), shape="box")  # 기본 형상은 humanoid
world = build_world(parts=[human])
print(world.model.body_mass[human.body_id])  # 70kg
print(world.model.geom_size[human.geom_id])  # 반길이: [0.85, 0.2, 0.1]
```

위 사람 단독 예제에서는 자유 관절로 이동·회전하며, 충돌과 추가 외력은 없습니다.
스텝을 진행하면 중력으로 떨어집니다. 줄 연결점은 위치 합의 전이라 생성하지 않습니다.

## 사람 움직임 보기

데스크톱 또는 WSLg처럼 GUI 창을 열 수 있는 환경에서 실행합니다.

```bash
source .venv/bin/activate
python -m sim.world.view_human
```

어깨·허리 윤곽이 있는 몸통, 타원형 머리와 얼굴, 팔꿈치·무릎·손발을 가진 사람과 주황·흰색 구명환이 함께 떨어집니다.
파란 점은 구명환의 로프 연결점입니다. 양팔은 환보다 머리 쪽에서 바깥으로 뻗습니다.
약 1초마다 시작 위치로 돌아와 반복합니다. 모든 부위와 구명환은 고정 연결되어 있습니다.
이 자유낙하 데모에서는 부력을 적용하지 않습니다. 팔 관절 운동과 접촉으로 몸을 지지하는 기능은 아직 없습니다.
창을 닫으면 종료합니다. 이는 독립 데모의 반복 동작이며 실제 구조 시나리오의 리셋 정책이 아닙니다.
렌더러가 EGL로 설정되어 창이 열리지 않으면 `MUJOCO_GL=glfw python -m sim.world.view_human`을 사용합니다.
화면이 없는 환경에서는 `python -m sim.world.view_human --headless`로 데모의 계산만 확인할 수 있습니다.
구명환은 사람 몸에 끼우지 않으므로 이 화면에는 사람만 나옵니다.
몸집은 `--size small|medium|large`로 고릅니다 (기본 medium). 예: `python -m sim.world.view_human --size large`.
구명환 검증만 실행하려면 `python -m pytest tests/world/test_lifebuoy.py -v`를 사용합니다.

## 로프 견인 시험 보기

> 이 시험은 사람이 구명환을 이미 왼손에 쥐었다고 가정하고 견인 단계만 떼어 본 것입니다. 처음부터 몸에 끼운 상태로는 시작하지 않습니다. 실제 흐름은 배가 줄+구명환을 건네주고 사람이 손으로 잡거나, 잡은 뒤 스스로 착용(`view_wear`)한 다음 견인합니다.

```bash
source .venv/bin/activate
python -m sim.world.view_tether
```

기본 시연은 중력을 끈 로프 단독 비교 실험입니다. 공유 config의 중력값은 바꾸지 않습니다.
화면의 두 끝점 거리(Endpoint gap)·로프 길이(Rope length)·기준 길이(Rest)·
남는 길이(Slack)·늘어난 길이·장력·사람 이동 거리와 다음 세 단계를 비교하세요.

1. 2초: 느슨한 줄(1.5m < 기준 2m), 장력 0N이며 사람이 정지합니다.
2. 1초: 실험 초기 배치를 리셋해 줄을 3m로 늘리고 잠시 정지 표시합니다 (장력 100N).
3. 4초: 일시정지를 풀면 로프 힘으로 사람·구명환이 이동합니다. 이후 반복합니다.

두 번째 단계의 위치 점프는 실험 배치 리셋이며 로프가 만든 움직임이 아닙니다.
회색 줄은 느슨함, 주황색 줄은 인장을 나타냅니다. 빨간 화살표는 시각화용 크기의 장력 방향이고,
파란 바닥 선은 해당 실험에서 사람의 시작 위치입니다. 바닥은 시각용이며 접촉력이 없습니다.
HUD는 MuJoCo 글꼴 호환을 위해 영문으로 표시합니다. 부력이나 실제 수평 해상 견인을 재현하지는 않습니다.
끝점이 가까워져도 로프가 기준 길이보다 짧게 표시되지 않습니다. 두 끝점 거리가 1.5m이고
기준 길이가 2m라면, 2m 길이의 회색 S 곡선으로 남는 0.5m를 표현합니다.
곡선은 표시하는 64개 구간 길이의 합을 기준 길이에 맞춥니다.
두 끝점 거리가 기준 길이 이상이면 탄성으로 늘어난 직선으로 표시합니다.
이 곡선은 물리적 처짐·접촉을 계산하는 로프가 아니라 화면용 중심선입니다.
기존 중력 낙하 시연은 `python -m sim.world.view_tether --pendulum`로 실행합니다.

```bash
python -m sim.world.view_tether --headless
python -m pytest tests/world/test_tether.py -v
```

단위 검사는 느슨함·늘어남·빠른 단축·양쪽 작용반작용을 엔진의 실제 가속도로 확인합니다.
통합 검사는 리셋 없이 사람·구명환 견인을 연속 1,000스텝 실행합니다.
추가 비교 검사는 느슨한 상태의 정지와 늘어난 로프를 놓은 뒤 이동을 확인합니다.
`python -m pytest tests/world/test_rope_visual.py -v`는 끝점 일치·표시 길이 보존·실제 표시 구간 길이를 검증합니다.

## 물 위에서 배가 사람을 견인하는 시험

> 이 시험은 사람이 구명환을 이미 왼손에 쥐었다고 가정하고 견인 단계만 떼어 본 것입니다. 처음부터 몸에 끼운 상태로는 시작하지 않습니다. 실제 흐름은 배가 줄+구명환을 건네주고 사람이 손으로 잡거나, 잡은 뒤 스스로 착용(`view_wear`)한 다음 견인합니다.

```bash
python -m sim.world.view_towing
python -m sim.world.view_towing --headless
python -m pytest tests/water/test_water_physics.py tests/water/test_buoyancy.py -q
```

기본 시연은 중력·부력·물 저항이 켜진 잔잔한 물의 견인 시험입니다.
최신 계약 선체(A형, 19.6kg, 1.5×1.2×0.2m)·좌우 추진기·로프·구명환·사람(합계 72.5kg)을 조립합니다.
초기 줄은 느슨하며, 전진 명령 0.2로 배가 움직이면 줄이 팽팽해져 사람을 당깁니다.
GUI는 리셋 없이 계속 실행하며 카메라가 배와 사람을 따라갑니다.
headless는 10초(5,000스텝)를 리셋 없이 실행합니다.

선체는 감싸는 box를 복셀로 나눈 뒤 A형 모양 밖 칸을 버리고 잠긴 부피와 부력 중심을 계산합니다.
사람과 고정 구명환은 외형의 합집합을 4cm 이하 복셀로 근사해 겹친 부피를 중복 계산하지 않습니다.
구명환이 상체를 띄우면서 사람은 다리가 물속으로 내려가는 자세로 회전할 수 있습니다.
파란 반투명 평면은 수면 표시이며 접촉력으로 물체를 받치지 않습니다.
물 저항은 물에 대한 상대 속도에 적용하고, 물 밖에서는 부력·저항이 사라집니다.
선체 정수 흘수는 손 계산값 약 0.022m(질량 ÷ (ρ · 수면 넓이 0.87m²))와 비교하며 기울기 복원도 실제 엔진에서 검사합니다.

수상 시연의 선미 연결점은 `Hull.TOW_POINT_B=(0.15, 0, 0.1)`(A형 홈 앞끝 가운데)을 사용합니다.
기본 로봇도 이 연결점을 생성하며, 독립 무중력 시험은 명시적 시험 좌표로 바꿀 수 있습니다.
사람 물성 `config.RescueWater`는 검증 전 시험값이며 실제 인체·구명환의 부유 성능을 보증하지 않습니다.
파도는 높이 함수만 구현되어 있고 수면 표시 변형·물 입자 속도·부가질량은 미구현입니다.
`build_world`가 물과 로봇을 자동으로 연결합니다. 흡입구(`Thruster.INTAKE_*_B`, 선체 바닥 아래 6cm)가 수면 밖에 나온 스텝에는 그쪽 추력이 0이 됩니다. 물이 없거나 둘 이상인 세계에서는 항상 물속으로 봅니다.
실패 감시와 자동 구조 단계는 아래 시연에서 확인합니다.

기존 무중력 힘 전달 시험은 다음 명령으로 실행합니다.

```bash
python -m sim.world.view_towing --dry
python -m sim.world.view_towing --dry --headless
```

이 모드는 초기 로프가 약 2.38m로 늘어난 배치에서 시작하고, 중력·부력·물 저항을 끕니다.
GUI는 2초를 반복하며 headless는 리셋 없이 1,000스텝을 실행합니다.

사람 형상은 단순 캡슐 인형에서 단면 mesh 몸통·분절 팔다리로 다듬었습니다.
반소매 상의·바지·신발과 피부를 구분하고 눈·코·입·귀·머리카락을 추가했습니다.
팔꿈치·무릎은 형상으로 구분되지만 개별 관절 운동은 아직 없습니다.
질량은 분절에 나눠 총 70kg을 유지하며, 얼굴·목 장식은 질량이 없습니다.
형상 변경으로 계산되는 무게중심·관성은 이전 캡슐 모델과 달라집니다.

## 목표 위치에 접근하고 멈추기

```bash
python -m sim.world.view_approach
python -m sim.world.view_approach --headless
python -m sim.world.view_approach --start 6 3 --yaw 180 --target 0 0 --heading 90
python -m pytest tests/robot/test_phase.py -q
```

기본 목표는 월드 `(4, 2)`m, 도착 방향은 +x(0도)입니다.
초록 표시가 목표 좌표이고 초록 화살표가 도착 방향입니다.
`--yaw`는 출발 방향, `--heading`은 도착 방향이며 CLI 단위는 도입니다.
코드의 `PhaseManager`와 `build_approach_demo` 각도 인자는 rad입니다.
이번 시험에는 배·잔잔한 물·목표 표시만 있고 사람·로프는 포함하지 않습니다.

`APPROACH`에서 위치와 수평 속도 오차로 필요한 힘을 구하고 방향을 맞춥니다.
뒤쪽 힘이 필요하면 역추력으로 제동하거나 후진합니다.
목표 5cm 안에 들어오면 `SETTLE`에서 목표 방향 정렬과 저속 제동을 수행합니다.
위치 오차 20cm, 속도 0.05m/s, 방향 오차 8도, yaw 속도 0.03rad/s 이하를
1초 동안 모두 만족하면 `ARRIVED`로 판정합니다.
도착 뒤에도 작은 제동·위치 보정을 계속합니다. 허용 범위를 벗어나면 정렬 또는 재접근합니다.
이 시험의 도착은 정지 유지 상태이며, 모든 추진을 영구적으로 끄는 단계는 아닙니다.

명령은 `Sim.CONTROL_HZ`(기본 50Hz), 물리는 `Sim.TIMESTEP`(0.002초)로 갱신합니다.
`ApproachSettings`의 계수·허용 오차는 현재 선체의 잔잔한 물 시험값입니다.
headless는 기본 60초 안에 도착하고 추가 8초 연속 정지 유지를 확인해야 성공합니다.
필요하면 `--timeout 90`으로 시뮬레이션 제한 시간을 늘릴 수 있습니다.
GUI는 리셋 없이 계속 실행하며 창을 닫으면 종료합니다.
이 단독 데모는 사람·로프를 포함하지 않습니다. 사람 접근·자동 연결·견인은 아래 전체 시나리오에서 실행합니다.
일반 장애물 회피·해류에서의 정밀 위치 유지는 후속 범위입니다.

## 구명환의 연결 위치로 접근하기

```bash
python -m sim.world.view_rescue --approach-only
python -m sim.world.view_rescue --no-ring              # 구명환 없는 사람에게 접근, 건네줄 자리에서 정지
python -m sim.world.view_rescue --approach-only --headless
python -m sim.world.view_rescue --approach-only --start -6 0 --yaw 0
python -m sim.world.view_rescue --approach-only --start 5 6 --yaw 180 --heading 90
python -m pytest tests/robot/test_rescue_approach.py -q
```

배와 실제 부유하는 사람·구명환을 함께 조립하고 **로프 없이** 연결 대기 위치에 접근합니다.
기본 배 출발은 `(6, 3)`m, 사람 초기 위치는 `(0, 0)`m, 최종 견인 방향은 +x입니다.
`--victim X Y`는 사람 몸체의 초기 위치이며 구명환 연결점 위치와는 다릅니다.
CLI 각도는 도, `build_rescue_demo`·`RescueApproach` API 각도는 rad입니다.

1. `TRANSIT`: 구명환 연결점 주위 반지름 4.5m의 바깥 경로를 거쳐 대기 위치로 이동합니다.
   반대편 출발도 사람을 가로지르는 직선 대신 바깥 원호의 경유점을 사용합니다.
2. `ALIGN_STERN`: 선미 연결점이 구명환에서 약 3m 떨어진 위치에서 견인 방향을 맞추고 정지합니다.
3. `BACK_IN`: 최대 목표 속도 0.2m/s로 접근해 연결점 간격 약 0.7m를 맞춥니다.
4. `READY`: 위치·방향·속도 조건과 연결점 간격·상대속도 조건을 만족하면 연결 대기합니다.

목표는 구명환의 현재 site 위치를 따라 갱신합니다. 사람 위치를 고정하거나 옮기지 않습니다.
초록 화살표는 현재 배의 목표 위치·방향이며, 카메라는 배와 구명환을 따라갑니다.
선미 연결점 위치는 `(-Hull.LENGTH/2, 0, -0.08)`인 시험값입니다.
최종 연결점 간격 오차 0.15m, 상대속도 0.06m/s 이하를 도착 판정과 함께 확인합니다.
추가로 1초를 유지해야 `READY`가 됩니다. headless는 그 뒤 5초 연속 대기를 확인합니다.
기본 headless 제한 시간은 240초이며 `--timeout`으로 바꿀 수 있습니다.

이 초기 시험은 배를 실제 구명환 연결점에서 4.5m 이상 떨어뜨려 출발해야 합니다.
선체와 사람·구명환의 실제 형상 간격을 50Hz로 확인하고 0.1m 미만이면 명령을 0으로 바꿔 시험을 중단합니다.
충돌 반응·실제 접촉 판정은 아직 꺼져 있습니다. 이 경로는 현재 형상의 잔잔한 물 시험용이며
일반 장애물 회피 계획기가 아닙니다. `READY`는 로프가 실제 연결됐다는 뜻이 아닙니다.
실제 연결과 견인은 아래 전체 시나리오에서 실행합니다.

## 구명환 발사 구조 시나리오 (2026-11-19, 기본 구조 방식)

```bash
python -m sim.world.view_launch --speed 3
python -m sim.world.view_launch --no-shadow     # 화면이 끊길 때
python -m sim.world.view_launch --headless
python -m sim.world.view_launch --start 5 6 --yaw 180 --heading 90
python -m pytest tests/robot/test_launcher.py -q
```

배 홈 위 발사대에 구명환을 고정해 싣고 출발합니다. 줄은 처음부터 배와 구명환에 연결되어 있고 느슨합니다.
`APPROACH`에서 바깥 경로를 돌아 선미를 사람 쪽으로 두고 줄 묶는 점이 사람 손에서 3 m(`Launcher.STAGING_GAP`)인
자리에 멈춥니다. 후진하지 않고 그 자리에서 `LAUNCH`: 발사대 weld를 풀고 0.05초 동안 밀어 쏩니다(반동은 배가 받음).
동시에 윈치를 발사 거리 + 2 m로 풀어 날아가는 동안 줄이 당기지 않습니다. 구명환이 물에 닿으면 `GRAB`에서
잡는 점이 손 0.45 m 안이고 느리면 사람이 잡은 것으로 보고 `TOW`로 끌고 갑니다.
공기 저항·바람·팔 뻗는 동작은 없습니다. 착수 속력이 약 4.4 m/s라 실제로는 사람 몸에 맞지 않게 겨눠야 합니다.
`--speed`는 화면 배속이며, 계산이 못 따라가면 실제 배속은 그보다 낮습니다. 실제 배속은 화면 왼쪽 위 `now x`에 나옵니다
(그래픽 카드 없는 WSL에서 x1은 실제 1.0배, x3은 실제 약 1.5배).
기본 배치(사람에게서 50m)에서 최고 약 4m/s로 다가와 약 22초에 발사하고 약 52초에 견인을 마칩니다.
추진기 100N·접근 4.5m/s·사람 근처 1.2m/s·견인 0.7m/s는 테스트값입니다(11/23 회의에서 확정).

## 접근 → 로프 연결 → 견인 → 정지 전체 시나리오 (착용 구명환, 이전 방식)

```bash
python -m sim.world.view_rescue
python -m sim.world.view_rescue --headless
python -m sim.world.view_rescue --start 5 6 --yaw 180 --heading 90
python -m sim.world.view_rescue --tow-distance 8
python -m pytest tests/world/test_tether.py tests/world/test_rescue_mission.py -q
```

이제 `view_rescue` 기본 실행은 전체 시나리오입니다. 기존 연결 대기 시험은 `--approach-only`로 선택합니다.
`APPROACH`에서 바깥 경로·선미 정렬·저속 접근을 마치고, `ATTACH`에서 현재 연결 조건을 다시 확인합니다.
조건이 유지되면 미리 생성해 둔 tendon의 스프링 계수를 켜고 `TOW`로 전환합니다.
연결 전에는 스프링·감쇠가 모두 0이며, 연결 시 위치·속도·로프 기준 길이 2m는 바꾸지 않습니다.
약 0.7m 간격에서 연결하므로 처음에는 느슨하고, 배가 전진해 2m 이상 벌어져야 장력이 생깁니다.

배의 목적지는 연결 순간의 선체 원점에서 지정 견인 방향으로 기본 6m 떨어진 위치입니다.
목표 속도는 최대 0.4m/s이고, 끝에서는 제동·방향 정렬로 정지합니다.
견인 중 목표 20cm 안에 들어오면 일찍 정렬해 선미가 크게 회전하는 것을 줄입니다.
배의 도착 조건뿐 아니라 사람의 실제 진행 2m 이상, 수평 속도 0.05m/s 이하,
잔여 장력 10N 이하를 1초 동안 만족해야 `COMPLETE`로 판정합니다.
headless는 완료 후에도 5초 연속 정지 유지를 확인합니다. 기본 제한은 240초입니다.
`--tow-distance`는 배의 이동 거리이며, 로프 길이와 최소 사람 진행 거리 합계인 4m보다 커야 합니다.

HUD에 연결 여부·장력·사람 진행 거리를 표시합니다. 로프는 미연결일 때 숨기고,
연결 후에는 길이를 유지하는 회색 느슨한 곡선 또는 주황 인장선으로 그립니다.
곡선은 화면 표시이며 실제 로프 처짐·접촉을 계산하지 않습니다.
월드 리셋 시 `RescueMission.reset`은 모델에 남아 있는 스프링 계수도 미연결 상태로 복원합니다.

연결은 거리·상대속도·방향 조건에 따른 자동 연결 근사이며 실제 접촉·후크 동작은 미구현입니다.
이번 검증은 현재 선체·인체 형상과 잔잔한 물 조건입니다.
시뮬레이션 시간 초과·수치 오류는 아래 Watchdog에서 처리합니다(전복 판정은 2026-11-19에 없앰). 추론 응답이 없으면 안전 모드로 멈추며, 로프 끊김은 후속 작업입니다.

## 실패 감지와 물리 중단 (Watchdog)

전체 시나리오에는 Watchdog가 기본 적용됩니다.

```bash
python -m sim.world.view_rescue
python -m sim.world.view_rescue --headless
# 명시적으로 오류를 주입해 중단을 확인하는 시연
python -m sim.world.view_rescue --headless --failure-demo timeout
python -m sim.world.view_rescue --headless --failure-demo nonfinite
python -m sim.world.view_rescue --headless --failure-demo warning
python -m sim.world.view_rescue --headless --failure-demo inference
python -m pytest tests/world/test_watchdog.py -q
```

`--failure-demo`는 시험을 위해 오류를 만드는 옵션입니다. 정상 실행에서는 오류를 주입하지 않습니다.
headless 실패 시 `STOPPED: 코드 at 시각 [단계] - 이유`를 출력하고 종료 코드 1을 반환합니다.
GUI는 `FAILED`와 중단 코드·단계·시각을 표시하며 창을 닫을 때까지 실패 장면을 유지합니다.
시간 초과는 실제 대기 시간이 아닌 **시뮬레이션 시간** 기준입니다.
headless의 `--timeout`은 추가 상한입니다. 더 크게 지정해도 Watchdog의 단계·전체 제한을 완화하지 않습니다.
긴 시나리오는 Python API의 `WatchdogSettings`도 함께 조정해야 합니다.

| 조건 | 기본 시험값 / 동작 |
| --- | --- |
| 접근 단계 제한 | 150초 (경유점·선미 정렬·후진 접근 포함) |
| 연결 단계 제한 | 5초 |
| 견인 단계 제한 | 100초 |
| 전체 실행 제한 | 240초; 완료 뒤 정지 유지에는 적용하지 않음 |
| 비정상 수치 | 위치·속도·가속도·외력 등에 NaN/무한대가 있거나 선체 quaternion이 무효 |
| MuJoCo 경고 | 누적 warning이 하나라도 있으면 중단 |
| 추론 무응답 | 실제 시간 0.5초 안에 응답이 없으면 안전 모드. 시뮬레이션 시간과 별개 |

이 값들은 `WatchdogSettings`의 시연용 기준입니다. 배가 크게 기울거나 뒤집혀도 자세만으로는 멈추지 않습니다.
필요하면 Python API로 기준을 전달할 수 있습니다.

```python
from sim.world.view_rescue import build_rescue_mission
from sim.world.watchdog import WatchdogSettings

mission = build_rescue_mission(
    watchdog_settings=WatchdogSettings(approach_timeout=180, tow_timeout=120),
)
```

Watchdog는 물리 진행 전후에 검사합니다. 스텝 시작 시 이미 존재하는 NaN은 MuJoCo에 넘기기 전에 감지합니다.
계산 도중 발생한 수치 오류·경고는 스텝 직후 검사해 추가 진행을 중단합니다.
실패하면 추진 명령과 누적·적용 외력을 0으로 만들고 이후 `step()`은 `False`를 반환하며
물리 시각·위치·속도를 더 진행시키지 않습니다. 이는 추진을 끈 뒤 관성으로 흘러가는 시뮬레이션과 다릅니다.
로프 연결 상태는 유지하고, 실패 원인과 단계·시각도 최초 기록을 유지합니다.
기존 형상 간격 검사에서 발생한 중단도 `CONTROL_ERROR`로 같은 실패 상태에 연결합니다.

수치 오류가 난 실제 물리 상태는 렌더러에 전달하지 않습니다.
GUI는 유효한 별도 표시 데이터를 사용하고, 실패 후 수치 표시는 마지막 정상 기록을 유지합니다.
시간 초과처럼 자세가 유효한 실패는 그 시점 자세를 표시합니다.
실패 상태는 월드 리셋만으로 자동 해제하지 않습니다. 재실행하려면 명시적으로 둘 다 초기화하세요.

```python
import mujoco
mujoco.mj_resetData(mission.world.model, mission.world.data)
mission.reset()
```

`--approach-only`는 기존 독립 접근 시험이므로 전체 시나리오 Watchdog와 오류 주입 옵션을 사용하지 않습니다.
로프 끊김과 고장 복구 정책은 아직 구현하지 않았습니다.

추론 감시만 실제 시간을 봅니다. 요청이 없으면 시계를 읽지 않으므로 기존 규칙 제어는 그대로 진행합니다.
응답이 제한 시간(기본 0.5초, `InferenceWatchdogSettings`의 시연값)을 넘기거나 오지 않으면
`INFERENCE_TIMEOUT`으로 고정하고 추진 명령과 외력을 0으로 둡니다. 로프는 끊지 않습니다.
재요청은 처음 요청 시각을 연장하지 않으며, 해제는 명시적 `reset`만 가능합니다.

```python
mission.request_inference()          # 정책에 추론을 보낸 시각
accepted = mission.receive_inference()  # 제시간이면 True. 늦으면 안전 모드로 멈추고 False
```

## 사람이 구명환을 받고 놓는 시험 (배 없이)

```bash
source .venv/bin/activate
python -m sim.world.view_grasp
python -m sim.world.view_grasp --headless
# 전달 신호가 없으면 로프가 사람을 직접 당기지 않는 비교
python -m sim.world.view_grasp --no-grab
python -m sim.world.view_grasp --headless --no-grab
# 중력·부력·물 저항을 켠 연결 안정성 확인
python -m sim.world.view_grasp --water
python -m sim.world.view_grasp --headless --water
python -m pytest tests/world/test_grasp.py -q
```

이번 범위는 `로프 → 구명환 → 손의 잡기 연결 → 사람`의 힘 전달입니다.
배 탑재·운항·구명환 전달 동작은 구현하지 않습니다. 파란 10kg 시험 질점이 배 쪽 연결점을 대신하고,
외력 제어로 움직입니다. 사람과 구명환의 위치·속도를 직접 수정해서 견인을 만들지 않습니다.

새 시험은 `Human(grip_sites=True)`와 `Lifebuoy(attached=False, grip_sites=True)`로
각각 독립 자유 강체를 만들며 처음부터 잡기 제약이 꺼져 있습니다.
사람은 구명환을 끼지 않고 환은 왼손 옆에 배치합니다. 이 초기 배치는 **전달 가능한 시험 조건**이며
배에서 손까지 구명환을 운반한 결과를 재현하는 장면은 아닙니다.
로프는 처음부터 시험 연결점과 구명환에 연결되어 있습니다.

기본 중력 없는 시연은 다음을 12초 동안 보여줍니다.

1. 0~2초: 독립 상태로 대기합니다. 녹색 점은 손과 구명환의 잡기 기준점입니다.
2. 2초: 시험용 전달 완료 신호를 입력하고 잡기 조건을 확인합니다.
3. 2~8초: 시험 연결점이 이동해 로프가 팽팽해지고 사람·구명환이 함께 움직입니다.
4. 8초: 놓기 신호로 잡기 제약을 해제합니다. 이후 사람은 관성으로 계속 움직이고 로프는 환만 당깁니다.

GUI는 12초 뒤 비교를 위해 초기화하여 반복합니다. headless는 리셋 없이 한 번 연속 실행합니다.
`--no-grab`은 전달 완료 신호를 보내지 않습니다. 로프에 장력이 생겨 환이 움직여도 사람은 정지해야 합니다.
중력 없는 비교로 연결의 힘 전달만 분리해서 확인하는 시험입니다.

잡기 조건은 전달 완료 신호, 손·환 site 간격 8cm 이하, 상대 선속도 0.1m/s 이하,
상대 각속도 0.5rad/s 이하입니다. `GraspSettings`의 검증 전 시험값입니다.
조건을 만족하면 그 순간의 상대 자세를 유지하는 weld 제약을 켭니다.
잡기·놓기 함수는 qpos·qvel·시간을 바꾸지 않지만, 연결 이후에는 물리 제약력이 작용합니다.
제약은 수치적으로 부드러워 하중 아래에서 약간의 상대 위치·각도 오차가 허용됩니다.
손가락·팔 관절의 실제 잡기·접촉 지지·구명환에 몸을 넣는 동작은 아직 구현하지 않았습니다.

`--water`는 손 옆 배치에서 t=0에 전달 완료를 입력해 잡은 뒤 6초에 놓습니다.
사람·환 각각의 부력·저항과 중력을 켜며, 배 모델 없이 `Water(include_boat=False)`를 사용합니다.
이 시험은 연결 안정성 검사이며 사람이 항상 수면에 떠 있거나 호흡을 유지한다는 검증이 아닙니다.
구명환을 놓은 사람의 부유 성능은 별도 물성 검증 대상입니다.

### 배 쪽 구현과 연결하는 방법

동경님 쪽에서는 배의 탑재 고정을 해제하고 실제 구명환을 수령 가능한 상태로 전달한 뒤
`delivery_ready=True`를 보냅니다. 이 신호가 있어도 잡기 모듈은 거리·속도를 직접 확인하며
멀거나 빠르면 연결하지 않습니다. `try_grab`이 True를 반환하면 전달 장치의 환 위치 제어를 끝내고
로프를 통한 견인으로 넘깁니다. 잡기 모듈이 배의 탑재 고정이나 전달 장치를 직접 조작하지는 않습니다.

```python
import mujoco
from sim.world.view_grasp import GraspDemo

demo = GraspDemo()  # 배 없는 시험 장면; 실제 통합에서는 같은 HumanGrasp를 조립
world = demo.world
mujoco.mj_forward(world.model, world.data)
received = demo.grasp.try_grab(world.model, world.data, delivery_ready=True)
# 놓을 때: 로프를 끊는 것이 아니라 사람↔환 잡기 제약만 해제
released = demo.grasp.release(world.model, world.data)
mujoco.mj_forward(world.model, world.data)
```

조립 순서는 사람·독립 구명환의 site 생성 뒤 `HumanGrasp.build`, compile, bind입니다.
새 에피소드에서는 `mj_resetData` 뒤 `HumanGrasp.reset(model, data)`로 참조 자세도 초기화합니다.
처음부터 끼운 착용(`attached=True`)은 쓰지 않습니다. 전달 흐름에서는 `Lifebuoy(grip_sites=True)`를,
처음부터 쥔 시험은 `Lifebuoy(held_by=human)` + `HumanGrasp(held=True)`를 씁니다. 기존 `view_rescue`의 배 제어·Watchdog에는 아직 합치지 않았습니다.

## 사람이 스스로 구명환을 끼는 시험

```bash
source .venv/bin/activate
python -m sim.world.view_wear
python -m sim.world.view_wear --headless
python -m pytest tests/world/test_wear.py -q
```

> 2026-11-19에 삭제했다가 2026-11-23 되살렸습니다. 처음부터 끼운 상태로 시작하지 않고, 구명환을 받은 뒤 사람이 스스로 끼는 흐름입니다.

구명환을 정면에 전달받은 상태에서 약 14초 동안 **양팔을 곧게 뻗어 동시에 통과**시킵니다.
배의 탑재·전달 동작은 포함하지 않는 **중력 0의 관절·접근 동작 시험**입니다.

1. `WAITING`: 사람과 구명환은 독립 강체이며, 로프는 환에 연결되어 있습니다.
2. `ALIGN_ARMS`: 전달 완료 신호를 받으면 양어깨를 동시에 돌려 팔을 나란히 뻗습니다. 팔꿈치와 손목의 목표 각도는 계속 0도입니다.
3. `INSERT_BOTH`: 환을 전달 위치에 지지하고 사람이 앞으로 접근합니다. 양손과 팔, 머리가 환의 구멍을 지나 몸통에 환이 도달합니다.
4. `WORN`: 실제 양팔 자세와 환의 정렬을 확인하고 몸통 착용 지지를 켭니다. 접근 시험 장치의 지지를 모두 해제하며, 팔은 계속 뻗어 있습니다.

`WearApproachFixture`는 시연 전용 외력 제어 장치입니다. 환을 건네는 위치에 지지하고 사람의 접근 이동을 재현합니다.
배 제어나 실제 수영 추진력 모델이 아니며, 실제 통합에서는 전달부·사람 이동 제어로 대체해야 합니다.
착용 후 힘은 몸통↔환의 weld 지지 근사로 전달됩니다. 겨드랑이 접촉·마찰 전체를 재현하는 모델은 아닙니다.
`SelfWear.measurements(data)['ready_to_tow']`가 True일 때 견인을 시작하세요.
기본(medium) 인체 형상과 정면 전달 배치에 맞춘 순서 제어이며, 수상 착용·다른 몸집·임의 자세의 충돌 회피는 검증하지 않았습니다.
headless는 50초 안에 착용이 완료되어야 성공합니다.

## 월미도 해안 구조 시나리오 (환경 3개)

```bash
source .venv/bin/activate
python -m sim.world.view_wolmido                    # calm: 만조·잔잔
python -m sim.world.view_wolmido --env waves        # 만조·파도
python -m sim.world.view_wolmido --env low_tide     # 간조: 해변·바위가 드러남
python -m sim.world.view_wolmido --env calm --headless
python -m pytest tests/world/test_map_wolmido.py -q
```

맵에는 방파제·산책로(가로등·벤치·구명환 보관함·대관람차), 경사 사석과 바위, 테트라포드 구간, 계단식 친수 호안,
여객선 선착장(폰툰·말뚝·연결 다리·묶인 배)이 있습니다.
인천 월미도 앞바다에서 사람이 떠내려간 상황입니다. 사람은 바다 한가운데(−38, 4)m에 구명환 없이 떠 있습니다.
구조 로봇은 구명환을 발사대에 싣고 해변 물가에서 출발해, 사람 둘레를 돌아 멈춘 뒤 구명환을 쏩니다(동경 발사 시나리오).
사람이 잡으면 해변 쪽으로 최대 8m 끌고 와 멈춥니다. 세 환경 모두 완료됩니다 (허우적대는 사람 기준 시뮬레이션 시간: calm 45초, waves 48초, low_tide 39초).
화면은 기본 1.5배속입니다. 계산이 못 따라가면 실제 배속은 더 낮고, 화면 왼쪽 위 `now x`에 나옵니다. `--speed 1`로 실제 속도, `--speed 4`로 더 빠르게 볼 수 있습니다.

## 배가 물에 뜬 사람에 부딪히는 충돌 시험

> 접근 단계처럼 사람은 구명환이 없습니다. 실제 흐름에서는 배가 닿지 않고 멈춰야 하며, 이 장면은 그 제어가 실패한 최악 상황입니다.

```bash
source .venv/bin/activate
python -m sim.world.view_contact                     # 발 쪽에서 1 m/s로 충돌
python -m sim.world.view_contact --side side         # 옆(구명환 쪽)에서 충돌. head는 머리 쪽
python -m sim.world.view_contact --speed 0.3         # 충돌 속도 바꾸기
python -m sim.world.view_contact --side side --push  # 정지 상태에서 최대 추력으로 계속 밀기
```

시작할 때 사람이 물에 자리 잡는 계산(약 5초)을 먼저 합니다. 자리 잡은 사람은 거의 수평으로 누워
얼굴 앞면이 수면 위로 조금 나온 자세라, 배는 수면 근처의 머리·몸통·다리에 닿습니다.
빨간 화살표는 접촉력, 작은 표시는 접촉점입니다. HUD에서 접촉 수, 현재/최대 겹침(mm),
사람 속도 변화와 완전 탄성 한계, 사람이 밀린 거리를 확인합니다. 3초(밀기는 6초) 뒤 잠시 멈췄다가 반복합니다.
옆에서 밀면 배가 마찰로 회전해 사람 둘레를 닿은 채 돌아가는데, 이는 관통이 아닌 정상 물리입니다.
화면 없이 확인하려면 `--headless`를 붙입니다. 같은 장면의 자동 검사는 `python -m pytest tests/world/test_contact.py -v`입니다.

## 하나개 해수욕장 참고 맵 (육지·바다·배)

육지는 맵 **동쪽 끝**에 붙이고, 익수자는 해안에서 약 **200m** 서쪽 바다에 둡니다.
배는 해안 근처에서 익수자 쪽을 봅니다. 물은 `Water` 부력·저항을 쓰고, 넓은 수면 표시는 맵이 그립니다.

```bash
source .venv/bin/activate
python -m sim.world.view_hanagae
python -m sim.world.view_hanagae --headless
python -m sim.world.view_hanagae --waves
python -m pytest tests/world/test_coast_map.py -q
```

지면은 수면 아래까지 내려 붙여 떠 보이지 않게 했습니다. 얕은 바다에 물장구·개헤엄·배영·다이빙·공놀이를 하는 사람 10명을 표시합니다. 출항 직선 한가운데에는 선체 길이의 약 5배인 돌섬이 수면 아래부터 위까지 솟아 배의 직진을 막습니다. 바위 여러 개가 겹친 모양입니다. 육지 충돌 비트·밀물 수면 상승·실측 GIS는 없습니다.

## 잡기·착용 시연의 코드 위치

실행 명령은 `python -m sim.world.view_grasp`, `python -m sim.world.view_wear`입니다.
`sim/world/scenarios.py`에는 GraspDemo·WearDemo와 시험용 PullFixture·WearApproachFixture가 있습니다. 착용 동작은 `sim/world/wear.py`(`SelfWear`)입니다.
자동 테스트는 뷰 대신 이 장면을 직접 사용합니다. `view_*.py`는 실행 옵션·화면·카메라를 담당합니다.
공통 물리 적분은 `world.advance_world`, 로프 표시는 `rope_visual.draw_rope`를 사용합니다.
시연별 제어 갱신·반복 리셋과 구조 단계의 Watchdog 실행 순서는 각 호출 측에서 유지합니다.

## 최신 main 배 모델 통합

EMILY 기반 선체·부품 질량/COM/관성·50N/20N 추진기·흡입구 판정 유지합니다.
`Robot(surface_height=None, *, position=None, tow_point_b=None)`에서 기본 배치는 정수 흘수이며
`position`·`tow_point_b`는 독립 시험에만 명시적으로 바꿉니다. 명령은 기존 `throttle/steer`와
시연의 `forward/steering` 키워드를 모두 지원하고, 유한한 범위 밖 명령은 main 계약대로 포화합니다.

접근 제어의 새 선체 시험 기본값은 속도 이득 60N/(m/s), 위치 이득 30N/m, 요 속도 이득 12N·m/(rad/s),
전진/후진 합력 제한 60/30N, 요 토크 제한 6N·m입니다. 추진기별 한계 안에서 회전 토크를 먼저 배분합니다.
새 COM은 앞뒤 중심에서 벗어나므로 정지·회전 시험은 정확한 영변위 대신 작은 정착 이동과 실제 힘·최종 속도를 검사합니다.
