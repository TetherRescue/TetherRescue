# 실행 방법

자세한 내용은 노션 "03. 개발 환경 & 실행 가이드"를 기준으로 합니다.

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
python -m pytest tests/test_world.py tests/test_forces.py -q
```

조립 함수만 사용하려면:

```python
from sim.world import build_world

# 회의에서 확정한 config 값(dt=0.002초, 중력=9.81, implicitfast)을 사용합니다.
world = build_world()
print(world.model.nbody, world.data.time)
```

현재 기본 조립에는 몸통·머리·양팔·양다리를 고정한 사람과 구명환이 포함됩니다.
물 모듈은 파도·감쇠 계산 일부가 구현됐고, 로봇 조립은 아직 빈 뼈대입니다.
기본 세계의 로프는 명시적으로 미연결이며 아래 독립 데모에서 시험합니다.
테스트는 빈 세계, 움직이는 시험용 몸체, 사람의 자유낙하에서 각각 1,000스텝을 확인합니다.
실제 인체·줄·부력 모델의 물리 정확성은 후속 구현과 검증 대상입니다.
설정 인자를 생략하면 `config.py`의 값을 사용하고, 미확정(`None`)이면 오류로 알립니다.

기존 상자 시험 모드로 크기·질량을 확인하려면:

```python
from sim.human import Human
from sim.world import build_world

human = Human(position=(0, 0, 1), shape="box")  # 기본 형상은 humanoid
world = build_world(parts=[human])
print(world.model.body_mass[human.body_id])  # 70kg
print(world.model.geom_size[human.geom_id])  # 반길이: [0.85, 0.2, 0.1]
```

사람은 자유 관절로 이동·회전하며, 현재 충돌과 추가 외력은 없습니다.
스텝을 진행하면 중력으로 떨어집니다. 줄 연결점은 위치 합의 전이라 생성하지 않습니다.

## 사람 움직임 보기

데스크톱 또는 WSLg처럼 GUI 창을 열 수 있는 환경에서 실행합니다.

```bash
source .venv/bin/activate
python -m sim.view_human
```

청록색 몸통·구 형태 머리·양팔·양다리와 주황·흰색 구명환이 함께 떨어집니다.
파란 점은 구명환의 로프 연결점입니다. 양팔은 환보다 머리 쪽에서 바깥으로 뻗습니다.
약 1초마다 시작 위치로 돌아와 반복합니다. 모든 부위와 구명환은 고정 연결되어 있습니다.
팔 관절 운동·접촉으로 몸을 지지하는 기능·부력은 아직 없습니다.
창을 닫으면 종료합니다. 이는 독립 데모의 반복 동작이며 실제 구조 시나리오의 리셋 정책이 아닙니다.
렌더러가 EGL로 설정되어 창이 열리지 않으면 `MUJOCO_GL=glfw python -m sim.view_human`을 사용합니다.
화면이 없는 환경에서는 `python -m sim.view_human --headless`로 데모의 계산만 확인할 수 있습니다.
기존 사람만 보려면 `python -m sim.view_human --human-only`로 실행합니다.
구명환 검증만 실행하려면 `python -m pytest tests/test_lifebuoy.py -v`를 사용합니다.

## 로프 견인 시험 보기

```bash
source .venv/bin/activate
python -m sim.view_tether
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
기존 중력 낙하 시연은 `python -m sim.view_tether --pendulum`로 실행합니다.

```bash
python -m sim.view_tether --headless
python -m pytest tests/test_tether.py -v
```

단위 검사는 느슨함·늘어남·빠른 단축·양쪽 작용반작용을 엔진의 실제 가속도로 확인합니다.
통합 검사는 리셋 없이 사람·구명환 견인을 연속 1,000스텝 실행합니다.
추가 비교 검사는 느슨한 상태의 정지와 늘어난 로프를 놓은 뒤 이동을 확인합니다.
`python -m pytest tests/test_rope_visual.py -v`는 끝점 일치·표시 길이 보존·실제 표시 구간 길이를 검증합니다.
