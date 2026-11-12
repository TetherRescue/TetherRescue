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

현재 기본 조립에는 상자 형태의 사람 강체가 포함됩니다. 물·로봇·줄 모듈은 아직 빈 뼈대입니다.
테스트는 빈 세계, 움직이는 시험용 몸체, 사람의 자유낙하에서 각각 1,000스텝을 확인합니다.
실제 인체·줄·부력 모델의 물리 정확성은 후속 구현과 검증 대상입니다.
설정 인자를 생략하면 `config.py`의 값을 사용하고, 미확정(`None`)이면 오류로 알립니다.

사람만 조립해서 크기·질량을 확인하려면:

```python
from sim.human import Human
from sim.world import build_world

human = Human(position=(0, 0, 1))  # 시험용: 수면 기준 1m 위
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

현재 사람은 상자 형태이며 중력으로 떨어집니다. 약 1초마다 시작 위치로 돌아와 반복합니다.
창을 닫으면 종료합니다. 이는 독립 데모의 반복 동작이며 실제 구조 시나리오의 리셋 정책이 아닙니다.
렌더러가 EGL로 설정되어 창이 열리지 않으면 `MUJOCO_GL=glfw python -m sim.view_human`을 사용합니다.
화면이 없는 환경에서는 `python -m sim.view_human --headless`로 데모의 계산만 확인할 수 있습니다.
