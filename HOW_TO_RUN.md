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
