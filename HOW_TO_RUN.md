# TetherRescue 실행 가이드 (`HOW_TO_RUN.md`)

이 문서는 TetherRescue 프로젝트의 Python 학습 파이프라인과 Unity
시뮬레이션을 실행하는 순서를 설명합니다.

> **버전 고정:** Unity 6.3 LTS (6000.3.x) + `com.unity.ml-agents` 4.0.x +
> Python `mlagents==1.1.0` 조합을 사용합니다. Unity 2022.3은 사용하지
> 않습니다. 팀 전원이 같은 Unity 패치 버전을 씁니다.
> [공식 설치 가이드](https://unity-technologies.github.io/ml-agents/Installation/)

## 1. 사전 준비

-   Ubuntu 24.04 LTS on WSL2
-   Conda 환경: `tether_rescue`
-   Python 3.10.x
-   프로젝트 의존성 설치 완료
-   Unity 6.3 LTS (6000.3.x), 프로젝트 폴더: `unity_env/` (Unity 작업은 동경 전담.
    다른 팀원은 `Builds/Linux/` 헤드리스 빌드를 Python으로 실행)
-   Unity 설정: Visible Meta Files, Force Text, Fixed Timestep 0.02 (50Hz)
-   필요한 설정 파일 및 스크립트가 저장소에 존재하는지 확인

저장소 루트에서 명령을 실행합니다. 예시 경로와 파일 이름은 실제 프로젝트
구조와 일치하는지 확인하세요.

## 2. Python 환경 활성화 및 확인

``` bash
conda activate tether_rescue

python --version
python -m pip check
mlagents-learn --help
```

핵심 라이브러리 import 및 버전을 확인합니다.

``` bash
python -c "import torch, onnx, onnxruntime, numpy, mlagents, gymnasium, stable_baselines3; print('torch:', torch.__version__); print('onnx:', onnx.__version__); print('onnxruntime:', onnxruntime.__version__); print('numpy:', numpy.__version__); print('gymnasium:', gymnasium.__version__); print('sb3:', stable_baselines3.__version__); print('ML-Agents import OK')"
```

`mlagents-learn --help`가 실행되고 `pip check`에서 의존성 충돌이 없어야
합니다. 이 검사는 Unity 연결이나 모델 추론까지 보장하지 않으므로 이후
단계도 확인합니다.

## 3. (선택) 테스트용 모의 데이터 생성

프로젝트에 해당 스크립트가 포함된 경우 실행합니다.

``` bash
python scripts/mock_data_generator.py --samples 5000 --seed 1004 --output data/raw/mock/mock_tether_data.csv
```

`data/raw/` 디렉터리가 없다면 먼저 만듭니다.

``` bash
mkdir -p data/raw/mock
```

실행 인자나 출력 경로가 다르면 다음으로 스크립트 사용법을 확인합니다.

``` bash
python scripts/mock_data_generator.py --help
```

## 4. PanicSim --- PPO 학습

### 4.1 Unity 환경 준비

Unity Hub에서 `unity_env/` 프로젝트를 엽니다. 씬과 Agent 설정에
ML-Agents 컴포넌트가 연결되어 있는지 확인합니다. 학습할 씬에서 필요한
Behavior Name(`Rescuer`)과 `configs/panic_sim_ppo.yaml`의 Behavior 설정이
일치해야 합니다.

### 4.2 학습 시작

저장소 루트에서 다음 명령을 실행합니다.

``` bash
conda activate tether_rescue
mlagents-learn configs/panic_sim_ppo.yaml --run-id=PanicSim_v1
```

명령을 실행한 뒤 Unity Editor에서 대상 씬을 열고 **Play**를 누릅니다.
Trainer가 환경 연결을 기다리는 동안 Unity 시뮬레이션이 시작되어야
합니다.

빌드된 Unity 환경을 사용하는 경우에는 환경 실행 파일 경로를 지정합니다.

``` bash
mlagents-learn configs/panic_sim_ppo.yaml --run-id=PanicSim_v1 --env=PATH_TO_UNITY_BUILD --no-graphics --num-envs=4
```

`PATH_TO_UNITY_BUILD`를 실제 빌드 실행 파일 경로로 바꾸세요.
`--no-graphics`는 시각적 관측이 필요하지 않은 빌드 환경에서만
사용하세요. 설치된 CLI의 옵션은 `mlagents-learn --help`로 확인할 수
있습니다.

학습 산출물은 일반적으로 `results/PanicSim_v1/` 아래에 저장됩니다. 실제
경로는 실행 로그와 설정을 기준으로 확인하세요.

### 4.3 TowAgent · FeedbackAgent (Python 먼저 → Unity 이식)

B(미현)·C(채연) 에이전트는 Unity 없이 Python Gymnasium 환경에서 먼저
설계·학습합니다(Stable-Baselines3 PPO). 환경 파일은
`airwaytow/tow_env.py`, `tethersense/feedback_env.py`이며 각각 PID,
규칙 기반 되먹임 베이스라인과 같은 시드로 비교합니다.

관측·행동·보상이 확정되면 동경이 `TowAgent.cs`, `FeedbackAgent.cs`로
이식하고, Unity 학습은 PanicSim과 같은 방식으로 실행합니다. Behavior
Name은 각각 `Tow`, `Feedback`입니다.

``` bash
mlagents-learn configs/airway_tow_ppo.yaml --run-id=Tow_v1 --env=PATH_TO_UNITY_BUILD --no-graphics
mlagents-learn configs/feedback_ppo.yaml --run-id=Feedback_v1 --env=PATH_TO_UNITY_BUILD --no-graphics
```

## 5. TetherSense --- 모델 학습 및 ONNX 내보내기

아래 명령은 프로젝트에 해당 스크립트와 설정 파일이 존재한다는
전제입니다.

### 5.1 모델 학습

``` bash
python tethersense/train_tethersense.py --config configs/tethersense_config.yaml
```

학습 스크립트가 요구하는 데이터 경로, 체크포인트 저장 위치 및 옵션은
설정 파일과 `--help` 출력으로 확인하세요.

``` bash
python tethersense/train_tethersense.py --help
```

### 5.2 ONNX 내보내기

``` bash
python tethersense/export_onnx.py \
  --checkpoint checkpoints/tethersense_best.pth \
  --output unity_env/Assets/Models/tethersense.onnx
```

필요한 폴더가 없다면 생성합니다.

``` bash
mkdir -p unity_env/Assets/Models
```

체크포인트 파일명이 실제 학습 결과와 다르면 `--checkpoint` 경로를
수정하세요. 내보내기 스크립트의 옵션이 다를 경우
`python tethersense/export_onnx.py --help`를 실행합니다.

### 5.3 ONNX 파일 확인

``` bash
python -c "import onnx; m = onnx.load('unity_env/Assets/Models/tethersense.onnx'); onnx.checker.check_model(m); print('ONNX model OK')"
```

ONNX 파일의 입출력 shape와 특성 순서가 학습 코드와 Unity 측 구현에서
동일한지 확인합니다. onnxruntime 출력과 PyTorch 출력의 차이가 1e-4 미만이어야
하고, 같은 입력 CSV로 Unity 추론 결과가 Python과 일치하는지(parity test)도
확인합니다.

## 6. Unity에서 통합 실행

1.  Unity Hub에서 `unity_env/` 프로젝트를 엽니다.
2.  `Assets/Scenes/MainRescueScenario.unity` 씬을 엽니다. 씬 파일이 다른
    위치에 있다면 실제 경로를 사용합니다.
3.  Inspector에서 `OnnxInferenceRunner`에 `tethersense.onnx`와
    `normalization.json`을 할당합니다. 특성 순서(SRS 5.1 `FEATURE_ORDER`),
    윈도우 길이(100), 정규화 방식이 Python과 같은지 확인합니다.
4.  PanicSim Agent, AirwayTow 제어기, 센서 및 UI 참조가 모두 연결되어
    있는지 확인합니다.
5.  Console의 오류를 확인한 뒤 **Play**를 눌러 시뮬레이션합니다.
6.  센서 입력, 상태 예측, 장력 제한 및 비상 안전 동작이 예상대로
    작동하는지 확인합니다.

**주의:** TetherSense 추론은 Unity 내부에서 실행합니다(Python 추론 서버를
쓰지 않음). Python ONNX Runtime은 변환 검증용입니다. ONNX 파일을 Unity
프로젝트에 복사하는 것만으로 추론 연결이 완성되지는 않으므로 parity test로
결과 일치를 확인하세요.

## 6.1 평가

``` bash
python scripts/evaluate_all.py --runs data/raw/{run_id} --out results/eval/{run_id}
```

## 7. 빠른 검증 체크리스트

-   [ ] `conda activate tether_rescue` 성공
-   [ ] `python -m pip check` 통과
-   [ ] `mlagents-learn --help` 실행 성공
-   [ ] 모의 데이터 생성 또는 학습 데이터 경로 확인
-   [ ] Unity 씬의 Agent 및 Behavior 설정 확인
-   [ ] PPO 학습 실행 및 체크포인트 생성 확인
-   [ ] TetherSense 학습 결과 체크포인트 확인
-   [ ] ONNX 내보내기 및 `onnx.checker` 검사 통과
-   [ ] Unity에서 모델 참조 및 추론 경로 확인
-   [ ] `DETACHED` 상태에서 안전 제어가 작동하는지 검증

## 8. 문제 해결

### `ModuleNotFoundError: No module named 'pkg_resources'`

활성화된 `tether_rescue` 환경에서 다음을 실행하고 다시 확인합니다.

``` bash
python -m pip install "setuptools==80.9.0"
python -c "import pkg_resources; print('pkg_resources OK')"
mlagents-learn --help
```

### `python -m pip check`에서 충돌 발생

``` bash
python -m pip check
python -m pip show mlagents mlagents-envs torch onnx onnxruntime grpcio setuptools gymnasium stable-baselines3
```

출력된 버전을 확인하고 `requirements.txt`와 실제 환경을 함께 조정합니다.
이미 동작하는 환경에서 원인 확인 없이 여러 패키지를 한꺼번에
업그레이드하지 마세요.

### Unity가 Trainer에 연결되지 않음

-   `mlagents-learn` 명령이 오류 없이 실행 중인지 확인합니다.
-   Unity 씬에서 올바른 Agent가 활성화되어 있는지 확인합니다.
-   Behavior Name 및 Behavior 설정이 YAML과 일치하는지 확인합니다.
-   Unity 패키지와 Python `mlagents` 버전 조합을 확인합니다.
-   WSL2에서 Unity Editor와 Python 프로세스가 서로 다른 환경에서
    실행되는 경우 네트워크 연결 및 실행 위치를 점검합니다.

### ONNX 모델을 불러오지 못함

-   체크포인트와 ONNX 출력 경로가 실제 파일 위치와 일치하는지
    확인합니다.
-   학습 및 추론의 입력 shape, 특성 순서, 정규화 방법이 동일한지
    확인합니다.
-   모델을 실행하는 런타임(Python ONNX Runtime 또는 Unity 측 런타임)이
    해당 모델 연산자를 지원하는지 확인합니다.

------------------------------------------------------------------------

## 9. 재현성 기록

각 실험마다 다음 정보를 함께 기록합니다.

-   Git commit hash
-   Python 및 패키지 버전
-   Unity Editor 및 `com.unity.ml-agents` 버전
-   설정 파일과 난수 시드
-   학습 시작/종료 시각 및 실행 명령
-   체크포인트 및 ONNX 파일 경로
-   평가 결과(구조 성공률, 전복률, 최대 장력, 분류 지표, 추론 지연시간)

성능 목표는 실제 측정 결과가 확보된 경우에만 달성으로 표시합니다.
