# 🚁 TetherRescue

> **비전 불능 해상 환경 대응 패닉 익수자 접촉 · 기도 보호 견인 · 로프 장력 신호 기반 상태 추정 통합 시뮬레이션**

---

## 📌 1. 프로젝트 개요 (Overview)
`TetherRescue`는 수상 구조로봇이 익수자에게 도달한 이후의 단계(접촉, 견인, 상태 판단)를 하나의 파이프라인으로 구현하는 가상환경 시뮬레이션 프로젝트입니다.

해상 구조 현장에서는 파도, 물보라, 익수자의 침수로 인해 카메라/비전 센서가 쉽게 무력화되는 **비전 불능 상황(Vision Breakdown)**이 발생합니다. 이러한 한계 극복을 위해 오염되지 않는 유일한 촉각 센서인 **로프의 장력 및 가속도 신호**를 활용해 익수자의 상태를 추정하는 독창적인 시뮬레이션 파이프라인을 구축합니다.

---

## 👥 2. 팀원 구성 및 역할 (Team R&R)

| 구분 | 담당 모듈 | 핵심 역할 및 기술 스택 |
| :--- | :--- | :--- |
| **팀원 A** | **PanicSim** | 익수자 FSM, 접근/접촉 RL 정책 학습 (PPO, Unity ML-Agents) |
| **팀원 B** | **AirwayTow** | 기도 보호 견인 제어, Composite Cost 최적화 (PID / RL) |
| **팀원 C** | **TetherSense** | 장력 시계열 ML 분류, Control Decoupling, ONNX 연동 (PyTorch) |
| **팀원 D** | **Core / Integration** | 통신 PoC, Mock Generator, 대시보드 UI, 로깅 및 시드 관리 |

- **작성 관점:** 개발자
- **개발 환경:** Unity + Python (PyTorch, ONNX, ML-Agents) / Blender (실물 로봇 없음, 저사양 PC 전제)

---

## 🧩 3. 핵심 모듈 구조 (Core Modules)

1. **PanicSim:** 패닉 행동을 가진 익수자 에이전트와 로봇이 안전하게 접근·접촉하도록 학습하는 RL 정책 (Unity ML-Agents Python SDK)
2. **AirwayTow:** 견인 중 익수자의 머리(기도) 수면 유지를 최적화하는 견인 제어 (NumPy / SciPy / RL)
3. **TetherSense:** 견인 로프의 장력 시계열 신호로 익수자 상태(매달림 / 놓침 / 의식 저하 추정)를 분류하는 모델 (PyTorch → ONNX 변환 및 실시간 추론)

### 3.1 전체 파이프라인 흐름
```text
[1. PanicSim (PPO 접근)]
       ↓
[2. AirwayTow (견인 제어)]
       ↓
[3. TetherSense (장력/가속도/제어명령 기반 상태 추정)]
       ↓
[4. Feedback (속도 조정 / 견인 중단 / 재시도)]
```

### 3.2 데이터 인터페이스 표준 규격
- **WickerState (익수자 상태):** `position`, `panic_level` ($0.0 \sim 1.0$), `stamina` ($0.0 \sim 1.0$), `airway_submerged_ratio`
- **TetherData (로프 센서 데이터 - Decoupling 반영):** `timestamp`, `tension_force`, `robot_accel`, `robot_control_cmd` (Correlation Leakage 방지)
- **TetherSenseOutput (ML 추론 결과):** `predicted_state` (매달림/놓침/의식저하), `confidence` ($0.0 \sim 1.0$)

---

## 🛠️ 4. 기술 스택 (Tech Stack)
- **OS:** Windows 10/11 / Ubuntu 22.04 LTS (WSL2 가능)
- **Python:** 3.10.x
- **Frameworks & ML:** PyTorch 2.1.x, Unity 2022.3 LTS, ML-Agents 0.30.0, ONNX Runtime 1.16.x
- **데이터 처리 및 시각화:** NumPy, pandas, scikit-learn, Matplotlib, Seaborn

---

## 📂 5. 프로젝트 구조 (Repository Structure)
```text
tetherRescue_structure/
├── airwaytow/
│   ├── __init__.py
│   ├── composite_cost.py
│   └── pid_controller.py
├── configs/
│   ├── config.yaml
│   ├── panic_sim_ppo.yaml
│   └── tethersense_config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_evaluation_plots.ipynb
├── panicsim/
│   ├── __init__.py
│   ├── env_wrapper.py
│   └── reward_functions.py
├── scripts/
│   ├── evaluate_all.py
│   ├── mock_data_generator.py
│   └── seed_fixer.py
├── tethersense/
│   ├── __init__.py
│   ├── dataset.py
│   ├── export_onnx.py
│   ├── models.py
│   └── train_tethersense.py
├── unity_env/
│   └── Assets/
│       ├── Models/
│       ├── Prefabs/
│       ├── Scenes/
│       └── Scripts/
│           ├── AirwayTow/
│           │   ├── TetherJointPhysics.cs
│           │   └── TowController.cs
│           ├── Core/
│           │   └── DataInterface.cs
│           ├── PanicSim/
│           │   └── RescueeFSM.cs
│           ├── TetherSense/
│           │   └── OnnxInferenceRunner.cs
│           └── UI/
│               └── CommandDashboardUI.cs
├── .gitignore
├── HOW_TO_RUN.md
├── README.md
├── requirements.txt
└── RISK_MANAGEMENT.md
```

---

## 🗓️ 6. 단계별 작업 목록 및 팀원 R&R (6개월 / 4인 체제)
- **1단계 (1~4주): 기반 구축 및 파이프라인 PoC**
  - 데이터 인터페이스 표준 고정, Unity-Python 통신 PoC 및 `mock_data_generator.py` 구축
- **2단계 (5~10주): PanicSim 개발 (마일스톤 1)**
  - PPO 기반 익수자 접근 환경 구축 및 직선 접근 대비 전복률 감소 수치 도출
- **3단계 (11~16주): AirwayTow 개발 (마일스톤 2)**
  - 스프링-댐퍼 물리 구현 및 Composite Cost($T_{sub}, T_{total}, F_{peak}$) 기반 파레토 곡선 도출
- **4단계 (17~21주): TetherSense 및 피드백 연동 (마일스톤 3)**
  - Domain Randomization 기반 데이터 수집, 1D CNN/LSTM 학습, ONNX 실시간 연동(2~5Hz) 및 되먹임 검증
- **5단계 (22~26주): 통합, 스트레스 테스트 및 최종 정리**
  - 파도/패닉/노이즈 극한 조절 스트레스 테스트(Breakdown Point 탐색), 대시보드 UI 및 데모 영상 제작

---

## 📊 7. 평가 지표 (Evaluation Metrics)
- **접촉 단계:** 전복률, 구조 성공률, 평균 소요 시간, 익수자 충돌 횟수
- **견인 단계:** 머리 수면 위 시간 비율($T_{sub}$), 평균 견인 속도, 로프 최대 장력, Composite Cost 점수
- **상태 추정:** Accuracy, 클래스별 Recall (특히 '놓침' 상태), 상태 전이 지연 시간
- **통합 성능:** 되먹임 적용 전후 성공률 변화, 환경 난이도별 성능 저하 곡선

---

## ⚠️ 8. 한계점 및 리스크 관리 (Limitations & Risks)
- **물리 근사:** 유체/인체 물리의 단순화로 실측 안전성 진단으로의 확대 해석을 금지하며 "시뮬레이션 기반 비교 연구"로 포지셔닝합니다.
- **데이터 편향 및 진단 제한:** 시뮬레이션 생성 데이터의 편향은 Domain Randomization으로 완화하고, '의식 저하' 판정은 의료적 진단이 아닌 제어 목적임을 명시합니다.
- **저사양 최적화:** Unity `-nographics` 헤드리스 학습, CSV 저장 주기 제한, TetherSense 추론 주기 2~5Hz 제한을 적용합니다.

---

## 🚀 9. 설치 및 실행 가이드 (Installation & Run)
자세한 실행 방법과 명령어는 [HOW_TO_RUN.md](./HOW_TO_RUN.md) 및 [RISK_MANAGEMENT.md](./RISK_MANAGEMENT.md)를 참고하세요.
```bash
# 가상환경 생성 및 패키지 설치
conda create -n tether_rescue python=3.10 -y
conda activate tether_rescue
python -m pip install -r requirements.txt
```