# 🚁 TetherRescue

> **비전 불능 해상 환경 대응 패닉 익수자 접촉 · 기도 보호 견인 · 로프 장력 신호 기반 상태 추정 통합 시뮬레이션**

---

## 📌 1. 프로젝트 개요 (Overview)
`TetherRescue`는 수상 구조로봇이 익수자에게 도달한 이후의 단계(접촉, 견인, 상태 판단)를 하나의 파이프라인으로 구현하는 가상환경 시뮬레이션 프로젝트입니다.

해상 구조 현장에서는 파도, 물보라, 익수자의 침수로 인해 카메라/비전 센서가 쉽게 무력화되는 **비전 불능 상황(Vision Breakdown)**이 발생합니다. 이러한 한계 극복을 위해 오염되지 않는 유일한 촉각 센서인 **로프의 장력 및 가속도 신호**를 활용해 익수자의 상태를 추정하는 독창적인 시뮬레이션 파이프라인을 구축합니다.

---

## 👥 2. 팀원 구성 및 역할 (Team R&R)

| 구분 | 팀원 | 담당 | 핵심 역할 |
| :--- | :--- | :--- | :--- |
| **A** | 가영 | **PanicSim** (Python) + 파이썬 베이스라인 | **RescuerAgent** 보상 수식, PPO 설정·학습 실행, 커리큘럼, 평가, 공용 도구(mock·시드·평가·env_wrapper·그래프) |
| **A · Unity** | 동경 (팀장) | **PanicSim**(고난도) + **Unity 전담** + 통합 | RescuerAgent 설계, PPO 튜닝 주도, 모든 Unity C#(물리·Agent 3종 이식·추론 연동·되먹임·대시보드), 헤드리스 빌드, 데모 |
| **B** | 미현 | **AirwayTow** (Python) | **TowAgent** 설계·Python 학습, PID 1D 시뮬·튜닝(베이스라인), 로프·기도 파라미터, Composite Cost, 파레토, 견인 실험 분석 |
| **C** | 채연 | **TetherSense** (Python) + 문서 검토 | **FeedbackAgent** 설계·Python 학습, 데이터셋·1D CNN·학습, ONNX·테스트 벡터, 규칙 감지·규칙 기반 되먹임(베이스라인), DR 설계 |

> Unity(C#)는 동경이 전담합니다. 나머지 팀원은 Python으로 로직·파라미터를 먼저 검증해 넘기고, 실험은 헤드리스 빌드를 Python으로 실행합니다.
>
> **파트별 에이전트:** 각 파트는 RL 에이전트 1개와 비교용 베이스라인을 가집니다. B·C 에이전트는 담당자가 Python Gymnasium 환경에서 먼저 설계·학습(Stable-Baselines3 PPO)하고, 검증된 관측·행동·보상을 동경이 Unity ML-Agents로 이식합니다.
>
> | 파트 | 에이전트 | 베이스라인 | 관측 → 행동 (초안) |
> | :--- | :--- | :--- | :--- |
> | A PanicSim | `RescuerAgent` | `StraightApproachBaseline` (직진 접근) | 익수자 위치·패닉 정도 → 추진·조향 |
> | B AirwayTow | `TowAgent` | PID `TowController` | 기도 높이·로프 장력·파도 → 견인 속도·방향 |
> | C TetherSense | `FeedbackAgent` | 규칙 기반 `FeedbackPolicy` | 상태 추정 결과·신뢰도 → 유지 / 감속 / 정지·재접근 |

- **개발 환경:** Unity 6.3 LTS + Python (PyTorch, ONNX, ML-Agents) (실물 로봇 없음, 저사양 PC 전제)

---

## 🧩 3. 핵심 모듈 구조 (Core Modules)

1. **PanicSim:** 패닉 행동을 가진 익수자(FSM)에게 로봇이 안전하게 접근·접촉하도록 학습하는 RL 에이전트 `RescuerAgent` (Unity ML-Agents)
2. **AirwayTow:** 견인 중 익수자의 머리(기도) 수면 유지를 최적화하는 RL 견인 에이전트 `TowAgent`, PID 베이스라인과 비교 (Gymnasium + SB3 → Unity ML-Agents)
3. **TetherSense:** 견인 로프의 장력 시계열 신호로 익수자 상태(매달림 / 놓침 / 의식 저하 추정)를 분류하는 모델 (PyTorch → ONNX 변환 및 실시간 추론) + 추정 결과로 대응을 고르는 RL 에이전트 `FeedbackAgent`, 규칙 기반 정책과 비교

### 3.1 전체 파이프라인 흐름
```text
[접근] PanicSim ── RescuerAgent(PPO, 25Hz) ↔ RescueeFSM
          │ 접촉 완료 → phase = TOW
          ▼
[견인] AirwayTow ── TowAgent(PPO) | 베이스라인 TowController(PID) + Tether 조인트(스프링-댐퍼)
          │ TetherData 50Hz
          ▼
[추정] TetherSense ── ① 규칙 기반 놓침 감지 (50Hz)
          │           ② 1D CNN 추론 (5Hz, 2초 슬라이딩 윈도우)
          ▼
[되먹임] FeedbackAgent(PPO) | 베이스라인 FeedbackPolicy(규칙) ── 유지 / 감속 / 정지·재접근
          + 안전망: 규칙 놓침 감지·InferenceWatchdog가 에이전트 결정을 덮어씀
```

### 3.2 데이터 인터페이스 표준 규격
상세 규격의 단일 기준은 노션 「02. 요구사항 명세서 (SRS)」 5장이며, 코드는 `common/interfaces.py`와 `unity_env/Assets/Scripts/Core/DataInterface.cs`에 둡니다.
- **RescueeStatus (익수자 상태):** `position`, `fsm_state`, `panic_level` (0~1), `stamina` (0~1), `is_conscious`, `airway_submerged`
- **TetherData (센서 입력, 50Hz):** `timestamp`, `tension_force`, `robot_accel_x/y/z`, `thrust_cmd`, `steer_cmd` → 모델 입력 6채널
- **TetherSenseOutput (상태 추정):** `predicted_state` (ATTACHED / DETACHED / UNRESPONSIVE), `confidence`, `probs`, `source` (RULE / MODEL)

---

## 🛠️ 4. 기술 스택 (Tech Stack)
- **OS:** Windows + WSL2 Ubuntu 24.04
- **Python:** 3.10.12
- **Frameworks & ML:** PyTorch 2.1.2, Unity 6.3 LTS (6000.3.x) + `com.unity.ml-agents` 4.0.x, ML-Agents (Python) 1.1.0, Gymnasium 0.29.1, Stable-Baselines3 2.4.1, ONNX 1.15.0, ONNX Runtime 1.14.1
- **데이터 처리 및 시각화:** NumPy, pandas, scikit-learn, Matplotlib, Seaborn

---

## 📂 5. 프로젝트 구조 (Repository Structure)
```text
TetherRescue/
├── .github/pull_request_template.md
├── common/                  # 데이터 규격 (SRS 5장)
│   └── interfaces.py
├── configs/
│   ├── config.yaml
│   ├── panic_sim_ppo.yaml      # RescuerAgent
│   ├── airway_tow_ppo.yaml     # TowAgent (Unity)
│   ├── feedback_ppo.yaml       # FeedbackAgent (Unity)
│   └── tethersense_config.yaml
├── data/{raw,processed}/    # git 제외
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_evaluation_plots.ipynb
├── panicsim/
│   ├── env_wrapper.py
│   └── reward_functions.py
├── airwaytow/
│   ├── composite_cost.py
│   ├── pareto.py
│   ├── pid_controller.py
│   └── tow_env.py           # TowAgent Python 학습 환경
├── tethersense/
│   ├── dataset.py
│   ├── export_onnx.py
│   ├── feedback_env.py      # FeedbackAgent Python 학습 환경
│   ├── feedback_policy.py
│   ├── models.py
│   ├── rule_detector.py
│   └── train_tethersense.py
├── scripts/
│   ├── evaluate_all.py
│   ├── make_plots.py
│   ├── mock_data_generator.py
│   └── seed_fixer.py
├── tests/
├── unity_env/Assets/
│   ├── Models/ · Prefabs/
│   ├── Scenes/              # MainRescueScenario (소유: 동경), Sandbox/ (개인 실험)
│   └── Scripts/
│       ├── Core/            # DataInterface, TelemetryLogger, PhaseManager, InferenceWatchdog
│       ├── Environment/     # WaveField, Buoyancy, RobotController
│       ├── PanicSim/        # RescueeFSM, RescuerAgent, RescueReward, StraightApproachBaseline
│       ├── AirwayTow/       # TetherJointPhysics, AirwaySensor, TowController, TowAgent
│       ├── TetherSense/     # OnnxInferenceRunner, SlidingWindowBuffer, DetachRuleDetector, FeedbackPolicy, FeedbackAgent
│       └── UI/              # CommandDashboardUI
├── HOW_TO_RUN.md
├── README.md
├── RISK_MANAGEMENT.md
└── requirements.txt
```

---

## 🗓️ 6. 일정 및 마일스톤 (2026-10-26 ~ 2027-03-07)
주간 작업은 노션 「주간 개발 일정표」에서 관리합니다.

| 단계 | 기간 | 마일스톤 |
| :--- | :--- | :--- |
| 1단계 기반 구축 | 10/26 ~ 11/22 (W1~4) | M0 파이프라인 PoC (11/22) |
| 2단계 모듈 v1 | 11/23 ~ 12/27 (W5~9) | M1 PanicSim 전복률 30% 감소 (12/27) |
| 3단계 시뮬 데이터 | 12/28 ~ 1/24 (W10~13) | M2 AirwayTow 파레토 곡선 (1/24) |
| 4단계 피드백 | 1/25 ~ 2/21 (W14~17, W16 설 연휴 버퍼) | M3 놓침 감지·되먹임 검증 (2/21) |
| 5단계 통합·마감 | 2/22 ~ 3/7 (W18~19) | 최종 제출 v1.0 (3/7) |
| 예비·발표 준비 | 3/8 ~ 3/23 | 종강 전 최종 발표 |

---

## 📊 7. 평가 지표 (Evaluation Metrics)
- **접촉 단계:** 전복률, 구조 성공률, 평균 소요 시간, 익수자 충돌 횟수
- **견인 단계:** 머리 수면 위 시간 비율($T_{sub}$), 평균 견인 속도, 로프 최대 장력, Composite Cost 점수
- **상태 추정:** Accuracy, 클래스별 Recall (특히 '놓침' 상태), 상태 전이 지연 시간
- **통합 성능:** 되먹임 적용 전후 성공률 변화, 환경 난이도별 성능 저하 곡선
- **에이전트 vs 베이스라인:** 파트별로 같은 시드·시나리오에서 에이전트와 베이스라인(직진 접근 / PID / 규칙 기반 되먹임)을 위 지표로 비교

---

## ⚠️ 8. 한계점 및 리스크 관리 (Limitations & Risks)
- **물리 근사:** 유체/인체 물리의 단순화로 실측 안전성 진단으로의 확대 해석을 금지하며 "시뮬레이션 기반 비교 연구"로 포지셔닝합니다.
- **데이터 편향 및 진단 제한:** 시뮬레이션 생성 데이터의 편향은 Domain Randomization으로 완화하고, '의식 저하' 판정은 의료적 진단이 아닌 제어 목적임을 명시합니다.
- **저사양 최적화:** `--no-graphics` 헤드리스 학습, step 로그 10Hz 제한, TetherSense 추론 5Hz 제한을 적용합니다.

---

## 🚀 9. 설치 및 실행 가이드 (Installation & Run)
자세한 실행 방법과 명령어는 [HOW_TO_RUN.md](./HOW_TO_RUN.md) 및 [RISK_MANAGEMENT.md](./RISK_MANAGEMENT.md)를 참고하세요.
```bash
# 가상환경 생성 및 패키지 설치
conda create -n tether_rescue python=3.10.12 -y
conda activate tether_rescue
python -m pip install -r requirements.txt
```