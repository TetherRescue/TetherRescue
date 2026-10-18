# 🛡️ TetherRescue Risk Management Matrix

| 리스크 항목 | 영향도 | 발생 가능성 | 예방 및 사전 조치 | 비상 대응책 (Plan B / Fallback) |
| :--- | :---: | :---: | :--- | :--- |
| **1. Unity-Python ONNX 실시간 연동 미세 병목** | High | Medium | • TetherSense 추론 주기를 RL 제어 주기와 분리 ($2 \sim 5\text{Hz}$ 차등화)<br>• Sliding Window 메모리 재할당 최소화 | • ONNX Runtime 통신을 C# Sentis 패키지 대신 간이 Socket/gRPC 백그라운드 프로세스로 분리 구동 |
| **2. PanicSim PPO 강화학습 수렴 실패** | High | Medium | • 보상 함수 단계별 커리큘럼 학습 적용<br>• 헤드리스 다중 환경 수 확충 | • 규칙 기반(FSM/APF) 접근 알고리즘으로 즉시 교체하여 베이스라인 확보 |
| **3. TetherSense Synthetic Data 과적합** | Medium | High | • 파도, 수중 마찰, 익수자 질량 $\pm 20\%$ Domain Randomization 적용 | • 1D CNN/LSTM 구조를 초소형 모델로 다이어트하고 L2 Regularization 및 Dropout 적용 |
| **4. 통신 지연 / 서버 응답 타임아웃 ($>200\text{ms}$)** | High | Low | • C# 연동부에 $200\text{ms}$ 타임아웃 타이머 설정 | • **NFR-ERR-05 Fallback 적용:** 이전 상태 유지하며 로봇 속도 $50\%$ 즉시 감속 안전모드 전환 |
| **5. 일정 지연** | High | Low | • 1주차 `mock_data_generator.py`로 의존성 원천 차단 | • TetherSense 통합이 지연될 경우, PanicSim + AirwayTow 2개 모듈로 마일스톤 데모 마감 |
