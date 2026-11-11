## 무엇을
<!-- 이번 PR에서 바꾼 내용 -->

## 왜
<!-- 관련 주간 작업 (예: W01 · 물 · 복셀 부력) -->

## 어떻게 테스트했나
<!-- 실행한 명령, 확인한 결과 (pytest -q 등) -->

## 체크리스트
- [ ] `git status`로 불필요한 파일(.venv/, 결과·영상)이 없는지 확인
- [ ] 이름·번호는 `sim/config.py` 상수만 사용 (문자열 직접 쓰지 않음)
- [ ] `xfrc_applied`에 직접 쓰지 않고 ForceAccumulator 사용
- [ ] 계약 파일(`config.py`·`forces.py`·`CONTRACT.md`)을 바꿨다면 가영 + 영향받는 담당자 태그
