"""🔒 좌표 변환 공통 헬퍼 (담당: 물·로봇 회의에서 정함, 통합 충돌 확인: 가영).

좌표 변환은 모든 모듈이 이 파일 함수만 씁니다. 각자 만들면 축이 조용히 어긋납니다 (노션 06 문제 2).
규칙: 변수명에 좌표계 접미사 — _w(월드), _b(몸체). ForceAccumulator 에는 월드 좌표만 넣습니다.
"""


def body_to_world(data, body_id, vec_b):
    """몸체 좌표 벡터 → 월드 좌표 (예: 추력 force_b → force_w)."""
    # TODO


def world_to_body(data, body_id, vec_w):
    """월드 좌표 벡터 → 몸체 좌표 (예: 물 저항 계산용 상대속도)."""
    # TODO


def body_velocity_w(model, data, body_id):
    """무게중심 기준 (선속도_w, 각속도_w).

    ⚠️ mj_objectVelocity(..., flg_local=1) 결과는 몸체 좌표가 아님 (PoC 발산 원인).
       월드 속도(flg_local=0)를 받고, 몸체 좌표가 필요하면 world_to_body 로 돌린다.
    """
    # TODO
