"""🔒 좌표 변환 공통 헬퍼 (담당: 동경, 통합 충돌 확인: 가영).

좌표 변환은 모든 모듈이 이 파일 함수만 씁니다. 각자 만들면 축이 조용히 어긋납니다 (노션 06 문제 2).
규칙: 변수명에 좌표계 접미사 — _w(월드), _b(몸체). ForceAccumulator 에는 월드 좌표만 넣습니다.

몸체 좌표 = body 의 frame (data.xmat). 관성 주축 frame(data.ximat)과 다를 수 있으니 섞지 않습니다.
"""
import mujoco
import numpy as np


def _rot(data, body_id):
    """몸체 → 월드 회전행렬 (3x3). 열 0·1·2 = 몸체 앞(F)·왼(L)·위(U) 축의 월드 방향."""
    return data.xmat[body_id].reshape(3, 3)


def body_to_world(data, body_id, vec_b):
    """몸체 좌표 벡터 → 월드 좌표 (예: 추력 force_b → force_w). 방향만 돌림 (위치 이동 없음)."""
    return _rot(data, body_id) @ np.asarray(vec_b, dtype=float)


def world_to_body(data, body_id, vec_w):
    """월드 좌표 벡터 → 몸체 좌표 (예: 물 저항 계산용 상대속도). 회전행렬의 전치 = 역회전."""
    return _rot(data, body_id).T @ np.asarray(vec_w, dtype=float)


def point_body_to_world(data, body_id, point_b):
    """몸체 좌표의 점 위치 → 월드 위치 (예: 복셀 중심, 추진기 위치). 회전 + body 원점 이동.

    point_b 는 (3,) 한 점 또는 (N, 3) 여러 점. 여러 점이면 (N, 3) 을 돌려줌 (복셀 128칸을 한 번에).
    """
    return np.asarray(point_b, dtype=float) @ _rot(data, body_id).T + data.xpos[body_id]


def body_velocity_w(model, data, body_id):
    """무게중심 기준 (선속도_w, 각속도_w).

    ⚠️ mj_objectVelocity(..., flg_local=1) 결과는 몸체 좌표가 아님 (PoC 발산 원인).
       월드 속도(flg_local=0)를 받고, 몸체 좌표가 필요하면 world_to_body 로 돌린다.
    """
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, res, 0)
    return res[3:].copy(), res[:3].copy()   # MuJoCo 순서는 (회전, 이동)
