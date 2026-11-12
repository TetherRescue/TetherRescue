"""로프 길이를 유지하는 표시용 중심선. 중간 지점의 물리·접촉은 계산하지 않는다."""

import operator

import numpy as np


def rope_curve(start, end, rest_length, *, segments=64):
    """두 끝 거리가 기준 길이보다 짧으면 같은 길이의 S 곡선을 반환한다.

    탄성으로 늘어난 구간은 직선, 느슨한 구간은 기준 길이를 유지한다.
    곡선 길이는 실제 그리는 꺾은선 구간 길이의 합으로 맞춘다.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if (start.shape != (3,) or end.shape != (3,)
            or not np.isfinite(start).all() or not np.isfinite(end).all()):
        raise ValueError("로프 양 끝점은 유한한 3차원 좌표여야 합니다.")
    rest_length = float(rest_length)
    if not np.isfinite(rest_length) or rest_length <= 0:
        raise ValueError("로프 기준 길이는 유한한 양수여야 합니다.")
    segments = operator.index(segments)
    if segments < 8:
        raise ValueError("표시용 로프 구간은 8개 이상이어야 합니다.")
    delta = end - start
    gap = np.linalg.norm(delta)
    if gap >= rest_length:
        return np.array([start, end])

    direction = delta / gap if gap > 1e-12 else np.array([1., 0., 0.])
    normal = np.cross([0., 0., 1.], direction)
    if np.linalg.norm(normal) < 1e-8:
        normal = np.cross([0., 1., 0.], direction)
    normal /= np.linalg.norm(normal)
    parameter = np.linspace(0., 1., segments + 1)
    baseline = start + parameter[:, None] * delta
    wave = np.sin(2 * np.pi * parameter)[:, None] * normal
    if gap < 1e-12:
        # 끝점이 같을 때 선을 겹쳐 접는 대신 닫힌 고리로 남는 길이를 표시한다.
        wave += 0.25 * (1 - np.cos(2 * np.pi * parameter))[:, None] * direction
    wave[[0, -1]] = 0.0

    low, high = 0., rest_length
    for _ in range(45):
        amplitude = (low + high) / 2
        points = baseline + amplitude * wave
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        if length < rest_length:
            low = amplitude
        else:
            high = amplitude
    points = baseline + ((low + high) / 2) * wave
    points[0], points[-1] = start, end
    return points
