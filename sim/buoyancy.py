"""고정 연결된 형상들의 합집합을 부력 시험용 복셀로 근사한다."""

import itertools

import numpy as np

from sim import config


def body_voxels(model, body_id, spacing):
    """body 좌표 중심점·반크기. 겹친 팔다리·구명환 부피는 한 번만 센다.

    사람·환의 부력용 수치 근사이며 표면 mesh의 고정밀 부피 적분은 아니다.
    """
    import mujoco

    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("복셀 간격은 유한한 양수여야 합니다.")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rotation = data.xmat[body_id].reshape(3, 3)
    origin = data.xpos[body_id]
    geoms = []
    bounds = []
    for gid in range(model.ngeom):
        if model.geom_group[gid] != config.Groups.PHYSICS:
            continue
        owner = int(model.geom_bodyid[gid])
        ancestor = owner
        while ancestor != body_id and ancestor != 0 and model.body_jntnum[ancestor] == 0:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != body_id:
            continue
        kind = int(model.geom_type[gid])
        size = model.geom_size[gid]
        geom_rotation = data.geom_xmat[gid].reshape(3, 3)
        center = (data.geom_xpos[gid] - origin) @ rotation
        to_body = rotation.T @ geom_rotation
        planes = None
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            vertices = np.asarray(model.mesh_vert[start:start + count], dtype=float)
            start, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            triangles = vertices[model.mesh_face[start:start + count]]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            # 이 시뮬레이터의 몸통 mesh는 볼록 외피로 처리한다.
            normals *= np.where(np.sum(normals * (triangles[:, 0] - vertices.mean(axis=0)), axis=1) < 0,
                                -1, 1)[:, None]
            planes = normals, np.sum(normals * triangles[:, 0], axis=1)
            local_bounds = vertices.min(axis=0), vertices.max(axis=0)
        else:
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = np.array([size[0], size[0], size[0] + size[1]])
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = np.full(3, size[0])
            elif kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                extent = size
            else:
                raise ValueError("부력 복셀은 box/sphere/ellipsoid/capsule/convex mesh만 지원합니다.")
            local_bounds = -extent, extent
        corners = np.array(list(itertools.product(*zip(*local_bounds))))
        bounds.append(corners @ to_body.T + center)
        geoms.append((kind, size, center, to_body, planes))
    if not geoms:
        raise ValueError("부력 대상 몸체에 지원하는 형상이 없습니다.")
    bounds = np.concatenate(bounds)
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    counts = np.maximum(1, np.ceil((high - low) / spacing - 1e-10).astype(int))
    cell = (high - low) / counts
    axes = [low[i] + (np.arange(counts[i]) + .5) * cell[i] for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    inside = np.zeros(len(points), dtype=bool)
    for kind, size, center, to_body, planes in geoms:
        local = (points - center) @ to_body
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            hit = np.all(np.abs(local) <= size, axis=1)
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            hit = np.sum((local / size)**2, axis=1) <= 1
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                local[:, 2] -= np.clip(local[:, 2], -size[1], size[1])
            hit = np.sum(local**2, axis=1) <= size[0]**2
        else:
            normals, offsets = planes
            hit = np.all(local @ normals.T <= offsets + 1e-10, axis=1)
        inside |= hit
    if not inside.any():
        raise ValueError("복셀 간격이 너무 커 부력 형상을 표현할 수 없습니다.")
    return points[inside], cell / 2, high - low
