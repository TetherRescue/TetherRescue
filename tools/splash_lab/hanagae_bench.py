"""하나개 크기 맵 렉 측정: 물 계산 범위(패치/바다 전체) x 격자 x 효과 끔/켬. 로봇 포함, 1280x720, 60 fps 기준.

CPU: 물리(로봇 힘 + mj_step) · 효과 계산(철썩 감지 + 물보라 갱신) · 수면 메시 · 장면 구성 · 효과 그리기 목록
GPU: 메시 업로드 · 렌더 — 각각 끝나고 1픽셀 읽기로 GPU 작업이 끝날 때까지 기다려 잼 (GPU 시간 포함)
실행: 저장소 루트에서  python tools/splash_lab/hanagae_bench.py [결과.md]
"""
import os
import sys
import time

root = os.environ.get("TETHER_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

import mujoco
import numpy as np

import splash_lab as L
import hanagae_lab as H

FX_ON = {"spray", "foamtex", "bulge"}
CONFIGS = [  # (이름, full, 메시 격자, 감지 격자, 잴 프레임 수)
    ("패치 50 m · 메시 0.5 m (본 시뮬과 같은 격자)", False, 0.5, 0.25, 120),
    ("패치 50 m · 메시 0.25 m (기본)", False, 0.25, 0.25, 120),
    ("패치 50 m · 메시 0.1 m", False, 0.1, 0.25, 90),
    ("바다 전체 · 메시 1.0 m · 감지 1.0 m", True, 1.0, 1.0, 30),
    ("바다 전체 · 메시 0.5 m · 감지 0.5 m", True, 0.5, 0.5, 20),
]
W, Hh = 1280, 720
STEPS = 8           # 60 fps 한 프레임 = 2 ms x 8 스텝
WARM = 4.0          # 파도가 해안·바위에 닿아 효과가 쌓이도록 먼저 돌리는 시간 [s]


def measure(name, full, mesh, grid, frames, fx_on):
    lab = H.HanagaeLab("fx" if fx_on else "off", fx=FX_ON if fx_on else set(), robot=True, throttle=0.6,
                       full=full, grid=grid, mesh_grid=mesh)
    t_fx = [0.0]
    for meth in ("detect_hits", "update_effects"):
        f = getattr(lab, meth)

        def wrapped(*a, _f=f, **k):
            s = time.perf_counter()
            try:
                return _f(*a, **k)
            finally:
                t_fx[0] += time.perf_counter() - s
        setattr(lab, meth, wrapped)

    r = mujoco.Renderer(lab.m, Hh, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 25, 30, -25
    px = np.zeros((1, 1, 3), np.uint8)
    one = mujoco.MjrRect(0, 0, 1, 1)
    ctx = r._mjr_context

    def sync():
        mujoco.mjr_readPixels(px, None, one, ctx)        # GPU 작업이 끝날 때까지 기다림

    acc = dict(phys=0.0, fx_ms=0.0, mesh=0.0, scene=0.0, draw=0.0, upload=0.0, render=0.0)
    ngeom = []
    n_frames = 0
    total = int(WARM * 60) + frames
    for i in range(total):
        measuring = i >= total - frames
        t_fx[0] = 0.0
        s = time.perf_counter()
        for _ in range(STEPS):
            lab.step()
        step_t = time.perf_counter() - s
        s = time.perf_counter()
        lab.update_water_mesh()
        mesh_t = time.perf_counter() - s
        if not measuring:
            continue
        cam.lookat[:] = (*lab.d.xpos[lab.boat_id][:2], 0.2)
        r._gl_context.make_current()
        sync()
        s = time.perf_counter()
        mujoco.mjr_uploadHField(lab.m, ctx, lab.hfield_id)
        sync()
        up_t = time.perf_counter() - s
        s = time.perf_counter()
        r.update_scene(lab.d, cam, opt)
        scene_t = time.perf_counter() - s
        n0 = r.scene.ngeom
        s = time.perf_counter()
        if lab.fx:
            lab.draw_effects(r.scene, L.cam_position(cam))
        draw_t = time.perf_counter() - s
        ngeom.append(r.scene.ngeom - n0)
        s = time.perf_counter()
        mujoco.mjr_render(r._rect, r.scene, ctx)
        sync()
        ren_t = time.perf_counter() - s
        acc["phys"] += step_t - t_fx[0]; acc["fx_ms"] += t_fx[0]; acc["mesh"] += mesh_t
        acc["scene"] += scene_t; acc["draw"] += draw_t; acc["upload"] += up_t; acc["render"] += ren_t
        n_frames += 1
    r.close()
    ms = {k: 1000 * v / n_frames for k, v in acc.items()}
    cpu = ms["phys"] + ms["fx_ms"] + ms["mesh"] + ms["scene"] + ms["draw"]
    gpu = ms["upload"] + ms["render"]
    return dict(name=name, fx="켬" if fx_on else "끔", mesh_pts=lab.vcol * lab.vrow, grid_cells=lab.ncol * lab.nrow,
                fxgeom=float(np.mean(ngeom)), cpu=cpu, gpu=gpu, frame=cpu + gpu, fps=1000 / (cpu + gpu), **ms)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    for name, full, mesh, grid, frames in CONFIGS:
        for fx_on in (False, True):
            t0 = time.perf_counter()
            row = measure(name, full, mesh, grid, frames, fx_on)
            rows.append(row)
            print(f"{row['name']} | 효과 {row['fx']} | CPU {row['cpu']:.1f} ms · GPU {row['gpu']:.1f} ms · "
                  f"{row['fps']:.0f} fps | ({time.perf_counter() - t0:.0f} s)", flush=True)
    head = ("| 물 계산 범위 · 격자 | 효과 | 물리 | 효과 계산 | 수면 메시 | 장면 구성 | 효과 그리기 | **CPU 합** "
            "| 메시 업로드 | 렌더 | **GPU 합** | **프레임** | fps | 메시 점 | 효과 geom |\n|" + "---|" * 15)
    lines = [head]
    for r in rows:
        lines.append(f"| {r['name']} | {r['fx']} | {r['phys']:.1f} | {r['fx_ms']:.1f} | {r['mesh']:.1f} | {r['scene']:.1f} | "
                     f"{r['draw']:.1f} | **{r['cpu']:.1f}** | {r['upload']:.1f} | {r['render']:.1f} | **{r['gpu']:.1f}** | "
                     f"**{r['frame']:.1f}** | {r['fps']:.0f} | {r['mesh_pts']:,} | {r['fxgeom']:.0f} |")
    table = "\n".join(lines)
    print("\n" + table)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write("단위: ms/프레임 (60 fps 기준 한 프레임 = 물리 8스텝). 예산 16.7 ms\n\n" + table + "\n")


if __name__ == "__main__":
    main()
