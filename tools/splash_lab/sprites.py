"""④ 물보라 스프라이트 그림 (투명 배경 RGBA PNG).

처음 실행할 때 이 파일 옆 sprites/ 폴더에 그림을 코드로 만들어 저장하고, 그 뒤로는 파일을 읽는다.
같은 이름의 정사각형 RGBA PNG 로 바꿔 끼우면 그 그림이 쓰인다 (실제 물보라 사진 등).

  puff.png  : 물안개 덩어리 (카메라를 향함)
  burst.png : 위로 솟는 물보라 (아래 가운데가 부딪힌 점, 위쪽이 하늘)
  foam.png  : 수면에 눕히는 거품 무늬
  spray 효과용 작은 조각 (여러 장을 입자처럼 수십 개 뿌림):
  drop0~2.png  : 물방울 무리
  sheet0~1.png : 물막 조각 (그림 위쪽 = 날아가는 방향)
  mist0~1.png  : 옅은 물안개
"""
import os
import struct
import zlib

import numpy as np

NAMES = ("puff", "burst", "foam")
SIZE = 256
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites")


def _value_noise(n, cells, rng):
    g = rng.random((cells + 1, cells + 1))
    t = np.linspace(0, cells, n, endpoint=False)
    i = t.astype(int)
    f = t - i
    f = f * f * (3 - 2 * f)
    a, b = g[np.ix_(i, i)], g[np.ix_(i, i + 1)]
    c, d = g[np.ix_(i + 1, i)], g[np.ix_(i + 1, i + 1)]
    fx, fy = f[None, :], f[:, None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _fbm(n, rng, octaves=(4, 8, 16, 32)):
    out = sum(_value_noise(n, c, rng) / (k + 1) for k, c in enumerate(octaves))
    return (out - out.min()) / (out.max() - out.min())


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _grid(n):
    v = np.linspace(-1, 1, n)
    return np.meshgrid(v, v)          # x: 왼→오, y: 위→아래 (이미지 행 방향)


def make_puff(rng, n=SIZE):
    x, y = _grid(n)
    r = np.hypot(x, y)
    noise = _fbm(n, rng)
    base = np.clip(1 - r / 0.92, 0, 1)
    alpha = _smooth(0.15, 0.65, base * 0.9 + (noise - 0.5) * 0.7) * base ** 0.4
    shade = 0.86 + 0.14 * noise
    return alpha, shade


def make_burst(rng, n=SIZE):
    x, y = _grid(n)
    v = (1 - y) / 2                   # 0 = 그림 아래, 1 = 위
    noise = _fbm(n, rng, (6, 12, 24, 48))
    # 아래쪽 물막(시트): 벽에 부딪혀 부채꼴로 넓게 퍼지고, 위로 갈수록 옅어짐
    cone = np.clip(1 - np.abs(x) / (0.25 + 1.1 * v), 0, 1) * np.clip(1 - v / 0.7, 0, 1)
    alpha = _smooth(0.2, 0.6, cone * (0.6 + 0.8 * noise))
    # 튀는 물방울: 위쪽·바깥으로 갈수록 작고 드묾
    px = np.arange(n)
    for _ in range(520):
        th = rng.normal(0, 0.8)
        s = rng.random() ** 0.6
        cx, cy = 0.95 * s * np.sin(th), 0.95 * s * max(np.cos(th), 0.15)          # cy: 위쪽(+)
        rad = (0.032 * (1 - 0.65 * s) + 0.006) * rng.uniform(0.6, 1.3)
        col, row = (cx + 1) / 2 * (n - 1), (1 - (2 * cy - 1)) / 2 * (n - 1)
        rp = rad * n / 2 + 2
        r0, r1 = int(max(row - rp, 0)), int(min(row + rp + 1, n))
        c0, c1 = int(max(col - rp, 0)), int(min(col + rp + 1, n))
        if r0 >= r1 or c0 >= c1:
            continue
        dd = np.hypot(px[c0:c1][None, :] - col, px[r0:r1][:, None] - row) / (rad * n / 2)
        alpha[r0:r1, c0:c1] = np.maximum(alpha[r0:r1, c0:c1], np.clip(1.3 - dd, 0, 1) * rng.uniform(.6, 1))
    shade = 0.9 + 0.1 * noise
    return np.clip(alpha, 0, 1), shade


def make_foam(rng, n=SIZE):
    x, y = _grid(n)
    r = np.hypot(x, y)
    n1 = _fbm(n, rng, (8, 16, 32, 64))
    n2 = _fbm(n, rng, (5, 10, 20, 40))
    veins = 1 - _smooth(0.0, 0.07, np.abs(n2 - 0.5))        # 거품 사이 그물 무늬
    blobs = _smooth(0.55, 0.75, n1)
    edge = np.clip(1 - r ** 3, 0, 1)
    alpha = np.clip(0.75 * veins + 0.6 * blobs, 0, 1) * edge
    return alpha, 0.92 + 0.08 * n1


def _png_rgba(alpha, shade):
    rgb = np.clip(np.stack([shade * 0.97, shade * 0.99, shade], -1) * 255, 0, 255)
    img = np.concatenate([rgb, (alpha * 255)[..., None]], -1).astype(np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# ── spray: 작은 조각 그림 여러 장 (입자처럼 수십 개를 뿌림) ─────────
def _dots(alpha, rng, n_dots, spread, rad_range, center=(0.0, 0.0), squash=1.0):
    """alpha 위에 부드러운 물방울 점들을 찍는다. 좌표는 -1..1."""
    n = alpha.shape[0]
    px = np.arange(n)
    for _ in range(n_dots):
        r = abs(rng.normal(0, spread))
        a = rng.uniform(0, 2 * np.pi)
        cx, cy = center[0] + r * np.cos(a), center[1] + r * np.sin(a) * squash
        rad = rng.uniform(*rad_range)
        col, row = (cx + 1) / 2 * (n - 1), (1 - cy) / 2 * (n - 1)
        rp = rad * n / 2 + 2
        r0, r1 = int(max(row - rp, 0)), int(min(row + rp + 1, n))
        c0, c1 = int(max(col - rp, 0)), int(min(col + rp + 1, n))
        if r0 >= r1 or c0 >= c1:
            continue
        dd = np.hypot(px[c0:c1][None, :] - col, px[r0:r1][:, None] - row) / (rad * n / 2)
        alpha[r0:r1, c0:c1] = np.maximum(alpha[r0:r1, c0:c1], np.clip(1.25 - dd, 0, 1) * rng.uniform(.65, 1))
    return alpha


def make_drops(rng, n=128):
    """물방울 무리: 가운데 몇 개 크고, 바깥으로 작은 방울이 흩어짐."""
    alpha = np.zeros((n, n))
    _dots(alpha, rng, 6, 0.18, (0.10, 0.18))
    _dots(alpha, rng, 40, 0.45, (0.03, 0.08))
    edge = np.clip(1 - np.hypot(*_grid(n)) ** 4, 0, 1)
    return alpha * edge, 0.92 + 0.08 * _fbm(n, rng, (4, 8))


def make_sheet(rng, n=128):
    """물막 조각: 위아래로 긴 덩어리 + 위쪽 끝에서 떨어져 나가는 방울 (진행 방향 = 그림 위쪽)."""
    x, y = _grid(n)
    noise = _fbm(n, rng, (4, 8, 16, 32))
    body = np.clip(1 - np.hypot(x / 0.32, (y + 0.15) / 0.75), 0, 1)
    alpha = _smooth(0.15, 0.55, body * (0.55 + 0.9 * noise))
    _dots(alpha, rng, 18, 0.22, (0.03, 0.08), center=(0.0, 0.62), squash=0.6)
    return np.clip(alpha, 0, 1), 0.9 + 0.1 * noise


def make_mist(rng, n=128):
    """옅은 물안개: puff 보다 성기고 가장자리가 부드러움."""
    x, y = _grid(n)
    noise = _fbm(n, rng, (3, 6, 12, 24))
    base = np.clip(1 - np.hypot(x, y) / 0.95, 0, 1)
    alpha = _smooth(0.1, 0.7, base * (0.5 + 0.9 * noise)) * base ** 0.6 * 0.8
    return alpha, 0.9 + 0.1 * noise


SPRAY_KINDS = {"drop": 3, "sheet": 2, "mist": 2}          # 종류별 그림 장수 (모양이 조금씩 다름)
NAMES = NAMES + tuple(f"{k}{i}" for k, c in SPRAY_KINDS.items() for i in range(c))


def load_sprites(seed=1):
    """{"spr_puff.png": bytes, ...} — MjModel.from_xml_string 의 assets 로 넘긴다."""
    os.makedirs(DIR, exist_ok=True)
    makers = {"puff": make_puff, "burst": make_burst, "foam": make_foam,
              "drop": make_drops, "sheet": make_sheet, "mist": make_mist}
    out = {}
    for idx, name in enumerate(NAMES):
        path = os.path.join(DIR, f"{name}.png")
        if not os.path.exists(path):
            rng = np.random.default_rng(seed * 1000 + idx)          # 그림마다 다른 시드 → 모양이 다름
            with open(path, "wb") as f:
                f.write(_png_rgba(*makers[name.rstrip("0123456789")](rng)))
        with open(path, "rb") as f:
            out[f"spr_{name}.png"] = f.read()
    return out
