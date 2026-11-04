"""Showcase figure for the README: the kind of map CrowdMap produces.

CityStreet only, because SCOUT terms forbid sharing SCOUT images. The base map
is Hong Kong Lands Department aerial imagery from web map tiles, so north is
up. The heat layer is our pipeline estimate (SAM head points + 4 marked ground
points per camera), not the official labels.

Needs the outputs of proof.py. Usage:
python showcase_map.py <proof dir> <output.jpg>
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'flowmap'))
import bundle as bundle_io  # noqa: E402
import heatmap  # noqa: E402
import transform  # noqa: E402

SCALE = 2  # upscale so labels stay readable
FONT = cv2.FONT_HERSHEY_SIMPLEX
BLACK, WHITE, GRAY = (0, 0, 0), (255, 255, 255), (60, 60, 60)


def boxed_text(img, text, center, size=.55):
    """Dark text on a white box, kept inside the image."""
    (tw, th), _ = cv2.getTextSize(text, FONT, size, 1)
    x = int(np.clip(center[0] - tw / 2, 4, img.shape[1] - tw - 8))
    y = int(np.clip(center[1], th + 8, img.shape[0] - 8))
    cv2.rectangle(img, (x - 4, y - th - 5), (x + tw + 4, y + 5), WHITE, -1)
    cv2.putText(img, text, (x, y), FONT, size, BLACK, 1, cv2.LINE_AA)


def north_arrow(img):
    x, y = img.shape[1] - 50, img.shape[0] - 100
    cv2.circle(img, (x, y + 34), 30, WHITE, -1, cv2.LINE_AA)
    cv2.fillConvexPoly(img, np.int32([(x, y + 8), (x - 12, y + 50), (x, y + 42), (x + 12, y + 50)]), BLACK, cv2.LINE_AA)
    cv2.putText(img, 'N', (x - 7, y + 4), FONT, .7, BLACK, 2, cv2.LINE_AA)


def scale_bar(img, m_per_px, meters=10):
    px, h = int(round(meters / m_per_px)), img.shape[0]
    cv2.rectangle(img, (16, h - 40), (34 + px, h - 10), WHITE, -1)
    cv2.line(img, (24, h - 18), (24 + px, h - 18), BLACK, 3)
    cv2.putText(img, f'{meters} m', (28, h - 24), FONT, .5, BLACK, 1, cv2.LINE_AA)


def main():
    proof, out = Path(sys.argv[1]), sys.argv[2]
    b = bundle_io.load(proof / 'bundle.json')
    det = json.loads((proof / 'detections.json').read_text())
    res = heatmap.run(b, det, 1.5, point='head')
    heat, covered = res['heat'], res['covered']

    img = res['base'].astype(np.float32)
    img[~covered] = img[~covered] * .45 + 255 * .25  # gray: no camera sees it
    vmax = max(float(np.percentile(heat[covered], 99.5)), 1e-9)
    color = cv2.applyColorMap(np.uint8(np.clip(heat / vmax, 0, 1) * 255), cv2.COLORMAP_JET).astype(np.float32)
    alpha = (np.clip(heat / vmax, 0, 1) ** .6 * .75 * covered)[..., None]
    img = np.uint8(np.clip(img * (1 - alpha) + color * alpha, 0, 255))
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)

    for i, cam in enumerate(b['cameras']):
        c = heatmap.PALETTE[i % len(heatmap.PALETTE)]
        area = transform.view_area(cam) * SCALE
        cv2.polylines(img, [np.int32(np.round(area))], True, c, 3, cv2.LINE_AA)
        p = np.asarray(cam['position'], float) * SCALE
        d = area.mean(0) - p
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        tri = np.int32([p + 26 * d, p - 12 * d + 14 * n, p - 12 * d - 14 * n])
        cv2.fillConvexPoly(img, tri, c, cv2.LINE_AA)
        cv2.polylines(img, [tri], True, WHITE, 2, cv2.LINE_AA)
        boxed_text(img, f'{cam["id"]} ({cam["height_m"]:.1f} m up)', p - 34 * d)
    north_arrow(img)
    scale_bar(img, b['meters_per_pixel'] / SCALE)

    w = img.shape[1]
    top = np.full((92, w, 3), 255, np.uint8)
    cv2.putText(top, 'What CrowdMap makes: a crowd heat map on a top-down map', (12, 28), FONT, .75, BLACK, 2, cv2.LINE_AA)
    top[44:56, 12:212] = cv2.applyColorMap(np.uint8(np.linspace(0, 255, 200))[None, :].repeat(12, 0), cv2.COLORMAP_JET)
    cv2.putText(top, f'0 .. {vmax:.2f} people per m2, mean of {len(res["frames"])} frames, 3 cameras.',
                (222, 55), FONT, .5, BLACK, 1, cv2.LINE_AA)
    cv2.putText(top, 'Outlines: ground each camera sees. Gray: no camera. North is up.', (12, 80), FONT, .5, GRAY, 1, cv2.LINE_AA)
    bottom = np.full((34, w, 3), 255, np.uint8)
    cv2.putText(bottom, 'Data: CityStreet (Zhang and Chan, CityU Hong Kong), CC BY-NC-SA 4.0. '
                        'Map data (c) Lands Department, HKSAR Government.', (12, 22), FONT, .45, GRAY, 1, cv2.LINE_AA)
    final = np.vstack([top, img, bottom])
    cv2.imwrite(out, final, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out, final.shape)


if __name__ == '__main__':
    main()
