"""Fetch Hong Kong Lands Department aerial imagery for the CityStreet junction and
register the official CityStreet scene map onto it.

Scale is fixed by the official calibration (0.305 m ground-plane cells), so the
search covers only rotation and translation. Matching uses doubled-angle
gradient orientation, which tolerates the different imagery sources.

Usage: python register_basemap.py <citystreet data dir> <output dir>
Imagery: Map data (c) Lands Department, HKSAR Government (mapapi.geodata.gov.hk).
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Shek Kip Mei Estate Ancillary Facilities Block (HK address lookup), beside the junction.
ANCHOR_LAT, ANCHOR_LON = 22.33316, 114.16713
ZOOM = 20  # highest zoom the imagery service provides here
TILE_URL = 'https://mapapi.geodata.gov.hk/gs/api/v1.0.0/xyz/imagery/WGS84/{z}/{x}/{y}.png'
TILES = range(-3, 5)  # 8 x 8 tiles around the anchor
SCENE_M_PER_PX = 0.305 / 2 * 392 / 438  # official: 392-wide scene = 2x grid of 0.305 m cells


def tile_xy(lat, lon, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def fetch_mosaic(out):
    tiles = out / 'tiles'
    tiles.mkdir(parents=True, exist_ok=True)
    cx, cy = (int(v) for v in tile_xy(ANCHOR_LAT, ANCHOR_LON, ZOOM))
    rows = []
    for dy in TILES:
        row = []
        for dx in TILES:
            x, y = cx + dx, cy + dy
            path = tiles / f'{ZOOM}_{x}_{y}.png'
            if not path.exists():  # cache: one polite request per missing tile
                with urllib.request.urlopen(TILE_URL.format(z=ZOOM, x=x, y=y), timeout=20) as r:
                    path.write_bytes(r.read())
                time.sleep(0.2)
            row.append(cv2.imread(str(path)))
        rows.append(np.hstack(row))
    origin = (cx + TILES[0], cy + TILES[0])
    return np.vstack(rows), origin


def orientation(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy) + 1e-6
    return (gx ** 2 - gy ** 2) / mag, 2 * gx * gy / mag


def register(scene, ortho, ortho_m):
    target = orientation(ortho)
    s = SCENE_M_PER_PX / ortho_m
    h, w = scene.shape[:2]
    best = None
    for angle in range(0, 360, 2):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
        c, si = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * si + w * c), int(h * c + w * si)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        valid = cv2.warpAffine(np.ones((h, w), np.float32), M, (nw, nh)) > .99
        a, b = orientation(cv2.warpAffine(scene, M, (nw, nh)))
        score_map = cv2.matchTemplate(target[0], a * valid, cv2.TM_CCORR) + cv2.matchTemplate(target[1], b * valid, cv2.TM_CCORR)
        _, peak, _, loc = cv2.minMaxLoc(score_map)
        score = peak / valid.sum()
        if best is None or score > best[0]:
            M[:, 2] += loc
            best = (score, angle, M)
    return best


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ortho, origin = fetch_mosaic(out)
    cv2.imwrite(str(out / 'mosaic.png'), ortho)
    ortho_m = 156543.03392 * math.cos(math.radians(ANCHOR_LAT)) / 2 ** ZOOM
    scene = cv2.imread(str(data / 'extracted/CityStreet/ROI_maps/Scene_map/scene_map.jpg'))
    score, angle, M = register(scene, ortho, ortho_m)
    result = {'score': score, 'angle_deg': angle, 'M_scene_to_ortho': M.tolist(), 'ortho_m_per_px': ortho_m,
              'zoom': ZOOM, 'tile_origin': origin,
              'attribution': 'Map data (c) Lands Department, HKSAR Government'}
    (out / 'registration.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ('score', 'angle_deg', 'ortho_m_per_px')}))


if __name__ == '__main__':
    main()
