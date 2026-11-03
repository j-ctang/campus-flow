"""SCOUT cameras on the top-down map: position, heading and view area.

The view area of a camera is the ground it can see: faces of the official
ground-only mesh whose corners project inside the frame. The distortion
model is only trusted where projecting and undistorting returns to the same
ray (round-trip check). Walls and other occluders are not tested, so the
area can include ground hidden behind buildings.

Usage: python cameras.py <scout data dir> <topdown dir> [--sequence 2] [--highlight cam10]
Writes <topdown dir>/cameras.json, view-areas.npz and camera-map.png.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(2)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topdown import Grid  # noqa: E402

FRAME_W, FRAME_H = 1920, 1080
ROUNDTRIP_TOL = 1e-3
MAX_RANGE_M = 60.0  # beyond this a 1080p camera sees people too small to detect


def load_grid(topdown):
    g = json.loads((Path(topdown) / 'map.json').read_text())
    grid = Grid(g['scene_x_at_px0'], 0, 0, g['scene_y_at_py0'], g['m_per_px'])
    grid.w, grid.h = g['width'], g['height']
    return grid


def sequence_cameras(data, sequence):
    stamps = json.loads((Path(data) / f'timestamps/timestamps_sequence_{sequence}.json').read_text())
    names = set()
    for v in stamps.values():
        names |= set(v)
    return sorted(names, key=lambda n: int(re.sub(r'\D', '', n)))


def calibration(data, name):
    c = json.loads((Path(data) / f'calibrations/cam_{int(re.sub(r"[^0-9]", "", name))}.json').read_text())
    K, R, T, D = (np.array(c[k], float) for k in ('K', 'R', 'T', 'dist'))
    return K, R, T.ravel(), D


def center_and_heading(R, T):
    center = -R.T @ T
    forward = R.T @ np.array([0, 0, 1.0])  # optical axis in scene coordinates
    return center, forward


def project(points, K, R, T, D):
    """Scene points -> frame pixels. NaN where behind the camera, outside the
    frame, or where the distortion model fails the round trip."""
    cam = points @ R.T + T
    z = cam[:, 2]
    px = np.full((len(points), 2), np.nan)
    front = z > 0.5
    if not front.any():
        return px
    p = cv2.projectPoints(np.ascontiguousarray(points[front]), cv2.Rodrigues(R)[0], T, K, D)[0].reshape(-1, 2)
    inside = (p[:, 0] >= 0) & (p[:, 0] < FRAME_W) & (p[:, 1] >= 0) & (p[:, 1] < FRAME_H)
    back = np.full_like(p, np.nan)
    if inside.any():
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
        back[inside] = cv2.undistortPoints(p[inside][:, None], K, D, criteria=crit).reshape(-1, 2)
    ideal = cam[front, :2] / z[front, None]
    good = inside & (np.linalg.norm(back - ideal, axis=1) < ROUNDTRIP_TOL)
    out = np.where(good[:, None], p, np.nan)
    px[front] = out
    return px


def read_ground(data):
    path = Path(data) / 'meshes/mesh_ground_only.ply'
    head = open(path, 'rb').read(4096)
    end = head.index(b'end_header\n') + len('end_header\n')
    nv = int(re.search(rb'element vertex (\d+)', head).group(1))
    nf = int(re.search(rb'element face (\d+)', head).group(1))
    v = np.fromfile(path, '<f4', nv * 3, offset=end).reshape(-1, 3).astype(np.float64)
    f = np.fromfile(path, np.dtype([('n', 'u1'), ('i', '<i4', 3)]), nf, offset=end + nv * 12)
    if not (f['n'] == 3).all():
        raise ValueError('expected triangles')
    return v, f['i']


def view_area(verts, faces, grid, calib, center):
    seen = ~np.isnan(project(verts, *calib)[:, 0])
    near = np.linalg.norm(verts[:, :2] - center[:2], axis=1) < MAX_RANGE_M
    ok = (seen & near)[faces].all(1)
    mask = np.zeros((grid.h, grid.w), np.uint8)
    tri = np.int32(np.round(grid.to_px(verts[faces[ok]][..., :2])))
    for t in tri:
        cv2.fillConvexPoly(mask, t, 1)
    return mask.astype(bool)


def height_above_ground(verts, center, radius_m=10.0):
    """Camera height over the median ground-mesh height within radius_m."""
    near = np.linalg.norm(verts[:, :2] - center[:2], axis=1) < radius_m
    return float(center[2] - np.median(verts[near, 2])) if near.any() else None


COLOR = (255, 170, 0)  # BGR, other cameras
HIGHLIGHT = (0, 60, 255)


def draw(base, grid, cams, areas, highlight):
    img = base.copy()
    for name in cams:
        if name == highlight:
            continue
        tint = img[areas[name]].astype(np.float32)
        img[areas[name]] = np.uint8(tint * .7 + np.array(COLOR) * .3)
    if highlight in areas:
        tint = img[areas[highlight]].astype(np.float32)
        img[areas[highlight]] = np.uint8(tint * .45 + np.array(HIGHLIGHT) * .55)
    placed = []
    for name, c in cams.items():
        p = grid.to_px(np.array(c['center'][:2]))
        row = sum(np.linalg.norm(p - q) < 40 for q in placed)  # stack labels of co-located cameras
        placed.append(p)
        draw_icon(img, grid, c, HIGHLIGHT if name == highlight else COLOR, name, name == highlight, row)
    return img


def draw_icon(img, grid, cam, color, label, big=False, row=0):
    p = grid.to_px(np.array(cam['center'][:2]))
    d = np.array(cam['forward'][:2]) * [1, -1]  # map y points down
    d = d / (np.linalg.norm(d) or 1)
    r = 16 if big else 11
    tip = p + d * r * 2.4
    side = np.array([-d[1], d[0]]) * r * 1.1
    cone = np.int32(np.round([p, tip + side, tip - side]))
    cv2.fillConvexPoly(img, cone, color, cv2.LINE_AA)
    c = tuple(int(v) for v in np.round(p))
    cv2.circle(img, c, r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, c, r, color, 4 if big else 3, cv2.LINE_AA)
    scale = .8 if big else .55
    (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = c[0] + r + 4 if c[0] + r + 4 + tw < img.shape[1] else c[0] - r - 4 - tw
    y = c[1] - r + row * (th_ + 10)
    for col, th in (((0, 0, 0), 5), ((255, 255, 255), 2 if big else 1)):
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('topdown')
    ap.add_argument('--sequence', type=int, default=2)
    ap.add_argument('--highlight', default='cam10')
    a = ap.parse_args()
    grid = load_grid(a.topdown)
    verts, faces = read_ground(a.data)
    cams, areas = {}, {}
    for name in sequence_cameras(a.data, a.sequence):
        calib = calibration(a.data, name)
        center, forward = center_and_heading(*calib[1:3])
        areas[name] = view_area(verts, faces, grid, calib, center)
        cams[name] = {'center': center.tolist(), 'forward': forward.tolist(),
                      'map_px': grid.to_px(center[:2]).round(1).tolist(),
                      'height_above_ground_m': height_above_ground(verts, center),
                      'view_area_m2': float(areas[name].sum() * grid.m ** 2)}
        print(name, cams[name]['map_px'], round(cams[name]['view_area_m2']), flush=True)
    out = Path(a.topdown)
    (out / 'cameras.json').write_text(json.dumps({'sequence': a.sequence, 'cameras': cams}, indent=1))
    np.savez_compressed(out / 'view-areas.npz', **areas)
    base = cv2.imread(str(out / 'map.png'))
    cv2.imwrite(str(out / 'camera-map.png'), draw(base, grid, cams, areas, a.highlight))


if __name__ == '__main__':
    main()
