"""Build a flow map bundle (format 0.1) for one SCOUT camera on the top-down map.

Stands in for a user marking 4 ground spots by hand: the spots are frame
pixels on mesh-verified ground (official ground-only mesh, round-trip
distortion check), nearest each corner of the covered area and pulled 15%
toward its center. Their map positions come from the official calibration.
The view area is the outline of the covered ground in the frame.

Also reports how far the 4-point homography lands from the calibrated
mesh position over all covered pixels.

Usage: python make_bundle.py <scout data dir> <topdown dir> <video> <output dir> [--camera cam10]
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'flowmap'))
import cameras as cams  # noqa: E402
import transform  # noqa: E402

PULL = .15


def ground_pixels(data, name, frame_w, frame_h):
    """Frame pixels (video size) on covered ground and their scene points."""
    verts, faces = cams.read_ground(data)
    calib = cams.calibration(data, name)
    center, _ = cams.center_and_heading(*calib[1:3])
    px = cams.project(verts, *calib)
    near = np.linalg.norm(verts[:, :2] - center[:2], axis=1) < cams.MAX_RANGE_M
    ok = (~np.isnan(px[:, 0]) & near)[faces].all(1)
    scale = np.array([frame_w / cams.FRAME_W, frame_h / cams.FRAME_H])
    mask = np.zeros((frame_h, frame_w), np.uint8)
    xyz = np.full((frame_h, frame_w, 3), np.nan)
    # Far faces first so near faces overwrite them.
    order = np.flatnonzero(ok)
    depth = np.linalg.norm(verts[faces[order]].mean(1) - center, axis=1)
    for i in order[np.argsort(-depth)]:
        tri = px[faces[i]] * scale
        cv2.fillConvexPoly(mask, np.int32(np.round(tri)), 255)
        x0, y0 = np.floor(tri.min(0)).astype(int)
        x1, y1 = np.ceil(tri.max(0)).astype(int)
        # Barycentric fill of scene coordinates for pixels inside this face.
        gx, gy = np.meshgrid(np.arange(max(x0, 0), min(x1 + 1, frame_w)), np.arange(max(y0, 0), min(y1 + 1, frame_h)))
        pts = np.column_stack([gx.ravel(), gy.ravel()]) + .5
        a, b, c = tri
        m = np.array([b - a, c - a]).T
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        uv = np.linalg.solve(m, (pts - a).T).T
        inside = (uv >= 0).all(1) & (uv.sum(1) <= 1)
        w = np.column_stack([1 - uv.sum(1), uv])[inside]
        q = pts[inside].astype(int)
        xyz[q[:, 1], q[:, 0]] = w @ verts[faces[i]]
    return mask > 0, xyz, center


def marks(covered):
    """Four covered spots spanning the largest quadrilateral, pulled inward."""
    ys, xs = np.nonzero(covered)
    pts = np.column_stack([xs, ys]).astype(float)
    hull = cv2.approxPolyDP(cv2.convexHull(np.float32(pts)), 4, True).reshape(-1, 2).astype(float)
    best, quad = -1, None
    for combo in itertools.combinations(range(len(hull)), 4):
        q = hull[list(combo)]
        area = abs(cv2.contourArea(np.float32(q)))
        if area > best:
            best, quad = area, q
    c = pts.mean(0)
    quad = quad + PULL * (c - quad)
    return np.array([pts[np.argmin(np.hypot(*(pts - p).T))] for p in quad])  # snap onto covered ground


def outline(covered):
    cs, _ = cv2.findContours(np.uint8(covered), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cs, key=cv2.contourArea)
    return cv2.approxPolyDP(big, 3, True).reshape(-1, 2).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('topdown')
    ap.add_argument('video')
    ap.add_argument('out')
    ap.add_argument('--camera', default='cam10')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    covered, xyz, center = ground_pixels(a.data, a.camera, w, h)
    covered &= np.isfinite(xyz[..., 0])
    grid = cams.load_grid(a.topdown)
    frame_pts = marks(covered)
    map_pts = grid.to_px(xyz[frame_pts[:, 1].astype(int), frame_pts[:, 0].astype(int), :2])
    meta = json.loads((Path(a.topdown) / 'cameras.json').read_text())['cameras'][a.camera]
    topdown = Path(a.topdown).resolve()
    ten_m = 10 / grid.m
    spec = {'version': '0.1',
            'map': {'image': os.path.relpath(topdown / 'map.png', out.resolve()),
                    'scale': {'points': [[0, 0], [round(ten_m, 3), 0]], 'meters': 10}},
            'cameras': [{'id': a.camera, 'video': os.path.relpath(Path(a.video).resolve(), out.resolve()),
                         'frame_size': [w, h], 'position': meta['map_px'],
                         'height_m': round(meta['height_above_ground_m'], 2),
                         'ground_points': [{'frame': f.round(1).tolist(), 'map': m.round(2).tolist()}
                                           for f, m in zip(frame_pts, map_pts)],
                         'view_area': outline(covered).tolist()}]}
    (out / 'bundle.json').write_text(json.dumps(spec, indent=1))
    # Homography error against the calibrated mesh position, all covered pixels.
    H = transform.fit(frame_pts, map_pts)
    ys, xs = np.nonzero(covered)
    truth = grid.to_px(xyz[ys, xs, :2])
    err = np.hypot(*(transform.apply(H, np.column_stack([xs, ys]) + .5) - truth).T) * grid.m
    report = {'covered_frame_fraction': float(covered.mean()), 'ground_points_frame': frame_pts.tolist(),
              'homography_vs_mesh_m': {'median': float(np.median(err)), 'p90': float(np.percentile(err, 90)),
                                       'max': float(err.max())}}
    (out / 'bundle-check.json').write_text(json.dumps(report, indent=1))
    ok, frame = cap.read()
    if ok:
        over = frame.copy()
        over[covered] = np.uint8(over[covered] * .55 + np.array([0, 160, 0]) * .45)
        cv2.polylines(over, [np.int32(spec['cameras'][0]['view_area'])], True, (0, 255, 255), 2, cv2.LINE_AA)
        for p in frame_pts:
            cv2.circle(over, tuple(int(v) for v in p), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.imwrite(str(out / 'ground-points.png'), over)
    print(json.dumps(report['homography_vs_mesh_m']))


if __name__ == '__main__':
    main()
