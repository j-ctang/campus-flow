"""Bundle + person detections -> crowd concentration heat map and report.

Detections JSON: {"<camera id>": {"<frame key>": [[x1, y1, x2, y2], ...]}}.
Frame keys are shared across cameras (synchronized cameras). Only frames that
every camera has are used, so each frame covers the same area.
Point modes: "feet" maps the box bottom center through the ground homography.
"head" maps a point near the box top and corrects for head height; it needs
each camera's height_m and is better in crowds where feet are hidden.
Usage: python heatmap.py <bundle.json> <detections.json> <output dir> [--sigma-m 1.5] [--point head]
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle as bundle_io  # noqa: E402
import transform  # noqa: E402

PALETTE = [(255, 170, 0), (60, 220, 60), (230, 80, 255), (0, 200, 255), (255, 255, 0), (80, 80, 255)]  # BGR


def ownership(cameras, shape):
    """Each map pixel belongs to the first listed camera that covers it, so
    overlapping views never count the same person twice."""
    owner = np.full(shape, -1, np.int16)
    for i, cam in enumerate(cameras):
        area = np.zeros(shape, np.uint8)
        cv2.fillPoly(area, [np.int32(np.round(transform.view_area(cam)))], 1)
        owner[(area > 0) & (owner < 0)] = i
    return owner


def to_map(cam, H, boxes, point='feet'):
    """Person boxes in one camera frame -> standing spots in map pixels."""
    if point == 'feet':
        return transform.apply(H, transform.foot_points(boxes))
    hits = transform.apply(H, transform.head_points(boxes))
    return transform.head_to_ground(hits, cam['position'], cam['height_m'])


def map_frame(cameras, homographies, owner, frame_boxes, point='feet'):
    """All people in one synchronized frame, in map pixels, each counted once."""
    out = []
    h, w = owner.shape
    for i, cam in enumerate(cameras):
        pts = to_map(cam, homographies[i], frame_boxes.get(cam['id'], []), point)
        if not len(pts):
            continue
        x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        keep = np.zeros(len(pts), bool)
        keep[ok] = owner[y[ok], x[ok]] == i
        out.append(pts[keep])
    return np.vstack(out) if out else np.zeros((0, 2))


def density(points, shape, sigma_px, covered):
    """People per map pixel, smoothed; mass outside coverage is dropped."""
    g = np.zeros(shape, np.float64)
    if len(points):
        x = np.clip(np.round(points[:, 0]).astype(int), 0, shape[1] - 1)
        y = np.clip(np.round(points[:, 1]).astype(int), 0, shape[0] - 1)
        np.add.at(g, (y, x), 1)
        g = cv2.GaussianBlur(g, (0, 0), sigma_px)
    return g * covered


def run(bundle, detections, sigma_m=1.5, point='feet'):
    if point not in ('feet', 'head'):
        raise ValueError('point must be "feet" or "head"')
    if point == 'head' and any('height_m' not in c for c in bundle['cameras']):
        raise ValueError('head mode needs height_m for every camera')
    base = cv2.imread(str(bundle['map_image']))
    if base is None:
        raise FileNotFoundError(bundle['map_image'])
    cams = bundle['cameras']
    Hs = [transform.fit(c['frame_points'], c['map_points']) for c in cams]
    owner = ownership(cams, base.shape[:2])
    covered = owner >= 0
    keys = sorted(set.intersection(*(set(detections.get(c['id'], {})) for c in cams)))
    if not keys:
        raise ValueError('no frame key is present for every camera')
    m = bundle['meters_per_pixel']
    per_frame = {k: map_frame(cams, Hs, owner, {c['id']: detections[c['id']][k] for c in cams}, point) for k in keys}
    # Smoothing is linear, so smoothing all frames' points at once equals
    # summing per-frame maps, at one blur instead of one per frame.
    total = density(np.vstack(list(per_frame.values())), base.shape[:2], sigma_m / m, covered)
    mean_per_m2 = total / len(keys) / m ** 2
    return {'base': base, 'owner': owner, 'covered': covered, 'heat': mean_per_m2, 'frames': keys,
            'points': per_frame, 'homographies': Hs}


def render(bundle, result, path, title):
    base, heat, covered = result['base'], result['heat'], result['covered']
    img = base.astype(np.float32)
    img[~covered] = img[~covered] * .45 + 255 * .25
    vmax = float(np.percentile(heat[covered], 99.5)) if covered.any() else 1.0
    vmax = max(vmax, 1e-9)
    color = cv2.applyColorMap(np.uint8(np.clip(heat / vmax, 0, 1) * 255), cv2.COLORMAP_JET).astype(np.float32)
    alpha = (np.clip(heat / vmax, 0, 1) ** .6 * .75 * covered)[..., None]
    img = np.uint8(np.clip(img * (1 - alpha) + color * alpha, 0, 255))
    for i, cam in enumerate(bundle['cameras']):
        c = PALETTE[i % len(PALETTE)]
        cv2.polylines(img, [np.int32(np.round(transform.view_area(cam)))], True, c, 2, cv2.LINE_AA)
        p = tuple(int(v) for v in np.round(cam['position']))
        cv2.circle(img, p, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 9, c, 3, cv2.LINE_AA)
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(img, cam['id'], (p[0] + 12, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .55, col, th, cv2.LINE_AA)
    pad = np.full((64, img.shape[1], 3), 255, np.uint8)
    cv2.putText(pad, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2, cv2.LINE_AA)
    bar = cv2.applyColorMap(np.uint8(np.linspace(0, 255, 200))[None, :].repeat(10, 0), cv2.COLORMAP_JET)
    pad[40:50, 10:210] = bar
    cv2.putText(pad, f'0 .. {vmax:.2f} people/m2 (mean over {len(result["frames"])} frames). Gray: no camera coverage.',
                (218, 50), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([pad, img]))
    return vmax


def report(bundle, result):
    m = bundle['meters_per_pixel']
    heat = result['heat']
    peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
    counts = [len(p) for p in result['points'].values()]
    return {'frames': len(result['frames']), 'people_per_frame_mean': float(np.mean(counts)),
            'people_per_frame_max': int(max(counts)), 'covered_area_m2': float(result['covered'].sum() * m ** 2),
            'peak_people_per_m2': float(heat.max()), 'peak_map_px': [int(peak[1]), int(peak[0])],
            'cameras': [c['id'] for c in bundle['cameras']]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bundle')
    ap.add_argument('detections')
    ap.add_argument('out')
    ap.add_argument('--sigma-m', type=float, default=1.5)
    ap.add_argument('--point', choices=('feet', 'head'), default='feet')
    ap.add_argument('--title', default='Crowd concentration')
    a = ap.parse_args()
    b = bundle_io.load(a.bundle)
    result = run(b, json.loads(Path(a.detections).read_text()), a.sigma_m, a.point)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    render(b, result, out / 'heatmap.png', a.title)
    rep = report(b, result)
    (out / 'report.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
