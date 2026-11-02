"""SCOUT demo map: whole-scene top-down render, every camera of the sequence
with its view area, and the crowd heat layer where processed cameras see.

Heat is drawn only on ground that a processed camera sees (mesh-verified
view area). Everything else is gray: no coverage, or not processed yet.

Usage: python demo_map.py <topdown dir> <bundle.json> <detections.json> <output dir>
       [--highlight cam10] [--sigma-m 1.0] [--point feet]
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'flowmap'))
import bundle as bundle_io  # noqa: E402
import cameras as cams  # noqa: E402
import heatmap  # noqa: E402

OTHER = (255, 170, 0)  # BGR
HIGHLIGHT = (0, 60, 255)


def gray(img, keep):
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = img.astype(np.float32)
    out[~keep] = g[~keep] * .55 + 255 * .3
    return out


def compose(base, grid, cam_meta, areas, heat, covered, highlight):
    img = gray(base, covered)
    vmax = max(float(np.percentile(heat[covered], 99.5)) if covered.any() else 1.0, 1e-9)
    level = np.clip(heat / vmax, 0, 1)
    color = cv2.applyColorMap(np.uint8(level * 255), cv2.COLORMAP_JET).astype(np.float32)
    alpha = (np.where(covered, .25 + .6 * level ** .6, 0))[..., None]
    img = np.uint8(np.clip(img * (1 - alpha) + color * alpha, 0, 255))
    for name, area in areas.items():
        cs, _ = cv2.findContours(np.uint8(area), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, HIGHLIGHT if name == highlight else OTHER, 3 if name == highlight else 2, cv2.LINE_AA)
    placed = []
    for name, c in cam_meta.items():
        p = grid.to_px(np.array(c['center'][:2]))
        row = sum(np.linalg.norm(p - q) < 40 for q in placed)
        placed.append(p)
        cams.draw_icon(img, grid, c, HIGHLIGHT if name == highlight else OTHER, name, name == highlight, row)
    return img, vmax


def legend(width, vmax, frames, seconds, people, highlight, n_cams):
    pad = np.full((150, width, 3), 255, np.uint8)
    t = lambda s, y, sc=.7, th=2: cv2.putText(pad, s, (16, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th, cv2.LINE_AA)  # noqa: E731
    t('SCOUT campus (EPFL): crowd concentration', 32, .9)
    t(f'Heat: {highlight}, {frames} frames ({seconds / 60:.0f} min), {people:.1f} people per frame on average.', 64, .55, 1)
    bar = cv2.applyColorMap(np.uint8(np.linspace(0, 255, 260))[None, :].repeat(14, 0), cv2.COLORMAP_JET)
    pad[80:94, 16:276] = bar
    cv2.putText(pad, f'0 .. {vmax:.2f} people/m2', (286, 93), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
    t(f'Outlines: ground each of the {n_cams} cameras sees. Gray: no data yet or no camera.', 120, .55, 1)
    t('Data: SCOUT dataset, CVLAB EPFL. Mesh render; scene axes, not north-up.', 142, .5, 1)
    return pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('topdown')
    ap.add_argument('bundle')
    ap.add_argument('detections')
    ap.add_argument('out')
    ap.add_argument('--highlight', default='cam10')
    ap.add_argument('--sigma-m', type=float, default=1.0)
    ap.add_argument('--point', default='feet', choices=('feet', 'head'))
    ap.add_argument('--fps', type=float, default=10.0)
    a = ap.parse_args()
    top, out = Path(a.topdown), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = cams.load_grid(top)
    meta = json.loads((top / 'cameras.json').read_text())['cameras']
    areas = dict(np.load(top / 'view-areas.npz'))
    b = bundle_io.load(a.bundle)
    result = heatmap.run(b, json.loads(Path(a.detections).read_text()), a.sigma_m, a.point)
    processed = [c['id'] for c in b['cameras']]
    covered = result['covered'] & np.any([areas[c] for c in processed], 0)
    m = b['meters_per_pixel']
    pts = {k: p for k, p in result['points'].items()}
    total = heatmap.density(np.vstack(list(pts.values())), covered.shape, a.sigma_m / m, covered)
    heat = total / len(pts) / m ** 2
    base = cv2.imread(str(top / 'map.png'))
    img, vmax = compose(base, grid, meta, areas, heat, covered, a.highlight)
    people = float(np.mean([len(p) for p in pts.values()]))
    final = np.vstack([legend(img.shape[1], vmax, len(pts), len(pts) / a.fps, people, a.highlight, len(meta)), img])
    cv2.imwrite(str(out / 'demo-map.png'), final)
    # Zoom on the processed area.
    ys, xs = np.nonzero(covered)
    pad = 120
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    zoom = img[y0:y1, x0:x1]
    zoom = cv2.resize(zoom, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out / 'demo-map-zoom.png'), np.vstack([legend(zoom.shape[1], vmax, len(pts), len(pts) / a.fps, people,
                                                                   a.highlight, len(meta)), zoom]))
    peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
    report = {'frames': len(pts), 'people_per_frame_mean': people,
              'people_per_frame_max': int(max(len(p) for p in pts.values())),
              'covered_area_m2': float(covered.sum() * m ** 2), 'peak_people_per_m2': float(heat.max()),
              'peak_map_px': [int(peak[1]), int(peak[0])], 'sigma_m': a.sigma_m, 'point': a.point}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
