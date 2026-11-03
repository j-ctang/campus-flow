"""Proof: does the simple bundle input (4 hand-marked ground points per camera)
reproduce CityStreet's ground-truth crowd concentration?

Steps (all local, cached SAM boxes only, no network):
1. Build a v0.1 bundle on the aerial base map. Ground point pairs are what a
   user would mark: 4 well-spread ground spots per camera. Their map positions
   come from the official calibration, standing in for careful clicking.
2. Run the generic flowmap pipeline on cached SAM person boxes.
3. Score against the official ground-plane labels (answer key), and against
   mapping the same boxes with the full official calibration.
4. Repeat with simulated clicking error on the frame points.
5. Repeat step 2 with head points instead of feet.

Usage: python proof.py <citystreet data dir> <basemap dir> <sam batch dir> <output dir>
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'flowmap'))
from cameras import VIEWS, center_mm, label_to_scene_original  # noqa: E402
import bundle as bundle_io  # noqa: E402
import heatmap  # noqa: E402
import transform  # noqa: E402

CAMERA_VIEW = {v['camera']: k for k, v in VIEWS.items()}
# Per-view world->label offsets, fit on TRAIN frames 636..1234 only (official
# readme offset (281.6, 417.6) leaves 1-2 m view-specific shifts vs the labels).
# Test frames 1236+ are never used for fitting.
LABEL_OFFSET = {'camera1': (292.27893186796325, 416.0818686204356),
                'camera3': (270.71468805313606, 429.47065420753427),
                'camera4': (284.6153152748718, 391.0081121591906)}
FIRST_TEST_FRAME = 1236
MARGIN = 40
CLICK_NOISE_PX = (0, 5, 15, 30)  # frame pixels at 2704x1520
TRIALS = 10


def load_official(data):
    path = data / 'extracted/CityStreet/projection_code/camera_proj_Zhang.py'
    spec = importlib.util.spec_from_file_location('camera_proj_Zhang', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Chain:
    """Official world/label coordinates -> pixels of the cropped aerial map."""

    def __init__(self, official, basemap):
        self.P = official
        reg = json.loads((basemap / 'registration.json').read_text())
        self.M = np.array(reg['M_scene_to_ortho'])
        self.m_per_px = reg['ortho_m_per_px']
        self.ortho = cv2.imread(str(basemap / 'mosaic.png'))
        self.crop = (0, 0)

    def label_to_map(self, label_xy):
        s = label_to_scene_original(label_xy)
        q = np.asarray(s, float).reshape(-1, 2) @ self.M[:, :2].T + self.M[:, 2]
        return q - self.crop

    def ground_to_map(self, camera, pixels):
        px = np.asarray(pixels, float).reshape(-1, 2)
        w = self.P.Image2World(CAMERA_VIEW[camera], px[:, 0], px[:, 1], np.zeros(len(px)))
        label = np.stack(w[:2], 1) / 76.25 + LABEL_OFFSET[camera]
        return self.label_to_map(label)


def ground_marks(data, camera, chain):
    """Four well-spread ground spots, like a user marking far-apart road marks
    on ground the camera clearly sees: frame pixels whose ground point lies in
    the official ground ROI of this view, nearest each image corner, pulled 15%
    toward their center. Points near the horizon are avoided."""
    v = CAMERA_VIEW[camera][-1]
    roi = cv2.imread(str(data / f'extracted/CityStreet/ROI_maps/ROIs/camera_view/mask{v}.jpg'), 0) > 127
    gp = np.load(data / f'extracted/CityStreet/ROI_maps/ROIs/ground_plane/view{v}_GP_mask.npz')['arr_0'] > 0
    ys, xs = np.nonzero(roi[::20, ::20])
    px = np.column_stack([xs * 20, ys * 20]).astype(float)
    w = chain.P.Image2World(CAMERA_VIEW[camera], px[:, 0], px[:, 1], np.zeros(len(px)))
    cell = (np.stack(w[:2], 1) / 76.25 + LABEL_OFFSET[camera]) / 4  # 160x192 ROI grid
    gx, gy = np.round(cell[:, 0]).astype(int), np.round(cell[:, 1]).astype(int)
    ok = (gx >= 0) & (gx < gp.shape[1]) & (gy >= 0) & (gy < gp.shape[0])
    ok[ok] = gp[gy[ok], gx[ok]]
    pts = px[ok]
    c = pts.mean(0)
    h, wd = roi.shape
    marks = [pts[np.argmin(np.hypot(*(pts - corner).T))] for corner in ((0, 0), (wd, 0), (wd, h), (0, h))]
    marks = np.array([p + .15 * (c - p) for p in marks])
    seen = np.zeros_like(roi)
    seen[ys[ok] * 20, xs[ok] * 20] = True
    return marks, seen


def sam_boxes(batch, camera, frame):
    files = glob.glob(str(batch / f'{camera}-{frame:04d}-*.json'))
    if len(files) < 4:
        return None
    boxes = []
    for p in files:
        ox, oy = map(int, re.search(r'-(\d+)_(\d+)\.json$', p).groups())
        r = json.loads(Path(p).read_text())
        text = ''.join(c.get('text', '') for o in r['output'] for c in o.get('content', []))
        boxes += [(a + ox, b + oy, c + ox, d + oy) for a, b, c, d in
                  (tuple(map(int, g)) for g in re.findall(r'box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+)', text))]
    keep = []  # tiles overlap: drop duplicates of the same person
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(b, k) < .5 for k in keep):
            keep.append(b)
    return [list(b) for b in keep]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def match(a, b, radius):
    if not len(a) or not len(b):
        return 0
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    used_a, used_b, n = set(), set(), 0
    for i, j in sorted(zip(*np.nonzero(d < radius)), key=lambda ij: d[ij]):
        if i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j)
            n += 1
    return n


def score(points, answers, shape, covered, m_per_px, sigma_m=1.5):
    frames = sorted(points)
    lab = sum(heatmap.density(answers[f], shape, sigma_m / m_per_px, covered) for f in frames) / len(frames)
    est = sum(heatmap.density(points[f], shape, sigma_m / m_per_px, covered) for f in frames) / len(frames)
    n_lab = [len(answers[f]) for f in frames]
    n_est = [len(points[f]) for f in frames]
    out = {'frames': len(frames), 'labels_per_frame': float(np.mean(n_lab)), 'detected_per_frame': float(np.mean(n_est)),
           'count_correlation': float(np.corrcoef(n_lab, n_est)[0, 1]),
           'map_correlation': float(np.corrcoef(lab[covered], est[covered])[0, 1])}
    for r in (1, 2):
        hits = sum(match(points[f], answers[f], r / m_per_px) for f in frames)
        out[f'recall_{r}m'] = hits / sum(n_lab)
        out[f'precision_{r}m'] = hits / max(sum(n_est), 1)
    top = lambda d: d >= np.percentile(d[covered], 90)  # noqa: E731
    out['hot_area_iou_top10pct'] = float(((top(lab) & top(est)) & covered).sum() / ((top(lab) | top(est)) & covered).sum())
    return out, lab, est


def main():
    data, basemap, batch, out = (Path(a) for a in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    chain = Chain(load_official(data), basemap)

    # 1. Bundle: crop the aerial map to the cameras and their marked areas.
    marks = {c: ground_marks(data, c, chain) for c in LABEL_OFFSET}
    full = {c: chain.ground_to_map(c, marks[c][0]) for c in marks}
    cams_full = {c: chain.label_to_map(np.asarray(center_mm(CAMERA_VIEW[c])[:2]) / 76.25 + LABEL_OFFSET[c])[0] for c in marks}
    allp = np.vstack(list(full.values()) + [np.array(list(cams_full.values()))])
    x0, y0 = np.maximum(allp.min(0).astype(int) - MARGIN, 0)
    x1, y1 = allp.max(0).astype(int) + MARGIN
    cv2.imwrite(str(out / 'map.png'), chain.ortho[y0:y1, x0:x1])
    chain.crop = (x0, y0)
    cameras = []
    for c in marks:
        frame_pts = marks[c][0]
        cameras.append({'id': c, 'video': f'image_frames/{c}/', 'frame_size': [2704, 1520],
                        'position': (cams_full[c] - (x0, y0)).round(1).tolist(),
                        'height_m': round(-float(center_mm(CAMERA_VIEW[c])[2]) / 1000, 2),  # world z points down
                        'ground_points': [{'frame': f.round(1).tolist(), 'map': m.round(2).tolist()}
                                          for f, m in zip(frame_pts, chain.ground_to_map(c, frame_pts))]})
    ten_m = 10 / chain.m_per_px
    spec = {'version': '0.1', 'map': {'image': 'map.png', 'scale': {'points': [[0, 0], [round(ten_m, 3), 0]], 'meters': 10}},
            'cameras': cameras}
    (out / 'bundle.json').write_text(json.dumps(spec, indent=1))
    b = bundle_io.load(out / 'bundle.json')

    # 2. Detections from cached SAM tiles, synchronized test frames only.
    frames = sorted({int(re.search(r'-(\d{4})-', p).group(1)) for p in glob.glob(str(batch / '*.json'))})
    frames = [f for f in frames if f >= FIRST_TEST_FRAME]  # offsets were fit on earlier frames
    det = {c: {} for c in marks}
    for f in frames:
        per = {c: sam_boxes(batch, c, f) for c in marks}
        if all(v is not None for v in per.values()):
            for c, v in per.items():
                det[c][str(f)] = v
    (out / 'detections.json').write_text(json.dumps(det))
    result = heatmap.run(b, det, 1.5)
    heatmap.render(b, result, out / 'heatmap-sam-4points.png', 'CityStreet: SAM + 4 marked points per camera')

    # 3. Answer key and full-calibration baseline, on the same map and coverage.
    with h5py.File(data / 'CityStreet_corrected_labels/Street_groundplane_pmap.h5') as f:
        gp = np.array(f['v_pmap_GP'])
    covered, owner, shape = result['covered'], result['owner'], result['covered'].shape

    def inside(pts, cam_index=None):
        x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
        ok = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
        keep = np.zeros(len(pts), bool)
        keep[ok] = covered[y[ok], x[ok]] if cam_index is None else owner[y[ok], x[ok]] == cam_index
        return pts[keep]

    answers = {k: inside(chain.label_to_map(gp[gp[:, 0] == int(k), 2:4])) for k in result['frames']}
    calib = {}
    for k in result['frames']:
        calib[k] = np.vstack([inside(chain.ground_to_map(c, transform.foot_points(det[c][k])), i) for i, c in enumerate(marks)])
    head = heatmap.run(b, det, 1.5, point='head')
    heatmap.render(b, head, out / 'heatmap-sam-4points-head.png', 'CityStreet: SAM head points + 4 marked points per camera')
    summary = {'sam_4points': score(result['points'], answers, shape, covered, chain.m_per_px)[0],
               'sam_4points_head': score(head['points'], answers, shape, covered, chain.m_per_px)[0],
               'sam_full_calibration': score(calib, answers, shape, covered, chain.m_per_px)[0]}

    # Mapping error of the 4-point homography vs official calibration, per camera, on ROI ground pixels.
    rng = np.random.default_rng(0)
    mapping = {}
    for i, c in enumerate(marks):
        frame_pts, roi = marks[c]
        ys, xs = np.nonzero(roi)
        grid = np.column_stack([xs, ys]).astype(float)
        truth = chain.ground_to_map(c, grid)
        good = covered[np.clip(truth[:, 1].astype(int), 0, shape[0] - 1), np.clip(truth[:, 0].astype(int), 0, shape[1] - 1)]
        mapping[c] = {}
        for noise in CLICK_NOISE_PX:
            errs = []
            for _ in range(TRIALS if noise else 1):
                noisy = frame_pts + rng.normal(0, noise, frame_pts.shape) if noise else frame_pts
                H = transform.fit(noisy, b['cameras'][i]['map_points'])
                errs.append(np.hypot(*(transform.apply(H, grid[good]) - truth[good]).T) * chain.m_per_px)
            e = np.concatenate(errs)
            mapping[c][f'click_noise_{noise}px'] = {'median_m': float(np.median(e)), 'p90_m': float(np.percentile(e, 90))}
    summary['mapping_error_vs_calibration'] = mapping

    # 4. End-to-end with clicking error (one seeded trial per noise level).
    noisy_runs = {}
    for noise in CLICK_NOISE_PX[1:]:
        nb = dict(b, cameras=[dict(cam, frame_points=cam['frame_points'] + rng.normal(0, noise, (4, 2))) for cam in b['cameras']])
        r = heatmap.run(nb, det, 1.5)
        noisy_runs[f'click_noise_{noise}px'] = score(r['points'], answers, shape, covered, chain.m_per_px)[0]
    summary['sam_4points_with_click_noise'] = noisy_runs

    # Answer-key heat map on the same canvas, same renderer.
    key = dict(result, heat=score(result['points'], answers, shape, covered, chain.m_per_px)[1] / chain.m_per_px ** 2)
    heatmap.render(b, key, out / 'heatmap-labels.png', 'CityStreet: answer key (official ground labels)')
    (out / 'proof.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
