"""Pre-made CityStreet map: aerial base, every camera, and the ground each sees.

Optional data layer: crowd concentration from the official labels, colored on
top of the same base. Run register_basemap.py first.
Usage: python camera_map.py <citystreet data dir> <basemap dir> <output dir>
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cameras import VIEWS, center_mm, label_to_scene_original, world_to_label  # noqa: E402
from scene import grid_to_scene, METERS_PER_GRID_CELL  # noqa: E402

COLORS = {'view1': (255, 170, 0), 'view2': (60, 220, 60), 'view3': (230, 80, 255)}  # BGR
MARGIN = 60
SCALE = 2  # final upscale for readable labels


def to_ortho(grid_map, M, shape, interp=cv2.INTER_LINEAR):
    """192x160 grid map -> aerial mosaic pixels, through the official scene padding."""
    scene = cv2.resize(grid_to_scene(grid_map).astype(np.float32), (438, 461), interpolation=interp)
    return cv2.warpAffine(scene, M, (shape[1], shape[0]), flags=interp)


def to_ortho_point(scene_xy, M):
    return M[:, :2] @ np.asarray(scene_xy, float) + M[:, 2]


def draw_camera(img, p, toward, color, label):
    p = np.asarray(p, float)
    d = np.asarray(toward, float) - p
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    tri = np.int32([p + 26 * d, p - 12 * d + 14 * n, p - 12 * d - 14 * n])
    cv2.fillConvexPoly(img, tri, color, cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(np.int32(p)), 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.int32(p)), 9, color, 3, cv2.LINE_AA)
    org = tuple(np.int32(p - 40 * d + (-60, 6)))
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)


def header(width, title, lines):
    pad = np.full((34 + 20 * len(lines), width, 3), 255, np.uint8)
    cv2.putText(pad, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(pad, line, (12, 50 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, .48, (60, 60, 60), 1, cv2.LINE_AA)
    return pad


def scale_bar(img, m_per_px):
    px = int(round(10 / m_per_px))
    h = img.shape[0]
    cv2.rectangle(img, (16, h - 36), (24 + px, h - 12), (255, 255, 255), -1)
    cv2.line(img, (20, h - 18), (20 + px, h - 18), (0, 0, 0), 3)
    cv2.putText(img, '10 m', (24, h - 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)


def build(data, basemap, out, heat_grid=None, heat_note=''):
    reg = json.loads((basemap / 'registration.json').read_text())
    M = np.array(reg['M_scene_to_ortho'])
    ortho = cv2.imread(str(basemap / 'mosaic.png'))
    rois = data / 'extracted/CityStreet/ROI_maps/ROIs/ground_plane'
    masks = {v: to_ortho(np.load(rois / f'{v}_GP_mask.npz')['arr_0'] > 0, M, ortho.shape, cv2.INTER_NEAREST) > .5 for v in VIEWS}
    union = np.any(list(masks.values()), axis=0)
    cams = {v: to_ortho_point(label_to_scene_original(world_to_label(center_mm(v)[:2])), M) for v in VIEWS}

    img = ortho.astype(np.float32)
    outside = ~union
    img[outside] = img[outside] * .45 + 255 * .25  # dim everything no camera sees
    if heat_grid is not None:
        heat = to_ortho(heat_grid, M, ortho.shape)
        vmax = float(np.percentile(heat[union], 99.5))
        color = cv2.applyColorMap(np.uint8(np.clip(heat / vmax, 0, 1) * 255), cv2.COLORMAP_JET).astype(np.float32)
        alpha = (np.clip(heat / vmax, 0, 1) ** .6 * .75 * union)[..., None]
        img = img * (1 - alpha) + color * alpha
    else:
        for v, m in masks.items():  # each camera's visible ground, tinted
            img[m] = img[m] * .72 + np.array(COLORS[v], np.float32) * .28
    img = np.uint8(np.clip(img, 0, 255))
    for v, m in masks.items():
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, COLORS[v], 2, cv2.LINE_AA)

    ys, xs = np.nonzero(union)
    pts = np.vstack([np.column_stack([xs, ys]), np.array(list(cams.values()))])
    x0, y0 = np.maximum(pts.min(0).astype(int) - MARGIN, 0)
    x1, y1 = np.minimum(pts.max(0).astype(int) + MARGIN, ortho.shape[1::-1])
    img = cv2.resize(img[y0:y1, x0:x1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    for v, p in cams.items():
        ys, xs = np.nonzero(masks[v])
        toward = (np.array([xs.mean(), ys.mean()]) - (x0, y0)) * SCALE
        height = -center_mm(v)[2] / 1000
        draw_camera(img, (p - (x0, y0)) * SCALE, toward, COLORS[v], f'{VIEWS[v]["camera"]} ({height:.1f} m up)')
    scale_bar(img, reg['ortho_m_per_px'] / SCALE)
    if heat_grid is None:
        title = 'CityStreet cameras and what each one sees'
        lines = ['Shek Kip Mei, Hong Kong. Tinted area: ground each camera sees (official ROI). Dimmed: no camera coverage.',
                 'Camera positions from official calibration. ' + reg['attribution'] + '.']
    else:
        title = 'CityStreet crowd concentration'
        lines = [heat_note + ' Blue = few people, red = most. Dimmed: no camera coverage.',
                 'Camera positions from official calibration. ' + reg['attribution'] + '.']
    final = np.vstack([header(img.shape[1], title, lines), img])
    cv2.imwrite(str(out), final)
    return {'file': str(out), 'size': final.shape[1::-1], 'cameras_px': {v: p.tolist() for v, p in cams.items()}}


def main():
    data, basemap, out = (Path(a) for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    result = [build(data, basemap, out / 'citystreet-cameras.png')]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'concentration'))
    from label_map import frame_maps, load_inputs
    labels, mask, _ = load_inputs(data)
    _, maps, _ = frame_maps(labels, mask, 1.5 / METERS_PER_GRID_CELL)
    note = f'Official labels, mean of {len(maps)} frames, 1.5 m kernel.'
    result.append(build(data, basemap, out / 'citystreet-concentration-labels.png', maps.mean(0), note))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
