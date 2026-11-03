"""Render CityStreet crowd concentration from the official ground-plane labels.

Ground truth only: no inference, no network. Usage:
  python label_map.py <citystreet data dir> <output dir>
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'concentration'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from density import kernel_grid  # noqa: E402
from scene import GRID_SHAPE, METERS_PER_GRID_CELL, grid_to_scene, label_to_grid  # noqa: E402

SIGMAS_M = (0.9, 1.5, 2.4)
UPSCALE = 2


def load_inputs(data):
    with h5py.File(data / 'CityStreet_corrected_labels/Street_groundplane_pmap.h5') as f:
        labels = np.array(f['v_pmap_GP'])
    rois = data / 'extracted/CityStreet/ROI_maps'
    mask = np.zeros(GRID_SHAPE, bool)
    for view in (1, 2, 3):
        mask |= np.load(rois / f'ROIs/ground_plane/view{view}_GP_mask.npz')['arr_0'] > 0
    scene = cv2.imread(str(rois / 'Scene_map/scene_map.jpg'))
    if scene is None:
        raise FileNotFoundError('scene_map.jpg missing')
    return labels, mask, scene


def frame_maps(labels, mask, sigma_cells):
    frames = np.unique(labels[:, 0]).astype(int)
    maps, dropped = [], 0
    for frame in frames:
        points = label_to_grid(labels[labels[:, 0] == frame, 2:4])
        inside = [p for p in points if mask[int(np.clip(p[1], 0, GRID_SHAPE[0] - 1)), int(np.clip(p[0], 0, GRID_SHAPE[1] - 1))]]
        dropped += len(points) - len(inside)
        maps.append(kernel_grid(inside, mask, sigma_cells))
    return frames, np.array(maps), dropped


def render(path, heat_per_m2, mask, scene, vmax, title, subtitle):
    h, w = scene.shape[:2]
    base = cv2.cvtColor(cv2.cvtColor(cv2.resize(scene, (392, 412)), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    heat = grid_to_scene(heat_per_m2)
    inside = grid_to_scene(mask.astype(np.uint8)) > 0
    color = cv2.applyColorMap(np.uint8(np.clip(heat / vmax, 0, 1) * 255), cv2.COLORMAP_INFERNO)
    out = (base * 0.35 + 90).astype(np.uint8)  # outside camera coverage: flat light gray
    out[inside] = (0.25 * base[inside] + 0.75 * color[inside]).astype(np.uint8)
    out = cv2.resize(out, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
    edge = cv2.resize(inside.astype(np.uint8), None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    pad = np.full((70, out.shape[1], 3), 255, np.uint8)
    cv2.putText(pad, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(pad, subtitle, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .42, (60, 60, 60), 1, cv2.LINE_AA)
    bar = cv2.applyColorMap(np.uint8(np.linspace(0, 255, 200))[None, :].repeat(10, 0), cv2.COLORMAP_INFERNO)
    pad[52:62, 8:208] = bar
    cv2.putText(pad, f'0 .. {vmax:.2f} people/m2   gray = no camera coverage', (216, 61), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([pad, out]))


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    labels, mask, scene = load_inputs(data)
    cell_area = METERS_PER_GRID_CELL ** 2
    report = {'source': 'official corrected ground-plane labels', 'labeled_frames': int(len(np.unique(labels[:, 0]))),
              'people_labels': int(len(labels)), 'meters_per_cell': METERS_PER_GRID_CELL, 'maps': []}
    for sigma_m in SIGMAS_M:
        sigma = sigma_m / METERS_PER_GRID_CELL
        frames, maps, dropped = frame_maps(labels, mask, sigma)
        n = len(frames)
        windows = {'all': slice(0, n), 'first50': slice(0, 50), 'middle50': slice(n // 2 - 25, n // 2 + 25), 'last50': slice(n - 50, n)}
        means = {k: maps[s].mean(0) / cell_area for k, s in windows.items()}
        vmax = max(float(np.percentile(m[mask], 99.5)) for m in means.values())
        for name, heat in means.items():
            sel = frames[windows[name]]
            file = out / f'citystreet-labels-{name}-sigma{sigma_m:.1f}m.png'
            render(file, heat, mask, scene, vmax, f'CityStreet crowd concentration (labels), {name}',
                   f'frames {sel[0]}-{sel[-1]} ({len(sel)} labeled), kernel sigma {sigma_m} m, mean over frames')
            report['maps'].append({'file': file.name, 'window': name, 'frames': [int(sel[0]), int(sel[-1]), len(sel)],
                                   'sigma_m': sigma_m, 'peak_people_per_m2': float(heat.max()),
                                   'mean_people_in_coverage': float(heat[mask].sum() * cell_area)})
        report.setdefault('labels_outside_coverage', {})[f'sigma{sigma_m}'] = int(dropped)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'maps'}))


if __name__ == '__main__':
    main()
