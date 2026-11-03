"""Export cached detections and tracks as separate viewer layers.

Usage: python export.py bundle.json detections.json output [--tracks tracks.json]
No inference, mesh loading, or video decoding is performed.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

import bundle as bundle_io
import flow
import heatmap
import transform


def export_result(bundle, detections, out, fps=10, sigma_m=1, point='feet', tracks=None,
                  cell_m=4, min_count=None, north_deg=None):
    """Write a new result folder atomically; existing folders are preserved."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    if north_deg is not None and not np.isfinite(north_deg):
        raise ValueError('north_deg must be finite')
    if not np.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError('sigma_m must be positive and finite')
    if not np.isfinite(cell_m) or cell_m <= 0:
        raise ValueError('cell_m must be positive and finite')
    if min_count is not None and min_count < 1:
        raise ValueError('min_count must be positive')
    cv2.setNumThreads(2)
    result = heatmap.run(bundle, detections, sigma_m, point)
    owner = result['owner']
    covered, heat = result['covered'], result['heat']
    height, width = covered.shape
    vmax = max(float(np.percentile(heat[covered], 99.5)) if covered.any() else 0, 1e-9)
    level = np.clip(heat / vmax, 0, 1)
    color = cv2.applyColorMap(np.uint8(level * 255), cv2.COLORMAP_JET)
    heat_layer = np.dstack([color, np.uint8(level ** .6 * .75 * covered * 255)])
    gray = np.full((height, width, 4), 128, np.uint8)
    gray[..., 3] = np.uint8(~covered) * 255
    cameras = []
    for i, cam in enumerate(bundle['cameras']):
        area = transform.view_area(cam)
        direction = area.mean(0) - cam['position']
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm else np.zeros(2)
        owned = np.where(owner == i, 0, -1)
        counts = [len(heatmap.map_frame([cam], [result['homographies'][i]],
                                      owned, {cam['id']: detections[cam['id']][k]}, point))
                  for k in result['frames']]
        cameras.append({'id': cam['id'], 'position': cam['position'].tolist(), 'direction': direction.tolist(),
                        'direction_source': 'view_area_centroid',
                        'height_m': cam.get('height_m'), 'view_area': area.tolist(),
                        'area_m2': float((owner == i).sum() * bundle['meters_per_pixel'] ** 2),
                        'people_per_frame_mean': float(np.mean(counts))})
    report = heatmap.report(bundle, result)
    report.update(minutes=len(result['frames']) / fps / 60, sigma_m=sigma_m, point=point,
                  heat_max_people_per_m2=vmax)
    arrows = []
    if tracks is not None:
        keep = {int(k) for k in result['frames']}
        selected = {c: {tid: [p for p in pts if int(p[0]) in keep] for tid, pts in per.items()}
                    for c, per in tracks.items()}
        field = flow.flow_field(bundle, selected, fps, cell_m, covered=covered,
                                duration_s=len(result['frames']) / fps)
        arrows = [{'x': float(c[0]), 'y': float(c[1]), 'dx': float(d[0]), 'dy': float(d[1]),
                   'people_per_min': rate} for c, d, rate in flow.arrows(field, min_count)]
        report['flow'] = flow.summary(field, min_count)
    else:
        report['flow'] = None
    files = {'map.json': {'m_per_px': bundle['meters_per_pixel'], 'width': width, 'height': height,
                          'north_deg': north_deg}, 'cameras.json': cameras, 'flow.json': arrows, 'report.json': report}
    # Validate JSON before creating any result files.
    encoded = {name: json.dumps(value, indent=2, allow_nan=False) + '\n' for name, value in files.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{out.name}-', dir=out.parent))
    try:
        for name, img in [('map.png', result['base']), ('heat.png', heat_layer), ('coverage.png', gray)]:
            if not cv2.imwrite(str(staging / name), img):
                raise OSError(f'could not write {name}')
        for name, value in encoded.items():
            (staging / name).write_text(value)
        if out.exists():
            raise FileExistsError(out)
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bundle')
    ap.add_argument('detections')
    ap.add_argument('out')
    ap.add_argument('--tracks')
    ap.add_argument('--fps', type=float, default=10)
    ap.add_argument('--sigma-m', type=float, default=1)
    ap.add_argument('--point', choices=('feet', 'head'), default='feet')
    ap.add_argument('--cell-m', type=float, default=4)
    ap.add_argument('--min-count', type=int)
    ap.add_argument('--north-deg', type=float, help='clockwise angle of north from image up; unknown if omitted')
    a = ap.parse_args()
    b = bundle_io.load(a.bundle)
    camera_id = b['cameras'][0]['id'] if len(b['cameras']) == 1 else None
    tracks = flow.load_tracks(a.tracks, camera_id) if a.tracks else None
    print(json.dumps(export_result(b, json.loads(Path(a.detections).read_text()), a.out,
                                  a.fps, a.sigma_m, a.point, tracks, a.cell_m, a.min_count, a.north_deg)))


if __name__ == '__main__':
    main()
