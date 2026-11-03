"""Input bundle: a top-down map, cameras, and hand-marked ground point pairs.

Format version 0.1 (may change). See services/perception/flowmap/README.md.
"""
import json
from pathlib import Path

import numpy as np

VERSION = '0.1'


class BundleError(ValueError):
    pass


def _points(value, where, minimum):
    try:
        arr = np.asarray(value, float)
    except (TypeError, ValueError):
        raise BundleError(f'{where}: points must be numbers') from None
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < minimum:
        raise BundleError(f'{where}: need at least {minimum} [x, y] points')
    return arr


MIN_CAMERA_HEIGHT_M = 2.0  # head mode needs the camera above heads


def meters_per_pixel(scale):
    a, b = _points(scale.get('points'), 'map.scale.points', 2)[:2]
    meters = float(scale.get('meters', 0))
    pixels = float(np.hypot(*(b - a)))
    if meters <= 0 or pixels == 0:
        raise BundleError('map.scale: need two distinct points and a positive distance in meters')
    return meters / pixels


def load(path):
    """Read and validate a bundle JSON. Relative file paths resolve next to it."""
    path = Path(path)
    raw = json.loads(path.read_text())
    if str(raw.get('version')) != VERSION:
        raise BundleError(f'version must be "{VERSION}"')
    m = raw.get('map') or {}
    if 'image' not in m or 'scale' not in m:
        raise BundleError('map needs "image" and "scale"')
    bundle = {'root': path.parent, 'map_image': path.parent / m['image'],
              'meters_per_pixel': meters_per_pixel(m['scale']), 'cameras': []}
    ids = set()
    for i, cam in enumerate(raw.get('cameras') or []):
        where = f'cameras[{i}]'
        cid = cam.get('id')
        if not cid or cid in ids:
            raise BundleError(f'{where}: needs a unique "id"')
        ids.add(cid)
        pairs = cam.get('ground_points') or []
        frame = _points([p.get('frame') for p in pairs], f'{where}.ground_points.frame', 4)
        mapped = _points([p.get('map') for p in pairs], f'{where}.ground_points.map', 4)
        entry = {'id': cid, 'frame_points': frame, 'map_points': mapped,
                 'position': _points([cam.get('position')], f'{where}.position', 1)[0],
                 'frame_size': tuple(int(v) for v in cam.get('frame_size', (0, 0))),
                 'video': cam.get('video')}
        if 'height_m' in cam:
            height = float(cam['height_m'])
            if height <= MIN_CAMERA_HEIGHT_M:
                raise BundleError(f'{where}.height_m: camera must be higher than a person')
            entry['height_m'] = height
        if 'view_area' in cam:
            entry['view_area'] = _points(cam['view_area'], f'{where}.view_area', 3)
        bundle['cameras'].append(entry)
    if not bundle['cameras']:
        raise BundleError('bundle needs at least one camera')
    return bundle
